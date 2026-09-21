"""
backend/services/pipeline/runner.py

Orchestrates all 8 ingestion pipeline stages for a dataset.
Runs synchronously in a background thread (called from FastAPI's
run_in_threadpool or a Celery task).

Stages
------
1. Validate & map columns      (validator.validate_and_map)
2. Normalise accounts          (normalizer.normalize_accounts)
3. Build graph & structural features (graph_engine.build_graph_and_features)
4. Behavioural features        (already computed inside normalizer, step 3 augments them)
5. Community detection         (community_engine.detect_communities)
6. Coordination detection      (coordination_engine.detect_coordination)
7. GNN inference               (inference_stage.run_inference)
8. Finalise dataset metadata   (update record_count, edge_count, status = ready)
"""
from __future__ import annotations

import json
import traceback
from typing import Dict, Callable, Optional

from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models.entities import Dataset, DatasetUpload, Job
from backend.services.pipeline.normalizer import normalize_accounts
from backend.services.pipeline.graph_engine import build_graph_and_features
from backend.services.pipeline.community_engine import detect_communities
from backend.services.pipeline.coordination_engine import detect_coordination
from backend.services.pipeline.inference_stage import run_inference


def _set_dataset_meta(db: Session, dataset_id: int, **kwargs):
    d = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if d:
        for k, v in kwargs.items():
            setattr(d, k, v)
        db.commit()


def _update_job(db: Session, job_id: str, **kwargs):
    j = db.query(Job).filter(Job.id == job_id).first()
    if j:
        for k, v in kwargs.items():
            setattr(j, k, v)
        db.commit()


def _cleanup_dataset_scoped_data(db: Session, dataset_id: int):
    """Remove any partial or existing scoped records for this dataset before ingestion."""
    from backend.models.entities import (
        PredictionSignal, Prediction, CommunityMember, Community,
        CoordinationMember, CoordinationCluster, Edge, AccountFeature,
        GraphFeature, Account
    )
    pred_ids = [p[0] for p in db.query(Prediction.id).filter(Prediction.dataset_id == dataset_id).all()]
    if pred_ids:
        db.query(PredictionSignal).filter(PredictionSignal.prediction_id.in_(pred_ids)).delete(synchronize_session=False)
    db.query(Prediction).filter(Prediction.dataset_id == dataset_id).delete(synchronize_session=False)

    comm_ids = [c[0] for c in db.query(Community.id).filter(Community.dataset_id == dataset_id).all()]
    if comm_ids:
        db.query(CommunityMember).filter(CommunityMember.community_id.in_(comm_ids)).delete(synchronize_session=False)
    db.query(Community).filter(Community.dataset_id == dataset_id).delete(synchronize_session=False)

    coord_ids = [cc[0] for cc in db.query(CoordinationCluster.id).filter(CoordinationCluster.dataset_id == dataset_id).all()]
    if coord_ids:
        db.query(CoordinationMember).filter(CoordinationMember.cluster_id.in_(coord_ids)).delete(synchronize_session=False)
    db.query(CoordinationCluster).filter(CoordinationCluster.dataset_id == dataset_id).delete(synchronize_session=False)

    db.query(Edge).filter(Edge.dataset_id == dataset_id).delete(synchronize_session=False)

    acc_ids = [a[0] for a in db.query(Account.id).filter(Account.dataset_id == dataset_id).all()]
    if acc_ids:
        db.query(AccountFeature).filter(AccountFeature.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(GraphFeature).filter(GraphFeature.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(Account).filter(Account.dataset_id == dataset_id).delete(synchronize_session=False)
    db.commit()


def run_ingestion_pipeline(
    dataset_id: int,
    df,                    # pandas DataFrame (already validated + column-mapped)
    job_id: Optional[str] = None,
) -> Dict:
    """
    Full 8-stage ingestion pipeline.
    Opens its own DB session (safe for background thread execution).
    Returns a summary dict.
    """
    db: Session = SessionLocal()
    summary: Dict = {"dataset_id": dataset_id, "stages": {}}

    def progress(stage: str, frac: float):
        """Update job progress in DB (0..1 per stage, overall progress across stages)."""
        stage_weights = {
            "normalise":     0.15,
            "graph":         0.20,
            "inference":     0.25,
            "community":     0.20,
            "coordination":  0.20,
        }
        stages_order = list(stage_weights.keys())
        completed_weight = sum(
            stage_weights[s] for s in stages_order
            if stages_order.index(s) < stages_order.index(stage)
        )
        overall = completed_weight + stage_weights.get(stage, 0.0) * frac
        if job_id:
            _update_job(db, job_id, progress=round(overall, 3))
        ingestion_meta = json.loads(
            db.query(Dataset).filter(Dataset.id == dataset_id)
            .with_entities(Dataset.ingestion_meta).scalar() or "{}"
        )
        ingestion_meta["stage"] = stage
        ingestion_meta["progress"] = round(overall, 3)
        _set_dataset_meta(db, dataset_id, ingestion_meta=json.dumps(ingestion_meta))

    try:
        _set_dataset_meta(db, dataset_id, status="ingesting",
                          ingestion_meta=json.dumps({"stage": "normalise", "progress": 0.0}))
        _cleanup_dataset_scoped_data(db, dataset_id)

        # ── Stage 1: Normalise ───────────────────────────────────────────
        progress("normalise", 0.0)
        n_accounts = normalize_accounts(dataset_id, df, db,
                                        progress_cb=lambda f: progress("normalise", f))
        db.commit()
        summary["stages"]["normalise"] = {"accounts_inserted": n_accounts}
        progress("normalise", 1.0)

        # ── Stage 2: Graph & structural features ─────────────────────────
        progress("graph", 0.0)
        n_graph = build_graph_and_features(dataset_id, db,
                                           progress_cb=lambda f: progress("graph", f))
        db.commit()
        summary["stages"]["graph"] = {"nodes_processed": n_graph}
        progress("graph", 1.0)

        # ── Stage 3: GNN Inference ───────────────────────────────────────
        progress("inference", 0.0)
        n_preds = run_inference(dataset_id, db,
                                progress_cb=lambda f: progress("inference", f))
        db.commit()
        summary["stages"]["inference"] = {"predictions_written": n_preds}
        progress("inference", 1.0)

        # ── Stage 4: Community detection (uses predictions) ──────────────
        progress("community", 0.0)
        n_comms = detect_communities(dataset_id, db,
                                     progress_cb=lambda f: progress("community", f))
        db.commit()
        summary["stages"]["community"] = {"communities_created": n_comms}
        progress("community", 1.0)

        # ── Stage 5: Coordination detection (uses predictions) ───────────
        progress("coordination", 0.0)
        n_clusters = detect_coordination(dataset_id, db,
                                         progress_cb=lambda f: progress("coordination", f))
        db.commit()
        summary["stages"]["coordination"] = {"clusters_created": n_clusters}
        progress("coordination", 1.0)

        # ── Stage 6: Finalise dataset metadata ──────────────────────────
        from backend.models.entities import Account, Edge
        rec_count = db.query(Account).filter(Account.dataset_id == dataset_id).count()
        edge_count = db.query(Edge).filter(Edge.dataset_id == dataset_id).count()
        _set_dataset_meta(
            db, dataset_id,
            status="ready",
            record_count=rec_count,
            edge_count=edge_count,
            ingestion_meta=json.dumps({"stage": "done", "progress": 1.0}),
        )
        if job_id:
            import datetime
            _update_job(db, job_id, status="completed", progress=1.0,
                        completed_at=datetime.datetime.utcnow())

        summary["status"] = "completed"
        return summary

    except Exception as exc:
        db.rollback()
        tb = traceback.format_exc()
        error_msg = f"{type(exc).__name__}: {exc}\n{tb}"
        print(f"[PIPELINE ERROR] Dataset {dataset_id}: {error_msg}")
        try:
            _set_dataset_meta(db, dataset_id, status="failed",
                              ingestion_meta=json.dumps({"stage": "error", "error": str(exc)}))
            if job_id:
                _update_job(db, job_id, status="failed", error_message=str(exc))
        except Exception as e2:
            print(f"[PIPELINE ERROR] Could not record failed status: {e2}")
        summary["status"] = "failed"
        summary["error"] = error_msg
        return summary
    finally:
        db.close()
