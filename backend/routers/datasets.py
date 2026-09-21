"""
backend/routers/datasets.py

Full dataset management router:
  GET    /api/v1/datasets                  – list all datasets
  GET    /api/v1/datasets/{id}             – single dataset detail
  POST   /api/v1/datasets/upload           – upload CSV, returns upload_id + column suggestions
  POST   /api/v1/datasets/confirm          – confirm column mapping, kick off pipeline
  GET    /api/v1/datasets/{id}/status      – ingestion status + progress
  GET    /api/v1/datasets/{id}/rejected-rows – rows rejected during validation
  PATCH  /api/v1/datasets/{id}             – update name/description
  DELETE /api/v1/datasets/{id}             – delete dataset and all scoped rows
"""
import json
import os
import threading
import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.entities import (
    Dataset, DatasetUpload, Account, Edge, Community, CoordinationCluster,
    Prediction, Job, AuditLog, PredictionSignal, CommunityMember,
    CoordinationMember, AccountFeature, GraphFeature, Post, Embedding
)
from backend.services.pipeline.validator import validate_and_map, infer_column_map_suggestions

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _dataset_summary(d: Dataset, db: Session) -> dict:
    bot_count = db.query(Account).filter(
        Account.dataset_id == d.id, Account.ground_truth == "bot").count()
    human_count = db.query(Account).filter(
        Account.dataset_id == d.id, Account.ground_truth == "human").count()
    susp_count = db.query(Account).filter(
        Account.dataset_id == d.id, Account.ground_truth == "suspicious").count()

    if bot_count == 0 and human_count == 0 and susp_count == 0:
        bot_count = db.query(Prediction).filter(
            Prediction.dataset_id == d.id, Prediction.predicted_class == "bot").count()
        human_count = db.query(Prediction).filter(
            Prediction.dataset_id == d.id, Prediction.predicted_class == "human").count()
        susp_count = db.query(Prediction).filter(
            Prediction.dataset_id == d.id, Prediction.predicted_class == "suspicious").count()

    meta = {}
    if d.ingestion_meta:
        try:
            meta = json.loads(d.ingestion_meta)
        except Exception:
            pass

    return {
        "id": d.id,
        "name": d.name,
        "description": d.description,
        "record_count": d.record_count,
        "edge_count": d.edge_count,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "is_active": d.is_active,
        "is_default": d.is_default,
        "source": d.source,
        "status": d.status,
        "ingestion_stage": meta.get("stage"),
        "ingestion_progress": meta.get("progress", 0.0),
        "breakdown": {
            "bot_count": bot_count,
            "human_count": human_count,
            "suspicious_count": susp_count,
        },
        "density": round(d.edge_count / max(1, d.record_count * (d.record_count - 1)), 8),
    }


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.id).all()
    return [_dataset_summary(d, db) for d in datasets]


@router.get("/{id}")
def get_dataset(id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    comm_count = db.query(Community).filter(Community.dataset_id == id).count()
    coord_count = db.query(CoordinationCluster).filter(CoordinationCluster.dataset_id == id).count()
    s = _dataset_summary(d, db)
    s["community_count"] = comm_count
    s["coordination_cluster_count"] = coord_count
    return s


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Stage 0: Accept a CSV/TSV file, run header-only validation,
    return column mapping suggestions so the UI can present the
    column-mapping step before committing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in (".csv", ".tsv", ".txt"):
        raise HTTPException(status_code=400, detail="Only .csv / .tsv files are accepted")

    raw = await file.read()
    if len(raw) > 200 * 1024 * 1024:  # 200 MB limit
        raise HTTPException(status_code=413, detail="File exceeds 200 MB limit")

    # Save to disk
    upload_id = str(uuid.uuid4())
    stored_path = os.path.join(UPLOAD_DIR, f"{upload_id}{ext}")
    with open(stored_path, "wb") as fh:
        fh.write(raw)

    # Column suggestions (header-only scan, fast)
    suggestions = infer_column_map_suggestions(raw)

    # Create a placeholder Dataset + DatasetUpload row
    ds = Dataset(
        name=os.path.splitext(file.filename)[0][:150],
        description=f"Uploaded from {file.filename}",
        source="upload",
        status="pending_confirm",
        is_default=False,
    )
    db.add(ds)
    db.flush()

    du = DatasetUpload(
        dataset_id=ds.id,
        original_filename=file.filename,
        stored_path=stored_path,
    )
    db.add(du)
    db.commit()
    db.refresh(ds)
    db.refresh(du)

    return {
        "upload_id": du.id,
        "dataset_id": ds.id,
        "original_filename": file.filename,
        "file_size_bytes": len(raw),
        "column_suggestions": suggestions,
        "message": "File accepted. Submit column_map via /confirm to start ingestion.",
    }


@router.post("/confirm")
def confirm_dataset(
    upload_id: int = Form(...),
    column_map: Optional[str] = Form(None),   # JSON string: {"raw_col": "canonical_col"}
    db: Session = Depends(get_db),
):
    """
    Stage 1+: Receive confirmed column mapping, validate the CSV in full,
    and kick off the background ingestion pipeline.
    """
    du = db.query(DatasetUpload).filter(DatasetUpload.id == upload_id).first()
    if not du:
        raise HTTPException(status_code=404, detail="Upload record not found")

    ds = db.query(Dataset).filter(Dataset.id == du.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset record not found")

    if ds.status not in ("pending_confirm", "failed"):
        raise HTTPException(status_code=409,
                            detail=f"Dataset status is '{ds.status}' — cannot re-confirm")

    col_map = {}
    if column_map:
        try:
            col_map = json.loads(column_map)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="column_map must be valid JSON")

    # Validate the file
    with open(du.stored_path, "rb") as fh:
        raw = fh.read()

    try:
        df_clean, rejected, warnings = validate_and_map(raw, col_map, du.original_filename)
    except ValueError as exc:
        ds.status = "failed"
        ds.ingestion_meta = json.dumps({"stage": "validation", "error": str(exc)})
        db.commit()
        raise HTTPException(status_code=422, detail=str(exc))

    # Persist column map + rejected info
    du.column_map = json.dumps(col_map)
    du.rejected_count = len(rejected)
    du.rejected_rows = json.dumps(rejected[:500])  # cap stored rows at 500
    db.commit()

    if len(df_clean) == 0:
        ds.status = "failed"
        ds.ingestion_meta = json.dumps({"stage": "validation", "error": "All rows were rejected"})
        db.commit()
        raise HTTPException(status_code=422, detail="All rows were rejected during validation. No data to ingest.")

    # Create Job record
    job_id = f"ingest_{ds.id}_{int(datetime.datetime.utcnow().timestamp())}"
    job = Job(
        id=job_id,
        job_type="ingest_dataset",
        dataset_id=ds.id,
        status="running",
        total_items=len(df_clean),
    )
    db.add(job)
    ds.status = "ingesting"
    ds.ingestion_meta = json.dumps({"stage": "queued", "progress": 0.0})
    db.commit()

    # Run pipeline in background thread
    from backend.services.pipeline.runner import run_ingestion_pipeline

    def _run():
        run_ingestion_pipeline(ds.id, df_clean, job_id=job_id)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {
        "dataset_id": ds.id,
        "job_id": job_id,
        "rows_accepted": len(df_clean),
        "rows_rejected": len(rejected),
        "warnings": warnings,
        "status": "ingesting",
        "message": "Ingestion pipeline started. Poll /status for progress.",
    }


@router.get("/{id}/status")
def get_dataset_status(id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")

    meta = {}
    if d.ingestion_meta:
        try:
            meta = json.loads(d.ingestion_meta)
        except Exception:
            pass

    latest_job = (
        db.query(Job)
        .filter(Job.dataset_id == id)
        .order_by(Job.started_at.desc())
        .first()
    )

    return {
        "dataset_id": id,
        "status": d.status,
        "stage": meta.get("stage"),
        "progress": meta.get("progress", 0.0),
        "error": meta.get("error"),
        "job": {
            "id": latest_job.id,
            "status": latest_job.status,
            "progress": latest_job.progress,
            "total_items": latest_job.total_items,
            "error_message": latest_job.error_message,
            "started_at": latest_job.started_at.isoformat() if latest_job.started_at else None,
            "completed_at": latest_job.completed_at.isoformat() if latest_job.completed_at else None,
        } if latest_job else None,
    }


@router.get("/{id}/rejected-rows")
def get_rejected_rows(id: int, db: Session = Depends(get_db)):
    du = db.query(DatasetUpload).filter(DatasetUpload.dataset_id == id).order_by(DatasetUpload.id.desc()).first()
    if not du:
        raise HTTPException(status_code=404, detail="No upload found for this dataset")
    rows = []
    if du.rejected_rows:
        try:
            rows = json.loads(du.rejected_rows)
        except Exception:
            pass
    return {
        "dataset_id": id,
        "rejected_count": du.rejected_count,
        "rows": rows,
    }


@router.patch("/{id}")
def patch_dataset(
    id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    d = db.query(Dataset).filter(Dataset.id == id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if name:
        d.name = name[:150]
    if description is not None:
        d.description = description
    db.commit()
    return {"dataset_id": id, "name": d.name, "description": d.description}


@router.delete("/{id}")
def delete_dataset(id: int, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.id == id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if d.is_default:
        raise HTTPException(status_code=403,
                            detail="Cannot delete the default (seed) dataset")

    # Clean cascading deletion of all scoped entities
    pred_ids = [p[0] for p in db.query(Prediction.id).filter(Prediction.dataset_id == id).all()]
    if pred_ids:
        db.query(PredictionSignal).filter(PredictionSignal.prediction_id.in_(pred_ids)).delete(synchronize_session=False)
    db.query(Prediction).filter(Prediction.dataset_id == id).delete(synchronize_session=False)

    comm_ids = [c[0] for c in db.query(Community.id).filter(Community.dataset_id == id).all()]
    if comm_ids:
        db.query(CommunityMember).filter(CommunityMember.community_id.in_(comm_ids)).delete(synchronize_session=False)
    db.query(Community).filter(Community.dataset_id == id).delete(synchronize_session=False)

    coord_ids = [cc[0] for cc in db.query(CoordinationCluster.id).filter(CoordinationCluster.dataset_id == id).all()]
    if coord_ids:
        db.query(CoordinationMember).filter(CoordinationMember.cluster_id.in_(coord_ids)).delete(synchronize_session=False)
    db.query(CoordinationCluster).filter(CoordinationCluster.dataset_id == id).delete(synchronize_session=False)

    db.query(Edge).filter(Edge.dataset_id == id).delete(synchronize_session=False)

    acc_ids = [a[0] for a in db.query(Account.id).filter(Account.dataset_id == id).all()]
    if acc_ids:
        db.query(AccountFeature).filter(AccountFeature.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(GraphFeature).filter(GraphFeature.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(Post).filter(Post.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(Embedding).filter(Embedding.account_id.in_(acc_ids)).delete(synchronize_session=False)
        db.query(Account).filter(Account.dataset_id == id).delete(synchronize_session=False)

    db.query(Job).filter(Job.dataset_id == id).delete(synchronize_session=False)
    db.query(DatasetUpload).filter(DatasetUpload.dataset_id == id).delete(synchronize_session=False)

    db.delete(d)
    db.commit()

    al = AuditLog(
        action="DELETE_DATASET",
        target_type="DATASET",
        target_id=str(id),
        details=f"Deleted dataset '{d.name}' (id={id})",
    )
    db.add(al)
    db.commit()

    return {"deleted": id}
