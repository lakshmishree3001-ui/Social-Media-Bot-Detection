"""
backend/services/pipeline/inference_stage.py

Stage 7: Run GNN inference for all accounts in a dataset.
Reuses the pre-warmed InferenceEngine singleton and persists
Prediction + PredictionSignal rows scoped to dataset_id.
"""
from __future__ import annotations

import json
import math
import time
from typing import List

from sqlalchemy.orm import Session

from backend.models.entities import (
    Account, AccountFeature, GraphFeature, ModelRun,
    Prediction, PredictionSignal
)
from backend.services.inference import InferenceEngine


def run_inference(
    dataset_id: int,
    db: Session,
    progress_cb=None,
) -> int:
    """
    Run inference over all accounts in dataset_id.
    Writes Prediction + PredictionSignal rows.
    Returns number of predictions written.
    """
    engine = InferenceEngine.get_instance()
    active_model: ModelRun = db.query(ModelRun).filter(ModelRun.is_active == True).first()
    if not active_model:
        raise RuntimeError("No active ModelRun found. Cannot run inference.")

    accounts: List[Account] = (
        db.query(Account).filter(Account.dataset_id == dataset_id).all()
    )
    if not accounts:
        return 0

    # Delete old predictions for this dataset
    old_preds = db.query(Prediction).filter(Prediction.dataset_id == dataset_id).all()
    for p in old_preds:
        db.delete(p)
    db.flush()

    af_map = {
        af.account_id: af
        for af in db.query(AccountFeature)
        .filter(AccountFeature.account_id.in_([a.id for a in accounts]))
        .all()
    }
    gf_map = {
        gf.account_id: gf
        for gf in db.query(GraphFeature)
        .filter(GraphFeature.account_id.in_([a.id for a in accounts]))
        .all()
    }

    predictions_written = 0
    total = len(accounts)

    for i, acc in enumerate(accounts):
        af = af_map.get(acc.id)
        gf = gf_map.get(acc.id)

        t0 = time.perf_counter()
        result = engine.predict_single(acc, af, gf)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        pred = Prediction(
            account_id=acc.id,
            dataset_id=dataset_id,
            model_run_id=active_model.id,
            bot_probability=result["bot_probability"],
            human_probability=result["human_probability"],
            suspicious_probability=result["suspicious_probability"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
        )
        db.add(pred)
        db.flush()

        # Prediction signals
        signals = result.get("signals", [])
        for s in signals:
            db.add(PredictionSignal(
                prediction_id=pred.id,
                signal_name=s["name"],
                signal_value=s["value"],
                signal_weight=s["weight"],
                signal_category=s.get("category", "graph"),
                description=s.get("description", ""),
            ))

        predictions_written += 1

        if progress_cb and i % 100 == 0:
            progress_cb(i / max(1, total))

    db.flush()

    if progress_cb:
        progress_cb(1.0)

    return predictions_written
