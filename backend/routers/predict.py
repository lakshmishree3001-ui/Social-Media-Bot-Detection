from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import datetime

from backend.database import get_db
from backend.models.entities import Account, Prediction, PredictionSignal, Job, AuditLog
from backend.services.inference import InferenceEngine
from backend.services.explainer import ExplainerService

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])

class PredictRequest(BaseModel):
    model_name: Optional[str] = "GAT"

class BatchPredictRequest(BaseModel):
    account_ids: List[int]
    model_name: Optional[str] = "GAT"

@router.post("/account/{id}")
def predict_account(id: int, req: Optional[PredictRequest] = None, db: Session = Depends(get_db)):
    acc = db.query(Account).filter(Account.id == id).first()
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")

    model_name = req.model_name if req and req.model_name else "GAT"
    engine = InferenceEngine.get_instance()
    pred_res = engine.predict_account(account_id=id, model_name=model_name, db=db)

    # Attach ego graph for interactive analyst review
    ego_graph = ExplainerService.get_ego_graph(account_id=id, hops=1, max_neighbors=25, db=db)
    attributions = ExplainerService.get_feature_attribution(account_id=id, db=db)

    return {
        **pred_res,
        "ego_graph": ego_graph,
        "feature_attributions": attributions,
    }

@router.post("/batch")
def predict_batch(req: BatchPredictRequest, db: Session = Depends(get_db)):
    if len(req.account_ids) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1,000 accounts per batch request")

    engine = InferenceEngine.get_instance()
    results = []
    for aid in req.account_ids:
        r = engine.predict_account(account_id=aid, model_name=req.model_name, db=db)
        results.append(r)

    # Record job
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    job = Job(
        id=job_id,
        job_type="batch_inference",
        status="completed",
        progress=1.0,
        total_items=len(req.account_ids),
        started_at=datetime.datetime.utcnow(),
        completed_at=datetime.datetime.utcnow(),
    )
    db.add(job)
    db.commit()

    return {
        "job_id": job_id,
        "status": "completed",
        "total_processed": len(results),
        "results": results,
    }
