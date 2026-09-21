from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json

from backend.database import get_db
from backend.models.entities import ModelRun, AuditLog

router = APIRouter(prefix="/api/v1/models", tags=["models"])

@router.get("")
def list_models(db: Session = Depends(get_db)):
    runs = db.query(ModelRun).order_by(ModelRun.accuracy.desc()).all()
    res = []
    for r in runs:
        params = {}
        if r.hyperparameters:
            try:
                params = json.loads(r.hyperparameters)
            except Exception:
                params = {}
        res.append({
            "id": r.id,
            "model_name": r.model_name,
            "architecture": r.architecture,
            "accuracy": r.accuracy,
            "f1_macro": r.f1_macro,
            "precision": r.precision,
            "recall": r.recall,
            "roc_auc": r.roc_auc,
            "hyperparameters": params,
            "checkpoint_path": r.checkpoint_path,
            "is_active": r.is_active,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return res

@router.post("/{id}/activate")
def activate_model(id: int, db: Session = Depends(get_db)):
    target = db.query(ModelRun).filter(ModelRun.id == id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Model run not found")

    # Set all other models to inactive
    db.query(ModelRun).update({ModelRun.is_active: False})
    target.is_active = True

    audit = AuditLog(
        action="MODEL_ACTIVATED",
        target_type="MODEL",
        target_id=str(target.id),
        details=f"Switched active production inference model to {target.model_name}.",
    )
    db.add(audit)
    db.commit()

    return {"message": f"Model {target.model_name} activated successfully", "active_model_id": target.id}
