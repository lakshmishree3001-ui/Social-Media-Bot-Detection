import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "graphwarden_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(name="tasks.run_batch_inference")
def run_batch_inference(account_ids, model_name="GAT"):
    from backend.database import SessionLocal
    from backend.services.inference import InferenceEngine
    db = SessionLocal()
    engine = InferenceEngine.get_instance()
    results = []
    for aid in account_ids:
        r = engine.predict_account(account_id=aid, model_name=model_name, db=db)
        results.append(r)
    db.close()
    return {"total": len(results), "status": "completed"}
