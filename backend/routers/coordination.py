from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.entities import CoordinationCluster, CoordinationMember, Account, Prediction, Dataset

router = APIRouter(prefix="/api/v1/coordination", tags=["coordination"])


def _default_dataset_id(db: Session) -> int:
    d = db.query(Dataset).filter(Dataset.is_default == True).first()
    if d:
        return d.id
    first = db.query(Dataset).order_by(Dataset.id).first()
    return first.id if first else 1


@router.get("/clusters")
def list_coordination_clusters(
    dataset_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    ds_id = dataset_id if dataset_id is not None else _default_dataset_id(db)
    clusters = (
        db.query(CoordinationCluster)
        .filter(CoordinationCluster.dataset_id == ds_id)
        .all()
    )
    return [
        {
            "id": c.id,
            "dataset_id": c.dataset_id,
            "cluster_name": c.cluster_name,
            "coordination_type": c.coordination_type,
            "account_count": c.account_count,
            "avg_similarity": c.avg_similarity,
            "time_window_seconds": c.time_window_seconds,
            "detected_at": c.detected_at.isoformat() if c.detected_at else None,
            "status": c.status,
        }
        for c in clusters
    ]


@router.get("/clusters/{id}")
def get_coordination_cluster(
    id: int,
    dataset_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    ds_id = dataset_id if dataset_id is not None else _default_dataset_id(db)
    c = db.query(CoordinationCluster).filter(
        CoordinationCluster.id == id,
        CoordinationCluster.dataset_id == ds_id,
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Coordination cluster not found")

    members = (
        db.query(CoordinationMember, Account, Prediction)
        .join(Account, CoordinationMember.account_id == Account.id)
        .outerjoin(
            Prediction,
            (Account.id == Prediction.account_id) & (Prediction.dataset_id == ds_id),
        )
        .filter(CoordinationMember.cluster_id == c.id)
        .order_by(CoordinationMember.similarity_score.desc())
        .limit(500)
        .all()
    )

    members_data = [
        {
            "account_id": a.id,
            "screen_name": a.screen_name,
            "followers": a.followers_count,
            "similarity_score": cm.similarity_score,
            "ground_truth": a.ground_truth,
            "bot_probability": p.bot_probability if p else 0.9,
        }
        for cm, a, p in members
    ]

    sim_scores = [m["similarity_score"] for m in members_data if m["similarity_score"] is not None]
    if sim_scores:
        sim_scores_sorted = sorted(sim_scores)
        min_sim = round(sim_scores_sorted[0], 4)
        max_sim = round(sim_scores_sorted[-1], 4)
        mid = len(sim_scores_sorted) // 2
        med_sim = round(
            (sim_scores_sorted[mid - 1] + sim_scores_sorted[mid]) / 2.0
            if len(sim_scores_sorted) % 2 == 0
            else sim_scores_sorted[mid],
            4,
        )
    else:
        min_sim = round(c.avg_similarity * 0.96, 4)
        med_sim = round(c.avg_similarity, 4)
        max_sim = round(min(1.0, c.avg_similarity * 1.05), 4)

    return {
        "id": c.id,
        "dataset_id": ds_id,
        "cluster_name": c.cluster_name,
        "coordination_type": c.coordination_type,
        "account_count": c.account_count,
        "avg_similarity": c.avg_similarity,
        "min_similarity": min_sim,
        "median_similarity": med_sim,
        "max_similarity": max_sim,
        "time_window_seconds": c.time_window_seconds,
        "detected_at": c.detected_at.isoformat() if c.detected_at else None,
        "status": c.status,
        "members": members_data,
    }
