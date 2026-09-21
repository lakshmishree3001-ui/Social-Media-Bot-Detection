from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from typing import Optional

from backend.database import get_db
from backend.models.entities import Community, CommunityMember, Account, Prediction, Dataset

router = APIRouter(prefix="/api/v1/communities", tags=["communities"])


def _default_dataset_id(db: Session) -> int:
    d = db.query(Dataset).filter(Dataset.is_default == True).first()
    if d:
        return d.id
    first = db.query(Dataset).order_by(Dataset.id).first()
    return first.id if first else 1


@router.get("")
def list_communities(
    dataset_id: Optional[int] = Query(None),
    sort_by: str = Query("bot_concentration", pattern="^(bot_concentration|size|internal_density|internal_edges)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    ds_id = dataset_id if dataset_id is not None else _default_dataset_id(db)
    query = db.query(Community).filter(Community.dataset_id == ds_id)

    if sort_by == "size":
        order_col = Community.size
    elif sort_by == "internal_density":
        order_col = Community.internal_density
    elif sort_by == "internal_edges":
        order_col = Community.internal_edges
    else:
        order_col = Community.bot_concentration

    if sort_order == "desc":
        query = query.order_by(desc(order_col))
    else:
        query = query.order_by(asc(order_col))

    comms = query.all()
    return [
        {
            "id": c.id,
            "dataset_id": c.dataset_id,
            "community_id": c.community_id,
            "algorithm": c.algorithm,
            "size": c.size,
            "n_bots": c.n_bots,
            "n_suspicious": c.n_suspicious,
            "n_humans": c.n_humans,
            "bot_concentration": c.bot_concentration,
            "suspicious_concentration": c.suspicious_concentration,
            "human_concentration": c.human_concentration,
            "coordinated_ratio": c.coordinated_ratio,
            "internal_edges": c.internal_edges,
            "internal_density": c.internal_density,
            "classification": c.classification,
        }
        for c in comms
    ]


@router.get("/{community_id}")
def get_community_detail(
    community_id: int,
    dataset_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    ds_id = dataset_id if dataset_id is not None else _default_dataset_id(db)
    c = (
        db.query(Community)
        .filter(Community.community_id == community_id, Community.dataset_id == ds_id)
        .first()
    )
    if not c:
        raise HTTPException(status_code=404, detail="Community not found")

    members = (
        db.query(CommunityMember, Account, Prediction)
        .join(Account, CommunityMember.account_id == Account.id)
        .outerjoin(
            Prediction,
            (Account.id == Prediction.account_id) & (Prediction.dataset_id == ds_id),
        )
        .filter(CommunityMember.community_id == c.id)
        .limit(50)
        .all()
    )

    member_list = []
    for cm, acc, pred in members:
        member_list.append({
            "account_id": acc.id,
            "screen_name": acc.screen_name,
            "followers": acc.followers_count,
            "following": acc.following_count,
            "ground_truth": acc.ground_truth,
            "role": cm.role,
            "predicted_class": pred.predicted_class if pred else acc.ground_truth,
            "bot_probability": pred.bot_probability if pred else 0.5,
        })

    return {
        "dataset_id": ds_id,
        "community_id": c.community_id,
        "algorithm": c.algorithm,
        "size": c.size,
        "n_bots": c.n_bots,
        "n_suspicious": c.n_suspicious,
        "n_humans": c.n_humans,
        "bot_concentration": c.bot_concentration,
        "suspicious_concentration": c.suspicious_concentration,
        "human_concentration": c.human_concentration,
        "coordinated_ratio": c.coordinated_ratio,
        "internal_edges": c.internal_edges,
        "internal_density": c.internal_density,
        "classification": c.classification,
        "members": member_list,
    }
