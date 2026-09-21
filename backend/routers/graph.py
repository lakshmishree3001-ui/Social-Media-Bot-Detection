from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from backend.database import get_db
from backend.models.entities import Account, Edge, AttentionWeight, CommunityMember, Community, Prediction, Dataset
from backend.services.explainer import ExplainerService

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


def _default_dataset_id(db: Session) -> int:
    d = db.query(Dataset).filter(Dataset.is_default == True).first()
    if d:
        return d.id
    first = db.query(Dataset).order_by(Dataset.id).first()
    return first.id if first else 1


@router.get("/subgraph")
def get_subgraph(
    dataset_id: Optional[int] = Query(None),
    limit_nodes: int = Query(200, ge=10, le=1000),
    community_id: Optional[int] = None,
    ground_truth: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Returns a connected topological subgraph for Cytoscape.js canvas rendering.
    All data is scoped to the specified dataset_id.
    """
    ds_id = dataset_id if dataset_id is not None else _default_dataset_id(db)
    account_query = db.query(Account).filter(Account.dataset_id == ds_id)

    if community_id is not None:
        account_query = (
            account_query
            .join(CommunityMember, CommunityMember.account_id == Account.id)
            .join(Community, CommunityMember.community_id == Community.id)
            .filter(Community.community_id == community_id, Community.dataset_id == ds_id)
        )

    if ground_truth is not None:
        account_query = account_query.filter(Account.ground_truth == ground_truth)

    accounts = account_query.limit(limit_nodes).all()
    node_ids = {a.id for a in accounts}

    preds = (
        db.query(Prediction)
        .filter(Prediction.account_id.in_(list(node_ids)), Prediction.dataset_id == ds_id)
        .all()
    )
    pred_map = {p.account_id: p for p in preds}

    cm_records = (
        db.query(CommunityMember, Community)
        .join(Community, CommunityMember.community_id == Community.id)
        .filter(CommunityMember.account_id.in_(list(node_ids)), Community.dataset_id == ds_id)
        .all()
    )
    comm_map = {cm.account_id: (c.community_id, c.classification) for cm, c in cm_records}

    nodes = []
    for a in accounts:
        p = pred_map.get(a.id)
        c_info = comm_map.get(a.id, (-1, "Unassigned"))
        nodes.append({
            "id": str(a.id),
            "label": a.screen_name,
            "followers": a.followers_count,
            "following": a.following_count,
            "ground_truth": a.ground_truth,
            "predicted_class": p.predicted_class if p else a.ground_truth,
            "bot_probability": p.bot_probability if p else 0.5,
            "confidence": p.confidence if p else 0.85,
            "community_id": c_info[0],
            "community_class": c_info[1],
            "dataset_id": a.dataset_id,
        })

    edges = (
        db.query(Edge)
        .filter(
            Edge.dataset_id == ds_id,
            Edge.source_id.in_(list(node_ids)),
            Edge.target_id.in_(list(node_ids)),
        )
        .limit(1500)
        .all()
    )

    edges_data = [
        {
            "id": f"e_{e.source_id}_{e.target_id}",
            "source": str(e.source_id),
            "target": str(e.target_id),
            "relation_type": e.relation_type,
            "weight": e.weight,
        }
        for e in edges
    ]

    return {
        "dataset_id": ds_id,
        "nodes": nodes,
        "edges": edges_data,
        "node_count": len(nodes),
        "edge_count": len(edges_data),
    }


@router.get("/ego/{account_id}")
def get_ego_graph(
    account_id: int,
    hops: int = Query(1, ge=1, le=2),
    max_neighbors: int = Query(35, ge=5, le=100),
    db: Session = Depends(get_db),
):
    acc = db.query(Account).filter(Account.id == account_id).first()
    if not acc:
        raise HTTPException(status_code=404, detail="Account not found")

    ego = ExplainerService.get_ego_graph(account_id=account_id, hops=hops, max_neighbors=max_neighbors, db=db)
    return ego
