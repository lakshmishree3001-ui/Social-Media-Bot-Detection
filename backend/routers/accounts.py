from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc, asc
from typing import Optional

from backend.database import get_db
from backend.models.entities import (
    Account,
    AccountFeature,
    GraphFeature,
    Post,
    Prediction,
    PredictionSignal,
    CommunityMember,
    Community,
    AuditLog,
    Dataset,
)
from backend.services.pdf_export import generate_forensic_dossier_pdf
from backend.services.explainer import ExplainerService

router = APIRouter(prefix="/api/v1/accounts", tags=["accounts"])


def _resolve_dataset(dataset_id: Optional[int], db: Session) -> int:
    """Return dataset_id, falling back to the default (seed) dataset."""
    if dataset_id is not None:
        return dataset_id
    default = db.query(Dataset).filter(Dataset.is_default == True).first()
    if default:
        return default.id
    # Fall back to lowest id
    first = db.query(Dataset).order_by(Dataset.id).first()
    return first.id if first else 1


@router.get("")
def list_accounts(
    dataset_id: Optional[int] = Query(None, description="Filter to a specific dataset. Defaults to the seed dataset."),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: Optional[str] = None,
    ground_truth: Optional[str] = None,
    predicted_class: Optional[str] = None,
    community_id: Optional[int] = None,
    verified: Optional[bool] = None,
    min_followers: Optional[int] = None,
    sort_by: str = Query("id", pattern="^(id|followers_count|following_count|bot_probability|confidence)$"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    ds_id = _resolve_dataset(dataset_id, db)
    query = db.query(Account).filter(Account.dataset_id == ds_id)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                Account.screen_name.ilike(search_pattern),
                Account.user_id_str.ilike(search_pattern),
            )
        )

    if ground_truth:
        query = query.filter(Account.ground_truth == ground_truth)

    if verified is not None:
        query = query.filter(Account.verified == verified)

    if min_followers is not None:
        query = query.filter(Account.followers_count >= min_followers)

    if community_id is not None:
        query = (
            query
            .join(CommunityMember, CommunityMember.account_id == Account.id)
            .join(Community, CommunityMember.community_id == Community.id)
            .filter(Community.community_id == community_id, Community.dataset_id == ds_id)
        )

    # Track whether Prediction is already joined to avoid ambiguous double-join
    _pred_joined = False

    if predicted_class is not None:
        query = (
            query
            .join(Prediction, (Prediction.account_id == Account.id) & (Prediction.dataset_id == ds_id))
            .filter(Prediction.predicted_class == predicted_class)
        )
        _pred_joined = True

    # Sorting
    if sort_by == "followers_count":
        order_col = Account.followers_count
    elif sort_by == "following_count":
        order_col = Account.following_count
    elif sort_by in ("bot_probability", "confidence"):
        if not _pred_joined:
            query = query.join(
                Prediction,
                (Prediction.account_id == Account.id) & (Prediction.dataset_id == ds_id)
            )
        order_col = Prediction.bot_probability if sort_by == "bot_probability" else Prediction.confidence
    else:
        order_col = Account.id

    if sort_order == "desc":
        query = query.order_by(desc(order_col))
    else:
        query = query.order_by(asc(order_col))

    total = query.count()
    accounts = query.offset((page - 1) * page_size).limit(page_size).all()

    account_ids = [a.id for a in accounts]
    predictions = (
        db.query(Prediction)
        .filter(Prediction.account_id.in_(account_ids), Prediction.dataset_id == ds_id)
        .all()
    )
    pred_map = {p.account_id: p for p in predictions}

    items = []
    for a in accounts:
        p = pred_map.get(a.id)
        items.append({
            "id": a.id,
            "user_id_str": a.user_id_str,
            "screen_name": a.screen_name,
            "name": a.name,
            "followers_count": a.followers_count,
            "following_count": a.following_count,
            "post_count": a.post_count,
            "verified": a.verified,
            "ground_truth": a.ground_truth,
            "dataset_id": a.dataset_id,
            "prediction": {
                "predicted_class": p.predicted_class if p else "unknown",
                "bot_probability": p.bot_probability if p else 0.0,
                "human_probability": p.human_probability if p else 0.0,
                "suspicious_probability": p.suspicious_probability if p else 0.0,
                "confidence": p.confidence if p else 0.0,
            } if p else None,
        })

    return {
        "dataset_id": ds_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "items": items,
    }


@router.get("/{id}")
def get_account_detail(
    id: int,
    dataset_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    a = db.query(Account).filter(Account.id == id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Account not found")

    ds_id = a.dataset_id  # use the account's own dataset

    af = a.account_features
    gf = a.graph_features
    pred = (
        db.query(Prediction)
        .filter(Prediction.account_id == a.id, Prediction.dataset_id == ds_id)
        .first()
    )
    signals = []
    if pred:
        signals = [
            {
                "name": s.signal_name,
                "value": s.signal_value,
                "weight": s.signal_weight,
                "category": s.signal_category,
                "description": s.description,
            }
            for s in pred.signals
        ]

    # Community info — scoped to dataset
    cm = (
        db.query(CommunityMember, Community)
        .join(Community, CommunityMember.community_id == Community.id)
        .filter(CommunityMember.account_id == a.id, Community.dataset_id == ds_id)
        .first()
    )
    comm_data = None
    if cm:
        comm_data = {
            "community_id": cm.Community.community_id,
            "role": cm.CommunityMember.role,
            "classification": cm.Community.classification,
            "bot_concentration": cm.Community.bot_concentration,
            "size": cm.Community.size,
        }

    # Recent posts
    from sqlalchemy import desc as _desc
    recent_posts = (
        db.query(Post)
        .filter(Post.account_id == a.id)
        .order_by(_desc(Post.id))
        .limit(5)
        .all()
    )

    posts_data = [
        {
            "id": p.id,
            "tweet_id_str": p.tweet_id_str,
            "text": p.text,
            "timestamp": p.timestamp,
            "retweet_count": p.retweet_count,
            "like_count": p.like_count,
            "reply_count": p.reply_count,
            "has_url": p.has_url,
            "has_hashtag": p.has_hashtag,
        }
        for p in recent_posts
    ]

    return {
        "id": a.id,
        "dataset_id": a.dataset_id,
        "user_id_str": a.user_id_str,
        "screen_name": a.screen_name,
        "name": a.name,
        "description": a.description,
        "location": a.location,
        "created_at": a.created_at,
        "account_age_days": a.account_age_days,
        "followers_count": a.followers_count,
        "following_count": a.following_count,
        "post_count": a.post_count,
        "listed_count": a.listed_count,
        "verified": a.verified,
        "has_profile_image": a.has_profile_image,
        "has_description": a.has_description,
        "default_profile": a.default_profile,
        "profile_completeness": a.profile_completeness,
        "ground_truth": a.ground_truth,
        "account_features": {
            "follower_friend_ratio": af.follower_friend_ratio if af else 0.0,
            "rep_score": af.rep_score if af else 0.0,
            "activity_rate": af.activity_rate if af else 0.0,
            "posts_per_day": af.posts_per_day if af else 0.0,
            "reply_ratio": af.reply_ratio if af else 0.0,
            "retweet_ratio": af.retweet_ratio if af else 0.0,
            "mention_ratio": af.mention_ratio if af else 0.0,
            "url_ratio": af.url_ratio if af else 0.0,
            "hashtag_ratio": af.hashtag_ratio if af else 0.0,
            "duplicate_content_ratio": af.duplicate_content_ratio if af else 0.0,
            "tweet_hour_entropy": af.tweet_hour_entropy if af else 0.0,
            "tweet_similarity_score": af.tweet_similarity_score if af else 0.0,
            "engagement_score": af.engagement_score if af else 0.0,
            "activity_consistency": af.activity_consistency if af else 0.0,
        } if af else None,
        "graph_features": {
            "in_degree": gf.in_degree if gf else 0,
            "out_degree": gf.out_degree if gf else 0,
            "total_degree": gf.total_degree if gf else 0,
            "in_degree_centrality": gf.in_degree_centrality if gf else 0.0,
            "out_degree_centrality": gf.out_degree_centrality if gf else 0.0,
            "pagerank": gf.pagerank if gf else 0.0,
            "betweenness_centrality": gf.betweenness_centrality if gf else 0.0,
            "clustering_coeff": gf.clustering_coeff if gf else 0.0,
            "k_core": gf.k_core if gf else 0,
        } if gf else None,
        "community": comm_data,
        "prediction": {
            "predicted_class": pred.predicted_class if pred else "unknown",
            "bot_probability": pred.bot_probability if pred else 0.0,
            "human_probability": pred.human_probability if pred else 0.0,
            "suspicious_probability": pred.suspicious_probability if pred else 0.0,
            "confidence": pred.confidence if pred else 0.0,
            "latency_ms": pred.latency_ms if pred else 0.0,
            "signals": signals,
        } if pred else None,
        "recent_posts": posts_data,
    }


@router.get("/{id}/dossier/pdf")
def export_dossier_pdf(id: int, db: Session = Depends(get_db)):
    a = db.query(Account).filter(Account.id == id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Account not found")

    ds_id = a.dataset_id
    pred = (
        db.query(Prediction)
        .filter(Prediction.account_id == a.id, Prediction.dataset_id == ds_id)
        .first()
    )
    attributions = ExplainerService.get_feature_attribution(a.id, db=db)

    account_data = {
        "id": a.id,
        "user_id_str": a.user_id_str,
        "screen_name": a.screen_name,
        "account_age_days": a.account_age_days,
        "followers_count": a.followers_count,
        "following_count": a.following_count,
        "verified": a.verified,
        "ground_truth": a.ground_truth,
    }

    prediction_data = {
        "predicted_class": pred.predicted_class if pred else "unknown",
        "confidence": pred.confidence if pred else 0.90,
        "latency_ms": pred.latency_ms if pred else 12.0,
        "probabilities": {
            "human": pred.human_probability if pred else 0.1,
            "bot": pred.bot_probability if pred else 0.85,
            "suspicious": pred.suspicious_probability if pred else 0.05,
        } if pred else {},
        "model_name": "Graph Attention Network (GAT)",
    }

    pdf_buffer = generate_forensic_dossier_pdf(account_data, prediction_data, attributions)

    audit = AuditLog(
        action="EXPORT_PDF_DOSSIER",
        target_type="ACCOUNT",
        target_id=str(a.id),
        details=f"Exported forensic dossier PDF for @{a.screen_name} (dataset_id={ds_id}).",
    )
    db.add(audit)
    db.commit()

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="graphwarden_dossier_{a.screen_name}.pdf"'
        }
    )
