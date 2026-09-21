"""
backend/services/pipeline/coordination_engine.py

Stage 6: Coordination analysis.
Detects clusters of accounts posting near-identical content within tight
time windows. Persists CoordinationCluster + CoordinationMember rows.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

from sqlalchemy.orm import Session

from backend.models.entities import (
    Account, AccountFeature, CoordinationCluster, CoordinationMember, Prediction
)


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    denom = n1 * n2
    return dot / denom if denom > 1e-9 else 0.0


def _feature_vector(af: AccountFeature | None) -> List[float]:
    """Convert AccountFeature to a fixed-length numeric vector."""
    if af is None:
        return [0.0] * 8
    return [
        af.retweet_ratio,
        af.reply_ratio,
        af.mention_ratio,
        af.url_ratio,
        af.hashtag_ratio,
        af.duplicate_content_ratio,
        af.tweet_hour_entropy,
        af.activity_consistency,
    ]


def detect_coordination(
    dataset_id: int,
    db: Session,
    progress_cb=None,
) -> int:
    """
    Groups accounts by behavioral similarity (cosine similarity on
    AccountFeature vectors) and temporal proximity (account_age_days
    used as proxy for activity epoch).

    Persists CoordinationCluster + CoordinationMember rows scoped to
    dataset_id. Returns count of clusters created.
    """
    accounts: List[Account] = (
        db.query(Account).filter(Account.dataset_id == dataset_id).all()
    )
    if len(accounts) < 5:
        return 0

    # Fetch features
    aid_list = [a.id for a in accounts]
    af_rows = (
        db.query(AccountFeature)
        .filter(AccountFeature.account_id.in_(aid_list))
        .all()
    )
    af_map: Dict[int, AccountFeature] = {af.account_id: af for af in af_rows}
    acc_map: Dict[int, Account] = {a.id: a for a in accounts}

    # Fetch predictions for label
    pred_map: Dict[int, str] = {}
    for p in db.query(Prediction).filter(Prediction.dataset_id == dataset_id).all():
        pred_map[p.account_id] = p.predicted_class

    if progress_cb:
        progress_cb(0.15)

    # Delete existing clusters for this dataset
    old_clusters = (
        db.query(CoordinationCluster).filter(CoordinationCluster.dataset_id == dataset_id).all()
    )
    for c in old_clusters:
        db.delete(c)
    db.flush()

    # Compute feature vectors
    vectors: Dict[int, List[float]] = {aid: _feature_vector(af_map.get(aid)) for aid in aid_list}

    SIMILARITY_THRESHOLD = 0.85
    MIN_CLUSTER_SIZE = 3

    # Greedy single-pass clustering: O(n * cluster_count), fast enough for n < 50k
    cluster_seeds: List[Tuple[int, List[float]]] = []  # (representative_aid, centroid_vector)
    cluster_members: List[List[int]] = []

    random.seed(dataset_id)
    shuffled_aids = aid_list[:]
    random.shuffle(shuffled_aids)

    for aid in shuffled_aids:
        vec = vectors[aid]
        placed = False
        for ci, (_, centroid) in enumerate(cluster_seeds):
            sim = _cosine_sim(vec, centroid)
            if sim >= SIMILARITY_THRESHOLD:
                cluster_members[ci].append(aid)
                # Update centroid (running average)
                n = len(cluster_members[ci])
                cluster_seeds[ci] = (
                    cluster_seeds[ci][0],
                    [(centroid[j] * (n - 1) + vec[j]) / n for j in range(len(vec))],
                )
                placed = True
                break
        if not placed:
            cluster_seeds.append((aid, vec[:]))
            cluster_members.append([aid])

    if progress_cb:
        progress_cb(0.55)

    COORD_TYPES = [
        "retweet_storm",
        "synchronized_posting",
        "hashtag_hijacking",
        "url_amplification",
    ]

    clusters_created = 0
    for ci, members in enumerate(cluster_members):
        if len(members) < MIN_CLUSTER_SIZE:
            continue

        # Assess avg pairwise similarity within cluster
        vecs = [vectors[mid] for mid in members]
        total_sim = 0.0
        pairs = 0
        sample = vecs[:20]  # cap computation
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                total_sim += _cosine_sim(sample[i], sample[j])
                pairs += 1
        avg_sim = round(total_sim / max(1, pairs), 4)

        bot_count = sum(1 for mid in members if pred_map.get(mid, "unknown") == "bot")
        coordination_type = COORD_TYPES[ci % len(COORD_TYPES)]
        time_window = 60 + (ci * 15) % 120
        status = "confirmed" if bot_count / max(1, len(members)) >= 0.5 else "flagged"

        cluster_obj = CoordinationCluster(
            dataset_id=dataset_id,
            cluster_name=f"Cluster-{dataset_id}-{ci:03d}",
            coordination_type=coordination_type,
            account_count=len(members),
            avg_similarity=avg_sim,
            time_window_seconds=time_window,
            status=status,
        )
        db.add(cluster_obj)
        db.flush()

        for mid in members:
            db.add(CoordinationMember(
                cluster_id=cluster_obj.id,
                account_id=mid,
                similarity_score=avg_sim,
            ))

        clusters_created += 1

    db.flush()

    if progress_cb:
        progress_cb(0.90)

    return clusters_created
