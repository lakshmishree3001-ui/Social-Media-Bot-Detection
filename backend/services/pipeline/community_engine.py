"""
backend/services/pipeline/community_engine.py

Stage 5: Community detection using the Louvain algorithm on the
dataset-scoped interaction graph. Persists Community + CommunityMember rows.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict

import networkx as nx
from sqlalchemy.orm import Session

from backend.models.entities import (
    Account, Edge, Community, CommunityMember, Prediction
)

try:
    from community import best_partition as louvain_partition  # python-louvain
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False


def _label_propagation_partition(G: nx.Graph) -> Dict:
    """Fallback when python-louvain is not installed."""
    communities_gen = nx.community.label_propagation_communities(G)
    partition: Dict[int, int] = {}
    for cid, members in enumerate(communities_gen):
        for node in members:
            partition[node] = cid
    return partition


def detect_communities(
    dataset_id: int,
    db: Session,
    progress_cb=None,
) -> int:
    """
    Build undirected graph for the dataset, run community detection,
    persist Community + CommunityMember rows scoped to dataset_id.
    Returns number of communities created.
    """
    # Load accounts
    accounts: List[Account] = (
        db.query(Account).filter(Account.dataset_id == dataset_id).all()
    )
    if not accounts:
        return 0

    id_set = {a.id for a in accounts}
    acc_map = {a.id: a for a in accounts}

    # Load ground-truth-aware prediction labels (use ground_truth as fallback)
    pred_rows = (
        db.query(Prediction)
        .filter(Prediction.dataset_id == dataset_id)
        .all()
    )
    pred_map = {p.account_id: p for p in pred_rows}

    def get_label(aid: int) -> str:
        p = pred_map.get(aid)
        if p:
            return p.predicted_class
        return acc_map[aid].ground_truth if aid in acc_map else "unknown"

    # Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(id_set)
    edges = db.query(Edge).filter(Edge.dataset_id == dataset_id).all()
    for e in edges:
        if e.source_id in id_set and e.target_id in id_set:
            if G.has_edge(e.source_id, e.target_id):
                G[e.source_id][e.target_id]["weight"] += e.weight
            else:
                G.add_edge(e.source_id, e.target_id, weight=e.weight)

    if progress_cb:
        progress_cb(0.20)

    # Run partition
    if _HAS_LOUVAIN:
        try:
            partition = louvain_partition(G, weight="weight", random_state=42)
        except Exception:
            partition = _label_propagation_partition(G)
    else:
        partition = _label_propagation_partition(G)

    if progress_cb:
        progress_cb(0.55)

    # Delete existing communities for this dataset
    old_comms = db.query(Community).filter(Community.dataset_id == dataset_id).all()
    for c in old_comms:
        db.delete(c)
    db.flush()

    # Group nodes by community
    comm_groups: Dict[int, List[int]] = {}
    for node_id, comm_id in partition.items():
        comm_groups.setdefault(comm_id, []).append(node_id)

    # Build edge set for internal-edge counting
    edge_set = set()
    for e in edges:
        u, v = min(e.source_id, e.target_id), max(e.source_id, e.target_id)
        edge_set.add((u, v))

    communities_created = 0
    for comm_idx, (comm_id, member_ids) in enumerate(comm_groups.items()):
        labels = [get_label(mid) for mid in member_ids]
        label_counts = Counter(labels)
        size = len(member_ids)
        n_bots = label_counts.get("bot", 0)
        n_susp = label_counts.get("suspicious", 0)
        n_humans = label_counts.get("human", 0)

        # Internal edges
        member_set = set(member_ids)
        int_edges = sum(
            1 for u, v in edge_set
            if u in member_set and v in member_set
        )
        max_edges = size * (size - 1) // 2
        density = round(int_edges / max_edges, 6) if max_edges > 0 else 0.0

        bot_conc = round(n_bots / max(1, size), 4)
        susp_conc = round(n_susp / max(1, size), 4)
        hum_conc = round(n_humans / max(1, size), 4)

        if bot_conc >= 0.60:
            classification = "Bot Farm"
        elif bot_conc + susp_conc <= 0.15:
            classification = "Authentic Cluster"
        elif density >= 0.40 and (n_bots + n_susp) / max(1, size) >= 0.40:
            classification = "Coordinated Network"
        else:
            classification = "Mixed"

        comm_obj = Community(
            dataset_id=dataset_id,
            community_id=comm_id,
            algorithm="Louvain" if _HAS_LOUVAIN else "LabelPropagation",
            size=size,
            n_bots=n_bots,
            n_suspicious=n_susp,
            n_humans=n_humans,
            bot_concentration=bot_conc,
            suspicious_concentration=susp_conc,
            human_concentration=hum_conc,
            coordinated_ratio=round((n_bots + n_susp) / max(1, size), 4),
            internal_edges=int_edges,
            internal_density=density,
            classification=classification,
        )
        db.add(comm_obj)
        db.flush()

        # Members
        for mid in member_ids:
            degree = G.degree(mid)
            if degree >= 10:
                role = "core"
            elif degree >= 3:
                role = "bridge"
            else:
                role = "peripheral"
            db.add(CommunityMember(
                community_id=comm_obj.id,
                account_id=mid,
                role=role,
            ))

        communities_created += 1

    db.flush()

    if progress_cb:
        progress_cb(0.90)

    return communities_created
