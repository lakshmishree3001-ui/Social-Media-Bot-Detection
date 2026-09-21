"""
backend/services/pipeline/graph_engine.py

Stage 4: Build NetworkX graph from account rows, compute per-node
graph-structural features (degree centrality, PageRank, betweenness,
clustering coefficient, k-core), and persist GraphFeature rows.

Also builds the Edge table rows for the dataset.
"""
from __future__ import annotations

import math
import random
from typing import List, Dict

import networkx as nx
from sqlalchemy.orm import Session

from backend.models.entities import Account, GraphFeature, Edge


def build_graph_and_features(
    dataset_id: int,
    db: Session,
    progress_cb=None,
) -> int:
    """
    1. Load all Account rows for dataset_id.
    2. Synthesize a realistic directed follow-graph (when no explicit edge
       data is available from the upload – edges from real uploads come from
       the normalizer stage and are already in the Edge table).
    3. Compute NetworkX centrality measures.
    4. Upsert GraphFeature rows.

    Returns the number of account nodes processed.
    """
    accounts: List[Account] = (
        db.query(Account).filter(Account.dataset_id == dataset_id).all()
    )
    if not accounts:
        return 0

    n = len(accounts)
    id_list = [a.id for a in accounts]
    id_to_idx = {aid: i for i, aid in enumerate(id_list)}

    # Check whether edges already exist for this dataset
    existing_edges = (
        db.query(Edge).filter(Edge.dataset_id == dataset_id).limit(1).first()
    )

    G = nx.DiGraph()
    G.add_nodes_from(id_list)

    if existing_edges is None:
        # Synthesize realistic follow graph:
        # Use followers_count / following_count as degree targets.
        # Small-world + preferential attachment hybrid.
        acc_map = {a.id: a for a in accounts}
        random.seed(dataset_id)

        # Build weighted probability for being followed (proportional to followers)
        weights = [max(1, acc_map[aid].followers_count) for aid in id_list]
        total_w = sum(weights)
        prob = [w / total_w for w in weights]

        edges_to_add: List[tuple] = []
        for i, src_id in enumerate(id_list):
            a = acc_map[src_id]
            desired_out = min(max(1, int(a.following_count * 0.05)), 30)
            targets = random.choices(id_list, weights=prob, k=desired_out * 2)
            targets = [t for t in targets if t != src_id]
            seen: set = set()
            for t in targets:
                if t not in seen and len(seen) < desired_out:
                    seen.add(t)
                    edges_to_add.append((src_id, t))
                    G.add_edge(src_id, t)

        # Persist synthesized edges
        edge_objs = [
            Edge(
                dataset_id=dataset_id,
                source_id=s,
                target_id=t,
                relation_type="follow",
                weight=1.0,
            )
            for s, t in edges_to_add
        ]
        db.bulk_save_objects(edge_objs)
        db.flush()
    else:
        # Load existing edges
        all_edges = (
            db.query(Edge).filter(Edge.dataset_id == dataset_id).all()
        )
        for e in all_edges:
            if e.source_id in id_to_idx and e.target_id in id_to_idx:
                G.add_edge(e.source_id, e.target_id, weight=e.weight)

    if progress_cb:
        progress_cb(0.30)

    # --- Compute centrality measures ---
    num_nodes = G.number_of_nodes()

    # Degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # PageRank (dampened, safe for large graphs)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
    except Exception:
        pr = {nid: 1.0 / max(1, num_nodes) for nid in G.nodes()}

    # Betweenness – approximate for large graphs
    if num_nodes > 2000:
        k_sample = min(500, num_nodes)
        try:
            bw = nx.betweenness_centrality(G, k=k_sample, normalized=True)
        except Exception:
            bw = {nid: 0.0 for nid in G.nodes()}
    else:
        try:
            bw = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            bw = {nid: 0.0 for nid in G.nodes()}

    # Clustering coefficient
    G_undirected = G.to_undirected()
    try:
        cc = nx.clustering(G_undirected)
    except Exception:
        cc = {nid: 0.0 for nid in G.nodes()}

    # K-core
    try:
        core_number = nx.core_number(G_undirected)
    except Exception:
        core_number = {nid: 0 for nid in G.nodes()}

    if progress_cb:
        progress_cb(0.65)

    norm = max(1, num_nodes - 1)

    # Upsert GraphFeature rows
    existing_gf = {
        gf.account_id: gf
        for gf in db.query(GraphFeature)
        .filter(GraphFeature.account_id.in_(id_list))
        .all()
    }

    for aid in id_list:
        gf = existing_gf.get(aid)
        ind = in_deg.get(aid, 0)
        outd = out_deg.get(aid, 0)
        if gf is None:
            gf = GraphFeature(account_id=aid)
            db.add(gf)
        gf.in_degree = ind
        gf.out_degree = outd
        gf.total_degree = ind + outd
        gf.in_degree_centrality = round(ind / norm, 6)
        gf.out_degree_centrality = round(outd / norm, 6)
        gf.pagerank = round(pr.get(aid, 0.0), 8)
        gf.betweenness_centrality = round(bw.get(aid, 0.0), 8)
        gf.clustering_coeff = round(cc.get(aid, 0.0), 6)
        gf.k_core = core_number.get(aid, 0)

    db.flush()

    if progress_cb:
        progress_cb(0.90)

    return n
