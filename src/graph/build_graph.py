"""
src/graph/build_graph.py
Module 6 - Building the Social Network Graph
"""
import json, warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
GRAPH_DIR     = ROOT / "data" / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def build_social_graph(edges_df, nodes_df):
    print(f"  Building DiGraph from {len(edges_df):,} edges...", flush=True)
    G = nx.DiGraph()
    G.add_nodes_from(nodes_df["user_id"].values)
    weight_col = "weight" if "weight" in edges_df.columns else None
    if weight_col:
        agg = edges_df.groupby(["source_id", "target_id"], as_index=False)[weight_col].sum()
        for row in agg.itertuples(index=False):
            G.add_edge(row.source_id, row.target_id, weight=float(getattr(row, weight_col)))
    else:
        for row in edges_df[["source_id", "target_id"]].drop_duplicates().itertuples(index=False):
            G.add_edge(row.source_id, row.target_id, weight=1.0)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)
    return G

def compute_graph_features(G, nodes_df):
    print("  Computing degree features...", flush=True)
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    print("  Computing PageRank...", flush=True)
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, weight="weight")
    print("  Computing clustering coefficients...", flush=True)
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))
    clust = nx.clustering(G_und)
    print("  Computing betweenness centrality (k=500 sample)...", flush=True)
    betw = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), normalized=True, seed=42)
    print("  Computing k-core numbers...", flush=True)
    kcore = nx.core_number(G_und)
    in_cent  = nx.in_degree_centrality(G)
    out_cent = nx.out_degree_centrality(G)
    records = []
    for uid in nodes_df["user_id"].values:
        records.append({
            "user_id":               uid,
            "in_degree":             in_deg.get(uid, 0),
            "out_degree":            out_deg.get(uid, 0),
            "total_degree":          in_deg.get(uid, 0) + out_deg.get(uid, 0),
            "in_degree_centrality":  in_cent.get(uid, 0.0),
            "out_degree_centrality": out_cent.get(uid, 0.0),
            "pagerank":              pr.get(uid, 0.0),
            "clustering_coeff":      clust.get(uid, 0.0),
            "betweenness":           betw.get(uid, 0.0),
            "k_core":                kcore.get(uid, 0),
        })
    return pd.DataFrame(records)

def build_edge_index(G, nodes_df):
    uid_to_idx = {uid: i for i, uid in enumerate(nodes_df["user_id"].values)}
    src_list, dst_list, wt_list = [], [], []
    for src, dst, data in G.edges(data=True):
        if src in uid_to_idx and dst in uid_to_idx:
            src_list.append(uid_to_idx[src])
            dst_list.append(uid_to_idx[dst])
            wt_list.append(data.get("weight", 1.0))
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(wt_list, dtype=torch.float)
    return edge_index, edge_attr, uid_to_idx

def run_graph_pipeline():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("=" * 70, flush=True)
    print("Module 6 - Social Network Graph Construction", flush=True)
    print("=" * 70, flush=True)

    nodes_df = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv", low_memory=False)
    edges_df = pd.read_csv(PROCESSED_DIR / "edges_clean.csv", low_memory=False)
    print(f"Loaded {len(nodes_df):,} nodes, {len(edges_df):,} edges.\n", flush=True)

    col_map = {}
    for c in edges_df.columns:
        lc = c.lower()
        if "source" in lc: col_map[c] = "source_id"
        elif "target" in lc or "dest" in lc: col_map[c] = "target_id"
    edges_df = edges_df.rename(columns=col_map)
    print(f"Edge columns: {list(edges_df.columns)}", flush=True)

    print("\n[Step 1/4] Building NetworkX DiGraph", flush=True)
    G = build_social_graph(edges_df, nodes_df)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    n_weakly_cc = nx.number_weakly_connected_components(G)
    print(f"  Nodes: {n_nodes:,} | Edges: {n_edges:,} | Density: {density:.6f}", flush=True)
    print(f"  Weakly connected components: {n_weakly_cc:,}", flush=True)

    print("\n[Step 2/4] Computing graph-theoretic node features", flush=True)
    graph_feat_df = compute_graph_features(G, nodes_df)
    graph_feat_df = graph_feat_df.merge(nodes_df[["user_id","account_type","label"]], on="user_id", how="left")
    feature_cols = [c for c in graph_feat_df.columns if c not in ["user_id","account_type","label"]]
    graph_feat_df[feature_cols] = graph_feat_df[feature_cols].fillna(0.0)
    graph_feat_df[feature_cols] = StandardScaler().fit_transform(graph_feat_df[feature_cols].astype(float))
    out_csv = FEATURES_DIR / "graph_features.csv"
    graph_feat_df.to_csv(out_csv, index=False)
    with open(FEATURES_DIR / "graph_feature_columns.json", "w") as f:
        json.dump({"graph_feature_columns": feature_cols, "n_graph_features": len(feature_cols)}, f, indent=2)
    print(f"  Saved graph_features.csv: shape={graph_feat_df.shape}  {out_csv.stat().st_size // 1024} KB", flush=True)

    print("\n[Step 3/4] Building PyG edge_index tensors", flush=True)
    edge_index, edge_attr, uid_to_idx = build_edge_index(G, nodes_df)
    torch.save(edge_index, GRAPH_DIR / "edge_index.pt")
    torch.save(edge_attr,  GRAPH_DIR / "edge_attr.pt")
    with open(GRAPH_DIR / "uid_to_idx.json", "w") as f:
        json.dump({str(k): v for k, v in uid_to_idx.items()}, f)
    print(f"  edge_index: {tuple(edge_index.shape)}  edge_attr: {tuple(edge_attr.shape)}", flush=True)

    print("\n[Step 4/4] Generating visualisations", flush=True)
    tc = {"human": "#4CAF50", "bot": "#F44336", "suspicious": "#FF9800"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, ["in_degree","out_degree","pagerank"], ["In-Degree","Out-Degree","PageRank"]):
        for atype, color in tc.items():
            sub = graph_feat_df[graph_feat_df["account_type"]==atype][metric]
            ax.hist(sub, bins=40, alpha=0.55, color=color, label=atype, density=True)
        ax.set_title(f"{title} Distribution", fontsize=10, fontweight="bold"); ax.legend(fontsize=8)
    plt.suptitle("Module 6 - Graph Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FEATURES_DIR / "graph_degree_distributions.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    for atype, color in tc.items():
        sub = graph_feat_df[graph_feat_df["account_type"]==atype]
        ax.scatter(sub["pagerank"], sub["clustering_coeff"], s=6, alpha=0.3, color=color, label=atype)
    ax.set_title("PageRank vs Clustering Coefficient", fontsize=12, fontweight="bold")
    ax.set_xlabel("PageRank (normalised)"); ax.set_ylabel("Clustering Coeff (normalised)"); ax.legend(markerscale=3)
    plt.tight_layout(); plt.savefig(FEATURES_DIR / "graph_pagerank_vs_clustering.png", dpi=120); plt.close()

    bot_nodes = graph_feat_df[graph_feat_df["account_type"]=="bot"]
    if len(bot_nodes) > 0:
        top_bot_uid = bot_nodes.sort_values("pagerank", ascending=False).iloc[0]["user_id"]
        ego = nx.ego_graph(G, top_bot_uid, radius=1)
        ego_nodes = list(ego.nodes)[:60]
        ego_s = ego.subgraph(ego_nodes)
        pos = nx.spring_layout(ego_s, seed=42)
        nldf = graph_feat_df.set_index("user_id")
        ncols = [tc.get(nldf.loc[n,"account_type"] if n in nldf.index else "human","#888") for n in ego_s.nodes()]
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw_networkx(ego_s, pos=pos, node_color=ncols, node_size=60, edge_color="#aaa",
                         arrows=True, with_labels=False, arrowsize=8, ax=ax)
        ax.set_title(f"Ego Network of Top-PageRank Bot (up to 60 nodes)", fontsize=11, fontweight="bold")
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in tc.items()], loc="upper right")
        ax.axis("off"); plt.tight_layout()
        plt.savefig(FEATURES_DIR / "graph_ego_network.png", dpi=120); plt.close()

    stats = {"n_nodes": n_nodes, "n_edges": n_edges, "density": round(density, 8),
             "weakly_cc": n_weakly_cc, "n_graph_features": len(feature_cols)}
    with open(GRAPH_DIR / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    assert graph_feat_df["user_id"].nunique() == len(graph_feat_df)
    print("\nAll assertions passed!", flush=True)
    print(f"  {len(feature_cols)} graph features x {len(graph_feat_df):,} users -> graph_features.csv", flush=True)
    print("Module 6 Graph Construction Completed Successfully!", flush=True)

if __name__ == "__main__":
    run_graph_pipeline()
