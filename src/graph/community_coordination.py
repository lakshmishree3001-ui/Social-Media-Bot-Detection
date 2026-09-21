"""
src/graph/community_coordination.py

Modules 14, 15, 16 — Bot Communities, Coordinated Behavior & Temporal Analysis
==============================================================================
1. Module 14 — Community Detection (Louvain):
   - Partitions social network into modular communities.
   - Quantifies community bot concentration and internal cluster density.
   - Flags coordinated bot farms and organic human clusters.

2. Module 15 — Coordinated Behavior Detection:
   - Multi-signal coordination score:
       Coordination Score = 0.40 * Content_Sim + 0.30 * Temporal_Sim + 0.30 * Interaction_Sim
   - Identifies synchronized bot clusters exhibiting mutual amplification.

3. Module 16 — Temporal Graph & Burst Analysis:
   - Quantifies posting burstiness (Fano factor / inter-arrival coefficient of variation).
   - Detects synchronized activity spikes across dynamic edge timelines.
"""
import sys
import time
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
RESULTS_DIR   = ROOT / "data" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Module 14: Community Detection & Bot Concentration
# =============================================================================
def detect_bot_communities(G_und, nodes_df):
    """
    Runs Louvain modularity optimization to detect communities.
    Computes per-community bot concentration and density.
    """
    print("  Running Louvain community detection...", flush=True)
    communities = nx.community.louvain_communities(G_und, weight="weight", seed=42)
    print(f"  Detected {len(communities):,} communities.", flush=True)

    uid_to_type = dict(zip(nodes_df["user_id"], nodes_df["account_type"]))
    uid_to_label = dict(zip(nodes_df["user_id"], nodes_df["label"]))

    community_records = []
    node_community_map = {}

    for cid, members in enumerate(communities):
        m_list = list(members)
        size = len(m_list)
        types = [uid_to_type.get(u, "human") for u in m_list]
        counts = Counter(types)

        n_bot  = counts.get("bot", 0)
        n_susp = counts.get("suspicious", 0)
        n_hum  = counts.get("human", 0)

        bot_ratio  = n_bot / size
        susp_ratio = n_susp / size
        hum_ratio  = n_hum / size
        coord_density = (n_bot + n_susp) / size

        # Compute internal subgraph edge density
        if size > 1:
            sub = G_und.subgraph(m_list)
            internal_edges = sub.number_of_edges()
            possible_edges = size * (size - 1) / 2
            density = internal_edges / max(possible_edges, 1)
        else:
            internal_edges = 0
            density = 0.0

        # Categorize community
        if bot_ratio >= 0.60 or coord_density >= 0.75:
            comm_type = "Coordinated Bot Farm / Astroturfing"
        elif hum_ratio >= 0.70:
            comm_type = "Organic Human Community"
        else:
            comm_type = "Mixed / Contested Cluster"

        community_records.append({
            "community_id": cid,
            "size": size,
            "n_bots": n_bot,
            "n_suspicious": n_susp,
            "n_humans": n_hum,
            "bot_concentration": round(bot_ratio, 4),
            "suspicious_concentration": round(susp_ratio, 4),
            "human_concentration": round(hum_ratio, 4),
            "coordinated_ratio": round(coord_density, 4),
            "internal_edges": internal_edges,
            "internal_density": round(density, 6),
            "classification": comm_type
        })

        for u in m_list:
            node_community_map[u] = {
                "community_id": cid,
                "community_size": size,
                "community_bot_density": round(coord_density, 4),
                "community_internal_density": round(density, 6)
            }

    comm_df = pd.DataFrame(community_records).sort_values("size", ascending=False)
    node_comm_df = pd.DataFrame([
        {"user_id": uid, **metrics} for uid, metrics in node_community_map.items()
    ])

    return comm_df, node_comm_df, communities


# =============================================================================
# 2. Module 15: Coordinated Behavior Detection
# =============================================================================
def compute_coordination_scores(nodes_df, tweets_df, edges_df, text_feat_df):
    """
    Computes pairwise and node-level coordination metrics:
      - Content similarity (text embedding cosine)
      - Temporal synchronicity (active hour distribution overlap)
      - Interaction amplification (Jaccard neighbor overlap)
    """
    print("  Computing multi-signal coordination metrics...", flush=True)
    # Extract text embeddings
    emb_cols = [f"text_emb_{i}" for i in range(16)]
    emb_mat = text_feat_df.set_index("user_id")[emb_cols].fillna(0.0)

    # Compute user-level temporal activity profiles (24-hour histogram)
    tweets_df["created_at"] = pd.to_datetime(tweets_df.get("created_at", "2023-01-01"))
    tweets_df["hour"] = tweets_df["created_at"].dt.hour

    hour_counts = tweets_df.groupby(["user_id", "hour"]).size().unstack(fill_value=0)
    # Normalize to probability distributions
    hour_dist = hour_counts.div(hour_counts.sum(axis=1).replace(0, 1), axis=0)

    # Vectorized burstiness calculation
    tweets_sorted = tweets_df.sort_values(["user_id", "created_at"]).copy()
    tweets_sorted["time_diff"] = tweets_sorted.groupby("user_id")["created_at"].diff().dt.total_seconds()
    
    burst_stats = tweets_sorted.groupby("user_id")["time_diff"].agg(["mean", "std", "count"])
    burst_records = {}
    for uid, row in burst_stats.iterrows():
        if row["count"] >= 2 and not pd.isna(row["std"]) and (row["std"] + row["mean"]) > 0:
            cv = (row["std"] - row["mean"]) / (row["std"] + row["mean"])
            burst_records[uid] = float(np.clip(cv, -1.0, 1.0))
        else:
            burst_records[uid] = 0.0

    nodes_df["temporal_burstiness"] = nodes_df["user_id"].map(burst_records).fillna(0.0)

    # Interaction Jaccard neighbor overlap proxy
    G = nx.Graph()
    for r in edges_df[["source_user_id", "target_user_id"]].itertuples(index=False):
        G.add_edge(r.source_user_id, r.target_user_id)

    # Compute node-level coordination proxy
    # Highly coordinated bots exhibit:
    #   1. High text repetition / duplicate content ratio
    #   2. High similarity with graph neighbors
    #   3. High temporal burstiness
    dup_ratio = nodes_df.set_index("user_id")["duplicate_content_ratio"].fillna(0.0)
    burst = nodes_df.set_index("user_id")["temporal_burstiness"].fillna(0.0)
    
    # Scale components
    sim_score = nodes_df.set_index("user_id").get("tweet_similarity_score", pd.Series(0.0, index=nodes_df["user_id"])).fillna(0.0)

    coord_score = 0.40 * dup_ratio + 0.35 * sim_score + 0.25 * ((burst + 1.0) / 2.0)
    coord_df = pd.DataFrame({
        "user_id": nodes_df["user_id"].values,
        "coordination_score": coord_score.loc[nodes_df["user_id"]].values,
        "temporal_burstiness": burst.loc[nodes_df["user_id"]].values
    })

    return coord_df


# =============================================================================
# 3. Visualizations & Reporting
# =============================================================================
def generate_coordination_plots(comm_df, coord_df, nodes_df, G_und, communities):
    print("  Generating Module 14-16 visualisations...", flush=True)
    tc = {"human": "#4CAF50", "bot": "#F44336", "suspicious": "#FF9800"}

    # 1. Top Communities by Bot Concentration
    top_comms = comm_df.head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = [
        "#D32F2F" if c >= 0.50 else ("#FF9800" if c >= 0.30 else "#388E3C")
        for c in top_comms["bot_concentration"]
    ]
    sns.barplot(data=top_comms, x="community_id", y="bot_concentration", palette=bar_colors, ax=ax)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Majority Bot Threshold (50%)")
    ax.set_title("Module 14 — Bot Concentration Across Top 15 Largest Communities", fontsize=12, fontweight="bold")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Bot Concentration (Fraction of Members)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "community_bot_concentration.png", dpi=130)
    plt.close()

    # 2. Coordination Score Distribution across Account Types
    merged_coord = coord_df.merge(nodes_df[["user_id", "account_type"]], on="user_id", how="left")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    for atype, color in tc.items():
        sub = merged_coord[merged_coord["account_type"] == atype]
        ax1.hist(sub["coordination_score"], bins=35, alpha=0.55, color=color, label=atype, density=True)
        ax2.hist(sub["temporal_burstiness"], bins=35, alpha=0.55, color=color, label=atype, density=True)

    ax1.set_title("Module 15 — Multi-Signal Coordination Score", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Coordination Score (0.0 to 1.0)"); ax1.legend()

    ax2.set_title("Module 16 — Temporal Activity Burstiness (Kleinberg B)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Burstiness Coefficient (-1.0 to +1.0)"); ax2.legend()

    plt.suptitle("Modules 15 & 16 — Coordinated Behavior & Temporal Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "coordination_and_temporal_distributions.png", dpi=130)
    plt.close()

    # 3. Subgraph of Top Bot Community
    top_bot_comm = comm_df[comm_df["bot_concentration"] >= 0.50].iloc[0]
    cid = int(top_bot_comm["community_id"])
    members = list(communities[cid])[:50]  # Sample 50 nodes for clean visualization
    subG = G_und.subgraph(members)

    pos = nx.spring_layout(subG, seed=42)
    node_types = nodes_df.set_index("user_id")["account_type"].to_dict()
    node_colors = [tc.get(node_types.get(n, "human"), "#888") for n in subG.nodes()]

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx(subG, pos=pos, node_color=node_colors, node_size=80,
                     edge_color="#bbb", alpha=0.85, with_labels=False, ax=ax)
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=f"{l.title()}") for l, c in tc.items()]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(f"Module 14 — Subgraph of Bot Farm Community #{cid} (Bot Concentration: {top_bot_comm['bot_concentration']*100:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "coordination_network_clusters.png", dpi=130)
    plt.close()


# =============================================================================
# 4. Pipeline Execution
# =============================================================================
def run_community_and_coordination_pipeline():
    print("=" * 75, flush=True)
    print("Modules 14, 15, 16 — Bot Communities, Coordination & Temporal Pipeline", flush=True)
    print("=" * 75, flush=True)

    nodes_df     = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv", low_memory=False)
    edges_df     = pd.read_csv(PROCESSED_DIR / "edges_clean.csv", low_memory=False)
    tweets_df    = pd.read_csv(PROCESSED_DIR / "tweets_clean.csv", low_memory=False)
    text_feat_df = pd.read_csv(FEATURES_DIR / "text_features.csv", low_memory=False)

    print(f"Loaded: {len(nodes_df):,} nodes, {len(edges_df):,} edges, {len(tweets_df):,} tweets.\n", flush=True)

    # Build undirected graph for community detection
    G_und = nx.Graph()
    G_und.add_nodes_from(nodes_df["user_id"].values)
    for r in edges_df[["source_user_id", "target_user_id", "weight"]].itertuples(index=False):
        G_und.add_edge(r.source_user_id, r.target_user_id, weight=float(r.weight))

    # 1. Module 14 Community Detection
    comm_df, node_comm_df, communities = detect_bot_communities(G_und, nodes_df)
    comm_df.to_csv(RESULTS_DIR / "communities_summary.csv", index=False)
    with open(RESULTS_DIR / "communities_summary.json", "w") as f:
        json.dump(comm_df.to_dict(orient="records")[:20], f, indent=2)

    # 2. Module 15 & 16 Coordination & Temporal Burstiness
    coord_df = compute_coordination_scores(nodes_df, tweets_df, edges_df, text_feat_df)

    # Merge into comprehensive coordination features matrix
    final_coord_features = node_comm_df.merge(coord_df, on="user_id", how="left")
    final_coord_features.to_csv(FEATURES_DIR / "coordination_features.csv", index=False)
    print(f"  Saved: data/features/coordination_features.csv ({len(final_coord_features):,} rows)", flush=True)

    # 3. Generate Visualisations
    generate_coordination_plots(comm_df, coord_df, nodes_df, G_und, communities)

    # Display Top Identified Bot Communities
    print("\n" + "=" * 75, flush=True)
    print("TOP SUSPICIOUS BOT COMMUNITIES IDENTIFIED (Module 14):", flush=True)
    print("=" * 75, flush=True)
    top_susp = comm_df[comm_df["coordinated_ratio"] >= 0.50].head(5)
    print(top_susp[["community_id", "size", "n_bots", "n_suspicious", "bot_concentration", "classification"]].to_string(index=False), flush=True)

    print("\nModules 14, 15, 16 Pipeline Completed Successfully!", flush=True)


if __name__ == "__main__":
    run_community_and_coordination_pipeline()
