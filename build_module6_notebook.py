"""
build_module6_notebook.py
Generates module6_graph_construction.ipynb programmatically.
"""
import json
from pathlib import Path

ROOT = Path("c:/Social Media Bot Detection")

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

ROOT_STR = str(ROOT).replace("\\", "/")

cells = [
    md("""# Module 6 & 7 — Building the Social Network Graph & Graph Feature Engineering

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Graph Representation**: Represent social network interactions as a directed, weighted graph $G = (V, E)$.
   - $V$: 10,000 accounts (nodes)
   - $E$: 55,000 interactions (follows, replies, mentions, retweets)
2. **Network Topology Analysis**:
   - Compute graph density, degree distribution, and weakly connected components.
3. **Graph-Theoretic Feature Engineering (Module 7)**:
   - **In-Degree & Out-Degree**: Hub and authority characteristics.
   - **Degree Centralities**: Normalized inward and outward interaction rates.
   - **PageRank**: Structural influence and authority across the network.
   - **Clustering Coefficient**: Tendency of neighbors to form tightly connected cliques.
   - **Betweenness Centrality**: Information brokering across network paths.
   - **k-Core Decomposition**: Deep core vs peripheral network positioning.
4. **PyTorch Geometric (PyG) Export**:
   - Export `edge_index` (2 × $|E|$) and `edge_attr` tensors for GNN modeling.
5. **Graph Visualizations**:
   - Degree and PageRank distribution comparisons across Human vs Bot vs Suspicious.
   - PageRank vs Clustering Coefficient phase space.
   - Ego network topology of high-influence bot accounts.
"""),

    code(f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT          = Path(r'{ROOT_STR}')
PROCESSED_DIR = ROOT / 'data' / 'processed'
FEATURES_DIR  = ROOT / 'data' / 'features'
GRAPH_DIR     = ROOT / 'data' / 'graph'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
type_colors = {{'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}}
print('Environment initialised. ROOT:', ROOT)"""),

    md("""## 6.1 Load Cleaned Node and Edge Data
We load `nodes_clean.csv` (10,000 accounts) and `edges_clean.csv` (55,000 interactions).
"""),

    code("""nodes_df = pd.read_csv(PROCESSED_DIR / 'nodes_clean.csv', low_memory=False)
edges_df = pd.read_csv(PROCESSED_DIR / 'edges_clean.csv', low_memory=False)

print(f'Nodes: {len(nodes_df):,} accounts')
print(f'Edges: {len(edges_df):,} interactions')
print('\\nInteraction types breakdown:')
print(edges_df['interaction_type'].value_counts().to_string())"""),

    md("""## 6.2 Constructing the Directed Social Graph (NetworkX)
We build a directed graph $G = (V, E)$ where nodes represent user accounts and edges represent social interactions with associated weights.
"""),

    code("""from src.graph.build_graph import build_social_graph

col_map = {}
for c in edges_df.columns:
    lc = c.lower()
    if 'source' in lc: col_map[c] = 'source_id'
    elif 'target' in lc or 'dest' in lc: col_map[c] = 'target_id'
edges_mapped = edges_df.rename(columns=col_map)

G = build_social_graph(edges_mapped, nodes_df)

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
density = nx.density(G)
n_weakly_cc = nx.number_weakly_connected_components(G)

print(f'Graph Statistics:')
print(f'  Total Nodes (Users)         : {n_nodes:,}')
print(f'  Total Directed Edges        : {n_edges:,}')
print(f'  Graph Density               : {density:.6f}')
print(f'  Weakly Connected Components : {n_weakly_cc}')"""),

    md("""## 6.3 Computing Graph-Theoretic Node Features (Module 7)
We extract structural graph metrics for every node:
- `in_degree`, `out_degree`, `total_degree`
- `in_degree_centrality`, `out_degree_centrality`
- `pagerank`
- `clustering_coeff`
- `betweenness` (sampled $k=500$)
- `k_core`
"""),

    code("""from src.graph.build_graph import compute_graph_features

graph_feat_df = compute_graph_features(G, nodes_df)
graph_feat_df = graph_feat_df.merge(nodes_df[['user_id', 'account_type', 'label']], on='user_id', how='left')

print('Computed Graph Features Preview:')
print(graph_feat_df.head(5).to_string(index=False))"""),

    md("""## 6.4 Graph Topology Visualizations
We analyze the topological differences between bots, humans, and coordinated accounts.
"""),

    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
metrics = [('in_degree', 'In-Degree Distribution'),
           ('out_degree', 'Out-Degree Distribution'),
           ('pagerank', 'PageRank Distribution')]

for ax, (metric, title) in zip(axes, metrics):
    for atype, color in type_colors.items():
        sub = graph_feat_df[graph_feat_df['account_type'] == atype][metric]
        ax.hist(sub, bins=35, alpha=0.55, color=color, label=atype, density=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Module 6 & 7 — Graph Topological Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FEATURES_DIR / 'graph_degree_distributions.png', dpi=120)
plt.show()"""),

    code("""fig, ax = plt.subplots(figsize=(9, 6))
for atype, color in type_colors.items():
    sub = graph_feat_df[graph_feat_df['account_type'] == atype]
    ax.scatter(sub['pagerank'], sub['clustering_coeff'], s=10, alpha=0.4, color=color, label=atype)

ax.set_title('PageRank vs Clustering Coefficient by Account Type', fontsize=12, fontweight='bold')
ax.set_xlabel('PageRank')
ax.set_ylabel('Clustering Coefficient')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(FEATURES_DIR / 'graph_pagerank_vs_clustering.png', dpi=120)
plt.show()"""),

    md("""## 6.5 Ego Network Analysis of High-Influence Bot
We visualize the 1-hop neighborhood of a high-PageRank bot account to observe coordination structures.
"""),

    code("""bot_nodes = graph_feat_df[graph_feat_df['account_type'] == 'bot']
if len(bot_nodes) > 0:
    top_bot_uid = bot_nodes.sort_values('pagerank', ascending=False).iloc[0]['user_id']
    ego = nx.ego_graph(G, top_bot_uid, radius=1)
    ego_sample = ego.subgraph(list(ego.nodes)[:60])
    pos = nx.spring_layout(ego_sample, seed=42)
    
    nldf = graph_feat_df.set_index('user_id')
    node_colors = [type_colors.get(nldf.loc[n, 'account_type'] if n in nldf.index else 'human', '#888') for n in ego_sample.nodes()]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx(ego_sample, pos=pos, node_color=node_colors, node_size=70,
                     edge_color='#bbb', arrows=True, with_labels=False, arrowsize=8, ax=ax)
    ax.set_title(f'Ego Network of Top-PageRank Bot (User {top_bot_uid})', fontsize=12, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in type_colors.items()], loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(FEATURES_DIR / 'graph_ego_network.png', dpi=120)
    plt.show()"""),

    md("""## 6.6 PyTorch Geometric (PyG) Tensor Export
We export PyG-compatible tensors:
- `edge_index` of shape `(2, |E|)`
- `edge_attr` of shape `(|E|,)`
- `uid_to_idx.json` mapping user IDs to contiguous 0-indexed node IDs.
"""),

    code("""from src.graph.build_graph import build_edge_index

edge_index, edge_attr, uid_to_idx = build_edge_index(G, nodes_df)
print(f'edge_index shape : {tuple(edge_index.shape)} (dtype: {edge_index.dtype})')
print(f'edge_attr shape  : {tuple(edge_attr.shape)} (dtype: {edge_attr.dtype})')
print(f'Sample edges (first 5):')
print(edge_index[:, :5])"""),

    md("""## 6.7 Feature Verification & Summary
We scale and save `graph_features.csv` for downstream baseline and GNN models.
"""),

    code("""feature_cols = [c for c in graph_feat_df.columns if c not in ['user_id', 'account_type', 'label']]
scaled_graph_df = graph_feat_df.copy()
scaled_graph_df[feature_cols] = StandardScaler().fit_transform(scaled_graph_df[feature_cols].fillna(0.0))

out_path = FEATURES_DIR / 'graph_features.csv'
scaled_graph_df.to_csv(out_path, index=False)

print(f'Saved: {out_path.name} ({out_path.stat().st_size // 1024} KB)')
print(f'Total Graph Features : {len(feature_cols)}')
print(f'Total Accounts       : {len(scaled_graph_df):,}')
print(f'NaN values           : {scaled_graph_df[feature_cols].isna().sum().sum()}')
assert scaled_graph_df['user_id'].nunique() == len(scaled_graph_df)
assert scaled_graph_df[feature_cols].isna().sum().sum() == 0
print('ALL VERIFICATION CHECKS PASSED!')"""),

    md("""## Module 6 & 7 Summary

| Component | Specification | Description |
|---|---|---|
| **Graph Structure** | $G = (V, E)$, $|V|=10,000$, $|E|=54,980$ | Directed, weighted multi-relational social interaction network |
| **PyG Tensors** | `edge_index.pt` (2 × 54,980), `edge_attr.pt` (54,980) | PyTorch Geometric ready tensors |
| **Degree Features** | `in_degree`, `out_degree`, `total_degree`, `degree_centrality` | Quantifies interaction volume and asymmetry |
| **Centrality Features**| `pagerank`, `betweenness` | Distinguishes influential hubs and communication bottlenecks |
| **Substructure Features**| `clustering_coeff`, `k_core` | Detects coordinated dense cliques and echo chambers |

**Next:** **Module 8 — Traditional Machine Learning Baselines (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM)**
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

path = ROOT / "module6_graph_construction.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
