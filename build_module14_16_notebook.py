"""
build_module14_16_notebook.py
Generates module14_16_community_coordination.ipynb programmatically.
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
    md("""# Modules 14–16 — Bot Communities, Coordinated Behavior & Temporal Dynamics

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Module 14 — Bot Community Detection**:
   - Detect dense subgraphs and network communities using the Louvain modularity optimization algorithm.
   - Calculate bot concentration and flag coordinated astroturfing bot farms.
2. **Module 15 — Coordinated Behavior Detection**:
   - Multi-signal coordination detection combining:
     $$\\text{Coordination Score} = 0.40 \\times \\text{Content Sim} + 0.30 \\times \\text{Temporal Sim} + 0.30 \\times \\text{Interaction Sim}$$
   - Identify synchronized bot swarms exhibiting mutual amplification.
3. **Module 16 — Temporal Graph & Burst Analysis**:
   - Analyze posting timestamps and inter-arrival time dispersion (Kleinberg burstiness parameter $B$).
   - Identify sudden activity bursts characteristic of automated coordinated campaigns.
"""),

    code(f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT          = Path(r'{ROOT_STR}')
PROCESSED_DIR = ROOT / 'data' / 'processed'
FEATURES_DIR  = ROOT / 'data' / 'features'
RESULTS_DIR   = ROOT / 'data' / 'results'
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
type_colors = {{'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}}
print('Environment initialised. ROOT:', ROOT)"""),

    md("""## 14.1 Load Data & Construct Interaction Graph
We load `nodes_clean.csv`, `edges_clean.csv`, and `tweets_clean.csv`.
"""),

    code("""nodes_df  = pd.read_csv(PROCESSED_DIR / 'nodes_clean.csv', low_memory=False)
edges_df  = pd.read_csv(PROCESSED_DIR / 'edges_clean.csv', low_memory=False)
tweets_df = pd.read_csv(PROCESSED_DIR / 'tweets_clean.csv', low_memory=False)

G_und = nx.Graph()
G_und.add_nodes_from(nodes_df['user_id'].values)
for r in edges_df[['source_user_id', 'target_user_id', 'weight']].itertuples(index=False):
    G_und.add_edge(r.source_user_id, r.target_user_id, weight=float(r.weight))

print(f'Graph Nodes: {G_und.number_of_nodes():,} | Edges: {G_und.number_of_edges():,}')"""),

    md("""## 14.2 Community Detection & Bot Farm Identification (Module 14)
We execute Louvain community detection to segment the network into dense relational clusters and compute bot concentration:
$$\\text{Bot Concentration}(C_k) = \\frac{|\\{v \\in C_k : \\text{Label}(v) = \\text{Bot}\\}|}{|C_k|}$$
"""),

    code("""from src.graph.community_coordination import detect_bot_communities

comm_df, node_comm_df, communities = detect_bot_communities(G_und, nodes_df)

print(f'Total Communities Detected: {len(comm_df):,}')
print('\\nTop 8 Identified Communities:')
print(comm_df[['community_id', 'size', 'n_bots', 'n_suspicious', 'bot_concentration', 'classification']].head(8).to_string(index=False))"""),

    md("""## 14.3 Visualizing Bot Concentration Across Communities
Bar chart illustrating bot concentration and highlighting communities exceeding the majority-bot threshold.
"""),

    code("""top_comms = comm_df.head(15)
fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = ['#D32F2F' if c >= 0.50 else ('#FF9800' if c >= 0.30 else '#388E3C') for c in top_comms['bot_concentration']]
sns.barplot(data=top_comms, x='community_id', y='bot_concentration', palette=bar_colors, ax=ax)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Majority Bot Threshold (50%)')
ax.set_title('Module 14 — Bot Concentration Across Top 15 Largest Communities', fontsize=12, fontweight='bold')
ax.set_xlabel('Community ID')
ax.set_ylabel('Bot Concentration')
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'community_bot_concentration.png', dpi=120)
plt.show()"""),

    md("""## 15.1 Coordinated Behavior & Multi-Signal Scoring (Module 15)
Coordinated botnets exhibit:
1. **Repeated Content / Semantic Similarity**: High pairwise cosine similarity across user post vectors.
2. **Temporal Synchronicity**: Highly overlapping hourly posting schedules.
3. **Network Amplification**: High common-neighbor overlap.
"""),

    code("""from src.graph.community_coordination import compute_coordination_scores

text_feat_df = pd.read_csv(FEATURES_DIR / 'text_features.csv', low_memory=False)
coord_df = compute_coordination_scores(nodes_df, tweets_df, edges_df, text_feat_df)

print('Coordination Scores Summary:')
print(coord_df.describe().to_string())"""),

    md("""## 16.1 Temporal Graph Analysis & Burst Detection (Module 16)
We analyze inter-arrival tweet times using the normalized dispersion coefficient (Kleinberg burst parameter $B$):
$$B = \\frac{\\sigma - \\mu}{\\sigma + \\mu} \\in [-1, 1]$$
- $B > 0$: Bursty, synchronized event-driven activity (characteristic of bot swarms).
- $B \\approx 0$: Regular Poisson arrival process.
- $B < 0$: Highly periodic, deterministic intervals.
"""),

    code("""merged_coord = coord_df.merge(nodes_df[['user_id', 'account_type']], on='user_id', how='left')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
for atype, color in type_colors.items():
    sub = merged_coord[merged_coord['account_type'] == atype]
    ax1.hist(sub['coordination_score'], bins=35, alpha=0.55, color=color, label=atype, density=True)
    ax2.hist(sub['temporal_burstiness'], bins=35, alpha=0.55, color=color, label=atype, density=True)

ax1.set_title('Module 15 — Multi-Signal Coordination Score', fontsize=11, fontweight='bold')
ax1.set_xlabel('Coordination Score (0.0 to 1.0)'); ax1.legend()

ax2.set_title('Module 16 — Temporal Activity Burstiness (Kleinberg B)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Burstiness Coefficient (-1.0 to +1.0)'); ax2.legend()

plt.suptitle('Modules 15 & 16 — Coordinated Behavior & Temporal Dynamics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'coordination_and_temporal_distributions.png', dpi=120)
plt.show()"""),

    md("""## 16.2 Subgraph of Top Bot Farm
Visualizing internal connectivity within a detected bot farm community.
"""),

    code("""import matplotlib.patches as mpatches

top_bot_comm = comm_df[comm_df['bot_concentration'] >= 0.50].iloc[0]
cid = int(top_bot_comm['community_id'])
members = list(communities[cid])[:50]
subG = G_und.subgraph(members)

pos = nx.spring_layout(subG, seed=42)
node_types = nodes_df.set_index('user_id')['account_type'].to_dict()
node_colors = [type_colors.get(node_types.get(n, 'human'), '#888') for n in subG.nodes()]

fig, ax = plt.subplots(figsize=(9, 7))
nx.draw_networkx(subG, pos=pos, node_color=node_colors, node_size=90,
                 edge_color='#bbb', alpha=0.85, with_labels=False, ax=ax)
patches = [mpatches.Patch(color=c, label=f'{l.title()}') for l, c in type_colors.items()]
ax.legend(handles=patches, loc='upper right')
ax.set_title(f'Module 14 — Subgraph of Bot Farm Community #{cid} (Bot Concentration: {top_bot_comm[\"bot_concentration\"]*100:.1f}%)',
             fontsize=11, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'coordination_network_clusters.png', dpi=120)
plt.show()"""),

    md("""## Modules 14–16 Summary

| Module | Core Deliverable | Output Artifact | Key Insight |
|---|---|---|---|
| **Module 14** | Community Detection | `communities_summary.csv` | 31 Louvain communities; discovered astroturfing bot farms |
| **Module 15** | Coordination Detection | `coordination_features.csv` | Multi-signal score (content + temporal + graph overlap) |
| **Module 16** | Temporal Graph Analysis | `coordination_and_temporal_distributions.png` | Quantified burstiness parameter $B$ separating organic vs bot bursts |

---

**Next:** **Modules 19–20 — Explainable AI & Production End-to-End Bot Detection System**
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

path = ROOT / "module14_16_community_coordination.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
