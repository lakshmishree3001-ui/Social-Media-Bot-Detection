"""
build_module20_notebook.py
Generates module20_final_system.ipynb programmatically.
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
    md("""# Module 20 — Final End-to-End Social Media Bot Detection System

**Social Media Bot Detection Using GNN**

---

### System Overview
This notebook demonstrates the complete, production-ready AI Bot Detection system integrating all components built across Modules 1 to 20:

```
             SOCIAL MEDIA DATA (Nodes, Tweets, Edges)
                                ↓
        ┌───────────────────────┼───────────────────────┐
        ↓                       ↓                       ↓
    User Data                 Posts                Interactions
        ↓                       ↓                       ↓
 Behavioral Features       NLP Features           Graph Creation
 (35 dimensions)         (36 dimensions)                ↓
        │                       │                 Graph Features
        │                       │                 (9 dimensions)
        └───────────────────────┼───────────────────────┘
                                ↓
                     Multimodal Feature Fusion
                                ↓
                  Graph Neural Network Engine
                     (GCN / GraphSAGE / GAT)
                                ↓
                       AI Decision Core
                                ↓
         ┌──────────────────────┴──────────────────────┐
         ↓                                             ↓
 Individual Detection                       Coordination Detection
         ↓                                             ↓
  Bot Probability & XAI Signals                 Suspicious Communities
```

### 5 Core Deliverables Demonstrated:
1. **Individual Account Prediction** ($P(\\text{Bot}), P(\\text{Suspicious}), P(\\text{Human})$)
2. **Behavioral Diagnostics** (Posting frequency, repetition rate, activity consistency)
3. **Graph-Based Detection** (Ego network inspection, neighbor homophily)
4. **Community Detection** (Louvain community decomposition, bot farm flagging)
5. **Explainable AI (XAI)** (Gradient $\\times$ Input feature attribution, attention distributions)
"""),

    code(f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT         = Path(r'{ROOT_STR}')
RESULTS_DIR  = ROOT / 'data' / 'results'
MODELS_DIR   = ROOT / 'data' / 'models'
FEATURES_DIR = ROOT / 'data' / 'features'
GRAPH_DIR    = ROOT / 'data' / 'graph'
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('End-to-End Bot Detection System Initialised.')"""),

    md("""## 20.1 Load Production Detector Engine
We initialize the unified `SocialMediaBotDetector` with our trained Graph Attention Network (GAT).
"""),

    code("""from src.system.bot_detector import SocialMediaBotDetector

detector = SocialMediaBotDetector(model_type='GAT')
print('Detector Model Loaded:', detector.model.__class__.__name__)
print(f'Total Users in Network : {detector.data.num_nodes:,}')
print(f'Total Directed Edges   : {detector.data.num_edges:,}')
print(f'Total Features Fused   : {detector.data.num_node_features}')"""),

    md("""## 20.2 Case Study 1: Coordinated Bot Account
We run an end-to-end scan on a confirmed bot account and print the structured 5-stage diagnostic report.
"""),

    code("""bot_uid = detector.nodes_df[detector.nodes_df['account_type'] == 'bot'].iloc[0]['user_id']
detector.print_diagnostic_card(bot_uid)"""),

    md("""## 20.3 Case Study 2: Authentic Human Account
We run the identical pipeline on an authentic human account to demonstrate low false-positive risk and clean organic separation.
"""),

    code("""human_uid = detector.nodes_df[detector.nodes_df['account_type'] == 'human'].iloc[0]['user_id']
detector.print_diagnostic_card(human_uid)"""),

    md("""## 20.4 Case Study 3: Suspicious Coordinated Account
We evaluate a borderline/suspicious account exhibiting astroturfing patterns.
"""),

    code("""susp_uid = detector.nodes_df[detector.nodes_df['account_type'] == 'suspicious'].iloc[0]['user_id']
detector.print_diagnostic_card(susp_uid)"""),

    md("""## 20.5 Coordinated Bot Farms & Communities
We query the community detection subsystem to identify bot farms and coordinated clusters across the network.
"""),

    code("""comm_summary = pd.read_csv(RESULTS_DIR / 'communities_summary.csv')
bot_farms = comm_summary[comm_summary['coordinated_ratio'] >= 0.50].head(8)

print('=' * 85)
print('IDENTIFIED COORDINATED BOT FARMS & ASTROTURFING NETWORKS (MODULE 14):')
print('=' * 85)
print(bot_farms[['community_id', 'size', 'n_bots', 'n_suspicious', 'bot_concentration', 'classification']].to_string(index=False))"""),

    md("""## 20.6 Overall System Dashboard
We render the master end-to-end performance and diagnostic dashboard.
"""),

    code("""from PIL import Image

dashboard_path = RESULTS_DIR / 'final_system_dashboard.png'
if dashboard_path.exists():
    img = Image.open(dashboard_path)
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    from src.system.bot_detector import generate_system_dashboard
    generate_system_dashboard()"""),

    md("""## 20.7 Complete Project Benchmark Summary

| Module | Core Deliverable | Output Artifact | Key Result |
|---|---|---|---|
| **Module 1–3** | Dataset Pipeline | `nodes_clean.csv`, `edges_clean.csv`, `tweets_clean.csv` | 10,000 accounts, 55,000 interactions, 137,774 tweets |
| **Module 4** | Behavioral Features | `node_features.csv` | 35 features (activity, timing entropy, ratios) |
| **Module 5** | NLP Content Features | `text_features.csv` | 36 features (stylometrics, TF-IDF LSA topics, embeddings) |
| **Module 6–7** | Graph & Structural | `graph_features.csv`, `edge_index.pt` | 9 features (PageRank, degrees, clustering, k-core) |
| **Module 8** | Traditional ML Baselines | `traditional_ml_metrics.csv` | 5 models evaluated (Logistic, DT, RF, XGBoost, SVM) |
| **Module 9–12** | GNN Architectures | `gnn_comparison_metrics.csv` | GCN, GraphSAGE, GAT with PyG |
| **Module 14–16** | Communities & Coordination | `coordination_features.csv` | 31 Louvain communities, coordination & burstiness scores |
| **Module 19** | Explainable AI | `gnn_explainer.py` | Local attribution, GAT attention weights, ego graphs |
| **Module 20** | End-to-End System | `bot_detector.py` | 5-stage automated diagnostic engine & master dashboard |

---
**Social Media Bot Detection Using GNN — All 20 Modules Successfully Completed!**
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

path = ROOT / "module20_final_system.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
