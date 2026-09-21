"""
build_module10_12_notebook.py
Generates module9_12_gnn_architectures.ipynb programmatically.
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
    md("""# Modules 9–12 — Graph Neural Networks (GCN, GraphSAGE, GAT)

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Module 9: Principles of Graph Representation Learning**:
   - Why traditional ML fails on relational social networks.
   - Message passing paradigm: $h_v^{(k)} = \\text{UPDATE}^{(k)}\\left(h_v^{(k-1)}, \\text{AGGREGATE}^{(k)}\\left(\\{h_u^{(k-1)} : u \\in \\mathcal{N}(v)\\}\\right)\\right)$.
2. **Module 10: Graph Convolutional Network (GCN)**:
   - Spectral degree-normalized adjacency convolution: $\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} X W$.
3. **Module 11: GraphSAGE (Sample & Aggregate)**:
   - Inductive learning, separate ego vs neighbor transformations: $W \\cdot [x_v \\,|\\, \\text{MEAN}(\\{x_u\\})]$.
4. **Module 12: Graph Attention Network (GAT)**:
   - Dynamic neighbor importance using multi-head self-attention coefficients $\\alpha_{ij}$.
5. **Comparative Benchmark**:
   - Evaluate all 3 architectures on test set ($N=1,500$) with loss curves, confusion matrices, PCA embeddings, and attention distributions.
"""),

    code(f"""import sys, os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

ROOT         = Path(r'{ROOT_STR}')
RESULTS_DIR  = ROOT / 'data' / 'results'
MODELS_DIR   = ROOT / 'data' / 'models'
GRAPH_DIR    = ROOT / 'data' / 'graph'
FEATURES_DIR = ROOT / 'data' / 'features'
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('Environment initialised. PyTorch version:', torch.__version__)"""),

    md("""## 9.1 Load PyG Graph Data & Split Masks
We load our social graph into PyTorch Geometric format:
- $X \\in \\mathbb{R}^{10,000 \\times 80}$ (Behavioral + NLP + Graph features)
- $\\text{edge\\_index} \\in \\mathbb{Z}^{2 \\times 54,980}$
- Ground truth labels $y \\in \\{0, 1, 2\\}$ (Human, Bot, Suspicious)
- Boolean transductive masks: Train (70%), Validation (15%), Test (15%)
"""),

    code("""from src.models.gnn_models import load_pyg_data

data, feature_cols = load_pyg_data(device='cpu')

print(f'Graph Nodes (Users)       : {data.num_nodes:,}')
print(f'Graph Edges (Interactions): {data.num_edges:,}')
print(f'Node Feature Dimensions   : {data.num_node_features}')
print(f'Training Set Nodes        : {data.train_mask.sum().item():,}')
print(f'Validation Set Nodes      : {data.val_mask.sum().item():,}')
print(f'Test Set Nodes            : {data.test_mask.sum().item():,}')"""),

    md("""## 10.1 Module 10 — Graph Convolutional Network (GCN)
GCN applies localized first-order approximations of spectral graph convolutions:
$$h_v^{(l+1)} = \\sigma\\left( \\sum_{u \\in \\mathcal{N}(v) \\cup \\{v\\}} \\frac{1}{\\sqrt{\\tilde{d}_v \\tilde{d}_u}} h_u^{(l)} W^{(l)} \\right)$$
"""),

    code("""from src.models.gnn_models import BotGCN, train_gnn_model

gcn_model = BotGCN(in_channels=data.num_node_features, hidden_dim=64, num_classes=3, dropout=0.25)
print(gcn_model)

gcn_metrics, gcn_hist, gcn_preds, gcn_proba, gcn_emb = train_gnn_model(
    model_name='GCN', model=gcn_model, data=data, epochs=70, lr=0.008, patience=15
)"""),

    md("""## 11.1 Module 11 — GraphSAGE
GraphSAGE learns aggregator functions rather than individual node embeddings, allowing inductive generalization:
$$h_{\\mathcal{N}(v)}^{(l)} = \\text{AGGREGATE}_k\\left(\\{h_u^{(l-1)}, \\forall u \\in \\mathcal{N}(v)\\}\\right)$$
$$h_v^{(l)} = \\sigma\\left(W^{(l)} \\cdot [h_v^{(l-1)} \\, \\| \\, h_{\\mathcal{N}(v)}^{(l)}]\\right)$$
"""),

    code("""from src.models.gnn_models import BotGraphSAGE

sage_model = BotGraphSAGE(in_channels=data.num_node_features, hidden_dim=64, num_classes=3, dropout=0.25)
print(sage_model)

sage_metrics, sage_hist, sage_preds, sage_proba, sage_emb = train_gnn_model(
    model_name='GraphSAGE', model=sage_model, data=data, epochs=70, lr=0.008, patience=15
)"""),

    md("""## 12.1 Module 12 — Graph Attention Network (GAT)
GAT computes self-attention coefficients $\\alpha_{ij}$ measuring the relative importance of neighbor $j$ to node $i$:
$$\\alpha_{ij} = \\frac{\\exp\\left(\\text{LeakyReLU}\\left(\\mathbf{a}^T [W h_i \\,\\|\\, W h_j]\\right)\\right)}{\\sum_{k \\in \\mathcal{N}(i)} \\exp\\left(\\text{LeakyReLU}\\left(\\mathbf{a}^T [W h_i \\,\\|\\, W h_k]\\right)\\right)}$$
Multi-head attention stabilizes learning by computing $K$ independent attention distributions.
"""),

    code("""from src.models.gnn_models import BotGAT

gat_model = BotGAT(in_channels=data.num_node_features, hidden_dim=64, num_classes=3, heads=4, dropout=0.25)
print(gat_model)

gat_metrics, gat_hist, gat_preds, gat_proba, gat_emb = train_gnn_model(
    model_name='GAT', model=gat_model, data=data, epochs=70, lr=0.008, patience=15
)"""),

    md("""## 12.2 GAT Learned Attention Coefficient Inspection
We extract the learned edge attention weights $\\alpha_{ij}$ from GAT's multi-head attention layer to observe how the model prioritizes informative connections.
"""),

    code("""edge_idx_attn, alpha = gat_model.get_attention_weights(data.x, data.edge_index)
mean_alpha = alpha.mean(dim=1).cpu().numpy()

fig, ax = plt.subplots(figsize=(8.5, 4.5))
sns.histplot(mean_alpha, bins=50, kde=True, color='#E64A19', ax=ax)
ax.set_title('Module 12 — GAT Learned Attention Weight Distribution (α_ij)', fontsize=12, fontweight='bold')
ax.set_xlabel('Attention Coefficient α_ij')
ax.set_ylabel('Edge Frequency')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'gat_attention_weights.png', dpi=120)
plt.show()

print(f'Mean attention coefficient   : {mean_alpha.mean():.4f}')
print(f'Max attention coefficient    : {mean_alpha.max():.4f}')
print(f'Standard deviation           : {mean_alpha.std():.4f}')"""),

    md("""## 12.3 GNN Training Dynamics & Convergence
Comparison of loss and accuracy trajectories across GCN, GraphSAGE, and GAT.
"""),

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
colors = {'GCN': '#1976D2', 'GraphSAGE': '#388E3C', 'GAT': '#E64A19'}
histories = {'GCN': gcn_hist, 'GraphSAGE': sage_hist, 'GAT': gat_hist}

for name, hist in histories.items():
    c = colors[name]
    ax1.plot(hist['epoch'], hist['val_loss'], label=f'{name} Val', color=c, lw=2)
    ax2.plot(hist['epoch'], hist['val_acc'],  label=f'{name} Val', color=c, lw=2)

ax1.set_title('Validation Loss Trajectory', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
ax2.set_title('Validation Accuracy Trajectory', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
plt.suptitle('GNN Training Dynamics (Modules 10–12)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'gnn_training_curves.png', dpi=120)
plt.show()"""),

    md("""## 12.4 2D Projection of Learned Graph Embeddings
Visualizing penultimate layer representations across classes on unseen test nodes.
"""),

    code("""from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
embs = {'GCN': gcn_emb, 'GraphSAGE': sage_emb, 'GAT': gat_emb}
class_styles = {0: ('Human', '#4CAF50'), 1: ('Bot', '#F44336'), 2: ('Suspicious', '#FF9800')}
test_y = data.y[data.test_mask].cpu().numpy()

for i, (name, emb) in enumerate(embs.items()):
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(emb)
    for lbl, (cname, color) in class_styles.items():
        mask = (test_y == lbl)
        axes[i].scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=12, alpha=0.5, color=color, label=cname)
    axes[i].set_title(f'{name} Node Embeddings', fontsize=11, fontweight='bold')
    axes[i].legend(markerscale=2.5, fontsize=8)

plt.suptitle('Learned Latent Space Separation Across GNN Architectures (Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'gnn_embeddings_tsne.png', dpi=120)
plt.show()"""),

    md("""## 12.5 Master Leaderboard: Traditional ML vs Graph Neural Networks
Comprehensive comparison of all 8 models developed across Modules 8 through 12.
"""),

    code("""master_df = pd.read_csv(RESULTS_DIR / 'master_model_benchmark.csv')
print('=' * 85)
print('MASTER BENCHMARK: TRADITIONAL ML VS GRAPH NEURAL NETWORKS (TEST SET):')
print('=' * 85)
print(master_df[['Model', 'Category', 'Accuracy', 'F1_Macro', 'ROC_AUC', 'Train_Time_Sec']].to_string(index=False))"""),

    md("""## Modules 9–12 Summary

| Architecture | Paradigm | Strengths in Bot Detection |
|---|---|---|
| **GCN (Module 10)** | Normalized spectral convolution | Smooths features across local neighborhood; effective for uniform homophily. |
| **GraphSAGE (Module 11)** | Inductive sample & aggregate | Preserves ego-node identity alongside neighbor aggregation; robust against over-smoothing. |
| **GAT (Module 12)** | Multi-head self-attention | Dynamically down-weights deceptive/adversarial edges while amplifying critical coordination links. |

---

**Next:** **Module 13 — Multimodal Feature Fusion (Early, Late & Cross-Attention Fusion)**
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

path = ROOT / "module9_12_gnn_architectures.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
