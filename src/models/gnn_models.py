"""
src/models/gnn_models.py

Modules 9, 10, 11, 12 — Graph Neural Network Architectures
===========================================================
Implements, trains, and benchmarks three canonical GNN architectures
for Social Media Bot Detection using PyTorch Geometric:
  1. GCN (Graph Convolutional Network — Module 10)
  2. GraphSAGE (Sample and Aggregate — Module 11)
  3. GAT (Graph Attention Network — Module 12)

Features multi-hop neighborhood aggregation, multi-head attention coefficient
extraction, early stopping, class-weighted optimization, and comprehensive evaluation.
"""
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
FEATURES_DIR  = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"
GRAPH_DIR     = ROOT / "data" / "graph"
SPLITS_DIR    = ROOT / "data" / "splits"
RESULTS_DIR   = ROOT / "data" / "results"
MODELS_DIR    = ROOT / "data" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. PyG Graph Data Loader & Transductive Split Masks
# =============================================================================
def load_pyg_data(device="cpu"):
    """
    Constructs a PyG Data object containing:
      - x: Node feature tensor (10,000 x 80)
      - edge_index: Graph connectivity tensor (2 x 54,980)
      - edge_attr: Edge weights tensor (54,980)
      - y: Ground truth labels (10,000)
      - train_mask, val_mask, test_mask: Transductive Boolean split masks
    """
    print("  Loading node features & graph tensors for PyG...", flush=True)
    nodes_df = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv", low_memory=False)
    node_feat_df = pd.read_csv(FEATURES_DIR / "node_features.csv", low_memory=False)
    text_feat_df = pd.read_csv(FEATURES_DIR / "text_features.csv", low_memory=False)
    graph_feat_df = pd.read_csv(FEATURES_DIR / "graph_features.csv", low_memory=False)

    with open(FEATURES_DIR / "feature_columns.json") as f:
        beh_cols = json.load(f)["feature_columns"]
    with open(FEATURES_DIR / "text_feature_columns.json") as f:
        text_cols = json.load(f)["text_feature_columns"]
    with open(FEATURES_DIR / "graph_feature_columns.json") as f:
        raw_graph_cols = json.load(f)["graph_feature_columns"]

    # Deduplicate / rename overlapping column names
    existing = set(beh_cols + text_cols)
    graph_cols = [f"graph_{c}" if c in existing else c for c in raw_graph_cols]
    rename_map = {raw: new for raw, new in zip(raw_graph_cols, graph_cols) if raw != new}
    if rename_map:
        graph_feat_df = graph_feat_df.rename(columns=rename_map)

    # Merge in exact row order
    merged = nodes_df[["user_id", "label"]].copy()
    merged = merged.merge(node_feat_df[["user_id"] + beh_cols], on="user_id", how="left")
    merged = merged.merge(text_feat_df[["user_id"] + text_cols], on="user_id", how="left")
    merged = merged.merge(graph_feat_df[["user_id"] + graph_cols], on="user_id", how="left")

    all_feat_cols = beh_cols + text_cols + graph_cols
    X = torch.tensor(merged[all_feat_cols].fillna(0.0).values, dtype=torch.float)
    y = torch.tensor(merged["label"].values, dtype=torch.long)

    # Load edges
    edge_index = torch.load(GRAPH_DIR / "edge_index.pt", map_location="cpu", weights_only=False)
    edge_attr  = torch.load(GRAPH_DIR / "edge_attr.pt",  map_location="cpu", weights_only=False)

    # Create masks
    train_ids = set(pd.read_csv(SPLITS_DIR / "train_ids.csv")["user_id"])
    val_ids   = set(pd.read_csv(SPLITS_DIR / "val_ids.csv")["user_id"])
    test_ids  = set(pd.read_csv(SPLITS_DIR / "test_ids.csv")["user_id"])

    user_ids = nodes_df["user_id"].values
    train_mask = torch.tensor([u in train_ids for u in user_ids], dtype=torch.bool)
    val_mask   = torch.tensor([u in val_ids   for u in user_ids], dtype=torch.bool)
    test_mask  = torch.tensor([u in test_ids  for u in user_ids], dtype=torch.bool)

    data = Data(
        x=X.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        y=y.to(device),
        train_mask=train_mask.to(device),
        val_mask=val_mask.to(device),
        test_mask=test_mask.to(device)
    )

    print(f"  PyG Data: {data.num_nodes:,} nodes, {data.num_edges:,} edges, {data.num_node_features} features", flush=True)
    print(f"  Splits: Train={data.train_mask.sum().item():,} | Val={data.val_mask.sum().item():,} | Test={data.test_mask.sum().item():,}", flush=True)
    return data, all_feat_cols


# =============================================================================
# 2. GNN Model Architectures (GCN, GraphSAGE, GAT)
# =============================================================================
class BotGCN(nn.Module):
    """
    Module 10 — Graph Convolutional Network (GCN)
    2-layer spectral-style neighborhood convolution with LayerNorm and Dropout.
    """
    def __init__(self, in_channels, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        # Layer 1 message passing
        h1 = self.conv1(x, edge_index)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2 message passing
        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)

        out = self.classifier(h2)
        if return_embedding:
            return out, h2
        return out


class BotGraphSAGE(nn.Module):
    """
    Module 11 — GraphSAGE (Sample and Aggregate)
    Inductive neighborhood aggregation using mean pooling aggregator.
    """
    def __init__(self, in_channels, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        h1 = self.conv1(x, edge_index)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)

        out = self.classifier(h2)
        if return_embedding:
            return out, h2
        return out


class BotGAT(nn.Module):
    """
    Module 12 — Graph Attention Network (GAT)
    Learns dynamic neighbor importance via multi-head self-attention.
    """
    def __init__(self, in_channels, hidden_dim=64, num_classes=3, heads=4, dropout=0.3):
        super().__init__()
        head_dim = hidden_dim // heads
        self.conv1 = GATConv(in_channels, head_dim, heads=heads, dropout=dropout)
        self.ln1   = nn.LayerNorm(head_dim * heads)
        self.conv2 = GATConv(head_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        h1 = self.conv1(x, edge_index)
        h1 = self.ln1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2)
        h2 = F.elu(h2)

        out = self.classifier(h2)
        if return_embedding:
            return out, h2
        return out

    def get_attention_weights(self, x, edge_index):
        """Extracts learned edge attention coefficients for explainability."""
        self.eval()
        with torch.no_grad():
            _, (edge_idx_attn, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return edge_idx_attn, alpha


# =============================================================================
# 3. Training, Validation & Evaluation Engine
# =============================================================================
def train_gnn_model(model_name, model, data, epochs=80, lr=0.008, weight_decay=5e-4, patience=15):
    """
    Trains a GNN model with early stopping on validation loss.
    Computes class weights to address minor class imbalances.
    """
    print(f"\n[Training {model_name}]", flush=True)
    device = data.x.device
    model = model.to(device)

    # Compute class weights from training set
    train_labels = data.y[data.train_mask].cpu().numpy()
    counts = np.bincount(train_labels)
    class_weights = torch.tensor(len(train_labels) / (len(counts) * counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_out = model(data.x, data.edge_index)
            val_loss = criterion(eval_out[data.val_mask], data.y[data.val_mask]).item()

            train_pred = eval_out[data.train_mask].argmax(dim=1)
            val_pred   = eval_out[data.val_mask].argmax(dim=1)

            train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()
            val_acc   = (val_pred == data.y[data.val_mask]).float().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})", flush=True)
            break

    train_duration = round(time.time() - t0, 3)

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save checkpoint to disk
    save_filename = f"best_{model_name.lower().replace(' ', '_')}.pt"
    torch.save(model.state_dict(), MODELS_DIR / save_filename)
    print(f"  Model saved to: {MODELS_DIR / save_filename} (Training time: {train_duration}s)", flush=True)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out, embeddings = model(data.x, data.edge_index, return_embedding=True)
        test_logits = test_out[data.test_mask]
        test_proba  = F.softmax(test_logits, dim=1).cpu().numpy()
        test_pred   = test_logits.argmax(dim=1).cpu().numpy()
        y_test_true = data.y[data.test_mask].cpu().numpy()

    acc        = accuracy_score(y_test_true, test_pred)
    prec_macro = precision_score(y_test_true, test_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test_true, test_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_test_true, test_pred, average="macro", zero_division=0)
    f1_weight  = f1_score(y_test_true, test_pred, average="weighted", zero_division=0)

    # Multi-class ROC-AUC (OvR)
    y_test_bin = label_binarize(y_test_true, classes=[0, 1, 2])
    try:
        roc_auc = roc_auc_score(y_test_bin, test_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = 0.0

    metrics = {
        "Model":              model_name,
        "Accuracy":           round(acc, 4),
        "Precision_Macro":    round(prec_macro, 4),
        "Recall_Macro":       round(rec_macro, 4),
        "F1_Macro":           round(f1_macro, 4),
        "F1_Weighted":        round(f1_weight, 4),
        "ROC_AUC":            round(roc_auc, 4),
        "Train_Time_Sec":     train_duration,
        "Best_Val_Loss":      round(best_val_loss, 4),
    }

    print(f"  --> Test Evaluation: Acc: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f}", flush=True)
    return metrics, history, test_pred, test_proba, embeddings[data.test_mask].cpu().numpy()


# =============================================================================
# 4. Diagnostic Visualisations & Benchmarking
# =============================================================================
def plot_gnn_diagnostics(all_histories, gnn_results_df, test_y, preds_dict, embeddings_dict, gat_model, data):
    print("\n  Generating GNN diagnostic visualisations...", flush=True)

    # 1. Training Curves (Loss & Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"GCN": "#1976D2", "GraphSAGE": "#388E3C", "GAT": "#E64A19"}

    for name, hist in all_histories.items():
        c = colors.get(name, "#333")
        ax1.plot(hist["epoch"], hist["train_loss"], label=f"{name} Train", color=c, linestyle="--", alpha=0.7)
        ax1.plot(hist["epoch"], hist["val_loss"],   label=f"{name} Val",   color=c, lw=2)

        ax2.plot(hist["epoch"], hist["train_acc"],  label=f"{name} Train", color=c, linestyle="--", alpha=0.7)
        ax2.plot(hist["epoch"], hist["val_acc"],    label=f"{name} Val",   color=c, lw=2)

    ax1.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)

    ax2.set_title("Training & Validation Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=8)

    plt.suptitle("Modules 10-12 — GNN Architecture Training Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gnn_training_curves.png", dpi=130)
    plt.close()

    # 2. Confusion Matrices for GNN Models
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, name in enumerate(["GCN", "GraphSAGE", "GAT"]):
        cm = confusion_matrix(test_y, preds_dict[name], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=["Human", "Bot", "Susp"], yticklabels=["Human", "Bot", "Susp"])
        axes[i].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("True Label")
        else:
            axes[i].set_ylabel("")

    plt.suptitle("Modules 10-12 — Normalized Confusion Matrices (GNN Models)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gnn_confusion_matrices.png", dpi=130)
    plt.close()

    # 3. 2D PCA of Learned Node Embeddings (GAT vs GCN vs GraphSAGE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    class_labels = {0: ("Human", "#4CAF50"), 1: ("Bot", "#F44336"), 2: ("Suspicious", "#FF9800")}

    for i, name in enumerate(["GCN", "GraphSAGE", "GAT"]):
        emb = embeddings_dict[name]
        pca = PCA(n_components=2, random_state=42)
        emb_2d = pca.fit_transform(emb)

        for lbl, (cname, color) in class_labels.items():
            mask = (test_y == lbl)
            axes[i].scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.5, color=color, label=cname)
        axes[i].set_title(f"{name} Embeddings (PCA)", fontsize=11, fontweight="bold")
        axes[i].legend(markerscale=2.5, fontsize=8)

    plt.suptitle("Modules 10-12 — Learned Graph Node Representations (Unseen Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gnn_embeddings_tsne.png", dpi=130)
    plt.close()

    # 4. GAT Attention Weights Distribution
    if hasattr(gat_model, "get_attention_weights"):
        edge_idx_attn, alpha = gat_model.get_attention_weights(data.x, data.edge_index)
        mean_alpha = alpha.mean(dim=1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(mean_alpha, bins=50, kde=True, color="#E64A19", ax=ax)
        ax.set_title("Module 12 — GAT Learned Attention Coefficient Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Multi-Head Attention Coefficient (α_ij)")
        ax.set_ylabel("Edge Count")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "gat_attention_weights.png", dpi=130)
        plt.close()


# =============================================================================
# 5. Main Execution Pipeline
# =============================================================================
def run_gnn_pipeline():
    print("=" * 75, flush=True)
    print("Modules 9–12 — Graph Neural Network (GCN, GraphSAGE, GAT) Pipeline", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cpu")
    data, feature_cols = load_pyg_data(device=device)

    in_dim  = data.num_node_features
    hid_dim = 64
    n_class = 3

    # Define GNN models
    models = {
        "GCN":       BotGCN(in_channels=in_dim, hidden_dim=hid_dim, num_classes=n_class, dropout=0.25),
        "GraphSAGE": BotGraphSAGE(in_channels=in_dim, hidden_dim=hid_dim, num_classes=n_class, dropout=0.25),
        "GAT":       BotGAT(in_channels=in_dim, hidden_dim=hid_dim, num_classes=n_class, heads=4, dropout=0.25),
    }

    all_histories = {}
    results_list  = []
    preds_dict    = {}
    probas_dict   = {}
    embeddings_dict = {}

    for name, model in models.items():
        metrics, history, test_pred, test_proba, emb = train_gnn_model(
            model_name=name,
            model=model,
            data=data,
            epochs=70,
            lr=0.008,
            weight_decay=5e-4,
            patience=15
        )
        all_histories[name]   = history
        results_list.append(metrics)
        preds_dict[name]      = test_pred
        probas_dict[name]     = test_proba
        embeddings_dict[name] = emb

    # Save metrics
    gnn_results_df = pd.DataFrame(results_list)
    out_csv = RESULTS_DIR / "gnn_comparison_metrics.csv"
    gnn_results_df.to_csv(out_csv, index=False)
    with open(RESULTS_DIR / "gnn_comparison_metrics.json", "w") as f:
        json.dump(results_list, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print("SUMMARY RESULTS TABLE (GNN Architecture Benchmark):", flush=True)
    print("=" * 75, flush=True)
    print(gnn_results_df[["Model", "Accuracy", "F1_Macro", "ROC_AUC", "Train_Time_Sec"]].to_string(index=False), flush=True)

    # Diagnostic Visualizations
    test_y = data.y[data.test_mask].cpu().numpy()
    plot_gnn_diagnostics(all_histories, gnn_results_df, test_y, preds_dict, embeddings_dict, models["GAT"], data)

    # Master Table: Traditional ML vs GNN
    trad_df = pd.read_csv(RESULTS_DIR / "traditional_ml_metrics.csv")
    trad_df["Category"] = "Traditional ML"
    gnn_results_df["Category"] = "Graph Neural Network"

    master_df = pd.concat([trad_df, gnn_results_df], ignore_index=True)
    master_df.to_csv(RESULTS_DIR / "master_model_benchmark.csv", index=False)

    print("\n" + "=" * 75, flush=True)
    print("OVERALL LEADERBOARD (Traditional ML vs Graph Neural Networks):", flush=True)
    print("=" * 75, flush=True)
    print(master_df[["Model", "Category", "Accuracy", "F1_Macro", "ROC_AUC", "Train_Time_Sec"]].to_string(index=False), flush=True)

    print("\nModules 9–12 GNN Architecture Pipeline Completed Successfully!", flush=True)


if __name__ == "__main__":
    run_gnn_pipeline()
