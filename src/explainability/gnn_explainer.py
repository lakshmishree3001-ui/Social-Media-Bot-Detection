"""
src/explainability/gnn_explainer.py

Module 19 — Explainable AI for Bot Detection
=============================================
Provides model interpretability and explainability:
  1. Feature Importance & Local Attribution (Gradient x Input)
  2. Neighborhood Influence & GAT Attention Analysis
  3. Neighbor Homophily & Echo Chamber Quantification
  4. Human-Readable Diagnostic Signals & Decision Rationale
  5. Local Explanation Visualisation (Ego Network + Signal Attribution)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
GRAPH_DIR     = ROOT / "data" / "graph"
RESULTS_DIR   = ROOT / "data" / "results"
MODELS_DIR    = ROOT / "data" / "models"


class BotGNNExplainer:
    """
    Explainability engine for GNN-based bot detection.
    """
    def __init__(self, model=None, data=None, feature_cols=None):
        self.device = torch.device("cpu")
        from src.models.gnn_models import load_pyg_data, BotGAT

        if data is None or feature_cols is None:
            self.data, self.feature_cols = load_pyg_data(device=self.device)
        else:
            self.data = data
            self.feature_cols = feature_cols

        if model is None:
            self.model = BotGAT(
                in_channels=self.data.num_node_features,
                hidden_dim=64,
                num_classes=3,
                heads=4,
                dropout=0.0
            ).to(self.device)
            state_dict = torch.load(MODELS_DIR / "best_gat.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
        else:
            self.model = model.to(self.device)

        self.model.eval()

        # Load metadata
        self.nodes_df = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv", low_memory=False)
        self.coord_df = pd.read_csv(FEATURES_DIR / "coordination_features.csv", low_memory=False)
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.nodes_df["user_id"])}
        self.idx_to_uid = {i: uid for i, uid in enumerate(self.nodes_df["user_id"])}

        # Build NetworkX graph for neighborhood queries
        edge_index_np = self.data.edge_index.cpu().numpy()
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.data.num_nodes))
        for src, dst in zip(edge_index_np[0], edge_index_np[1]):
            self.G.add_edge(src, dst)

    def explain_account(self, user_id, top_k_features=6):
        """
        Generates full explainability profile for a given user_id.
        """
        if user_id not in self.uid_to_idx:
            raise ValueError(f"User ID {user_id} not found in graph.")

        node_idx = self.uid_to_idx[user_id]
        x = self.data.x.clone().detach().requires_grad_(True)
        edge_index = self.data.edge_index

        # 1. Prediction & Probabilities
        out = self.model(x, edge_index)
        probs = F.softmax(out[node_idx], dim=0).detach().cpu().numpy()
        pred_label = int(np.argmax(probs))
        label_names = {0: "HUMAN", 1: "BOT", 2: "SUSPICIOUS"}
        pred_name = label_names.get(pred_label, "UNKNOWN")

        # 2. Local Gradient x Input Attribution
        target_score = out[node_idx, pred_label]
        target_score.backward()
        grads = x.grad[node_idx].detach().cpu().numpy()
        inputs = x[node_idx].detach().cpu().numpy()
        attribution = grads * inputs

        top_attr_indices = np.argsort(np.abs(attribution))[-top_k_features:][::-1]
        top_features = [
            {
                "feature": self.feature_cols[i],
                "attribution": round(float(attribution[i]), 4),
                "normalized_value": round(float(inputs[i]), 4)
            }
            for i in top_attr_indices
        ]

        # 3. Neighborhood & Attention Analysis
        neighbors_in  = list(self.G.predecessors(node_idx))
        neighbors_out = list(self.G.successors(node_idx))
        all_neighbors = list(set(neighbors_in + neighbors_out))

        neighbor_types = [self.nodes_df.iloc[n]["account_type"] for n in all_neighbors] if all_neighbors else []
        counts = pd.Series(neighbor_types).value_counts().to_dict() if neighbor_types else {}
        n_tot = len(all_neighbors)
        bot_neighbor_ratio = counts.get("bot", 0) / max(n_tot, 1)
        susp_neighbor_ratio = counts.get("suspicious", 0) / max(n_tot, 1)

        # GAT Attention weights
        attn_weights = []
        if hasattr(self.model, "get_attention_weights"):
            edge_idx_attn, alpha = self.model.get_attention_weights(self.data.x, edge_index)
            alpha_mean = alpha.mean(dim=1).cpu().numpy()
            src_nodes = edge_idx_attn[0].cpu().numpy()
            dst_nodes = edge_idx_attn[1].cpu().numpy()

            for n in all_neighbors:
                # Find edges between node_idx and n
                mask = (src_nodes == n) & (dst_nodes == node_idx)
                if np.any(mask):
                    attn_weights.append({
                        "neighbor_id": int(self.idx_to_uid[n]),
                        "neighbor_type": self.nodes_df.iloc[n]["account_type"],
                        "attention_weight": round(float(alpha_mean[mask].mean()), 4)
                    })

        # 4. Behavioral & Coordination Signals
        node_row = self.nodes_df.iloc[node_idx]
        coord_row = self.coord_df[self.coord_df["user_id"] == user_id]
        coord_score = coord_row["coordination_score"].values[0] if len(coord_row) > 0 else 0.0
        burstiness  = coord_row["temporal_burstiness"].values[0] if len(coord_row) > 0 else 0.0
        community_id = int(coord_row["community_id"].values[0]) if len(coord_row) > 0 else -1

        signals = []
        if float(node_row.get("posts_per_day", 0)) > 15:
            signals.append("Extremely high posting frequency (>15 posts/day)")
        if float(node_row.get("duplicate_content_ratio", 0)) > 0.50:
            signals.append(f"Highly repetitive content ({node_row.get('duplicate_content_ratio', 0):.1%} duplicates)")
        if bot_neighbor_ratio >= 0.50:
            signals.append(f"Dense bot-like neighborhood ({bot_neighbor_ratio:.1%} bot neighbors)")
        if coord_score > 0.50:
            signals.append(f"High coordination score ({coord_score:.2f})")
        if burstiness > 0.40:
            signals.append(f"Pronounced activity burstiness (B = {burstiness:.2f})")
        if float(node_row.get("followers_following_ratio", 0)) < 0.10:
            signals.append("Unusual follower/following asymmetry (<0.10 ratio)")

        if not signals:
            signals.append("Organic posting and interaction patterns consistent with genuine human behavior.")

        explanation = {
            "user_id": int(user_id),
            "prediction": pred_name,
            "probabilities": {
                "human": round(float(probs[0]), 4),
                "bot": round(float(probs[1]), 4),
                "suspicious": round(float(probs[2]), 4)
            },
            "community_id": community_id,
            "coordination_score": round(float(coord_score), 4),
            "temporal_burstiness": round(float(burstiness), 4),
            "neighbor_analysis": {
                "total_neighbors": n_tot,
                "bot_neighbors": counts.get("bot", 0),
                "human_neighbors": counts.get("human", 0),
                "suspicious_neighbors": counts.get("suspicious", 0),
                "bot_neighbor_ratio": round(float(bot_neighbor_ratio), 4)
            },
            "top_contributing_features": top_features,
            "top_attended_neighbors": sorted(attn_weights, key=lambda x: x["attention_weight"], reverse=True)[:5],
            "key_diagnostic_signals": signals
        }
        return explanation

    def visualize_explanation(self, user_id, save_path=None):
        """
        Generates explanation graphic: Ego network with attention weights + feature attribution bar chart.
        """
        exp = self.explain_account(user_id)
        node_idx = self.uid_to_idx[user_id]
        tc = {"human": "#4CAF50", "bot": "#F44336", "suspicious": "#FF9800"}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # 1. Feature attribution bar chart
        feats = [f["feature"] for f in exp["top_contributing_features"]]
        attrs = [f["attribution"] for f in exp["top_contributing_features"]]
        colors = ["#D32F2F" if a > 0 else "#1976D2" for a in attrs]
        ax1.barh(feats[::-1], attrs[::-1], color=colors[::-1], alpha=0.85)
        ax1.set_title(f"Local Feature Attribution (User {user_id} -> {exp['prediction']})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Attribution (Gradient x Input)")

        # 2. Local Ego Network with attention
        ego = nx.ego_graph(self.G, node_idx, radius=1)
        sub_nodes = list(ego.nodes)[:30]
        ego_s = ego.subgraph(sub_nodes)
        pos = nx.spring_layout(ego_s, seed=42)

        node_colors = [
            "#9C27B0" if n == node_idx else tc.get(self.nodes_df.iloc[n]["account_type"], "#888")
            for n in ego_s.nodes()
        ]
        nx.draw_networkx(ego_s, pos=pos, node_color=node_colors, node_size=120, edge_color="#bbb",
                         arrows=True, with_labels=False, arrowsize=8, ax=ax2)
        ax2.set_title(f"1-Hop Neighborhood (Center: User {user_id} [{exp['prediction']}])", fontsize=11, fontweight="bold")
        patches = [
            mpatches.Patch(color="#9C27B0", label="Target Account"),
            mpatches.Patch(color=tc["human"], label="Human Neighbor"),
            mpatches.Patch(color=tc["bot"], label="Bot Neighbor"),
            mpatches.Patch(color=tc["suspicious"], label="Suspicious Neighbor")
        ]
        ax2.legend(handles=patches, loc="upper right", fontsize=8)
        ax2.axis("off")

        plt.suptitle(f"Module 19 — Explainable AI Diagnostic Report: User {user_id} ({exp['prediction']} - Prob: {exp['probabilities']['bot']*100:.1f}%)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save_path is None:
            save_path = RESULTS_DIR / f"explanation_user_{user_id}.png"
        plt.savefig(save_path, dpi=130)
        plt.close()
        return save_path


def run_explainability_demo():
    print("=" * 75, flush=True)
    print("Module 19 — Explainable AI (XAI) Diagnostic Engine", flush=True)
    print("=" * 75, flush=True)

    explainer = BotGNNExplainer()

    # Pick sample accounts (one bot, one human, one suspicious)
    sample_bot = explainer.nodes_df[explainer.nodes_df["account_type"] == "bot"].iloc[0]["user_id"]
    sample_human = explainer.nodes_df[explainer.nodes_df["account_type"] == "human"].iloc[0]["user_id"]

    for uid in [sample_bot, sample_human]:
        exp = explainer.explain_account(uid)
        print(f"\n--- Diagnostic Profile for User ID: {exp['user_id']} ---", flush=True)
        print(f"  Prediction       : {exp['prediction']} (Bot Prob: {exp['probabilities']['bot']:.4f})", flush=True)
        print(f"  Community ID     : #{exp['community_id']}", flush=True)
        print(f"  Coordination Sc. : {exp['coordination_score']}", flush=True)
        print(f"  Neighbors        : {exp['neighbor_analysis']['total_neighbors']} ({exp['neighbor_analysis']['bot_neighbor_ratio']*100:.1f}% Bots)", flush=True)
        print("  Key Signals Detected:", flush=True)
        for s in exp["key_diagnostic_signals"]:
            print(f"    [+] {s}", flush=True)
        print("  Top Contributing Features:", flush=True)
        for f in exp["top_contributing_features"][:3]:
            print(f"    - {f['feature']:25s} | Attr: {f['attribution']:+.4f} | Val: {f['normalized_value']:+.4f}", flush=True)

        img_path = explainer.visualize_explanation(uid)
        print(f"  Saved explanation graphic to: {img_path.name}", flush=True)

    print("\nModule 19 Explainable AI Pipeline Completed Successfully!", flush=True)


if __name__ == "__main__":
    run_explainability_demo()
