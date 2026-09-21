"""
src/system/bot_detector.py

Module 20 — Final End-to-End Social Media Bot Detection System
=============================================================
Unified AI System combining:
  1. Profile & Behavioral Analysis (Module 4)
  2. Text & Content NLP Analysis (Module 5)
  3. Directed Interaction Graph (Module 6 & 7)
  4. Traditional ML & GNN Inference (Modules 8–12)
  5. Multimodal Feature Fusion (Module 13)
  6. Coordinated Bot Community Detection (Modules 14–16)
  7. Explainable AI Diagnostics (Module 19)

Outputs:
  1. Individual Account Prediction (User ID, Label, Probability)
  2. Behavioral Diagnostics (Posting rate, Repetition, Consistency)
  3. Graph-Based Neighborhood Risk (Bot Homophily, Centrality)
  4. Coordinated Community Context (Community ID, Size, Bot Density)
  5. Transparent Decision Signals (XAI Explainability Rationale)
"""
import sys
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
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
GRAPH_DIR     = ROOT / "data" / "graph"
RESULTS_DIR   = ROOT / "data" / "results"
MODELS_DIR    = ROOT / "data" / "models"


class SocialMediaBotDetector:
    """
    End-to-End Production Bot Detection Engine.
    """
    def __init__(self, model_type="GAT"):
        self.device = torch.device("cpu")
        self.model_type = model_type

        # 1. Load Data
        from src.models.gnn_models import load_pyg_data, BotGCN, BotGraphSAGE, BotGAT
        self.data, self.feature_cols = load_pyg_data(device=self.device)

        # 2. Load Model
        in_dim = self.data.num_node_features
        if model_type == "GAT":
            self.model = BotGAT(in_channels=in_dim, hidden_dim=64, num_classes=3, heads=4, dropout=0.0).to(self.device)
            self.model.load_state_dict(torch.load(MODELS_DIR / "best_gat.pt", map_location=self.device, weights_only=False))
        elif model_type == "GraphSAGE":
            self.model = BotGraphSAGE(in_channels=in_dim, hidden_dim=64, num_classes=3, dropout=0.0).to(self.device)
            self.model.load_state_dict(torch.load(MODELS_DIR / "best_graphsage.pt", map_location=self.device, weights_only=False))
        else:
            self.model = BotGCN(in_channels=in_dim, hidden_dim=64, num_classes=3, dropout=0.0).to(self.device)
            self.model.load_state_dict(torch.load(MODELS_DIR / "best_gcn.pt", map_location=self.device, weights_only=False))

        self.model.eval()

        # 3. Load Auxiliary Metadata
        self.nodes_df = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv", low_memory=False)
        self.coord_df = pd.read_csv(FEATURES_DIR / "coordination_features.csv", low_memory=False)
        self.comm_df  = pd.read_csv(RESULTS_DIR / "communities_summary.csv", low_memory=False)

        self.uid_to_idx = {uid: i for i, uid in enumerate(self.nodes_df["user_id"])}
        self.idx_to_uid = {i: uid for i, uid in enumerate(self.nodes_df["user_id"])}

        # 4. Load Explainability Engine
        from src.explainability.gnn_explainer import BotGNNExplainer
        self.explainer = BotGNNExplainer(model=self.model, data=self.data, feature_cols=self.feature_cols)

    def analyze_account(self, user_id):
        """
        Runs full 5-stage AI diagnostic evaluation for an individual account.
        """
        if user_id not in self.uid_to_idx:
            raise ValueError(f"User ID {user_id} not recognized.")

        node_idx = self.uid_to_idx[user_id]
        node_row = self.nodes_df.iloc[node_idx]
        coord_row = self.coord_df[self.coord_df["user_id"] == user_id].iloc[0]

        # Model Inference
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits[node_idx], dim=0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        labels_map = {0: "HUMAN", 1: "BOT", 2: "SUSPICIOUS / COORDINATED"}
        pred_label = labels_map[pred_idx]

        # Stage 1: Individual Prediction
        stage1 = {
            "user_id": int(user_id),
            "prediction": pred_label,
            "bot_probability": round(float(probs[1]), 4),
            "suspicious_probability": round(float(probs[2]), 4),
            "human_probability": round(float(probs[0]), 4),
        }

        # Stage 2: Behavioral Analysis
        ppd = float(node_row.get("posts_per_day", 0))
        freq_rating = "Very High" if ppd > 20 else ("High" if ppd > 10 else ("Moderate" if ppd > 3 else "Normal"))

        dup_rate = float(node_row.get("duplicate_content_ratio", 0))
        rep_rating = "High" if dup_rate > 0.50 else ("Moderate" if dup_rate > 0.20 else "Low")

        ff_ratio = float(node_row.get("followers_following_ratio", 0))
        net_rating = "Abnormal / Highly Asymmetric" if ff_ratio < 0.15 or ff_ratio > 10.0 else "Balanced"

        c_score = float(coord_row.get("coordination_score", 0))
        coord_rating = "High" if c_score > 0.50 else ("Moderate" if c_score > 0.30 else "Low")

        stage2 = {
            "posting_frequency": f"{freq_rating} ({ppd:.1f} posts/day)",
            "content_repetition": f"{rep_rating} ({dup_rate:.1%} duplicates)",
            "network_connectivity": f"{net_rating} (Ratio: {ff_ratio:.2f})",
            "coordination_score": f"{coord_rating} ({c_score:.2f})",
        }

        # Stage 3: Graph-Based Detection
        exp = self.explainer.explain_account(user_id)
        neigh = exp["neighbor_analysis"]
        b_ratio = neigh["bot_neighbor_ratio"]
        neigh_risk = "High Risk (Bot Echo Chamber)" if b_ratio > 0.50 else ("Moderate" if b_ratio > 0.25 else "Low Risk / Organic")

        stage3 = {
            "total_neighbors": neigh["total_neighbors"],
            "bot_neighbors": neigh["bot_neighbors"],
            "human_neighbors": neigh["human_neighbors"],
            "bot_neighbor_ratio": round(float(b_ratio), 4),
            "neighborhood_assessment": neigh_risk
        }

        # Stage 4: Community Detection
        cid = int(coord_row.get("community_id", -1))
        comm_info = self.comm_df[self.comm_df["community_id"] == cid]
        if len(comm_info) > 0:
            c_row = comm_info.iloc[0]
            stage4 = {
                "community_id": cid,
                "community_members": int(c_row["size"]),
                "potential_bots_in_community": int(c_row["n_bots"] + c_row["n_suspicious"]),
                "community_bot_concentration": round(float(c_row["bot_concentration"]), 4),
                "community_classification": c_row["classification"]
            }
        else:
            stage4 = {"community_id": cid, "community_members": 0}

        # Stage 5: Explainable Signals
        stage5 = {
            "important_signals": exp["key_diagnostic_signals"],
            "top_attributing_features": exp["top_contributing_features"][:4]
        }

        return {
            "account_prediction": stage1,
            "behavioral_analysis": stage2,
            "graph_detection": stage3,
            "community_detection": stage4,
            "explainable_prediction": stage5
        }

    def print_diagnostic_card(self, user_id):
        """Prints beautifully formatted 5-stage report to terminal."""
        res = self.analyze_account(user_id)
        p = res["account_prediction"]
        b = res["behavioral_analysis"]
        g = res["graph_detection"]
        c = res["community_detection"]
        e = res["explainable_prediction"]

        print("\n" + "=" * 65)
        print(f"  AI BOT DETECTION DIAGNOSTIC REPORT: USER {p['user_id']}")
        print("=" * 65)

        print("\n1. Individual Account Prediction")
        print(f"   User ID         : {p['user_id']}")
        print(f"   Prediction      : {p['prediction']}")
        print(f"   Bot Probability : {p['bot_probability']:.2f}")

        print("\n2. Behavioral Analysis")
        print(f"   Posting Frequency    : {b['posting_frequency']}")
        print(f"   Content Repetition   : {b['content_repetition']}")
        print(f"   Network Connectivity : {b['network_connectivity']}")
        print(f"   Coordination Score   : {b['coordination_score']}")

        print("\n3. Graph-Based Detection")
        print(f"   Account -> Neighbors : {g['total_neighbors']} total ({g['bot_neighbor_ratio']*100:.1f}% bots)")
        print(f"   Suspicious Network   : {g['neighborhood_assessment']}")

        print("\n4. Community Detection")
        print(f"   Community            : #{c.get('community_id', 'N/A')}")
        print(f"   Members              : {c.get('community_members', 0)}")
        print(f"   Potential Bots       : {c.get('potential_bots_in_community', 0)}")
        print(f"   Classification       : {c.get('community_classification', 'N/A')}")

        print("\n5. Explainable Prediction")
        print(f"   Prediction -> {p['prediction']}")
        print("   Important signals:")
        for s in e["important_signals"]:
            print(f"     [+] {s}")
        print("=" * 65)


def generate_system_dashboard():
    """
    Generates the master multi-panel visual dashboard summarizing the entire system.
    """
    print("  Generating End-to-End System Dashboard...", flush=True)
    detector = SocialMediaBotDetector()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Master Model Benchmark Leaderboard
    ax1 = axes[0, 0]
    master_df = pd.read_csv(RESULTS_DIR / "master_model_benchmark.csv")
    sns.barplot(data=master_df, x="Model", y="Accuracy", hue="Category", ax=ax1, palette="Blues_d")
    ax1.set_ylim(0.70, 1.05)
    ax1.set_title("1. Model Architecture Benchmark (Test Accuracy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis='x', rotation=25)

    # Panel 2: Account Population Distribution
    ax2 = axes[0, 1]
    type_counts = detector.nodes_df["account_type"].value_counts()
    colors = ["#4CAF50", "#F44336", "#FF9800"]
    ax2.pie(type_counts, labels=[f"{l.title()} ({c:,})" for l, c in zip(type_counts.index, type_counts)],
            colors=colors, autopct="%1.1f%%", startangle=140, explode=[0.02, 0.02, 0.02])
    ax2.set_title("2. Social Network Population Breakdown (N=10,000)", fontsize=11, fontweight="bold")

    # Panel 3: Top Coordinated Bot Communities
    ax3 = axes[1, 0]
    top_comms = detector.comm_df.head(8)
    x = np.arange(len(top_comms))
    width = 0.35
    ax3.bar(x - width/2, top_comms["n_bots"], width, label="Confirmed Bots", color="#F44336", alpha=0.85)
    ax3.bar(x + width/2, top_comms["n_humans"], width, label="Humans", color="#4CAF50", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Comm #{cid}" for cid in top_comms["community_id"]], rotation=20)
    ax3.set_title("3. Bot Farm Detection Across Communities (Module 14)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Account Count")
    ax3.legend()

    # Panel 4: Sample Account Diagnostic Case Study
    ax4 = axes[1, 1]
    ax4.axis("off")
    bot_sample_uid = detector.nodes_df[detector.nodes_df["account_type"] == "bot"].iloc[0]["user_id"]
    res = detector.analyze_account(bot_sample_uid)
    p = res["account_prediction"]
    b = res["behavioral_analysis"]
    c = res["community_detection"]

    card_text = (
        f"4. AI DIAGNOSTIC CASE STUDY (User {p['user_id']})\n"
        f"----------------------------------------------------\n"
        f"Prediction           : {p['prediction']} (Confidence: {p['bot_probability']*100:.1f}%)\n"
        f"Posting Frequency    : {b['posting_frequency']}\n"
        f"Content Repetition   : {b['content_repetition']}\n"
        f"Network Asymmetry    : {b['network_connectivity']}\n"
        f"Coordination Score   : {b['coordination_score']}\n"
        f"Community Context    : #{c.get('community_id')} ({c.get('community_classification')})\n\n"
        f"Key Diagnostic Signals:\n"
    )
    for s in res["explainable_prediction"]["important_signals"]:
        card_text += f"  • {s}\n"

    ax4.text(0.05, 0.95, card_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#FAFAFA', edgecolor='#D32F2F', lw=2))

    plt.suptitle("Module 20 — End-to-End Social Media Bot Detection AI System", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_system_dashboard.png", dpi=130)
    plt.close()
    print("  Saved dashboard: data/results/final_system_dashboard.png", flush=True)


def run_system_demo():
    print("=" * 75, flush=True)
    print("Module 20 — Final End-to-End Bot Detection System Execution", flush=True)
    print("=" * 75, flush=True)

    detector = SocialMediaBotDetector(model_type="GAT")

    # Sample Bot Account
    sample_bot = detector.nodes_df[detector.nodes_df["account_type"] == "bot"].iloc[0]["user_id"]
    detector.print_diagnostic_card(sample_bot)

    # Sample Human Account
    sample_human = detector.nodes_df[detector.nodes_df["account_type"] == "human"].iloc[0]["user_id"]
    detector.print_diagnostic_card(sample_human)

    # Generate visual dashboard
    generate_system_dashboard()

    print("\nModule 20 Complete AI System Executed Successfully!", flush=True)


if __name__ == "__main__":
    run_system_demo()
