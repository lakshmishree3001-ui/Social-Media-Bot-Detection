"""
src/models/baseline_models.py

Module 8 — Traditional Machine Learning Baseline
=================================================
Trains and benchmarks traditional ML models before GNNs:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost
  5. Support Vector Machine (SVM)

Evaluates on unseen test set (N=1,500) using strict train/val/test splits.
Computes Accuracy, Precision, Recall, F1-score (macro/weighted), ROC-AUC, and latency.
Generates confusion matrices, ROC curves, and feature importance visualisations.
"""
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[2]
FEATURES_DIR  = ROOT / "data" / "features"
SPLITS_DIR    = ROOT / "data" / "splits"
RESULTS_DIR   = ROOT / "data" / "results"
MODELS_DIR    = ROOT / "data" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_and_splits(include_graph_features=True):
    """
    Loads node behavioral features, text NLP features, and graph features.
    Aligns them strictly with the pre-defined train/val/test splits.
    """
    print("  Loading feature matrices...", flush=True)
    node_feat_df = pd.read_csv(FEATURES_DIR / "node_features.csv", low_memory=False)
    text_feat_df = pd.read_csv(FEATURES_DIR / "text_features.csv", low_memory=False)
    
    with open(FEATURES_DIR / "feature_columns.json") as f:
        beh_cols = json.load(f)["feature_columns"]
    with open(FEATURES_DIR / "text_feature_columns.json") as f:
        text_cols = json.load(f)["text_feature_columns"]

    # Base features: Behavioral + Text
    merged = node_feat_df[["user_id", "label"] + beh_cols].merge(
        text_feat_df[["user_id"] + text_cols], on="user_id", how="left"
    )
    feature_cols = list(beh_cols) + list(text_cols)

    # Optional Graph features
    if include_graph_features:
        graph_feat_df = pd.read_csv(FEATURES_DIR / "graph_features.csv", low_memory=False)
        with open(FEATURES_DIR / "graph_feature_columns.json") as f:
            raw_graph_cols = json.load(f)["graph_feature_columns"]
        
        # Rename overlapping columns to avoid clashes
        rename_map = {}
        for c in raw_graph_cols:
            if c in feature_cols:
                rename_map[c] = f"graph_{c}"
        if rename_map:
            graph_feat_df = graph_feat_df.rename(columns=rename_map)
        
        graph_cols = [rename_map.get(c, c) for c in raw_graph_cols]
        merged = merged.merge(graph_feat_df[["user_id"] + graph_cols], on="user_id", how="left")
        feature_cols += graph_cols

    # Ensure no NaNs and proper float types
    merged[feature_cols] = merged[feature_cols].fillna(0.0).astype(np.float32)

    # Load official split IDs
    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv")["user_id"].values
    val_ids   = pd.read_csv(SPLITS_DIR / "val_ids.csv")["user_id"].values
    test_ids  = pd.read_csv(SPLITS_DIR / "test_ids.csv")["user_id"].values

    df_indexed = merged.set_index("user_id")

    X_train = df_indexed.loc[train_ids, feature_cols].values
    y_train = df_indexed.loc[train_ids, "label"].values.astype(int)

    X_val   = df_indexed.loc[val_ids, feature_cols].values
    y_val   = df_indexed.loc[val_ids, "label"].values.astype(int)

    X_test  = df_indexed.loc[test_ids, feature_cols].values
    y_test  = df_indexed.loc[test_ids, "label"].values.astype(int)

    print(f"  Features loaded: {len(feature_cols)} dimensions ({len(beh_cols)} Behavioral, {len(text_cols)} NLP"
          f"{f', {len(graph_cols)} Graph' if include_graph_features else ''})", flush=True)
    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}", flush=True)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def get_baseline_models():
    """
    Instantiates standard traditional ML classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    }
    return models


def evaluate_model(name, model, X_train, y_train, X_test, y_test, classes=(0, 1, 2)):
    """
    Trains model, records execution time, and computes all required test metrics.
    """
    print(f"  --> Training {name}...", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 4)

    t0_inf = time.time()
    y_pred = model.predict(X_test)
    inference_time = round((time.time() - t0_inf) * 1000, 2)  # ms

    # Predict probabilities for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_proba = exp_df / np.sum(exp_df, axis=1, keepdims=True)

    acc        = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weight  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Compute Multi-class ROC-AUC (OvR)
    roc_auc = 0.0
    if y_proba is not None:
        try:
            y_test_bin = label_binarize(y_test, classes=classes)
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = 0.0

    print(f"      Acc: {acc*100:.2f}% | F1-Macro: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f} | Train Time: {train_time}s", flush=True)

    metrics = {
        "Model":              name,
        "Accuracy":           round(acc, 4),
        "Precision_Macro":    round(prec_macro, 4),
        "Recall_Macro":       round(rec_macro, 4),
        "F1_Macro":           round(f1_macro, 4),
        "F1_Weighted":        round(f1_weight, 4),
        "ROC_AUC":            round(roc_auc, 4),
        "Train_Time_Sec":     train_time,
        "Inference_Time_ms":  inference_time,
    }

    return metrics, y_pred, y_proba


def generate_visualisations(results_list, models_dict, y_test, preds_dict, probas_dict, feature_cols):
    """
    Creates comprehensive plots:
      1. Model benchmark metrics bar chart
      2. Multi-model confusion matrices
      3. ROC Curves (OvR)
      4. Top Feature Importances (Random Forest & XGBoost)
    """
    print("  Generating diagnostic visualisations...", flush=True)
    df_metrics = pd.DataFrame(results_list)
    class_names = ["Human (0)", "Bot (1)", "Suspicious (2)"]

    # 1. Benchmark Barplot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    metrics_to_plot = ["Accuracy", "F1_Macro", "ROC_AUC"]
    df_melted = df_metrics.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", ax=ax, palette="Blues_d")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Module 8 — Traditional ML Baseline Model Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.legend(loc="lower right")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_barplot.png", dpi=130)
    plt.close()

    # 2. Confusion Matrices
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8))
    for i, (name, _) in enumerate(models_dict.items()):
        ax = axes[i] if n_models > 1 else axes
        cm = confusion_matrix(y_test, preds_dict[name], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Human", "Bot", "Susp"], yticklabels=["Human", "Bot", "Susp"])
        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("True")
        else:
            ax.set_ylabel("")
    plt.suptitle("Module 8 — Normalized Confusion Matrices (Unseen Test Split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices_traditional_ml.png", dpi=130)
    plt.close()

    # 3. ROC Curves (OvR)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    colors = ["#1976D2", "#388E3C", "#D32F2F", "#7B1FA2", "#FF8F00"]

    for (name, _), color in zip(models_dict.items(), colors):
        y_prob = probas_dict[name]
        if y_prob is not None:
            fpr_list, tpr_list = [], []
            for c in range(3):
                fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob[:, c])
                fpr_list.append(fpr)
                tpr_list.append(tpr)
            # Micro-average ROC
            all_fpr = np.unique(np.concatenate(fpr_list))
            mean_tpr = np.zeros_like(all_fpr)
            for c in range(3):
                mean_tpr += np.interp(all_fpr, fpr_list[c], tpr_list[c])
            mean_tpr /= 3.0
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color=color, lw=2, label=f"{name} (Macro AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Module 8 — Receiver Operating Characteristic (ROC) Curves (Macro OvR)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves_traditional_ml.png", dpi=130)
    plt.close()

    # 4. Feature Importance (Random Forest & XGBoost)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    rf = models_dict.get("Random Forest")
    xgb_m = models_dict.get("XGBoost")

    if rf is not None and hasattr(rf, "feature_importances_"):
        rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=ax1, palette="viridis")
        ax1.set_title("Random Forest — Top 20 Most Predictive Features", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Gini Importance")

    if xgb_m is not None and hasattr(xgb_m, "feature_importances_"):
        xgb_imp = pd.Series(xgb_m.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=ax2, palette="magma")
        ax2.set_title("XGBoost — Top 20 Most Predictive Features", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Gain Importance")

    plt.suptitle("Module 8 — Feature Importance Rankings (Tree Ensemble Baselines)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_traditional_ml.png", dpi=130)
    plt.close()


def run_baseline_pipeline():
    print("=" * 75, flush=True)
    print("Module 8 — Traditional Machine Learning Baseline Pipeline", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Data (Behavioral + NLP + Graph)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_dataset_and_splits(
        include_graph_features=True
    )

    models_dict = get_baseline_models()
    results_list = []
    preds_dict   = {}
    probas_dict  = {}

    print(f"\nTraining & Evaluating {len(models_dict)} Baseline Classifiers on Test Set (N={len(y_test):,}):", flush=True)
    for name, model in models_dict.items():
        metrics, y_pred, y_proba = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results_list.append(metrics)
        preds_dict[name]  = y_pred
        probas_dict[name] = y_proba

        # Save trained model artifact
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{clean_name}.joblib")

    # 2. Save Metrics Table
    results_df = pd.DataFrame(results_list)
    out_csv = RESULTS_DIR / "traditional_ml_metrics.csv"
    results_df.to_csv(out_csv, index=False)
    with open(RESULTS_DIR / "traditional_ml_metrics.json", "w") as f:
        json.dump(results_list, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print("SUMMARY RESULTS TABLE (Module 8 Traditional ML Baselines):", flush=True)
    print("=" * 75, flush=True)
    print(results_df[["Model", "Accuracy", "F1_Macro", "ROC_AUC", "Train_Time_Sec"]].to_string(index=False), flush=True)

    # 3. Generate Visualisations
    generate_visualisations(results_list, models_dict, y_test, preds_dict, probas_dict, feature_cols)

    # 4. Feature Ablation: Behavioral+NLP vs Behavioral+NLP+Graph
    print("\nConducting Feature Ablation (Behavioral+NLP vs +Graph):", flush=True)
    (X_tr_no_g, y_tr_no_g), _, (X_te_no_g, y_te_no_g), cols_no_g = load_dataset_and_splits(include_graph_features=False)
    rf_no_g = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_no_g.fit(X_tr_no_g, y_tr_no_g)
    acc_no_g = accuracy_score(y_te_no_g, rf_no_g.predict(X_te_no_g))
    f1_no_g  = f1_score(y_te_no_g, rf_no_g.predict(X_te_no_g), average="macro")

    rf_with_g = models_dict["Random Forest"]
    acc_with_g = results_df[results_df["Model"] == "Random Forest"]["Accuracy"].values[0]
    f1_with_g  = results_df[results_df["Model"] == "Random Forest"]["F1_Macro"].values[0]

    ablation = {
        "Behavioral_plus_NLP": {"features": len(cols_no_g), "accuracy": round(acc_no_g, 4), "f1_macro": round(f1_no_g, 4)},
        "Behavioral_plus_NLP_plus_Graph": {"features": len(feature_cols), "accuracy": round(acc_with_g, 4), "f1_macro": round(f1_with_g, 4)},
        "Delta_Accuracy": round(acc_with_g - acc_no_g, 4),
        "Delta_F1_Macro": round(f1_with_g - f1_no_g, 4),
    }
    with open(RESULTS_DIR / "ablation_graph_lift.json", "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"  Ablation Results: Behavioral+NLP (71 feats) -> Acc: {acc_no_g*100:.2f}%, F1: {f1_no_g:.4f}", flush=True)
    print(f"                    + Graph Feats   (80 feats) -> Acc: {acc_with_g*100:.2f}%, F1: {f1_with_g:.4f}", flush=True)
    print(f"                    Lift from Graph Features    -> Delta Acc: {ablation['Delta_Accuracy']*100:+.2f}%, Delta F1: {ablation['Delta_F1_Macro']:+.4f}", flush=True)

    print("\nModule 8 Traditional ML Pipeline Completed Successfully!", flush=True)


if __name__ == "__main__":
    run_baseline_pipeline()
