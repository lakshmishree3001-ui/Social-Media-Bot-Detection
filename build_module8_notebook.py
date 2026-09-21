"""
build_module8_notebook.py
Generates module8_traditional_ml_baseline.ipynb programmatically.
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
    md("""# Module 8 — Traditional Machine Learning Baselines

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Benchmark Classical Classifiers**: Before introducing Graph Neural Networks, establish rigorous baselines using 5 standard machine learning models:
   - **Logistic Regression** (linear reference with L2 regularization)
   - **Decision Tree** (non-linear tree baseline)
   - **Random Forest** (ensemble bagging)
   - **XGBoost** (gradient boosted decision trees)
   - **Support Vector Machine (SVM)** (RBF kernel with probability calibration)
2. **Feature Configurations**:
   - Configuration A: **Behavioral + NLP Features** (71 dimensions)
   - Configuration B: **Behavioral + NLP + Graph Features** (80 dimensions)
3. **Rigorous Evaluation on Test Set ($N=1,500$)**:
   - Accuracy, Precision, Recall, F1-Score (Macro & Weighted), ROC-AUC (One-vs-Rest)
   - Training duration & inference latency
4. **Diagnostic Visualizations**:
   - Normalized confusion matrices
   - ROC curves (Macro OvR)
   - Top-20 feature importance rankings
5. **Bridge to GNNs (Module 9)**:
   - Understand the i.i.d. assumption limitation of traditional ML and motivate relational graph learning.
"""),

    code(f"""import sys, os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

ROOT          = Path(r'{ROOT_STR}')
FEATURES_DIR  = ROOT / 'data' / 'features'
SPLITS_DIR    = ROOT / 'data' / 'splits'
RESULTS_DIR   = ROOT / 'data' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('Environment initialised. ROOT:', ROOT)"""),

    md("""## 8.1 Load Multimodal Features & Train/Val/Test Splits
We load:
- `node_features.csv` (35 behavioral features)
- `text_features.csv` (36 NLP features)
- `graph_features.csv` (9 graph features)
Aligned across the official train (70%), validation (15%), and test (15%) splits.
"""),

    code("""from src.models.baseline_models import load_dataset_and_splits

(X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_dataset_and_splits(include_graph_features=True)

print(f'Total Features Loaded : {len(feature_cols)}')
print(f'Training Set Size     : {X_train.shape[0]:,} samples')
print(f'Validation Set Size   : {X_val.shape[0]:,} samples')
print(f'Test Set Size         : {X_test.shape[0]:,} samples')
print('\\nClass distribution in Test Set:')
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    name = {0: 'Human', 1: 'Bot', 2: 'Suspicious'}.get(u)
    print(f'  Class {u} ({name}): {c:,} accounts ({c/len(y_test)*100:.1f}%)')"""),

    md("""## 8.2 Instantiate Baseline Classifiers
We initialize all 5 models with standardized, reproducible configurations.
"""),

    code("""from src.models.baseline_models import get_baseline_models

models_dict = get_baseline_models()
for name, m in models_dict.items():
    print(f'• {name:25s} -> {m.__class__.__name__}')"""),

    md("""## 8.3 Model Training & Test Set Evaluation
We train each model on the 7,000 training accounts and evaluate on the 1,500 unseen test accounts.
"""),

    code("""from src.models.baseline_models import evaluate_model

results_list = []
preds_dict   = {}
probas_dict  = {}

for name, model in models_dict.items():
    metrics, y_pred, y_proba = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results_list.append(metrics)
    preds_dict[name]  = y_pred
    probas_dict[name] = y_proba

results_df = pd.DataFrame(results_list)
print('\\n' + '=' * 80)
print('SUMMARY BASELINE PERFORMANCE COMPARISON (TEST SET):')
print('=' * 80)
print(results_df[['Model', 'Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'ROC_AUC', 'Train_Time_Sec']].to_string(index=False))"""),

    md("""## 8.4 Model Comparison Visualization
Bar chart comparing Accuracy, Macro F1, and ROC-AUC across all algorithms.
"""),

    code("""fig, ax = plt.subplots(figsize=(11, 5))
df_melted = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'F1_Macro', 'ROC_AUC'], var_name='Metric', value_name='Score')
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=ax, palette='Blues_d')
ax.set_ylim(0.5, 1.05)
ax.set_title('Module 8 — Traditional ML Baseline Model Performance Comparison', fontsize=12, fontweight='bold')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_comparison_barplot.png', dpi=120)
plt.show()"""),

    md("""## 8.5 Confusion Matrices Across All Models
Normalized confusion matrices verifying precision and recall per class (Human, Bot, Suspicious).
"""),

    code("""fig, axes = plt.subplots(1, len(models_dict), figsize=(20, 3.8))
for i, (name, _) in enumerate(models_dict.items()):
    cm = confusion_matrix(y_test, preds_dict[name], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[i],
                xticklabels=['Human', 'Bot', 'Susp'], yticklabels=['Human', 'Bot', 'Susp'])
    axes[i].set_title(name, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    if i == 0:
        axes[i].set_ylabel('True Label')
    else:
        axes[i].set_ylabel('')

plt.suptitle('Module 8 — Normalized Confusion Matrices (Unseen Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrices_traditional_ml.png', dpi=120)
plt.show()"""),

    md("""## 8.6 Receiver Operating Characteristic (ROC) Curves
Multi-class One-vs-Rest macro ROC curves.
"""),

    code("""fig, ax = plt.subplots(figsize=(8.5, 5.5))
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = ['#1976D2', '#388E3C', '#D32F2F', '#7B1FA2', '#FF8F00']

for (name, _), color in zip(models_dict.items(), colors):
    y_prob = probas_dict[name]
    if y_prob is not None:
        fpr_list, tpr_list = [], []
        for c in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob[:, c])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(3):
            mean_tpr += np.interp(all_fpr, fpr_list[c], tpr_list[c])
        mean_tpr /= 3.0
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=color, lw=2, label=f'{name} (Macro AUC = {macro_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Module 8 — Receiver Operating Characteristic (ROC) Curves (Macro OvR)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'roc_curves_traditional_ml.png', dpi=120)
plt.show()"""),

    md("""## 8.7 Feature Importance Rankings
Gini importance from Random Forest and Gain importance from XGBoost.
"""),

    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
rf = models_dict['Random Forest']
xgb_m = models_dict['XGBoost']

rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=ax1, palette='viridis')
ax1.set_title('Random Forest — Top 15 Most Predictive Features', fontsize=11, fontweight='bold')
ax1.set_xlabel('Gini Importance')

xgb_imp = pd.Series(xgb_m.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=ax2, palette='magma')
ax2.set_title('XGBoost — Top 15 Most Predictive Features', fontsize=11, fontweight='bold')
ax2.set_xlabel('Gain Importance')

plt.suptitle('Module 8 — Feature Importance Rankings (Tree Ensemble Baselines)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'feature_importance_traditional_ml.png', dpi=120)
plt.show()"""),

    md("""## 8.8 Why Traditional ML Is Not Sufficient & Transition to GNNs (Module 9)

While traditional machine learning models achieve high accuracy on standard tabular and content features, they suffer from fundamental architectural limitations when applied to social networks:

1. **The i.i.d. Assumption**:
   - Traditional ML assumes each account is an independent and identically distributed data point ($P(Y_i | X_i)$).
   - In reality, social media accounts are deeply interdependent ($P(Y_i | X_i, X_j, A_{ij})$).
2. **Blindness to Coordinated Camouflage**:
   - Sophisticated bot farms distribute activity across hundreds of accounts where no single account looks obviously automated in isolation.
   - Only by tracing interaction topology (who retweets whom, who replies to whom in synchronized bursts) can coordinated botnets be exposed.
3. **Information Loss from Hand-Crafted Graph Metrics**:
   - Hand-crafted metrics like PageRank or degree reduce a complex multi-hop neighborhood into a few scalar numbers.
   - GNNs learn continuous neural message passing over multi-hop paths directly.

---

**Next:** **Module 9 — Introduction to Graph Neural Networks (Neighborhood Aggregation & Message Passing)**
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

path = ROOT / "module8_traditional_ml_baseline.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
