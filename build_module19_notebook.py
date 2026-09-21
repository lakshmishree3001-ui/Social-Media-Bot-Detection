"""
build_module19_notebook.py
Generates module19_explainable_ai.ipynb programmatically.
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
    md("""# Module 19 — Explainable AI (XAI) for Social Media Bot Detection

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Model Interpretability in High-Stakes Detection**:
   - Rather than acting as a black box that merely outputs "Bot" or "Human", the AI must output transparent, actionable diagnostic rationale.
2. **Local Feature Attribution (Gradient $\\times$ Input)**:
   - Quantify which behavioral, NLP, and graph features pushed the prediction toward Bot or Human.
3. **Neighborhood Attention Analysis (GAT)**:
   - Inspect multi-head attention weights $\\alpha_{ij}$ to identify influential connections.
4. **Neighbor Homophily & Echo Chamber Quantification**:
   - Calculate the proportion of 1-hop neighbors that are bots vs authentic accounts.
5. **Multi-Panel Visual Explanation Card**:
   - Generate local feature attribution bars and 1-hop neighborhood ego graphs.
"""),

    code(f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT         = Path(r'{ROOT_STR}')
RESULTS_DIR  = ROOT / 'data' / 'results'
MODELS_DIR   = ROOT / 'data' / 'models'
FEATURES_DIR = ROOT / 'data' / 'features'
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('Explainable AI Engine Initialised.')"""),

    md("""## 19.1 Initialize BotGNNExplainer
We instantiate the explainability engine with our trained GAT model.
"""),

    code("""from src.explainability.gnn_explainer import BotGNNExplainer

explainer = BotGNNExplainer()
print('Explainer loaded with GAT model.')
print(f'Total Accounts in Explainer Context: {len(explainer.nodes_df):,}')"""),

    md("""## 19.2 Explaining a Bot Account (Case Study)
We generate the full explainability profile and inspect:
- Decision probability
- Local feature attribution
- Neighbor breakdown
- Key diagnostic rationale
"""),

    code("""bot_uid = explainer.nodes_df[explainer.nodes_df['account_type'] == 'bot'].iloc[0]['user_id']
exp_bot = explainer.explain_account(bot_uid)

print(f'--- Diagnostic Rationale for User {exp_bot[\"user_id\"]} ---')
print(f'Prediction      : {exp_bot[\"prediction\"]} (Confidence: {exp_bot[\"probabilities\"][\"bot\"]*100:.1f}%)')
print(f'Community ID    : #{exp_bot[\"community_id\"]}')
print(f'Coordination Sc : {exp_bot[\"coordination_score\"]}')
print(f'Neighbors       : {exp_bot[\"neighbor_analysis\"][\"total_neighbors\"]} total ({exp_bot[\"neighbor_analysis\"][\"bot_neighbor_ratio\"]*100:.1f}% Bots)')

print('\\nKey Diagnostic Signals Detected:')
for s in exp_bot['key_diagnostic_signals']:
    print(f'  [+] {s}')

print('\\nTop Attributing Features:')
for f in exp_bot['top_contributing_features']:
    print(f'  • {f[\"feature\"]:25s} | Attr: {f[\"attribution\"]:+.4f} | Val: {f[\"normalized_value\"]:+.4f}')"""),

    md("""## 19.3 Visualizing Bot Explanation (Ego Network + Feature Attribution)
We render the dual-panel explanation card.
"""),

    code("""img_path = explainer.visualize_explanation(bot_uid)

from PIL import Image
img = Image.open(img_path)
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.imshow(img)
ax.axis('off')
plt.tight_layout()
plt.show()"""),

    md("""## 19.4 Explaining an Authentic Human Account
We examine an authentic human account to verify that explanation signals correctly reflect organic behavior.
"""),

    code("""human_uid = explainer.nodes_df[explainer.nodes_df['account_type'] == 'human'].iloc[0]['user_id']
exp_human = explainer.explain_account(human_uid)

print(f'--- Diagnostic Rationale for User {exp_human[\"user_id\"]} ---')
print(f'Prediction      : {exp_human[\"prediction\"]} (Confidence: {exp_human[\"probabilities\"][\"human\"]*100:.1f}%)')
print(f'Community ID    : #{exp_human[\"community_id\"]}')
print(f'Coordination Sc : {exp_human[\"coordination_score\"]}')
print(f'Neighbors       : {exp_human[\"neighbor_analysis\"][\"total_neighbors\"]} total ({exp_human[\"neighbor_analysis\"][\"bot_neighbor_ratio\"]*100:.1f}% Bots)')

print('\\nKey Diagnostic Signals Detected:')
for s in exp_human['key_diagnostic_signals']:
    print(f'  [+] {s}')

print('\\nTop Attributing Features:')
for f in exp_human['top_contributing_features']:
    print(f'  • {f[\"feature\"]:25s} | Attr: {f[\"attribution\"]:+.4f} | Val: {f[\"normalized_value\"]:+.4f}')"""),

    md("""## 19.5 Visualizing Human Explanation
Rendering the explanation graphic for the human profile.
"""),

    code("""img_human_path = explainer.visualize_explanation(human_uid)
img_h = Image.open(img_human_path)
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.imshow(img_h)
ax.axis('off')
plt.tight_layout()
plt.show()"""),

    md("""## Module 19 Summary

| Component | Technical Implementation | Benefit |
|---|---|---|
| **Local Feature Attribution** | Gradient $\\times$ Input | Isolates specific profile/content metrics driving the decision |
| **Neighborhood Attention** | GAT $\\alpha_{ij}$ coefficients | Shows which adjacent connections the model relied upon |
| **Echo Chamber Risk** | Neighbor bot concentration | Distinguishes isolated bots from coordinated bot farms |
| **Human-Readable Rationale** | Rule-based signal translator | Provides plain-language justifications for safety reviewers |

---

**Next:** **Module 20 — Final End-to-End Bot Detection System**
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

path = ROOT / "module19_explainable_ai.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
