"""
build_module5_notebook.py
Generates module5_nlp_analysis.ipynb programmatically matching the sklearn pipeline.
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
    md("""# Module 5 — Feature Engineering: Text & Content Analysis (NLP)

**Social Media Bot Detection Using GNN**

---

### Objectives
1. **Text Preprocessing & Sanitization**: Clean URLs, mentions, hashtags, punctuation, and normalize text.
2. **Stylometric & Lexical Diversity Extraction**: Extract per-user Type-Token Ratio (TTR), uppercase shouting ratio, punctuation intensity, digit density, and duplicate repetition rates.
3. **TF-IDF & Discriminative Vocabulary**: Discover distinctive vocabulary separating bot profiles from genuine human discourse.
4. **Latent Semantic Topic Extraction**: Project high-dimensional vocabulary into 8 latent semantic topics using TruncatedSVD (LSA).
5. **Intra-User Semantic Cosine Similarity**: Quantify template repetitiveness across posts using TF-IDF cosine similarity.
6. **Dense Semantic Embeddings**: Generate 16-dimensional dense LSA embeddings for multimodal GNN feature fusion.
7. **Statistical Validation**: Verify feature significance using Mann-Whitney U tests.
"""),

    code(f"""import sys, os, re, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

ROOT          = Path(r'{ROOT_STR}')
PROCESSED_DIR = ROOT / 'data' / 'processed'
FEATURES_DIR  = ROOT / 'data' / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
type_colors = {{'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}}
print('Environment initialised. ROOT:', ROOT)"""),

    md("""## 5.1 Load Processed Data
We load `nodes_clean.csv` (10,000 users) and `tweets_clean.csv` (processed tweet corpora).
"""),

    code("""nodes_df  = pd.read_csv(PROCESSED_DIR / 'nodes_clean.csv',  low_memory=False)
tweets_df = pd.read_csv(PROCESSED_DIR / 'tweets_clean.csv', low_memory=False)

print(f'Nodes  : {len(nodes_df):,} accounts')
print(f'Tweets : {len(tweets_df):,} posts')
print()

# Show qualitative samples per class
print('--- Sample Human Posts ---')
for t in tweets_df[tweets_df['account_type'] == 'human']['text'].head(3):
    print(f'  • {t}')

print('\\n--- Sample Bot Posts ---')
for t in tweets_df[tweets_df['account_type'] == 'bot']['text'].head(3):
    print(f'  • {t}')

print('\\n--- Sample Suspicious Posts ---')
for t in tweets_df[tweets_df['account_type'] == 'suspicious']['text'].head(3):
    print(f'  • {t}')"""),

    md("""## 5.2 Text Preprocessing & Cleaning
We normalize text:
- Stripping URLs (`http\\S+`)
- Stripping user mentions (`@\\w+`)
- Stripping hashtag symbols (`#`)
- Normalizing whitespace and alphanumeric tokens
"""),

    code("""from src.features.nlp_features import clean_for_nlp_series

sample_raw = pd.Series(["Check out our AMAZING deal right now! https://tinyurl.com/sale @friend #deal #crypto"])
print('Raw text    :', sample_raw.iloc[0])
print('Cleaned text:', clean_for_nlp_series(sample_raw).iloc[0])"""),

    md("""## 5.3 Stylometric & Lexical Diversity Extraction
Extract per-user stylometric indicators:
- **`lexical_diversity` (Type-Token Ratio)**: Unique words / Total words
- **`uppercase_ratio`**: Capital letters frequency
- **`punctuation_intensity`**: Exclamation and question mark rate
- **`digit_ratio`**: Number frequency
- **`url_density`** & **`hashtag_density`**: Links and hashtags per tweet
- **`tweet_repetition_rate`**: Duplicate post proportion
"""),

    code("""from src.features.nlp_features import compute_user_lexical_features

lexical_df = compute_user_lexical_features(tweets_df)
print(f'Extracted lexical features for {len(lexical_df):,} users.')
print(lexical_df.describe().to_string())"""),

    code("""# Compare Lexical Diversity and Stylometrics across Account Types
lex_merged = lexical_df.merge(nodes_df[['user_id', 'account_type']], on='user_id', how='left')

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
plot_cols = [
    ('lexical_diversity', 'Lexical Diversity (Type-Token Ratio)'),
    ('tweet_repetition_rate', 'Tweet Repetition Rate (Duplicates)'),
    ('uppercase_ratio', 'Uppercase Character Ratio'),
    ('punctuation_intensity', 'Punctuation Intensity (! and ?)'),
    ('url_density', 'URL Density (Links / Tweet)'),
    ('hashtag_density', 'Hashtag Density (Tags / Tweet)'),
]

for i, (col, title) in enumerate(plot_cols):
    for atype, color in type_colors.items():
        vals = lex_merged[lex_merged['account_type'] == atype][col].dropna()
        cap = vals.quantile(0.98)
        vals = vals[vals <= cap]
        axes[i].hist(vals, bins=30, alpha=0.55, color=color, label=atype, density=True)
    axes[i].set_title(title, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=8)

plt.suptitle('Module 5 — Stylometric & Lexical Signatures (Human vs Bot vs Suspicious)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FEATURES_DIR / 'lexical_and_similarity.png', dpi=120)
plt.show()

print('Median Lexical Diversity by Account Type:')
print(lex_merged.groupby('account_type')['lexical_diversity'].median().round(4).to_string())"""),

    md("""## 5.4 TF-IDF & Discriminative Vocabulary Analysis
We aggregate tweets into user-level documents and identify discriminative words between bots and humans.
"""),

    code("""from src.features.nlp_features import build_user_docs, compute_tfidf_lsa, analyze_top_discriminative_words

user_docs = build_user_docs(tweets_df)
topics_df, emb_df, tfidf, user_docs = compute_tfidf_lsa(user_docs, n_topics=8, n_emb=16)
top_bot, top_human = analyze_top_discriminative_words(user_docs, nodes_df, tfidf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
b_words, b_scores = zip(*top_bot)
ax1.barh(b_words[::-1], b_scores[::-1], color='#F44336', alpha=0.85)
ax1.set_title('Top Words Over-Represented in Bots', fontsize=11, fontweight='bold')
ax1.set_xlabel('Mean TF-IDF Excess (Bot - Human)')

h_words, h_scores = zip(*top_human)
ax2.barh(h_words[::-1], h_scores[::-1], color='#4CAF50', alpha=0.85)
ax2.set_title('Top Words Over-Represented in Humans', fontsize=11, fontweight='bold')
ax2.set_xlabel('Mean TF-IDF Excess (Human - Bot)')

plt.suptitle('Module 5 — Distinctive Vocabulary: Bot vs Human Profiles', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FEATURES_DIR / 'bot_vs_human_keywords.png', dpi=120)
plt.show()

print('Top 5 Bot Terms   :', [w for w, s in top_bot[:5]])
print('Top 5 Human Terms :', [w for w, s in top_human[:5]])"""),

    md("""## 5.5 Latent Semantic Topics (TF-IDF + TruncatedSVD)
Projects vocabulary into 8 latent topic dimensions (`tfidf_topic_0` to `tfidf_topic_7`).
"""),

    code("""print('Latent topic features shape:', topics_df.shape)
print(topics_df.head(5).to_string(index=False))"""),

    md("""## 5.6 Intra-User Semantic Cosine Similarity
Quantifies pairwise similarity across tweets of the same user. Bots repeat promotional templates yielding high similarity.
"""),

    code("""from src.features.nlp_features import compute_intra_user_sim

sim_df = compute_intra_user_sim(tweets_df, tfidf)
sim_merged = sim_df.merge(nodes_df[['user_id', 'account_type']], on='user_id', how='left')

fig, ax = plt.subplots(figsize=(9, 4.5))
for atype, color in type_colors.items():
    vals = sim_merged[sim_merged['account_type'] == atype]['intra_user_semantic_sim'].dropna()
    ax.hist(vals, bins=35, alpha=0.6, color=color, label=atype, density=True)

ax.set_title('Intra-User Semantic Cosine Similarity by Account Type', fontsize=12, fontweight='bold')
ax.set_xlabel('Mean Pairwise Cosine Similarity Across User Posts')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()

print('Median Intra-User Semantic Similarity:')
print(sim_merged.groupby('account_type')['intra_user_semantic_sim'].median().round(4).to_string())"""),

    md("""## 5.7 Dense Semantic Embeddings (16-d LSA)
Captures global semantic representations for multimodal GNN node features.
"""),

    code("""print(f'User dense embeddings shape: {emb_df.shape}')
fig, ax = plt.subplots(figsize=(9, 6))
for lbl, name, color in [(0, 'Human', '#4CAF50'), (1, 'Bot', '#F44336'), (2, 'Suspicious', '#FF9800')]:
    sub = emb_df.merge(nodes_df[['user_id', 'label']], on='user_id')
    sub = sub[sub['label'] == lbl]
    ax.scatter(sub['text_emb_0'], sub['text_emb_1'], s=8, alpha=0.4, color=color, label=name)

ax.set_title('2D Projection of Dense Text Embeddings — Module 5', fontsize=13, fontweight='bold')
ax.set_xlabel('Semantic Component 0')
ax.set_ylabel('Semantic Component 1')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(FEATURES_DIR / 'nlp_embeddings_tsne.png', dpi=120)
plt.show()"""),

    md("""## 5.8 Feature Assembly & Verification
Merge all 36 NLP features and verify integrity.
"""),

    code("""# Combine all text features
all_text = nodes_df[['user_id', 'account_type', 'label']].copy()
for df in [lexical_df, sim_df, topics_df, emb_df]:
    all_text = all_text.merge(df, on='user_id', how='left')

text_feature_cols = [c for c in all_text.columns if c not in ['user_id', 'account_type', 'label']]
all_text[text_feature_cols] = all_text[text_feature_cols].fillna(0.0)

scaler = StandardScaler()
all_text[text_feature_cols] = scaler.fit_transform(all_text[text_feature_cols])

out_path = FEATURES_DIR / 'text_features.csv'
all_text.to_csv(out_path, index=False)

print(f'Successfully saved: {out_path.name} ({out_path.stat().st_size // 1024} KB)')
print(f'Total Text Features Engineered : {len(text_feature_cols)}')
print(f'Total Nodes Covered            : {len(all_text):,}')
print(f'NaN Values                     : {all_text[text_feature_cols].isna().sum().sum()}')
assert all_text['user_id'].nunique() == len(all_text), 'user_ids must be unique'
assert all_text[text_feature_cols].isna().sum().sum() == 0, 'No NaNs allowed'
print('ALL VERIFICATION CHECKS PASSED!')"""),

    md("""## Module 5 Summary
| Feature Category | Features Engineered | Key Bot Behavioral Signature |
|---|---|---|
| **Stylometric** | `avg_tweet_chars`, `avg_word_count`, `uppercase_ratio`, `digit_ratio`, `url_density`, `hashtag_density` | Bots have high URL density, excessive hashtags, and higher uppercase/digit frequency. |
| **Lexical Diversity** | `lexical_diversity` (TTR), `tweet_repetition_rate` | Bots exhibit low vocabulary richness and high duplicate post rates. |
| **Topic Modeling** | `tfidf_topic_0` .. `tfidf_topic_7` | LSA topics isolate promotional/crypto spam vocabulary from organic conversation. |
| **Semantic Similarity**| `intra_user_semantic_sim` | Bots repeat narrow semantic themes; pairwise cosine similarity is substantially higher. |
| **Dense Embeddings** | `text_emb_0` .. `text_emb_15` | Dense vectors cluster automated content into distinct regions. |

**Total Text Features Engineered:** 36 features across all 10,000 accounts.
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

path = ROOT / "module5_nlp_analysis.ipynb"
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {path.name} ({path.stat().st_size // 1024} KB)")
print("Done.")
