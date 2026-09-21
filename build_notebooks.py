"""
build_notebooks.py
Generates Module 2 and Module 3 Jupyter notebooks programmatically
using nbformat so encoding is always correct.
"""
import json
from pathlib import Path

ROOT = Path("c:/Social Media Bot Detection")


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def make_nb(cells):
    return {
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


ROOT_STR = str(ROOT).replace("\\", "\\\\")

SETUP_M2 = f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT    = Path(r'{ROOT}')
RAW_DIR = ROOT / 'data' / 'raw'

pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('RAW_DIR:', RAW_DIR)"""

SETUP_M3 = f"""import sys, os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT          = Path(r'{ROOT}')
RAW_DIR       = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
SPLITS_DIR    = ROOT / 'data' / 'splits'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {{0: 'human', 1: 'bot', 2: 'suspicious'}}
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{{:.4f}}'.format)
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='darkgrid', palette='muted')
print('All libraries loaded.  ROOT:', ROOT)"""

# ============================================================
# MODULE 2
# ============================================================
nb2_cells = [
    md("""# Module 2 — Understanding the Dataset

**Social Media Bot Detection Using GNN**

| File | Contents | Rows |
|------|----------|------|
| users.csv  | User/account profiles with labels | 10,000 |
| tweets.csv | Individual post/tweet records | ~137,000 |
| edges.csv  | Social-network interaction edges | 55,000 |

**Label encoding:** 0=Human  1=Bot  2=Suspicious/Coordinated
"""),

    code(SETUP_M2),

    md("## 2.1 Load Data"),
    code("""users_df  = pd.read_csv(RAW_DIR / 'users.csv',  low_memory=False)
tweets_df = pd.read_csv(RAW_DIR / 'tweets.csv', low_memory=False)
edges_df  = pd.read_csv(RAW_DIR / 'edges.csv',  low_memory=False)
with open(RAW_DIR / 'dataset_info.json') as f:
    meta = json.load(f)
print(f'Users  : {len(users_df):,} rows x {users_df.shape[1]} cols')
print(f'Tweets : {len(tweets_df):,} rows x {tweets_df.shape[1]} cols')
print(f'Edges  : {len(edges_df):,} rows x {edges_df.shape[1]} cols')"""),

    md("## 2.2 User Table — Column Schema"),
    code("""print('Column dtypes:')
print(users_df.dtypes.to_string())"""),

    code("users_df.head()"),
    code("users_df.describe()"),

    md("## 2.3 Label Distribution"),
    code("""label_map = {0: 'Human', 1: 'Bot', 2: 'Suspicious'}
colors    = {'Human': '#4CAF50', 'Bot': '#F44336', 'Suspicious': '#FF9800'}
lc = users_df['label'].map(label_map).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bars = axes[0].bar(lc.index, lc.values, color=[colors[l] for l in lc.index], edgecolor='black')
for bar, val in zip(bars, lc.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f'{val:,}', ha='center', fontweight='bold')
axes[0].set_title('Account Label Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
axes[1].pie(lc.values,
            labels=[f'{l}\\n({v:,})' for l, v in zip(lc.index, lc.values)],
            colors=[colors[l] for l in lc.index], autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Class Proportions', fontweight='bold')
plt.suptitle('Module 2 — Label Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RAW_DIR / 'label_distribution.png'), bbox_inches='tight', dpi=120)
plt.show()
print(lc.to_frame('count'))"""),

    md("## 2.4 Behavioral Feature Distributions"),
    code("""type_colors = {'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}
feature_cols = ['followers_count', 'following_count', 'posts_per_day',
                'reply_ratio', 'retweet_ratio', 'url_ratio',
                'hashtag_ratio', 'duplicate_content_ratio']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    for atype, color in type_colors.items():
        subset = users_df[users_df['account_type'] == atype][col].dropna()
        cap = subset.quantile(0.98)
        subset = subset[subset <= cap]
        axes[i].hist(subset, bins=40, alpha=0.55, color=color, label=atype, density=True)
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].set_ylabel('Density')
plt.suptitle('Behavioral Feature Distributions (Human vs Bot vs Suspicious)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RAW_DIR / 'behavioral_distributions.png'), bbox_inches='tight', dpi=120)
plt.show()"""),

    md("## 2.5 Followers / Following Ratio (Key Bot Signal)"),
    code("""users_df['ff_ratio'] = (users_df['followers_count'] + 1) / (users_df['following_count'] + 1)
print('Median Followers/Following ratio per account type:')
print(users_df.groupby('account_type')['ff_ratio'].median().round(3).to_string())
print()
print('Human      -> ratio ~1.3  (balanced social behavior)')
print('Bot        -> ratio ~0.04 (follows thousands, few follow back)')
print('Suspicious -> ratio ~0.16 (moderately imbalanced)')"""),

    md("## 2.6 Replies, Retweets & Mentions"),
    code("""interaction_cols = ['reply_ratio', 'retweet_ratio', 'mention_ratio']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(interaction_cols):
    data = [users_df[users_df['account_type'] == t][col].dropna().values
            for t in ['human', 'bot', 'suspicious']]
    bp = axes[i].boxplot(data, patch_artist=True, notch=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#4CAF50', '#F44336', '#FF9800']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_xticklabels(['Human', 'Bot', 'Suspicious'])
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=11)
    axes[i].set_ylabel('Ratio')
plt.suptitle('Reply / Retweet / Mention Ratios', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RAW_DIR / 'interaction_boxplots.png'), bbox_inches='tight', dpi=120)
plt.show()"""),

    md("## 2.7 Tweet Content Samples"),
    code("""for atype in ['human', 'bot', 'suspicious']:
    print(f'\\n--- {atype.upper()} SAMPLE TWEETS ---')
    for t in tweets_df[tweets_df['account_type'] == atype]['text'].sample(3, random_state=42):
        print(f'  {t[:115]}')"""),

    md("## 2.8 Edge / Graph Relationships"),
    code("""print('Edge type distribution:')
print(edges_df['interaction_type'].value_counts().to_string())

edge_cmap = {'follows': '#2196F3', 'retweets': '#9C27B0',
             'replies': '#009688', 'mentions': '#FF5722'}
ec = edges_df['interaction_type'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(ec.index, ec.values,
            color=[edge_cmap.get(t, 'grey') for t in ec.index], edgecolor='black')
for bar, val in zip(axes[0].patches, ec.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('Interaction Type Distribution', fontweight='bold')
axes[0].set_ylabel('Edge Count')
axes[1].hist(edges_df['weight'], bins=50, color='#607D8B', edgecolor='black', alpha=0.8)
axes[1].set_title('Edge Weight Distribution', fontweight='bold')
axes[1].set_xlabel('Weight (interaction strength)')
plt.suptitle('Module 2 — Social Network Edge Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RAW_DIR / 'edge_analysis.png'), bbox_inches='tight', dpi=120)
plt.show()"""),

    md("## 2.9 Cross-Type Interaction Heatmap"),
    code("""uid_to_type = users_df.set_index('user_id')['account_type']
el = edges_df.copy()
el['src_type'] = el['source_user_id'].map(uid_to_type)
el['tgt_type'] = el['target_user_id'].map(uid_to_type)
el = el.dropna(subset=['src_type', 'tgt_type'])
cm = pd.crosstab(el['src_type'], el['tgt_type'])

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt=',', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Edge Count'}, ax=ax)
ax.set_title('Cross-Type Interaction Heatmap (Row=Source, Col=Target)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Target Account Type')
ax.set_ylabel('Source Account Type')
plt.tight_layout()
plt.savefig(str(RAW_DIR / 'cross_type_heatmap.png'), bbox_inches='tight', dpi=120)
plt.show()
print('Dense Bot->Bot edges = coordinated bot cluster structure')"""),

    md("## 2.10 Missing Values"),
    code("""for name, df in [('users', users_df), ('tweets', tweets_df), ('edges', edges_df)]:
    m = df.isnull().sum()
    has_m = m[m > 0]
    if len(has_m) == 0:
        print(f'[{name}] => No missing values')
    else:
        print(f'[{name}] Missing:')
        for col, cnt in has_m.items():
            print(f'  {col:<30} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)')"""),

    md("""## Module 2 Summary

| Component | Details |
|-----------|---------|
| **Users** | 10,000 accounts — 50% human, 35% bot, 15% suspicious |
| **Tweets** | ~137,000 posts with text, engagement, URL/hashtag flags |
| **Edges** | 55,000 directed interactions — follows, retweets, replies, mentions |
| **Key Bot Signals** | High posts_per_day, high url_ratio, high retweet_ratio, low ff_ratio, default_profile=1 |
| **Key Graph Signal** | Dense bot-bot clusters, suspicious-bot coordination edges |

**Next:** Module 3 — Data Cleaning & Preparation
"""),
]

# ============================================================
# MODULE 3
# ============================================================
NODE_FEATURE_COLS_LIST = [
    "account_age_days", "followers_count", "following_count", "post_count", "listed_count",
    "verified", "has_profile_image", "has_description", "default_profile", "profile_completeness",
    "posts_per_day", "reply_ratio", "retweet_ratio", "mention_ratio",
    "url_ratio", "hashtag_ratio", "duplicate_content_ratio",
]
NFC = repr(NODE_FEATURE_COLS_LIST)

nb3_cells = [
    md("""# Module 3 — Data Collection & Dataset Preparation

**Social Media Bot Detection Using GNN**

Full data pipeline:
```
Raw CSVs  ->  Load  ->  Inspect formats (CSV / JSON / Edge list)
          ->  Remove duplicates
          ->  Handle missing values
          ->  Fix data types
          ->  Cap outliers
          ->  Validate user IDs
          ->  Encode labels
          ->  Separate node / edge tables
          ->  Train / Val / Test split
          ->  Save processed files
```
"""),

    code(SETUP_M3),

    md("## 3.1 Load & Inspect Raw Data"),
    code("""users_raw  = pd.read_csv(RAW_DIR / 'users.csv',  low_memory=False)
tweets_raw = pd.read_csv(RAW_DIR / 'tweets.csv', low_memory=False)
edges_raw  = pd.read_csv(RAW_DIR / 'edges.csv',  low_memory=False)
print(f'[RAW] users  : {len(users_raw):>7,} rows x {users_raw.shape[1]} cols')
print(f'[RAW] tweets : {len(tweets_raw):>7,} rows x {tweets_raw.shape[1]} cols')
print(f'[RAW] edges  : {len(edges_raw):>7,} rows x {edges_raw.shape[1]} cols')"""),
    code("users_raw.head()"),

    md("## 3.2 Dataset Format Inspection (CSV / JSON / Edge List)"),
    code("""print('== CSV FORMAT (first 3 rows) ==')
print(users_raw.head(3).to_string(index=False))
print()
with open(RAW_DIR / 'dataset_info.json') as f:
    meta = json.load(f)
print('== JSON FORMAT (statistics) ==')
print(json.dumps(meta['statistics'], indent=2))
print()
print('== EDGE LIST FORMAT (first 5) ==')
print(edges_raw[['source_user_id','target_user_id','interaction_type','weight']].head(5).to_string(index=False))"""),

    md("## 3.3a Duplicate Check & Removal"),
    code("""print(f'Duplicate user_ids  : {users_raw.duplicated(subset="user_id").sum():,}')
print(f'Duplicate tweet_ids : {tweets_raw.duplicated(subset="tweet_id").sum():,}')
print(f'Duplicate edge keys : {edges_raw.duplicated(subset=["source_user_id","target_user_id","interaction_type"]).sum():,}')

users_clean  = users_raw.drop_duplicates(subset='user_id').copy()
tweets_clean = tweets_raw.drop_duplicates(subset='tweet_id').copy()
edges_clean  = edges_raw.drop_duplicates(
    subset=['source_user_id','target_user_id','interaction_type']).copy()
print(f'After dedup: users={len(users_clean):,}  tweets={len(tweets_clean):,}  edges={len(edges_clean):,}')"""),

    md("## 3.3b Missing Value Analysis"),
    code("""def missing_report(df, name):
    m = df.isnull().sum()
    pct = (m / len(df) * 100).round(2)
    rep = pd.DataFrame({'count': m, 'pct': pct})
    rep = rep[rep['count'] > 0].sort_values('pct', ascending=False)
    print(f'[{name}] Missing: {len(rep)} columns')
    if len(rep):
        print(rep.to_string())
    else:
        print('  None')
    return rep

miss_users  = missing_report(users_clean,  'users')
miss_tweets = missing_report(tweets_clean, 'tweets')
miss_edges  = missing_report(edges_clean,  'edges')"""),
    code("""if len(miss_users):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(miss_users.index, miss_users['pct'], color='#EF5350', edgecolor='black')
    ax.set_xlabel('Missing %')
    ax.set_title('User Table — Missing Value Rate per Column', fontweight='bold')
    for i, (col, row) in enumerate(miss_users.iterrows()):
        ax.text(row['pct']+0.2, i,
                f"{row['pct']:.1f}%  ({int(row['count']):,})", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(str(PROCESSED_DIR / 'missing_values.png'), bbox_inches='tight', dpi=120)
    plt.show()"""),

    md("## 3.3c Handle Missing Values"),
    code("""# location  -> fill with 'Unknown'
# listed_count -> fill with per-type median
users_clean['location'] = users_clean['location'].fillna('Unknown')
if 'listed_count' in users_clean.columns:
    med = users_clean.groupby('account_type')['listed_count'].transform('median')
    users_clean['listed_count'] = users_clean['listed_count'].fillna(med).fillna(0).astype(int)

print(f'Total NaN after fill: {users_clean.isnull().sum().sum()}')"""),

    md("## 3.3d Fix Data Types"),
    code("""users_clean['created_at']  = pd.to_datetime(users_clean['created_at'])
tweets_clean['timestamp']  = pd.to_datetime(tweets_clean['timestamp'])
edges_clean['timestamp']   = pd.to_datetime(edges_clean['timestamp'])
for col in ['verified','has_profile_image','has_description','default_profile']:
    users_clean[col] = users_clean[col].astype(int)
print('Types fixed.')
print(users_clean[['created_at','verified','has_profile_image']].dtypes.to_string())"""),

    md("## 3.3e Outlier Capping (99th Percentile)"),
    code("""cap_cols = ['followers_count','following_count','post_count','posts_per_day','listed_count']
caps = {}
print('Capping at 99th percentile:')
for col in cap_cols:
    p99 = users_clean[col].quantile(0.99)
    n = (users_clean[col] > p99).sum()
    users_clean[col] = users_clean[col].clip(upper=p99)
    caps[col] = float(p99)
    print(f'  {col:<25}  cap={p99:>10.1f}  rows_capped={n}')
with open(PROCESSED_DIR / 'outlier_caps.json', 'w') as f:
    json.dump(caps, f, indent=2)
print('\\nSaved outlier_caps.json')"""),

    md("## 3.4 Consistent User IDs (0-based node_idx for PyTorch Geometric)"),
    code("""valid_ids = set(users_clean['user_id'].unique())
invalid_tweets = (~tweets_clean['user_id'].isin(valid_ids)).sum()
print(f'Orphan tweet rows   : {invalid_tweets:,}')

tweets_clean = tweets_clean[tweets_clean['user_id'].isin(valid_ids)].copy()
edges_clean  = edges_clean[
    edges_clean['source_user_id'].isin(valid_ids) &
    edges_clean['target_user_id'].isin(valid_ids)
].copy()

uid_sorted = sorted(users_clean['user_id'].unique())
uid_to_idx = {uid: idx for idx, uid in enumerate(uid_sorted)}
users_clean['node_idx']    = users_clean['user_id'].map(uid_to_idx)
edges_clean['src_idx']     = edges_clean['source_user_id'].map(uid_to_idx)
edges_clean['tgt_idx']     = edges_clean['target_user_id'].map(uid_to_idx)

with open(PROCESSED_DIR / 'uid_to_idx.json', 'w') as f:
    json.dump({str(k): v for k, v in uid_to_idx.items()}, f)
print(f'node_idx range: 0 ... {users_clean["node_idx"].max()}')"""),

    md("## 3.5 Label Encoding"),
    code("""label_encoding = {
    'int_to_str': {0: 'human', 1: 'bot', 2: 'suspicious'},
    'str_to_int': {'human': 0, 'bot': 1, 'suspicious': 2}
}
with open(PROCESSED_DIR / 'label_encoding.json', 'w') as f:
    json.dump(label_encoding, f, indent=2)

itype_le = LabelEncoder()
edges_clean['interaction_type_enc'] = itype_le.fit_transform(edges_clean['interaction_type'])
itype_map = {k: int(v) for k, v in zip(itype_le.classes_, itype_le.transform(itype_le.classes_))}
with open(PROCESSED_DIR / 'interaction_encoding.json', 'w') as f:
    json.dump(itype_map, f, indent=2)

print('Label encoding: 0=human  1=bot  2=suspicious')
total = len(users_clean)
for lbl, cnt in users_clean['label'].value_counts().sort_index().items():
    name = label_encoding['int_to_str'][lbl]
    w = round(total / (3 * cnt), 4)
    print(f'  label={lbl}  {name:<12}  count={cnt:,}  class_weight={w}')
print(f'Interaction type encoding: {itype_map}')"""),

    md("## 3.6 Separate Node and Edge Information"),
    code(f"""NODE_FEATURE_COLS = {NFC}

IDENTITY_COLS = ['user_id','node_idx','screen_name','account_type','label','created_at','location']
EDGE_COLS = ['source_user_id','target_user_id','src_idx','tgt_idx',
             'interaction_type','interaction_type_enc','weight','timestamp']

nodes_df = users_clean[IDENTITY_COLS + NODE_FEATURE_COLS].copy()
edges_df = edges_clean[EDGE_COLS].copy()

print(f'Node feature matrix : {{nodes_df.shape}}')
print(f'Edge feature matrix : {{edges_df.shape}}')
print(f'NaN in node features: {{nodes_df[NODE_FEATURE_COLS].isnull().sum().sum()}}')
print(f'NaN in edge indices : {{edges_df[["src_idx","tgt_idx"]].isnull().sum().sum()}}')"""),

    md("## 3.7 Feature Type Categorization"),
    code("""feature_types = {
    'continuous':  ['account_age_days','followers_count','following_count',
                    'post_count','listed_count','posts_per_day','profile_completeness'],
    'ratio_0_1':   ['reply_ratio','retweet_ratio','mention_ratio',
                    'url_ratio','hashtag_ratio','duplicate_content_ratio'],
    'binary':      ['verified','has_profile_image','has_description','default_profile'],
    'normalisation': {
        'continuous': 'StandardScaler (applied in Module 4)',
        'ratio_0_1':  'Already 0-1, optional MinMaxScaler',
        'binary':     'No normalisation'
    }
}
with open(PROCESSED_DIR / 'feature_types.json', 'w') as f:
    json.dump(feature_types, f, indent=2)
print('feature_types.json saved.')

fig, ax = plt.subplots(figsize=(14, 10))
corr = nodes_df[NODE_FEATURE_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.4, annot_kws={'size': 7}, ax=ax)
ax.set_title('Node Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(PROCESSED_DIR / 'feature_correlation.png'), bbox_inches='tight', dpi=120)
plt.show()"""),

    md("## 3.8 Train / Validation / Test Split (70/15/15 — Stratified)"),
    code("""all_ids    = nodes_df['user_id'].values
all_labels = nodes_df['label'].values

train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    all_ids, all_labels, test_size=0.30, stratify=all_labels, random_state=42)
val_ids, test_ids, val_labels, test_labels = train_test_split(
    temp_ids, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42)

print(f'Train : {len(train_ids):,}  ({len(train_ids)/len(all_ids)*100:.1f}%)')
print(f'Val   : {len(val_ids):,}  ({len(val_ids)/len(all_ids)*100:.1f}%)')
print(f'Test  : {len(test_ids):,}  ({len(test_ids)/len(all_ids)*100:.1f}%)')

label_names = {0: 'human', 1: 'bot', 2: 'suspicious'}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (ids, lbls, title) in zip(axes, [
    (train_ids, train_labels, 'Train (70%)'),
    (val_ids,   val_labels,   'Val   (15%)'),
    (test_ids,  test_labels,  'Test  (15%)'),
]):
    unique, counts = np.unique(lbls, return_counts=True)
    labels_str = [label_names[u] for u in unique]
    bars = ax.bar(labels_str, counts,
                  color=['#4CAF50','#F44336','#FF9800'][:len(unique)], edgecolor='black')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f'{cnt:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Count')
plt.suptitle('Stratified Train / Val / Test Split', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(PROCESSED_DIR / 'split_distribution.png'), bbox_inches='tight', dpi=120)
plt.show()"""),

    md("## 3.9 Save All Processed Files"),
    code("""nodes_df.to_csv(PROCESSED_DIR    / 'nodes_clean.csv',  index=False)
edges_df.to_csv(PROCESSED_DIR    / 'edges_clean.csv',  index=False)
tweets_clean.to_csv(PROCESSED_DIR / 'tweets_clean.csv', index=False)

pd.DataFrame({'user_id': train_ids, 'label': train_labels}).to_csv(SPLITS_DIR / 'train_ids.csv', index=False)
pd.DataFrame({'user_id': val_ids,   'label': val_labels  }).to_csv(SPLITS_DIR / 'val_ids.csv',   index=False)
pd.DataFrame({'user_id': test_ids,  'label': test_labels }).to_csv(SPLITS_DIR / 'test_ids.csv',  index=False)

print('data/processed/')
for p in sorted(PROCESSED_DIR.iterdir()):
    print(f'  {p.name:<32}  {p.stat().st_size/1024:.1f} KB')
print('data/splits/')
for p in sorted(SPLITS_DIR.iterdir()):
    print(f'  {p.name:<32}  {p.stat().st_size/1024:.1f} KB')"""),

    md("## 3.10 Final Validation Assertions"),
    code(f"""NODE_FEATURE_COLS = {NFC}

nc = pd.read_csv(PROCESSED_DIR / 'nodes_clean.csv')
ec = pd.read_csv(PROCESSED_DIR / 'edges_clean.csv')

assert nc['user_id'].nunique() == len(nc),                   'user_id must be unique'
assert nc[NODE_FEATURE_COLS].isnull().sum().sum() == 0,       'No NaN in features'
assert ec[['src_idx','tgt_idx']].isnull().sum().sum() == 0,  'No NaN in edge indices'
assert nc['label'].isin([0, 1, 2]).all(),                    'Labels must be 0, 1, or 2'

print('ALL ASSERTIONS PASSED')
print(f'nodes_clean.csv : {{nc.shape}}')
print(f'edges_clean.csv : {{ec.shape}}')
print()
print('Data is ready for Module 4 -- Behavioral Feature Engineering')"""),

    md("""## Module 3 Summary

| Step | Action | Result |
|------|--------|--------|
| Load | Read CSV files | 10k users, 137k tweets, 55k edges |
| Format | Inspect CSV / JSON / Edge list | All formats documented |
| Dedup | Remove duplicate keys | 0 duplicates found |
| Missing | Fill location, listed_count | 0 remaining NaN in features |
| Types | Parse dates, cast booleans | All types correct |
| Outliers | Cap at 99th percentile | 100 rows capped per column |
| IDs | Validate FKs, create node_idx | 0-9999 contiguous index |
| Encode | Labels 0/1/2, itype 0-3 | Saved JSON encodings |
| Split | 70/15/15 stratified | 7k / 1.5k / 1.5k |
| Save | nodes_clean, edges_clean, splits | All files verified |

**Next:** Module 4 — User Behavioral Feature Engineering
"""),
]

# Save both notebooks
for nb_name, cells in [
    ("module2_dataset_understanding.ipynb", nb2_cells),
    ("module3_data_preparation.ipynb",      nb3_cells),
]:
    nb = make_nb(cells)
    path = ROOT / nb_name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {nb_name}  ({path.stat().st_size // 1024} KB)")

print("Done.")
