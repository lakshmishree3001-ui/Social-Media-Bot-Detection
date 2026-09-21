"""
verify_pipeline.py  --  Runs the complete Module 2 + Module 3 pipeline
without Jupyter so we can confirm correctness immediately.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style='darkgrid', palette='muted')

RAW_DIR       = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
SPLITS_DIR    = Path('data/splits')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: 'human', 1: 'bot', 2: 'suspicious'}

NODE_FEATURE_COLS = [
    'account_age_days', 'followers_count', 'following_count',
    'post_count', 'listed_count', 'verified',
    'has_profile_image', 'has_description', 'default_profile',
    'profile_completeness', 'posts_per_day',
    'reply_ratio', 'retweet_ratio', 'mention_ratio',
    'url_ratio', 'hashtag_ratio', 'duplicate_content_ratio',
]
IDENTITY_COLS = ['user_id', 'node_idx', 'screen_name', 'account_type', 'label', 'created_at', 'location']

print("\n" + "="*60)
print("  MODULE 2 + 3 PIPELINE VERIFICATION")
print("="*60)

# =============================================================================
# MODULE 2 -- Load & Understand
# =============================================================================
print("\n[MODULE 2] Loading raw data ...")
users_raw  = pd.read_csv(RAW_DIR / 'users.csv',  low_memory=False)
tweets_raw = pd.read_csv(RAW_DIR / 'tweets.csv', low_memory=False)
edges_raw  = pd.read_csv(RAW_DIR / 'edges.csv',  low_memory=False)
with open(RAW_DIR / 'dataset_info.json') as f:
    meta = json.load(f)

print(f"  users  : {len(users_raw):>7,} rows x {users_raw.shape[1]} cols")
print(f"  tweets : {len(tweets_raw):>7,} rows x {tweets_raw.shape[1]} cols")
print(f"  edges  : {len(edges_raw):>7,} rows x {edges_raw.shape[1]} cols")

# Label distribution
label_counts = users_raw['label'].value_counts().sort_index()
print("\n  Label distribution:")
for lbl, cnt in label_counts.items():
    pct = cnt / len(users_raw) * 100
    bar = '#' * int(pct / 2)
    print(f"    {LABEL_MAP[lbl]:<12} {cnt:,}  ({pct:.1f}%)  {bar}")

# Followers/Following ratio
users_raw['ff_ratio'] = (users_raw['followers_count'] + 1) / (users_raw['following_count'] + 1)
ff_median = users_raw.groupby('account_type')['ff_ratio'].median().round(3)
print("\n  Followers/Following ratio (median):")
for atype, val in ff_median.items():
    print(f"    {atype:<12} {val}")

# Edge type distribution
edge_counts = edges_raw['interaction_type'].value_counts()
print("\n  Edge type distribution:")
for etype, cnt in edge_counts.items():
    print(f"    {etype:<12} {cnt:,}")

# Save a quick chart
type_colors = {'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lc = users_raw['label'].map(LABEL_MAP).value_counts()
axes[0].bar(lc.index, lc.values, color=[type_colors[l] for l in lc.index], edgecolor='black')
axes[0].set_title('Label Distribution')
axes[0].set_ylabel('Count')
ec = edges_raw['interaction_type'].value_counts()
axes[1].bar(ec.index, ec.values,
            color=['#2196F3','#9C27B0','#009688','#FF5722'][:len(ec)], edgecolor='black')
axes[1].set_title('Edge Type Distribution')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(RAW_DIR / 'module2_overview.png', bbox_inches='tight', dpi=100)
plt.close()
print("\n  Chart saved: data/raw/module2_overview.png")

# =============================================================================
# MODULE 3 -- Data Cleaning & Preparation
# =============================================================================
print("\n" + "-"*60)
print("[MODULE 3] Cleaning & preparing data ...")

# 3a. Deduplication
users_clean  = users_raw.drop_duplicates(subset='user_id').copy()
tweets_clean = tweets_raw.drop_duplicates(subset='tweet_id').copy()
edges_clean  = edges_raw.drop_duplicates(
    subset=['source_user_id','target_user_id','interaction_type']).copy()
print(f"\n  After dedup  -> users={len(users_clean):,}  "
      f"tweets={len(tweets_clean):,}  edges={len(edges_clean):,}")

# 3b. Missing values
print("\n  Missing values (before fill):")
for col, cnt in users_clean.isnull().sum()[users_clean.isnull().sum() > 0].items():
    print(f"    {col:<30} {cnt:>5,}  ({cnt/len(users_clean)*100:.1f}%)")

users_clean['location'] = users_clean['location'].fillna('Unknown')
if 'listed_count' in users_clean.columns:
    median_by_type = users_clean.groupby('account_type')['listed_count'].transform('median')
    users_clean['listed_count'] = users_clean['listed_count'].fillna(median_by_type).fillna(0).astype(int)

remaining = users_clean.isnull().sum().sum()
print(f"\n  Missing values after fill: {remaining}")

# 3c. Data types
users_clean['created_at']     = pd.to_datetime(users_clean['created_at'])
tweets_clean['timestamp']     = pd.to_datetime(tweets_clean['timestamp'])
edges_clean['timestamp']      = pd.to_datetime(edges_clean['timestamp'])
for col in ['verified','has_profile_image','has_description','default_profile']:
    users_clean[col] = users_clean[col].astype(int)

# 3d. Outlier capping
caps = {}
cap_cols = ['followers_count','following_count','post_count','posts_per_day','listed_count']
print("\n  Outlier capping (99th pct):")
for col in cap_cols:
    p99 = users_clean[col].quantile(0.99)
    n_capped = (users_clean[col] > p99).sum()
    users_clean[col] = users_clean[col].clip(upper=p99)
    caps[col] = float(p99)
    print(f"    {col:<25}  cap={p99:>10.1f}  rows_capped={n_capped}")
with open(PROCESSED_DIR / 'outlier_caps.json', 'w') as f:
    json.dump(caps, f, indent=2)

# 3e. Consistent user IDs (node_idx)
uid_sorted = sorted(users_clean['user_id'].unique())
uid_to_idx = {uid: idx for idx, uid in enumerate(uid_sorted)}
users_clean['node_idx']   = users_clean['user_id'].map(uid_to_idx)
valid_ids                  = set(users_clean['user_id'].unique())
tweets_clean              = tweets_clean[tweets_clean['user_id'].isin(valid_ids)].copy()
edges_clean               = edges_clean[
    edges_clean['source_user_id'].isin(valid_ids) &
    edges_clean['target_user_id'].isin(valid_ids)
].copy()
edges_clean['src_idx'] = edges_clean['source_user_id'].map(uid_to_idx)
edges_clean['tgt_idx'] = edges_clean['target_user_id'].map(uid_to_idx)
with open(PROCESSED_DIR / 'uid_to_idx.json', 'w') as f:
    json.dump({str(k): v for k, v in uid_to_idx.items()}, f)
print(f"\n  node_idx range: 0...{users_clean['node_idx'].max()}")

# 3f. Label encoding
label_encoding = {
    'int_to_str': {0:'human',1:'bot',2:'suspicious'},
    'str_to_int': {'human':0,'bot':1,'suspicious':2},
}
with open(PROCESSED_DIR / 'label_encoding.json', 'w') as f:
    json.dump(label_encoding, f, indent=2)

itype_le = LabelEncoder()
edges_clean['interaction_type_enc'] = itype_le.fit_transform(edges_clean['interaction_type'])
itype_map = {k: int(v) for k, v in zip(itype_le.classes_, itype_le.transform(itype_le.classes_))}
with open(PROCESSED_DIR / 'interaction_encoding.json', 'w') as f:
    json.dump(itype_map, f, indent=2)
print(f"  Interaction encoding: {itype_map}")

# 3g. Separate node / edge tables
nodes_feature_df = users_clean[IDENTITY_COLS + NODE_FEATURE_COLS].copy()
EDGE_COLS = ['source_user_id','target_user_id','src_idx','tgt_idx',
             'interaction_type','interaction_type_enc','weight','timestamp']
edges_feature_df = edges_clean[EDGE_COLS].copy()

# Feature type metadata
feature_types = {
    'continuous': ['account_age_days','followers_count','following_count',
                   'post_count','listed_count','posts_per_day','profile_completeness'],
    'ratio_0_1':  ['reply_ratio','retweet_ratio','mention_ratio',
                   'url_ratio','hashtag_ratio','duplicate_content_ratio'],
    'binary':     ['verified','has_profile_image','has_description','default_profile'],
}
with open(PROCESSED_DIR / 'feature_types.json', 'w') as f:
    json.dump(feature_types, f, indent=2)

# 3h. Train / Val / Test split (70/15/15 stratified)
all_ids    = nodes_feature_df['user_id'].values
all_labels = nodes_feature_df['label'].values
train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    all_ids, all_labels, test_size=0.30, stratify=all_labels, random_state=42)
val_ids, test_ids, val_labels, test_labels = train_test_split(
    temp_ids, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42)
print(f"\n  Split sizes -> train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,}")

# 3i. Save processed files
nodes_feature_df.to_csv(PROCESSED_DIR / 'nodes_clean.csv',  index=False)
edges_feature_df.to_csv(PROCESSED_DIR / 'edges_clean.csv',  index=False)
tweets_clean.to_csv(PROCESSED_DIR     / 'tweets_clean.csv', index=False)
pd.DataFrame({'user_id':train_ids,'label':train_labels}).to_csv(SPLITS_DIR/'train_ids.csv', index=False)
pd.DataFrame({'user_id':val_ids,  'label':val_labels  }).to_csv(SPLITS_DIR/'val_ids.csv',   index=False)
pd.DataFrame({'user_id':test_ids, 'label':test_labels }).to_csv(SPLITS_DIR/'test_ids.csv',  index=False)

# 3j. Assertions
assert nodes_feature_df['user_id'].nunique() == len(nodes_feature_df)
assert nodes_feature_df[NODE_FEATURE_COLS].isnull().sum().sum() == 0
assert edges_feature_df[['src_idx','tgt_idx']].isnull().sum().sum() == 0
assert nodes_feature_df['label'].isin([0,1,2]).all()

print("\n" + "="*60)
print("  ALL CHECKS PASSED")
print("="*60)
print(f"\n  data/processed/")
for f in sorted(PROCESSED_DIR.iterdir()):
    print(f"    {f.name:<30}  {f.stat().st_size/1024:.1f} KB")
print(f"\n  data/splits/")
for f in sorted(SPLITS_DIR.iterdir()):
    print(f"    {f.name:<30}  {f.stat().st_size/1024:.1f} KB")
print("\n  Ready for Module 4 -- Behavioral Feature Engineering")
print("="*60 + "\n")
