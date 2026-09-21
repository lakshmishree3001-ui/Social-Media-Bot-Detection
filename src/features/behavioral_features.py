"""
src/features/behavioral_features.py

Module 4 — User Behavioral Feature Engineering
===============================================
Derives richer features from nodes_clean.csv and tweets_clean.csv
to improve GNN classification quality.

New features computed:
  Activity patterns:
    - account_age_months
    - posts_per_day (already in data; we recompute for consistency)
    - tweet_hour_entropy       (uniformity of posting hours)
    - tweet_day_entropy        (uniformity of posting days of week)
    - active_hours_count       (distinct hours active)

  Content-based:
    - avg_tweet_length
    - avg_hashtags_per_tweet
    - avg_urls_per_tweet
    - avg_mentions_per_tweet
    - avg_retweet_count
    - avg_favorite_count
    - tweet_similarity_score   (mean pairwise Jaccard within top-20 tweets per user)

  Network-derived:
    - in_degree                (how many accounts interact WITH this user)
    - out_degree               (how many accounts this user interacts with)
    - degree_ratio             (in / (in + out + 1))
    - weighted_in_degree
    - weighted_out_degree

  Derived ratios:
    - followers_following_ratio  (followers+1 / following+1)
    - engagement_score           (avg_retweet + avg_favorite, log-scaled)
    - activity_consistency       (coefficient of variation of daily tweets, inverted)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. TWEET-AGGREGATED FEATURES
# =============================================================================

def compute_tweet_features(tweets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-tweet records into per-user statistics.

    Parameters
    ----------
    tweets_df : pd.DataFrame
        Raw or cleaned tweets with columns:
        user_id, timestamp, text, hashtag_count, url_count,
        mention_count, retweet_count, favorite_count, is_retweet

    Returns
    -------
    pd.DataFrame  indexed by user_id with new behavioral columns.
    """
    tweets_df = tweets_df.copy()
    tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"], errors="coerce")
    tweets_df = tweets_df.dropna(subset=["timestamp"])

    tweets_df["tweet_hour"] = tweets_df["timestamp"].dt.hour
    tweets_df["tweet_dow"] = tweets_df["timestamp"].dt.dayofweek  # 0=Mon
    tweets_df["tweet_length"] = tweets_df["text"].astype(str).str.len()

    def hour_entropy(s):
        counts = np.bincount(s, minlength=24).astype(float)
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        return float(scipy_entropy(probs + 1e-9, base=2))

    def dow_entropy(s):
        counts = np.bincount(s, minlength=7).astype(float)
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        return float(scipy_entropy(probs + 1e-9, base=2))

    def jaccard_similarity(texts, n_sample=20):
        """Mean Jaccard similarity of word-set pairs from up to n_sample tweets."""
        if len(texts) < 2:
            return 0.0
        sample = texts[:n_sample]
        sets = [set(str(t).lower().split()) for t in sample]
        scores = []
        for i in range(len(sets) - 1):
            for j in range(i + 1, min(i + 4, len(sets))):  # only adjacent pairs
                u = sets[i] | sets[j]
                if len(u) == 0:
                    continue
                scores.append(len(sets[i] & sets[j]) / len(u))
        return float(np.mean(scores)) if scores else 0.0

    # Column name normalisation — handle both raw and processed tweet schemas
    col_map = {
        "like_count":    "favorite_count",
        "has_hashtag":   "hashtag_count",
        "has_url":       "url_count",
        "reply_count":   "mention_count",  # approx — reply_count as proxy for mention activity
    }
    for old, new in col_map.items():
        if old in tweets_df.columns and new not in tweets_df.columns:
            tweets_df[new] = tweets_df[old]

    # Ensure all expected columns exist
    for col in ["hashtag_count", "url_count", "mention_count", "retweet_count", "favorite_count"]:
        if col not in tweets_df.columns:
            tweets_df[col] = 0

    agg = tweets_df.groupby("user_id").agg(
        tweet_count=("tweet_length", "count"),
        avg_tweet_length=("tweet_length", "mean"),
        avg_hashtags_per_tweet=("hashtag_count", "mean"),
        avg_urls_per_tweet=("url_count", "mean"),
        avg_mentions_per_tweet=("mention_count", "mean"),
        avg_retweet_count=("retweet_count", "mean"),
        avg_favorite_count=("favorite_count", "mean"),
        tweet_hour_entropy=("tweet_hour", hour_entropy),
        tweet_day_entropy=("tweet_dow", dow_entropy),
        active_hours_count=("tweet_hour", "nunique"),
    ).reset_index()

    # Jaccard similarity — run per user over text column
    jac = (
        tweets_df.groupby("user_id")["text"]
        .apply(lambda x: jaccard_similarity(x.tolist()))
        .reset_index(name="tweet_similarity_score")
    )
    agg = agg.merge(jac, on="user_id", how="left")
    agg["tweet_similarity_score"] = agg["tweet_similarity_score"].fillna(0.0)

    return agg


# =============================================================================
# 2. GRAPH DEGREE FEATURES
# =============================================================================

def compute_graph_features(edges_df: pd.DataFrame, all_user_ids: list) -> pd.DataFrame:
    """
    Computes in/out degree and weighted versions from the interaction edge list.

    Parameters
    ----------
    edges_df : pd.DataFrame   with columns: source_user_id, target_user_id, weight
    all_user_ids : list       ensures all users appear even if isolated

    Returns
    -------
    pd.DataFrame indexed by user_id with degree features.
    """
    out_deg = edges_df.groupby("source_user_id").agg(
        out_degree=("target_user_id", "count"),
        weighted_out_degree=("weight", "sum"),
    ).reset_index().rename(columns={"source_user_id": "user_id"})

    in_deg = edges_df.groupby("target_user_id").agg(
        in_degree=("source_user_id", "count"),
        weighted_in_degree=("weight", "sum"),
    ).reset_index().rename(columns={"target_user_id": "user_id"})

    base = pd.DataFrame({"user_id": all_user_ids})
    df = base.merge(out_deg, on="user_id", how="left").merge(in_deg, on="user_id", how="left")
    df = df.fillna(0)
    df["out_degree"] = df["out_degree"].astype(int)
    df["in_degree"] = df["in_degree"].astype(int)

    # Degree ratio: how much of total interactions are INCOMING?
    df["degree_ratio"] = df["in_degree"] / (df["in_degree"] + df["out_degree"] + 1)

    return df


# =============================================================================
# 3. MERGE & NORMALISE
# =============================================================================

def engineer_all_features(
    nodes_df: pd.DataFrame,
    tweets_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: compute tweet + graph features, merge with node table,
    add derived ratio features, normalise, return final matrix.

    Parameters
    ----------
    nodes_df  : cleaned node table (nodes_clean.csv)
    tweets_df : cleaned tweets (tweets_clean.csv)
    edges_df  : cleaned edges  (edges_clean.csv)
    save      : if True, writes results to data/features/

    Returns
    -------
    pd.DataFrame with all engineered features (one row per user).
    """
    print("[1/5] Computing tweet-level aggregated features ...")
    tweet_feats = compute_tweet_features(tweets_df)

    print("[2/5] Computing graph degree features ...")
    graph_feats = compute_graph_features(edges_df, nodes_df["user_id"].tolist())

    print("[3/5] Merging with node profile features ...")
    df = nodes_df.merge(tweet_feats, on="user_id", how="left")
    df = df.merge(graph_feats,       on="user_id", how="left")

    # Fill users with no tweets
    tweet_fill_cols = [
        "tweet_count", "avg_tweet_length", "avg_hashtags_per_tweet",
        "avg_urls_per_tweet", "avg_mentions_per_tweet",
        "avg_retweet_count", "avg_favorite_count",
        "tweet_hour_entropy", "tweet_day_entropy", "active_hours_count",
        "tweet_similarity_score",
    ]
    df[tweet_fill_cols] = df[tweet_fill_cols].fillna(0)
    df[["in_degree", "out_degree", "weighted_in_degree",
        "weighted_out_degree", "degree_ratio"]] = df[[
        "in_degree", "out_degree", "weighted_in_degree",
        "weighted_out_degree", "degree_ratio",
    ]].fillna(0)

    print("[4/5] Computing derived ratio features ...")
    df["followers_following_ratio"] = (
        (df["followers_count"] + 1) / (df["following_count"] + 1)
    )
    df["engagement_score"] = np.log1p(
        df["avg_retweet_count"] + df["avg_favorite_count"]
    )

    # Activity consistency: low CoV = consistent (bot-like); high CoV = sporadic (human-like)
    # We approximate using posting entropy as inverse measure
    max_h_ent = np.log2(24)
    df["activity_consistency"] = 1.0 - (df["tweet_hour_entropy"] / max_h_ent).clip(0, 1)

    print("[5/5] Normalising continuous features (StandardScaler) ...")
    continuous = [
        "account_age_days", "followers_count", "following_count", "post_count",
        "listed_count", "posts_per_day", "avg_tweet_length",
        "avg_hashtags_per_tweet", "avg_urls_per_tweet", "avg_mentions_per_tweet",
        "avg_retweet_count", "avg_favorite_count",
        "in_degree", "out_degree", "weighted_in_degree", "weighted_out_degree",
        "followers_following_ratio", "engagement_score",
    ]
    # Only normalise columns that actually exist in df
    continuous = [c for c in continuous if c in df.columns]
    scaler = StandardScaler()
    df[continuous] = scaler.fit_transform(df[continuous].astype(float))

    # Save scaler parameters
    scaler_params = {
        col: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
        for i, col in enumerate(continuous)
    }

    if save:
        df.to_csv(FEATURES_DIR / "node_features.csv", index=False)
        with open(FEATURES_DIR / "scaler_params.json", "w") as f:
            json.dump(scaler_params, f, indent=2)
        with open(FEATURES_DIR / "feature_columns.json", "w") as f:
            feature_cols = (
                continuous
                + [
                    "profile_completeness", "tweet_hour_entropy", "tweet_day_entropy",
                    "active_hours_count", "tweet_similarity_score",
                    "degree_ratio", "activity_consistency",
                    "reply_ratio", "retweet_ratio", "mention_ratio",
                    "url_ratio", "hashtag_ratio", "duplicate_content_ratio",
                    "verified", "has_profile_image", "has_description", "default_profile",
                ]
            )
            feature_cols = [c for c in feature_cols if c in df.columns]
            feature_cols = list(dict.fromkeys(feature_cols))  # dedupe, preserve order
            json.dump({"feature_columns": feature_cols, "n_features": len(feature_cols)}, f, indent=2)
        print(f"\nSaved to data/features/")
        print(f"  node_features.csv   {(FEATURES_DIR / 'node_features.csv').stat().st_size // 1024} KB")
        print(f"  scaler_params.json")
        print(f"  feature_columns.json  ({len(feature_cols)} features)")

    return df, scaler_params


# =============================================================================
# CLI  —  run directly: python -m src.features.behavioral_features
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 4 — Behavioral Feature Engineering")
    print("=" * 60)

    nodes_df  = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv",  low_memory=False)
    tweets_df = pd.read_csv(PROCESSED_DIR / "tweets_clean.csv", low_memory=False)
    edges_df  = pd.read_csv(PROCESSED_DIR / "edges_clean.csv",  low_memory=False)

    print(f"\nLoaded: {len(nodes_df):,} nodes | {len(tweets_df):,} tweets | {len(edges_df):,} edges\n")

    result_df, _ = engineer_all_features(nodes_df, tweets_df, edges_df, save=True)

    print(f"\nFinal feature matrix: {result_df.shape}")
    print(f"Labels: {dict(result_df['label'].value_counts().sort_index())}")
    print("\nSample features (first row):")
    NEW_FEATURE_COLS = [
        "avg_tweet_length", "tweet_hour_entropy", "tweet_day_entropy",
        "active_hours_count", "tweet_similarity_score",
        "in_degree", "out_degree", "degree_ratio",
        "followers_following_ratio", "engagement_score", "activity_consistency",
    ]
    sample_row = result_df[NEW_FEATURE_COLS].head(1)
    for col in NEW_FEATURE_COLS:
        val = float(sample_row[col].iloc[0])
        print(f"  {col:<35} {val:>8.4f}")

    print("\n" + "=" * 60)
    print("  Module 4 COMPLETE — Ready for Module 5 (NLP Features)")
    print("=" * 60)
