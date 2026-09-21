"""
=============================================================================
data_loader.py
Social Media Bot Detection — Module 3
Reusable Data Loading & Preprocessing Utilities

Provides:
  - load_raw()         → raw DataFrames
  - load_processed()   → cleaned DataFrames
  - build_graph()      → NetworkX DiGraph
  - get_stats()        → quick summary dict
=============================================================================
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict, Optional

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR    = BASE_DIR / "data" / "splits"

LABEL_MAP = {0: "human", 1: "bot", 2: "suspicious"}
COLOR_MAP = {0: "#4CAF50", 1: "#F44336", 2: "#FF9800"}   # green / red / orange


# =============================================================================
# 1. Raw data loaders
# =============================================================================

def load_raw(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV files from data/raw/.

    Returns
    -------
    users_df, tweets_df, edges_df
    """
    users_df  = pd.read_csv(RAW_DIR / "users.csv",  low_memory=False)
    tweets_df = pd.read_csv(RAW_DIR / "tweets.csv", low_memory=False)
    edges_df  = pd.read_csv(RAW_DIR / "edges.csv",  low_memory=False)

    if verbose:
        print(f"[load_raw] users={len(users_df):,}  tweets={len(tweets_df):,}  edges={len(edges_df):,}")

    return users_df, tweets_df, edges_df


def load_metadata() -> dict:
    """Load dataset_info.json metadata."""
    with open(RAW_DIR / "dataset_info.json") as f:
        return json.load(f)


def load_processed(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned/processed CSV files from data/processed/."""
    nodes_df  = pd.read_csv(PROCESSED_DIR / "nodes_clean.csv",  low_memory=False)
    edges_df  = pd.read_csv(PROCESSED_DIR / "edges_clean.csv",  low_memory=False)
    tweets_df = pd.read_csv(PROCESSED_DIR / "tweets_clean.csv", low_memory=False)

    if verbose:
        print(f"[load_processed] nodes={len(nodes_df):,}  "
              f"edges={len(edges_df):,}  tweets={len(tweets_df):,}")
    return nodes_df, edges_df, tweets_df


def load_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test user_id arrays."""
    train = pd.read_csv(SPLITS_DIR / "train_ids.csv")["user_id"].values
    val   = pd.read_csv(SPLITS_DIR / "val_ids.csv")["user_id"].values
    test  = pd.read_csv(SPLITS_DIR / "test_ids.csv")["user_id"].values
    return train, val, test


# =============================================================================
# 2. Graph construction
# =============================================================================

def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    interaction_filter: Optional[list] = None,
    add_node_attrs: bool = True,
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from processed data.

    Parameters
    ----------
    nodes_df           : cleaned node DataFrame
    edges_df           : cleaned edge DataFrame
    interaction_filter : list of interaction types to include (None = all)
    add_node_attrs     : whether to attach node attributes (label, etc.)

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # Add nodes
    for _, row in nodes_df.iterrows():
        attrs = {"label": int(row["label"]), "account_type": row["account_type"]}
        if add_node_attrs:
            for col in ["followers_count", "following_count", "post_count",
                        "posts_per_day", "account_age_days", "verified",
                        "has_profile_image", "has_description", "default_profile"]:
                if col in row.index:
                    attrs[col] = row[col]
        G.add_node(int(row["user_id"]), **attrs)

    # Filter edges
    df = edges_df.copy()
    if interaction_filter:
        df = df[df["interaction_type"].isin(interaction_filter)]

    # Add edges
    for _, row in df.iterrows():
        G.add_edge(
            int(row["source_user_id"]),
            int(row["target_user_id"]),
            interaction_type=row["interaction_type"],
            weight=float(row.get("weight", 1.0)),
        )

    return G


# =============================================================================
# 3. Quick statistics
# =============================================================================

def get_stats(users_df: pd.DataFrame,
              tweets_df: pd.DataFrame,
              edges_df: pd.DataFrame) -> Dict:
    """Return a concise statistics dictionary."""
    label_counts = users_df["label"].value_counts().to_dict()
    edge_counts  = edges_df["interaction_type"].value_counts().to_dict()
    missing_users  = users_df.isnull().sum()
    missing_tweets = tweets_df.isnull().sum()

    return {
        "users": {
            "total":      len(users_df),
            "human":      label_counts.get(0, 0),
            "bot":        label_counts.get(1, 0),
            "suspicious": label_counts.get(2, 0),
            "missing_per_column": missing_users[missing_users > 0].to_dict(),
        },
        "tweets": {
            "total":             len(tweets_df),
            "unique_users":      tweets_df["user_id"].nunique(),
            "avg_per_user":      round(len(tweets_df) / max(1, tweets_df["user_id"].nunique()), 2),
            "missing_per_column": missing_tweets[missing_tweets > 0].to_dict(),
        },
        "edges": {
            "total":       len(edges_df),
            "unique_src":  edges_df["source_user_id"].nunique(),
            "unique_tgt":  edges_df["target_user_id"].nunique(),
            "by_type":     edge_counts,
        },
    }


# =============================================================================
# 4. Label encoding helper
# =============================================================================

def encode_labels(series: pd.Series) -> Tuple[pd.Series, dict]:
    """
    Encode string labels to integers and return mapping.

    Returns (encoded_series, label_map)
    """
    label_map = {v: k for k, v in LABEL_MAP.items()}
    encoded = series.map(label_map)
    return encoded, label_map
