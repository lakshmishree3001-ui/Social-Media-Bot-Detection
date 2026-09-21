"""
backend/services/pipeline/validator.py

Stage 0: Validate uploaded CSV/TSV, enforce minimum required columns,
apply column mapping, and return clean DataFrame + list of rejected rows.

Required canonical columns (after mapping):
  screen_name, followers_count, following_count, post_count, account_age_days

All other columns are optional and will be filled with defaults if absent.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Tuple

import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "screen_name",
    "followers_count",
    "following_count",
    "post_count",
    "account_age_days",
]

OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "user_id_str": None,          # generated from row index if missing
    "name": "",
    "description": "",
    "location": "",
    "created_at": None,
    "listed_count": 0,
    "verified": False,
    "has_profile_image": True,
    "has_description": True,
    "default_profile": False,
    "profile_completeness": 0.5,
    "ground_truth": "unknown",
}

SYNONYMS: Dict[str, List[str]] = {
    "screen_name":      ["username", "user_name", "handle", "login", "screenname"],
    "followers_count":  ["followers", "follower_count", "num_followers"],
    "following_count":  ["following", "friends_count", "friends", "num_following"],
    "post_count":       ["tweets", "tweet_count", "statuses_count", "posts", "status_count"],
    "account_age_days": ["age_days", "account_age", "age"],
    "user_id_str":      ["user_id", "id", "uid"],
    "description":      ["bio", "profile_description", "about"],
    "listed_count":     ["lists", "list_count"],
    "ground_truth":     ["label", "target", "class", "is_bot", "bot_label"],
}


def detect_delimiter(raw: bytes) -> str:
    """Heuristic: count commas vs tabs in first 4 kB."""
    sample = raw[:4096].decode("utf-8", errors="replace")
    return "\t" if sample.count("\t") > sample.count(",") else ","


def validate_and_map(
    file_bytes: bytes,
    column_map: Dict[str, str] | None = None,
    original_filename: str = "upload.csv",
) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
    """
    Returns
    -------
    df_clean     : DataFrame with canonical column names, all required cols present.
    rejected     : list of {row_index, row_preview, reason} dicts.
    warnings     : list of non-fatal warning strings.
    """
    sep = detect_delimiter(file_bytes)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, dtype=str, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not parse file '{original_filename}': {exc}") from exc

    # Apply caller-supplied column renaming
    if column_map:
        df = df.rename(columns=column_map)

    # Auto-map known aliases for any canonical columns not yet present
    lower_to_col = {c.lower(): c for c in df.columns}
    for canonical, aliases in SYNONYMS.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in lower_to_col:
                    orig = lower_to_col[alias]
                    df = df.rename(columns={orig: canonical})
                    break

    # Identify missing required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Upload is missing required columns after mapping: {missing}. "
            f"Detected columns: {list(df.columns)}"
        )

    # Fill optional missing columns with defaults
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Generate user_id_str from index when missing
    df["user_id_str"] = df["user_id_str"].fillna(
        df.index.to_series().apply(lambda i: f"upload_{i:08d}")
    )

    # Coerce numeric types, track rows that fail
    numeric_cols = {
        "followers_count": 0,
        "following_count": 0,
        "post_count": 0,
        "account_age_days": 0,
        "listed_count": 0,
        "profile_completeness": 0.5,
    }

    rejected: List[Dict] = []
    bad_indices: List[int] = []

    for col, fallback in numeric_cols.items():
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            bad_mask = numeric_series.isna() & df[col].notna()
            for idx in df.index[bad_mask]:
                rejected.append({
                    "row_index": int(idx),
                    "row_preview": df.loc[idx, "screen_name"] if "screen_name" in df.columns else str(idx),
                    "reason": f"Non-numeric value in column '{col}': {df.loc[idx, col]!r}",
                })
                bad_indices.append(idx)
            df[col] = numeric_series.fillna(fallback)

    # Drop rows with blank screen_name
    blank_mask = df["screen_name"].isna() | (df["screen_name"].astype(str).str.strip() == "")
    for idx in df.index[blank_mask]:
        if idx not in bad_indices:
            rejected.append({
                "row_index": int(idx),
                "row_preview": str(idx),
                "reason": "Blank screen_name",
            })
            bad_indices.append(idx)

    bad_set = set(bad_indices)
    df_clean = df[~df.index.isin(bad_set)].copy().reset_index(drop=True)

    # Normalize boolean columns
    for bcol in ("verified", "has_profile_image", "has_description", "default_profile"):
        if bcol in df_clean.columns:
            df_clean[bcol] = (
                df_clean[bcol]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "t"])
            )

    # Normalize ground_truth labels (e.g. 1/0, "bot"/"human", "suspicious")
    if "ground_truth" in df_clean.columns:
        def _normalize_gt(val):
            if pd.isna(val) or val is None:
                return "unknown"
            s = str(val).strip().lower()
            if s in ("1", "1.0", "bot", "b", "true", "yes", "t"):
                return "bot"
            if s in ("0", "0.0", "human", "h", "false", "no", "f", "legit", "authentic"):
                return "human"
            if s in ("2", "2.0", "suspicious", "suspect", "susp"):
                return "suspicious"
            if s in ("bot", "human", "suspicious", "unknown"):
                return s
            return "unknown"

        df_clean["ground_truth"] = df_clean["ground_truth"].apply(_normalize_gt)

    warnings: List[str] = []
    if len(rejected) > 0:
        warnings.append(f"{len(rejected)} rows rejected during validation and will not be ingested.")

    return df_clean, rejected, warnings


def infer_column_map_suggestions(file_bytes: bytes) -> Dict[str, str]:
    """
    Return a suggested {raw_column -> canonical_column} mapping for the
    column-mapping UI. Only returns entries where a non-trivial mapping
    is detected; already-canonical columns are excluded.
    """
    sep = detect_delimiter(file_bytes)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, nrows=0)
    except Exception:
        return {}

    raw_cols = list(df.columns)
    suggestions: Dict[str, str] = {}
    for canonical, aliases in SYNONYMS.items():
        for raw in raw_cols:
            if raw.lower() in aliases and raw != canonical:
                suggestions[raw] = canonical
                break

    return suggestions

