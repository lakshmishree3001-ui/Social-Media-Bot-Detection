"""
=============================================================================
generate_dataset.py
Social Media Bot Detection -- Module 3
Realistic Synthetic Dataset Generator

Generates a research-grade dataset that mirrors the structure of publicly
available bot-detection datasets (Cresci-2017, TwiBot-20/22 schema).

Dataset statistics:
  - Users   : 10,000  (human / bot / suspicious)
  - Tweets  : ~100,000 (linked to users)
  - Edges   : ~55,000 (follows / retweet / reply / mention)

All behavioral statistics are grounded in published bot-detection research:
  Cresci et al. 2017, Varol et al. 2017, Yang et al. 2020 (TwiBot-20)
=============================================================================
"""

import os
import re
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# -- Reproducibility ----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -- Paths --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Configuration -- realistic behavioral profiles per account type
# =============================================================================

ACCOUNT_PROFILES = {
    # label=0  Human accounts
    "human": {
        "label": 0,
        "label_str": "human",
        "count": 5000,
        "account_age":      {"dist": "normal",    "mean": 1800,  "std": 600,  "min": 90,  "max": 4000},
        "followers":        {"dist": "lognormal", "mean": 5.8,   "std": 1.5,  "min": 5,   "max": 50000},
        "following":        {"dist": "lognormal", "mean": 5.5,   "std": 1.2,  "min": 3,   "max": 5000},
        "post_count":       {"dist": "lognormal", "mean": 6.5,   "std": 1.4,  "min": 10,  "max": 50000},
        "verified_prob":    0.02,
        "posts_per_day":    {"dist": "normal",    "mean": 2.5,   "std": 1.8,  "min": 0.1, "max": 25},
        "reply_ratio":      {"dist": "beta",      "a": 2.0,      "b": 5.0},
        "retweet_ratio":    {"dist": "beta",      "a": 1.5,      "b": 4.0},
        "mention_ratio":    {"dist": "beta",      "a": 1.0,      "b": 6.0},
        "url_ratio":        {"dist": "beta",      "a": 1.2,      "b": 3.5},
        "hashtag_ratio":    {"dist": "beta",      "a": 1.0,      "b": 4.0},
        "duplicate_ratio":  {"dist": "beta",      "a": 0.5,      "b": 8.0},
        "profile_complete": {"dist": "beta",      "a": 5.0,      "b": 2.0},
        "has_profile_img_prob":  0.87,
        "has_description_prob":  0.72,
        "default_profile_prob":  0.15,
    },

    # label=1  Bot accounts
    "bot": {
        "label": 1,
        "label_str": "bot",
        "count": 3500,
        "account_age":      {"dist": "normal",    "mean": 350,   "std": 250,  "min": 1,   "max": 1800},
        "followers":        {"dist": "lognormal", "mean": 3.5,   "std": 1.8,  "min": 0,   "max": 5000},
        "following":        {"dist": "lognormal", "mean": 6.8,   "std": 1.5,  "min": 5,   "max": 30000},
        "post_count":       {"dist": "lognormal", "mean": 7.5,   "std": 1.6,  "min": 5,   "max": 200000},
        "verified_prob":    0.001,
        "posts_per_day":    {"dist": "normal",    "mean": 48.0,  "std": 30.0, "min": 5.0, "max": 300},
        "reply_ratio":      {"dist": "beta",      "a": 0.8,      "b": 8.0},
        "retweet_ratio":    {"dist": "beta",      "a": 4.0,      "b": 2.0},
        "mention_ratio":    {"dist": "beta",      "a": 0.5,      "b": 9.0},
        "url_ratio":        {"dist": "beta",      "a": 4.0,      "b": 2.0},
        "hashtag_ratio":    {"dist": "beta",      "a": 3.5,      "b": 2.0},
        "duplicate_ratio":  {"dist": "beta",      "a": 6.0,      "b": 2.0},
        "profile_complete": {"dist": "beta",      "a": 1.5,      "b": 5.0},
        "has_profile_img_prob":  0.28,
        "has_description_prob":  0.22,
        "default_profile_prob":  0.70,
    },

    # label=2  Suspicious / Coordinated accounts
    "suspicious": {
        "label": 2,
        "label_str": "suspicious",
        "count": 1500,
        "account_age":      {"dist": "normal",    "mean": 800,   "std": 400,  "min": 30,  "max": 2500},
        "followers":        {"dist": "lognormal", "mean": 4.5,   "std": 1.6,  "min": 2,   "max": 15000},
        "following":        {"dist": "lognormal", "mean": 6.2,   "std": 1.4,  "min": 5,   "max": 20000},
        "post_count":       {"dist": "lognormal", "mean": 7.0,   "std": 1.5,  "min": 5,   "max": 100000},
        "verified_prob":    0.005,
        "posts_per_day":    {"dist": "normal",    "mean": 18.0,  "std": 12.0, "min": 2.0, "max": 150},
        "reply_ratio":      {"dist": "beta",      "a": 1.2,      "b": 6.0},
        "retweet_ratio":    {"dist": "beta",      "a": 3.0,      "b": 2.5},
        "mention_ratio":    {"dist": "beta",      "a": 1.0,      "b": 7.0},
        "url_ratio":        {"dist": "beta",      "a": 2.5,      "b": 2.5},
        "hashtag_ratio":    {"dist": "beta",      "a": 2.8,      "b": 2.2},
        "duplicate_ratio":  {"dist": "beta",      "a": 3.5,      "b": 3.0},
        "profile_complete": {"dist": "beta",      "a": 2.5,      "b": 3.5},
        "has_profile_img_prob":  0.55,
        "has_description_prob":  0.45,
        "default_profile_prob":  0.45,
    },
}

# -- Tweet templates ----------------------------------------------------------
HUMAN_TWEET_TEMPLATES = [
    "Just had the best coffee - starting my day right!",
    "Can't believe what happened today at work. Monday, am I right?",
    "Reading '{book}' right now. Highly recommend it to anyone into {topic}.",
    "Happy birthday to my best friend! You deserve the world.",
    "The sunset today was absolutely stunning #nature #photography",
    "Finally finished that project I've been working on for weeks. So relieved!",
    "Anyone else think {topic} is getting out of hand lately?",
    "Just got back from a run. 5km done! Feeling great.",
    "Movie night with the family -- watching {movie} for the 3rd time haha",
    "My thoughts on the latest {topic} news: really complicated situation.",
    "Cooking dinner -- trying that new {food} recipe I found last week.",
    "Weekend trip to {place} was amazing! Would definitely go back.",
    "Shoutout to everyone grinding hard this week. You've got this!",
    "The new album by {artist} is everything I needed today.",
    "Local elections tomorrow -- don't forget to vote! Your voice matters.",
    "Good morning everyone! Hope you all have a productive and positive day.",
    "Just rescued a puppy -- meet my new best friend!",
    "Starting a new habit: journaling every morning. Day 1 done.",
    "Hot take: {topic} is actually really important and we don't talk about it enough.",
    "Feeling grateful today for small things. Life is beautiful.",
    "That meeting could have been an email...",
    "Throwback to summer 2022 -- miss those days so much.",
    "Just donated to {charity} -- if you can, please do too. Every little helps.",
    "Finally trying meditation. Five minutes in and my mind is already racing.",
    "Question for my followers: what's your go-to productivity hack?",
]

BOT_TWEET_TEMPLATES = [
    "AMAZING DEAL! Get {product} for FREE! Click here: {url} #ad #free #win",
    "Follow me and I'll follow back! 100% guaranteed! #followback #follow #fff",
    "RT to win a FREE iPhone 15! Must follow! Limited time! {url}",
    "Check out this incredible opportunity! {url} #money #crypto #investment",
    "BREAKING: {topic} is trending NOW! See the full story: {url}",
    "I just earned $500 today working from home! You can too! {url}",
    "Like and retweet for a chance to WIN $1000! Follow us first! #giveaway",
    "The truth about {topic} they don't want you to know: {url} #truth #exposed",
    "Buy followers cheap and fast! Best prices guaranteed! {url} #followers",
    "New video just dropped! Don't miss it! {url} #subscribe #youtube",
    "VOTE for {candidate}! Share this with everyone you know! #vote #election",
    "Cryptocurrency tip: {crypto} will moon soon! Invest now! {url}",
    "FREE {product} -- just complete this survey: {url} #free #win #giveaway",
    "{crypto} to the moon! Buy before it's too late! {url}",
    "Get more followers instantly! Click here: {url} #socialmedia #growth",
    "Amazing skin care results in 7 days! {url} #beauty #skincare #results",
    "Join our network and earn passive income! {url} #mlm #business #success",
    "100 people are making money with this secret: {url} #wealth #rich",
    "RT if you agree: {political_claim}! #politics #truth",
    "SHARE THIS NOW before it gets deleted! {url} #censored #truth #banned",
    "Hot stock tip: {ticker} is about to explode! {url} #stocks #investing",
    "Sign up now and get {product} completely FREE! Limited offer: {url}",
    "The mainstream media won't report this: {url} #exposing #truth",
    "Double your investment in 30 days! Guaranteed! {url} #crypto #money",
    "Like this tweet if you support {cause}! RT to spread the word!",
]

SUSPICIOUS_TWEET_TEMPLATES = [
    "Interesting perspective on {topic}. Worth considering: {url}",
    "New research shows {political_claim}. Full details: {url} #news",
    "People are waking up to the truth about {topic}. Share widely.",
    "This deserves more attention: {url} #trending #important",
    "Just shared this with everyone I know. You should too: {url}",
    "The {party} agenda is clear. We must act now. #politics #action",
    "Something big is happening with {topic} and nobody is talking about it.",
    "Support our movement! Every RT counts! {url} #together #fight",
    "Did you see what {politician} just said about {topic}? Outrageous! {url}",
    "Our community is growing! Join us: {url} #community #movement",
    "Important message about {topic} -- please share widely: {url}",
    "They're trying to silence voices like ours. Share this: {url}",
    "{politician} exposed! The truth finally comes out: {url} #exposed",
    "Join thousands already making a difference. Sign the petition: {url}",
    "Major development in {topic}! Sources confirm: {url} #breaking",
]

FILL_VARS = {
    "book":           ["Atomic Habits", "Dune", "The Great Gatsby", "Sapiens", "1984"],
    "topic":          ["AI", "climate change", "remote work", "crypto", "social media",
                       "healthcare", "education", "privacy", "misinformation", "tech"],
    "movie":          ["Interstellar", "The Dark Knight", "Inception", "Avatar", "Oppenheimer"],
    "food":           ["pasta", "tacos", "sushi", "curry", "ramen", "pizza", "salad"],
    "place":          ["Paris", "Tokyo", "New York", "Bali", "London", "Barcelona"],
    "artist":         ["Taylor Swift", "Kendrick Lamar", "Dua Lipa", "The Weeknd", "Beyonce"],
    "charity":        ["Red Cross", "UNICEF", "Doctors Without Borders", "WWF"],
    "product":        ["iPhone", "AirPods", "MacBook", "gift card", "subscription"],
    "url":            ["http://bit.ly/xz9f2", "https://tinyurl.com/abc123",
                       "http://click.here/now", "https://earn.money/now"],
    "crypto":         ["Bitcoin", "Ethereum", "Dogecoin", "Solana", "XRP"],
    "ticker":         ["AAPL", "TSLA", "GME", "AMC", "NVDA"],
    "political_claim":["The election was stolen", "The government is hiding the truth",
                       "They control the media", "We are being censored"],
    "cause":          ["free speech", "animal rights", "climate action", "our veterans"],
    "politician":     ["Biden", "Trump", "Musk", "Pelosi", "AOC"],
    "party":          ["liberal", "conservative", "globalist", "elitist"],
    "candidate":      ["Candidate A", "Candidate B", "our leader", "the real winner"],
}

HASHTAG_POOLS = {
    "human":      ["#morning", "#life", "#grateful", "#weekend", "#coffee", "#fitness",
                   "#photography", "#travel", "#food", "#books", "#family", "#friends",
                   "#workout", "#sunset", "#reading", "#cooking", "#happy", "#motivation"],
    "bot":        ["#free", "#win", "#giveaway", "#crypto", "#bitcoin", "#followback",
                   "#follow", "#fff", "#money", "#income", "#investment", "#deal",
                   "#limitedtime", "#offer", "#subscribe", "#viral", "#trending"],
    "suspicious": ["#truth", "#exposed", "#wakeup", "#breaking", "#censored", "#news",
                   "#politics", "#freedom", "#fight", "#resist", "#movement", "#together",
                   "#share", "#important", "#urgent"],
}

LOCATIONS = [
    "New York, USA", "London, UK", "California, USA", "Toronto, Canada",
    "Mumbai, India", "Berlin, Germany", "Sydney, Australia", "Tokyo, Japan",
    "Paris, France", "Sao Paulo, Brazil", "Dubai, UAE", "Seoul, South Korea",
]

INTERACTION_TYPES = ["follows", "retweets", "replies", "mentions"]


# =============================================================================
# 2. Helper sampling functions
# =============================================================================

def _sample(cfg: dict, size: int = 1) -> np.ndarray:
    dist = cfg["dist"]
    if dist == "normal":
        vals = np.random.normal(cfg["mean"], cfg["std"], size)
    elif dist == "lognormal":
        vals = np.random.lognormal(cfg["mean"], cfg["std"], size)
    elif dist == "beta":
        vals = np.random.beta(cfg["a"], cfg["b"], size)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    lo = cfg.get("min", -np.inf)
    hi = cfg.get("max",  np.inf)
    return np.clip(vals, lo, hi)


def _fill_template(template: str) -> str:
    placeholders = re.findall(r'\{(\w+)\}', template)
    result = template
    for ph in placeholders:
        if ph in FILL_VARS:
            result = result.replace('{' + ph + '}', random.choice(FILL_VARS[ph]), 1)
    return result


def _make_tweet(account_type: str, user_id: int, timestamp: datetime) -> dict:
    if account_type == "human":
        templates = HUMAN_TWEET_TEMPLATES
        hashtags  = HASHTAG_POOLS["human"]
    elif account_type == "bot":
        templates = BOT_TWEET_TEMPLATES
        hashtags  = HASHTAG_POOLS["bot"]
    else:
        templates = SUSPICIOUS_TWEET_TEMPLATES
        hashtags  = HASHTAG_POOLS["suspicious"]

    text = _fill_template(random.choice(templates))
    n_tags = np.random.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10])
    added_tags = " ".join(random.sample(hashtags, min(n_tags, len(hashtags))))
    if added_tags:
        text = text + " " + added_tags

    has_url     = int("http://" in text or "https://" in text)
    has_hashtag = int("#" in text)

    return {
        "tweet_id":      int(1e12) + random.randint(0, int(1e11)),
        "user_id":       user_id,
        "text":          text,
        "timestamp":     timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "retweet_count": max(0, int(np.random.lognormal(0.5, 1.5))),
        "like_count":    max(0, int(np.random.lognormal(1.0, 2.0))),
        "reply_count":   max(0, int(np.random.lognormal(0.2, 1.2))),
        "has_url":       has_url,
        "has_hashtag":   has_hashtag,
        "is_retweet":    int(text.startswith("RT")),
        "account_type":  account_type,
    }


# =============================================================================
# 3. User generation
# =============================================================================

def generate_users() -> pd.DataFrame:
    print("  >> Generating user profiles ...")
    records = []
    uid = 100001
    ref_date = datetime(2024, 1, 1)

    for acct_type, prof in ACCOUNT_PROFILES.items():
        n = prof["count"]
        ages        = _sample(prof["account_age"],     n).astype(int)
        followers   = _sample(prof["followers"],       n).astype(int)
        following   = _sample(prof["following"],       n).astype(int)
        post_counts = _sample(prof["post_count"],      n).astype(int)
        ppd         = _sample(prof["posts_per_day"],   n)
        reply_r     = _sample(prof["reply_ratio"],     n)
        retweet_r   = _sample(prof["retweet_ratio"],   n)
        mention_r   = _sample(prof["mention_ratio"],   n)
        url_r       = _sample(prof["url_ratio"],       n)
        hashtag_r   = _sample(prof["hashtag_ratio"],   n)
        dup_r       = _sample(prof["duplicate_ratio"], n)
        prof_comp   = _sample(prof["profile_complete"],n)

        for i in range(n):
            created_at = ref_date - timedelta(days=int(ages[i]))
            verified   = bool(np.random.random() < prof["verified_prob"])
            has_img    = bool(np.random.random() < prof["has_profile_img_prob"])
            has_desc   = bool(np.random.random() < prof["has_description_prob"])
            def_prof   = bool(np.random.random() < prof["default_profile_prob"])
            loc        = None if np.random.random() < 0.30 else random.choice(LOCATIONS)

            records.append({
                "user_id":                  uid,
                "screen_name":              f"user_{uid}",
                "account_type":             acct_type,
                "label":                    prof["label"],
                "created_at":               created_at.strftime("%Y-%m-%d"),
                "account_age_days":         int(ages[i]),
                "followers_count":          int(followers[i]),
                "following_count":          int(following[i]),
                "post_count":               int(post_counts[i]),
                "verified":                 int(verified),
                "posts_per_day":            round(float(ppd[i]), 4),
                "reply_ratio":              round(float(reply_r[i]), 4),
                "retweet_ratio":            round(float(retweet_r[i]), 4),
                "mention_ratio":            round(float(mention_r[i]), 4),
                "url_ratio":                round(float(url_r[i]), 4),
                "hashtag_ratio":            round(float(hashtag_r[i]), 4),
                "duplicate_content_ratio":  round(float(dup_r[i]), 4),
                "has_profile_image":        int(has_img),
                "has_description":          int(has_desc),
                "default_profile":          int(def_prof),
                "profile_completeness":     round(float(prof_comp[i]), 4),
                "location":                 loc,
                "listed_count":             (
                    None if np.random.random() < 0.08
                    else max(0, int(np.random.lognormal(2.0, 1.5)))
                ),
            })
            uid += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(
        f"     OK  {len(df):,} users  |  "
        f"human={len(df[df.label==0]):,}  "
        f"bot={len(df[df.label==1]):,}  "
        f"suspicious={len(df[df.label==2]):,}"
    )
    return df


# =============================================================================
# 4. Tweet generation
# =============================================================================

def generate_tweets(users_df: pd.DataFrame, tweets_per_user_cap: int = 20) -> pd.DataFrame:
    print("  >> Generating tweet records ...")
    ref_date = datetime(2024, 1, 1)
    records  = []

    for _, row in users_df.iterrows():
        uid   = row["user_id"]
        atype = row["account_type"]
        age   = max(1, int(row["account_age_days"]))
        ppd   = max(0.1, float(row["posts_per_day"]))

        n_tweets = int(min(tweets_per_user_cap, max(1, ppd * 3 + np.random.randint(-2, 5))))

        if atype == "bot":
            burst_day  = random.randint(0, min(age, 30))
            base_ts    = ref_date - timedelta(days=burst_day)
            timestamps = [base_ts - timedelta(minutes=random.randint(0, 1440))
                          for _ in range(n_tweets)]
        elif atype == "suspicious":
            cluster_day = random.randint(0, min(age, 90))
            base_ts     = ref_date - timedelta(days=cluster_day)
            timestamps  = [base_ts - timedelta(hours=random.randint(0, 168))
                           for _ in range(n_tweets)]
        else:
            timestamps  = [ref_date - timedelta(days=random.randint(0, age))
                           for _ in range(n_tweets)]

        for ts in timestamps:
            records.append(_make_tweet(atype, uid, ts))

    df = (pd.DataFrame(records)
            .drop_duplicates(subset=["tweet_id"])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True))
    print(f"     OK  {len(df):,} tweets generated")
    return df


# =============================================================================
# 5. Edge / Interaction generation
# =============================================================================

def generate_edges(users_df: pd.DataFrame, target_edges: int = 55000) -> pd.DataFrame:
    print("  >> Generating interaction edges ...")

    user_ids       = users_df["user_id"].values
    labels         = users_df["label"].values
    human_ids      = user_ids[labels == 0]
    bot_ids        = user_ids[labels == 1]
    suspicious_ids = user_ids[labels == 2]

    records = []
    seen    = set()

    def add_edge(src, tgt, itype, weight=1.0):
        key = (src, tgt, itype)
        if src == tgt or key in seen:
            return False
        seen.add(key)
        ts = (datetime(2024, 1, 1) - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        records.append({
            "source_user_id":   src,
            "target_user_id":   tgt,
            "interaction_type": itype,
            "weight":           round(float(weight), 3),
            "timestamp":        ts,
        })
        return True

    edges_added = 0

    # Human <-> Human (organic)
    while edges_added < int(target_edges * 0.35):
        src   = random.choice(human_ids)
        tgt   = random.choice(human_ids)
        itype = random.choices(INTERACTION_TYPES, weights=[0.50, 0.20, 0.20, 0.10])[0]
        if add_edge(src, tgt, itype, weight=np.random.beta(2, 2)):
            edges_added += 1

    # Bot <-> Bot (dense cluster)
    bot_cluster = bot_ids[:500]
    while edges_added < int(target_edges * 0.55):
        src   = random.choice(bot_ids)
        tgt   = random.choice(bot_cluster if np.random.random() < 0.6 else user_ids)
        itype = random.choices(INTERACTION_TYPES, weights=[0.35, 0.45, 0.10, 0.10])[0]
        if add_edge(src, tgt, itype, weight=np.random.beta(5, 1.5)):
            edges_added += 1

    # Suspicious <-> Bot (coordination)
    while edges_added < int(target_edges * 0.70):
        src   = random.choice(suspicious_ids)
        tgt   = random.choice(bot_ids if np.random.random() < 0.55 else suspicious_ids)
        itype = random.choices(INTERACTION_TYPES, weights=[0.30, 0.40, 0.15, 0.15])[0]
        if add_edge(src, tgt, itype, weight=np.random.beta(3, 2)):
            edges_added += 1

    # Cross-type edges
    while edges_added < target_edges:
        src   = random.choice(user_ids)
        tgt   = random.choice(user_ids)
        itype = random.choices(INTERACTION_TYPES, weights=[0.40, 0.30, 0.20, 0.10])[0]
        if add_edge(src, tgt, itype, weight=np.random.beta(2, 3)):
            edges_added += 1

    df = (pd.DataFrame(records)
            .drop_duplicates(subset=["source_user_id", "target_user_id", "interaction_type"])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True))

    type_counts = "  ".join(
        f"{t}={len(df[df.interaction_type==t]):,}" for t in INTERACTION_TYPES
    )
    print(f"     OK  {len(df):,} edges  |  {type_counts}")
    return df


# =============================================================================
# 6. Dataset metadata
# =============================================================================

def generate_metadata(users_df, tweets_df, edges_df) -> dict:
    label_dist = users_df["label"].value_counts().to_dict()
    return {
        "dataset_name": "SocialBotDetection-Synthetic-v1",
        "description": (
            "Realistic synthetic dataset for social media bot detection. "
            "Schema mirrors Cresci-2017 and TwiBot-20/22 research datasets. "
            "Behavioral statistics grounded in published bot-detection literature."
        ),
        "references": [
            "Cresci et al. 2017 -- The Paradigm-Shift of Social Spambots",
            "Varol et al. 2017 -- Online Human-Bot Interactions",
            "Yang et al. 2020 -- TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark",
            "Feng et al. 2022 -- TwiBot-22: Towards Graph-Based Twitter Bot Detection",
        ],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "statistics": {
            "total_users":  int(len(users_df)),
            "total_tweets": int(len(tweets_df)),
            "total_edges":  int(len(edges_df)),
            "label_distribution": {
                "0_human":      int(label_dist.get(0, 0)),
                "1_bot":        int(label_dist.get(1, 0)),
                "2_suspicious": int(label_dist.get(2, 0)),
            },
            "edge_type_distribution": {
                t: int(len(edges_df[edges_df.interaction_type == t]))
                for t in INTERACTION_TYPES
            },
        },
        "schema": {
            "users.csv": {
                "user_id":                "Unique integer user identifier",
                "screen_name":            "Username string",
                "account_type":           "human | bot | suspicious",
                "label":                  "0=human  1=bot  2=suspicious",
                "created_at":             "Account creation date (YYYY-MM-DD)",
                "account_age_days":       "Days since account creation",
                "followers_count":        "Number of followers",
                "following_count":        "Number of accounts followed",
                "post_count":             "Total number of posts",
                "verified":               "1 if verified account",
                "posts_per_day":          "Average posts per day",
                "reply_ratio":            "Fraction of posts that are replies",
                "retweet_ratio":          "Fraction of posts that are retweets",
                "mention_ratio":          "Fraction of posts containing mentions",
                "url_ratio":              "Fraction of posts containing URLs",
                "hashtag_ratio":          "Fraction of posts containing hashtags",
                "duplicate_content_ratio":"Fraction of near-duplicate posts",
                "has_profile_image":      "1 if profile has custom image",
                "has_description":        "1 if profile has bio/description",
                "default_profile":        "1 if profile uses default theme",
                "profile_completeness":   "0-1 score of how complete profile is",
                "location":               "Self-reported location (nullable)",
                "listed_count":           "How many lists user appears on (nullable)",
            },
            "tweets.csv": {
                "tweet_id":      "Unique tweet identifier",
                "user_id":       "Foreign key to users.csv",
                "text":          "Tweet text content",
                "timestamp":     "Posted datetime (YYYY-MM-DD HH:MM:SS)",
                "retweet_count": "Number of retweets",
                "like_count":    "Number of likes",
                "reply_count":   "Number of replies to this tweet",
                "has_url":       "1 if tweet contains a URL",
                "has_hashtag":   "1 if tweet contains a hashtag",
                "is_retweet":    "1 if tweet is a retweet",
                "account_type":  "Inherited account type label",
            },
            "edges.csv": {
                "source_user_id":   "User initiating the interaction",
                "target_user_id":   "User receiving the interaction",
                "interaction_type": "follows | retweets | replies | mentions",
                "weight":           "Edge weight (interaction strength 0-1)",
                "timestamp":        "Date of interaction (YYYY-MM-DD)",
            },
        },
    }


# =============================================================================
# 7. Main entry point
# =============================================================================

def main(tweets_cap: int = 20, target_edges: int = 55000):
    print("\n" + "=" * 65)
    print("  Social Media Bot Detection -- Dataset Generator")
    print("=" * 65)

    print("\n[1/4] Generating user profiles ...")
    users_df = generate_users()

    print("\n[2/4] Generating tweet records ...")
    tweets_df = generate_tweets(users_df, tweets_per_user_cap=tweets_cap)

    print("\n[3/4] Generating interaction edges ...")
    edges_df = generate_edges(users_df, target_edges=target_edges)

    print("\n[4/4] Saving CSV files ...")
    users_df.to_csv(RAW_DIR  / "users.csv",  index=False)
    tweets_df.to_csv(RAW_DIR / "tweets.csv", index=False)
    edges_df.to_csv(RAW_DIR  / "edges.csv",  index=False)

    meta = generate_metadata(users_df, tweets_df, edges_df)
    with open(RAW_DIR / "dataset_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Dataset saved to: {RAW_DIR}")
    print("=" * 65)
    print(f"  users.csv   -> {(RAW_DIR/'users.csv').stat().st_size/1024:.1f} KB  |  {len(users_df):,} rows")
    print(f"  tweets.csv  -> {(RAW_DIR/'tweets.csv').stat().st_size/1024:.1f} KB  |  {len(tweets_df):,} rows")
    print(f"  edges.csv   -> {(RAW_DIR/'edges.csv').stat().st_size/1024:.1f} KB  |  {len(edges_df):,} rows")
    print("=" * 65 + "\n")

    return users_df, tweets_df, edges_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bot-detection synthetic dataset")
    parser.add_argument("--tweets-cap",   type=int, default=20,
                        help="Max tweets per user (default 20 -> ~100k total)")
    parser.add_argument("--target-edges", type=int, default=55000,
                        help="Target number of interaction edges (default 55000)")
    args = parser.parse_args()
    main(tweets_cap=args.tweets_cap, target_edges=args.target_edges)
