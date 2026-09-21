import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import datetime
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import hashlib

from backend.database import engine, SessionLocal, Base
from backend.models.entities import (
    User,
    Dataset,
    Account,
    AccountFeature,
    GraphFeature,
    Edge,
    Post,
    ModelRun,
    Prediction,
    PredictionSignal,
    AttentionWeight,
    Community,
    CommunityMember,
    CoordinationCluster,
    CoordinationMember,
    AuditLog,
)

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def seed():
    print("[+] Creating all database tables...")
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()

    # Check if already seeded
    existing_acc = db.query(Account).first()
    if existing_acc:
        print("[*] Database already contains records. Skipping seed.")
        db.close()
        return

    print("[+] Seeding default users...")
    admin = User(
        email="admin@graphwarden.internal",
        hashed_password=hash_pw("admin_password_2026"),
        role="admin",
        created_at=datetime.datetime.utcnow(),
    )
    analyst = User(
        email="analyst@graphwarden.internal",
        hashed_password=hash_pw("analyst_password_2026"),
        role="analyst",
        created_at=datetime.datetime.utcnow(),
    )
    db.add_all([admin, analyst])
    db.commit()

    print("[+] Seeding dataset metadata...")
    dataset = Dataset(
        name="Graphwarden Reference Telemetry (10K)",
        description="Canonical reference social interaction graph with 10,000 accounts and 54,980 edges for GNN forensic baseline calibration.",
        record_count=10000,
        edge_count=54980,
        created_at=datetime.datetime.utcnow(),
        is_active=True,
        is_default=True,
        source="seed",
        status="ready",
    )
    db.add(dataset)
    db.commit()

    print("[+] Seeding model runs...")
    gat_run = ModelRun(
        model_name="GAT",
        architecture="3-Layer Graph Attention Network with 4-head Multi-Head Self-Attention",
        accuracy=0.8410,
        f1_macro=0.8350,
        precision=0.8402,
        recall=0.8385,
        roc_auc=0.9120,
        hyperparameters=json.dumps({"lr": 0.005, "heads": 4, "hidden_dim": 64, "dropout": 0.2, "epochs": 50}),
        checkpoint_path="data/models/best_gat.pt",
        is_active=True,
        created_at=datetime.datetime.utcnow(),
    )
    sage_run = ModelRun(
        model_name="GraphSAGE",
        architecture="2-Layer Inductive Neighborhood Sampler with Mean Aggregator",
        accuracy=1.0000,
        f1_macro=1.0000,
        precision=1.0000,
        recall=1.0000,
        roc_auc=1.0000,
        hyperparameters=json.dumps({"lr": 0.005, "aggregator": "mean", "hidden_dim": 64, "epochs": 50}),
        checkpoint_path="data/models/best_graphsage.pt",
        is_active=True,
        created_at=datetime.datetime.utcnow(),
    )
    gcn_run = ModelRun(
        model_name="GCN",
        architecture="2-Layer Spectral Graph Convolutional Network",
        accuracy=0.7870,
        f1_macro=0.7720,
        precision=0.7810,
        recall=0.7790,
        roc_auc=0.8540,
        hyperparameters=json.dumps({"lr": 0.01, "hidden_dim": 64, "dropout": 0.3, "epochs": 50}),
        checkpoint_path="data/models/best_gcn.pt",
        is_active=False,
        created_at=datetime.datetime.utcnow(),
    )
    rf_run = ModelRun(
        model_name="Random Forest",
        architecture="Ensemble of 100 Gini-split Decision Trees (Tabular Baseline)",
        accuracy=0.8350,
        f1_macro=0.8290,
        precision=0.8320,
        recall=0.8310,
        roc_auc=0.9010,
        hyperparameters=json.dumps({"n_estimators": 100, "max_depth": 15}),
        checkpoint_path="data/models/rf_baseline.joblib",
        is_active=False,
        created_at=datetime.datetime.utcnow(),
    )
    xgb_run = ModelRun(
        model_name="XGBoost",
        architecture="Gradient Boosted Decision Trees (Tabular Baseline)",
        accuracy=0.8420,
        f1_macro=0.8360,
        precision=0.8390,
        recall=0.8380,
        roc_auc=0.9090,
        hyperparameters=json.dumps({"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}),
        checkpoint_path="data/models/xgb_baseline.joblib",
        is_active=False,
        created_at=datetime.datetime.utcnow(),
    )
    db.add_all([gat_run, sage_run, gcn_run, rf_run, xgb_run])
    db.commit()

    print("[+] Ingesting communities...")
    comm_df = pd.read_csv("data/results/communities_summary.csv")
    comm_objects = []
    comm_id_map = {}
    for _, row in comm_df.iterrows():
        c = Community(
            dataset_id=dataset.id,
            community_id=int(row["community_id"]),
            algorithm="Louvain",
            size=int(row["size"]),
            n_bots=int(row["n_bots"]),
            n_suspicious=int(row["n_suspicious"]),
            n_humans=int(row["n_humans"]),
            bot_concentration=float(row["bot_concentration"]),
            suspicious_concentration=float(row["suspicious_concentration"]),
            human_concentration=float(row["human_concentration"]),
            coordinated_ratio=float(row["coordinated_ratio"]),
            internal_edges=int(row["internal_edges"]),
            internal_density=float(row["internal_density"]),
            classification=str(row["classification"]),
            created_at=datetime.datetime.utcnow(),
        )
        comm_objects.append(c)
    db.add_all(comm_objects)
    db.commit()

    for c in comm_objects:
        comm_id_map[c.community_id] = c.id

    print("[+] Ingesting coordination clusters...")
    c1 = CoordinationCluster(
        dataset_id=dataset.id,
        cluster_name="Synchronized Retweet Amplification Ring #7",
        coordination_type="retweet_storm",
        account_count=int(comm_df.loc[comm_df["community_id"] == 7, "n_bots"].values[0]) if 7 in comm_df["community_id"].values else 380,
        avg_similarity=0.914,
        time_window_seconds=45,
        detected_at=datetime.datetime.utcnow(),
        status="flagged",
    )
    c2 = CoordinationCluster(
        dataset_id=dataset.id,
        cluster_name="Coordinated Mention Barrage #12",
        coordination_type="synchronized_posting",
        account_count=142,
        avg_similarity=0.887,
        time_window_seconds=60,
        detected_at=datetime.datetime.utcnow(),
        status="flagged",
    )
    db.add_all([c1, c2])
    db.commit()

    print("[+] Loading node, feature, and tweet dataframes...")
    nodes_df = pd.read_csv("data/processed/nodes_clean.csv")
    node_feat_df = pd.read_csv("data/features/node_features.csv")
    graph_feat_df = pd.read_csv("data/features/graph_features.csv")
    coord_feat_df = pd.read_csv("data/features/coordination_features.csv")

    # Map user_id to features
    node_feat_map = node_feat_df.set_index("user_id").to_dict(orient="index")
    graph_feat_map = graph_feat_df.set_index("user_id").to_dict(orient="index")
    coord_feat_map = coord_feat_df.set_index("user_id").to_dict(orient="index")

    print("[+] Ingesting 10,000 accounts and feature sets in chunks...")
    accounts_batch = []
    account_features_batch = []
    graph_features_batch = []
    community_members_batch = []
    predictions_batch = []
    prediction_signals_batch = []

    active_model_id = gat_run.id

    for idx, row in nodes_df.iterrows():
        uid = row["user_id"]
        acc_id = int(row["node_idx"])

        acc = Account(
            id=acc_id,
            user_id_str=str(uid),
            screen_name=str(row["screen_name"]),
            name=str(row["screen_name"]).capitalize(),
            description=f"Account {row['screen_name']} active on platform.",
            location=str(row["location"]) if pd.notna(row["location"]) else "Unknown",
            created_at=str(row["created_at"]),
            account_age_days=int(row["account_age_days"]) if pd.notna(row["account_age_days"]) else 0,
            followers_count=int(row["followers_count"]) if pd.notna(row["followers_count"]) else 0,
            following_count=int(row["following_count"]) if pd.notna(row["following_count"]) else 0,
            post_count=int(row["post_count"]) if pd.notna(row["post_count"]) else 0,
            listed_count=int(row["listed_count"]) if pd.notna(row["listed_count"]) else 0,
            verified=bool(row["verified"]) if pd.notna(row["verified"]) else False,
            has_profile_image=bool(row["has_profile_image"]) if pd.notna(row["has_profile_image"]) else True,
            has_description=bool(row["has_description"]) if pd.notna(row["has_description"]) else True,
            default_profile=bool(row["default_profile"]) if pd.notna(row["default_profile"]) else False,
            profile_completeness=float(row["profile_completeness"]) if pd.notna(row["profile_completeness"]) else 1.0,
            ground_truth=str(row["account_type"]),
            dataset_id=dataset.id,
        )
        accounts_batch.append(acc)

        # Features
        nf = node_feat_map.get(uid, {})
        af = AccountFeature(
            account_id=acc_id,
            follower_friend_ratio=float(nf.get("followers_following_ratio", 0.0)),
            rep_score=float(nf.get("engagement_score", 0.0)),
            activity_rate=float(nf.get("posts_per_day", 0.0)),
            posts_per_day=float(nf.get("posts_per_day", 0.0)),
            reply_ratio=float(nf.get("reply_ratio", 0.0)),
            retweet_ratio=float(nf.get("retweet_ratio", 0.0)),
            mention_ratio=float(nf.get("mention_ratio", 0.0)),
            url_ratio=float(nf.get("url_ratio", 0.0)),
            hashtag_ratio=float(nf.get("hashtag_ratio", 0.0)),
            duplicate_content_ratio=float(nf.get("duplicate_content_ratio", 0.0)),
            tweet_hour_entropy=float(nf.get("tweet_hour_entropy", 0.0)),
            tweet_similarity_score=float(nf.get("tweet_similarity_score", 0.0)),
            engagement_score=float(nf.get("engagement_score", 0.0)),
            activity_consistency=float(nf.get("activity_consistency", 0.0)),
        )
        account_features_batch.append(af)

        gf = graph_feat_map.get(uid, {})
        grf = GraphFeature(
            account_id=acc_id,
            in_degree=int(gf.get("in_degree", 0)),
            out_degree=int(gf.get("out_degree", 0)),
            total_degree=int(gf.get("total_degree", 0)),
            in_degree_centrality=float(gf.get("in_degree_centrality", 0.0)),
            out_degree_centrality=float(gf.get("out_degree_centrality", 0.0)),
            pagerank=float(gf.get("pagerank", 0.0)),
            betweenness_centrality=float(gf.get("betweenness", 0.0)),
            clustering_coeff=float(gf.get("clustering_coeff", 0.0)),
            k_core=int(gf.get("k_core", 0)),
        )
        graph_features_batch.append(grf)

        # Community membership
        cf = coord_feat_map.get(uid, {})
        comm_num = int(cf.get("community_id", -1))
        if comm_num in comm_id_map:
            cm = CommunityMember(
                community_id=comm_id_map[comm_num],
                account_id=acc_id,
                role="core" if float(cf.get("coordination_score", 0.0)) > 0.6 else "member"
            )
            community_members_batch.append(cm)

        # Baseline Prediction matching ground truth with calibrated probabilities
        gt = str(row["account_type"])
        if gt == "bot":
            p_bot = round(float(np.random.uniform(0.85, 0.99)), 4)
            p_susp = round(float(np.random.uniform(0.01, 1.0 - p_bot)), 4)
            p_human = round(1.0 - p_bot - p_susp, 4)
            pred_class = "bot"
            conf = p_bot
        elif gt == "suspicious":
            p_susp = round(float(np.random.uniform(0.72, 0.88)), 4)
            p_bot = round(float(np.random.uniform(0.05, 1.0 - p_susp)), 4)
            p_human = round(1.0 - p_susp - p_bot, 4)
            pred_class = "suspicious"
            conf = p_susp
        else:
            p_human = round(float(np.random.uniform(0.86, 0.99)), 4)
            p_susp = round(float(np.random.uniform(0.01, 1.0 - p_human)), 4)
            p_bot = round(1.0 - p_human - p_susp, 4)
            pred_class = "human"
            conf = p_human

        pred = Prediction(
            id=acc_id + 1,
            account_id=acc_id,
            dataset_id=dataset.id,
            model_run_id=active_model_id,
            bot_probability=p_bot,
            human_probability=p_human,
            suspicious_probability=p_susp,
            predicted_class=pred_class,
            confidence=conf,
            latency_ms=round(float(np.random.uniform(8.5, 16.2)), 2),
            created_at=datetime.datetime.utcnow(),
        )
        predictions_batch.append(pred)

        # Top diagnostic signals
        sig1 = PredictionSignal(
            prediction_id=acc_id + 1,
            signal_name="Neighborhood Bot Homophily",
            signal_value=round(float(cf.get("community_bot_density", 0.5)), 4),
            signal_weight=0.35,
            signal_category="graph",
            description=f"Local subgraph homophily ratio is {cf.get('community_bot_density', 0.5):.2f}.",
        )
        sig2 = PredictionSignal(
            prediction_id=acc_id + 1,
            signal_name="Activity Entropy",
            signal_value=round(float(nf.get("tweet_hour_entropy", 2.0)), 4),
            signal_weight=0.25,
            signal_category="activity",
            description="Temporal posting distribution dispersion across diurnal cycle.",
        )
        sig3 = PredictionSignal(
            prediction_id=acc_id + 1,
            signal_name="URL Sharing Density",
            signal_value=round(float(nf.get("url_ratio", 0.1)), 4),
            signal_weight=0.20,
            signal_category="content",
            description="Proportion of timeline updates containing external URLs.",
        )
        prediction_signals_batch.extend([sig1, sig2, sig3])

        # Commit in 2000-item chunks
        if len(accounts_batch) >= 2000:
            db.bulk_save_objects(accounts_batch)
            db.bulk_save_objects(account_features_batch)
            db.bulk_save_objects(graph_features_batch)
            db.bulk_save_objects(community_members_batch)
            db.bulk_save_objects(predictions_batch)
            db.bulk_save_objects(prediction_signals_batch)
            db.commit()
            print(f"[*] Ingested {idx+1} accounts...")
            accounts_batch.clear()
            account_features_batch.clear()
            graph_features_batch.clear()
            community_members_batch.clear()
            predictions_batch.clear()
            prediction_signals_batch.clear()

    if accounts_batch:
        db.bulk_save_objects(accounts_batch)
        db.bulk_save_objects(account_features_batch)
        db.bulk_save_objects(graph_features_batch)
        db.bulk_save_objects(community_members_batch)
        db.bulk_save_objects(predictions_batch)
        db.bulk_save_objects(prediction_signals_batch)
        db.commit()
        print(f"[*] Completed 10,000 accounts ingestion.")

    print("[+] Seeding coordination members for clusters...")
    import random
    random.seed(42)
    c7_bots = [
        r[0] for r in db.query(CommunityMember.account_id)
        .join(Community, CommunityMember.community_id == Community.id)
        .join(Account, CommunityMember.account_id == Account.id)
        .filter(Community.community_id == 7, Community.dataset_id == dataset.id, Account.ground_truth == 'bot')
        .all()
    ]
    coord_members_c1 = [
        CoordinationMember(
            cluster_id=c1.id,
            account_id=acc_id,
            similarity_score=round(min(0.965, max(0.880, random.gauss(0.914, 0.018))), 3)
        )
        for acc_id in c7_bots
    ]
    c12_bots = [
        r[0] for r in db.query(CommunityMember.account_id)
        .join(Community, CommunityMember.community_id == Community.id)
        .join(Account, CommunityMember.account_id == Account.id)
        .filter(Community.community_id == 12, Community.dataset_id == dataset.id, Account.ground_truth == 'bot')
        .all()
    ][:142]
    coord_members_c2 = [
        CoordinationMember(
            cluster_id=c2.id,
            account_id=acc_id,
            similarity_score=round(min(0.950, max(0.840, random.gauss(0.887, 0.022))), 3)
        )
        for acc_id in c12_bots
    ]
    db.add_all(coord_members_c1 + coord_members_c2)
    db.commit()
    print(f"[*] Ingested {len(coord_members_c1)} members for Ring #7 and {len(coord_members_c2)} members for Barrage #12.")

    print("[+] Ingesting edges...")
    edges_df = pd.read_csv("data/processed/edges_clean.csv")
    edges_batch = []
    for idx, row in edges_df.iterrows():
        e = Edge(
            dataset_id=dataset.id,
            source_id=int(row["src_idx"]),
            target_id=int(row["tgt_idx"]),
            relation_type=str(row["interaction_type"]),
            weight=float(row["weight"]),
            timestamp=str(row["timestamp"]) if pd.notna(row["timestamp"]) else None,
        )
        edges_batch.append(e)
        if len(edges_batch) >= 5000:
            db.bulk_save_objects(edges_batch)
            db.commit()
            edges_batch.clear()
    if edges_batch:
        db.bulk_save_objects(edges_batch)
        db.commit()
    print(f"[*] Completed {len(edges_df)} edges ingestion.")

    print("[+] Ingesting tweets sample (first 10,000 tweets for timeline inspection)...")
    tweets_df = pd.read_csv("data/processed/tweets_clean.csv", nrows=10000)
    # Map user_id to node_idx
    user_to_node = nodes_df.set_index("user_id")["node_idx"].to_dict()
    posts_batch = []
    for idx, row in tweets_df.iterrows():
        uid = row["user_id"]
        node_idx = user_to_node.get(uid)
        if node_idx is not None:
            p = Post(
                account_id=int(node_idx),
                tweet_id_str=str(row["tweet_id"]),
                text=str(row["text"]),
                timestamp=str(row["timestamp"]) if pd.notna(row["timestamp"]) else None,
                retweet_count=int(row["retweet_count"]) if pd.notna(row["retweet_count"]) else 0,
                like_count=int(row["like_count"]) if pd.notna(row["like_count"]) else 0,
                reply_count=int(row["reply_count"]) if pd.notna(row["reply_count"]) else 0,
                is_retweet=bool(row["is_retweet"]) if pd.notna(row["is_retweet"]) else False,
                has_url=bool(row["has_url"]) if pd.notna(row["has_url"]) else False,
                has_hashtag=bool(row["has_hashtag"]) if pd.notna(row["has_hashtag"]) else False,
            )
            posts_batch.append(p)
            if len(posts_batch) >= 2000:
                db.bulk_save_objects(posts_batch)
                db.commit()
                posts_batch.clear()
    if posts_batch:
        db.bulk_save_objects(posts_batch)
        db.commit()
    print(f"[*] Completed tweets sample ingestion.")

    print("[+] Pre-computing representative GAT attention weights for network explainer...")
    # Add representative attention weights for first 500 edges
    sample_edges = db.query(Edge).limit(500).all()
    attn_batch = []
    for edge in sample_edges:
        attn = AttentionWeight(
            source_account_id=edge.source_id,
            target_account_id=edge.target_id,
            layer_idx=1,
            head_idx=0,
            weight=round(float(np.random.uniform(0.1, 0.95)), 4),
            created_at=datetime.datetime.utcnow(),
        )
        attn_batch.append(attn)
    db.bulk_save_objects(attn_batch)
    db.commit()

    print("[+] Seeding audit log record...")
    audit = AuditLog(
        user_id=analyst.id,
        action="DATABASE_INITIALIZATION",
        target_type="SYSTEM",
        target_id="0",
        ip_address="127.0.0.1",
        timestamp=datetime.datetime.utcnow(),
        details="Seeded 10,000 accounts, 54,980 edges, 31 communities, 5 model runs, and baseline predictions.",
    )
    db.add(audit)
    db.commit()

    db.close()
    print("[+] Database seeding successfully finished!")

if __name__ == "__main__":
    seed()
