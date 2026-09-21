import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="analyst")  # analyst, admin, auditor
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)
    record_count = Column(Integer, default=0)
    edge_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    # Multi-dataset additions
    source = Column(String(50), default="seed")          # seed | upload
    status = Column(String(50), default="ready")         # ready | ingesting | failed
    is_default = Column(Boolean, default=False)          # exactly one row is True
    ingestion_meta = Column(Text, nullable=True)         # JSON blob: progress, stage, error

    accounts = relationship("Account", back_populates="dataset", cascade="all, delete-orphan")
    uploads = relationship("DatasetUpload", back_populates="dataset", cascade="all, delete-orphan")


class DatasetUpload(Base):
    """
    One row per upload attempt. Stores raw upload metadata, column mapping
    chosen during the confirm step, and rejected-row details after validation.
    """
    __tablename__ = "dataset_uploads"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=True)
    stored_path = Column(String(500), nullable=True)       # absolute path on disk
    column_map = Column(Text, nullable=True)               # JSON: {"screen_name": "user_name", ...}
    rejected_count = Column(Integer, default=0)
    rejected_rows = Column(Text, nullable=True)            # JSON array of {row, reason}
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    dataset = relationship("Dataset", back_populates="uploads")


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id_str = Column(String(100), index=True, nullable=False)
    screen_name = Column(String(100), index=True, nullable=False)
    name = Column(String(150), nullable=True)
    description = Column(Text, nullable=True)
    location = Column(String(150), nullable=True)
    created_at = Column(String(100), nullable=True)
    account_age_days = Column(Integer, default=0)
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    post_count = Column(Integer, default=0)
    listed_count = Column(Integer, default=0)
    verified = Column(Boolean, default=False)
    has_profile_image = Column(Boolean, default=True)
    has_description = Column(Boolean, default=True)
    default_profile = Column(Boolean, default=False)
    profile_completeness = Column(Float, default=1.0)
    ground_truth = Column(String(50), default="unknown")  # human, bot, suspicious
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)

    dataset = relationship("Dataset", back_populates="accounts")
    account_features = relationship("AccountFeature", back_populates="account", uselist=False, cascade="all, delete-orphan")
    graph_features = relationship("GraphFeature", back_populates="account", uselist=False, cascade="all, delete-orphan")
    posts = relationship("Post", back_populates="account", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="account", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="account", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_account_dataset_id", "dataset_id"),
        Index("idx_account_dataset_screen_name", "dataset_id", "screen_name"),
        Index("idx_accounts_dataset_user", "dataset_id", "user_id_str"),
    )


class AccountFeature(Base):
    __tablename__ = "account_features"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), unique=True, index=True, nullable=False)
    follower_friend_ratio = Column(Float, default=0.0)
    rep_score = Column(Float, default=0.0)
    activity_rate = Column(Float, default=0.0)
    posts_per_day = Column(Float, default=0.0)
    reply_ratio = Column(Float, default=0.0)
    retweet_ratio = Column(Float, default=0.0)
    mention_ratio = Column(Float, default=0.0)
    url_ratio = Column(Float, default=0.0)
    hashtag_ratio = Column(Float, default=0.0)
    duplicate_content_ratio = Column(Float, default=0.0)
    tweet_hour_entropy = Column(Float, default=0.0)
    tweet_similarity_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    activity_consistency = Column(Float, default=0.0)

    account = relationship("Account", back_populates="account_features")


class GraphFeature(Base):
    __tablename__ = "graph_features"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), unique=True, index=True, nullable=False)
    in_degree = Column(Integer, default=0)
    out_degree = Column(Integer, default=0)
    total_degree = Column(Integer, default=0)
    in_degree_centrality = Column(Float, default=0.0)
    out_degree_centrality = Column(Float, default=0.0)
    pagerank = Column(Float, default=0.0)
    betweenness_centrality = Column(Float, default=0.0)
    clustering_coeff = Column(Float, default=0.0)
    k_core = Column(Integer, default=0)

    account = relationship("Account", back_populates="graph_features")


class Edge(Base):
    __tablename__ = "edges"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    source_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    target_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    relation_type = Column(String(50), default="interacts")  # retweet, mention, reply, follow
    weight = Column(Float, default=1.0)
    timestamp = Column(String(100), nullable=True)

    __table_args__ = (
        Index("idx_edge_dataset_source_target", "dataset_id", "source_id", "target_id"),
    )


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    tweet_id_str = Column(String(100), index=True, nullable=False)
    text = Column(Text, nullable=False)
    timestamp = Column(String(100), nullable=True)
    retweet_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    is_retweet = Column(Boolean, default=False)
    has_url = Column(Boolean, default=False)
    has_hashtag = Column(Boolean, default=False)

    account = relationship("Account", back_populates="posts")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    embedding_type = Column(String(50), default="gnn")  # gnn, text, node2vec
    vector_json = Column(Text, nullable=False)  # JSON-serialized vector
    dim = Column(Integer, default=64)
    model_name = Column(String(100), default="GAT")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    account = relationship("Account", back_populates="embeddings")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)  # GAT, GraphSAGE, GCN, Random Forest, XGBoost
    architecture = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=False)
    f1_macro = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    hyperparameters = Column(Text, nullable=True)  # JSON string
    checkpoint_path = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model_run", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), index=True, nullable=False)
    bot_probability = Column(Float, nullable=False)
    human_probability = Column(Float, nullable=False)
    suspicious_probability = Column(Float, nullable=False)
    predicted_class = Column(String(50), nullable=False)  # human, bot, suspicious
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    account = relationship("Account", back_populates="predictions")
    model_run = relationship("ModelRun", back_populates="predictions")
    signals = relationship("PredictionSignal", back_populates="prediction", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_prediction_dataset_account", "dataset_id", "account_id"),
    )


class PredictionSignal(Base):
    __tablename__ = "prediction_signals"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), index=True, nullable=False)
    signal_name = Column(String(100), nullable=False)
    signal_value = Column(Float, nullable=False)
    signal_weight = Column(Float, nullable=False)
    signal_category = Column(String(50), default="graph")  # profile, activity, content, graph, coordination
    description = Column(Text, nullable=True)

    prediction = relationship("Prediction", back_populates="signals")


class AttentionWeight(Base):
    __tablename__ = "attention_weights"

    id = Column(Integer, primary_key=True, index=True)
    source_account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    target_account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    layer_idx = Column(Integer, default=0)
    head_idx = Column(Integer, default=0)
    weight = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Community(Base):
    __tablename__ = "communities"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    community_id = Column(Integer, index=True, nullable=False)
    algorithm = Column(String(50), default="Louvain")
    size = Column(Integer, default=0)
    n_bots = Column(Integer, default=0)
    n_suspicious = Column(Integer, default=0)
    n_humans = Column(Integer, default=0)
    bot_concentration = Column(Float, default=0.0)
    suspicious_concentration = Column(Float, default=0.0)
    human_concentration = Column(Float, default=0.0)
    coordinated_ratio = Column(Float, default=0.0)
    internal_edges = Column(Integer, default=0)
    internal_density = Column(Float, default=0.0)
    classification = Column(String(50), default="Mixed")  # Bot Farm, Authentic Cluster, Coordinated Network
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    members = relationship("CommunityMember", back_populates="community", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_communities_dataset_cid", "dataset_id", "community_id", unique=True),
    )


class CommunityMember(Base):
    __tablename__ = "community_members"

    id = Column(Integer, primary_key=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), index=True, nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    role = Column(String(50), default="member")  # core, bridge, peripheral

    community = relationship("Community", back_populates="members")


class CoordinationCluster(Base):
    __tablename__ = "coordination_clusters"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    cluster_name = Column(String(100), nullable=False)
    coordination_type = Column(String(50), default="retweet_storm")
    account_count = Column(Integer, default=0)
    avg_similarity = Column(Float, default=0.0)
    time_window_seconds = Column(Integer, default=60)
    detected_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), default="flagged")  # flagged, confirmed, resolved

    members = relationship("CoordinationMember", back_populates="cluster", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_coord_cluster_dataset", "dataset_id"),
    )


class CoordinationMember(Base):
    __tablename__ = "coordination_members"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("coordination_clusters.id"), index=True, nullable=False)
    account_id = Column(Integer, ForeignKey("accounts.id"), index=True, nullable=False)
    similarity_score = Column(Float, default=1.0)

    cluster = relationship("CoordinationCluster", back_populates="members")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(100), primary_key=True, index=True)
    job_type = Column(String(100), nullable=False)  # batch_inference, community_detection, coordination_scan, ingest_dataset
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True, index=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    total_items = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100), nullable=False)
    target_type = Column(String(100), nullable=False)
    target_id = Column(String(100), nullable=True)
    ip_address = Column(String(50), default="127.0.0.1")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(Text, nullable=True)
