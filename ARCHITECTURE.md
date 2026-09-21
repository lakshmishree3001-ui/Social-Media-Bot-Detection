# GRAPHWARDEN // SYSTEM ARCHITECTURE SPECIFICATION
**Version:** 1.0.0  
**Classification:** Technical Engineering Architecture  
**Scope:** End-to-End Multimodal Graph Neural Network Bot Detection Platform

---

## 1. System Topology Overview

```
                           +---------------------------+
                           |  Analyst Web Console      |
                           |  (React 18 + TS + Vite)   |
                           +-------------+-------------+
                                         | HTTP / JSON / PDF
                                         v
                           +---------------------------+
                           |  FastAPI Gateway (Port 8000)|
                           |  - SlowAPI Rate Limiting  |
                           |  - JWT Bearer Auth        |
                           +-------------+-------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+           +-------------------+           +-------------------+
| SQL Storage      |           | Warm PyG Engine   |           | Forensic Pipeline |
| - PostgreSQL 16  |           | - PyG Graph Tensors|           | - Louvain Clusters|
| - SQLite Fallback|           | - GAT (4-Head Attn)|           | - Ego Subgraphs   |
| - 18 Data Tables |           | - GraphSAGE / GCN |           | - PDF Dossier Gen |
+------------------+           +-------------------+           +-------------------+
```

---

## 2. Multimodal Representation Pipeline

Each account vertex $v_i \in V$ is represented by a concatenated 80-dimensional feature vector $\mathbf{x}_i \in \mathbb{R}^{80}$:

$$\mathbf{x}_i = [\mathbf{x}_i^{\text{behavioral}} \,\|\, \mathbf{x}_i^{\text{lexical}} \,\|\, \mathbf{x}_i^{\text{topological}}]$$

1. **Behavioral & Profile Attributes (35 features):**
   * Follower-to-following ratio, reputation index, account longevity.
   * Diurnal posting entropy, URL link ratio, hashtag density, lexical duplication quotient.
2. **NLP Lexical Attributes (36 features):**
   * TF-IDF n-gram distributions over timeline text, vocabulary richness, sentiment polarity variance.
3. **Graph Topological Metrics (9 features):**
   * Directed in-degree, out-degree, total degree, PageRank, betweenness centrality, clustering coefficient, k-core number.

---

## 3. Graph Neural Network Architectures

### Graph Attention Network (GAT)
Computes dynamic attention weights between connected accounts:

$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)\right)}$$

* Layer 1: Multi-head attention (4 heads, output dimension 64) with LayerNorm and ELU activation.
* Layer 2: Graph Attention aggregation into 64-dim embedding space.
* Classifier: Linear projection into 3 output logits: `[Human, Bot, Suspicious]`.

### GraphSAGE
Inductive neighborhood aggregator using mean pooling over sampled multi-hop connections.

### GCN
Spectral graph convolution utilizing symmetric normalized adjacency $\mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$.

---

## 4. Database Schema (18 Tables)

| Table | Role | Key Fields |
| :--- | :--- | :--- |
| `users` | Analyst identities | `email`, `hashed_password`, `role`, `created_at` |
| `datasets` | Benchmark corpora metadata | `name`, `record_count`, `edge_count`, `density` |
| `accounts` | Indexed account nodes | `node_idx`, `user_id_str`, `screen_name`, `followers`, `following` |
| `account_features` | 35 behavioral features | `tweet_hour_entropy`, `url_ratio`, `duplicate_content_ratio` |
| `graph_features` | 9 topological metrics | `pagerank`, `betweenness_centrality`, `k_core` |
| `edges` | Topological interactions | `source_id`, `target_id`, `relation_type`, `weight` |
| `posts` | Sampled timeline tweets | `tweet_id_str`, `text`, `timestamp`, `retweet_count` |
| `embeddings` | 64-dim hidden representations | `vector_json`, `dim`, `model_name` |
| `model_runs` | MLflow experiment runs | `model_name`, `accuracy`, `f1_macro`, `roc_auc` |
| `predictions` | Calibrated posteriors | `bot_probability`, `human_probability`, `suspicious_probability` |
| `prediction_signals` | Feature attributions | `signal_name`, `signal_value`, `signal_weight`, `category` |
| `attention_weights` | Top GAT edge coefficients | `source_account_id`, `target_account_id`, `weight` |
| `communities` | Louvain partitions | `community_id`, `size`, `bot_concentration`, `classification` |
| `community_members` | Account cluster mapping | `community_id`, `account_id`, `role` |
| `coordination_clusters` | Amplification rings | `cluster_name`, `avg_similarity`, `time_window_seconds` |
| `coordination_members` | Ring participants | `cluster_id`, `account_id`, `similarity_score` |
| `jobs` | Async batch tasks | `job_type`, `status`, `progress`, `total_items` |
| `audit_log` | Forensic audit trail | `user_id`, `action`, `target_type`, `target_id`, `timestamp` |

---

## 5. Security & Operational Hardening
* **Authentication:** JWT Bearer tokens with SHA-256 password hashing.
* **Rate Limiting:** SlowAPI token bucket limiting at 200 requests/minute per client IP.
* **Audit Trail:** Every inference evaluation, model activation, and PDF export is written to `audit_log`.
