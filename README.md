# Graphwarden: Multimodal Graph Neural Network Bot Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.x-3C2179.svg)](https://pyg.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

**Graphwarden** is an end-to-end, enterprise-grade forensic platform designed to uncover coordinated social media bot networks, synchronized disinformation campaigns, and automated malicious accounts using **Multimodal Graph Neural Networks (GNNs)** combined with topological graph metrics, NLP lexical features, and temporal posting dynamics.

---

## 🌟 Key Architecture & Capabilities

- **🧠 Multimodal Graph Neural Networks:**
  - **Graph Attention Networks (GAT)** with 4-head attention mechanisms to weight suspicious neighboring interactions.
  - **GraphSAGE** for inductive neighborhood aggregation and large-scale graph inference.
  - **Graph Convolutional Networks (GCN)** & Traditional ML baselines (Random Forest, XGBoost, SVM).
- **🕸️ Interactive Ego-Graph Forensics:**
  - Visual exploration of 1-hop and 2-hop ego subgraphs powered by Cytoscape and D3.js.
  - Identification of Louvain modularity communities and dense core clusters.
- **⚡ Coordinated Campaign Detection:**
  - Uncovers synchronous posting bursts, identical narrative amplification, and synchronized retweet rings.
- **📄 Automated Forensic Dossiers:**
  - Real-time generation of forensic PDF dossiers complete with risk scoring, radar charts, and topological metrics.
- **🛡️ Enterprise API & Microservices:**
  - FastAPI with rate limiting (`slowapi`), JWT authentication, and OpenAPI documentation (`/api/docs`).
  - PostgreSQL 16 with `pgvector`, Redis caching, Celery asynchronous workers, and MinIO storage.

---

## 🏗️ System Topology

```
                         +---------------------------+
                         |    Analyst Web Console    |
                         |   (React 18 + TS + Vite)  |
                         +-------------+-------------+
                                       | HTTP / JSON / PDF
                                       v
                         +---------------------------+
                         |  FastAPI Gateway (:8000)  |
                         |  - Rate Limiting (SlowAPI)|
                         |  - JWT Bearer Auth        |
                         +-------------+-------------+
                                       |
       +-------------------------------+-------------------------------+
       |                               |                               |
       v                               v                               v
+------------------+           +-------------------+           +-------------------+
| SQL & Embeddings |           | Warm PyG Engine   |           | Forensic Pipeline |
| - PostgreSQL 16  |           | - PyG Tensors     |           | - Louvain Clusters|
| - pgvector       |           | - GAT (4-Head)    |           | - Ego Subgraphs   |
| - SQLite Fallback|           | - GraphSAGE / GCN |           | - PDF Dossier Gen |
+------------------+           +-------------------+           +-------------------+
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Docker & Docker Compose

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/lakshmishree3001-ui/Social-Media-Bot-Detection.git
cd Social-Media-Bot-Detection

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run database setup & seeding
python backend/seed_database.py

# Start FastAPI backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/api/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🐳 Production Deployment (Docker Compose)

Deploy the complete multi-container stack (Postgres, Redis, Celery, FastAPI, Nginx Frontend, MinIO, MLflow) with one command:

```bash
# Copy and adjust environment variables if needed
cp .env.example .env

# Build and start all services
docker compose up -d --build

# Seed initial forensic datasets
docker compose exec api python backend/seed_database.py
```

Access the application at `http://localhost` (Port 80) and API docs at `http://localhost:8000/api/docs`.

---

## 📊 Evaluation & Machine Learning Modules

The repository includes reproducible research notebooks documenting the entire pipeline from data ingestion to explainability:
- `module2_dataset_understanding.ipynb`: Distribution analysis of verified, bot, and human accounts.
- `module3_data_preparation.ipynb`: Imputation, outlier handling, and data normalization.
- `module4_feature_engineering.ipynb`: Behavioral, lexical, and topological feature extraction (80-dim representation).
- `module5_nlp_analysis.ipynb`: TF-IDF, sentiment variance, and linguistic entropy.
- `module6_graph_construction.ipynb`: Adjacency matrices, PyG `Data` tensors, edge construction.
- `module8_traditional_ml_baseline.ipynb`: Benchmarking RF, XGBoost, Decision Trees, and Logistic Regression.
- `module9_12_gnn_architectures.ipynb`: GCN, GraphSAGE, and GAT multi-head attention implementations.
- `module14_16_community_coordination.ipynb`: Louvain community detection and coordinated campaign discovery.
- `module19_explainable_ai.ipynb`: Integrated Gradients, attention weight extraction, and GNNExplainer.
- `module20_final_system.ipynb`: Full end-to-end integration and forensic evaluation.

---

## 📜 License
This project is developed for educational and forensic research purposes.
