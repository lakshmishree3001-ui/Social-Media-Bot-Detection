# GRAPHWARDEN // DISASTER RECOVERY & BARE-METAL RESTORE RUNBOOK
**Classification:** Forensic Engineering Operations  
**Target RTO (Recovery Time Objective):** < 3 minutes  
**Target RPO (Recovery Point Objective):** 0 seconds (Persistent Graph & Weight Checkpoints)

---

## 1. Prerequisites
Ensure the host machine satisfies:
* Python 3.10+ (or virtual environment with PyTorch & PyG installed)
* Node.js v20+ / npm (or the bundled standalone toolchain in `tools/node/`)
* Docker & Docker Compose (optional for containerized multi-node orchestration)

---

## 2. Fast Bare-Metal Restore (Local Python + Vite)

### Step 1: Environment Setup
```powershell
# In project root:
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install fastapi uvicorn sqlalchemy pydantic slowapi reportlab python-jose cryptography
```

### Step 2: Database Initialization & Seeding
```powershell
# Ingests 10,000 accounts, 54,980 edges, 31 communities, and 5 model runs:
.\venv\Scripts\python.exe backend/seed_database.py
```
*Verification:* Seed output ends with `[+] Database seeding successfully finished!`. Row counts should confirm `10000` accounts and `55000` edges.

### Step 3: Launch FastAPI Backend Server
```powershell
# Run API server on port 8000:
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Health Check:* Open `http://localhost:8000/api/health` -> should return `{"status": "healthy", ...}`.

### Step 4: Build & Launch Frontend Interface
```powershell
# In frontend directory:
cd frontend
..\tools\run_npm.bat install
..\tools\run_npm.bat run build
# To launch development preview:
..\tools\run_npm.bat run dev
```
*Verification:* Open `http://localhost:5173` to access the live Graphwarden forensic console.

---

## 3. Containerized Orchestration (Docker Compose)

To launch the full production cluster with PostgreSQL 16, pgvector, Redis, MinIO, MLflow, and Celery:

```bash
docker-compose up -d --build
```

### Port Mapping Reference
| Service | Internal Port | Host Port | Role |
| :--- | :--- | :--- | :--- |
| `web` | 80 | 80 | React SPA / Nginx Reverse Proxy |
| `api` | 8000 | 8000 | FastAPI REST Inference Service |
| `postgres` | 5432 | 5432 | PostgreSQL 16 + pgvector |
| `redis` | 6379 | 6379 | Celery Broker & Result Cache |
| `minio` | 9000 / 9001 | 9000 / 9001 | Object Storage (Weights & Artifacts) |
| `mlflow` | 5000 | 5000 | Experiment Registry & Metrics |

---

## 4. Emergency State Reset
To purge and recreate all databases from source CSVs:
```powershell
Remove-Item graphwarden.db -Force
.\venv\Scripts\python.exe backend/seed_database.py
```
