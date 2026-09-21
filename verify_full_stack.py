import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

def test_full_api():
    print("[*] Initializing TestClient on Graphwarden API...")
    client = TestClient(app)

    # 1. Health Check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("  [+] Health check passed:", res.json()["status"])

    # 2. Datasets
    res = client.get("/api/v1/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) >= 1
    print(f"  [+] Datasets check passed: {len(datasets)} dataset(s), records: {datasets[0]['record_count']}")

    # 3. Accounts List & Filter
    res = client.get("/api/v1/accounts?page=1&page_size=10&sort_by=bot_probability&sort_order=desc")
    assert res.status_code == 200
    acc_page = res.json()
    assert acc_page["total"] == 10000
    assert len(acc_page["items"]) == 10
    top_acc = acc_page["items"][0]
    print(f"  [+] Accounts list check passed: Total {acc_page['total']} accounts. Top bot: @{top_acc['screen_name']} ({top_acc['prediction']['bot_probability']*100:.1f}%)")

    # 4. Account Detail
    res = client.get(f"/api/v1/accounts/{top_acc['id']}")
    assert res.status_code == 200
    acc_detail = res.json()
    assert "account_features" in acc_detail
    assert "graph_features" in acc_detail
    print(f"  [+] Account detail check passed: UID={acc_detail['user_id_str']}, features loaded successfully.")

    # 5. PDF Dossier Export
    res = client.get(f"/api/v1/accounts/{top_acc['id']}/dossier/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
    print(f"  [+] PDF dossier export passed: Streamed {len(res.content)} bytes of archival case file.")

    # 6. Topological Subgraph
    res = client.get("/api/v1/graph/subgraph?limit_nodes=50")
    assert res.status_code == 200
    subgraph = res.json()
    assert subgraph["node_count"] == 50
    assert subgraph["edge_count"] > 0
    print(f"  [+] Subgraph check passed: {subgraph['node_count']} nodes, {subgraph['edge_count']} edges.")

    # 7. Ego Graph with GAT Attention
    res = client.get(f"/api/v1/graph/ego/{top_acc['id']}?hops=1&max_neighbors=15")
    assert res.status_code == 200
    ego = res.json()
    assert ego["node_count"] >= 1
    print(f"  [+] Ego network check passed: {ego['node_count']} local nodes around center account.")

    # 8. Communities & Bot Farm
    res = client.get("/api/v1/communities")
    assert res.status_code == 200
    comms = res.json()
    assert len(comms) == 31
    farm_7 = next((c for c in comms if c["community_id"] == 7), None)
    assert farm_7 is not None
    print(f"  [+] Communities check passed: 31 Louvain communities. Community #7 Bot Concentration: {farm_7['bot_concentration']*100:.1f}%.")

    # 9. Coordination Clusters
    res = client.get("/api/v1/coordination/clusters")
    assert res.status_code == 200
    clusters = res.json()
    assert len(clusters) >= 2
    print(f"  [+] Coordination check passed: {len(clusters)} rings detected, avg similarity: {clusters[0]['avg_similarity']*100:.1f}%.")

    # 10. Temporal Diurnal Distributions
    res = client.get("/api/v1/temporal/bursts")
    assert res.status_code == 200
    temporal = res.json()
    assert len(temporal["diurnal_distribution"]) == 24
    assert len(temporal["burst_events"]) >= 3
    print(f"  [+] Temporal check passed: 24-hour diurnal profile and {len(temporal['burst_events'])} burst events.")

    # 11. Model Registry
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) == 5
    gat_m = next(m for m in models if m["model_name"] == "GAT")
    print(f"  [+] Model registry check passed: 5 benchmark runs. GAT Test Accuracy: {gat_m['accuracy']*100:.2f}%.")

    # 12. Warm Inference On-Demand
    res = client.post(f"/api/v1/predict/account/{top_acc['id']}", json={"model_name": "GAT"})
    assert res.status_code == 200
    pred = res.json()
    assert "probabilities" in pred
    assert "signals" in pred
    print(f"  [+] Live inference check passed: Latency {pred['latency_ms']}ms, Bot Prob: {pred['probabilities']['bot']*100:.1f}%.")

    print("\n[SUCCESS] ALL 12 API VERIFICATION GATES PASSED CLEANLY WITH ZERO FAILURES!")

if __name__ == "__main__":
    test_full_api()
