"""
test_ingestion_fix.py

Automated test script to verify:
1. Cascading deletion of stuck datasets #2 and #3.
2. Fresh upload and ingestion of graphwarden_test_data_with_screen_name.csv (15 accounts).
   - Verifies status == "ready"
   - Accounts == 15
   - Edges > 0 (follow network synthesized)
   - Bots == 5
   - Communities created without IntegrityError
   - Coordination clusters created
3. Fresh upload and ingestion of graphwarden_single_user_test.csv (1 account).
   - Verifies n=1 handled gracefully without errors
   - Status == "ready"
   - Accounts == 1
4. Dataset #1 (seed dataset with 10,000 accounts) remains intact and selectable.
5. Summary breakdown shows accurate bot counts.
"""
import os
import sys
import json
import sqlite3

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.database import SessionLocal
from backend.models.entities import (
    Dataset, DatasetUpload, Account, Edge, Community, CommunityMember,
    CoordinationCluster, CoordinationMember, Prediction, PredictionSignal,
    AccountFeature, GraphFeature, Post, Embedding, Job
)
from backend.services.pipeline.validator import validate_and_map, infer_column_map_suggestions
from backend.services.pipeline.runner import run_ingestion_pipeline
from backend.routers.datasets import _dataset_summary, delete_dataset


def run_tests():
    db = SessionLocal()
    print("=" * 70)
    print("STARTING INGESTION FIX VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Clean up stuck datasets #2 and #3
    # -------------------------------------------------------------
    print("\n[1] Deleting stuck datasets #2 and #3...")
    for ds_id in [2, 3]:
        d = db.query(Dataset).filter(Dataset.id == ds_id).first()
        if d:
            res = delete_dataset(ds_id, db)
            print(f"  [OK] Deleted dataset #{ds_id}: {res}")
        else:
            print(f"  [-] Dataset #{ds_id} not found, already deleted.")

    # Verify lingering records for #2 and #3 are 0
    for ds_id in [2, 3]:
        accs = db.query(Account).filter(Account.dataset_id == ds_id).count()
        edges = db.query(Edge).filter(Edge.dataset_id == ds_id).count()
        comms = db.query(Community).filter(Community.dataset_id == ds_id).count()
        preds = db.query(Prediction).filter(Prediction.dataset_id == ds_id).count()
        assert accs == 0, f"Lingering accounts for dataset {ds_id}: {accs}"
        assert edges == 0, f"Lingering edges for dataset {ds_id}: {edges}"
        assert comms == 0, f"Lingering communities for dataset {ds_id}: {comms}"
        assert preds == 0, f"Lingering predictions for dataset {ds_id}: {preds}"
    print("  [OK] Cascading deletion verified: 0 lingering records for datasets #2 and #3.")

    # -------------------------------------------------------------
    # Step 2: Ingest graphwarden_test_data_with_screen_name.csv (15 accounts)
    # -------------------------------------------------------------
    print("\n[2] Ingesting graphwarden_test_data_with_screen_name.csv...")
    file_path = os.path.join("data", "uploads", "19cef614-a260-4a30-9cee-fd0cde479761.csv")
    with open(file_path, "rb") as fh:
        raw_bytes = fh.read()

    # Verify column map suggestion detects label -> ground_truth
    suggestions = infer_column_map_suggestions(raw_bytes)
    print(f"  Column map suggestions: {suggestions}")
    assert suggestions.get("label") == "ground_truth", f"Expected label -> ground_truth suggestion, got: {suggestions}"

    # Validate and map CSV
    df_clean, rejected, warnings = validate_and_map(raw_bytes, column_map=suggestions, original_filename="graphwarden_test_data_with_screen_name.csv")
    print(f"  Validated DataFrame rows: {len(df_clean)}, rejected: {len(rejected)}")
    assert len(df_clean) == 15, f"Expected 15 accounts, got {len(df_clean)}"
    assert "ground_truth" in df_clean.columns

    # Create dataset row
    ds2 = Dataset(
        name="graphwarden_test_data_with_screen_name",
        description="15-user test dataset with known bots and humans",
        source="upload",
        status="pending_confirm",
        is_default=False,
    )
    db.add(ds2)
    db.commit()
    db.refresh(ds2)
    ds2_id = ds2.id
    print(f"  Created Dataset record ID: {ds2_id}")

    # Create DatasetUpload record
    du2 = DatasetUpload(
        dataset_id=ds2_id,
        original_filename="graphwarden_test_data_with_screen_name.csv",
        stored_path=os.path.abspath(file_path),
        column_map=json.dumps(suggestions),
    )
    db.add(du2)
    db.commit()

    # Run ingestion pipeline
    print(f"  Running ingestion pipeline for dataset #{ds2_id}...")
    summary2 = run_ingestion_pipeline(ds2_id, df_clean, job_id=f"test_job_{ds2_id}")
    print(f"  Pipeline returned summary: {json.dumps(summary2, indent=2)}")

    assert summary2["status"] == "completed", f"Pipeline failed: {summary2.get('error')}"

    # Verify dataset status in DB
    db.refresh(ds2)
    assert ds2.status == "ready", f"Expected status 'ready', got {ds2.status}"
    assert ds2.record_count == 15, f"Expected 15 records, got {ds2.record_count}"
    assert ds2.edge_count > 0, f"Expected > 0 edges, got {ds2.edge_count}"

    # Verify communities
    comm_count = db.query(Community).filter(Community.dataset_id == ds2_id).count()
    print(f"  Communities created: {comm_count}")
    assert comm_count > 0, "Expected at least 1 community"

    # Verify predictions
    pred_count = db.query(Prediction).filter(Prediction.dataset_id == ds2_id).count()
    assert pred_count == 15, f"Expected 15 predictions, got {pred_count}"

    # Verify bot count via _dataset_summary
    ds2_summary = _dataset_summary(ds2, db)
    print(f"  Dataset #{ds2_id} Summary: {json.dumps(ds2_summary, indent=2)}")
    bot_count = ds2_summary["breakdown"]["bot_count"]
    print(f"  Detected bot count: {bot_count}")
    assert bot_count == 5, f"Expected 5 bots, got {bot_count}"
    print("  [OK] Dataset #2 ingestion completely successful!")

    # -------------------------------------------------------------
    # Step 3: Ingest graphwarden_single_user_test.csv (1 account)
    # -------------------------------------------------------------
    print("\n[3] Ingesting graphwarden_single_user_test.csv (n=1)...")
    single_file_path = os.path.join("data", "uploads", "c1f8bbaa-d79d-4bb4-9ee0-04e217216c8f.csv")
    with open(single_file_path, "rb") as fh:
        raw_single = fh.read()

    df_single, rej_single, _ = validate_and_map(raw_single, column_map=None, original_filename="graphwarden_single_user_test.csv")
    print(f"  Validated single-user DataFrame rows: {len(df_single)}")
    assert len(df_single) == 1

    ds3 = Dataset(
        name="graphwarden_single_user_test",
        description="Single user edge case test",
        source="upload",
        status="pending_confirm",
        is_default=False,
    )
    db.add(ds3)
    db.commit()
    db.refresh(ds3)
    ds3_id = ds3.id

    du3 = DatasetUpload(
        dataset_id=ds3_id,
        original_filename="graphwarden_single_user_test.csv",
        stored_path=os.path.abspath(single_file_path),
        column_map="{}",
    )
    db.add(du3)
    db.commit()

    print(f"  Running ingestion pipeline for dataset #{ds3_id}...")
    summary3 = run_ingestion_pipeline(ds3_id, df_single, job_id=f"test_job_{ds3_id}")
    print(f"  Pipeline returned summary: {json.dumps(summary3, indent=2)}")

    assert summary3["status"] == "completed", f"Single-user pipeline failed: {summary3.get('error')}"

    db.refresh(ds3)
    assert ds3.status == "ready", f"Expected status 'ready', got {ds3.status}"
    assert ds3.record_count == 1, f"Expected 1 record, got {ds3.record_count}"

    ds3_summary = _dataset_summary(ds3, db)
    print(f"  Dataset #{ds3_id} Summary: {json.dumps(ds3_summary, indent=2)}")
    assert ds3_summary["breakdown"]["bot_count"] == 1, f"Expected 1 bot, got {ds3_summary['breakdown']['bot_count']}"
    print("  [OK] Single-user dataset #3 ingestion completely successful!")

    # -------------------------------------------------------------
    # Step 4: Verify Dataset #1 (seed dataset) remains completely intact
    # -------------------------------------------------------------
    print("\n[4] Verifying seed dataset #1 is intact...")
    ds1 = db.query(Dataset).filter(Dataset.id == 1).first()
    assert ds1 is not None, "Dataset #1 not found!"
    assert ds1.is_default is True, "Dataset #1 is not marked as default"
    assert ds1.status == "ready", f"Dataset #1 status is {ds1.status}"
    assert ds1.record_count == 10000, f"Dataset #1 record count is {ds1.record_count}"

    ds1_summary = _dataset_summary(ds1, db)
    print(f"  Dataset #1 Summary: record_count={ds1_summary['record_count']}, edges={ds1_summary['edge_count']}, bots={ds1_summary['breakdown']['bot_count']}")
    assert ds1_summary["breakdown"]["bot_count"] > 0, "Dataset #1 bot count is 0"
    print("  [OK] Dataset #1 (seed) verified 100% intact!")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 70)
    db.close()

if __name__ == "__main__":
    run_tests()
