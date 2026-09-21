"""
migrate_schema.py
Applies additive schema changes to the live SQLite database
without destroying existing data.

Run once: python migrate_schema.py
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))

from sqlalchemy import text, inspect
from backend.database import engine, Base

# Import all models so Base.metadata is fully populated
from backend.models import entities  # noqa

def column_exists(inspector, table, col):
    return any(c["name"] == col for c in inspector.get_columns(table))

def table_exists(inspector, table):
    return table in inspector.get_table_names()

def run():
    insp = inspect(engine)

    with engine.begin() as conn:
        # ── datasets ──────────────────────────────────────────────────────
        if table_exists(insp, "datasets"):
            if not column_exists(insp, "datasets", "source"):
                conn.execute(text("ALTER TABLE datasets ADD COLUMN source VARCHAR(50) DEFAULT 'seed'"))
                print("[+] datasets.source added")
            if not column_exists(insp, "datasets", "status"):
                conn.execute(text("ALTER TABLE datasets ADD COLUMN status VARCHAR(50) DEFAULT 'ready'"))
                print("[+] datasets.status added")
            if not column_exists(insp, "datasets", "is_default"):
                conn.execute(text("ALTER TABLE datasets ADD COLUMN is_default BOOLEAN DEFAULT 0"))
                print("[+] datasets.is_default added")
                # Mark the first dataset as default
                conn.execute(text(
                    "UPDATE datasets SET is_default = 1 WHERE id = (SELECT MIN(id) FROM datasets)"
                ))
                print("[+] First dataset marked as is_default=1")
            if not column_exists(insp, "datasets", "ingestion_meta"):
                conn.execute(text("ALTER TABLE datasets ADD COLUMN ingestion_meta TEXT"))
                print("[+] datasets.ingestion_meta added")

        # ── edges ─────────────────────────────────────────────────────────
        if table_exists(insp, "edges"):
            if not column_exists(insp, "edges", "dataset_id"):
                conn.execute(text("ALTER TABLE edges ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)"))
                print("[+] edges.dataset_id added")
                # Back-fill with default dataset id
                conn.execute(text(
                    "UPDATE edges SET dataset_id = (SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL"
                ))
                print("[+] edges.dataset_id back-filled")

        # ── communities ───────────────────────────────────────────────────
        if table_exists(insp, "communities"):
            if not column_exists(insp, "communities", "dataset_id"):
                conn.execute(text("ALTER TABLE communities ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)"))
                print("[+] communities.dataset_id added")
                conn.execute(text(
                    "UPDATE communities SET dataset_id = (SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL"
                ))
                print("[+] communities.dataset_id back-filled")
            # Remove UNIQUE constraint on community_id (SQLite can't drop, but create_all handles new tables)

        # ── coordination_clusters ─────────────────────────────────────────
        if table_exists(insp, "coordination_clusters"):
            if not column_exists(insp, "coordination_clusters", "dataset_id"):
                conn.execute(text("ALTER TABLE coordination_clusters ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)"))
                print("[+] coordination_clusters.dataset_id added")
                conn.execute(text(
                    "UPDATE coordination_clusters SET dataset_id = (SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL"
                ))
                print("[+] coordination_clusters.dataset_id back-filled")

        # ── predictions ───────────────────────────────────────────────────
        if table_exists(insp, "predictions"):
            if not column_exists(insp, "predictions", "dataset_id"):
                conn.execute(text("ALTER TABLE predictions ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)"))
                print("[+] predictions.dataset_id added")
                conn.execute(text(
                    "UPDATE predictions SET dataset_id = (SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL"
                ))
                print("[+] predictions.dataset_id back-filled")

        # ── jobs ──────────────────────────────────────────────────────────
        if table_exists(insp, "jobs"):
            if not column_exists(insp, "jobs", "dataset_id"):
                conn.execute(text("ALTER TABLE jobs ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)"))
                print("[+] jobs.dataset_id added")

    # Create any brand-new tables (DatasetUpload) without touching existing ones
    Base.metadata.create_all(bind=engine)
    print("[+] create_all completed (new tables created, existing tables unchanged)")
    print("[+] Migration finished successfully.")

if __name__ == "__main__":
    run()
