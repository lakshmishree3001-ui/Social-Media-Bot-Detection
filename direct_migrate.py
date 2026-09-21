"""
direct_migrate.py - direct SQLite migration bypassing SQLAlchemy locking issues
"""
import sqlite3
import os
import glob

# Find the database
candidates = ["graphwarden.db", "data/graphwarden.db", "backend/graphwarden.db"]
candidates += glob.glob("**/*.db", recursive=True)
db_path = None
for c in candidates:
    if os.path.exists(c):
        db_path = c
        break

if not db_path:
    print("ERROR: Could not find .db file")
    exit(1)

print(f"Using database: {db_path}")
con = sqlite3.connect(db_path, timeout=30)
con.execute("PRAGMA journal_mode=WAL")

def add_col(table, col, defn):
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
        print(f"  [+] {table}.{col} added")
    except Exception as e:
        print(f"  [~] {table}.{col}: {e}")

def safe_exec(sql, *args):
    try:
        con.execute(sql, args if args else ())
    except Exception as e:
        print(f"  [!] {sql[:60]}: {e}")

# datasets
add_col("datasets", "source", "VARCHAR(50) DEFAULT 'seed'")
add_col("datasets", "status", "VARCHAR(50) DEFAULT 'ready'")
add_col("datasets", "is_default", "BOOLEAN DEFAULT 0")
add_col("datasets", "ingestion_meta", "TEXT")
safe_exec("UPDATE datasets SET is_default=1 WHERE id=(SELECT MIN(id) FROM datasets)")
safe_exec("UPDATE datasets SET source='seed' WHERE source IS NULL")
safe_exec("UPDATE datasets SET status='ready' WHERE status IS NULL")
print("  [+] datasets defaults set")

# edges
add_col("edges", "dataset_id", "INTEGER REFERENCES datasets(id)")
safe_exec("UPDATE edges SET dataset_id=(SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL")
print("  [+] edges.dataset_id backfilled")

# communities
add_col("communities", "dataset_id", "INTEGER REFERENCES datasets(id)")
safe_exec("UPDATE communities SET dataset_id=(SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL")
print("  [+] communities.dataset_id backfilled")

# coordination_clusters
add_col("coordination_clusters", "dataset_id", "INTEGER REFERENCES datasets(id)")
safe_exec("UPDATE coordination_clusters SET dataset_id=(SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL")
print("  [+] coordination_clusters.dataset_id backfilled")

# predictions
add_col("predictions", "dataset_id", "INTEGER REFERENCES datasets(id)")
safe_exec("UPDATE predictions SET dataset_id=(SELECT MIN(id) FROM datasets) WHERE dataset_id IS NULL")
print("  [+] predictions.dataset_id backfilled")

# jobs
add_col("jobs", "dataset_id", "INTEGER REFERENCES datasets(id)")

# dataset_uploads
safe_exec("""CREATE TABLE IF NOT EXISTS dataset_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    original_filename VARCHAR(255),
    stored_path VARCHAR(500),
    column_map TEXT,
    rejected_count INTEGER DEFAULT 0,
    rejected_rows TEXT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
print("  [+] dataset_uploads table ready")

con.commit()
con.close()
print("\nMigration complete.")
