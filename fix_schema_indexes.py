"""
fix_schema_indexes.py

Safely migrates indexes on `communities` and `accounts` in graphwarden.db:
1. Drops global UNIQUE index on communities.community_id and creates non-unique index.
2. Creates composite UNIQUE index on communities (dataset_id, community_id).
3. Drops global UNIQUE index on accounts.user_id_str and creates non-unique index.
4. Creates composite index on accounts (dataset_id, user_id_str).
"""
import sqlite3
import os
import glob

def find_db():
    candidates = ["graphwarden.db", "data/graphwarden.db", "backend/graphwarden.db"]
    candidates += glob.glob("**/*.db", recursive=True)
    for c in candidates:
        if os.path.exists(c) and not c.endswith("-shm") and not c.endswith("-wal"):
            return c
    return "graphwarden.db"

def run_migration():
    db_path = find_db()
    print(f"[+] Connecting to SQLite database: {db_path}")
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()

    # 1. Inspect existing indexes
    indexes = cur.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'").fetchall()
    idx_map = {row[0]: (row[1], row[2] or "") for row in indexes}

    print("\n--- Current Indexes on communities & accounts ---")
    for name, (tbl, sql) in idx_map.items():
        if tbl in ("communities", "accounts"):
            print(f"  {tbl}.{name}: {sql}")

    # 2. Fix communities.community_id
    # If ix_communities_community_id is unique, drop it
    print("\n[+] Re-configuring indexes for 'communities'...")
    cur.execute("DROP INDEX IF EXISTS ix_communities_community_id")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_communities_community_id ON communities (community_id)")
    print("  [OK] Index ix_communities_community_id (non-unique) ensured")

    cur.execute("DROP INDEX IF EXISTS idx_community_dataset_cid")
    cur.execute("DROP INDEX IF EXISTS idx_communities_dataset_cid")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_communities_dataset_cid ON communities (dataset_id, community_id)")
    print("  [OK] Unique composite index idx_communities_dataset_cid (dataset_id, community_id) ensured")

    # 3. Fix accounts.user_id_str
    print("\n[+] Re-configuring indexes for 'accounts'...")
    cur.execute("DROP INDEX IF EXISTS ix_accounts_user_id_str")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_accounts_user_id_str ON accounts (user_id_str)")
    print("  [OK] Index ix_accounts_user_id_str (non-unique) ensured")

    cur.execute("DROP INDEX IF EXISTS idx_accounts_dataset_user")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_accounts_dataset_user ON accounts (dataset_id, user_id_str)")
    print("  [OK] Composite index idx_accounts_dataset_user (dataset_id, user_id_str) ensured")

    con.commit()

    # Verify final index definitions
    print("\n--- Verified Indexes ---")
    verified = cur.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND tbl_name IN ('communities', 'accounts')"
    ).fetchall()
    for name, tbl, sql in verified:
        print(f"  {tbl}.{name}: {sql}")

    con.close()
    print("\n[OK] Schema index migration completed successfully.")

if __name__ == "__main__":
    run_migration()
