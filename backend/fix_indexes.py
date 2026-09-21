import sqlite3

def run():
    con = sqlite3.connect("graphwarden.db")
    cur = con.cursor()

    print("Dropping old unique index on communities...")
    cur.execute("DROP INDEX IF EXISTS ix_communities_community_id;")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_communities_community_id ON communities (community_id);")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_communities_dataset_cid ON communities (dataset_id, community_id);")

    print("Dropping old unique index on accounts...")
    cur.execute("DROP INDEX IF EXISTS ix_accounts_user_id_str;")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_accounts_user_id_str ON accounts (user_id_str);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_accounts_dataset_user ON accounts (dataset_id, user_id_str);")

    con.commit()

    print("Updated indexes:")
    for row in cur.execute("SELECT tbl_name, name, sql FROM sqlite_master WHERE type='index' AND tbl_name IN ('communities', 'accounts')"):
        print(" ", row)

    con.close()
    print("Done.")

if __name__ == "__main__":
    run()
