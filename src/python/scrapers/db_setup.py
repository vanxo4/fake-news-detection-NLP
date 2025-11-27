import sqlite3
import sys
import os

# Adjust path to find project root manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Paths
DB_PATH = "data/database.db"
SCHEMA_PATH = "src/sql/database.sql"

def init_db():
    print(f"🔨 Initializing database at: {DB_PATH}")
    
    try:
        with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
            sql_script = f.read()
            print("📜 SQL schema read successfully.")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {SCHEMA_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.executescript(sql_script)
        conn.commit()
        print("✅ Success!")
        
    except sqlite3.Error as e:
        print(f"❌ SQLite Error: {e}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()