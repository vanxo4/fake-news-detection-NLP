import sys
import os
import sqlite3
import time
import pandas as pd
import feedparser
from newspaper import Article, Config

# --- 1. Path Configuration (Manual approach) ---
# We need to go up 3 levels: scrapers -> python -> src -> ROOT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Define paths relative to the calculated root
# We use os.path.dirname logic to ensure stability
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")
SOURCES_CSV = os.path.join(BASE_DIR, "data", "sources.csv")

# Browser User-Agent configuration
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

def init_db():
    """Creates the necessary tables if they don't exist."""
    print(f"🔨 Connecting to database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table: 'articles'
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            source TEXT,
            title TEXT,
            text TEXT,
            authors TEXT,
            publish_date TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_processed INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def save_article_to_db(data):
    """
    Inserts a new article into the database.
    Returns True if saved, False if duplicate.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR IGNORE INTO articles 
            (url, source, title, text, authors, publish_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)
        
        if c.rowcount > 0:
            print(f"   💾 Saved: {data[2][:40]}...")
            conn.commit()
            return True
        else:
            # Uncomment for debugging duplicates
            # print(f"   ⏭️  Skipped (Duplicate): {data[2][:40]}...")
            return False
            
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
        return False
    finally:
        conn.close()

def process_feed(source_name, rss_url):
    """
    Scrapes a single feed. Returns the count of new articles saved.
    """
    print(f"\n📡 Scanning: {source_name}...")
    saved_count = 0
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"   ❌ Network/Parse Error: {e}")
        return 0

    if not feed.entries:
        print("   ⚠️  No entries found in feed.")
        return 0

    print(f"   Found {len(feed.entries)} entries. Processing latest 5...")

    # Newspaper4k Config
    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 10

    # Limit to 5 per feed to avoid bottlenecks in this demo version
    # (Increase this number or remove the slice for full production)
    for entry in feed.entries:
        url = entry.link
        
        try:
            # Download & Parse
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            title = article.title
            text = article.text.strip()
            
            # Clean missing metadata
            authors = ", ".join(article.authors) if article.authors else "Unknown"
            pub_date = str(article.publish_date) if article.publish_date else "Unknown"

            # Quality Filter: Ignore very short texts
            if len(text) > 150:
                article_data = (url, source_name, title, text, authors, pub_date)
                
                if save_article_to_db(article_data):
                    saved_count += 1
            else:
                pass # Silently ignore short content to keep logs clean

            # Polite delay
            time.sleep(0.5)

        except Exception as e:
            # Most common error: 403 Forbidden or 404
            # We just print a small 'x' to signify a skipped item
            print(f"   x Failed: {url} ({str(e)[:50]})")
            
    return saved_count

if __name__ == "__main__":
    print("--- 🦅 MASS HUNTER AGENT STARTED ---")
    
    # 1. Check if CSV exists
    if not os.path.exists(SOURCES_CSV):
        print(f"❌ Error: Source file not found at {SOURCES_CSV}")
        exit()
        
    # 2. Init DB
    init_db()
    
    # 3. Load Sources from CSV
    print("Loading sources from CSV...")
    df_sources = pd.read_csv(SOURCES_CSV)
    print(f"📋 Loaded {len(df_sources)} sources to monitor.")
    
    total_new_articles = 0
    stats = {}
    
    # 4. Main Loop
    for index, row in df_sources.iterrows():
        source_name = row['source_name']
        rss_url = row['url']
        
        # Run the scraper for this source
        new_count = process_feed(source_name, rss_url)
        
        total_new_articles += new_count
        stats[source_name] = new_count
        
    # 5. Final Report
    print("\n" + "="*50)
    print("🏁 MISSION COMPLETE: INGEST REPORT")
    print("="*50)
    print(f"📈 TOTAL NEW ARTICLES: {total_new_articles}")
    print("-" * 50)
    
    # Show only sources that produced data to keep the report clean
    for source, count in stats.items():
        if count > 0:
            print(f"🟢 {source:<25}: {count} new")
            
    print("="*50)
    print(f"Database size updated at: {DB_PATH}")