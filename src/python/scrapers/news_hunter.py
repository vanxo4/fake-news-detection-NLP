################################################################################
# HUNTER AGENT: RSS & Web Scraper
# ------------------------------------------------------------------------------
# This script acts as the primary data ingestion engine.
# 1. Fetches latest URLs from RSS feeds (Real, Satire, and Mixed sources).
# 2. Uses 'newspaper4k' to download and parse the full article content.
# 3. Stores the clean data into the SQLite Data Lake.
################################################################################

import feedparser
import sqlite3
import time
from newspaper import Article, Config

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# --- Configuration ---
# Path to the Data Lake (SQLite Database)
DB_PATH = "data/database.db"

# List of RSS Feeds to monitor
# We include diverse sources to build a robust dataset:
# - Mainstream (Reuters, NYT, BBC) -> Likely TRUE
# - Satire/Parody (The Onion, Babylon Bee) -> Likely FAKE (Style-wise)
# - Sensationalist (Daily Mail) -> Hard examples
RSS_SOURCES = {
    'Reuters World': 'http://feeds.reuters.com/reuters/worldNews',
    'NY Times World': 'https://rss.nytimes.com/services/xml/rss/nyt/World.xml',
    'BBC News': 'http://feeds.bbci.co.uk/news/world/rss.xml',
    'The Onion (Satire)': 'https://www.theonion.com/rss',
    'Babylon Bee (Satire)': 'https://babylonbee.com/feed', 
    'Daily Mail': 'https://www.dailymail.co.uk/news/index.rss'
}

# Browser User-Agent configuration to avoid being blocked by anti-bot systems
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# --- Database Initialization ---

def init_db():
    """Creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table: 'articles'
    # Stores the raw scraped data. URL is unique to prevent duplicates.
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
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

# --- Core Functions ---

def save_article_to_db(data):
    """
    Inserts a new article into the database.
    Ignores the insert if the URL already exists (deduplication).
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # data tuple: (url, source, title, text, authors, publish_date)
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
            print(f"   ⏭️  Skipped (Duplicate): {data[2][:40]}...")
            return False
            
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
        return False
    finally:
        conn.close()

def process_feed(source_name, rss_url):
    """
    Orchestrates the scraping process for a single RSS feed.
    """
    print(f"\n📡 Scanning Feed: {source_name}...")
    
    saved_count = 0

    # 1. Parse RSS to get links
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"   ❌ Error reaching RSS: {e}")
        return

    print(f"   Found {len(feed.entries)} entries. Processing latest 5...")

    # Configure Newspaper4k
    config = Config()
    config.browser_user_agent = USER_AGENT
    config.request_timeout = 10

    # Process only the latest 5 entries to be polite and fast
    for entry in feed.entries[:5]:
        url = entry.link
        
        try:
            # 2. Download & Parse with Newspaper4k
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            # 3. Extract Fields
            title = article.title
            text = article.text.strip()
            
            # Handle missing metadata gracefully
            authors = ", ".join(article.authors) if article.authors else "Unknown"
            pub_date = str(article.publish_date) if article.publish_date else "Unknown"

            # Filter: Ignore empty articles or video-only pages
            if len(text) > 150:
                article_data = (url, source_name, title, text, authors, pub_date)
                if(save_article_to_db(article_data)):
                    saved_count += 1
            else:
                print(f"   ⚠️  Content too short (<150 chars), discarded.")

            # Polite delay between requests
            time.sleep(1)

        except Exception as e:
            print(f"   ❌ Scrape Error ({url}): {e}")
    return saved_count

# --- Main Execution ---

if __name__ == "__main__":
    print("--- SCRAPPER STARTED ---")
    print(f"Database: {DB_PATH}")
    
    # 1. Ensure DB exists
    init_db()
    
    stats = {}
    total_saved = 0

    # 2. Run the cycle
    for name, url in RSS_SOURCES.items():
        n_saved = process_feed(name, url)
        
        # Guardamos el dato
        stats[name] = n_saved
        total_saved += n_saved
    
    print("="*50)
    print(f"📈 TOTAL NEW ARTICLES ADDED: {total_saved}")
    print(stats)

        
    