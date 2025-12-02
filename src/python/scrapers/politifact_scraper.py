import sys
import os
import sqlite3
import time
import requests
from bs4 import BeautifulSoup

# --- 1. Path Configuration ---
# Adjust path to find project root manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")

# --- 2. Configuration ---
# Target: PolitiFact's fact-check list
BASE_URL = "https://www.politifact.com/factchecks/list/?page={}"
PAGES_TO_SCRAPE = 20  # Start with 5 pages (approx 100 facts)

def save_fact_check(data):
    """Inserts the fact-check into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Table 'fact_checks' must exist (created by db_setup.py)
        c.execute('''
            INSERT OR IGNORE INTO fact_checks 
            (claim, verdict, source_url, checker_site)
            VALUES (?, ?, ?, ?)
        ''', data)
        
        if c.rowcount > 0:
            print(f"   ⚖️  Verdict saved: [{data[1]}] {data[0][:40]}...")
            conn.commit()
        else:
            # Duplicate found
            pass
            
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
    finally:
        conn.close()

def scrape_politifact_page(page_number):
    """Scrapes a single pagination page from PolitiFact."""
    url = BASE_URL.format(page_number)
    print(f"\n📄 Analyzing page {page_number}: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"   ❌ Error {response.status_code}: Could not fetch page.")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all fact-check items (PolitiFact structure as of 2024/2025)
        # They usually use <li class="o-listicle__item">
        items = soup.find_all('li', class_='o-listicle__item')
        
        if not items:
            print("   ⚠️ No items found. Did the website structure change?")
            return

        count = 0
        for item in items:
            try:
                # 1. Extract The Statement (Claim)
                quote_div = item.find('div', class_='m-statement__quote')
                if not quote_div: continue
                
                link_tag = quote_div.find('a')
                claim_text = link_tag.text.strip()
                full_link = "https://www.politifact.com" + link_tag['href']
                
                # 2. Extract The Verdict (True/False/etc)
                # It's usually in an image alt tag or class
                meter_div = item.find('div', class_='m-statement__meter')
                if meter_div:
                    img_tag = meter_div.find('img')
                    verdict_raw = img_tag.get('alt', 'Unknown')
                    
                    # Clean the verdict (e.g. "true" -> "TRUE", "pants-fire" -> "FAKE")
                    verdict = verdict_raw.lower()
                else:
                    verdict = "unknown"

                # 3. Store Data
                data = (claim_text, verdict, full_link, "PolitiFact")
                save_fact_check(data)
                count += 1
                
            except Exception as e:
                print(f"   ⚠️ Parsing error on item: {e}")
                continue
                
        print(f"   ✅ Extracted {count} facts from page {page_number}.")

    except Exception as e:
        print(f"   ❌ Network error: {e}")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        print("Run 'db_setup.py' first.")
    else:
        print("--- 🦅 JUDGE AGENT STARTED (PolitiFact) ---")
        
        for i in range(1, PAGES_TO_SCRAPE + 1):
            scrape_politifact_page(i)
            time.sleep(1) # Be polite to the server
            
        print("\n--- 🏁 JUDGEMENT DAY COMPLETE ---")
        print("Check the 'fact_checks' table in your database.")