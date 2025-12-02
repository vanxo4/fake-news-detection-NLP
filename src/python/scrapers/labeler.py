import sys
import os
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. Path Configuration ---
# Adjust path to find project root manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")

# --- 2. Configuration ---
# Threshold: How similar must the texts be to consider them a match?
# 0.85 is a safe conservative value. 0.75 might catch more but risks errors.
SIMILARITY_THRESHOLD = 0.85

# Model: 'all-MiniLM-L6-v2' is small, fast, and very good for semantic search.
# It runs on CPU easily.
MODEL_NAME = 'all-MiniLM-L6-v2'

def get_unlabeled_articles():
    """Fetch articles from 'articles' table that don't have a label yet."""
    conn = sqlite3.connect(DB_PATH)
    # We select ID and Title (Titles contain the main claim usually)
    query = "SELECT id, title FROM articles WHERE verified_label IS NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_fact_checks():
    """Fetch all claims from the fact-checker database."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, claim, verdict FROM fact_checks"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def map_verdict_to_label(verdict_text):
    """
    Converts PolitiFact text verdicts to our binary system.
    Returns: 1 (FAKE), 0 (REAL), or None (Ambiguous)
    """
    v = verdict_text.lower()
    
    # FAKE indicators
    if any(x in v for x in ['false', 'pants-fire', 'barely-true']):
        return 1
    # REAL indicators
    elif any(x in v for x in ['true', 'mostly-true']):
        return 0
    
    return None # we ignore half-true

def update_article_label(article_id, label, source_info):
    """Updates the article with the Ground Truth label."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE articles 
            SET verified_label = ?, label_source = ?
            WHERE id = ?
        ''', (label, source_info, article_id))
        
        conn.commit()
        print(f"      💾 Tagged Article #{article_id} as: {'FAKE 🚨' if label==1 else 'REAL ✅'}")
        
    except Exception as e:
        print(f"      ❌ DB Error: {e}")
    finally:
        conn.close()

def run_matcher():
    print(f"--- 🧠 SEMANTIC MATCHER AGENT STARTED ---")
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Load Data
    print("Loading data from Data Lake...")
    articles = get_unlabeled_articles()
    facts = get_fact_checks()
    
    if articles.empty:
        print("❌ No pending articles to check.")
        return
    if facts.empty:
        print("❌ No fact-checks available. Run the Judge Agent first.")
        return

    print(f"   Loaded {len(articles)} articles and {len(facts)} fact-checks.")
    
    # 2. Encode Fact-Checks (The Knowledge Base)
    # We do this once because it's our reference library.
    print("Encoding fact-checks to vectors...")
    fact_embeddings = model.encode(facts['claim'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # 3. Iterate & Match
    print("\n🔍 Scanning articles for matches...")
    
    # We iterate articles one by one (or in batches)
    # Ideally, encode all articles first for speed, but loop gives better control for logging
    
    article_embeddings = model.encode(articles['title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    
    # Calculate Cosine Similarity Matrix
    # Result is a matrix of size (num_articles, num_facts)
    cosine_scores = util.cos_sim(article_embeddings, fact_embeddings)
    
    matches_found = 0
    count = 0
    for i in range(len(articles)):
        # Find the best match for this article among all facts
        best_score_tensor = cosine_scores[i].max()
        best_score = float(best_score_tensor)
        
        if best_score > 0.5:
            best_fact_idx = cosine_scores[i].argmax().item()
            article_row = articles.iloc[i]
            fact_row = facts.iloc[best_fact_idx]
            print(f"👀 Potential Match ({best_score:.2f}):")
            print(f"   News: {article_row['title'][:50]}...")
            print(f"   Fact: {fact_row['claim'][:50]}...")
            count += 1

        if best_score >= SIMILARITY_THRESHOLD:
            # Get the index of the best match
            best_fact_idx = int(cosine_scores[i].argmax())
            
            # Extract data
            article_row = articles.iloc[i]
            fact_row = facts.iloc[best_fact_idx]
            
            # Map the text verdict (e.g. "Pants on fire") to number (1)
            numeric_label = map_verdict_to_label(fact_row['verdict'])
            
            if numeric_label is not None:
                print(f"\n✅ MATCH FOUND ({best_score:.2f})")
                print(f"   📰 News:  {article_row['title']}")
                print(f"   ⚖️  Fact:  {fact_row['claim']} -> {fact_row['verdict']}")
                
                # Update the article directly!
                source_info = f"Match with PolitiFact (Sim: {best_score:.2f})"
                update_article_label(article_row['id'], numeric_label, source_info)
                matches_found += 1
            else:
                # Similarity was high, but verdict was ambiguous (e.g. "Half True")
                pass 
                
    print(f"\n--- 🏁 MATCHING COMPLETE ---")
    print(f"Auto-labeled {matches_found} articles based on verified facts.")
    print(f"oissible mastches {count}")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
    else:
        run_matcher()