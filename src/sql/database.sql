-- ==========================================
-- NEWS LAKE SCHEMA
-- Version: 1.1 (Updated with Verification Columns)
-- ==========================================

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    source TEXT,
    title TEXT,
    text TEXT,
    authors TEXT,
    publish_date TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_processed INTEGER DEFAULT 0,
    -- New columns for verification (v1.1)
    verified_label INTEGER,  -- 0: Real, 1: Fake (from Fact-Checkers)
    label_source TEXT        -- Source of the verification (e.g. "PolitiFact Match")
);

CREATE TABLE IF NOT EXISTS fact_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim TEXT UNIQUE NOT NULL,
    verdict TEXT,
    source_url TEXT,
    checker_site TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    article_id INTEGER,
    model_version TEXT,
    predicted_label INTEGER,
    confidence_score REAL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(article_id) REFERENCES articles(id)
);