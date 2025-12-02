import sys
import os
import sqlite3
import pandas as pd
import joblib

# --- 1. Path Configuration (Manual) ---
# Subimos 2 niveles: src/python -> src -> ROOT
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bow_logit_v1.joblib")

# --- 2. Load Model ---
print(f"🧠 Loading Brain from: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model not found. Train it first!")
    exit()

def get_pending_articles():
    """Fetches articles that haven't been analyzed yet."""
    conn = sqlite3.connect(DB_PATH)
    # Leemos solo lo necesario: ID, Título y Texto
    query = "SELECT id, title, text, source FROM articles WHERE is_processed = 0"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def save_predictions(df_results):
    """Saves predictions to the 'predictions' table and updates 'articles'."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print(f"💾 Saving {len(df_results)} predictions to database...")
    
    try:
        # 1. Insertar en tabla de predicciones
        for index, row in df_results.iterrows():
            c.execute('''
                INSERT INTO predictions 
                (article_id, model_version, predicted_label, confidence_score)
                VALUES (?, ?, ?, ?)
            ''', (row['id'], 'bow_logit_v1', row['pred_label'], row['confidence']))
            
            # 2. Marcar artículo como procesado
            c.execute('UPDATE articles SET is_processed = 1 WHERE id = ?', (row['id'],))
            
        conn.commit()
        print("✅ Database updated.")
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        conn.rollback()
    finally:
        conn.close()

# --- 3. Main Execution ---
if __name__ == "__main__":
    print("--- 🔮 LAKE PREDICTOR STARTED ---")
    
    # 1. Get Data
    print("🎣 Fishing for new articles in the lake...")
    df = get_pending_articles()
    
    if df.empty:
        print("💤 No pending articles found. Run the Hunter first!")
        exit()
        
    print(f"   Found {len(df)} new articles to analyze.")
    
    # 2. Predict
    print("⚡ Analyzing content...")
    # El pipeline se encarga de vectorizar el texto crudo
    predictions = model.predict(df['text'])
    probs = model.predict_proba(df['text'])
    
    # 3. Process Results
    # Extraemos la confianza de la clase ganadora
    confidences = [probs[i][pred] for i, pred in enumerate(predictions)]
    
    # Añadimos columnas al DataFrame temporal
    df['pred_label'] = predictions
    df['confidence'] = confidences
    
    # 4. Show Report (Preview)
    print("\n" + "="*80)
    print(f"{'SOURCE':<15} | {'VERDICT':<8} | {'CONF':<6} | {'HEADLINE'}")
    print("="*80)
    
    fake_count = 0
    
    for i, row in df.head(15).iterrows(): # Mostramos solo las primeras 15
        label = "FAKE 🚨" if row['pred_label'] == 1 else "REAL ✅"
        conf = row['confidence']
        title = (row['title'][:45] + '..') if len(row['title']) > 45 else row['title']
        
        if row['pred_label'] == 1: fake_count += 1
        
        print(f"{row['source'][:15]:<15} | {label:<8} | {conf:.0%}   | {title}")
        
    print("="*80)
    print(f"📊 SUMMARY: Found {fake_count} potential FAKES out of {len(df)} articles.")
    
    # 5. Save to DB
    save_predictions(df)