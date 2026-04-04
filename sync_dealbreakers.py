import pandas as pd
import json
import os
import re

# --- CONFIGURATION ---
DATA_PATH = "data/singapore_airlines_reviews_core4.csv"
DB_PATH = "models/dealbreaker_words.json"

def clean_text_local(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def has_db(text, word_set):
    words = set(str(text).split())
    return 1 if words & word_set else 0

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    print(f"🔄 Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Load Dealbreakers
    try:
        with open(DB_PATH, "r") as f:
            db_data = json.load(f)
            # Use negative dealbreaker categories (1 and 2 in your system)
            neg_db = set(db_data.get("1", []) + db_data.get("2", []))
            print(f"📡 Loaded {len(neg_db)} negative dealbreaker keywords.")
    except Exception as e:
        print(f"⚠️ Error loading dealbreakers: {e}")
        return

    # Ensure clean_text exists for synchronization
    if 'clean_text' not in df.columns or df['clean_text'].isnull().any():
        print("🏗️ Re-generating clean text for sync...")
        df['clean_text'] = df['text'].apply(clean_text_local)

    # RUN SYNC
    print("🚦 Synchronizing 'Pain Point' sensors (has_negative_dealbreaker)...")
    df['has_negative_dealbreaker'] = df['clean_text'].apply(lambda t: has_db(t, neg_db))
    
    hits = df['has_negative_dealbreaker'].sum()
    print(f"✅ Sync Complete. Total Dealbreaker Hits: {hits} ({ (hits/len(df)*100):.1f}%)")

    # SAVE
    df.to_csv(DATA_PATH, index=False)
    print(f"💾 Dataset updated and saved to: {DATA_PATH}")

if __name__ == "__main__":
    main()
