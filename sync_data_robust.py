import csv
import json
import os
import re

# --- CONFIGURATION ---
DATA_PATH = "data/singapore_airlines_reviews_core4.csv"
DB_PATH = "models/dealbreaker_words.json"
TEMP_PATH = "data/temp_sync.csv"

def clean_text_local(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def has_db(text, word_set):
    words = set(str(text).split())
    return '1' if words & word_set else '0'

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    # Load Dealbreakers
    try:
        with open(DB_PATH, "r") as f:
            db_data = json.load(f)
            neg_db = set(db_data.get("1", []) + db_data.get("2", []))
            print(f"📡 Loaded {len(neg_db)} negative dealbreaker keywords.")
    except Exception as e:
        print(f"⚠️ Error loading dealbreakers: {e}")
        return

    hits = 0
    total = 0

    print("🚦 Synchronizing Data Layer (No-Dependency Mode)...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f_in, \
         open(TEMP_PATH, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            # Build clean text if missing or for sync
            clean = clean_text_local(row.get('text', ''))
            val = has_db(clean, neg_db)
            row['has_negative_dealbreaker'] = val
            if val == '1': hits += 1
            writer.writerow(row)

    # Replace original with synced version
    os.replace(TEMP_PATH, DATA_PATH)
    
    print(f"✅ Final Sync Complete. Total Hits: {hits} / {total} ({ (hits/total*100):.1f}%)")
    print(f"💾 Dataset secured at: {DATA_PATH}")

if __name__ == "__main__":
    main()
