import pandas as pd
import requests
import json
import re
import time
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"
INPUT_CSV = "data/singapore_airlines_reviews.csv"
OUTPUT_CSV = "data/singapore_airlines_reviews_core4.csv"
DEALBREAKER_PATH = "models/dealbreaker_words.json"
SAVE_INTERVAL = 10  # Reduced for the 100-review test batch
TEST_LIMIT = 100    # Added to handle the user's specific request

# --- NLP INITIALIZATION ---
from app.utils import clean_text as clean_text_local, get_vader_min, has_dealbreaker as has_db, get_llm_sentiment

# --- ENGINE HELPER IMPORTS COMPLETED ---

# --- MAIN ENGINE ---
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    # 1. LOAD DATA
    if os.path.exists(OUTPUT_CSV):
        print(f"🔄 Resuming from existing Core Four File: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        print(f"🚀 Initializing new Core Four File from: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        # Initialize strictly the Core Four columns
        for col in ["clean_text", "vader_min", "has_negative_dealbreaker", "llm_sentiment_score"]:
            if col not in df.columns: df[col] = 0.0

    # 2. RUN FAST FEATURES ('Core Four' Standard)
    if 'clean_text' not in df.columns or df['clean_text'].iloc[0] == 0.0 or any(pd.isna(df['clean_text'])):
        print("🏗️ Generating 'Core Four' Fast Features (Pain-Point & Dealbreakers)...")
        # Load Dealbreakers (Negative only for Core Four)
        try:
            with open(DEALBREAKER_PATH, "r") as f:
                db_data = json.load(f)
                neg_db = set(db_data.get("1", []) + db_data.get("2", []))
        except:
            neg_db = set()
            print("⚠️ Warning: dealbreaker_words.json missing. Skipping DB flags.")

        df['clean_text'] = df['text'].apply(clean_text_local)
        df['vader_min'] = df['text'].fillna("").apply(get_vader_min)
        df['has_negative_dealbreaker'] = df['clean_text'].apply(lambda t: has_db(t, neg_db))
        
        # Initial Save
        df.to_csv(OUTPUT_CSV, index=False)
        print("✅ Core Four Fast Features Complete & Saved.")

    # 3. RUN SLOW FEATURES (Ollama AI Audit with Checkpointing)
    # Find rows where LLM Score is missing (using 0.0 as marker)
    # We only process where rating != 0 and text isn't empty
    to_process = df[(df['llm_sentiment_score'] == 0.0) & (df['text'].notna())]
    
    if to_process.empty:
        print("🏁 All 28k rows are fully LLM-enriched. Nothing to do!")
        return

    print(f"📡 {len(to_process)} reviews remaining to be processed by Llama-3.")
    print("🚦 Use 'Ctrl+C' to Pause anytime. Progress is saved every 25 entries.")
    
    start_time = time.time()
    count = 0
    
    try:
        for idx in to_process.index:
            text = df.at[idx, 'text']
            llm_val = get_llm_sentiment(text)
            df.at[idx, 'llm_sentiment_score'] = llm_val
            
            count += 1
            if count % SAVE_INTERVAL == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                elapsed = time.time() - start_time
                avg = elapsed / count
                remaining = (len(to_process) - count) * avg
                status_msg = f"✨ Progress: {count}/{len(to_process)} ({ (count/len(to_process)*100):.1f}%) | Speed: {avg:.2f}s/rev | Est. Left: {remaining/60:.1f}m"
                print(status_msg)
                # Write to Live Status File
                with open("enrichment_status.txt", "w") as f_status:
                    f_status.write(f"{status_msg} | Last Update: {time.strftime('%H:%M:%S')}")
                
        # Final Save
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Full Enrichment Complete! Saved to: {OUTPUT_CSV}")

    except KeyboardInterrupt:
        print("\n⏸️ Pausing... Saving current progress to CSV...")
        df.to_csv(OUTPUT_CSV, index=False)
        print("💾 Progress saved. Re-run this script anytime to resume.")

if __name__ == "__main__":
    main()
