import pandas as pd
import requests
import json
import re
import time
import os

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"
INPUT_CSV = "data/singapore_airlines_reviews.csv"
OUTPUT_CSV = "data/singapore_airlines_reviews_llm.csv"
SAMPLE_SIZE = 500  # Start with a manageable Golden Sample

def get_llm_sentiment(text):
    """
    Pings the local Ollama API to get a high-nuance sentiment score.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    # Prompt engineering for strict numerical output
    prompt = f"""
    Analyze the emotional sentiment of the following airline review. 
    Output ONLY a single decimal number between -1.0 (extremely negative/hated) 
    and +1.0 (extremely positive/delighted). 
    
    Do NOT provide any explanations, headers, or conversational filler.
    Only output the number.

    Review Content: "{text[:1000]}"
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0  # Force determinism
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("response", "0.0").strip()
        
        # Extract the first float found in the response string (in case it is verbose)
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", raw_text)
        if matches:
            score = float(matches[0])
            return max(-1.0, min(1.0, score))  # Clamp to range
        return 0.0
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return 0.0

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"🚀 Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Shuffle and sample 500 "Golden Sample" reviews
    print(f"🏗️ Preparing Golden Sample of {SAMPLE_SIZE} reviews...")
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).copy()
    
    scores = []
    start_time = time.time()
    
    print(f"📡 Igniting {MODEL_NAME} via Ollama. Estimated time: {SAMPLE_SIZE * 2} seconds...")
    
    for i, row in enumerate(df_sample.itertuples()):
        score = get_llm_sentiment(row.text)
        scores.append(score)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (SAMPLE_SIZE - (i + 1)) * avg_time
            print(f"✨ Processed {i+1}/{SAMPLE_SIZE} | Est. Remaining: {remaining:.1f}s")
            
    df_sample['llm_sentiment_score'] = scores
    
    # Save the enriched sample
    df_sample.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ LLM Feature Enrichment Complete! Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
