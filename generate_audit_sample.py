import pandas as pd
import pickle
import json
import os

# --- CONFIGURATION ---
DATA_PATH = "data/singapore_airlines_reviews_+9columns.csv"
MODELS_DIR = "models_exp"
OUTPUT_SAMPLE = "data/audit_sample_4k.csv"

def load_committee(prefix):
    models = {}
    for name in ["lr", "rf", "svm"]:
        with open(f"{MODELS_DIR}/{prefix}_{name}_model.pkl", 'rb') as f:
            models[name] = pickle.load(f)
    return models

def get_consensus(models, X):
    # Winner takes all voting
    votes = []
    for name, m in models.items():
        votes.append(int(m.predict(X)[0]))
    return max(set(votes), key=votes.count), votes

def main():
    print("🧠 Selecting Audit Samples (Sarcasm & Nuance Detection)...")
    df = pd.read_csv(DATA_PATH)
    df_audit = df[df['llm_sentiment_score'] != 0.0].head(4000).copy()
    
    # Target: Reviews where VADER and LLM disagree (Strong Sarcasm indicator)
    df_audit['sentiment_delta'] = (df_audit['vader_score'] - df_audit['llm_sentiment_score']).abs()
    # Sort by delta to find the most "Confusing" reviews for a traditional model
    top_nuance = df_audit.sort_values(by='sentiment_delta', ascending=False).head(20).copy()
    
    base_comm = load_committee("base")
    boost_comm = load_committee("boost")
    
    audit_rows = []
    
    for idx in top_nuance.index:
        row = top_nuance.loc[idx]
        X_row = top_nuance.loc[[idx]] # Single row as DataFrame
        
        # Committee A (Base)
        base_cols = [base_comm['lr'].named_steps['pre'].transformers[0][2]] + ['vader_score', 'vader_class', 'vader_min', 'vader_max', 'vader_range', 'has_negative_dealbreaker', 'has_pos_dealbreaker']
        base_verdict, base_votes = get_consensus(base_comm, X_row[base_cols])
        
        # Committee B (Boost)
        boost_cols = [boost_comm['lr'].named_steps['pre'].transformers[0][2]] + ['vader_score', 'vader_class', 'vader_min', 'vader_max', 'vader_range', 'has_negative_dealbreaker', 'has_pos_dealbreaker', 'llm_sentiment_score']
        boost_verdict, boost_votes = get_consensus(boost_comm, X_row[boost_cols])
        
        audit_rows.append({
            "Review Snippet": row['text'][:150] + "...",
            "VADER Score": f"{row['vader_score']:.2f}",
            "Llama-3 Score": f"{row['llm_sentiment_score']:.2f}",
            "Base Committee (Votes)": f"{base_votes}",
            "Boost Committee (Votes)": f"{boost_votes}",
            "Base Verdict": base_verdict,
            "Boost Verdict": boost_verdict,
            "Actual Rating": int(row['rating']),
            "Improved?": "✅" if boost_verdict == int(row['rating']) and base_verdict != int(row['rating']) else ("❌" if boost_verdict != int(row['rating']) and base_verdict == int(row['rating']) else "-")
        })
        
    df_output = pd.DataFrame(audit_rows)
    df_output.to_csv(OUTPUT_SAMPLE, index=False)
    print(f"✅ Audit Sample of 20 high-nuance reviews saved to: {OUTPUT_SAMPLE}")

if __name__ == "__main__":
    main()
