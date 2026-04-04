import pandas as pd
import pickle
import json
import os
import numpy as np

# --- CONFIGURATION ---
DATA_PATH = "data/singapore_airlines_reviews_+9columns.csv"
MODELS_DIR = "models_exp"
OUTPUT_METRICS = "data/exp_metrics_4k.json"

def load_committee(prefix):
    models = {}
    for name in ["lr", "rf", "svm"]:
        with open(f"{MODELS_DIR}/{prefix}_{name}_model.pkl", 'rb') as f:
            models[name] = pickle.load(f)
    return models

def get_committee_predictions(models, X, cols):
    # Mean of predictions (for simpler consensus logic in metrics)
    preds = []
    for name, m in models.items():
        preds.append(m.predict(X[cols]))
    # Take majority vote
    stacked = np.stack(preds, axis=1)
    consensus = [max(set(row), key=list(row).count) for row in stacked]
    return np.array(consensus)

def main():
    print("📈 Calculating Performance Delta & Feature Importance...")
    df = pd.read_csv(DATA_PATH)
    df_4k = df[df['llm_sentiment_score'] != 0.0].head(4000).copy()
    
    base_comm = load_committee("base")
    boost_comm = load_committee("boost")
    
    base_cols = [base_comm['lr'].named_steps['pre'].transformers[0][2]] + ['vader_score', 'vader_class', 'vader_min', 'vader_max', 'vader_range', 'has_negative_dealbreaker', 'has_pos_dealbreaker']
    boost_cols = base_cols + ['llm_sentiment_score']
    
    y_actual = df_4k['rating'].astype(int).values
    
    print("   - Running Inference for Committee A (8-Col)...")
    y_base = get_committee_predictions(base_comm, df_4k, base_cols)
    
    print("   - Running Inference for Committee B (9-Col)...")
    y_boost = get_committee_predictions(boost_comm, df_4k, boost_cols)
    
    # CALCULATE SARCASM RESOLUTION
    # Definition: Baseline was WRONG, but Boosted was RIGHT
    fixed_mask = (y_base != y_actual) & (y_boost == y_actual)
    total_fixed = int(fixed_mask.sum())
    
    # Definition: Baseline was RIGHT, but Boosted was WRONG (The "Collateral Damage")
    broken_mask = (y_base == y_actual) & (y_boost != y_actual)
    total_broken = int(broken_mask.sum())
    
    # CONFUSED PROMOTERS (Sarcasm specific subset)
    # Filter for Rating 1-2, VADER > 0, LLM < 0
    cp_mask = (df_4k['rating'] <= 2) & (df_4k['vader_score'] > 0.1) & (df_4k['llm_sentiment_score'] < -0.1)
    cp_count = int(cp_mask.sum())
    cp_fixed = int(((y_base[cp_mask] != y_actual[cp_mask]) & (y_boost[cp_mask] == y_actual[cp_mask])).sum())
    
    # FEATURE IMPORTANCE (From Random Forest)
    rf_model = boost_comm['rf']
    importances = rf_model.named_steps['clf'].feature_importances_
    
    # Get feature names from preprocessor
    # Note: Tfidf returns 2500, we aggregate them for the chart
    feature_names = []
    # TF-IDF
    tfidf_count = len(rf_model.named_steps['pre'].transformers_[0][1].get_feature_names_out())
    # Num
    num_features = rf_model.named_steps['pre'].transformers_[1][2]
    # OHE
    ohe_features = list(rf_model.named_steps['pre'].transformers_[2][1].get_feature_names_out())
    # Flags
    flag_features = rf_model.named_steps['pre'].transformers_[3][2]
    # LLM
    llm_feature = ['llm_sentiment_score']
    
    # Grouped Importances
    tfidf_imp = float(np.sum(importances[:tfidf_count]))
    vader_imp = float(np.sum(importances[tfidf_count : tfidf_count+len(num_features)]))
    ohe_imp = float(np.sum(importances[tfidf_count+len(num_features) : tfidf_count+len(num_features)+len(ohe_features)]))
    flag_imp = float(np.sum(importances[tfidf_count+len(num_features)+len(ohe_features) : -1]))
    llm_imp = float(importances[-1])
    
    importance_data = [
        {"Feature": "Llama-3 Score (Nuance)", "Importance": llm_imp},
        {"Feature": "Text Context (TF-IDF)", "Importance": tfidf_imp},
        {"Feature": "VADER Scores (Emotion)", "Importance": vader_imp},
        {"Feature": "Dealbreaker Flags", "Importance": flag_imp},
        {"Feature": "VADER Categories", "Importance": ohe_imp}
    ]
    
    metrics = {
        "sarcasm_resolution": {
            "total_reviews": 4000,
            "net_accurate_gain": int(total_fixed - total_broken),
            "total_fixed": total_fixed,
            "total_broken": total_broken,
            "confused_promoters_fixed": f"{cp_fixed}/{cp_count} ({ (cp_fixed/cp_count*100):.1f}% )" if cp_count > 0 else "0/0"
        },
        "feature_importance": sorted(importance_data, key=lambda x: x['Importance'], reverse=True)
    }
    
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Performance Delta & Feature MVP report saved to: {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
