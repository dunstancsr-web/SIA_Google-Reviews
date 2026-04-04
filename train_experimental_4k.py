import pandas as pd
import pickle
import json
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATA_PATH = "data/singapore_airlines_reviews_+9columns.csv"
AUDIT_TARGET_COUNT = 4000
MODELS_DIR = "models_exp" # Separate directory for experimental sandbox models

os.makedirs(MODELS_DIR, exist_ok=True)

def train_committee(df, feature_cols, prefix):
    """
    Trains a committee of 3 models (LR, RF, SVM) on a specific set of features.
    Saves models to the MODELS_DIR with the given prefix.
    """
    X = df[feature_cols]
    y = df['rating'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessor 
    preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(max_features=2500, stop_words='english'), 'clean_text'),
        ('num', 'passthrough', ['vader_score', 'vader_min', 'vader_max', 'vader_range']),
        ('class', OneHotEncoder(handle_unknown='ignore'), ['vader_class']),
        ('flags', 'passthrough', ['has_negative_dealbreaker', 'has_pos_dealbreaker'])
    ])
    
    # Add LLM feature if provided
    if 'llm_sentiment_score' in feature_cols:
        preprocessor.transformers.append(('llm', 'passthrough', ['llm_sentiment_score']))
        
    models = {
        "lr": LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42),
        "svm": SVC(probability=True, kernel='linear', C=0.5, class_weight='balanced', random_state=42)
    }
    
    results = {}
    for name, clf in models.items():
        print(f"   - Training {prefix}_{name}...")
        pipe = Pipeline([
            ('pre', preprocessor),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        # Save model
        with open(f"{MODELS_DIR}/{prefix}_{name}_model.pkl", 'wb') as f:
            pickle.dump(pipe, f)
            
        # Save metadata
        meta = {
            "accuracy": float(acc),
            "features": feature_cols,
            "trained_at": "Experimental Audit 4k"
        }
        with open(f"{MODELS_DIR}/{prefix}_{name}_meta.json", 'w') as f:
            json.dump(meta, f)
            
    return results

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    print(f"🚀 Loading Data for the 4K Audit Sandbox...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter for processed rows where llm_sentiment_score != 0
    df_audit = df[df['llm_sentiment_score'] != 0.0].head(AUDIT_TARGET_COUNT).copy()
    
    if len(df_audit) < AUDIT_TARGET_COUNT:
        print(f"⚠️ Warning: Only {len(df_audit)} enriched records found. Using what we have.")
    else:
        print(f"✅ Extracted {len(df_audit)} reviews for experimental audit.")

    # 1. TRAIN COMMITTEE A: BASELINE (8 Features)
    print("\n🏗️ Training Committee A: Baseline (VADER + TF-IDF)...")
    base_features = [
        'clean_text', 'vader_score', 'vader_class', 
        'vader_min', 'vader_max', 'vader_range', 
        'has_negative_dealbreaker', 'has_pos_dealbreaker'
    ]
    base_acc = train_committee(df_audit, base_features, "base")

    # 2. TRAIN COMMITTEE B: BOOSTED (9 Features)
    print("\n🚀 Training Committee B: Boosted (Llama-3 Integration)...")
    boost_features = base_features + ['llm_sentiment_score']
    boost_acc = train_committee(df_audit, boost_features, "boost")

    # 3. SUMMARY REPORT
    print("\n" + "="*40)
    print("EXPERIMENTAL AUDIT (A/B TESTING) RESULTS")
    print("="*40)
    print(f"{'Model':<15} | {'Baseline (8)':<15} | {'Boosted (9)':<15} | {'Lift':<10}")
    print("-" * 60)
    
    for name in base_acc:
        b_acc = base_acc[name]
        bt_acc = boost_acc[name]
        lift = bt_acc - b_acc
        print(f"{name.upper():<15} | {b_acc:.2%}         | {bt_acc:.2%}         | {lift:+.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
