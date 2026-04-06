import pandas as pd
import numpy as np
import re
import pickle
import os
import json
import nltk
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

os.makedirs('models/baseline', exist_ok=True)
os.makedirs('./nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')
nltk.download('vader_lexicon', download_dir='./nltk_data', quiet=True)
nltk.download('stopwords', download_dir='./nltk_data', quiet=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def get_vader_min(text, analyzer):
    sentences = re.split(r'[.!?]', str(text))
    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences if s.strip()]
    return min(scores) if scores else 0.0

def has_dealbreaker(text, word_set):
    words = set(text.split())
    return 1 if words & word_set else 0

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading Baseline data (Aggressive Undersampling)...")
    MASTER_PATH = 'data/singapore_airlines_reviews_core4.csv'
    if os.path.exists(MASTER_PATH):
        df = pd.read_csv(MASTER_PATH)
    else:
        df = pd.read_csv('data/singapore_airlines_reviews.csv')
    df = df.dropna(subset=['text', 'rating'])
    
    # --- ORIGINAL AGGRESSIVE UNDERSAMPLING (THE ACCURACY CHOKEPOINT) ---
    df_1 = df[df['rating'] == 1]
    df_2 = df[df['rating'] == 2]
    df_3 = df[df['rating'] == 3]
    df_4 = df[df['rating'] == 4].sample(n=min(1200, len(df[df['rating'] == 4])), random_state=42)
    df_5 = df[df['rating'] == 5].sample(n=min(1500, len(df[df['rating'] == 5])), random_state=42)
    
    # Oversample 2★ slightly as per original code
    if len(df_2) < 900:
        df_2_extra = df_2.sample(n=900 - len(df_2), replace=True, random_state=42)
        df_2 = pd.concat([df_2, df_2_extra])
    
    df_bal = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42)
    print(f"Baseline Data Size: {len(df_bal)}")

    # Features
    analyzer = SentimentIntensityAnalyzer()
    df_bal['clean_text'] = df_bal['text'].apply(clean_text)
    df_bal['vader_min'] = df_bal['text'].apply(lambda x: get_vader_min(x, analyzer))
    
    stop_words = list(nltk.corpus.stopwords.words('english')) + ['singapore', 'airlines', 'airline', 'flight']
    
    # Pass 1: Extract words
    X_p1 = df_bal[['clean_text', 'vader_min']]
    y = df_bal['rating'].astype(int)
    X_train_p, X_test_p, y_train, y_test = train_test_split(X_p1, y, test_size=0.2, random_state=42, stratify=y)
    
    prep_p1 = ColumnTransformer([('tfidf', TfidfVectorizer(max_features=10000, stop_words=stop_words, ngram_range=(1, 2)), 'clean_text'), ('v_min', 'passthrough', ['vader_min'])])
    
    lr_scout = Pipeline([('prep', prep_p1), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
    lr_scout.fit(X_train_p, y_train)
    
    # Feature 3: Dealbreaker
    vocab = lr_scout.named_steps['prep'].named_transformers_['tfidf'].get_feature_names_out()
    coefs = lr_scout.named_steps['clf'].coef_
    neg_words = set()
    for i in [0, 1]:  # Class 1 and 2
        top_idx = coefs[i][:len(vocab)].argsort()[-50:][::-1]
        neg_words.update([vocab[j] for j in top_idx])
    
    df_bal['has_negative_dealbreaker'] = df_bal['clean_text'].apply(lambda t: has_dealbreaker(t, neg_words))
    
    # Full Training
    feature_cols = ['clean_text', 'vader_min', 'has_negative_dealbreaker', 'llm_sentiment_score']
    X = df_bal[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words=stop_words,
                                   ngram_range=(1, 2), sublinear_tf=True), 'clean_text'),
        ('v_min', 'passthrough', ['vader_min']),
        ('flags', 'passthrough', ['has_negative_dealbreaker']),
        ('llm',   'passthrough', ['llm_sentiment_score']),
    ])
    
    models_to_train = {
        "lr":  LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=42),
        "svc": CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual=False))
    }
    
    display_names = {"lr": "Baseline LR", "rf": "Baseline RF", "svc": "Baseline SVM"}
    
    print("\nTraining Baseline Engines (Old Logic)...")
    
    for name, clf in models_to_train.items():
        pipeline = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipeline.fit(X_train, y_train)
        
        # Evaluations (Smart)
        smart_preds = pipeline.predict(X_test)
        smart_acc = accuracy_score(y_test, smart_preds)
        
        # Evaluations (Standard)
        X_test_std = X_test.copy()
        X_test_std['llm_sentiment_score'] = 0.0
        std_preds = pipeline.predict(X_test_std)
        std_acc = accuracy_score(y_test, std_preds)
        
        print(f"  - {display_names[name]} Smart Acc: {smart_acc:.1%} | Standard Acc: {std_acc:.1%}")
        
        # Save
        # Proactively fix multi_class for compatibility before saving
        if name == "lr":
            pipeline.named_steps['clf'].multi_class = 'auto'

        with open(f'models/baseline/{name}_model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
            
        with open(f'models/baseline/{name}_meta.json', 'w') as f:
            json.dump({
                "standard_test_accuracy": round(float(std_acc), 4),
                "smart_test_accuracy": round(float(smart_acc), 4),
                "test_accuracy": round(float(smart_acc), 4),
                "display_name": display_names[name],
                "trained_at": datetime.now().isoformat(),
                "feature_streams": 4,
                "features": feature_cols
            }, f, indent=2)

if __name__ == "__main__":
    main()
