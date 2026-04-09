import pandas as pd
import numpy as np
import re
import pickle
import os
import json
import nltk
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from scipy.stats import uniform, loguniform
from nltk.sentiment.vader import SentimentIntensityAnalyzer

os.makedirs('./nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')
nltk.download('vader_lexicon', download_dir='./nltk_data', quiet=True)
nltk.download('stopwords', download_dir='./nltk_data', quiet=True)

from utils import clean_text, get_vader_min, has_dealbreaker

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    MASTER_PATH = 'data/singapore_airlines_reviews_core4.csv'
    if os.path.exists(MASTER_PATH):
        print(f"✨ Found 'Core Four' Master Source: {MASTER_PATH}")
        df = pd.read_csv(MASTER_PATH)
    else:
        df = pd.read_csv('data/singapore_airlines_reviews.csv')
    df = df.dropna(subset=['text', 'rating'])
    
    # --- CLASS BALANCING (HYDRATED) ---
    print("Hydrating dataset (Zero truncation, balanced weights)...")
    df_1 = df[df['rating'] == 1]
    df_2 = df[df['rating'] == 2]
    df_3 = df[df['rating'] == 3]
    df_4 = df[df['rating'] == 4]
    df_5 = df[df['rating'] == 5]
    
    # Mild oversampling for extreme minority (2★)
    if len(df_2) < 1000:
        df_2_extra = df_2.sample(n=1000 - len(df_2), replace=True, random_state=42)
        df_2 = pd.concat([df_2, df_2_extra])
    
    df_bal = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42)
    print(f"Balanced Dataset Size: {len(df_bal)}")
    print(f"  Class distribution: {dict(df_bal['rating'].value_counts().sort_index())}")

    # --- FEATURE ENGINEERING ---
    analyzer = SentimentIntensityAnalyzer()
    if 'vader_min' in df_bal.columns and 'has_negative_dealbreaker' in df_bal.columns:
        print("⚡ Core Four features detected. Skipping on-the-fly NLP computation...")
    else:
        print("Cleaning text and engineering Core Four features...")
        df_bal['clean_text'] = df_bal['text'].apply(clean_text)
        df_bal['vader_min'] = df_bal['text'].apply(lambda x: get_vader_min(x, analyzer))
    
    stop_words = list(nltk.corpus.stopwords.words('english')) + [
        'singapore', 'airlines', 'airline', 'flight'
    ]
    
    os.makedirs('models/optimized', exist_ok=True)
    
    # ════════════════════════════════════════════════════════════════════════
    # PASS 1 — Train a lightweight LR to extract dealbreaker words
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PASS 1: Training scout LR to extract dealbreaker words...")
    print("="*60)
    
    X_pass1 = df_bal[['clean_text', 'vader_min']]
    y = df_bal['rating'].astype(int)
    
    X_train_p1, X_test_p1, y_train, y_test = train_test_split(
        X_pass1, y, test_size=0.2, random_state=42, stratify=y
    )
    
    prep_pass1 = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=25000, stop_words=stop_words,
                                   ngram_range=(1, 3), sublinear_tf=True), 'clean_text'),
        ('sent',  'passthrough', ['vader_min']),
    ])
    
    lr_scout = Pipeline([
        ('prep', prep_pass1),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    lr_scout.fit(X_train_p1, y_train)
    scout_preds = lr_scout.predict(X_test_p1)
    print(f"Scout LR Accuracy: {accuracy_score(y_test, scout_preds):.1%}")
    
    tfidf_vocab = lr_scout.named_steps['prep'].named_transformers_['tfidf'].get_feature_names_out()
    lr_coefs = lr_scout.named_steps['clf'].coef_
    n_tfidf = len(tfidf_vocab)
    
    dealbreakers = {}
    for i, star in enumerate([1, 2, 3, 4, 5]):
        tfidf_coefs = lr_coefs[i, :n_tfidf]
        top_idx = tfidf_coefs.argsort()[-50:][::-1]
        dealbreakers[star] = [tfidf_vocab[j] for j in top_idx]
    
    with open('models/dealbreaker_words.json', 'w') as f:
        json.dump(dealbreakers, f, indent=2)
    
    # ════════════════════════════════════════════════════════════════════════
    # PASS 2 — Full training with Grid Searches & Ensemble
    # ════════════════════════════════════════════════════════════════════════
    neg_words = set(dealbreakers[1] + dealbreakers[2])
    if 'has_negative_dealbreaker' not in df_bal.columns:
        df_bal['has_negative_dealbreaker'] = df_bal['clean_text'].apply(lambda t: has_dealbreaker(t, neg_words))
    
    feature_cols = ['clean_text', 'vader_min', 'has_negative_dealbreaker', 'llm_sentiment_score']
    X = df_bal[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=25000, stop_words=stop_words,
                                   ngram_range=(1, 3), sublinear_tf=True), 'clean_text'),
        ('v_min', 'passthrough', ['vader_min']),
        ('flags', 'passthrough', ['has_negative_dealbreaker']),
        ('llm',   'passthrough', ['llm_sentiment_score']),
    ])
    
    # We will build base models, tune them slightly, then ensemble
    print("\n" + "="*40 + "\nTraining Logistic Regression...\n" + "="*40)
    t_lr_start = time.time()
    lr_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_params = {'clf__C': [0.1, 0.5, 1.0, 5.0]}
    lr_grid = RandomizedSearchCV(lr_pipeline, lr_params, n_iter=3, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    t_lr = time.time() - t_lr_start
    
    print("\n" + "="*40 + "\nTraining Linear SVM...\n" + "="*40)
    t_svc_start = time.time()
    svc_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', CalibratedClassifierCV(LinearSVC(random_state=42, dual=False)))
    ])
    svc_pipeline.fit(X_train, y_train)
    best_svc = svc_pipeline
    t_svc = time.time() - t_svc_start
    
    print("\n" + "="*40 + "\nTraining Random Forest Ensembler (Soft Voting over 3 models)...\n" + "="*40)
    t_rf_start = time.time()
    # Replaced XGBoost with an unconstrained Random Forest to avoid missing Mac libomp dependencies
    # Capped n_estimators=100 and max_depth=35 to keep the .pkl file under 100MB for GitHub compatibility
    rf = RandomForestClassifier(n_estimators=100, max_depth=35, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    
    # The ultimate ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('lr', best_lr.named_steps['clf']),
            ('svc', best_svc.named_steps['clf']),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    ensemble_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', ensemble)
    ])
    
    ensemble_pipeline.fit(X_train, y_train)
    t_rf = time.time() - t_rf_start
    
    # Evaluation and Saving
    models_to_save = {
        "lr": {"model": best_lr, "display": f"Logistic Regression (tuned C={lr_grid.best_params_['clf__C']})", "time": t_lr},
        "svc": {"model": best_svc, "display": "Linear SVM (Calibrated)", "time": t_svc},
        "rf": {"model": ensemble_pipeline, "display": "Super Ensemble (RF+LR+SVM)", "time": t_rf}  # We override 'rf' to avoid breaking dashboard layout
    }
    
    for name, config in models_to_save.items():
        pipeline = config["model"]
        display_name = config["display"]
        
        # 1. Smart Mode Evaluation (Actual features)
        smart_preds = pipeline.predict(X_test)
        smart_relaxed_acc = np.mean(np.abs(y_test - smart_preds) <= 1)
        smart_strict_acc = accuracy_score(y_test, smart_preds)
        
        # 2. Standard Mode Evaluation (Simulated: zero out LLM features)
        X_test_std = X_test.copy()
        X_test_std['llm_sentiment_score'] = 0.0
        std_preds = pipeline.predict(X_test_std)
        std_relaxed_acc = np.mean(np.abs(y_test - std_preds) <= 1)
        
        # 3. Train Evaluation (Smart)
        train_preds = pipeline.predict(X_train)
        train_relaxed_acc = np.mean(np.abs(y_train - train_preds) <= 1)
        
        print(f"\n{display_name}:")
        print(f"  - Smart Relaxed (±1) ACC: {smart_relaxed_acc:.1%} (Strict: {smart_strict_acc:.1%})")
        print(f"  - Standard Relaxed (±1) ACC: {std_relaxed_acc:.1%}")
        
        # --- PROACTIVE PATCHING FOR SERIALIZATION COMPATIBILITY ---
        # Ensure any LogisticRegression in the pipeline or ensemble has 'multi_class' set
        # This prevents AttributeError when unpickling in different scikit-learn versions
        def patch_lr(obj):
            if isinstance(obj, LogisticRegression):
                if not hasattr(obj, 'multi_class'):
                    obj.multi_class = 'auto'
            elif hasattr(obj, 'named_steps'): # Pipeline
                for step in obj.named_steps.values(): patch_lr(step)
            elif hasattr(obj, 'estimators_'): # VotingClassifier
                for est in obj.estimators_: patch_lr(est)
            elif hasattr(obj, 'named_estimators_'): # Also for VotingClassifier
                for est in obj.named_estimators_.values(): patch_lr(est)
        
        patch_lr(pipeline)

        # Save
        with open(f'models/optimized/{name}_model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
            
        with open(f'models/optimized/{name}_meta.json', 'w') as f:
            json.dump({
                "standard_test_accuracy": round(float(std_relaxed_acc), 4),
                "smart_test_accuracy": round(float(smart_relaxed_acc), 4),
                "test_accuracy": round(float(smart_relaxed_acc), 4), # Legacy sync
                "strict_accuracy": round(float(smart_strict_acc), 4),
                "train_accuracy": round(float(train_relaxed_acc), 4),
                "accuracy": round(float(smart_relaxed_acc), 4),
                "display_name": display_name,
                "trained_at": datetime.now().isoformat(),
                "training_time_s": round(config["time"], 2),
                "feature_streams": 4,
                "features": feature_cols
            }, f, indent=2)
            
    print("\nAll models upgraded! Generating Aspect Engine...\n")
    
    # Ensure aspect model pipeline exists
    ASPECT_TAXONOMY = {
        "Food & Beverage": ["food", "meal", "drink", "water", "wine", "chicken", "beef", "breakfast", "lunch", "dinner", "taste"],
        "Seat & Comfort": ["seat", "comfort", "legroom", "recline", "space", "cramped", "narrow", "sleep", "bed", "aisle"],
        "Staff & Service": ["crew", "staff", "attendant", "steward", "rude", "friendly", "polite", "helpful", "service", "smile"],
        "Flight Punctuality": ["delay", "delayed", "late", "wait", "cancel", "cancelled", "hours", "schedule"],
        "Baggage Handling": ["baggage", "bag", "luggage", "lost", "belt", "claim", "carousel", "damaged"],
        "Inflight Entertainment": ["wifi", "internet", "movie", "screen", "krisworld", "tv", "entertainment"],
        "Booking & Check-in": ["website", "app", "check-in", "checkin", "online", "booking", "system", "payment"]
    }
    
    df_all = df.dropna(subset=['text'])
    df_all['clean_text'] = df_all['text'].apply(clean_text)
    
    aspect_labels = []
    for text in df_all['clean_text']:
        words = set(text.split())
        aspect_labels.append([1 if any(kw in words for kw in kws) else 0 for kws in ASPECT_TAXONOMY.values()])
    
    y_aspects, X_aspects = np.array(aspect_labels), df_all['clean_text']
    mask = y_aspects.sum(axis=1) > 0
    X_aspects, y_aspects = X_aspects[mask], y_aspects[mask]
    
    from sklearn.multioutput import MultiOutputClassifier
    aspect_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, stop_words=stop_words, ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced', random_state=42, dual=False)))
    ])
    
    aspect_pipeline.fit(X_aspects, y_aspects)
    with open('models/aspect_model.pkl', 'wb') as f:
        pickle.dump(aspect_pipeline, f)
        
    print("ALL ENGINES OPTIMIZED. Pipeline complete.")

if __name__ == "__main__":
    main()
