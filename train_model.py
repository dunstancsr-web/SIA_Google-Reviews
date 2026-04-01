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
from sklearn.preprocessing import OneHotEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def get_vader_class(text, analyzer):
    """Classify overall review tone into a discrete label."""
    score = analyzer.polarity_scores(str(text))['compound']
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

def get_sentence_vader_stats(text, analyzer):
    """Return (min, max, range) of per-sentence VADER scores."""
    sentences = re.split(r'[.!?]', str(text))
    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences if s.strip()]
    if not scores:
        return 0.0, 0.0, 0.0
    return min(scores), max(scores), max(scores) - min(scores)

def has_dealbreaker(text, word_set):
    words = set(text.split())
    return 1 if words & word_set else 0

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv('data/singapore_airlines_reviews.csv')
    df = df.dropna(subset=['text', 'rating'])
    
    # --- CLASS BALANCING ---
    # Under-sample the majority classes AND oversample the sparse 2★ class
    print("Re-balancing classes (Under-sampling + 2★ Oversampling)...")
    df_1 = df[df['rating'] == 1]                                                    # ~1057
    df_2 = df[df['rating'] == 2]                                                    # ~543
    df_3 = df[df['rating'] == 3]                                                    # ~1009
    df_4 = df[df['rating'] == 4].sample(n=min(1200, len(df[df['rating'] == 4])), random_state=42)
    df_5 = df[df['rating'] == 5].sample(n=min(1500, len(df[df['rating'] == 5])), random_state=42)
    
    # Oversample 2★ to ~1000 by duplicating (prevents class collapse)
    if len(df_2) < 900:
        df_2_extra = df_2.sample(n=900 - len(df_2), replace=True, random_state=42)
        df_2 = pd.concat([df_2, df_2_extra])
    
    df_bal = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42)
    print(f"Balanced Dataset Size: {len(df_bal)}")
    print(f"  Class distribution: {dict(df_bal['rating'].value_counts().sort_index())}")

    
    # --- FEATURE ENGINEERING ---
    print("Cleaning text and engineering features...")
    analyzer = SentimentIntensityAnalyzer()
    
    df_bal['clean_text'] = df_bal['text'].apply(clean_text)
    df_bal['vader_score'] = df_bal['text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    df_bal['vader_class'] = df_bal['text'].apply(
        lambda x: get_vader_class(x, analyzer)
    )
    
    # Sentence-level VADER features
    sentence_stats = df_bal['text'].apply(lambda x: get_sentence_vader_stats(x, analyzer))
    df_bal['vader_min'] = sentence_stats.apply(lambda x: x[0])
    df_bal['vader_max'] = sentence_stats.apply(lambda x: x[1])
    df_bal['vader_range'] = sentence_stats.apply(lambda x: x[2])
    
    stop_words = list(nltk.corpus.stopwords.words('english')) + [
        'singapore', 'airlines', 'airline', 'flight'
    ]
    
    os.makedirs('models', exist_ok=True)
    
    # ════════════════════════════════════════════════════════════════════════
    # PASS 1 — Train a lightweight LR to extract dealbreaker words
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PASS 1: Training scout LR to extract dealbreaker words...")
    print("="*60)
    
    X_pass1 = df_bal[['clean_text', 'vader_score', 'vader_class',
                       'vader_min', 'vader_max', 'vader_range']]
    y = df_bal['rating'].astype(int)
    
    X_train_p1, X_test_p1, y_train, y_test = train_test_split(
        X_pass1, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pass-1 preprocessor (no dealbreaker flags yet)
    prep_pass1 = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words=stop_words,
                                   ngram_range=(1, 2), sublinear_tf=True), 'clean_text'),
        ('sent',  'passthrough', ['vader_score', 'vader_min', 'vader_max', 'vader_range']),
        ('class', OneHotEncoder(handle_unknown='ignore'), ['vader_class']),
    ])
    
    lr_scout = Pipeline([
        ('prep', prep_pass1),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    lr_scout.fit(X_train_p1, y_train)
    scout_preds = lr_scout.predict(X_test_p1)
    scout_acc = accuracy_score(y_test, scout_preds)
    print(f"Scout LR Accuracy: {scout_acc:.1%}")
    
    # --- Extract dealbreaker words ---
    tfidf_vocab = lr_scout.named_steps['prep'].named_transformers_['tfidf'].get_feature_names_out()
    lr_coefs = lr_scout.named_steps['clf'].coef_  # shape: (5, n_features)
    n_tfidf = len(tfidf_vocab)
    
    dealbreakers = {}
    for i, star in enumerate([1, 2, 3, 4, 5]):
        # Only look at the TF-IDF portion of the coefficient vector
        tfidf_coefs = lr_coefs[i, :n_tfidf]
        top_idx = tfidf_coefs.argsort()[-50:][::-1]
        dealbreakers[star] = [tfidf_vocab[j] for j in top_idx]
    
    with open('models/dealbreaker_words.json', 'w') as f:
        json.dump(dealbreakers, f, indent=2)
    print(f"Saved dealbreaker words for all 5 classes → models/dealbreaker_words.json")
    
    # Print top 10 for visibility
    for star in [1, 5]:
        print(f"  Top-10 dealbreakers for {star}★: {dealbreakers[star][:10]}")
    
    # ════════════════════════════════════════════════════════════════════════
    # PASS 2 — Full training with dealbreaker flags
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PASS 2: Full training with 5-stream feature set...")
    print("="*60)
    
    neg_words = set(dealbreakers[1] + dealbreakers[2])
    pos_words = set(dealbreakers[4] + dealbreakers[5])
    
    df_bal['has_neg_dealbreaker'] = df_bal['clean_text'].apply(lambda t: has_dealbreaker(t, neg_words))
    df_bal['has_pos_dealbreaker'] = df_bal['clean_text'].apply(lambda t: has_dealbreaker(t, pos_words))
    
    # Full feature set
    feature_cols = ['clean_text', 'vader_score', 'vader_class',
                    'vader_min', 'vader_max', 'vader_range',
                    'has_neg_dealbreaker', 'has_pos_dealbreaker']
    X = df_bal[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Full 5-stream preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words=stop_words,
                                   ngram_range=(1, 2), sublinear_tf=True), 'clean_text'),
        ('sent',  'passthrough', ['vader_score', 'vader_min', 'vader_max', 'vader_range']),
        ('class', OneHotEncoder(handle_unknown='ignore'), ['vader_class']),
        ('flags', 'passthrough', ['has_neg_dealbreaker', 'has_pos_dealbreaker']),
    ])
    
    models_to_train = {
        "lr":  LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, class_weight='balanced', random_state=42),
        "svc": CalibratedClassifierCV(LinearSVC(C=0.1, class_weight='balanced', random_state=42, dual=False))
    }
    
    display_names = {"lr": "Logistic Regression", "rf": "Random Forest", "svc": "Linear SVM"}
    
    print("\nStarting 'Strict Mode' Model Tournament...\n" + "="*40)
    
    for name, clf in models_to_train.items():
        print(f"\nTraining {display_names[name]}...")
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate on Test
        preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        
        # Evaluate on Train
        train_preds = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        
        print(f"{display_names[name]} Test Accuracy: {test_acc:.1%}")
        print(f"{display_names[name]} Train Accuracy: {train_acc:.1%}")
        print(classification_report(y_test, preds, target_names=['1★','2★','3★','4★','5★'], zero_division=0))
        
        # Save Pipeline
        filename = f'models/{name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Saved Pipeline → {filename}")
        
        # Save metadata (dynamic benchmarks)
        meta = {
            "test_accuracy": round(float(test_acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "accuracy": round(float(test_acc), 4),  # Legacy fallback
            "display_name": display_names[name],
            "trained_at": datetime.now().isoformat(),
            "feature_streams": 5,
            "features": feature_cols
        }
        with open(f'models/{name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata → models/{name}_meta.json")
    
    print("="*40 + "\nAll Strict Models trained! Moving to Aspect Engine...\n" + "="*40)
    
    # --- AUTONOMOUS ASPECT ENGINE (MULTI-LABEL) ---
    print("Preparing Aspect Training Data (Silver Labeling)...")
    ASPECT_TAXONOMY = {
        "Food & Beverage": ["food", "meal", "drink", "water", "wine", "chicken", "beef", "breakfast", "lunch", "dinner", "taste", "menu", "beverage", "hungry", "thirsty"],
        "Seat & Comfort": ["seat", "comfort", "legroom", "recline", "space", "cramped", "narrow", "sleep", "bed", "aisle", "window", "sore", "uncomfortable"],
        "Staff & Service": ["crew", "staff", "attendant", "steward", "stewardess", "rude", "friendly", "polite", "helpful", "service", "smile", "ignored", "professional", "attendants"],
        "Flight Punctuality": ["delay", "delayed", "late", "wait", "cancel", "cancelled", "hours", "schedule", "missed", "connection", "waiting"],
        "Baggage Handling": ["baggage", "bag", "luggage", "lost", "belt", "claim", "carousel", "damaged"],
        "Inflight Entertainment": ["wifi", "internet", "movie", "screen", "krisworld", "tv", "entertainment", "monitor", "headphone", "movies"],
        "Booking & Check-in": ["website", "app", "check-in", "checkin", "online", "booking", "system", "payment", "error", "counter", "boarding", "ticket", "tickets"]
    }
    
    df_all = df.dropna(subset=['text'])
    df_all['clean_text'] = df_all['text'].apply(clean_text)
    
    aspect_labels = []
    for text in df_all['clean_text']:
        words = set(text.split())
        row_labels = []
        for cat, keywords in ASPECT_TAXONOMY.items():
            row_labels.append(1 if any(kw in words for kw in keywords) else 0)
        aspect_labels.append(row_labels)
    
    y_aspects = np.array(aspect_labels)
    X_aspects = df_all['clean_text']
    
    mask = y_aspects.sum(axis=1) > 0
    X_aspects = X_aspects[mask]
    y_aspects = y_aspects[mask]
    
    print(f"Extracted {len(X_aspects)} tagged reviews for Aspect Training.")
    
    from sklearn.multioutput import MultiOutputClassifier
    
    aspect_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, stop_words=stop_words,
                                   ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced', random_state=42, dual=False)))
    ])
    
    print("Training Autonomous Aspect Engine...")
    aspect_pipeline.fit(X_aspects, y_aspects)
    
    aspect_filename = 'models/aspect_model.pkl'
    with open(aspect_filename, 'wb') as f:
        pickle.dump(aspect_pipeline, f)
    
    print(f"Saved Aspect Engine → {aspect_filename}")
    print("="*60)
    print("ALL ENGINES OPTIMIZED. Dashboard ready for Strict Mode Intelligence.")
    print("="*60)

if __name__ == "__main__":
    main()
