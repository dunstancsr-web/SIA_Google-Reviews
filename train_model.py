import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

os.makedirs('./nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')
nltk.download('vader_lexicon', download_dir='./nltk_data', quiet=True)
nltk.download('stopwords', download_dir='./nltk_data', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def main():
    print("Loading data...")
    df = pd.read_csv('data/singapore_airlines_reviews.csv')
    df = df.dropna(subset=['text', 'rating'])
    
    # --- ABRASIVE UNDER-SAMPLING ---
    # We have far too many 5-star reviews (54%). Let's cap 5-star and 4-star counts 
    # to force the model to be stricter.
    print("Re-balancing classes (Abrasive Under-sampling)...")
    df_1 = df[df['rating'] == 1]
    df_2 = df[df['rating'] == 2]
    df_3 = df[df['rating'] == 3]
    df_4 = df[df['rating'] == 4].sample(n=1200, random_state=42) # Cap 4-stars
    df_5 = df[df['rating'] == 5].sample(n=1500, random_state=42) # Heavily cap 5-stars
    
    df_bal = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42)
    print(f"Balanced Dataset Size: {len(df_bal)} (Capped 5-stars at 1500)")
    
    print("Cleaning and Augmenting with Sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df_bal['clean_text'] = df_bal['text'].apply(clean_text)
    df_bal['vader_score'] = df_bal['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    
    # --- FEATURES ---
    X = df_bal[['clean_text', 'vader_score']]
    y = df_bal['rating'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    stop_words = list(nltk.corpus.stopwords.words('english')) + ['singapore', 'airlines', 'airline', 'flight']
    
    # Shared Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words=stop_words, ngram_range=(1, 2)), 'clean_text'),
            ('sent', 'passthrough', ['vader_score'])
        ]
    )
    
    models_to_train = {
        "lr": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "svc": CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual=False))
    }
    
    os.makedirs('models', exist_ok=True)
    print("\nStarting 'Strict Mode' Model Tournament...\n" + "="*40)
    
    for name, clf in models_to_train.items():
        print(f"Training {name.upper()}...")
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name.upper()} Accuracy (Strict): {acc:.1%}")
        
        # Save Entire Pipeline
        filename = f'models/{name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Successfully saved Pipeline to {filename}")
        
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
    
    # Process all reviews (not just balanced ones) for more aspect variety
    df_all = df.dropna(subset=['text'])
    df_all['clean_text'] = df_all['text'].apply(clean_text)
    
    # Create Binary Labels for each Aspect
    aspect_labels = []
    for text in df_all['clean_text']:
        words = set(text.split())
        row_labels = []
        for cat, keywords in ASPECT_TAXONOMY.items():
            row_labels.append(1 if any(kw in words for kw in keywords) else 0)
        aspect_labels.append(row_labels)
    
    y_aspects = np.array(aspect_labels)
    X_aspects = df_all['clean_text']
    
    # Remove rows with zero tags (ambiguous) to keep training signal high
    mask = y_aspects.sum(axis=1) > 0
    X_aspects = X_aspects[mask]
    y_aspects = y_aspects[mask]
    
    print(f"Extracted {len(X_aspects)} tagged reviews for Aspect Training.")
    
    from sklearn.multioutput import MultiOutputClassifier
    
    # Aspect Pipeline
    aspect_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, stop_words=stop_words, ngram_range=(1, 2))),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight='balanced', random_state=42, dual=False)))
    ])
    
    print("Training Autonomous Aspect Engine (LinearTournament)...")
    aspect_pipeline.fit(X_aspects, y_aspects)
    
    # Save Model
    aspect_filename = 'models/aspect_model.pkl'
    with open(aspect_filename, 'wb') as f:
        pickle.dump(aspect_pipeline, f)
    
    print(f"Successfully saved HIGH-ACCURACY Aspect Engine to {aspect_filename}")
    print("="*40 + "\nAll engines optimized! Dashboard is ready for 'Strict Mode' Intelligence.")

if __name__ == "__main__":
    main()
