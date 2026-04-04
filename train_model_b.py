import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── CONFIGURATION — SETUP B ───────────────────────────────────────────────────
MODEL_DIR = 'models_b'
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print(f"🚀 Initializing SETUP B (AI Mapping - 1 Feature) Training...")
    
    # --- DATA LOADING ---
    MASTER_PATH = 'data/singapore_airlines_reviews_core4.csv'
    if not os.path.exists(MASTER_PATH):
        print(f"❌ Error: {MASTER_PATH} not found. Please run master data script first.")
        return
        
    df = pd.read_csv(MASTER_PATH)
    # Ensure we have the required LLM feature and drop rows where it might be missing
    df = df.dropna(subset=['llm_sentiment_score', 'rating'])
    
    # --- CLASS BALANCING ---
    # Setup B relies heavily on the quality of the LLM mapping, so we maintain broad balance.
    print("Re-balancing classes...")
    df_1 = df[df['rating'] == 1]
    df_2 = df[df['rating'] == 2]
    df_3 = df[df['rating'] == 3]
    df_4 = df[df['rating'] == 4].sample(n=min(1200, len(df[df['rating'] == 4])), random_state=42)
    df_5 = df[df['rating'] == 5].sample(n=min(1500, len(df[df['rating'] == 5])), random_state=42)
    
    # Oversample sparse classes for a robust continuum
    if len(df_2) < 900:
        df_2_extra = df_2.sample(n=900 - len(df_2), replace=True, random_state=42)
        df_2 = pd.concat([df_2, df_2_extra])
        
    df_bal = pd.concat([df_1, df_2, df_3, df_4, df_5]).sample(frac=1, random_state=42)
    print(f"Balanced Dataset Size: {len(df_bal)}")
    
    # --- FEATURE SELECTION ---
    # Mode B is 'Pure AI Mapping' - using ONLY the LLM score
    feature_cols = ['llm_sentiment_score']
    X = df_bal[feature_cols]
    y = df_bal['rating'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Minimalist Preprocessor (just passthrough the numeric score)
    preprocessor = ColumnTransformer(transformers=[
        ('llm', 'passthrough', ['llm_sentiment_score']),
    ])
    
    models_to_train = {
        "lr":  LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42),
        "svc": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual=False))
    }
    
    display_names = {"lr": "Logistic Regression (B)", "rf": "Random Forest (B)", "svc": "Linear SVM (B)"}
    
    print("\nStarting SETUP B Model Tournament...\n" + "="*40)
    
    for name, clf in models_to_train.items():
        print(f"\nTraining {display_names[name]}...")
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        
        print(f"{display_names[name]} Test Accuracy: {test_acc:.1%}")
        
        # Save Pipeline
        filename = f'{MODEL_DIR}/{name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)
            
        # Save metadata
        meta = {
            "test_accuracy": round(float(test_acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "display_name": display_names[name],
            "trained_at": datetime.now().isoformat(),
            "feature_streams": 1,
            "features": feature_cols
        }
        with open(f'{MODEL_DIR}/{name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
            
    print("\n" + "="*40)
    print(f"✅ Setup B Training Complete! Models saved to: {MODEL_DIR}/")
    print("="*40)

if __name__ == "__main__":
    main()
