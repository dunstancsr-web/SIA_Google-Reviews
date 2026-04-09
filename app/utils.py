import pandas as pd
import numpy as np
import re
import requests
import json
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLP INITIALIZATION ---
os.makedirs('./nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')
try:
    nltk.download('vader_lexicon', download_dir='./nltk_data', quiet=True)
except Exception:
    pass

_default_analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    """
    Cleans text of punctuation and non-alphanumeric characters.
    Used uniformly across training scripts, API wrappers, and UI dashboards.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def get_vader_min(text, analyzer=None):
    """
    Isolates the single angriest sentence (The Pain Point).
    Args:
        text (str): Input text (the review).
        analyzer: Provide a VADER SentimentIntensityAnalyzer instance, or Uses default.
    """
    if analyzer is None:
        analyzer = _default_analyzer
    sentences = re.split(r'[.!?]', str(text))
    scores = [analyzer.polarity_scores(s)['compound'] for s in sentences if s.strip()]
    return min(scores) if scores else 0.0

def has_dealbreaker(text, word_set):
    """
    Checks if the given text contains any of the critical dealbreaker words.
    Args:
        text (str): Input text
        word_set (set): A set of words considered dealbreakers.
    """
    if not isinstance(text, str): 
        return 0
    words = set(text.split())
    return 1 if words & word_set else 0

def get_llm_sentiment(text, model_name="llama3:latest", url="http://localhost:11434/api/generate", timeout=15):
    """
    Pings the local Ollama API for a high-nuance sentiment score (-1.0 to 1.0).
    Args:
        text (str): Input review.
        model_name (str): Llama3 or equivalent model.
        url (str): Endpoint for generation.
        timeout (int): API timeout.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    prompt = f'Analyze sentiment of this review. Output ONLY a decimal from -1.0 to 1.0. No text. Review: "{text[:800]}"'
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            raw_text = response.json().get("response", "0.0").strip()
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", raw_text)
            if matches:
                return max(-1.0, min(1.0, float(matches[0])))
    except Exception:
        pass
        
    return 0.0
