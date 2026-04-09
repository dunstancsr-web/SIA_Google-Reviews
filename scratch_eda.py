import pandas as pd
import json
import os
import numpy as np

def run_eda():
    data_path = "data/singapore_airlines_reviews_core4.csv"
    if not os.path.exists(data_path):
        data_path = "data/singapore_airlines_reviews.csv"
        
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip("\"").str.strip()
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    df["published_date"] = df["published_date"].dt.tz_convert(None)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text_length"] = df["text"].fillna("").str.len()
    
    # Filter out missing dates or ratings
    df = df.dropna(subset=["published_date", "rating"])
    
    output = {}
    output["Total Reviews"] = len(df)
    output["Date Range"] = f"{df['published_date'].min().strftime('%Y-%m-%d')} to {df['published_date'].max().strftime('%Y-%m-%d')}"
    
    # Chart 1 & 2: Volume and Avg Rating over Time
    time_series = df.set_index("published_date").resample("MS").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()
    time_series["published_date"] = time_series["published_date"].dt.strftime("%Y-%m")
    
    output["Time Series (Monthly)"] = time_series.dropna().to_dict('records')
    
    # Chart 3: Rating Distribution
    rating_counts = df.groupby("rating").agg(
        count=("rating", "count"),
        avg_length=("text_length", "mean"),
        median_length=("text_length", "median")
    ).reset_index()
    output["Rating Distribution"] = rating_counts.to_dict('records')
    
    # Chart 4: Sentiment Share (uses llm_sentiment_score)
    if "llm_sentiment_score" in df.columns:
        pos_f = df[df["llm_sentiment_score"] > 0.05]
        neu_f = df[df["llm_sentiment_score"].between(-0.05, 0.05)]
        neg_f = df[df["llm_sentiment_score"] < -0.05]
        
        output["Sentiment Share"] = {
            "Positive": len(pos_f),
            "Neutral": len(neu_f),
            "Negative": len(neg_f)
        }
    else:
        output["Sentiment Share"] = "llm_sentiment_score not found in dataset"
        
    # Chart 5 & 6: Platform Volume and Avg Rating
    if "published_platform" in df.columns:
        platform_stats = df.groupby("published_platform").agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_length=("text_length", "mean")
        ).reset_index()
        output["Platform Stats"] = platform_stats.to_dict('records')
    else:
        output["Platform Stats"] = "published_platform not found"
        
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    run_eda()
