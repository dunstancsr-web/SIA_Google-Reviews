import pandas as pd
import streamlit as st
import altair as alt
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
import pickle
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import html
import json
import ast
import textwrap

os.makedirs('./nltk_data', exist_ok=True)
nltk.data.path.append('./nltk_data')
try:
    nltk.download('vader_lexicon', download_dir='./nltk_data', quiet=True)
except Exception:
    pass

@st.cache_resource
def load_ml_models():
    """
    Loads rating models, aspect engine, dealbreaker words, and dynamic benchmarks.
    """
    try:
        models = {}
        aspect_model = None
        benchmarks = {}
        dealbreaker_words = {"neg": set(), "pos": set()}
        
        model_files = {
            "Logistic Regression": ("models/lr_model.pkl", "models/lr_meta.json"),
            "Random Forest": ("models/rf_model.pkl", "models/rf_meta.json"),
            "Linear SVM": ("models/svc_model.pkl", "models/svc_meta.json")
        }
        
        for name, (pkl_path, meta_path) in model_files.items():
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    m = pickle.load(f)
                    if name == "Logistic Regression":
                        clf_step = m.named_steps.get('clf')
                        if clf_step and not hasattr(clf_step, 'multi_class'):
                            clf_step.multi_class = 'auto'
                    models[name] = m
            # Load dynamic benchmark accuracy
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    m_meta = json.load(f)
                    # Load Test Acc as primary benchmark
                    benchmarks[name] = m_meta.get("test_accuracy", m_meta.get("accuracy", 0.5))
                    # Load Train Acc for the audit table
                    benchmarks[name + "_train"] = m_meta.get("train_accuracy", 0.8)
            else:
                benchmarks[name] = 0.5
                    
        # Load the Autonomous Aspect Engine
        aspect_path = "models/aspect_model.pkl"
        if os.path.exists(aspect_path):
            with open(aspect_path, 'rb') as f:
                aspect_model = pickle.load(f)
        
        # Load dealbreaker words
        db_path = "models/dealbreaker_words.json"
        if os.path.exists(db_path):
            with open(db_path) as f:
                db = json.load(f)
                dealbreaker_words["neg"] = set(db.get("1", []) + db.get("2", []))
                dealbreaker_words["pos"] = set(db.get("4", []) + db.get("5", []))
                    
        return {"rating": models, "aspect": aspect_model,
                "benchmarks": benchmarks, "dealbreakers": dealbreaker_words}
    except Exception as e:
        return {"rating": {}, "aspect": None,
                "benchmarks": {}, "dealbreakers": {"neg": set(), "pos": set()}}

def render_star_rating(score, color="#ca8a04"):
    """
    Renders 5 stars with partial fill based on the fractional score using inline SVGs.
    Utilizes a custom CSS tooltip system.
    """
    hover_text = f"{score:.1f} STARS"
    stars_html = f'<div class="star-v2-container" data-tooltip="{hover_text}" style="gap: 4px; align-items: center;">'
    for i in range(1, 6):
        fill_pct = max(0, min(100, (score - (i - 1)) * 100))
        grad_id = f"starGrad_{i}_{str(score).replace('.','_')}"
        
        stars_html += f"""
            <svg width="22" height="22" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0px 1px 1px rgba(0,0,0,0.05)); pointer-events: none;">
                <defs>
                    <linearGradient id="{grad_id}">
                        <stop offset="{fill_pct}%" stop-color="{color}"/>
                        <stop offset="{fill_pct}%" stop-color="#e5e7eb"/>
                    </linearGradient>
                </defs>
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="url(#{grad_id})"/>
            </svg>
        """
    stars_html += '</div>'
    return stars_html

@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()


st.set_page_config(page_title="Self‑Service Data Hub", page_icon="SIA", layout="wide")

# Global font styling (Streamlit UI)
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: Arial, sans-serif;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.01rem;
        }
        .stDownloadButton button {
            background-color: transparent;
            border: 1px solid #000000;
            color: #000000;
            font-size: 1.3rem;
            font-weight: 600;
            padding: 1.2rem 1.8rem;
        }
        .stDownloadButton button:hover {
            background-color: #30d158;
            border: 1px solid #30d158;
            color: #ffffff;
        }
        [data-testid="stButton"] button {
            transition: background-color 0.2s ease, color 0.2s ease;
        }
        [data-testid="stButton"] button[kind="primary"] {
            background-color: transparent !important;
            color: #111827 !important;
            border: 1px solid #9ca3af !important;
        }
        [data-testid="stButton"] button:hover {
            background-color: #30d158 !important;
            color: #ffffff !important;
            border-color: #30d158 !important;
        }
        [data-baseweb="switch"] input:checked + div {
            background-color: #16a34a !important;
            border-color: #16a34a !important;
        }
        [data-baseweb="switch"] input:checked + div > div {
            background-color: #ffffff !important;
        }
        .guide-hover-wrap {
            position: relative;
            display: inline-block;
            margin-bottom: 0.25rem;
        }
        .guide-hover-trigger {
            display: inline-block;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            cursor: help;
        }
        .guide-hover-trigger:hover {
            background-color: #f4f6f8;
        }
        .guide-hover-tooltip {
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.15s ease;
            position: absolute;
            left: 0;
            top: calc(100% + 6px);
            z-index: 1000;
            font-size: 20px;
            line-height: 1.35;
            background: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 10px 12px;
            white-space: nowrap;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
        }
        .guide-hover-wrap:hover .guide-hover-tooltip {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stExpander"] details summary:hover {
            background-color: #30d158;
            color: #ffffff;
            border-radius: 8px;
        }
        [data-testid="stExpander"] details summary:hover * {
            color: #ffffff !important;
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-bottom: 0.35rem;
        }
        [data-baseweb="tab-list"] button,
        [data-baseweb="tab-list"] [role="tab"],
        [data-baseweb="tab"] {
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            padding: 1rem 2rem !important;
            min-height: 3.5rem !important;
            border-radius: 0 !important;
            background-color: transparent !important;
            color: #1f2937 !important;
            transition: background-color 0.2s ease, color 0.2s ease;
        }
        [data-baseweb="tab-list"] button *,
        [data-baseweb="tab-list"] [role="tab"] *,
        [data-baseweb="tab"] * {
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        [data-baseweb="tab-list"] button:hover,
        [data-baseweb="tab-list"] [role="tab"]:hover,
        [data-baseweb="tab"]:hover {
            background-color: #f3fdf6 !important;
            color: #15803d !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"],
        [data-baseweb="tab-list"] [role="tab"][aria-selected="true"],
        [data-baseweb="tab"][aria-selected="true"] {
            background-color: transparent !important;
            color: #15803d !important;
            border: none !important;
            border-bottom: 3px solid #30d158 !important;
            box-shadow: none !important;
            text-decoration: none !important;
        }
        [data-baseweb="tab-list"] button,
        [data-baseweb="tab-list"] [role="tab"],
        [data-baseweb="tab"] {
            border: none !important;
            border-bottom: 3px solid transparent !important;
            box-shadow: none !important;
            text-decoration: none !important;
        }
        [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
            height: 0 !important;
        }
        [data-baseweb="tab-border"],
        [data-baseweb="tab-list"]::before,
        [data-baseweb="tab-list"]::after {
            display: none !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            height: 0 !important;
        }
        .title-hover-wrap {
            position: relative;
            display: inline-block;
            width: fit-content;
        }
        .title-hover-tooltip {
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.15s ease;
            position: absolute;
            left: 300px;
            bottom: calc(100% - 20px);
            z-index: 1000;
            font-size: 30px;
            line-height: 1.35;
            color: #6b6b6b;
            background: #ffffff;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 12px 14px;
            white-space: nowrap;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
        }
        .title-hover-wrap:hover .title-hover-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .stButton button:active {
            transform: scale(0.98);
        }
        
        /* Premium Verdict Hero CSS */
        .verdict-hero {
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 8px solid #D4AF37;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .verdict-hero-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #4b5563;
        }
        .verdict-score-big {
            font-size: 3.5rem;
            font-weight: 800;
            line-height: 1;
            margin-right: 1rem;
        }
        
        /* Consensus Progress Bar */
        .consensus-container {
            margin-top: 1rem;
            background: rgba(255, 255, 255, 0.5);
            padding: 1rem;
            border-radius: 8px;
        }
        .consensus-track {
            height: 12px;
            background-color: #e5e7eb;
            border-radius: 6px;
            width: 100%;
            overflow: visible;
            position: relative;
            margin-top: 0.5rem;
        }
        .consensus-fill {
            height: 100%;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 6px;
        }
        
        /* Voting Breakdown Toggle (CSS checkbox hack) */
        .voting-breakdown-panel {
            display: none;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.3s ease;
        }
        #voting-toggle:checked ~ .voting-breakdown-panel {
            display: block;
        }
        #voting-toggle:checked ~ label[for="voting-toggle"] {
            opacity: 0.5;
        }
        label[for="voting-toggle"]:hover {
            opacity: 0.8 !important;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Tick markers for Model Votes */
        .tick-marker {
            position: absolute;
            top: -8px;
            width: 12px;
            height: 28px;
            background: rgba(255, 255, 255, 0.8);
            z-index: 20;
            cursor: help;
            pointer-events: auto;
            border: 1px solid rgba(200, 200, 200, 0.6);
            border-radius: 3px;
            margin-left: -6px;
            transition: all 0.2s ease;
        }
        .tick-marker:hover {
            background: rgba(255, 255, 255, 1);
            height: 32px;
            top: -10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            border-color: rgba(100, 100, 100, 0.8);
        }
        .tick-marker::after {
            content: attr(data-label);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
            pointer-events: none;
            margin-bottom: 8px;
            opacity: 0;
            transition: opacity 0.2s ease;
            z-index: 30;
        }
        .tick-marker:hover::after {
            opacity: 1;
        }
        
        /* Actionable Chips */
        .sia-chip {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
            border: 1px solid rgba(0,0,0,0.1);
            background-color: #ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        /* Custom Star Tooltip System */
        .star-v2-container {
            position: relative;
            display: inline-flex;
            cursor: help;
        }
        .star-v2-container::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
            pointer-events: none;
            opacity: 0;
            transition: all 0.2s ease;
            z-index: 9999;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .star-v2-container:hover::after {
            opacity: 1;
        }
        
        /* Custom HTML Native Tooltip for Hero Card Headers */
        .sia-tooltip-wrap:hover .sia-popover {
            visibility: visible !important;
            opacity: 1 !important;
        }
        .sia-tooltip-wrap:hover svg {
            fill: #4b5563 !important;
        }
        .star-v2-container:hover::after {
            opacity: 1;
        }
        
        /* Sausage (Math Audit) Tooltip - Genuine HTML Container */
        .sausage-wrap {
            position: relative;
            display: inline-block;
            cursor: help;
            border-bottom: 1.5px dashed #6b7280;
        }
        .sausage-content {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            background: #ffffff;
            color: #111827;
            padding: 20px;
            border-radius: 12px;
            font-size: 0.85rem;
            line-height: 1.6;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
            white-space: nowrap;
            z-index: 10005;
            min-width: 400px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.25), 0 8px 15px rgba(0,0,0,0.1);
            border: 1px solid #d1d5db;
            margin-top: 10px;
            text-align: left;
            text-transform: none; /* Prevent uppercase inheritance */
            letter-spacing: normal;
        }
        .sausage-wrap:hover .sausage-content {
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global font styling (Altair charts)
@alt.theme.register("arial_theme", enable=True)
def _altair_arial_theme():
    return alt.theme.ThemeConfig({
        "config": {
            "font": "Arial",
            "title": {"font": "Arial"},
            "axis": {"labelFont": "Arial", "titleFont": "Arial"},
            "legend": {"labelFont": "Arial", "titleFont": "Arial"},
            "header": {"labelFont": "Arial", "titleFont": "Arial"},
        }
    })

REQUIRED_COLUMNS = {
    "published_date",
    "rating",
    "helpful_votes",
    "text",
    "title",
    "published_platform",
    "type",
}

COLUMN_DEFINITIONS = {
    "published_date": "Date the review was published (UTC converted to local).",
    "rating": "Customer rating score (1 = lowest, 5 = highest).",
    "helpful_votes": "Number of users who marked the review as helpful.",
    "title": "Review title or headline.",
    "text": "Full review text.",
    "published_platform": "Where the review was posted (site/app/source).",
    "type": "Review category/type (e.g., cabin class, route, or review source type).",
}

COLUMN_ORDER = list(COLUMN_DEFINITIONS.keys())

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip("\"")
        .str.strip()
    )
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
    df["published_date"] = df["published_date"].dt.tz_convert(None)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce")
    df["text_length"] = df["text"].fillna("").str.len()
    return df

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def infer_type(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "Date"
    if is_numeric_dtype(series):
        s = series.dropna()
        if not s.empty and (s % 1 == 0).all():
            return "Integer"
        return "Decimal"
    return "Text"

def format_example(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "—"
    val = s.iloc[0]
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    text = str(val)
    if len(text) > 80:
        return text[:77] + "..."
    return text

def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, col in enumerate(COLUMN_ORDER, start=1):
        series = df[col] if col in df.columns else pd.Series([], dtype="object")
        rows.append(
            {
                "#": idx,
                "Column": col,
                "Type": infer_type(series),
                "Definition": COLUMN_DEFINITIONS.get(col, ""),
                "Example": format_example(series),
            }
        )
    return pd.DataFrame(rows)

def apply_filters(
    df: pd.DataFrame,
    date_range,
    rating_range,
    platforms,
) -> pd.DataFrame:
    filtered = df
    if platforms:
        filtered = filtered[filtered["published_platform"].isin(platforms)]
    if rating_range:
        filtered = filtered[filtered["rating"].between(rating_range[0], rating_range[1])]
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            filtered = filtered[
                (filtered["published_date"] >= pd.Timestamp(start_date))
                & (filtered["published_date"] <= pd.Timestamp(end_date))
            ]
    return filtered

def generate_key_takeaways(df: pd.DataFrame) -> str:
    """Generate 3 key insights from filtered data."""
    if len(df) == 0:
        return "No data available."
    
    negative_pct = df["rating"].between(1, 2).mean() * 100
    avg_rating = df["rating"].mean()
    top_platform = df["published_platform"].value_counts().idxmax() if df["published_platform"].notna().any() else "Unknown"
    
    return f"📊 {negative_pct:.1f}% of reviews are negative | ⭐ Avg rating: {avg_rating:.2f} | 📱 Most reviews from {top_platform}"

def _format_date_window(df: pd.DataFrame) -> str:
    date_data = df["published_date"].dropna() if "published_date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    if date_data.empty:
        return "Unknown period"
    return f"{date_data.min():%b %Y} to {date_data.max():%b %Y}"

def _format_date_range_duration(start_date, end_date) -> str:
    if not start_date or not end_date:
        return "Unknown duration"

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        start, end = end, start

    total_months = (end.year - start.year) * 12 + (end.month - start.month)
    years = total_months // 12
    months = total_months % 12

    parts = []
    if years:
        parts.append(f"{years} year" + ("s" if years != 1 else ""))
    if months:
        parts.append(f"{months} month" + ("s" if months != 1 else ""))

    if not parts:
        return "0 months"
    return " ".join(parts)

def build_download_recommendations(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Create a plain-language guide for what to download by common user goals."""
    return pd.DataFrame(
        [
            {
                "Goal": "Track overall customer health",
                "Suggested filters": "Use current date + platform filters",
                "Columns to download": "published_date, rating, published_platform",
                "Why this is enough": "Lets you monitor volume, sentiment trend, and channel performance quickly.",
            },
            {
                "Goal": "Investigate complaints",
                "Suggested filters": "Set rating range to 1-2",
                "Columns to download": "published_date, rating, published_platform, title, text, helpful_votes",
                "Why this is enough": "Helps identify pain points, when they happened, where they happened, and urgency.",
            },
            {
                "Goal": "Find what customers love",
                "Suggested filters": "Set rating range to 4-5",
                "Columns to download": "published_date, rating, published_platform, title, text",
                "Why this is enough": "Useful for best-practice examples, success stories, and experience benchmarks.",
            },
            {
                "Goal": "Prioritize high-impact feedback",
                "Suggested filters": "Any rating range; sort by helpful_votes after download",
                "Columns to download": "published_date, rating, published_platform, helpful_votes, title, text",
                "Why this is enough": "Focuses on feedback that other users found most useful and credible.",
            },
        ]
    )

def build_at_a_glance_trend_summary(time_series: pd.DataFrame) -> str:
    """Build compact trend context from overview's At-a-glance trend charts."""
    if time_series is None or len(time_series) < 2:
        return "Trend: insufficient monthly data"

    trend_df = time_series.sort_values("published_date").copy()
    trend_df = trend_df.dropna(subset=["published_date", "review_count", "avg_rating"])
    if len(trend_df) < 2:
        return "Trend: insufficient monthly data"

    first_row = trend_df.iloc[0]
    last_row = trend_df.iloc[-1]

    review_change = float(last_row["review_count"] - first_row["review_count"])
    rating_change = float(last_row["avg_rating"] - first_row["avg_rating"])

    peak_volume_row = trend_df.loc[trend_df["review_count"].idxmax()]
    best_rating_row = trend_df.loc[trend_df["avg_rating"].idxmax()]
    worst_rating_row = trend_df.loc[trend_df["avg_rating"].idxmin()]

    volume_direction = "increasing" if review_change > 0 else "decreasing" if review_change < 0 else "stable"
    rating_direction = "improving" if rating_change > 0 else "declining" if rating_change < 0 else "stable"

    return (
        f"Trend: review volume {volume_direction} ({review_change:+.0f} from "
        f"{first_row['published_date']:%b %Y} to {last_row['published_date']:%b %Y}); "
        f"average rating {rating_direction} ({rating_change:+.2f}); "
        f"peak volume in {peak_volume_row['published_date']:%b %Y} ({peak_volume_row['review_count']:.0f} reviews); "
        f"best rating in {best_rating_row['published_date']:%b %Y} ({best_rating_row['avg_rating']:.2f}); "
        f"lowest rating in {worst_rating_row['published_date']:%b %Y} ({worst_rating_row['avg_rating']:.2f})"
    )

def generate_dataset_overview(df: pd.DataFrame, time_series: pd.DataFrame, model_choice: str = "auto") -> str:
    """Generate AI-powered narrative dataset overview to orient first-time users.
    
    Args:
        df: Filtered dataframe
        model_choice: 'groq', 'ollama', or 'auto' (default)
    """
    import os
    
    if len(df) == 0:
        return "📊 **Dataset Overview:** No data available with current filters."
    
    # Create dataset summary for AI to contextualize
    total_reviews = len(df)
    avg_rating = df["rating"].mean()
    negative_pct = df["rating"].between(1, 2).mean() * 100
    positive_pct = df["rating"].between(4, 5).mean() * 100
    date_range = f"{df['published_date'].min().strftime('%b %Y')} to {df['published_date'].max().strftime('%b %Y')}" if "published_date" in df.columns else "Unknown period"
    platforms = ", ".join(df["published_platform"].value_counts().head(3).index.tolist()) if df["published_platform"].notna().any() else "Multiple platforms"
    trend_summary = build_at_a_glance_trend_summary(time_series)
    
    data_summary = (
        f"Reviews: {total_reviews}, Avg Rating: {avg_rating:.2f}, Negative: {negative_pct:.0f}%, "
        f"Positive: {positive_pct:.0f}%, Period: {date_range}, Platforms: {platforms}, {trend_summary}"
    )
    
    prompt = f"Briefly describe what this Singapore Airlines review dataset shows (1-2 sentences). Context: {data_summary}"
    
    # Route to selected model
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if model_choice == "groq":
        if groq_api_key:
            return _generate_groq_insight(prompt, groq_api_key)
        else:
            return "📊 **Dataset Overview:** Groq not configured. Using Ollama...\n" + _generate_ollama_insight(prompt)
    elif model_choice == "ollama":
        return _generate_ollama_insight(prompt)
    else:  # auto
        if groq_api_key:
            return _generate_groq_insight(prompt, groq_api_key)
        return _generate_ollama_insight(prompt)

def generate_chart_insight(df: pd.DataFrame, x_col: str, y_col: str, agg_label: str) -> str:
    """Generate an insight about the chart data."""
    if y_col == "(count)":
        max_group = df.groupby(x_col, dropna=True).size().idxmax()
        return f"💡 **Insight:** The highest concentration is in the '{max_group}' category."
    else:
        max_val = df.groupby(x_col, dropna=True)[y_col].agg(agg_label).max()
        min_val = df.groupby(x_col, dropna=True)[y_col].agg(agg_label).min()
        diff = max_val - min_val
        return f"💡 **Insight:** There's a {diff:.2f} difference between the highest and lowest {agg_label} values across categories."

def format_column_with_type(col: str, df: pd.DataFrame) -> str:
    """Format column name with data type icon."""
    if col == "(count)":
        return "📊 (count)"
    if is_numeric_dtype(df[col]):
        return f"📊 {col}"
    else:
        return f"🔤 {col}"

def get_formatted_columns(cols: list, df: pd.DataFrame) -> dict:
    """Return mapping of formatted names to original column names."""
    return {format_column_with_type(col, df): col for col in cols}

def generate_ai_insight(df: pd.DataFrame, x_col: str, y_col: str, agg_label: str, model_choice: str = "auto") -> str:
    """Generate AI-powered insight using specified model or auto-detect.
    
    Args:
        model_choice: 'groq', 'ollama', or 'auto' (default)
    """
    import os
    
    # Generate data summary (limit to top 10 items to avoid token limits)
    if y_col == "(count)":
        summary = f"Review frequency by {x_col}"
        grouped = df.groupby(x_col, dropna=True).size().sort_values(ascending=False)
    else:
        summary = f"{agg_label.capitalize()} of {y_col} by {x_col}"
        grouped = df.groupby(x_col, dropna=True)[y_col].agg(agg_label).sort_values(ascending=False)
    
    # Limit to top 10 items to keep prompt small
    data_desc = grouped.head(10).to_dict()
    
    # Shorter, simpler prompt to reduce token count
    prompt = f"Data: {data_desc}. One insight about '{summary}' in airline reviews (1-2 sentences)."
    
    # Route to selected model
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    if model_choice == "groq":
        if groq_api_key:
            return _generate_groq_insight(prompt, groq_api_key)
        else:
            return "💡 **AI Insight:** Groq model selected but GROQ_API_KEY not set. Using Ollama fallback..."
    elif model_choice == "ollama":
        return _generate_ollama_insight(prompt)
    else:  # auto
        # Try Groq first if available
        if groq_api_key:
            return _generate_groq_insight(prompt, groq_api_key)
        # Fallback to Ollama
        return _generate_ollama_insight(prompt)

def _generate_groq_insight(prompt: str, api_key: str) -> str:
    """Generate insight using Groq API."""
    try:
        from groq import Groq
    except ImportError as e:
        # More detailed error message for debugging
        import sys
        return f"💡 **AI Insight:** Groq not found. Paths checked: {sys.path[-3:]}. Error: {str(e)}"
    
    try:
        # Validate API key format
        if not api_key or not api_key.startswith("gsk_"):
            return "💡 **AI Insight:** Invalid Groq API key format. Key should start with 'gsk_'."
        
        client = Groq(api_key=api_key)
        
        # Ensure prompt is not empty
        if not prompt or len(prompt.strip()) == 0:
            return "💡 **AI Insight:** Empty prompt provided."
        
        message = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",  # Fast, free tier model
            max_tokens=100,  # Reduced to avoid token limits
        )
        insight_text = message.choices[0].message.content.strip()
        return f"✨ **AI Insight (Groq):** {insight_text}"
    except Exception as e:
        error_msg = str(e)
        # Show more of the error for debugging
        return f"💡 **AI Insight:** Groq error. {error_msg[:100]} Contact support or check API key."

def _generate_ollama_insight(prompt: str) -> str:
    """Generate insight using Ollama (local only)."""
    try:
        import requests
    except ImportError:
        return "💡 **AI Insight:** Install requests library (pip install requests) to enable insights."
    
    # Check if running on Streamlit Cloud
    import os
    is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
    
    if is_cloud:
        return "💡 **AI Insight:** Set GROQ_API_KEY env var for cloud AI insights. (Or run locally with Ollama: `ollama serve`)"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            insight_text = result.get('response', 'No insight generated').strip()
            return f"✨ **AI Insight (Ollama):** {insight_text}"
        else:
            return "💡 **AI Insight:** Ollama API error. Ensure Ollama is running: `ollama serve`"
    
    except requests.exceptions.ConnectionError:
        return "💡 **AI Insight:** Ollama not running. Start it with: `ollama serve`"
    except requests.exceptions.Timeout:
        return "💡 **AI Insight:** Ollama timeout. Model may still be loading."
    except Exception as e:
        return f"💡 **AI Insight:** Error ({str(e)[:40]})"

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except Exception:
    WordCloud = None
    STOPWORDS = set()
    WORDCLOUD_AVAILABLE = False

def render_wordcloud(text_series, stopwords, title):
    if not WORDCLOUD_AVAILABLE:
        st.info("Keyword clouds require the `wordcloud` package. Install with `pip install wordcloud`.")
        return
    text = " ".join(text_series.dropna().astype(str).tolist())
    if not text.strip():
        st.info("Not enough text for a keyword cloud.")
        return
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
    ).generate(text)
    st.write(title)
    st.image(wc.to_array(), use_container_width=True)

st.markdown(
    """
    <div class="title-hover-wrap">
        <h1>Self‑Service Data Hub: SIA Google Reviews</h1>
        <span class="title-hover-tooltip">Created by Stan</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("A self-service dashboard to explore the data profile, filter, and export Singapore Airlines review data for your analysis needs. Use the sidebar/filters panel to customize your view and download the dataset that fits your goals.")

df = load_data("data/singapore_airlines_reviews.csv")

missing_cols = REQUIRED_COLUMNS - set(df.columns)
if missing_cols:
    st.error(f"Missing columns in dataset: {', '.join(sorted(missing_cols))}")
    st.stop()

min_date = df["published_date"].min()
max_date = df["published_date"].max()
default_end = max_date.date() if pd.notnull(max_date) else None
default_start = ((max_date - pd.DateOffset(months=12)).date() if pd.notnull(max_date) else None)

min_rating = int(df["rating"].min()) if df["rating"].notna().any() else 1
max_rating = int(df["rating"].max()) if df["rating"].notna().any() else 5

with st.sidebar:
    download_container = st.container()
    st.header(
        "Filters",
        help="Quickly narrow the dataset and download the filtered results.",
    )

    with st.expander("Date range", expanded=True):
        if pd.isnull(min_date) or pd.isnull(max_date):
            date_range = None
        else:
            month_starts = pd.date_range(
                min_date.to_period("M").to_timestamp(),
                max_date.to_period("M").to_timestamp(),
                freq="MS",
            ).date
            def to_month_start(value):
                if not value:
                    return None
                return pd.Timestamp(value).to_period("M").to_timestamp().date()

            fixed_start_date = pd.Timestamp("2023-03-01").date()
            fixed_end_date = pd.Timestamp("2024-03-01").date()
            if fixed_start_date < month_starts[0]:
                fixed_start_date = month_starts[0]
            if fixed_end_date > month_starts[-1]:
                fixed_end_date = month_starts[-1]

            if st.button("↺ Reset", key="reset_date", use_container_width=True):
                st.session_state["date_range_slider"] = (fixed_start_date, fixed_end_date)

            stored_range = st.session_state.get("date_range_slider")
            if stored_range:
                stored_start = to_month_start(stored_range[0])
                stored_end = to_month_start(stored_range[1])
                if stored_start in month_starts and stored_end in month_starts:
                    slider_value = (stored_start, stored_end)
                else:
                    slider_value = (fixed_start_date, fixed_end_date)
            else:
                slider_value = (fixed_start_date, fixed_end_date)

            date_range = st.select_slider(
                "Date range",
                options=list(month_starts),
                value=slider_value,
                format_func=lambda d: d.strftime("%b %Y"),
                key="date_range_slider",
            )
            selected_duration = _format_date_range_duration(date_range[0], date_range[1])
            st.caption(
                f"Selected: {date_range[0].strftime('%b %Y')} → {date_range[1].strftime('%b %Y')} ({selected_duration})"
            )

    with st.expander("Rating range", expanded=False):
        rating_range = st.slider(
            "Rating range",
            min_value=min_rating,
            max_value=max_rating,
            value=(min_rating, max_rating),
            step=1,
            help="Overall customer rating (1 = lowest, 5 = highest).",
        )

    platform_options = sorted(df["published_platform"].dropna().unique())
    with st.expander("Platform", expanded=False):
        platforms = st.multiselect(
            "Platform",
            options=platform_options,
            default=platform_options,
            help="Where the review was published (e.g., site/app/source).",
        )
    
    with st.expander("✨ AI Model", expanded=False):
        ai_model_choice = st.radio(
            "Select AI model for insights:",
            options=["Auto", "Groq (Cloud)", "Ollama (Local)"],
            help="Auto: Uses Groq if API key is set, otherwise Ollama\nGroq: Fast cloud-based model (requires API key)\nOllama: Offline local model (requires running service)",
            horizontal=False,
        )
        # Map radio selection to model choice
        model_map = {"Auto": "auto", "Groq (Cloud)": "groq", "Ollama (Local)": "ollama"}
        st.session_state.selected_ai_model = model_map[ai_model_choice]
    
    with st.expander("", expanded=False):
        st.caption("Hidden Feature - Columns Selection: Select which columns to include in your download (Above filters select which rows to include, this selects which columns).")
        
        # Initialize checkbox states on first load
        for col in COLUMN_ORDER:
            checkbox_key = f"col_checkbox_{col}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = True  # Default all to checked
        
        # Initialize selected_columns if not set
        if "selected_columns" not in st.session_state:
            st.session_state.selected_columns = list(COLUMN_ORDER)
        
        # Toggle Select All / Deselect All button
        all_selected = len(st.session_state.selected_columns) == len(COLUMN_ORDER)
        button_label = "Deselect All" if all_selected else "Select All"
        button_help = "Uncheck all columns" if all_selected else "Check all columns"
        
        if st.button(button_label, key="toggle_all", use_container_width=True, help=button_help):
            if all_selected:
                st.session_state.selected_columns = []
                # Update all checkbox states to unchecked
                for col in COLUMN_ORDER:
                    st.session_state[f"col_checkbox_{col}"] = False
            else:
                st.session_state.selected_columns = list(COLUMN_ORDER)
                # Update all checkbox states to checked
                for col in COLUMN_ORDER:
                    st.session_state[f"col_checkbox_{col}"] = True
            st.rerun()
        
        # Checkbox list for column selection
        st.markdown("**Columns to export:**")
        
        # Render checkboxes and build selected columns list
        selected_cols = []
        for col in COLUMN_ORDER:
            if st.checkbox(col, key=f"col_checkbox_{col}"):
                selected_cols.append(col)
        
        # Update session state with current selection
        st.session_state.selected_columns = selected_cols
        
        # Visual feedback
        col_count = len(st.session_state.selected_columns)
        if col_count == 0:
            st.warning("⚠️ Select at least one column")
        else:
            st.caption(f"✓ {col_count} of {len(COLUMN_ORDER)} columns selected")



date_filter = date_range
filtered = apply_filters(df, date_filter, rating_range, platforms)

with download_container:
    # Get selected columns or use all if none selected
    download_cols = st.session_state.get("selected_columns", list(COLUMN_ORDER))
    if not download_cols:
        download_cols = list(COLUMN_ORDER)
    
    # Filter dataframe to selected columns
    download_df = filtered[download_cols]
    col_count = len(download_cols)
    
    st.download_button(
        f"⬇ Download filtered .csv file ({col_count} column{'s' if col_count != 1 else ''})",
        data=to_csv_bytes(download_df),
        file_name="sia_reviews_filtered.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
        help=f"Export {len(filtered):,} reviews with {col_count} selected columns",
    )

if len(filtered) == 0:
    st.info("No reviews match the current filters.")
    st.stop()

# Shared time series for overview + trends
time_series = (
    filtered.dropna(subset=["published_date"])
    .set_index("published_date")
    .resample("MS")
    .agg(
        review_count=("rating", "size"),
        avg_rating=("rating", "mean"),
    )
    .reset_index()
)

tab_ml_predict, tab_overview, tab_explore, tab_export = st.tabs(
    ["🔍 Review Analyzer", "Overview", "Data Exploration", "Data & Export"]
)

with tab_export:
    st.caption("Understand column characteristics and data context, preview a sample of the raw dataset, and download when you're ready.")
    
    st.markdown(
        """
        <div class="guide-hover-wrap">
            <span class="guide-hover-trigger">ℹ️ New here? Hover for help</span>
            <span class="guide-hover-tooltip">Start with Download Planning Guide.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("🧭 Download planning guide", expanded=False):
        st.caption("For non-technical users: use this quick guide to decide what data to export for your task.")

        in_scope_reviews = len(filtered)
        total_reviews = len(df)
        scope_pct = (in_scope_reviews / total_reviews * 100) if total_reviews else 0
        neg_reviews = int(filtered["rating"].between(1, 2).sum())
        pos_reviews = int(filtered["rating"].between(4, 5).sum())
        selected_platform_count = filtered["published_platform"].dropna().nunique()
        selected_period = _format_date_window(filtered)

        summary_cols = st.columns(4)
        summary_cols[0].metric("Reviews in current scope", f"{in_scope_reviews:,}", f"{scope_pct:.1f}% of total")
        summary_cols[1].metric("Date period", selected_period)
        summary_cols[2].metric("Platforms covered", f"{selected_platform_count}")
        summary_cols[3].metric("Negative vs positive", f"{neg_reviews} / {pos_reviews}")

        st.info(
            "Start with the smallest dataset that answers your question. "
            "If you only need trends, download fewer columns; include `title`/`text` only when you need detailed root-cause analysis."
        )

        st.markdown("**What should I download?**")
        recommendation_df = build_download_recommendations(filtered)
        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "Dataset dictionary",
        help="Use this field guide to understand each column and select only what you need before downloading.",
    )
    dict_df = build_data_dictionary(df)
    st.dataframe(
        dict_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Example": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader(
        "Raw Data",
        help="Preview the actual filtered rows to confirm relevance before downloading.",
    )
    show_cols = ["published_date", "rating", "published_platform", "title", "text", "helpful_votes"]
    row_option = st.radio(
        "Rows to display",
        ["First 10", "Last 10", "Custom range"],
        horizontal=True,
    )
    base_df = filtered[show_cols]
    if row_option == "First 10":
        sample_df = base_df.head(10)
    elif row_option == "Last 10":
        sample_df = base_df.tail(10)
    else:
        total_rows = len(base_df)
        default_end = min(total_rows, 10)
        start_row, end_row = st.slider(
            "Row range (1-based)",
            min_value=1,
            max_value=max(total_rows, 1),
            value=(1, default_end),
            step=1,
        )
        start_idx = max(start_row - 1, 0)
        sample_df = base_df.iloc[start_idx:end_row]
    st.dataframe(
        sample_df,
        use_container_width=True,
        hide_index=True,
    )

with tab_overview:
    st.caption("Get a high-level summary of your filtered dataset with AI-generated insights, key metrics, and trend visualizations to quickly understand review patterns and sentiment.")
    
    st.info(
        generate_key_takeaways(filtered),
        icon="💡"
    )

    # Get selected AI model from session state for Dataset Overview
    selected_model = st.session_state.get("selected_ai_model", "auto")
    st.markdown(generate_dataset_overview(filtered, time_series, selected_model))

    st.divider()

    st.subheader("At-a-glance trends")
    st.caption("Track how review volume and ratings have changed over time to spot trends and patterns.")
    trend_cols = st.columns(2)

    with trend_cols[0]:
        volume_chart = (
            alt.Chart(time_series)
            .mark_line(point=True, color="#ff7f0e")
            .encode(
                x=alt.X("published_date:T", title="Month"),
                y=alt.Y("review_count:Q", title="Reviews"),
                tooltip=[
                    alt.Tooltip("published_date:T", title="Month"),
                    alt.Tooltip("review_count:Q", title="Reviews"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(volume_chart, use_container_width=True)

    with trend_cols[1]:
        rating_trend_chart = (
            alt.Chart(time_series)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x=alt.X("published_date:T", title="Month"),
                y=alt.Y("avg_rating:Q", title="Average rating", scale=alt.Scale(domain=[0, 5])),
                tooltip=[
                    alt.Tooltip("published_date:T", title="Month"),
                    alt.Tooltip("avg_rating:Q", title="Avg rating", format=".2f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(rating_trend_chart, use_container_width=True)

    st.subheader("Ratings & sentiment")
    st.caption("See how reviews are distributed across ratings and overall sentiment breakdown.")
    dist_cols = st.columns(2)

    with dist_cols[0]:
        rating_counts = (
            filtered.groupby("rating", dropna=True)
            .size()
            .reset_index(name="count")
            .sort_values("rating")
        )
        rating_chart = (
            alt.Chart(rating_counts)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("rating:O", title="Rating", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Reviews"),
                tooltip=["rating:O", "count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(rating_chart, use_container_width=True)

    with dist_cols[1]:
        sentiment_df = pd.DataFrame(
            {
                "Sentiment": ["Positive (4-5)", "Neutral (3)", "Negative (1-2)"],
                "Count": [
                    filtered["rating"].between(4, 5).sum(),
                    filtered["rating"].eq(3).sum(),
                    filtered["rating"].between(1, 2).sum(),
                ],
            }
        )
        sentiment_df["Percent"] = (sentiment_df["Count"] / sentiment_df["Count"].sum()).fillna(0)

        sentiment_chart = (
            alt.Chart(sentiment_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Sentiment:N",
                    scale=alt.Scale(range=["#2ca02c", "#ffbf00", "#d62728"]),
                ),
                tooltip=[
                    "Sentiment:N",
                    alt.Tooltip("Count:Q", title="Reviews"),
                    alt.Tooltip("Percent:Q", title="Share", format=".1%"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(sentiment_chart, use_container_width=True)

    st.subheader("Where reviews come from")
    st.caption("Compare review volume and average ratings across different platforms.")
    platform_cols = st.columns(2)

    with platform_cols[0]:
        platform_volume = (
            filtered.groupby("published_platform", dropna=True)
            .size()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
        )
        platform_volume_chart = (
            alt.Chart(platform_volume)
            .mark_bar(color="#9467bd")
            .encode(
                x=alt.X("value:Q", title="Reviews"),
                y=alt.Y("published_platform:N", title=None, sort="-x"),
                tooltip=["published_platform:N", "value:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(platform_volume_chart, use_container_width=True)

    with platform_cols[1]:
        platform_avg = (
            filtered.groupby("published_platform", dropna=True)["rating"]
            .mean()
            .reset_index(name="value")
            .sort_values("value", ascending=False)
        )
        platform_avg_chart = (
            alt.Chart(platform_avg)
            .mark_bar(color="#2ca02c")
            .encode(
                x=alt.X("value:Q", title="Average rating", scale=alt.Scale(domain=[0, 5])),
                y=alt.Y("published_platform:N", title=None, sort="-x"),
                tooltip=["published_platform:N", alt.Tooltip("value:Q", format=".2f")],
            )
            .properties(height=280)
        )
        st.altair_chart(platform_avg_chart, use_container_width=True)

    with st.expander("📝 Text Insights", expanded=False):
        st.subheader("Text length vs rating")
        st.caption("Explore if longer reviews tend to receive higher or lower ratings.")
        text_length_chart = (
            alt.Chart(filtered.dropna(subset=["rating", "text_length"]))
            .mark_circle(size=60, opacity=0.35, color="#8c564b")
            .encode(
                x=alt.X("rating:Q", title="Rating", scale=alt.Scale(domain=[1, 5])),
                y=alt.Y("text_length:Q", title="Text length (characters)"),
                tooltip=[
                    alt.Tooltip("rating:Q", title="Rating"),
                    alt.Tooltip("text_length:Q", title="Text length"),
                    alt.Tooltip("published_platform:N", title="Platform"),
                ],
            )
            .properties(height=320)
        )

        trend_line = text_length_chart.transform_regression(
            "rating",
            "text_length",
        ).mark_line(color="#1f77b4")

        st.altair_chart(text_length_chart + trend_line, use_container_width=True)

        st.subheader("Helpfulness insights")
        st.caption("Understand which reviews are marked as helpful and what characteristics they share.")
        helpful_data = filtered.dropna(subset=["rating", "helpful_votes", "text_length"])
        helpful_cols = st.columns(2)

        with helpful_cols[0]:
            helpful_by_rating = (
                helpful_data.groupby("rating", dropna=True)["helpful_votes"]
                .mean()
                .reset_index(name="avg_helpful")
                .sort_values("rating")
            )
            helpful_rating_chart = (
                alt.Chart(helpful_by_rating)
                .mark_bar(color="#e377c2")
                .encode(
                    x=alt.X("rating:O", title="Rating"),
                    y=alt.Y("avg_helpful:Q", title="Avg helpful votes"),
                    tooltip=["rating:O", alt.Tooltip("avg_helpful:Q", format=".2f")],
                )
                .properties(height=280)
            )
            st.altair_chart(helpful_rating_chart, use_container_width=True)

        with helpful_cols[1]:
            helpful_length_chart = (
                alt.Chart(helpful_data)
                .mark_circle(size=60, opacity=0.35, color="#7f7f7f")
                .encode(
                    x=alt.X("text_length:Q", title="Text length (characters)"),
                    y=alt.Y("helpful_votes:Q", title="Helpful votes"),
                    tooltip=[
                        alt.Tooltip("text_length:Q", title="Text length"),
                        alt.Tooltip("helpful_votes:Q", title="Helpful votes"),
                        alt.Tooltip("rating:Q", title="Rating"),
                    ],
                )
                .properties(height=280)
            )
            helpful_trend = helpful_length_chart.transform_regression(
                "text_length",
                "helpful_votes",
            ).mark_line(color="#1f77b4")
            st.altair_chart(helpful_length_chart + helpful_trend, use_container_width=True)

        st.subheader("Keyword clouds")
        st.caption("Discover the most frequently mentioned words in positive and negative reviews.")
        base_stopwords = set(STOPWORDS) if STOPWORDS else set()
        base_stopwords.update(
            {
                "singapore",
                "airlines",
                "airline",
                "flight",
                "flights",
                "crew",
                "seat",
                "seats",
                "plane",
                "aircraft",
            }
        )

        pos_reviews = filtered[filtered["rating"].between(4, 5)]["text"]
        neg_reviews = filtered[filtered["rating"].between(1, 2)]["text"]

        pos_col, neg_col = st.columns(2)
        with pos_col:
            st.subheader("Positive reviews (4–5)")
            render_wordcloud(pos_reviews, base_stopwords, "Top positive keywords")
        with neg_col:
            st.subheader("Negative reviews (1–2)")
            render_wordcloud(neg_reviews, base_stopwords, "Top negative keywords")

        st.subheader("Title vs full review keywords")
        title_col, text_col = st.columns(2)
        with title_col:
            st.subheader("Titles")
            render_wordcloud(filtered["title"], base_stopwords, "Top title keywords")
        with text_col:
            st.subheader("Full reviews")
            render_wordcloud(filtered["text"], base_stopwords, "Top full-review keywords")

with tab_explore:
    
    st.caption("Explore your data through visualizations. Choose a preset or build your own.")

    explore_cols = filtered.columns.tolist()
    numeric_cols = [col for col in explore_cols if is_numeric_dtype(filtered[col])]
    
    # Create formatted column mappings
    x_col_mapping = get_formatted_columns(explore_cols, filtered)
    y_col_options = ["(count)"] + numeric_cols
    y_col_mapping = get_formatted_columns(y_col_options, filtered)

    # ===== POPULAR EXPLORATIONS (Preset Templates) =====
    st.subheader("📊 Popular Explorations", help="Click any preset to quickly explore common insights")
    
    preset_cols = st.columns(3)
    presets = [
        ("📈 Rating Trends", "published_date", "rating", "mean"),
        ("📱 Platform Comparison", "published_platform", "rating", "mean"),
        ("💬 Review Volume by Platform", "published_platform", "(count)", "count"),
    ]
    
    preset_selected = None
    for idx, (preset_name, x, y, agg) in enumerate(presets):
        with preset_cols[idx % 3]:
            if st.button(preset_name, use_container_width=True, help=f"Explore {preset_name.lower()}"):
                preset_selected = (x, y, agg)
                st.session_state.preset_choice = preset_selected
    
    st.divider()

    # ===== CHART BUILDER =====
    st.subheader("🛠️ Custom Chart Builder", help="Build your own exploration by selecting columns")
    
    with st.expander("✏️ Customize Chart", expanded=True):
        control_cols = st.columns([1.5, 1.5, 1.5])
        with control_cols[0]:
            chart_type = st.selectbox(
                "Chart type",
                ["Bar", "Line", "Area", "Scatter", "Box Plot"],
                help="Bar: Categories | Line: Time series | Area: Trends over time | Scatter: Relationships | Box: Distribution"
            )
        with control_cols[1]:
            x_col_formatted = st.selectbox(
                "X axis",
                options=list(x_col_mapping.keys()),
                index=list(x_col_mapping.keys()).index(format_column_with_type("published_date", filtered)) if "published_date" in explore_cols else 0,
                help="📊 Numeric or date | 🔤 Text/category"
            )
            x_col = x_col_mapping[x_col_formatted]
        with control_cols[2]:
            y_col_formatted = st.selectbox(
                "Y axis",
                options=list(y_col_mapping.keys()),
                index=0 if not numeric_cols else 1,
                help="(count) = review frequency | Metric = aggregate values"
            )
            y_col = y_col_mapping[y_col_formatted]

        if y_col == "(count)":
            agg_label = "count"
    
    # Use preset if selected, otherwise use custom selection
    if "preset_choice" in st.session_state:
        x_col, y_col, agg_type = st.session_state.preset_choice
        chart_type = "Line" if x_col == "published_date" else "Bar"
        agg_label = agg_type
        # Clear preset for next interaction
        del st.session_state.preset_choice
    
    if y_col == "(count)":
        agg_label = "count"
        plot_df = (
            filtered.groupby(x_col, dropna=True)
            .size()
            .reset_index(name="value")
        )
    else:
        if "agg_label" not in locals():
            agg_func = st.selectbox("Aggregation", ["mean", "sum", "median"], index=0)
            agg_label = agg_func
        plot_df = (
            filtered.groupby(x_col, dropna=True)[y_col]
            .agg(agg_label)
            .reset_index(name="value")
        )

    x_dtype = "T" if is_datetime64_any_dtype(filtered[x_col]) else "O"
    if chart_type == "Line" and x_dtype == "O":
        st.info("💡 Tip: Line charts work best with time or numeric X axes. Consider a bar chart if X is categorical.")

    st.markdown(f"**Chart:** {agg_label.capitalize()} of {y_col if y_col != '(count)' else 'reviews'} by {x_col}")
    
    # Economist-style minimalist color palette
    chart_color = "#6b6b6b"  # Subtle grey
    accent_color = "#003d7a"  # Deep blue accent
    
    chart = alt.Chart(plot_df).encode(
        x=alt.X(f"{x_col}:{x_dtype}", title=x_col),
        y=alt.Y("value:Q", title=f"{agg_label} of {y_col if y_col != '(count)' else 'rows'}"),
        tooltip=[alt.Tooltip(f"{x_col}:{x_dtype}", title=x_col), alt.Tooltip("value:Q", title=agg_label)],
    ).properties(height=320)
    
    # Apply chart type
    if chart_type == "Bar":
        chart = chart.mark_bar(color=chart_color)
    elif chart_type == "Line":
        chart = chart.mark_line(point=True, color=chart_color, interpolate='monotone')
    elif chart_type == "Area":
        chart = chart.mark_area(color=chart_color, opacity=0.6, interpolate='monotone')
    elif chart_type == "Scatter":
        chart = chart.mark_point(color=accent_color, size=100)
    elif chart_type == "Box Plot":
        # Box plots require raw data, not aggregated
        if y_col != "(count)":
            chart = alt.Chart(filtered).encode(
                x=alt.X(f"{x_col}:{x_dtype}", title=x_col),
                y=alt.Y(f"{y_col}:Q", title=y_col),
                tooltip=[alt.Tooltip(f"{x_col}:{x_dtype}", title=x_col), alt.Tooltip(f"{y_col}:Q", title=y_col)],
            ).mark_boxplot(color=chart_color).properties(height=320)
        else:
            st.warning("📊 Box plots require a numeric Y axis. Please select a metric instead of (count).")

    st.altair_chart(chart, use_container_width=True)
    
    st.markdown(generate_chart_insight(filtered, x_col, y_col, agg_label))
    # Get selected AI model from session state (default to auto)
    selected_model = st.session_state.get("selected_ai_model", "auto")
    st.markdown(generate_ai_insight(filtered, x_col, y_col, agg_label, selected_model))

models = load_ml_models()

with tab_ml_predict:
    st.caption("Analyze a customer review to estimate star rating and identify the right team for follow-up.")
    
    if not models:
        st.error("ML models not found. Please run `train_model.py` first.")
    else:
        # Define aspect taxonomy
        ASPECT_TAXONOMY = {
            "Food & Beverage": ["food", "meal", "drink", "water", "wine", "chicken", "beef", "breakfast", "lunch", "dinner", "taste", "menu", "beverage", "hungry", "thirsty"],
            "Seat & Comfort": ["seat", "comfort", "legroom", "recline", "space", "cramped", "narrow", "sleep", "bed", "aisle", "window", "sore", "uncomfortable"],
            "Staff & Service": ["crew", "staff", "attendant", "steward", "stewardess", "rude", "friendly", "polite", "helpful", "service", "smile", "ignored", "professional", "attendants"],
            "Flight Punctuality": ["delay", "delayed", "late", "wait", "cancel", "cancelled", "hours", "schedule", "missed", "connection", "waiting"],
            "Baggage Handling": ["baggage", "bag", "luggage", "lost", "belt", "claim", "carousel", "damaged"],
            "Inflight Entertainment": ["wifi", "internet", "movie", "screen", "krisworld", "tv", "entertainment", "monitor", "headphone", "movies"],
            "Booking & Check-in": ["website", "app", "check-in", "checkin", "online", "booking", "system", "payment", "error", "counter", "boarding", "ticket", "tickets"]
        }
        
        def extract_aspect_tags(text, use_llm=True):
            analyzer = get_vader_analyzer()
            # Split text keeping delimiters to reconstruct original text
            segments = re.split(r'([.!?\n]+)', text)
            analysis_meta = {
                "engine_label": "Standard ML + VADER",
                "engine_tier": "Local",
                "status": "Using local routing engine",
                "llm_attempted": False,
                "llm_used": False,
            }
            
            # --- CONTEXT-AWARE LLM UPGRADE ---
            llm_sentiments = {} # Dictionary mapping segment index -> (topics_list, sentiment_score)
            
            model_choice = st.session_state.get("selected_ai_model", "auto")
            api_key = os.environ.get("GROQ_API_KEY", "")
            use_groq = (model_choice == "groq" and api_key) or (model_choice == "auto" and api_key)
            use_ollama = (model_choice == "ollama") or (model_choice == "auto" and not api_key)
            
            if use_llm and (use_groq or use_ollama):
                analysis_meta["llm_attempted"] = True
                if use_groq:
                    analysis_meta["engine_label"] = "Groq LLM"
                    analysis_meta["engine_tier"] = "Cloud"
                    analysis_meta["status"] = "Using Groq for context-aware routing"
                elif use_ollama:
                    analysis_meta["engine_label"] = "Ollama LLM"
                    analysis_meta["engine_tier"] = "Local LLM"
                    analysis_meta["status"] = "Using Ollama for context-aware routing"

                valid_sentences = []
                for idx, seg in enumerate(segments):
                    if seg.strip() and not re.fullmatch(r'[.!?\n]+', seg):
                        words = set(re.findall(r'\b\w+\b', seg.lower()))
                        if words:
                            valid_sentences.append({"id": idx, "text": seg.strip()})
                            
                if valid_sentences:
                    prompt = (
                        "You are a specialized aviation data parser. Respond ONLY with a valid JSON array. No preamble. No markdown. No explanations.\n\n"
                        "STRICT OUTPUT SCHEMA:\n"
                        '[{"id": integer, "topics": array of strings, "sentiment": "Positive"|"Negative"|"Neutral"}]\n\n'
                        "TAXONOMY RULES:\n"
                        "1. Use ONLY these topics: ['Food & Beverage', 'Seat & Comfort', 'Staff & Service', 'Flight Punctuality', 'Baggage Handling', 'Inflight Entertainment', 'Booking & Check-in'].\n"
                        "2. EXPLICIT COMPETITOR MENTIONS: If they praise a competitor (e.g., 'Emirates has better seats'), that is a 'Negative' sentiment for SIA.\n"
                        "3. SARCASM: Phrases like 'Thanks for the 5-hour wait' are 'Negative'.\n"
                        "4. FACTUAL: If no topic is mentioned, return [].\n\n"
                        "ONE-SHOT EXAMPLE:\n"
                        '[{"id": 0, "topics": ["Food & Beverage"], "sentiment": "Positive"}, {"id": 1, "topics": [], "sentiment": "Neutral"}]\n\n'
                        "INPUT DATA:\n"
                    )
                    prompt += json.dumps(valid_sentences)

                    
                    try:
                        llm_out = ""
                        if use_groq:
                            from groq import Groq
                            client = Groq(api_key=api_key)
                            resp = client.chat.completions.create(
                                messages=[{"role": "system", "content": "You are a JSON API. Output raw JSON array only, no markdown."}, {"role": "user", "content": prompt}],
                                model="llama-3.1-8b-instant", temperature=0
                            )
                            llm_out = resp.choices[0].message.content or ""
                        else:
                            import requests
                            res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3", "prompt": prompt, "system": "Output bare JSON array only.", "stream": False}, timeout=60)
                            if res.status_code != 200:
                                raise Exception(f"Ollama returned {res.status_code}: {res.text}")
                            llm_out = res.json().get("response", "") or ""

                        llm_out = str(llm_out)
                            
                        if not llm_out.strip():
                            raise Exception("LLM returned an entirely blank string.")
                        
                        # Strip markdown code blocks if present
                        llm_out = re.sub(r'```(?:json)?\s*', '', llm_out)
                        llm_out = re.sub(r'\s*```', '', llm_out)
                            
                        # Robust JSON array extraction
                        start_idx = llm_out.find('[')
                        end_idx = llm_out.rfind(']')
                        
                        if start_idx == -1 or end_idx == -1:
                            raise Exception(f"Failed to find JSON boundaries in LLM response: {llm_out[:100]}...")
                            
                        clean_json = llm_out[start_idx:end_idx+1]

                        def _normalize_json_text(raw_text):
                            txt = raw_text.strip()
                            txt = txt.replace("\u201c", '"').replace("\u201d", '"')
                            txt = txt.replace("\u2018", "'").replace("\u2019", "'")
                            # Remove trailing commas before object/array close.
                            txt = re.sub(r",\s*([}\]])", r"\1", txt)
                            # Quote common bare keys if present.
                            txt = re.sub(r'([{,]\s*)(id|topics|sentiment)(\s*:)', r'\1"\2"\3', txt)
                            # Remove any leading/trailing control characters
                            txt = ''.join(c for c in txt if ord(c) >= 32 or c in '\n\t')
                            return txt

                        annotations = None
                        parse_attempts = [
                            clean_json,
                            _normalize_json_text(clean_json),
                        ]

                        for attempt in parse_attempts:
                            try:
                                annotations = json.loads(attempt)
                                break
                            except json.JSONDecodeError:
                                pass

                        if annotations is None:
                            # Try Python-literal parsing for single-quoted payloads.
                            try:
                                literal_obj = ast.literal_eval(_normalize_json_text(clean_json))
                                if isinstance(literal_obj, list):
                                    annotations = literal_obj
                            except Exception:
                                annotations = None

                        if annotations is None:
                            raise ValueError("Malformed JSON payload from LLM")
                        
                        for ann in annotations:
                            sent_str = ann.get("sentiment", "Neutral")
                            score = 0.0
                            if "positive" in sent_str.lower(): score = 0.5
                            elif "negative" in sent_str.lower(): score = -0.5
                            llm_sentiments[ann.get("id")] = (ann.get("topics", []), score)
                        if llm_sentiments:
                            analysis_meta["llm_used"] = True
                            analysis_meta["status"] = f"Active: {analysis_meta['engine_label']}"
                    except Exception as e:
                        st.toast("LLM output format issue detected. Using standard local routing.", icon="⚠️")
                        analysis_meta["engine_label"] = "Standard ML + VADER"
                        analysis_meta["engine_tier"] = "Local"
                        analysis_meta["status"] = "Fallback activated: local routing engine"
                        pass # Fallback cleanly to VADER lexicons if LLM fails
            # ----------------------------------
            
            aspect_scores = {category: [] for category in ASPECT_TAXONOMY}
            aspect_lengths = {category: 0.0 for category in ASPECT_TAXONOMY}
            aspect_mentions = {category: 0 for category in ASPECT_TAXONOMY}
            sentiment_lengths = {"Positive": 0, "Negative": 0, "Neutral": 0}
            sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
            general_length = 0
            
            highlighted_html = []
            
            for idx, segment in enumerate(segments):
                if not segment.strip() or re.fullmatch(r'[.!?\n]+', segment):
                    # Replace pure newlines with <br> for HTML rendering, escape others
                    clean_seg = html.escape(segment).replace('\n', '<br>')
                    highlighted_html.append(clean_seg)
                    continue
                    
                words = set(re.findall(r'\b\w+\b', segment.lower()))
                if not words:
                    highlighted_html.append(html.escape(segment))
                    continue
                
                sent_len = len(segment)
                matched_cats = []
                sent_score = 0.0
                
                # Fetch contextual tags from LLM dict first
                if idx in llm_sentiments:
                    topics, sent_score = llm_sentiments[idx]
                    for t in topics:
                        for key in ASPECT_TAXONOMY.keys():
                            if t.lower().replace("&", "") in key.lower().replace("&", ""):
                                if key not in matched_cats: matched_cats.append(key)
                                
                # If LLM failed, missed it, or wasn't used, fallback to Autonomous ML Engine or Keyword matching
                if not matched_cats and idx not in llm_sentiments:
                    sent_score = analyzer.polarity_scores(segment)['compound']
                    
                    # --- AUTONOMOUS ML ENGINE UPGRADE ---
                    if aspect_engine:
                        # Multi-label prediction
                        try:
                            # The model expects a list/Series of strings
                            pred_binary = aspect_engine.predict([segment.lower()])[0]
                            categories_list = list(ASPECT_TAXONOMY.keys())
                            for i, val in enumerate(pred_binary):
                                if val == 1:
                                    matched_cats.append(categories_list[i])
                        except:
                            pass
                    
                    # Final fallback to standard keywords if ML engine failed or is missing
                    if not matched_cats:
                        for category, keywords in ASPECT_TAXONOMY.items():
                            if any(kw in words for kw in keywords):
                                matched_cats.append(category)
                
                # Track sentiment distribution lengths
                if sent_score > 0.1:
                    sentiment_lengths["Positive"] += sent_len
                    sentiment_counts["Positive"] += 1
                elif sent_score < -0.1:
                    sentiment_lengths["Negative"] += sent_len
                    sentiment_counts["Negative"] += 1
                else:
                    sentiment_lengths["Neutral"] += sent_len
                    sentiment_counts["Neutral"] += 1
                
                segment_safe = html.escape(segment)
                
                if matched_cats:
                    # Distribute length
                    split_len = sent_len / len(matched_cats)
                    for cat in matched_cats:
                        aspect_scores[cat].append(sent_score)
                        aspect_lengths[cat] += split_len
                        aspect_mentions[cat] += 1
                        
                    # Build highlight HTML
                    tags_html = "".join([f"<span style='font-size: 0.7em; font-weight: bold; color: #4b5563; background: #e5e7eb; padding: 2px 6px; margin-left: 4px; border-radius: 12px; white-space: nowrap;'>{c}</span>" for c in matched_cats])
                    
                    # Sentiment color for the block background
                    bg_color = "#f3f4f6" # neutral gray
                    border_color = "#e5e7eb"
                    if sent_score > 0.1:
                        bg_color = "#dcfce7" # subtle green
                        border_color = "#bbf7d0"
                    elif sent_score < -0.1:
                        bg_color = "#fee2e2" # subtle red
                        border_color = "#fecaca"
                        
                    highlight = f"<span style='background-color: {bg_color}; border: 1px solid {border_color}; padding: 2px 4px; border-radius: 6px; line-height: 2.2;'>{segment_safe}{tags_html}</span>"
                    highlighted_html.append(highlight)
                else:
                    general_length += sent_len
                    highlighted_html.append(segment_safe)
            
            final_tags = []
            distribution = []
            total_len = sum(aspect_lengths.values()) + general_length
            
            for category, scores in aspect_scores.items():
                if scores:
                    # Average the sentiment across all sentences mentioning this aspect
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0.05:
                        final_tags.append(f"🟢 {category}")
                    elif avg_score < -0.05:
                        final_tags.append(f"🔴 {category}")
                    else:
                        final_tags.append(f"⚪ {category}")
                        
                if aspect_mentions[category] > 0:
                    distribution.append({"Topic": category, "Count": aspect_mentions[category]})
                    
            mention_total = sum(aspect_mentions.values())
            for row in distribution:
                row["Share (%)"] = (row["Count"] / mention_total) if mention_total else 0
            
            # Sentiment breakdown distribution
            sent_distribution = []
            sentiment_total = sum(sentiment_counts.values())
            for sent_type, s_count in sentiment_counts.items():
                if sentiment_total > 0:
                    sent_distribution.append({"Sentiment": sent_type, "Count": s_count, "Percentage": s_count / sentiment_total})
                
            annotated_text = "".join(highlighted_html)
            return (final_tags if final_tags else ["⚪ General Feedback"]), distribution, sent_distribution, annotated_text, analysis_meta
            
        # Global Model Handling for this Tab
        model_bundle = load_ml_models()
        rating_models = model_bundle.get("rating", {})
        aspect_engine = model_bundle.get("aspect")
        dealbreakers = model_bundle.get("dealbreakers", {"neg": set(), "pos": set()})

        with st.container():
            if "review_input_text" not in st.session_state:
                st.session_state["review_input_text"] = ""

            def _apply_sample_review(text: str):
                st.session_state["review_input_text"] = text
                st.session_state["review_analyzed"] = False

            review_input = st.text_area(
                "Customer Review Text",
                height=140,
                key="review_input_text",
                placeholder="Paste or type one review, then click Analyze Review. e.g., The food was excellent but the flight was delayed by 3 hours...",
            )

            st.markdown("**Try sample reviews**")
            st.caption("Use a preset to quickly stress test negative, mixed, and positive scenarios.")

            sample_reviews = {
                "1_star": (
                    "My recent return flights Singapore to Rome on Singapore Airlines were the worst airline experience I've had in my 50 years of international travel and I paid a premium for it. "
                    "I booked my flights thinking I would treat myself to a quality experience and also paid over SGD50 to reserve my preferred seats. "
                    "A longwinded online check in which didn't allow me to select carry on only. "
                    "I needed the airport check in counter to sort that out. "
                    "Then I noticed I'm not in my reserved seat, instead I have been seated in one of the worst seats in the plane, opposite the toilets. "
                    "The flight was late, service was chaotic and I was constantly disturbed by the toilet use. "
                    "I contacted customer service after the flight. "
                    "'Operational/technical reasons' were the excuse for my seat change (no change of plane type); no they wouldn't refund the seat reservation fee (they are reviewing but no word for 5 weeks). "
                    "Also the reserved seat on my return flight was also unavailable (to me) and no, no refund, they would try to find a similar seat. "
                    "I was passed around a series of CSRs, all of whom were brusque, intransigent and unhelpful. "
                    "For my return flight, online check in would not let me register my Kris Flyer account because someone at SA had altered my name details. "
                    "This took over an hour to sort out. "
                    "In the end I missed the flight because I mistook 12pm for midnight. "
                    "I contacted SA and was offered a seat on the next flight for SGD1400 'as a courtesy'. "
                    "Even upset I said no to that. "
                    "I booked with Gulf Air for SGD700, but I noticed in my searches the flight the CSR had offered, available online for SGD1000. "
                    "The worst customer service and flight experience I have ever had from an airline that markets itself as offering a premium experience. "
                    "By all means use SA if it's the cheapest deal, you'll get the same experience as anywhere else. "
                    "I wouldn't pay a premium to fly with them because you won't get anything for your money. "
                    "The airline knows how to market itself and maybe at one time, its service was above standard, but no longer. "
                    "Reviews reveal that quality has tumbled post Covid. "
                    "Clearly the bean counters have taken over as the airline tries to recoup its losses. "
                    "In my recent experience, SA has been the rudest, most unhelpful and grudging airline I have ever dealt with and I will be avoiding them in future."
                ),
                "2_5_star": (
                    "Overall disappointing from Singapore Airlines. Late and disorganized boarding. The A350-900 is a tired old thing. "
                    "Does anyone seriously use the coat hanger button on the seat back in front of you? This aircraft has no air outlets above seats and made the trip stuffy. "
                    "The entertainment system was very old and dated movies, yes even the so called \"recent releases\". TV selections were abysmal. "
                    "This was the first international flight I watched nothing at all on. Food was below par and drinks service patchy. "
                    "I used to think Singapore Airlines was a great airline but after this flight I beg to differ. One good point is the staff on board professional and friendly."
                ),
                "5_star": (
                    "Excellent for economy. Five hours into flight and the seat was still comfortable - soft and firm in the right spots. "
                    "The breakfast was good for economy. Large enough portions. Taste enough with variation on the tray to add interest. "
                    "Cabin staff outstanding. Friendly and attentive. Quality tv screen. Easy to operate with a wide variety of programs. "
                    "My transmitter that enables my wireless headphones to connect, doesn't always work on left and right. No problems on this flight. "
                    "Power outlets for mains and USB. The best boarding process I have used, why don't all airlines do this? "
                    "After showing boarding pass and entering the gate lounge, we were seated in boarding groups. "
                    "Then each seated are group was called up in turn. Much more efficient than calling out the number of the boarding group and hoping every one has heard and no one tries to queue jump."
                ),
            }

            preset_cols = st.columns([1, 1, 1])
            with preset_cols[0]:
                st.button(
                    "1★ Severe service failure",
                    use_container_width=True,
                    on_click=_apply_sample_review,
                    args=(sample_reviews["1_star"],),
                )
            with preset_cols[1]:
                st.button(
                    "2.5★ Mixed experience",
                    use_container_width=True,
                    on_click=_apply_sample_review,
                    args=(sample_reviews["2_5_star"],),
                )
            with preset_cols[2]:
                st.button(
                    "5★ Excellent journey",
                    use_container_width=True,
                    on_click=_apply_sample_review,
                    args=(sample_reviews["5_star"],),
                )

            st.write("") # Add a bit of spatial breathing room

            # 1. Primary Modifier - The AI Toggle
            selected_ai_backend = st.session_state.get("selected_ai_model", "auto")
            groq_api_ready = bool(os.environ.get("GROQ_API_KEY", ""))

            # Build dynamic help text with current status
            _use_llm_current = st.session_state.get("_smart_ai_toggle", False)
            if not _use_llm_current:
                current_status = ":blue[●] Basic keyword detection (fastest, works offline)"
            elif (selected_ai_backend == "groq" and groq_api_ready) or (selected_ai_backend == "auto" and groq_api_ready):
                current_status = ":green[●] Smart AI active (Groq cloud)"
            elif selected_ai_backend == "ollama" or (selected_ai_backend == "auto" and not groq_api_ready):
                current_status = ":green[●] Smart AI active (local Ollama)"
            else:
                current_status = ":blue[●] Basic keyword detection (no AI service found)"

            help_text = (
                f"**Current mode:** {current_status}\n\n---\n\n"
                "**When ON:** A cloud-based AI identifies specific topics in the review "
                "(e.g., food quality, seat comfort) for richer, more nuanced analysis.\n\n"
                "**When OFF:** Uses basic keyword detection only — fastest and works offline, but less nuanced."
            )

            use_llm = st.toggle(
                "Use Smart AI Analysis",
                value=False,
                key="_smart_ai_toggle",
                help=help_text,
            )

            # 2. Secondary Modifiers - Advanced Settings Expander
            if rating_models:
                engine_options = list(rating_models.keys())
                selected_model_name = st.session_state.get("selected_rating_engine", engine_options[0])
                if selected_model_name not in engine_options:
                    selected_model_name = engine_options[0]

                with st.expander("Advanced settings", expanded=False):
                    st.markdown("🎯 **Consensus Calibration (Fine-Tuning)**")
                    st.info("The **1.0x Baseline** reflects the AI model's accuracy on the most recent hold-out test set from the training phase. Adjust the multipliers below to increase or decrease a model's influence on the final consensus rating.")
                    
                    for m_name in engine_options:
                        st.slider(
                            f"Weight Multiplier: {m_name}",
                            min_value=0.5,
                            max_value=2.0,
                            value=1.0,
                            step=0.05,
                            key=f"weight_mult_{m_name}",
                            help=f"Manually adjust the influence of {m_name} in the final committee vote."
                        )
            else:
                selected_model_name = "None"

            st.write("")
            
            # 3. Call to Action Button
            if st.button("Analyze Review", type="primary", use_container_width=True):
                st.session_state["review_analyzed"] = True
            
            if st.session_state.get("review_analyzed") and review_input.strip() and rating_models:
                input_clean = re.sub(r'[^a-z0-9\s]', '', review_input.lower()).strip()
                _analyzer = get_vader_analyzer()
                vader_score = _analyzer.polarity_scores(review_input)['compound']
                
                # VADER class label
                if vader_score > 0.05:
                    vader_class = "Positive"
                elif vader_score < -0.05:
                    vader_class = "Negative"
                else:
                    vader_class = "Neutral"
                
                # Sentence-level VADER features
                _sentences = re.split(r'[.!?]', review_input)
                _sent_scores = [_analyzer.polarity_scores(s)['compound'] for s in _sentences if s.strip()]
                vader_min = min(_sent_scores) if _sent_scores else 0.0
                vader_max = max(_sent_scores) if _sent_scores else 0.0
                vader_range = vader_max - vader_min
                
                # Dealbreaker flags
                _words = set(input_clean.split())
                has_neg_db = 1 if _words & dealbreakers.get("neg", set()) else 0
                has_pos_db = 1 if _words & dealbreakers.get("pos", set()) else 0
                
                # Build the full 5-stream DataFrame expected by the new Pipeline
                X_input = pd.DataFrame({
                    "clean_text": [input_clean],
                    "vader_score": [vader_score],
                    "vader_class": [vader_class],
                    "vader_min": [vader_min],
                    "vader_max": [vader_max],
                    "vader_range": [vader_range],
                    "has_neg_dealbreaker": [has_neg_db],
                    "has_pos_dealbreaker": [has_pos_db]
                })
                
                # --- TOURNAMENT PREDICTIONS ---
                # Dynamic benchmarks loaded from model metadata
                benchmarks = model_bundle.get("benchmarks", {})
                
                all_results = []
                model_weighted_scores = []
                total_weight = 0
                
                for name, m in rating_models.items():
                    res = m.predict(X_input)[0]
                    probs = m.predict_proba(X_input)[0]
                    conf = max(probs)
                    
                    # Individual Weighted Rating
                    weighted_rating = sum((i+1) * probs[i] for i in range(5))
                    
                    # Accumulate for Ensemble Consensus
                    weight = benchmarks.get(name, 0.5)
                    model_weighted_scores.append(weighted_rating * weight)
                    total_weight += weight
                    
                    all_results.append({
                        "Model": name, 
                        "Predicted Rating": f"{res} ⭐", 
                        "In-Text Confidence": f"{conf:.1%}",
                        "Train Accuracy": f"{benchmarks.get(name + '_train', 0.8)*100:.1f}%",
                        "Test Accuracy": f"{weight*100:.1f}%",
                        "Rating Logic": f"{weighted_rating:.2f} ⭐"
                    })
                
                # --- CONSENSUS CALCULATIONS ---
                sausage_lines = []
                weighted_sum = 0
                total_weight = 0
                # Collector for Categorical Votes (Winner-Takes-All)
                categorical_votes = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
                vote_breakdown = []

                for name, m in rating_models.items():
                    # Base Benchmark Weight (from training accuracy)
                    base_w = benchmarks.get(name, 0.5)
                    # User Manual Multiplier
                    multiplier = st.session_state.get(f"weight_mult_{name}", 1.0)
                    # Effective Weight used in consensus
                    effective_w = base_w * multiplier
                    
                    # 1. THE WINNER VOTE
                    res = int(m.predict(X_input)[0])
                    categorical_votes[res] += effective_w
                    vote_breakdown.append({"name": name, "star": res, "base_w": base_w, "mult": multiplier, "eff_w": effective_w})
                    
                    # 2. THE CONTINUUM CALC (Math Balance Sheet)
                    s = [sum((i+1) * r[i] for i in range(5)) for r in [m.predict_proba(X_input)[0]]][0]
                    sausage_lines.append(f"• {name.upper()}:\n  {s:.2f} Rating × {base_w:.2f} (Base Acc) × {multiplier:.1f} (Manual) = {s*effective_w:.3f}")
                    weighted_sum += (s * effective_w)
                    total_weight += effective_w
                
                # DERIVE FINAL TOURNAMENT VALUES
                consensus_winner = max(categorical_votes, key=categorical_votes.get)
                continuum_score = weighted_sum / total_weight if total_weight > 0 else 3.0
                
                consensus_score = weighted_sum / total_weight if total_weight > 0 else 3.0
                
                # Build the Voting Breakdown HTML
                # Section 1: Individual Votes
                votes_html_lines = ""
                for v in vote_breakdown:
                    votes_html_lines += (
                        f"<div style='margin-bottom: 10px; padding: 8px; background: #f9fafb; border-radius: 6px; border: 1px solid #e5e7eb;'>"
                        f"<b>{v['name'].upper()}</b> voted <b style='color: #111827;'>{v['star']}★</b><br>"
                        f"<span style='font-size: 0.8rem; color: #6b7280;'>Weight: {v['base_w']:.2f} (Test Accuracy) × {v['mult']:.1f} (Multiplier) = <b>{v['eff_w']:.2f}</b></span>"
                        f"</div>"
                    )
                
                # Section 2: Vote Tally
                tally_lines = ""
                for star in sorted(categorical_votes.keys(), reverse=True):
                    w = categorical_votes[star]
                    if w > 0:
                        # Show which models contributed
                        contributors = [v['name'].upper() for v in vote_breakdown if v['star'] == star]
                        contrib_str = " + ".join([f"{v['eff_w']:.2f}" for v in vote_breakdown if v['star'] == star])
                        highlight = ' style="font-weight: 800; color: #16a34a;"' if star == consensus_winner else ''
                        tally_lines += f"<div{highlight}>{star}★ bucket: {contrib_str} = <b>{w:.2f}</b> ({', '.join(contributors)})</div>"
                
                voting_breakdown_html = f"""
                <div style='font-family: "Source Sans Pro", -apple-system, sans-serif; font-size: 0.88rem; line-height: 1.7;'>
                    <div style='border-bottom: 2px solid #111827; padding-bottom: 8px; margin-bottom: 12px;'>
                        <b style='font-size: 0.9rem;'>CONSENSUS VOTING BREAKDOWN</b><br>
                        <span style='font-size: 0.75rem; color: #6b7280;'>How the Predicted Rating is Derived</span>
                    </div>
                    <b style='color: #15803d; font-size: 0.8rem;'>FORMULA</b><br>
                    <div style='background: #f0fdf4; padding: 8px; border-radius: 6px; margin: 6px 0 12px 0; border: 1px solid #bbf7d0;'>
                        Consensus Winner = Star rating with highest Σ(Test Accuracy × Multiplier)
                    </div>
                    <b style='color: #1f2937; font-size: 0.8rem;'>STEP 1: INDIVIDUAL MODEL VOTES</b><br>
                    <div style='margin: 6px 0 12px 0;'>{votes_html_lines}</div>
                    <b style='color: #1f2937; font-size: 0.8rem;'>STEP 2: VOTE TALLY</b><br>
                    <div style='background: #f9fafb; padding: 12px; border-radius: 8px; margin: 6px 0; border: 1px solid #e5e7eb;'>
                        {tally_lines}
                    </div>
                    <div style='background: #f0fdf4; padding: 12px; border-radius: 8px; margin-top: 8px; border: 2px solid #16a34a;'>
                        <span style='font-size: 1rem; font-weight: 800; color: #111827;'>CONSENSUS WINNER: {consensus_winner}★</span><br>
                        <span style='font-size: 0.75rem; color: #6b7280;'>Highest weighted tally: {categorical_votes[consensus_winner]:.2f}</span>
                        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #bbf7d0; font-size: 0.75rem; color: #6b7280; line-height: 1.5;'>
                            The star rating with the highest combined weight wins the consensus.<br>
                            Each model's weight is based on its test accuracy — more accurate models have a stronger vote.
                        </div>
                    </div>
                </div>
                """
                
                # Divergence (Standard Deviation of the 3 model opinions)
                # Helps identify ambiguous/challenging reviews
                individual_ratings = [sum((i+1) * m.predict_proba(X_input)[0][i] for i in range(5)) for m in rating_models.values()]
                divergence = np.std(individual_ratings) if len(individual_ratings) > 1 else 0
                
                consensus_agreement = "High" if divergence < 0.2 else ("Medium" if divergence < 0.5 else "Low (Ambiguous)")
                
                # Main Active Model (Selected by user for the detailed deep-dive)
                active_model = rating_models.get(selected_model_name)
                prediction = active_model.predict(X_input)[0]
                probabilities = active_model.predict_proba(X_input)[0]
                
                # Aspect extraction
                detected_tags, aspect_dist, sent_dist, annotated_text, analysis_meta = extract_aspect_tags(review_input, use_llm=use_llm)
                
                st.divider()
                st.caption("Results are organized from summary to detail. Start with the verdict card, then open optional detail sections as needed.")
                
                # --- 0. MULTI-MODEL TOURNAMENT SUMMARY ---
                st.subheader("🧬 Model Tournament Consensus", help="**How is this calculated?** During training, we use an 80/20 split. **Train Accuracy:** How well the model learned the historical data. **Test Accuracy:** How well it handles brand new data. We use the Test Accuracy as the 'Expert Weight' for the final star consensus.")
                with st.expander("Consensus Ledger & Stability Audit", expanded=False):
                    st.table(all_results)
                    st.info("**What do these columns mean?**\n- **Train Accuracy:** The AI's performance on the 'study guide' (data it saw during training).\n- **Test Accuracy (Benchmark):** The performance on the 'final exam' (brand new data). We use this as the model's 'Expert Weight.'\n- **In-Text Confidence:** How sure the AI is about *this particular review* right now.\n- **Rating Logic:** A probability-weighted formula: $(1 \\times P_{1★}) + (2 \\times P_{2★}) + (3 \\times P_{3★}) + (4 \\times P_{4★}) + (5 \\times P_{5★})$. This explains the 'continuum' of sentiment even if a model picks a single winner.")
                st.write("")
                
                # --- 1. THE PREMIUM VERDICT HERO (UI/UX) ---
                # Use the Consensus Winner (Categorical) for the primary card display
                check_score = float(consensus_winner)
                
                # Determine colors STRICTLY based on Consensus Winner (Integer)
                # Success Zone (Green for 5★ and 4★)
                if check_score >= 4.0:
                    h_bg, h_border, h_text = "#f0fdf4", "#16a34a", "#166534" # Strong Green
                    if check_score == 5.0:
                        result_label = "🏆 Elite Selection (5★ Verdict)"
                    else:
                        result_label = "⭐ Strong Selection (4★ Verdict)"
                # Warning Zone (Amber for 3★)
                elif check_score == 3.0:
                    h_bg, h_border, h_text = "#fffbeb", "#f59e0b", "#92400e" # Balanced Amber
                    result_label = "📊 Balanced (3★ Verdict)"
                # Critical Zone (Red for 1★ and 2★)
                else:
                    h_bg, h_border, h_text = "#fef2f2", "#ef4444", "#991b1b" # Warning Red
                    result_label = f"⚠️ Warning ({int(check_score)}★ Verdict)"
                
                # Agreement Badge HTML
                agreement_colors = {"High": "#16a34a", "Medium": "#d97706", "Low (Ambiguous)": "#dc2626"}
                ag_color = agreement_colors.get(consensus_agreement, "#4b5563")
                
                # Consensus Meter Percentage (based on categorical winner)
                meter_pct = round(((consensus_winner - 1) / 4) * 100, 1)
                
                # Calculate individual model ticks (grouped by predicted star)
                tick_groups = {}
                for name, m in rating_models.items():
                    m_pred = int(m.predict(X_input)[0])
                    tick_groups.setdefault(m_pred, []).append(name)
                
                ticks_html = ""
                for star, names in tick_groups.items():
                    tick_pos = round(((star - 1) / 4) * 100, 1)
                    label = " | ".join([f"{n}: {star} ★" for n in names])
                    ticks_html += f'<div class="tick-marker" data-label="{label}" style="left: {tick_pos}%;"></div>'
                
                # Build tags HTML for Actionable Routing
                sentiment_order = {"🟢": 0, "🔴": 1, "⚪": 2}
                sorted_tags = sorted(
                    detected_tags,
                    key=lambda tag: (sentiment_order.get(tag[:1], 3), tag[2:].strip().lower()),
                )
                
                tags_html_list = []
                for tag in sorted_tags:
                    if tag.startswith("🟢"):
                        tags_html_list.append(f'<span class="sia-chip" style="color: #166534; background: #dcfce7; border-color: #86efac;">● {tag[2:].strip()}</span>')
                    elif tag.startswith("🔴"):
                        tags_html_list.append(f'<span class="sia-chip" style="color: #991b1b; background: #fee2e2; border-color: #fca5a5;">● {tag[2:].strip()}</span>')
                    elif tag.startswith("⚪"):
                        tags_html_list.append(f'<span class="sia-chip" style="color: #374151; background: #f3f4f6; border-color: #d1d5db;">● {tag[2:].strip()}</span>')
                    else:
                        tags_html_list.append(f'<span class="sia-chip">{tag}</span>')
                tags_html = "".join(tags_html_list)

                # Stars rendering
                stars_display = render_star_rating(consensus_winner, h_border)
                
                raw_html = f"""
                <div class="verdict-hero" style="background-color: {h_bg}; border-left-color: {h_border};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div>
                            <span class="verdict-hero-title">MODEL CONSENSUS SCORE</span>
                            <div style="display: flex; align-items: baseline; margin-top: 4px;">
                                <span class="verdict-score-big" style="color: {h_text};">{int(consensus_winner)}.0</span>
                                <span style="font-size: 1.5rem; font-weight: 700; color: {h_text}; opacity: 0.7;">/ 5</span>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; padding-top: 10px;">
                            {stars_display}
                        </div>
                    </div>
                    
                    <div class="consensus-container" style="margin-top: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; font-weight: 700; color: #4b5563; margin-bottom: 6px;">
                            <span class="sia-tooltip-wrap" style="position: relative; display: inline-flex; align-items: center; cursor: help;">
                                HOW EACH MODEL VOTED (COMMITTEE CLUSTERING)
                                <svg style="margin-left: 6px; fill: #9ca3af; width: 14px; height: 14px;" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z"/>
                                </svg>
                                <!-- The raw HTML pure CSS tooltip -->
                                <div class="sia-popover" style="position: absolute; bottom: 135%; left: 50%; transform: translateX(-50%); background: #ffffff; color: #111827; border: 1px solid #e5e7eb; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 500; line-height: 1.4; width: 280px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); pointer-events: none; z-index: 99999; visibility: hidden; opacity: 0; transition: opacity 0.2s ease, visibility 0.2s ease;">
                                    White markers show each AI model's predicted star rating. The shaded bar indicates the consensus position.
                                </div>
                            </span>
                        </div>
                        <div class="consensus-track">
                            <div class="consensus-fill" style="width: {meter_pct}%; background-color: {h_border}; opacity: 0.3;"></div>
                            {ticks_html}
                        </div>
                    </div>
                    <input type="checkbox" id="voting-toggle" style="display: none;">
                    <label for="voting-toggle" style="display: block; margin-top: 12px; font-size: 0.75rem; font-weight: 600; color: {h_border}; cursor: pointer; text-align: center; padding: 6px; border: 1px dashed transparent; border-radius: 6px; opacity: 0.25; transition: all 0.2s;">
                        How is the consensus score determined? (click to reveal)
                    </label>
                    <div class="voting-breakdown-panel">
                        {voting_breakdown_html}
                    </div>
                </div>
                """
                # The 'Nuclear' Fix: Remove all newlines and indentation to force the browser to treat it as one continuous HTML block
                verdict_html = "".join([line.strip() for line in raw_html.split("\n")])
                
                st.subheader(
                    "🎯 MODEL CONSENSUS SCORE",
                    help=(
                        "**MODEL CONSENSUS SCORE:**\n"
                        "The weighted consensus sentiment rating predicted across all AI models (1-5 stars). "
                        "A consensus score that blends multiple machine learning engines.\n\n"
                        "**How Each Model Voted:**\n"
                        "Individual predictions from each AI model are shown as white markers on the progress bar. "
                        "Hover over the markers to see each model's exact score.\n\n"
                        "This helps you understand model confidence — if all markers align (high agreement), the prediction is highly reliable."
                    )
                )
                st.markdown(verdict_html, unsafe_allow_html=True)
                
                if consensus_agreement == "Low (Ambiguous)":
                    st.warning("⚠️ **Model Divergence Detected:** The AI engines are in strong disagreement. This review has been flagged for manual administrative audit.", icon="🧐")
                
                st.write("")
                
                # --- 2. ANALYTICAL BREAKDOWN (THE 'WHY') ---
                st.subheader(
                    "📊 Analytical Breakdown",
                    help="Use this section for deeper context on sentiment and service-driver distribution.",
                )
                engine_colors = {
                    "Cloud": "#dbeafe",
                    "Local LLM": "#dcfce7",
                    "Local": "#f3f4f6",
                }
                engine_bg = engine_colors.get(analysis_meta.get("engine_tier"), "#f3f4f6")
                status_color = "#166534" if analysis_meta.get("llm_used") else "#374151"
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb; border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.9rem; background:#ffffff;">
                        <div style="display:flex; flex-wrap:wrap; gap:0.55rem; align-items:center;">
                            <span style="font-size:0.75rem; font-weight:700; letter-spacing:0.02em; color:#6b7280; text-transform:uppercase;">Analysis engine</span>
                            <span style="background:{engine_bg}; color:#111827; border:1px solid #d1d5db; border-radius:999px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:700;">{analysis_meta.get('engine_label', 'Standard ML + VADER')}</span>
                            <span style="font-size:0.82rem; color:{status_color}; font-weight:600;">{analysis_meta.get('status', '')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                a_col1, a_col2 = st.columns([1, 1])
                
                with a_col1:
                    if sent_dist:
                        st.subheader(
                            "Review Sentiment (Mood)",
                            help="Shows whether the customer's review is overall positive, negative, or neutral. The chart breaks down what percentage of their comments fall into each category.",
                        )
                        sent_df = pd.DataFrame(sent_dist)
                        sentiment_labels = {
                            "Positive": "Positive feedback",
                            "Neutral": "Neutral feedback",
                            "Negative": "Negative feedback",
                        }
                        sent_df["Sentiment Label"] = sent_df["Sentiment"].map(sentiment_labels).fillna(sent_df["Sentiment"])
                        sent_df["Display"] = sent_df.apply(
                            lambda row: f"{row['Percentage']:.0%} ({int(row['Count'])})",
                            axis=1,
                        )
                        
                        # Move takeaway to top (under header) per user request
                        top_sent = sent_df.sort_values("Percentage", ascending=False).iloc[0]
                        st.caption(
                            f"Takeaway: {top_sent['Sentiment Label']} is most common at {top_sent['Percentage']:.0%} ({int(top_sent['Count'])})."
                        )

                        sent_order = ["Positive feedback", "Neutral feedback", "Negative feedback"]
                        label_inside_threshold = 0.18

                        sent_chart = alt.Chart(sent_df).mark_bar(cornerRadiusEnd=5).encode(
                            x=alt.X("Percentage:Q", axis=alt.Axis(format=".0%", title=None)),
                            y=alt.Y("Sentiment Label:N", title=None, sort=sent_order),
                            color=alt.Color("Sentiment:N", scale=alt.Scale(
                                domain=["Positive", "Neutral", "Negative"],
                                range=["#10b981", "#9ca3af", "#ef4444"]
                            ), legend=None),
                            tooltip=[
                                alt.Tooltip("Sentiment Label:N", title="Mood"),
                                alt.Tooltip("Percentage:Q", title="Share", format=".1%"),
                                alt.Tooltip("Count:Q", title="Sentence count"),
                            ]
                        ).properties(height=180)
                        
                        sent_text_inside = (
                            alt.Chart(sent_df)
                            .transform_filter(f"datum.Percentage >= {label_inside_threshold}")
                            .transform_calculate(LabelX="datum.Percentage / 2")
                            .mark_text(align="center", baseline="middle", fontWeight=700, color="#ffffff")
                            .encode(
                                x=alt.X("LabelX:Q"),
                                y=alt.Y("Sentiment Label:N", sort=sent_order),
                                text=alt.Text("Display:N"),
                            )
                        )
                        sent_text_outside = (
                            alt.Chart(sent_df)
                            .transform_filter(f"datum.Percentage < {label_inside_threshold}")
                            .mark_text(align="left", baseline="middle", dx=6, fontWeight=600, color="#374151")
                            .encode(
                                x=alt.X("Percentage:Q"),
                                y=alt.Y("Sentiment Label:N", sort=sent_order),
                                text=alt.Text("Display:N"),
                            )
                        )
                        st.altair_chart(sent_chart + sent_text_inside + sent_text_outside, use_container_width=True)
                
                with a_col2:
                    if aspect_dist:
                        st.subheader(
                            "Service Drivers (Share of Voice)",
                            help="Shows which service topics (like food, seats, staff, or delays) the customer mentions most. Topics mentioned more often are typically more important to them.",
                        )
                        dist_df = pd.DataFrame(aspect_dist)
                        dist_df = dist_df.sort_values("Share (%)", ascending=False)
                        
                        # Move takeaway to top (under header) per user request
                        top_driver = dist_df.iloc[0]
                        st.caption(
                            f"Takeaway: {top_driver['Topic']} is the top service driver at {top_driver['Share (%)']:.0%} ({int(top_driver['Count'])})."
                        )

                        topic_order = dist_df["Topic"].tolist()
                        dist_df["Display"] = dist_df.apply(
                            lambda row: f"{row['Share (%)']:.0%} ({int(row['Count'])})",
                            axis=1,
                        )
                        driver_label_inside_threshold = 0.18

                        drivers_chart = alt.Chart(dist_df).mark_bar(color="#4f46e5", cornerRadiusEnd=5).encode(
                            x=alt.X("Share (%):Q", axis=alt.Axis(format=".0%", title=None)),
                            y=alt.Y("Topic:N", title=None, sort=topic_order),
                            tooltip=[
                                alt.Tooltip("Topic:N", title="Driver"),
                                alt.Tooltip("Share (%):Q", title="Share", format=".1%"),
                                alt.Tooltip("Count:Q", title="Mentions"),
                            ],
                        ).properties(height=180)
                        drivers_text_inside = (
                            alt.Chart(dist_df)
                            .transform_filter(f"datum['Share (%)'] >= {driver_label_inside_threshold}")
                            .transform_calculate(LabelX="datum['Share (%)'] / 2")
                            .mark_text(align="center", baseline="middle", fontWeight=700, color="#ffffff")
                            .encode(
                                x=alt.X("LabelX:Q"),
                                y=alt.Y("Topic:N", sort=topic_order),
                                text=alt.Text("Display:N"),
                            )
                        )
                        drivers_text_outside = (
                            alt.Chart(dist_df)
                            .transform_filter(f"datum['Share (%)'] < {driver_label_inside_threshold}")
                            .mark_text(align="left", baseline="middle", dx=6, fontWeight=600, color="#374151")
                            .encode(
                                x=alt.X("Share (%):Q"),
                                y=alt.Y("Topic:N", sort=topic_order),
                                text=alt.Text("Display:N"),
                            )
                        )
                        st.altair_chart(drivers_chart + drivers_text_inside + drivers_text_outside, use_container_width=True)

                st.subheader(
                    "🎯 Actionable Routing Tags",
                    help="Recommended follow-up routing based on the detected issues and sentiment.",
                )
                st.markdown(f"<div style='margin-top: 0.2rem;'>{tags_html}</div>", unsafe_allow_html=True)
                
                st.write("")
                
                # --- 3. EVIDENCE LAYER (DEEP DIVE) ---
                st.divider()
                with st.expander("🔍 Evidence Deep-Dive: Annotated Context", expanded=True):
                    # Added max-height and overflow-y: auto to handle 10,000+ word reviews without breaking UI
                    st.markdown(
                        f"""
                        <div style='background-color: #ffffff; color: #1f2937; border: 1px solid #e5e7eb; 
                                    padding: 1.5rem; border-radius: 0.5rem; font-size: 1.1rem; line-height: 1.8; 
                                    font-family: ui-sans-serif, system-ui, sans-serif; 
                                    max-height: 400px; overflow-y: scroll; scrollbar-gutter: stable;'>
                            {annotated_text}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        """
                        <div style='text-align: center; margin-top: 0.75rem; padding: 0.55rem 0.5rem 0.65rem 0.5rem;'>
                            <div style='display: inline-flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0.45rem 0.5rem; padding: 0.15rem 0;'>
                                <span style='font-size: 0.78rem; letter-spacing: 0.03em; font-weight: 700; color: #6b7280; text-transform: uppercase; margin-right: 0.15rem;'>Legend</span>
                                <span title='Positive comments: the sentence expresses a favorable customer view.' style='display: inline-flex; align-items: center; gap: 0.25rem; background: #dcfce7; color: #166534; border: 1px solid #86efac; border-radius: 999px; padding: 0.24rem 0.62rem; font-size: 0.82rem; font-weight: 700;'>
                                    <span aria-hidden='true'>●</span> Positive
                                </span>
                                <span title='Negative comments: the sentence describes dissatisfaction or issues.' style='display: inline-flex; align-items: center; gap: 0.25rem; background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; border-radius: 999px; padding: 0.24rem 0.62rem; font-size: 0.82rem; font-weight: 700;'>
                                    <span aria-hidden='true'>●</span> Negative
                                </span>
                                <span title='Neutral comments: the sentence is factual or emotionally balanced.' style='display: inline-flex; align-items: center; gap: 0.25rem; background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; border-radius: 999px; padding: 0.24rem 0.62rem; font-size: 0.82rem; font-weight: 700;'>
                                    <span aria-hidden='true'>●</span> Neutral
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            elif st.session_state.get("review_analyzed"):
                st.warning("Please enter some text to begin analysis.")

