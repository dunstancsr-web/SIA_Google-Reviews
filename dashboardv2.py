import pandas as pd
import streamlit as st
import altair as alt
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Global font styling (Altair charts)
def _altair_arial_theme():
    return {
        "config": {
            "font": "Arial",
            "title": {"font": "Arial"},
            "axis": {"labelFont": "Arial", "titleFont": "Arial"},
            "legend": {"labelFont": "Arial", "titleFont": "Arial"},
            "header": {"labelFont": "Arial", "titleFont": "Arial"},
        }
    }

alt.themes.register("arial_theme", _altair_arial_theme)
alt.themes.enable("arial_theme")

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

def generate_ai_insight(df: pd.DataFrame, x_col: str, y_col: str, agg_label: str) -> str:
    """Generate AI-powered insight using Groq API (cloud-compatible) or Ollama (local fallback)."""
    import os
    
    # Generate data summary
    if y_col == "(count)":
        summary = f"Review frequency by {x_col}"
        data_desc = df.groupby(x_col, dropna=True).size().to_dict()
    else:
        summary = f"{agg_label.capitalize()} of {y_col} by {x_col}"
        data_desc = df.groupby(x_col, dropna=True)[y_col].agg(agg_label).to_dict()
    
    prompt = f"Based on this data: {data_desc}. Provide ONE concise business insight (1-2 sentences) about '{summary}' in Singapore Airlines reviews. Be specific and actionable."
    
    # Check for Groq API (cloud-compatible)
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        return _generate_groq_insight(prompt, groq_api_key)
    
    # Fallback to Ollama (local only)
    return _generate_ollama_insight(prompt)

def _generate_groq_insight(prompt: str, api_key: str) -> str:
    """Generate insight using Groq API."""
    try:
        from groq import Groq
    except ImportError:
        return "💡 **AI Insight:** Install groq library to enable Groq insights (pip install groq)."
    
    try:
        client = Groq(api_key=api_key)
        message = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="mixtral-8x7b-32768",  # Free tier model
            max_tokens=150,
        )
        return f"🤖 **AI Insight:** {message.choices[0].message.content.strip()}"
    except Exception as e:
        return f"💡 **AI Insight:** Groq error ({str(e)[:40]}). Check API key."

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
            return f"🤖 **AI Insight:** {result.get('response', 'No insight generated').strip()}"
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

st.title("Self‑Service Data Hub: SIA Google Reviews")
st.caption("A quick look at Singapore Airlines review sentiment and trends.")

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
            st.caption(
                f"Selected: {date_range[0].strftime('%b %Y')} → {date_range[1].strftime('%b %Y')}"
            )

    with st.expander("Rating range", expanded=True):
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



date_filter = date_range
filtered = apply_filters(df, date_filter, rating_range, platforms)

with download_container:

    st.download_button(
        "⬇ Download filtered data",
        data=to_csv_bytes(filtered),
        file_name="sia_reviews_filtered.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
        help="Export the reviews that match your filters.",
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

tab_profile, tab_overview, tab_explore, tab_text = st.tabs(
    ["Data Profile", "Overview", "Data Exploration", "Text Insights"]
)

with tab_profile:
    st.subheader("Dataset dictionary")
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

    st.subheader("Raw Data")
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
    st.info(
        generate_key_takeaways(filtered),
        icon="💡"
    )
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", f"{len(filtered):,}")
    col2.metric("Average rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "—")
    col3.metric("% Positive (4-5)", f"{(filtered['rating'].ge(4).mean() * 100):.1f}%" if len(filtered) else "—")
    col4.metric("Median length", f"{filtered['text_length'].median():.0f} chars" if len(filtered) else "—")

    avg_rating = filtered["rating"].mean()
    positive_pct = filtered["rating"].ge(4).mean() * 100
    negative_pct = filtered["rating"].le(2).mean() * 100
    st.markdown(
        f"**Summary:** {len(filtered):,} reviews | "
        f"Avg rating {avg_rating:.2f} | "
        f"Positive (4–5) {positive_pct:.1f}% | "
        f"Negative (1–2) {negative_pct:.1f}%"
    )

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
                x=alt.X("rating:O", title="Rating"),
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

with tab_explore:
    st.subheader("Data Exploration")
    st.caption("Select columns for X and Y to build a quick bar or line chart to explore relationships.")

    explore_cols = filtered.columns.tolist()
    numeric_cols = [col for col in explore_cols if is_numeric_dtype(filtered[col])]
    
    # Create formatted column mappings
    x_col_mapping = get_formatted_columns(explore_cols, filtered)
    y_col_options = ["(count)"] + numeric_cols
    y_col_mapping = get_formatted_columns(y_col_options, filtered)

    with st.expander("🛠️ Chart Builder", expanded=True):
        control_cols = st.columns([1.5, 1.5, 1.5])
        with control_cols[0]:
            chart_type = st.selectbox("Chart type", ["Bar", "Line"])
        with control_cols[1]:
            x_col_formatted = st.selectbox(
                "X axis",
                options=list(x_col_mapping.keys()),
                index=list(x_col_mapping.keys()).index(format_column_with_type("published_date", filtered)) if "published_date" in explore_cols else 0,
            )
            x_col = x_col_mapping[x_col_formatted]
        with control_cols[2]:
            y_col_formatted = st.selectbox(
                "Y axis",
                options=list(y_col_mapping.keys()),
                index=0 if not numeric_cols else 1,
            )
            y_col = y_col_mapping[y_col_formatted]

        if y_col == "(count)":
            agg_label = "count"
            plot_df = (
                filtered.groupby(x_col, dropna=True)
                .size()
                .reset_index(name="value")
            )
        else:
            agg_func = st.selectbox("Aggregation", ["mean", "sum", "median"], index=0)
            agg_label = agg_func
            plot_df = (
                filtered.groupby(x_col, dropna=True)[y_col]
                .agg(agg_func)
                .reset_index(name="value")
            )

    x_dtype = "T" if is_datetime64_any_dtype(filtered[x_col]) else "O"
    if chart_type == "Line" and x_dtype == "O":
        st.info("Line charts work best with time or numeric X axes. Consider a bar chart if X is categorical.")

    st.markdown(f"**Chart:** {agg_label.capitalize()} of {y_col if y_col != '(count)' else 'reviews'} by {x_col}")
    
    chart = (
        alt.Chart(plot_df)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X(f"{x_col}:{x_dtype}", title=x_col),
            y=alt.Y("value:Q", title=f"{agg_label} of {y_col if y_col != '(count)' else 'rows'}"),
            tooltip=[alt.Tooltip(f"{x_col}:{x_dtype}", title=x_col), alt.Tooltip("value:Q", title=agg_label)],
        )
        .properties(height=320)
    )

    if chart_type == "Line":
        chart = chart.mark_line(point=True, color="#1f77b4")

    st.altair_chart(chart, use_container_width=True)
    
    st.markdown(generate_chart_insight(filtered, x_col, y_col, agg_label))
    st.markdown(generate_ai_insight(filtered, x_col, y_col, agg_label))

with tab_text:
    st.info(
        generate_key_takeaways(filtered),
        icon="💡"
    )
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", f"{len(filtered):,}")
    col2.metric("Average rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "—")
    col3.metric("% Positive (4-5)", f"{(filtered['rating'].ge(4).mean() * 100):.1f}%" if len(filtered) else "—")
    col4.metric("Median length", f"{filtered['text_length'].median():.0f} chars" if len(filtered) else "—")

    avg_rating = filtered["rating"].mean()
    positive_pct = filtered["rating"].ge(4).mean() * 100
    negative_pct = filtered["rating"].le(2).mean() * 100
    st.markdown(
        f"**Summary:** {len(filtered):,} reviews | "
        f"Avg rating {avg_rating:.2f} | "
        f"Positive (4–5) {positive_pct:.1f}% | "
        f"Negative (1–2) {negative_pct:.1f}%"
    )

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
