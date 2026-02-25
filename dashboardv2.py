import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="SIA Review Pulse", page_icon="SIA", layout="wide")

REQUIRED_COLUMNS = {
    "published_date",
    "rating",
    "helpful_votes",
    "text",
    "title",
    "published_platform",
    "type",
}

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

def apply_filters(
    df: pd.DataFrame,
    date_range,
    rating_range,
    platforms,
    review_types,
) -> pd.DataFrame:
    filtered = df
    if platforms:
        filtered = filtered[filtered["published_platform"].isin(platforms)]
    if review_types:
        filtered = filtered[filtered["type"].isin(review_types)]
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

st.title("SIA Review Pulse")
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
    st.header("Filters")
    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end) if default_start and default_end else None,
        min_value=min_date.date() if pd.notnull(min_date) else None,
        max_value=max_date.date() if pd.notnull(max_date) else None,
    )
    rating_range = st.slider(
        "Rating range",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
        step=1,
    )
    platforms = st.multiselect(
        "Platform",
        options=sorted(df["published_platform"].dropna().unique()),
        default=sorted(df["published_platform"].dropna().unique()),
    )
    review_types = st.multiselect(
        "Review type",
        options=sorted(df["type"].dropna().unique()),
        default=sorted(df["type"].dropna().unique()),
    )

filtered = apply_filters(df, date_range, rating_range, platforms, review_types)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total reviews", f"{len(filtered):,}")
col2.metric("Average rating", f"{filtered['rating'].mean():.2f}" if len(filtered) else "—")
col3.metric("% Positive (4-5)", f"{(filtered['rating'].ge(4).mean() * 100):.1f}%" if len(filtered) else "—")
col4.metric("Median length", f"{filtered['text_length'].median():.0f} chars" if len(filtered) else "—")

if len(filtered):
    avg_rating = filtered["rating"].mean()
    positive_pct = filtered["rating"].ge(4).mean() * 100
    negative_pct = filtered["rating"].le(2).mean() * 100
    st.markdown(
        f"**Summary:** {len(filtered):,} reviews | "
        f"Avg rating {avg_rating:.2f} | "
        f"Positive (4–5) {positive_pct:.1f}% | "
        f"Negative (1–2) {negative_pct:.1f}%"
    )
else:
    st.info("No reviews match the current filters.")
    st.stop()

st.download_button(
    "Download filtered data",
    data=to_csv_bytes(filtered),
    file_name="sia_reviews_filtered.csv",
    mime="text/csv",
)

tab_overview, tab_trends, tab_text, tab_samples = st.tabs(
    ["Overview", "Trends", "Text Insights", "Samples"]
)

with tab_overview:
    left, right = st.columns((2, 3))

    with left:
        st.subheader("Ratings distribution")
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
            .properties(height=300)
        )
        st.altair_chart(rating_chart, use_container_width=True)

    with right:
        st.subheader("Platform comparison")
        platform_metric = st.radio(
            "View",
            options=["Average rating", "Review volume"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if platform_metric == "Average rating":
            platform_stats = (
                filtered.groupby("published_platform", dropna=True)["rating"]
                .mean()
                .reset_index(name="value")
                .sort_values("value", ascending=False)
            )
            x_title = "Average rating"
            x_scale = alt.Scale(domain=[0, 5])
            tooltip = ["published_platform:N", alt.Tooltip("value:Q", format=".2f")]
            color = "#2ca02c"
        else:
            platform_stats = (
                filtered.groupby("published_platform", dropna=True)
                .size()
                .reset_index(name="value")
                .sort_values("value", ascending=False)
            )
            x_title = "Reviews"
            x_scale = alt.Scale()
            tooltip = ["published_platform:N", "value:Q"]
            color = "#9467bd"

        platform_chart = (
            alt.Chart(platform_stats)
            .mark_bar(color=color)
            .encode(
                x=alt.X("value:Q", title=x_title, scale=x_scale),
                y=alt.Y("published_platform:N", title=None, sort="-x"),
                tooltip=tooltip,
            )
            .properties(height=300)
        )
        st.altair_chart(platform_chart, use_container_width=True)

    st.subheader("Review type analysis")
    type_stats = (
        filtered.groupby("type", dropna=True)
        .agg(
            review_count=("rating", "size"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )

    type_cols = st.columns(2)
    with type_cols[0]:
        type_rating_chart = (
            alt.Chart(type_stats)
            .mark_bar(color="#17becf")
            .encode(
                x=alt.X("avg_rating:Q", title="Average rating", scale=alt.Scale(domain=[0, 5])),
                y=alt.Y("type:N", title=None, sort="-x"),
                tooltip=["type:N", alt.Tooltip("avg_rating:Q", format=".2f")],
            )
            .properties(height=280)
        )
        st.altair_chart(type_rating_chart, use_container_width=True)

    with type_cols[1]:
        type_volume_chart = (
            alt.Chart(type_stats)
            .mark_bar(color="#bcbd22")
            .encode(
                x=alt.X("review_count:Q", title="Reviews"),
                y=alt.Y("type:N", title=None, sort="-x"),
                tooltip=["type:N", "review_count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(type_volume_chart, use_container_width=True)

with tab_trends:
    st.subheader("Review volume and rating trends over time")
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

    volume_chart = (
        alt.Chart(time_series)
        .mark_line(point=True, color="#ff7f0e")
        .encode(
            x=alt.X("published_date:T", title="Month"),
            y=alt.Y("review_count:Q", title="Reviews"),
            tooltip=[
                alt.Tooltip("published_date:T", title="Month"),
                alt.Tooltip("review_count:Q", title="Reviews"),
                alt.Tooltip("avg_rating:Q", title="Avg rating", format=".2f"),
            ],
        )
        .properties(height=320)
    )

    rating_trend_chart = (
        alt.Chart(time_series)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("published_date:T", title="Month"),
            y=alt.Y("avg_rating:Q", title="Average rating", scale=alt.Scale(domain=[0, 5])),
            tooltip=[
                alt.Tooltip("published_date:T", title="Month"),
                alt.Tooltip("avg_rating:Q", title="Avg rating", format=".2f"),
                alt.Tooltip("review_count:Q", title="Reviews"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(volume_chart, use_container_width=True)
    st.altair_chart(rating_trend_chart, use_container_width=True)

with tab_text:
    st.subheader("Text length vs rating")
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

with tab_samples:
    st.subheader("Sample reviews")
    show_cols = ["published_date", "rating", "published_platform", "title", "text", "helpful_votes"]
    st.dataframe(
        filtered[show_cols].sort_values("published_date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )
