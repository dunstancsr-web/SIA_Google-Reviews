---
title: Singapore Airlines Review Dashboard
emoji: ✈️
colorFrom: blue
colorTo: yellow
sdk: streamlit
app_file: app/dashboardv2.py
pinned: false
---
> **Note to self:** The block above is for Hugging Face settings. 
> Do not delete it or the app won't boot.


# To open file in local host, type the following in the terminal:
python -m streamlit run app/dashboardv2.py

# Singapore Airlines Review Dashboard

An interactive Streamlit dashboard for analyzing Singapore Airlines customer reviews with AI-powered insights.

## Features

- **4 Interactive Tabs**:
  - **Data Profile**: Dataset dictionary and raw data exploration
  - **Overview**: Key metrics, trends, ratings breakdown, and platform analysis
  - **Data Exploration**: Chart builder with color-coded columns and AI insights
  - **Text Insights**: Text analysis, sentiment distribution, and keyword clouds

- **Smart Filtering**: Date range (month-labeled slider), rating range, and platform selection
- **AI-Powered Insights**: Real-time analysis using Groq API (cloud) or local Ollama (offline)
- **Downloadable Data**: Export filtered results as CSV

## Quick Start

### Local Setup (Ollama + Groq API)

1. **Clone/navigate to project**:
   ```bash
   cd ~/Documents/GitHub/SIA_Google-Reviews
   ```


3. **Set Groq API Key** (for cloud AI insights):
   ```bash
   export GROQ_API_KEY="gsk_your_key_here"
   ```
   Get your free key at: https://console.groq.com

4. **Run Streamlit**:
   ```bash
   make run
   ```

Dashboard opens at `http://localhost:8501`

### Local Run Commands (Quick Reference for running Streamlit locally on desktop)

1. Open a terminal.
2. Go to the project folder:
   ```bash
   cd /Users/stan/Documents/GitHub/SIA_Google-Reviews
   ```
3. Use one of these commands:
   ```bash
   make run
   make status
   make stop
   ```

Note: Copy only the command lines inside the code block. Do not copy the ` ```bash ` or closing ` ``` ` markers.

- `make run`: Start the dashboard on `127.0.0.1:8501`.
- `make status`: Check if anything is listening on port `8501`.
- `make stop`: Stop the local dashboard process on port `8501`.

### Optional: Local AI (Ollama)

For offline AI insights without Groq API:

1. **Install Ollama** (if not already done):
   ```bash
   brew install ollama
   ```

2. **Start Ollama service**:
   ```bash
   brew services start ollama
   ```

3. **Pull llama2 model** (one-time):
   ```bash
   ollama pull llama2
   ```

4. **Run dashboard** (no GROQ_API_KEY needed):
   ```bash
   make run
   ```

App auto-detects Ollama and uses it for AI insights. If both Groq and Ollama are available, Groq takes priority.

## Cloud Deployment (Streamlit Cloud)

### Prerequisites
- GitHub repo with this code
- Groq API key (https://console.groq.com)
- Streamlit Cloud account (https://streamlit.io/cloud)

### Steps

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial dashboard commit"
   git remote add origin https://github.com/yourusername/repo-name.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "New app" → Select your repo, branch, and main file
   - Enter `app/dashboardv2.py` as the main file path

3. **Add Groq API Secret**:
   - In Streamlit Cloud app settings (⋮ menu), select "Secrets"
   - Add secret:
     ```
     GROQ_API_KEY = "gsk_your_key_here"
     ```
   - Save and app will auto-rerun

4. **Test**: Navigate to all tabs, apply filters, and verify AI insights work

**Note**: Ollama (localhost:11434) is not available on cloud. Groq API is required for cloud deployment.

## Data

**Source**: `data/singapore_airlines_reviews.csv`

**Columns**:
- `published_date`: Review publication date
- `rating`: 1-5 star rating
- `helpful_votes`: Number of helpful votes
- `text`: Full review text
- `title`: Review title
- `published_platform`: Source platform (e.g., Skytrax, Google)
- `type`: Review type (e.g., "Verified purchase")

## Environment Variables

- `GROQ_API_KEY`: Groq API key for cloud AI insights (optional, recommended)
- `STREAMLIT_SERVER_HEADLESS`: Auto-detected on Streamlit Cloud (no action needed)

## Technical Stack

- **Framework**: Streamlit
- **Data Processing**: pandas, altair
- **Visualization**: Altair (charts), wordcloud
- **AI Backends**: Groq API (cloud), Ollama llama2 (local)
- **Python**: 3.9+

## Methodology & Model Architecture

### Feature Pipeline ("Core Four")

The ML rating-prediction models use four engineered features:

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `clean_text` | Review text | TF-IDF vectorized (up to 25k features, 1–3 n-grams) |
| 2 | `vader_min` | Review text | VADER compound score of the most negative sentence ("pain-point detector") |
| 3 | `has_negative_dealbreaker` | Review text | Binary flag for critical service-failure keywords (extracted via scout LR) |
| 4 | `llm_sentiment_score` | Review text via Llama-3 | LLM-generated sentiment score (-1.0 to 1.0) |

### LLM-Assisted Mode (Transparency Note)

This project demonstrates an **LLM-augmented ML pipeline**. The `llm_sentiment_score` feature is generated by prompting a large language model (Llama-3) to rate the sentiment of the same review text whose star rating the model predicts. This creates a strong correlation between the feature and the target:

- **Pearson r = 0.87** between `llm_sentiment_score` and `rating`
- The LLM score alone can predict ratings with **~90% relaxed (±1) accuracy**

This is by design — the architecture uses the LLM as a **pre-scorer** that captures nuance (sarcasm, mixed sentiment, context) that traditional NLP features miss. The ML ensemble then acts as a structured wrapper around this signal.

**Two operating modes reflect this honestly:**

| Mode | Features Used | Test Accuracy (±1) | What It Measures |
|------|---------------|---------------------|------------------|
| **Standard ML** | TF-IDF + VADER + Dealbreakers | 81–89% | The ML model's own learned patterns |
| **LLM-Assisted** | All four features | 95–97% | LLM doing the heavy lifting + ML ensemble refinement |

The "Standard ML" accuracy represents the model's genuine predictive ability. The "LLM-Assisted" accuracy reflects the combined system where the LLM contributes most of the signal.

### Model Ensemble

Three models are trained and ensembled via accuracy-weighted soft voting:
- **Logistic Regression** (tuned C via RandomizedSearchCV)
- **Linear SVM** (Calibrated)
- **Random Forest** (capped at 100 estimators / depth 35 for portability)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### AI Insights not appearing (local)
**Ollama connection error**: Ensure Ollama service is running:
```bash
brew services start ollama
ollama serve  # Or check if already running
```

### AI Insights not appearing (cloud)
**Missing GROQ_API_KEY secret**: Add your Groq API key to Streamlit Cloud Secrets (⋮ menu → Secrets).

### Slow chart generation
Chart rendering may take a few seconds depending on data size. Use the filter panel to narrow results for faster exploration.

## Support

For issues, check the code comments or consult:
- Streamlit docs: https://docs.streamlit.io
- Groq API docs: https://console.groq.com/docs
- Ollama docs: https://ollama.ai



# To open file in local host, type the following in the terminal:
python -m streamlit run app/dashboardv2.py