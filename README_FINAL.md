# Restaurant Review Insights  
### Hybrid Sentiment Analytics (VADER) + Local LLM (Ollama) for Actionable Owner Reports

A **privacy-friendly analytics tool** that converts unstructured customer reviews into **actionable insights** for a restaurant owner/manager (and also supports competitor benchmarking and other product/service domains). It reads a review CSV export, performs cleaning + filtering, computes sentiment KPIs, extracts evidence-based themes using a **local LLM**, generates charts and structured exports, and can send a weekly report email with attachments.

---

## 1) RAMI 4.0 positioning (project context)

**Functional Layer:**  
This tool is positioned as a **Functional Layer capability** because it provides analytics functions (sentiment scoring, evidence extraction, summarization, KPI suggestions) that enable decision-support.

**Information Layer:**  
Consumes review data (CSV) and produces structured information artifacts (CSV/JSON) and traceable evidence lines.

**Business Layer:**  
Supports managerial decisions with owner-ready summaries, quick wins, long-term actions, KPIs, and email reporting.

---

## 2) Who is it for? (primary user + reuse)

**Primary user:** Restaurant owner / manager  
**Also useful for:**
- **Competitor monitoring:** Compare strengths/weaknesses of competitors using their public reviews.
- **Multi-branch monitoring:** Run the same pipeline per branch and compare results.
- **Other domains:** Any **product/service company** that receives review-like feedback (retail, hospitality, e-commerce, customer support) can reuse the pipeline with minimal changes (same input type: text reviews).

---

## 3) What the tool does (end-to-end)

### Inputs
A CSV export containing:
- **Required:** review text column (ideally `review_text`)
- **Optional:** rating/stars column (e.g., `stars`, `rating`, `review_rating`)
- **Optional:** timestamp/date column (e.g., `review_datetime_utc`, `date`, `timestamp`, `created_at`)

### Processing pipeline
1. **CSV ingest + validation**
2. **Cleaning & filtering**
   - minimum text length  
   - de-duplication  
   - optional language filter (`auto|de|en`)
3. **Sentiment scoring (VADER / NLTK)**
   - produces `vader_score` and `sentiment_bucket`
4. **Local LLM extraction (Ollama API) with strict JSON prompts**
   - extracts only evidence-backed insights for:
     - **Kitchen** (food quality)
     - **Service** (speed, friendliness, cleanliness, order accuracy)
     - **Management** (pricing, ambience, organization)
5. **Owner summary generation (LLM, chunked + merged + final structured JSON)**
   - strengths, improvements, **7-day quick wins**, **30-day actions**, KPIs
6. **Charts + exports**
7. **Owner email generation + optional SMTP email sending**

---

## 4) Folder structure (per-run outputs)

Each execution creates a timestamped run folder:

```
project_root/
  .env
  Project_Sentiment_Analysis_22.12.1.py      # MAIN pipeline (runs everything)
  owner_outputs.py                            # builds weekly email + flat summary table
  send_weekly_report.py                       # sends email with attachments
  email_reporter.py                           # SMTP helper
  runs/
    last_run.txt                              # points to latest run folder
    YYYY-MM-DD_HHMMSS/
      cache_reviews.csv                        # cache for resuming LLM extraction
      reviews_processed.csv                    # processed dataset (sentiment + extractions)
      owner_summary.json                       # structured owner summary
      owner_summary_readable.txt               # same JSON, human-readable text
      owner_summary_flat.csv                   # flattened strengths/improvements table
      weekly_owner_email.txt                   # final email body
      chart_stars.png                          # star distribution
      chart_sentiment_pie.png                  # sentiment distribution
      chart_trend.png                          # monthly sentiment trend
```

Why per-run foldering matters:
- traceability (what was generated when)
- reproducibility (rerun and compare)
- safe auditing/debugging using stored intermediate artifacts

---

## 5) Requirements

### Software
- **Python 3.10+**
- **Ollama** installed and running locally (recommended for privacy)

### Python dependencies
Install with pip (example):
```bash
pip install pandas nltk matplotlib tqdm requests
```

NLTK resources are **auto-downloaded** on first run:
- `stopwords`
- `vader_lexicon`

---

## 6) Setup & configuration

### Step A — Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step B — Start Ollama and pull a model
```bash
ollama serve
ollama pull phi3.5
```

### Step C — Configure `.env`
Minimal keys you typically set:

**Restaurant / input**
- `RESTAURANT_NAME=Pizza House`
- `INPUT_CSV=Burger King Data.csv` (optional; auto-detect works if CSV is in the project folder)
- `MAX_REVIEWS=50` (optional for faster testing)
- `LANGUAGE=auto|de|en` (optional)
- `MIN_TEXT_LEN=10` (optional)

**LLM**
- `LLM_URL=http://localhost:11434/api/generate`
- `MODEL_NAME=phi3.5`

> Tip: Switching LLM model is designed to be “drop-in”: you typically only change `MODEL_NAME`.

**Email (optional)**
- `SMTP_HOST=smtp.gmail.com`
- `SMTP_PORT=587`
- `SMTP_USER=...`
- `SMTP_PASS=...`
- `OWNER_EMAIL=...` or `OWNER_EMAILS=a@x.com,b@y.com`

**Important (Gmail):** Use an **App Password** instead of your normal password if your account requires it.

---

## 7) How to run (one command)

Run the full pipeline:

```bash
python Project_Sentiment_Analysis_22.12.1.py
```

This will automatically:
1. create a new run folder under `runs/`
2. load CSV (from `INPUT_CSV` or auto-detect)
3. clean/filter + VADER sentiment scoring
4. run strict LLM extraction + summary generation
5. generate charts
6. run `owner_outputs.py` (build email + flat CSV)
7. run `send_weekly_report.py` (email report if SMTP configured)

---

## 8) Scripts explained (what each file does)

### A) `Project_Sentiment_Analysis_22.12.1.py` (MAIN pipeline)
Responsible for:
- reading `.env` (so subprocess scripts inherit config)
- creating **RUN_DIR** and writing `runs/last_run.txt`
- detecting input CSV + key columns
- cleaning, filtering, de-duplication, date parsing, star clamping (1..5)
- VADER sentiment scoring + bucketing
- calling local LLM with strict JSON prompts for Kitchen/Service/Management
- generating `reviews_processed.csv`, `owner_summary.json`, charts
- orchestrating the rest of the pipeline scripts

### B) `owner_outputs.py` (Owner email builder)
- reads `owner_summary.json` and `reviews_processed.csv`
- creates a **compact** `weekly_owner_email.txt` (owner-ready)
- writes `owner_summary_flat.csv` (for long-term tracking / BI / dashboards)

### C) `send_weekly_report.py` (SMTP sender)
- resolves the correct run folder from `RUN_DIR` or `runs/last_run.txt`
- attaches charts + summary JSON/TXT + processed CSV + flat CSV
- sends to `OWNER_EMAIL` or `OWNER_EMAILS`

### D) `email_reporter.py` (SMTP helper)
- `send_email_with_attachments(...)`
- handles MIME types and attachments safely using TLS

---

## 9) Outputs (what the owner should look at)

**Best “owner-facing” artifact:**
- `weekly_owner_email.txt` (what gets emailed)

**For validation / debugging:**
- `reviews_processed.csv` (row-level view: sentiment + extracted fields)

**For automation / dashboards:**
- `owner_summary.json` (structured decision-support output)
- `owner_summary_flat.csv` (tracking themes over time)

**Visual attachments:**
- `chart_stars.png`
- `chart_sentiment_pie.png`
- `chart_trend.png`

---

## 10) Troubleshooting

**LLM not reachable**
- ensure: `ollama serve`
- ensure `.env`: `LLM_URL=http://localhost:11434/api/generate`

**NLTK download fails**
- run once with internet access so NLTK can download resources
- or pre-download stopwords + vader lexicon in a connected environment

**Star chart looks wrong**
- check if star column exists in input CSV
- confirm `stars` in `reviews_processed.csv` are numeric 1..5

**Email sending fails**
- Gmail often requires an App Password
- ensure SMTP settings are correct and `SMTP_PASS` is set

---

## 11) Limitations (current scope)

- VADER may miss sarcasm/irony and domain-specific slang.
- LLM quality depends on `MODEL_NAME` (bigger/better models usually produce cleaner summaries).
- Batch-oriented execution (weekly/monthly); not real-time streaming.

---

## 12) Future extensions (not required for current submission)

- real-time ingestion (API/stream)
- multilingual expansion
- competitor benchmarking automation (compare multiple restaurants automatically)
- dashboarding using the stored `owner_summary_flat.csv` across runs

---

## 13) GitHub: create a repository and upload

### Step A — create `.gitignore` (important)
Make sure you do NOT commit secrets or huge outputs:
- `.env`
- `runs/` (optional: you can keep one sample run under `examples/`)

Example `.gitignore`:
```text
.env
.venv/
__pycache__/
runs/
*.pyc
```

### Step B — initialize git and commit
```bash
git init
git add .
git commit -m "Initial release: Restaurant Review Insights"
```

### Step C — create a GitHub repo and push
1. GitHub → New repository → name it (example): `restaurant-review-insights`
2. Then:
```bash
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/restaurant-review-insights.git
git push -u origin main
```

Tip: Add an `.env.example` (safe template) instead of committing `.env`.

---

## 14) Authors / project context

**Authors:** Bony Martin, Bharathraj Govindaraj  
**Context:** Data Science & Analytics master project  
**Domain:** Customer feedback analytics for QSR (case study: Burger King reviews)
