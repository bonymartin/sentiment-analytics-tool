README — Sentiment Analytics Tool for Customer Feedback (RAMI 4.0 Functional Layer)

Name: Bony Martin, Bharathraj Govindaraj
Topic: Review-based sentiment analytics + LLM extraction for customer review insights
Course/Context: Data Science and Analytics
Target user: Restaurant owner/manager (also reusable for competitor analysis and other product-based domains)

------------------------------------------------------------
1) What the project does (in one paragraph)
------------------------------------------------------------
This project reads customer review CSV exports (text + optional stars + optional timestamps),
cleans and standardizes them, calculates sentiment using VADER (NLTK), and uses a local
LLM (Ollama endpoint) to extract evidence-based positives/negatives for Kitchen, Service,
and Management. It then generates charts, a structured owner summary (JSON/TXT), a compact
weekly email text, and sends the email with attachments via SMTP. Each execution creates a
new timestamped run folder under /runs for traceability and reproducibility.

Positioning in RAMI 4.0:
- Functional Layer: analytics functions (sentiment scoring, extraction, summarization, KPI suggestions)
- Information Layer: structured outputs (CSV/JSON) and evidence lines
- Business Layer: decision support (owner summary + weekly email)

------------------------------------------------------------
2) Directory structure / files
------------------------------------------------------------
project_root/
  .env
  Project_Sentiment_Analysis_22.12.1.py       (MAIN pipeline; runs everything)
  owner_outputs.py                             (creates weekly_owner_email.txt + owner_summary_flat.csv)
  send_weekly_report.py                        (sends email with attachments from RUN_DIR or runs/last_run.txt)
  email_reporter.py                            (SMTP helper)
  runs/
    last_run.txt                               (points to latest run folder)
    YYYY-MM-DD_HHMMSS/
      cache_reviews.csv                        (resume cache for LLM extraction)
      reviews_processed.csv                    (final processed dataset)
      owner_summary.json                       (LLM owner summary JSON)
      owner_summary_readable.txt               (human-readable summary text)
      owner_summary_flat.csv                   (flat table for tracking themes over time)
      weekly_owner_email.txt                   (final email body)
      chart_stars.png                          (star distribution)
      chart_sentiment_pie.png                  (sentiment distribution)
      chart_trend.png                          (monthly sentiment trend)

------------------------------------------------------------
3) Setup / initialization
------------------------------------------------------------
3.1 Python environment
- Recommended: create a virtual environment
  python -m venv .venv
  .venv\Scripts\activate     (Windows)
  source .venv/bin/activate  (Linux/Mac)

3.2 Install dependencies
  pip install pandas nltk matplotlib tqdm requests

3.3 NLTK resources (auto-download on first run)
The code automatically downloads:
- stopwords
- vader_lexicon

3.4 Local LLM (Ollama)
- Install and run Ollama
- Ensure endpoint is reachable:
  http://localhost:11434/api/generate

3.5 Configure .env
Example .env keys:
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
- OWNER_EMAIL (or OWNER_EMAILS)
- LLM_URL (and/or OLLAMA_URL)
- MODEL_NAME (change model here to switch LLM without code changes)
- RESTAURANT_NAME
- INPUT_CSV (optional; if not set the script auto-detects a CSV in project folder)
- MAX_REVIEWS (optional)

IMPORTANT:
For Gmail SMTP, use an App Password (not your normal password) if required by account security.

------------------------------------------------------------
4) How to run (one command)
------------------------------------------------------------
Run the main script:
  python Project_Sentiment_Analysis_22.12.1.py

What happens automatically:
1) Creates runs/YYYY-MM-DD_HHMMSS and writes runs/last_run.txt
2) Loads input CSV (INPUT_CSV from .env, else auto-detect newest csv in project folder)
3) Cleans + filters reviews (min length, duplicates; optional language filters)
4) VADER sentiment scoring
5) LLM extraction per review (Kitchen/Service/Management + sentiment label)
6) Generates owner summary JSON via chunking + merging + final LLM prompt
7) Generates charts (stars, sentiment pie, trend)
8) Runs owner_outputs.py (writes weekly_owner_email.txt + owner_summary_flat.csv)
9) Runs send_weekly_report.py (sends email + attachments)

------------------------------------------------------------
5) Description of each script (what it does)
------------------------------------------------------------
A) Project_Sentiment_Analysis_22.12.1.py (MAIN)
- Loads .env so subprocess scripts inherit settings
- Creates RUN_DIR per run (runs/YYYY-MM-DD_HHMMSS)
- Detects input CSV and columns (review_text, stars/rating, date)
- Cleans text, normalizes dates, clamps stars to 1..5, drops duplicates
- Computes VADER sentiment and buckets
- Calls local LLM with strict JSON prompts:
  - Kitchen extraction
  - Service extraction
  - Management extraction
  - Sentiment label (text-only)
- Writes outputs into RUN_DIR:
  - cache_reviews.csv (resume)
  - reviews_processed.csv
  - owner_summary.json + owner_summary_readable.txt
  - chart_*.png

B) owner_outputs.py
- Reads RUN_DIR owner_summary.json and reviews_processed.csv
- Builds compact weekly_owner_email.txt (subject + run date + snapshot + top items)
- Writes owner_summary_flat.csv for long-term tracking (trend dashboard later)

C) send_weekly_report.py
- Resolves RUN_DIR from env or runs/last_run.txt
- Reads weekly_owner_email.txt
- Attaches charts + summary JSON/TXT + processed CSV + flat CSV
- Sends via SMTP to OWNER_EMAIL / OWNER_EMAILS

D) email_reporter.py
- SMTP helper using EmailMessage
- Attaches files and sends safely with TLS

------------------------------------------------------------
6) Outputs per run (what to look at)
------------------------------------------------------------
- weekly_owner_email.txt: what the owner reads (executive-style)
- owner_summary.json: structured summary for programmatic usage
- reviews_processed.csv: row-level dataset for validation/debugging
- chart_stars.png, chart_sentiment_pie.png, chart_trend.png: visuals
- owner_summary_flat.csv: can be aggregated across runs to build dashboards

------------------------------------------------------------
7) Notes / known limitations
------------------------------------------------------------
- VADER can be imperfect for sarcasm or very domain-specific slang.
- LLM output quality depends on the local model (MODEL_NAME).
- The pipeline is batch-based (periodic), not real-time streaming.
- For best reliability, keep temperature=0 and enforce JSON output (already done).

------------------------------------------------------------
8) Submission note (zip packaging)
------------------------------------------------------------
Put all files (report PDF, slides PDF if required, .tex, .bib if used, python scripts,
and the runs folder with example outputs) into ONE zip file as required.
Do not submit by email if submission rules say otherwise (see course guideline).
