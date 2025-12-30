import os
import re
import time
import json
import requests
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime


# -----------------------------
# PATHS (robust)
# -----------------------------
HERE = Path(__file__).resolve().parent

# -----------------------------
# RUN OUTPUT FOLDER (one folder per run)
# -----------------------------
RUNS_ROOT = HERE / "runs"

def get_run_dir() -> Path:
    """Create/resolve a per-run output folder and expose it via env vars.
    - If RUN_DIR is already set, reuse it.
    - Otherwise create runs/YYYY-MM-DD_HHMMSS and write runs/last_run.txt.
    """
    env_dir = os.getenv("RUN_DIR", "").strip()
    if env_dir:
        d = Path(env_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = RUNS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)

    (RUNS_ROOT / "last_run.txt").write_text(str(d), encoding="utf-8")

    os.environ["RUN_DIR"] = str(d)
    os.environ["RUN_STARTED_AT"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return d


# -----------------------------
# ENV LOADER
# -----------------------------
def load_dotenv(path: Path, overwrite: bool = False):
    """Minimal .env loader (no extra packages). Loads key=value pairs."""
    if not path.exists():
        print(f"[ENV] .env not found at: {path}")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if overwrite or (k not in os.environ):
            os.environ[k] = v


# Load env vars early so subprocess inherits them
load_dotenv(HERE / ".env", overwrite=False)

# -----------------------------
# SETTINGS (from env where possible)
# -----------------------------
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "").strip() or "Restaurant"
MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "phi3.5"
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/generate").strip()

LANGUAGE = os.getenv("LANGUAGE", "auto").strip().lower()   # auto|de|en
SUMMARY_LANGUAGE = os.getenv("SUMMARY_LANGUAGE", "auto").strip().lower()
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "10").strip() or "10")

MAX_REVIEWS_ENV = os.getenv("MAX_REVIEWS", "").strip()
MAX_REVIEWS = int(MAX_REVIEWS_ENV) if MAX_REVIEWS_ENV.isdigit() else None

# Ensure helper scripts inherit these too
os.environ["MODEL_NAME"] = MODEL_NAME
os.environ["RESTAURANT_NAME"] = RESTAURANT_NAME
os.environ["LLM_URL"] = LLM_URL


# -----------------------------
# OUTPUT FILENAMES (generic + backward compatible)
# -----------------------------
CACHE_FILE_GENERIC = "cache_reviews.csv"
PROCESSED_FILE_GENERIC = "reviews_processed.csv"
OWNER_JSON_GENERIC = "owner_summary.json"
OWNER_TXT_GENERIC = "owner_summary_readable.txt"
OWNER_FLAT_GENERIC = "owner_summary_flat.csv"
EMAIL_TXT_GENERIC = "weekly_owner_email.txt"

# backward-compat names (also written)
CACHE_FILE_BK = "bk_cache_final.csv"
PROCESSED_FILE_BK = "bk_final_report.csv"
OWNER_JSON_BK = "bk_owner_summary.json"
OWNER_TXT_BK = "bk_owner_summary.txt"

CHART_STARS = "chart_stars.png"
CHART_SENTIMENT = "chart_sentiment_pie.png"
CHART_TREND = "chart_trend.png"

# backward-compat chart names
CHART1_BK = "chart_1_stars.png"
CHART2_BK = "chart_2_sentiment_pie.png"
CHART3_BK = "chart_3_trend.png"

# If you still need the old bk_* outputs, set WRITE_LEGACY_OUTPUTS=1 in .env
WRITE_LEGACY_OUTPUTS = os.getenv("WRITE_LEGACY_OUTPUTS", "0").strip().lower() in ("1", "true", "yes", "y")


# -----------------------------
# NLTK SAFE SETUP
# -----------------------------
def ensure_nltk_resources():
    try:
        _ = stopwords.words("german")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        _ = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


ensure_nltk_resources()
sia = SentimentIntensityAnalyzer()
GERMAN_STOPS = set(stopwords.words("german"))
ENGLISH_STOPS = set(stopwords.words("english"))


# -----------------------------
# INPUT CSV AUTO-DETECT
# -----------------------------
def detect_input_csv() -> Path:
    """Choose input CSV:
    1) INPUT_CSV from env (absolute or relative to script folder)
    2) else: choose the most recently modified CSV in script folder (excluding runs/ and obvious outputs)
    """
    env_csv = os.getenv("INPUT_CSV", "").strip()
    if env_csv:
        p = Path(env_csv)
        if not p.is_absolute():
            p = HERE / p
        return p

    candidates = []
    for p in HERE.glob("*.csv"):
        name = p.name.lower()
        if name.startswith("cache_"):
            continue
        if "processed" in name or "owner_summary" in name or "weekly_owner_email" in name:
            continue
        if name.startswith("bk_"):
            continue
        candidates.append(p)

    # If nothing found, fallback to common legacy file if present
    legacy = HERE / "Burger King Data.csv"
    if legacy.exists():
        return legacy

    if not candidates:
        return legacy  # will error later with a clear message

    # Most recently modified
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# UTILS
# -----------------------------
def get_vader_score(text: str) -> float:
    return sia.polarity_scores(str(text))["compound"]


def vader_bucket(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def is_german_review(text: str) -> bool:
    """Heuristic German detection for filtering."""
    t = str(text).lower()
    if len(t) < 5:
        return False
    if any(ch in t for ch in "äöüß"):
        return True
    words = set(t.split())
    return len(words.intersection(GERMAN_STOPS)) >= len(words.intersection(ENGLISH_STOPS))


def is_english_review(text: str) -> bool:
    t = str(text).lower()
    words = set(t.split())
    # simple heuristic: more English stopwords than German
    return len(words.intersection(ENGLISH_STOPS)) > len(words.intersection(GERMAN_STOPS))


def parse_date(date_str):
    s = str(date_str).strip()
    if not s or s.lower() == "nan":
        return pd.NaT
    dt = pd.to_datetime(s, dayfirst=False, errors="coerce")
    if pd.notna(dt):
        return dt
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def clean_review_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clamp_star(x):
    """Return int 1..5 or NaN."""
    try:
        v = float(str(x).replace("★", "").strip())
        if v != v:
            return pd.NA
        v = int(round(v))
        if v < 1 or v > 5:
            return pd.NA
        return v
    except Exception:
        return pd.NA


def call_llm(prompt, timeout=240, retries=3):
    """Ollama /api/generate compatible call. Change MODEL_NAME only in env."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "format": "json",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(LLM_URL, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("response", "")
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)

        print(f"LLM Warning (attempt {attempt}/{retries}): {last_err}")
        time.sleep(2 * attempt)

    return "{}"


def parse_json_response(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"(\{.*\})", str(text), re.DOTALL)
        if match:
            clean = match.group(1).replace("'", '"')
            try:
                return json.loads(clean)
            except Exception:
                pass
    return {}


def normalize_llm_label(label: str) -> str:
    """Normalize various label spellings/languages to positive/neutral/negative."""
    s = (label or "").strip().lower()
    # common German outputs
    if s.startswith("pos") or "positiv" in s:
        return "positive"
    if s.startswith("neg") or "negativ" in s:
        return "negative"
    if s.startswith("neu") or "neutral" in s:
        return "neutral"
    return "neutral"


def llm_label_to_score(label: str, conf: float) -> float:
    """Map LLM label+confidence to a score in [-1, +1]."""
    lab = normalize_llm_label(label)
    try:
        c = float(conf)
    except Exception:
        c = 0.6
    c = max(0.0, min(1.0, c))
    if lab == "positive":
        return +c
    if lab == "negative":
        return -c
    return 0.0


def fuse_sentiment(vader_compound: float, llm_label: str, llm_conf: float,
                   gate: float = 0.25,
                   pos_th: float = 0.10,
                   neg_th: float = -0.10):
    """
    Gated fusion of VADER (compound) and LLM (label+confidence).
    Returns: fused_label, fused_score, w_vader, w_llm
    """
    try:
        v = float(vader_compound)
    except Exception:
        v = 0.0

    l = llm_label_to_score(llm_label, llm_conf)

    # If VADER is unsure (close to 0), trust LLM more.
    if abs(v) < gate:
        w_v, w_l = 0.25, 0.75
    else:
        w_v, w_l = 0.55, 0.45

    fused_score = (w_v * v) + (w_l * l)

    if fused_score >= pos_th:
        fused_label = "Positive"
    elif fused_score <= neg_th:
        fused_label = "Negative"
    else:
        fused_label = "Neutral"

    return fused_label, fused_score, w_v, w_l


# -----------------------------
# STRICT PROMPTS
# -----------------------------
def get_kitchen_prompt(text, stars):
    return f"""
Du bist ein Daten-Analyst. Extrahiere FAKTEN über das ESSEN.

Regeln:
1. Suche NUR nach: Geschmack, Temperatur, Frische, Portionen, Belag/Toppings, Zubereitung (zu trocken/zu dunkel/zu roh).
2. Ignoriere: Service, Preis, Wartezeit.
3. Wenn nichts erwähnt wird, antworte mit "n/a".
4. Kopiere kurze Zitate aus dem Text. Erfinde nichts.
5. Sterne sind nur Metadaten. Interpretiere sie NICHT. Nutze nur den Text.

Antworte als JSON:
{{
  "kitchen_pros": "Kurzes positives Zitat (oder 'n/a')",
  "kitchen_cons": "Kurzes negatives Zitat (oder 'n/a')"
}}

Sterne: {stars}
Text: "{text}"
""".strip()


def get_service_prompt(text, stars):
    return f"""
Du bist ein Daten-Analyst. Extrahiere FAKTEN über SERVICE & LIEFERUNG.

Regeln:
1. Suche nach: Wartezeit, Fahrer, Freundlichkeit, Bestellgenauigkeit, Kommunikation, Sauberkeit.
2. Ignoriere: Geschmack des Essens.
3. Kopiere kurze Zitate. Erfinde nichts.
4. Wenn nichts erwähnt wird, antworte mit "n/a".
5. Sterne sind nur Metadaten. Interpretiere sie NICHT. Nutze nur den Text.

Antworte als JSON:
{{
  "service_pros": "Kurzes positives Zitat (oder 'n/a')",
  "service_cons": "Kurzes negatives Zitat (oder 'n/a')"
}}

Sterne: {stars}
Text: "{text}"
""".strip()


def get_mgmt_prompt(text, stars):
    return f"""
Du bist ein Manager. Extrahiere FAKTEN über MANAGEMENT & PREIS.

Regeln:
1. Suche nach: Preis/Leistung, Ambiente, Sauberkeit/Organisation, Konkurrenzvergleich (z.B. McDonald's), Verbesserungsvorschläge.
2. Ignoriere Essen/Service (außer wenn es strategische Hinweise enthält, z.B. "zu teuer", "schlechte Organisation").
3. Kopiere kurze Zitate. Erfinde nichts.
4. Wenn nichts erwähnt wird, antworte mit "n/a".
5. Sterne sind nur Metadaten. Interpretiere sie NICHT. Nutze nur den Text.

Antworte als JSON:
{{
  "mgmt_pros": "Kurzes positives Zitat (oder 'n/a')",
  "mgmt_cons": "Kurzes negatives Zitat (oder 'n/a')"
}}

Sterne: {stars}
Text: "{text}"
""".strip()


def get_sentiment_prompt(text, stars):
    return f"""
Classify the overall sentiment based ONLY on the review text (ignore star rating).
Return STRICT JSON with exactly these keys:
{{
  "label": "positive" | "neutral" | "negative",
  "confidence": 0.0-1.0
}}

Stars (context only, do NOT use for the decision): {stars}
Review: "{text}"
""".strip()


# -----------------------------
# OWNER SUMMARY HELPERS
# -----------------------------
def chunk_list(items, chunk_size=10):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def build_evidence_lines(final_df: pd.DataFrame) -> list:
    lines = []
    for _, r in final_df.iterrows():
        text = str(r.get("review_text", "")).strip()
        stars = r.get("stars", "")
        vader = r.get("vader_score", 0.0)
        bucket = r.get("sentiment_bucket", "")

        if len(text) > 340:
            text = text[:340] + "..."

        kp = str(r.get("kitchen_pros", "")).strip()
        kc = str(r.get("kitchen_cons", "")).strip()
        sp = str(r.get("service_pros", "")).strip()
        sc = str(r.get("service_cons", "")).strip()
        mp = str(r.get("mgmt_pros", "")).strip()
        mc = str(r.get("mgmt_cons", "")).strip()

        line = f'- Stars:{stars} | VADER:{float(vader):.2f} | {bucket} | Review:"{text}"'
        extras = []
        if kp:
            extras.append(f"Kitchen+:{kp}")
        if kc:
            extras.append(f"Kitchen-:{kc}")
        if sp:
            extras.append(f"Service+:{sp}")
        if sc:
            extras.append(f"Service-:{sc}")
        if mp:
            extras.append(f"Mgmt+:{mp}")
        if mc:
            extras.append(f"Mgmt-:{mc}")

        if extras:
            line += " | " + " | ".join(extras)

        lines.append(line)
    return lines


def chunk_summary_prompt(evidence_lines: list) -> str:
    joined = "\n".join(evidence_lines)
    return f"""
Du bist ein Restaurant-Berater. Du analysierst Kundenfeedback und extrahierst NUR belegbare Punkte aus den Evidence-Lines.

WICHTIG (harte Regeln):
- Erfinde nichts. Nutze nur die Evidence-Lines.
- Verwende kurze wörtliche Belege direkt aus den Lines (Evidence-Zitate).
- Sterne sind nur Metadaten. Interpretiere sie NICHT.
- Gib NUR gültiges JSON zurück (kein Markdown, keine Kommentare).
- Wenn du keinen Punkt belegen kannst: gib leere Listen zurück.
- Jeder Improvement-Punkt MUSS "what_to_fix" UND "suggested_action" haben.
- Jeder Excellence-Punkt MUSS "what_to_keep" haben.

Ausgabeformat (strikt):
{{
  "excellence": [
    {{
      "theme": "kurzer Titel",
      "area": "Kitchen|Service|Management",
      "what_to_keep": "1 Satz",
      "how_to_market_it": "optional",
      "evidence": ["Zitat 1", "Zitat 2"]
    }}
  ],
  "improvements": [
    {{
      "theme": "kurzer Titel",
      "area": "Kitchen|Service|Management",
      "what_to_fix": "1 Satz",
      "suggested_action": "1 Satz",
      "evidence": ["Zitat 1", "Zitat 2"]
    }}
  ]
}}

EVIDENCE-LINES:
{joined}
""".strip()


def final_owner_summary_prompt(merged: dict) -> str:
    blob = json.dumps(merged, ensure_ascii=False)
    return f"""
Du bist ein Restaurant-Owner-Advisor. Erstelle eine klare, gut strukturierte Zusammenfassung für den Restaurantinhaber.

Regeln:
- Nutze nur die Themen + Belege aus dem JSON.
- Keine neuen Fakten erfinden.
- Priorisiere die wichtigsten 3-5 Punkte je Kategorie.
- Gib konkrete Handlungsempfehlungen (Quick Wins vs. langfristig).
- Output strikt als JSON.

Format:
{{
  "headline": "...",
  "areas_of_excellence": [
    {{"theme":"...", "area":"...", "what_to_keep":"...", "how_to_market_it":"...", "evidence":["..."]}}
  ],
  "areas_of_improvement": [
    {{"theme":"...", "area":"...", "what_to_fix":"...", "suggested_action":"...", "evidence":["..."]}}
  ],
  "quick_wins_next_7_days": ["...","..."],
  "longer_term_30_days": ["...","..."],
  "suggested_kpis": ["...","..."]
}}

INPUT THEMES JSON:
{blob}
""".strip()


def merge_chunk_outputs(chunk_outputs: list) -> dict:
    merged = {"excellence": [], "improvements": []}
    seen = set()

    def norm(x: str) -> str:
        return re.sub(r"\s+", " ", str(x).strip().lower())

    def add_items(key: str, items):
        if not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            theme = str(it.get("theme", "")).strip()
            area = str(it.get("area", "")).strip()
            if not theme or not area:
                continue
            sig = (norm(theme), norm(area))
            if sig in seen:
                continue
            seen.add(sig)
            merged[key].append(it)

    for out in chunk_outputs:
        if not isinstance(out, dict):
            continue
        add_items("excellence", out.get("excellence", []))
        add_items("improvements", out.get("improvements", []))

    return merged


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 60)
    print(f"RESTAURANT INSIGHTS PIPELINE — {RESTAURANT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    os.chdir(HERE)
    run_dir = get_run_dir()
    print(f"[RUN FOLDER] {run_dir}")

    # 1) LOAD DATA
    input_csv = detect_input_csv()
    if not input_csv.exists():
        print(f"ERROR: Input CSV not found: {input_csv}")
        print("Tip: set INPUT_CSV in .env or place a CSV in the script folder.")
        return

    print(f"[INPUT] Using CSV: {input_csv.name}")
    df = pd.read_csv(input_csv)

    # 2) PRE-PROCESS (generic + safe)
    print("[1] Processing & Filtering...")

    # Find review_text column
    if "review_text" not in df.columns:
        # try common alternatives
        for alt in ["text", "review", "content", "comment", "reviews"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "review_text"})
                break

    if "review_text" not in df.columns:
        print("Error: 'review_text' column missing.")
        print("Columns found:", list(df.columns))
        return

    # stars column discovery
    star_col = None
    for c in ["review_rating", "stars", "rating", "score"]:
        if c in df.columns:
            star_col = c
            break

    # date column discovery
    date_col = None
    for c in ["review_datetime_utc", "date", "timestamp", "created_at", "time"]:
        if c in df.columns:
            date_col = c
            break

    df["review_text"] = df["review_text"].astype(str).map(clean_review_text)
    df = df[df["review_text"].str.lower() != "nan"].copy()
    df = df[df["review_text"].str.len() >= MIN_TEXT_LEN].copy()

    # remove duplicates (common in exports)
    df = df.drop_duplicates(subset=["review_text"]).copy()

    if date_col:
        df["parsed_date"] = df[date_col].apply(parse_date)
    else:
        df["parsed_date"] = pd.NaT

    if star_col:
        df["stars"] = df[star_col].apply(clamp_star)
    else:
        df["stars"] = pd.NA

    # language filtering
    if LANGUAGE in ("de", "german"):
        df = df[df["review_text"].apply(is_german_review)].copy()
    elif LANGUAGE in ("en", "english"):
        df = df[df["review_text"].apply(is_english_review)].copy()
    # else auto: keep all

    print(f"    Reviews after filtering: {len(df)}")

    if MAX_REVIEWS:
        df = df.head(MAX_REVIEWS).copy()
        print(f"    Processing subset: {MAX_REVIEWS} rows.")

    # 3) ANALYSIS LOOP
    print("\n[2] Extracting Facts (Strict Mode + LLM)...")
    results = []

    cache_path = run_dir / CACHE_FILE_GENERIC
    cache_bk_path = run_dir / CACHE_FILE_BK

    if cache_path.exists():
        try:
            results = pd.read_csv(cache_path).to_dict("records")
            print(f"    Resumed {len(results)} from cache.")
        except Exception:
            pass

    processed_ids = set([str(r.get("idx")) for r in results])

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        idx_str = str(idx)
        if idx_str in processed_ids:
            continue

        text = row["review_text"]
        stars = row.get("stars", pd.NA)
        date_val = row.get("parsed_date", pd.NaT)

        json_k = parse_json_response(call_llm(get_kitchen_prompt(text, stars)))
        json_s = parse_json_response(call_llm(get_service_prompt(text, stars)))
        json_m = parse_json_response(call_llm(get_mgmt_prompt(text, stars)))
        json_sem = parse_json_response(call_llm(get_sentiment_prompt(text, stars)))

        vscore = get_vader_score(text)

        llm_label_raw = json_sem.get("label", json_sem.get("sentiment", "neutral"))
        llm_conf = json_sem.get("confidence", 0.6)
        llm_label_norm = normalize_llm_label(llm_label_raw)
        llm_score = llm_label_to_score(llm_label_norm, llm_conf)
        fused_label, fused_score, w_v, w_l = fuse_sentiment(vscore, llm_label_norm, llm_conf)

        results.append({
            "idx": idx_str,
            "date": date_val,
            "stars": stars,
            "review_text": text,
            "vader_score": vscore,
            "sentiment_bucket": vader_bucket(vscore),
            "sentiment_label": llm_label_norm.title(),  # legacy-friendly
            "llm_label": llm_label_norm,
            "llm_confidence": llm_conf,
            "llm_score": llm_score,
            "fused_label": fused_label,
            "fused_score": fused_score,
            "fused_w_vader": w_v,
            "fused_w_llm": w_l,
            "kitchen_pros": json_k.get("kitchen_pros", "n/a"),
            "kitchen_cons": json_k.get("kitchen_cons", "n/a"),
            "service_pros": json_s.get("service_pros", "n/a"),
            "service_cons": json_s.get("service_cons", "n/a"),
            "mgmt_pros": json_m.get("mgmt_pros", "n/a"),
            "mgmt_cons": json_m.get("mgmt_cons", "n/a"),
        })

        # save cache often
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(cache_path, index=False)
            if WRITE_LEGACY_OUTPUTS:
                pd.DataFrame(results).to_csv(cache_bk_path, index=False)

        time.sleep(0.02)

    final_df = pd.DataFrame(results)

    # cleanup: remove n/a
    for col in final_df.columns:
        if "pros" in col or "cons" in col:
            final_df[col] = final_df[col].replace(["n/a", "N/A", "nichts", "keine"], "")

    # --- FIX: ensure stars are numeric 1..5 for distribution/chart ---
    final_df["stars"] = final_df["stars"].apply(clamp_star)

    # Save processed outputs (canonical; optional legacy)
    processed_path = run_dir / PROCESSED_FILE_GENERIC
    processed_bk_path = run_dir / PROCESSED_FILE_BK

    final_df.to_csv(processed_path, index=False)
    if WRITE_LEGACY_OUTPUTS:
        final_df.to_csv(processed_bk_path, index=False)

    if WRITE_LEGACY_OUTPUTS:
        print(f"\n[3] Data saved to {processed_path.name} (+ legacy {processed_bk_path.name})")
    else:
        print(f"\n[3] Data saved to {processed_path.name}")

    if final_df.empty:
        print("No data to summarize.")
        return

    # 4) OWNER SUMMARY JSON
    print("\n" + "=" * 60)
    print("OWNER SUMMARY (LLM - ALL REVIEWS)")
    print("=" * 60)

    evidence_lines = build_evidence_lines(final_df)

    chunk_outputs = []
    for chunk in chunk_list(evidence_lines, chunk_size=10):
        prompt = chunk_summary_prompt(chunk)
        resp = parse_json_response(call_llm(prompt, timeout=180, retries=3))
        if isinstance(resp, dict) and resp:
            chunk_outputs.append(resp)

    merged = merge_chunk_outputs(chunk_outputs)
    final_prompt = final_owner_summary_prompt(merged)
    owner_summary = parse_json_response(call_llm(final_prompt, timeout=180, retries=3))
    if not isinstance(owner_summary, dict):
        owner_summary = {}

    owner_json_path = run_dir / OWNER_JSON_GENERIC
    owner_json_bk_path = run_dir / OWNER_JSON_BK
    with open(owner_json_path, "w", encoding="utf-8") as f:
        json.dump(owner_summary, f, ensure_ascii=False, indent=2)
    if WRITE_LEGACY_OUTPUTS:
        with open(owner_json_bk_path, "w", encoding="utf-8") as f:
            json.dump(owner_summary, f, ensure_ascii=False, indent=2)

    # write txt (readable later by owner_outputs.py too)
    owner_txt_path = run_dir / OWNER_TXT_GENERIC
    owner_txt_bk_path = run_dir / OWNER_TXT_BK
    owner_txt_path.write_text(json.dumps(owner_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if WRITE_LEGACY_OUTPUTS:
        owner_txt_bk_path.write_text(json.dumps(owner_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved owner summary: {owner_json_path.name}" + (f" (and {owner_json_bk_path.name})" if WRITE_LEGACY_OUTPUTS else ""))

    # 5) CHARTS
    print("\n[4] Generating charts...")

    # Chart: star distribution (use numeric stars only)
    stars_series = pd.to_numeric(final_df["stars"], errors="coerce").dropna().astype(int)
    plt.figure(figsize=(8, 5))
    stars_series.value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0).plot(kind="bar", edgecolor="black")
    plt.title("Star Rating Distribution", fontsize=14)
    plt.xlabel("Stars")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(run_dir / CHART_STARS)
    if WRITE_LEGACY_OUTPUTS:
        plt.savefig(run_dir / CHART1_BK)
    plt.close()

    # Chart: sentiment pie
    plt.figure(figsize=(6, 6))
    label_col = "fused_label" if "fused_label" in final_df.columns else "sentiment_bucket"
    counts = final_df[label_col].value_counts()
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title(f"Overall Sentiment ({'Fused' if label_col=='fused_label' else 'VADER Buckets'})", fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(run_dir / CHART_SENTIMENT)
    if WRITE_LEGACY_OUTPUTS:
        plt.savefig(run_dir / CHART2_BK)
    plt.close()

    # Chart: trend
    trend_df = final_df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["date"]).sort_values("date")

    if not trend_df.empty:
        trend_df = trend_df.set_index("date")
        score_col = "fused_score" if "fused_score" in trend_df.columns else "vader_score"
        monthly_trend = trend_df[score_col].resample("ME").mean()

        plt.figure(figsize=(10, 5))
        plt.plot(monthly_trend.index, monthly_trend.values, marker="o", linestyle="-", linewidth=2)
        plt.axhline(0, linestyle="--")
        plt.title(f"Customer Satisfaction Trend (Monthly Average - {'Fused' if score_col=='fused_score' else 'VADER'})", fontsize=14)
        plt.ylabel("Sentiment Score (-1.0 to +1.0)")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / CHART_TREND)
        if WRITE_LEGACY_OUTPUTS:
            plt.savefig(run_dir / CHART3_BK)
        plt.close()



    # -----------------------------
    # TERMINAL SUMMARY (quick KPIs)
    # -----------------------------
    try:
        print("\n========== RUN SUMMARY ==========")
        total = len(final_df)
        print(f"Total reviews processed: {total}")

        # Stars
        if "stars" in final_df.columns:
            stars_series = pd.to_numeric(final_df["stars"], errors="coerce").dropna()
            if len(stars_series) > 0:
                print(f"Average star rating: {stars_series.mean():.2f} / 5")
                dist = stars_series.value_counts().sort_index()
                print("Stars distribution:")
                for s, c in dist.items():
                    try:
                        s_int = int(float(s))
                    except Exception:
                        s_int = s
                    print(f"  {s_int}★ : {int(c)}")

        # Sentiment bucket (VADER)
        if "sentiment_bucket" in final_df.columns:
            counts = final_df["sentiment_bucket"].fillna("Unknown").value_counts()
            print("Sentiment distribution (VADER):")
            for k, v in counts.items():
                print(f"  {k}: {int(v)}")

        # Fused sentiment (recommended)
        if "fused_label" in final_df.columns:
            fused_counts = final_df["fused_label"].fillna("Neutral").value_counts()
            print("Sentiment distribution (Fused):")
            for k, v in fused_counts.items():
                print(f"  {k}: {int(v)}")

            # Disagreement rates (useful sanity check)
            try:
                llm_title = final_df["llm_label"].fillna("neutral").apply(normalize_llm_label).str.title()
                vader_title = final_df["sentiment_bucket"].fillna("Neutral")
                disagree_llm_vader = (llm_title != vader_title).mean() * 100.0
                disagree_fused_vader = (final_df["fused_label"].fillna("Neutral") != vader_title).mean() * 100.0
                print(f"Disagreement (LLM vs VADER): {disagree_llm_vader:.1f}%")
                print(f"Disagreement (Fused vs VADER): {disagree_fused_vader:.1f}%")
            except Exception:
                pass


        # Sentiment label from LLM (optional)
        if "llm_label" in final_df.columns:
            counts2 = final_df["llm_label"].fillna("neutral").apply(normalize_llm_label).value_counts()
            # show top 5 to avoid spam
            print("Sentiment labels (LLM) — top 5 (normalized):")
            for k, v in counts2.head(5).items():
                print(f"  {k}: {int(v)}")

        print("================================\n")
    except Exception as e:
        print("[WARN] Could not print run summary:", e)


    print("\nDone. Run folder:")
    print(f"- {run_dir}")


def run_subprocess_step(script_name: str):
    """Run a helper script without killing the whole pipeline."""
    import sys
    import subprocess

    script_path = HERE / script_name
    if not script_path.exists():
        print(f"[WARN] Missing script: {script_name} (skipping)")
        return

    print(f"\n[STEP] Running: {script_name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(HERE),
        env=os.environ.copy(),
        check=False
    )
    if result.returncode == 0:
        print(f"[OK] {script_name} finished successfully.")
    else:
        print(f"[WARN] {script_name} finished with return code {result.returncode}.")


if __name__ == "__main__":
    main()
    run_subprocess_step("owner_outputs.py")
    run_subprocess_step("send_weekly_report.py")
