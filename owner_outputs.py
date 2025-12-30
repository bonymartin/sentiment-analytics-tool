import json
import os
import re
from datetime import datetime
from pathlib import Path
from collections import Counter
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "").strip() or "Restaurant"
MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "LLM"
WRITE_LEGACY_OUTPUTS = os.getenv("WRITE_LEGACY_OUTPUTS", "0").strip().lower() in ("1", "true", "yes", "y")

RUNS_ROOT = BASE_DIR / "runs"


def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        t = x.strip()
        if not t:
            return []
        return [t]
    return [x]


def normalize_area(s: str) -> str:
    s = clean_text(s)
    if not s:
        return "General"
    # Keep short, avoid multi slashes explosion
    s = s.replace("\\", "/")
    s = re.sub(r"\s*/\s*", "/", s)
    return s


def get_run_dir() -> Path:
    env_dir = os.getenv("RUN_DIR", "").strip()
    if env_dir:
        d = Path(env_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    last = RUNS_ROOT / "last_run.txt"
    if last.exists():
        try:
            d = Path(last.read_text(encoding="utf-8").strip())
            if d.exists():
                os.environ["RUN_DIR"] = str(d)
                return d
        except Exception:
            pass

    # fallback: newest run folder
    RUNS_ROOT.mkdir(exist_ok=True)
    runs = sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        os.environ["RUN_DIR"] = str(runs[0])
        return runs[0]

    # last fallback: create one
    d = RUNS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M")
    d.mkdir(parents=True, exist_ok=True)
    os.environ["RUN_DIR"] = str(d)
    return d


def load_owner_summary(run_dir: Path) -> dict:
    # Prefer canonical names; fall back to legacy
    cand = [
        run_dir / "owner_summary.json",
        run_dir / "bk_owner_summary.json",
        BASE_DIR / "owner_summary.json",
        BASE_DIR / "bk_owner_summary.json",
    ]
    for p in cand:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def load_processed_reviews(run_dir: Path) -> pd.DataFrame:
    cand = [
        run_dir / "reviews_processed.csv",
        run_dir / "bk_final_report.csv",
        BASE_DIR / "reviews_processed.csv",
        BASE_DIR / "bk_final_report.csv",
    ]
    for p in cand:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    return pd.DataFrame()


def split_phrases(cell) -> list:
    """
    Extract short phrases from pros/cons cells.
    Handles:
      - NaN
      - single sentence
      - semicolon separated phrases
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    # split by ; or | first, then by newlines
    parts = re.split(r"[;\n\|]+", s)
    out = []
    for p in parts:
        t = clean_text(p)
        # keep reasonably short phrases
        if 2 <= len(t) <= 140:
            out.append(t)
    return out


def top_phrases(df: pd.DataFrame, positive: bool = True, k: int = 3):
    if df.empty:
        return []

    # determine label column preference
    label_col = None
    if "fused_label" in df.columns:
        label_col = "fused_label"
        pos_key, neg_key = "Positive", "Negative"
    elif "sentiment_bucket" in df.columns:
        label_col = "sentiment_bucket"
        pos_key, neg_key = "Positive", "Negative"
    elif "sentiment_label" in df.columns:
        label_col = "sentiment_label"
        pos_key, neg_key = "Positive", "Negative"

    if not label_col:
        return []

    target = pos_key if positive else neg_key
    sub = df[df[label_col].astype(str).str.lower() == target.lower()].copy()
    if sub.empty:
        return []

    cols = ["kitchen_pros", "service_pros", "mgmt_pros"] if positive else ["kitchen_cons", "service_cons", "mgmt_cons"]
    existing = [c for c in cols if c in sub.columns]
    if not existing:
        return []

    cnt = Counter()
    for c in existing:
        for cell in sub[c].tolist():
            for phrase in split_phrases(cell):
                cnt[phrase] += 1

    return cnt.most_common(k)


def compute_snapshot(df: pd.DataFrame) -> dict:
    snap = {
        "n_reviews": int(len(df)) if df is not None else 0,
        "avg_stars": None,
        "star_counts": {},
        "sent_counts": {},
        "overall_score": None,
        "score_source": None,
        "label_source": None,
    }
    if df is None or df.empty:
        return snap

    # stars
    if "stars" in df.columns:
        stars = pd.to_numeric(df["stars"], errors="coerce")
        stars = stars.dropna()
        if not stars.empty:
            snap["avg_stars"] = float(stars.mean())
            vc = stars.round().astype(int).value_counts().to_dict()
            snap["star_counts"] = {f"{i}★": int(vc.get(i, 0)) for i in [1, 2, 3, 4, 5]}

    # sentiment (prefer fused)
    if "fused_label" in df.columns:
        lab = df["fused_label"].astype(str)
        vc = lab.value_counts().to_dict()
        snap["sent_counts"] = {k: int(v) for k, v in vc.items()}
        snap["label_source"] = "Fused"
    elif "sentiment_bucket" in df.columns:
        lab = df["sentiment_bucket"].astype(str)
        vc = lab.value_counts().to_dict()
        snap["sent_counts"] = {k: int(v) for k, v in vc.items()}
        snap["label_source"] = "VADER"

    # overall score (prefer fused_score, else vader_score)
    if "fused_score" in df.columns:
        sc = pd.to_numeric(df["fused_score"], errors="coerce").dropna()
        if not sc.empty:
            snap["overall_score"] = float(sc.mean())
            snap["score_source"] = "Fused"
    elif "vader_score" in df.columns:
        sc = pd.to_numeric(df["vader_score"], errors="coerce").dropna()
        if not sc.empty:
            snap["overall_score"] = float(sc.mean())
            snap["score_source"] = "VADER"

    return snap


def build_email_text(summary: dict, snapshot: dict, top_pos, top_neg) -> str:
    """
    Keep the *previous version* structure, but show fused metrics when available.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_date = os.getenv("RUN_STARTED_AT", "") or now

    # subject line stays exactly like previous version
    subject = f"Subject: Weekly Review Summary — {RESTAURANT_NAME} — {datetime.now().strftime('%Y-%m-%d')}"

    headline = clean_text(summary.get("headline", "Restaurant Performance Enhancement Guide for Owner"))
    lines = []
    lines.append(subject)
    lines.append("")
    lines.append(f"{headline} — {RESTAURANT_NAME}")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append(f"Run date: {run_date}")
    lines.append("")

    lines.append("DATA SNAPSHOT")
    lines.append(f"• Reviews analyzed: {snapshot.get('n_reviews', 0)}")

    score = snapshot.get("overall_score", None)
    score_src = snapshot.get("score_source", "")
    if score is None:
        lines.append("• Overall sentiment: n/a")
    else:
        # show source explicitly, but keep old formatting
        tag = f"{score_src} avg" if score_src else "avg"
        lines.append(f"• Overall sentiment ({tag}): {score:.2f}  (-1 to +1)")

    # sentiment counts (prefer fused)
    sent = snapshot.get("sent_counts", {}) or {}
    label_src = snapshot.get("label_source", "")
    if sent:
        # normalize keys to Positive/Neutral/Negative order
        def pick(d, key):
            for k in d.keys():
                if str(k).lower() == key.lower():
                    return int(d[k])
            return 0
        p = pick(sent, "Positive")
        n = pick(sent, "Negative")
        u = pick(sent, "Neutral")
        tag = label_src if label_src else "Sentiment"
        lines.append(f"• Sentiment ({tag}): Positive={p}, Neutral={u}, Negative={n}")

    stars = snapshot.get("star_counts", {}) or {}
    if stars:
        star_str = ", ".join([f"{k}={stars.get(k,0)}" for k in ["1★", "2★", "3★", "4★", "5★"]])
        lines.append(f"• Stars: {star_str}")
    lines.append("")

    # strengths
    lines.append("TOP STRENGTHS (keep & promote)")
    exc = as_list(summary.get("areas_of_excellence", []))[:3]
    if not exc:
        lines.append("• No strong repeated strengths detected in this run.")
    else:
        for i, it in enumerate(exc, 1):
            area = normalize_area(it.get("area", ""))
            theme = clean_text(it.get("theme", ""))
            keep = clean_text(it.get("what_to_keep", "")) or "Beibehalten und standardisieren."
            promo = clean_text(it.get("how_to_market_it", "")) or "In Google/Online-Bewertungen hervorheben."
            ev = [clean_text(x) for x in as_list(it.get("evidence", [])) if clean_text(x)]
            lines.append(f"{i}. [{area}] {theme}")
            lines.append(f"   Keep: {keep}")
            lines.append(f"   Promote: {promo}")
            if ev:
                lines.append(f'   Evidence: "{ev[0]}"')
    lines.append("")

    # issues
    lines.append("TOP ISSUES (prioritized)")
    imp = as_list(summary.get("areas_of_improvement", []))[:5]
    if not imp:
        lines.append("• No major repeated issues detected in this run.")
    else:
        for i, it in enumerate(imp, 1):
            area = normalize_area(it.get("area", ""))
            theme = clean_text(it.get("theme", ""))
            fix = clean_text(it.get("what_to_fix", "")) or "Problem genauer beschreiben und messen."
            action = clean_text(it.get("suggested_action", "")) or "Klaren Standardprozess definieren und schulen."
            ev = [clean_text(x) for x in as_list(it.get("evidence", [])) if clean_text(x)]
            lines.append(f"{i}. [{area}] {theme}")
            lines.append(f"   Fix: {fix}")
            lines.append(f"   Action: {action}")
            if ev:
                lines.append(f'   Evidence: "{ev[0]}"')
    lines.append("")

    # quotes
    lines.append("RECURRING QUOTES (from extraction)")
    if top_pos:
        lines.append("• Positive:")
        for phrase, cnt in top_pos[:3]:
            lines.append(f'  - "{phrase}" ({cnt}x)')
    if top_neg:
        lines.append("• Negative:")
        for phrase, cnt in top_neg[:3]:
            lines.append(f'  - "{phrase}" ({cnt}x)')
    if not top_pos and not top_neg:
        lines.append("• No recurring phrases extracted in this run.")
    lines.append("")

    # action plan (support both schemas; fill from improvements if missing)
    lines.append("ACTION PLAN")

    quick = as_list(summary.get("quick_wins_next_7_days", []))[:3]
    longer = as_list(summary.get("longer_term_30_days", []))[:3]

    # If missing, derive from improvements suggested actions
    if not quick and imp:
        quick = [clean_text(x.get("suggested_action", "")) for x in imp[:3] if clean_text(x.get("suggested_action", ""))]
        quick = quick[:3]
    if not longer and imp:
        longer = [clean_text(x.get("suggested_action", "")) for x in imp[3:8] if clean_text(x.get("suggested_action", ""))]
        longer = longer[:3]

    lines.append("• Quick wins (next 7 days):")
    if quick:
        for x in quick:
            lines.append(f"  - {clean_text(x)}")
    else:
        lines.append("  - Standardize one service/kitchen checklist for every shift.")
        lines.append("  - Review top 5 negative reviews with staff and agree on fixes.")

    lines.append("• Longer term (next 30 days):")
    if longer:
        for x in longer:
            lines.append(f"  - {clean_text(x)}")
    else:
        lines.append("  - Reduce repeated complaints by adding process + training + owner checks.")

    lines.append("")

    # KPIs (support both schemas; fall back to sensible defaults)
    kpis = as_list(summary.get("suggested_kpis", []))[:4]
    if not kpis:
        # Some older schemas used different key names
        kpis = as_list(summary.get("suggested_kpi_for_30_days", []))[:4]

    lines.append("KPIs TO TRACK (simple)")
    if kpis:
        for x in kpis[:4]:
            lines.append(f"• {clean_text(x)}")
    else:
        lines.append("• Average Wait Time (Min.) — Timer: 20 Bestellungen pro Schicht messen.")
        lines.append("• Order Accuracy (%) — Stichprobe: 30 Bestellungen/Woche prüfen.")
        lines.append("• Food Quality Rating (1–5) — QR-Umfrage auf Bon / Tischaufsteller.")
        lines.append("• Cleanliness Checklist Score — Tägliche Checkliste + wöchentliches Audit.")

    lines.append("")
    lines.append(f"Generated on: {now}")
    return "\n".join(lines)


def write_flat_csv(summary: dict, run_dir: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for it in as_list(summary.get("areas_of_excellence", [])):
        rows.append({
            "generated_at": now,
            "category": "excellence",
            "area": normalize_area(it.get("area", "")),
            "theme": clean_text(it.get("theme", "")),
            "keep_or_fix": clean_text(it.get("what_to_keep", "")),
            "action_or_marketing": clean_text(it.get("how_to_market_it", "")),
        })
    for it in as_list(summary.get("areas_of_improvement", [])):
        rows.append({
            "generated_at": now,
            "category": "improvement",
            "area": normalize_area(it.get("area", "")),
            "theme": clean_text(it.get("theme", "")),
            "keep_or_fix": clean_text(it.get("what_to_fix", "")),
            "action_or_marketing": clean_text(it.get("suggested_action", "")),
        })
    if not rows:
        return

    out = run_dir / "owner_summary_flat.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    if WRITE_LEGACY_OUTPUTS:
        pd.DataFrame(rows).to_csv(run_dir / "bk_owner_summary_flat.csv", index=False, encoding="utf-8")


def main():
    run_dir = get_run_dir()
    summary = load_owner_summary(run_dir)
    df = load_processed_reviews(run_dir)

    # snapshot + quotes
    snapshot = compute_snapshot(df)
    top_pos = top_phrases(df, positive=True, k=3)
    top_neg = top_phrases(df, positive=False, k=3)

    # write owner_summary_readable + weekly email
    email_text = build_email_text(summary, snapshot, top_pos, top_neg)

    readable = run_dir / "owner_summary_readable.txt"
    readable.write_text(email_text, encoding="utf-8")
    if WRITE_LEGACY_OUTPUTS:
        (run_dir / "bk_owner_summary.txt").write_text(email_text, encoding="utf-8")

    weekly = run_dir / "weekly_owner_email.txt"
    weekly.write_text(email_text, encoding="utf-8")

    # also copy JSON to run folder (if summary came from base dir)
    out_json = run_dir / "owner_summary.json"
    if not out_json.exists() and summary:
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if WRITE_LEGACY_OUTPUTS and summary:
        (run_dir / "bk_owner_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_flat_csv(summary, run_dir)

    print("OWNER OUTPUTS")
    print("=" * 40)
    print(f"Run dir: {run_dir}")
    print(f"Wrote: {weekly.name}, {readable.name}, owner_summary_flat.csv")


if __name__ == "__main__":
    main()
