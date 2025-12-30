import os
from datetime import datetime
from pathlib import Path
from email_reporter import send_email_with_attachments

BASE_DIR = Path(__file__).resolve().parent
RUNS_ROOT = BASE_DIR / "runs"


def load_dotenv(path: Path, overwrite: bool = True):
    """Minimal .env loader (overwrite=True avoids old env values)."""
    if not path.exists():
        print(f"[ENV] .env not found at: {path}")
        return

    print(f"[ENV] Loading .env from: {path}")
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


def resolve_run_dir() -> Path:
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

    return BASE_DIR


def load_subject_and_body(path: Path):
    content = path.read_text(encoding="utf-8").strip()
    subject = "Weekly Restaurant Review Report"
    body = content

    if content.lower().startswith("subject:"):
        lines = content.splitlines()
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()

    return subject, body


def parse_recipients(value: str):
    if not value:
        return []
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    return [p for p in parts if p]


def main():
    print("=" * 60)
    print("SENDING WEEKLY OWNER REPORT (Email)")
    print("=" * 60)

    # Always reload env
    load_dotenv(BASE_DIR / ".env", overwrite=True)

    run_dir = resolve_run_dir()
    print(f"[RUN FOLDER] {run_dir}")

    email_text_file = run_dir / "weekly_owner_email.txt"
    if not email_text_file.exists():
        raise SystemExit(f"Missing email text file: {email_text_file} (run owner_outputs.py first)")

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int((os.getenv("SMTP_PORT", "587").strip() or "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()

    recipients = parse_recipients(os.getenv("OWNER_EMAILS", "").strip())
    if not recipients:
        recipients = parse_recipients(os.getenv("OWNER_EMAIL", "").strip())

    missing = [k for k in ["SMTP_HOST", "SMTP_USER", "SMTP_PASS"] if not os.getenv(k)]
    if missing:
        raise SystemExit("Missing env vars: " + ", ".join(missing))
    if not recipients:
        raise SystemExit("Missing recipient. Set OWNER_EMAIL (or OWNER_EMAILS) in .env")

    subject, body = load_subject_and_body(email_text_file)

    # ensure date visible even if subject line didn't include it
    if "—" not in subject:
        subject = f"{subject} — {datetime.now().strftime('%Y-%m-%d')}"

    # Attach whatever exists (generic + backward compatible)
    attachment_names = [
        # charts
        "chart_stars.png",
        "chart_sentiment_pie.png",
        "chart_trend.png",
        # summaries
        "owner_summary.json",
        "owner_summary_readable.txt",
        "owner_summary_flat.csv",
        "weekly_owner_email.txt",
        # processed data
        "reviews_processed.csv",
        "cache_reviews.csv",
    ]

    attachments = []
    for name in attachment_names:
        p = run_dir / name
        if p.exists():
            attachments.append(str(p))

    print("\n[ENV CHECK]")
    print("  SMTP_HOST:", smtp_host)
    print("  SMTP_PORT:", smtp_port)
    print("  SMTP_USER:", smtp_user)
    print("  RECIPIENTS:", ", ".join(recipients))

    print("\n[ATTACHMENTS]")
    if attachments:
        for a in attachments:
            print("  +", a)
    else:
        print("  (No attachments found!)")

    ok_all = True
    for to_email in recipients:
        print(f"\n[SENDING EMAIL] -> {to_email}")
        ok = send_email_with_attachments(
            to_email=to_email,
            subject=subject,
            body=body,
            attachments=attachments,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_pass=smtp_pass,
        )
        print("  RESULT:", "✅ Sent!" if ok else "❌ Failed")
        ok_all = ok_all and ok

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
