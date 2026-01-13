import os
import mimetypes
import smtplib
import ssl
from pathlib import Path
from email.message import EmailMessage


def _dedupe_attachments(attachments: list) -> list:
    """Remove duplicate attachments safely.

    - Dedupes by resolved path first.
    - If two different files share a name, keeps both (email clients show names),
      but you can still avoid accidental duplicates by passing a clean list.
    """
    seen = set()
    out = []
    for a in attachments or []:
        if a is None:
            continue
        p = Path(a).expanduser()
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def send_email_with_attachments(
    to_email: str,
    subject: str,
    body: str,
    attachments: list,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
) -> bool:
    """Send an email with optional attachments.

    attachments: list of file paths (str/Path). Missing files are skipped.
    """
    try:
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body or "")

        # Attach files (deduped + skip missing)
        for path in _dedupe_attachments(attachments):
            if not path.exists() or not path.is_file():
                print(f"  [SKIP] Attachment not found: {path}")
                continue

            ctype, encoding = mimetypes.guess_type(str(path))
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)

            with open(path, "rb") as f:
                data = f.read()

            msg.add_attachment(
                data,
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )

        context = ssl.create_default_context()

        # 465 = implicit TLS, 587 = STARTTLS
        if int(smtp_port) == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        return True

    except Exception as e:
        print("Email send failed:", e)
        return False
