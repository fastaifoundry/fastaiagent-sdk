"""One-shot demo bootstrap so a human can verify the new
``?redact=true`` UI toggle in a real browser.

Run::

    zsh -lc 'python scripts/_demo_redaction_ui.py'

What it does:

1. Seeds a tmpdir SQLite trace DB with two known-secret spans
   (gen_ai.response.content and agent.output both contain
   ``sk-PROD…`` and a card number).
2. Installs a ``RedactionPolicy(mode="both")`` so the same flag
   covers capture *and* read paths. The seeded data was inserted
   raw — only the read path will mask it for this demo.
3. Starts the FastAPI Local UI on port 7891.
4. Opens the trace detail page in the default browser.

You should see a "Mask secrets" toggle in the top-right of the
page. Flip it and watch ``gen_ai.response.content`` swap between
the raw key and ``[REDACTED]``. Ctrl-C this script when done.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from fastaiagent.trace import RedactionPolicy, set_redaction_policy
from fastaiagent.ui.db import init_local_db
from fastaiagent.ui.server import build_app

TRACE_ID = "demo-redact-trace"
SPAN_ID = "demo-redact-span"
PORT = 7891
SECRET_LINE = "Here is your secret key: sk-PROD12345678901234567890ABCDEFGH and card 4111-1111-1111-1111"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fastaiagent-redact-demo-"))
    db_path = tmp / "local.db"
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    attrs = {
        "agent.name": "demo-leaky-bot",
        "fastaiagent.runner.type": "agent",
        "agent.input": "Give me a key",
        "agent.output": SECRET_LINE,
        "gen_ai.response.content": SECRET_LINE,
        "gen_ai.request.model": "gpt-4o-mini",
        "fastaiagent.cost.total_usd": 0.0012,
    }
    db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            SPAN_ID,
            TRACE_ID,
            None,
            "agent.demo-leaky-bot",
            now,
            now,
            "OK",
            json.dumps(attrs),
            "[]",
        ),
    )
    db.close()

    set_redaction_policy(
        RedactionPolicy(
            patterns=(
                r"sk-[A-Za-z0-9]{20,}",
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
            ),
            replacement="[REDACTED]",
            mode="both",
        )
    )

    app = build_app(db_path=str(db_path), no_auth=True)
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait until the server is responsive (uvicorn doesn't expose a ready
    # signal directly; poll the auth-status endpoint instead of sleeping).
    import urllib.request

    for _ in range(60):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/api/auth/status", timeout=0.5
            ):
                break
        except Exception:
            time.sleep(0.1)
    else:
        print("UI server failed to start", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{PORT}/traces/{TRACE_ID}"
    print()
    print("─" * 64)
    print(f"  UI running:  {url}")
    print()
    print("  In your browser:")
    print("    1. The trace shows a known fake key and card number.")
    print("    2. Top-right of the page: a 'Mask secrets' toggle.")
    print("    3. Flip it ON  → both values render as [REDACTED].")
    print("    4. Flip it OFF → the raw values come back.")
    print()
    print("  Ctrl-C to stop.")
    print("─" * 64)
    print()

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down…")
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
