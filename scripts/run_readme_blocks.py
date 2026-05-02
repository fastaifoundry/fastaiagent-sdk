"""Extract every fenced ``python`` code block from README.md and run it.

Skips blocks that look like prose-only snippets (no runnable
statement) and blocks that depend on a running platform server.
Real LLM calls — keys come from ~/.zshrc.

Each block runs in its own subprocess with a 60s timeout.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

README = Path("/Users/upendrabhandari/fastaiagent-sdk/README.md")
BLOCK_RE = re.compile(
    r"^```python\s*\n(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


def is_runnable(block: str) -> tuple[bool, str]:
    """Heuristic: skip blocks that are pure imports, or that reference
    objects we can't bring up locally (Platform endpoints, custom code
    not defined in the block, etc).
    """
    s = block.strip()
    if not s:
        return False, "empty"
    # Pure import-only blocks — handled by the doc-snippet test, not us.
    lines_no_imports = [
        ln for ln in s.splitlines()
        if ln.strip()
        and not ln.lstrip().startswith(("from ", "import ", "#"))
    ]
    if not lines_no_imports:
        return False, "imports only"
    # References Platform server or external endpoints we don't bring up.
    if "platform_url" in s or "PlatformConfig" in s:
        return False, "needs Platform"
    if "fa.connect(" in s or "fastaiagent.connect(" in s:
        return False, "needs Platform creds"
    # Heuristic: a runnable block must call ``.run(`` or print()
    # something or assign a variable that gets used.
    if ".run(" not in s and ".execute(" not in s and ".acomplete" not in s:
        # Could still be useful (e.g. configuration recipe). Run anyway.
        return True, "no .run() — running anyway as best-effort"
    return True, "runnable"


def main() -> int:
    text = README.read_text()
    blocks = list(BLOCK_RE.finditer(text))
    print(f"▸ extracted {len(blocks)} python blocks from README")
    print()

    # Each block is run in isolation; we don't share state between them.
    # That matches how a reader would copy-paste from the README.
    results = []
    for i, m in enumerate(blocks, 1):
        body = m["body"]
        # README blocks often elide the imports for readability — re-add
        # the canonical preamble so the snippet stands alone.
        preamble = textwrap.dedent("""
            import os
            os.environ.setdefault("FASTAIAGENT_TRACE", "false")  # quiet output
        """)
        # Skip untrusted heuristics: just always try to run and report.
        runnable, reason = is_runnable(body)
        if not runnable:
            print(f"  [block {i}] skipping ({reason})")
            results.append({"block": i, "status": "skipped", "reason": reason})
            continue

        full = preamble + "\n" + body
        start = time.time()
        try:
            proc = subprocess.run(
                ["zsh", "-lc", "python -"],
                input=full,
                capture_output=True,
                text=True,
                timeout=60,
            )
            took = round(time.time() - start, 1)
            ok = proc.returncode == 0
            print(
                f"  [block {i}] {'✓' if ok else '✗'} "
                f"exit={proc.returncode}  ({took}s)  reason={reason}"
            )
            if not ok:
                tail = (proc.stderr.splitlines()[-3:]
                        if proc.stderr else proc.stdout.splitlines()[-3:])
                for line in tail:
                    print(f"      {line}")
            results.append(
                {
                    "block": i,
                    "status": "passed" if ok else "failed",
                    "took_s": took,
                    "exit": proc.returncode,
                    "stderr_tail": proc.stderr.splitlines()[-5:] if proc.stderr else [],
                    "preview": body.splitlines()[0][:80],
                }
            )
        except subprocess.TimeoutExpired:
            print(f"  [block {i}] ⏱ timeout (>60s)")
            results.append({"block": i, "status": "timeout"})

    print("\n" + "=" * 60)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status in ("passed", "failed", "timeout", "skipped"):
        if counts.get(status):
            print(f"  {status:<10s} {counts[status]}")
    print(f"  total      {len(results)}")
    failures = [r for r in results if r["status"] in ("failed", "timeout")]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
