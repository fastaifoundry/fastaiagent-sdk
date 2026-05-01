"""Run every example in examples/, capture outcomes, print a report.

Real LLM calls — API keys come from the user's ~/.zshrc (we re-exec
each example via ``zsh -lc`` so the keys propagate). Per-example
timeout caps hangs at 90 seconds.

Examples that need infra we don't have here are listed in SKIP and
included in the report as "skipped (reason)".

Output: a markdown table that fits in a terminal, plus a JSON
artifact at /tmp/example-run-report.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/Users/upendrabhandari/fastaiagent-sdk")
EXAMPLES = sorted(REPO.glob("examples/[0-9]*.py"))

# Examples that need external infra we don't bring up here.
SKIP: dict[str, str] = {
    "10_platform_sync.py": "needs Platform cloud credentials",
    "19_connect_e2e.py": "needs Platform cloud credentials",
    "34_platform_kb.py": "needs Platform cloud credentials",
    "29_kb_qdrant.py": "needs running qdrant server",
    "08_trace_langchain.py": "needs langchain + LangSmith credentials",
    "09_otel_export.py": "needs OTel collector listening on localhost",
    # Long-running uvicorn servers — we boot them separately as a smoke check.
    "33_deploy_fastapi.py": "long-running uvicorn server (smoke-tested separately)",
    "47_workflow_topology.py": "long-running uvicorn server (smoke-tested separately)",
    # Local UI server example — boots a server too.
    "35_local_ui.py": "long-running uvicorn server (smoke-tested separately)",
    "37_kb_ui.py": "long-running uvicorn server (smoke-tested separately)",
}

TIMEOUT = 90


def run_one(path: Path) -> dict:
    name = path.name
    if name in SKIP:
        return {"name": name, "status": "skipped", "reason": SKIP[name], "took_s": 0}

    start = time.time()
    try:
        proc = subprocess.run(
            ["zsh", "-lc", f"cd {REPO} && python examples/{name}"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        took = round(time.time() - start, 1)
        # Truncate output for the report
        stdout_tail = proc.stdout.splitlines()[-3:] if proc.stdout else []
        stderr_tail = proc.stderr.splitlines()[-3:] if proc.stderr else []
        ok = proc.returncode == 0
        return {
            "name": name,
            "status": "passed" if ok else "failed",
            "exit_code": proc.returncode,
            "took_s": took,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail if not ok else [],
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "timeout",
            "took_s": TIMEOUT,
            "reason": f"exceeded {TIMEOUT}s",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "name": name,
            "status": "error",
            "took_s": round(time.time() - start, 1),
            "reason": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    print(f"▸ running {len(EXAMPLES)} examples (4 at a time, {TIMEOUT}s timeout each)")
    print(f"  skip list ({len(SKIP)}): {', '.join(sorted(SKIP))}")
    print()

    results: list[dict] = []
    # 4 in parallel — most examples spend their time waiting on the LLM,
    # so this gives near-linear speedup without thrashing the laptop.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, p): p for p in EXAMPLES}
        for fut in as_completed(futures):
            r = fut.result()
            sym = {
                "passed": "✓",
                "failed": "✗",
                "timeout": "⏱",
                "skipped": "—",
                "error": "!",
            }[r["status"]]
            extra = ""
            if r["status"] == "failed":
                extra = (
                    f"  exit={r.get('exit_code')}  "
                    f"err={(r.get('stderr_tail') or [''])[-1][:120]}"
                )
            elif r["status"] == "skipped":
                extra = f"  ({r['reason']})"
            elif r["status"] == "timeout":
                extra = "  (timed out)"
            print(f"  {sym} {r['name']:<40s}  {r.get('took_s', 0):>5.1f}s{extra}")
            results.append(r)

    # Sort by example number for the final summary.
    results.sort(key=lambda r: r["name"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status in ("passed", "failed", "timeout", "skipped", "error"):
        print(f"  {status:<10s} {counts.get(status, 0)}")
    print(f"  total      {len(results)}")

    # Detailed failure dump
    failures = [r for r in results if r["status"] in ("failed", "timeout", "error")]
    if failures:
        print("\n--- failure detail ---")
        for r in failures:
            print(f"\n{r['name']} ({r['status']}):")
            for line in r.get("stderr_tail", []) or [r.get("reason", "")]:
                print(f"    {line}")

    Path("/tmp/example-run-report.json").write_text(json.dumps(results, indent=2))
    print(f"\nfull report: /tmp/example-run-report.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
