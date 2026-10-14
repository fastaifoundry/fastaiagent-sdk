"""1.89.0 Phase 4 (replay isolation and provenance) — seed, then drive and assert.

Called by ``scripts/capture-replay-isolation.sh``:

    python scripts/capture_replay_isolation.py seed
    python scripts/capture_replay_isolation.py drive <base-url>

What it proves, on screen:

* a rerun runs only the tools you give it: ``refund_order`` is blocked by
  default (the real function is never called), Live is disabled in a UI process
  that has no implementation, a stub returns the canned value;
* the rerun says what happened (per-tool chips, the blocked-call warning, the
  local-only note) and the diff lists the changed tool call;
* replay traces carry a source badge, the Traces list can hide them, and
  Analytics / Overview / Agents leave them out and say so (Include experiments
  brings them back);
* a rerun's trace links back to the trace it replays.

Offline: the agent is a ``FunctionModel``; reruns use recorded LLM responses.
"""

from __future__ import annotations

import os
import re
import sys

AGENT = "refund-desk"


def build_agent(calls: list[str]):  # type: ignore[no-untyped-def]
    from fastaiagent import Agent, FunctionTool
    from fastaiagent.testing import FunctionModel

    def refund_order(order_id: str) -> str:
        """Refund an order (charges the card back — a real side effect)."""
        calls.append(order_id)
        return f"refund issued for {order_id}"

    def responder(messages):  # type: ignore[no-untyped-def]
        tool_results = [m for m in messages if str(getattr(m, "role", "")).endswith("tool")]
        if tool_results:
            return f"Done: {tool_results[-1].content}."
        question = next(str(m.content) for m in reversed(messages) if m.role == "user")
        order = re.search(r"ORD-\d+", question)
        return ("", [{"name": "refund_order", "arguments": {"order_id": order.group(0)}}])

    return Agent(
        name=AGENT,
        system_prompt="You process refunds with the refund_order tool.",
        llm=FunctionModel(responder, model="gpt-4o-mini"),
        tools=[FunctionTool(name="refund_order", fn=refund_order)],
    )


def seed() -> int:
    from fastaiagent.trace.replay import Replay

    calls: list[str] = []
    agent = build_agent(calls)
    first = agent.run("Please refund ORD-1001")
    agent.run("Please refund ORD-1002")
    print(f"▸ two production runs; refund_order ran {len(calls)}x")

    before = len(calls)
    result = Replay.load(first.trace_id).fork_at(0).with_determinism("recorded").rerun()
    print(
        f"▸ python rerun {result.trace_id}: dispositions={result.tool_dispositions} "
        f"blocked={result.blocked_tool_calls} exported={result.exported}"
    )
    if len(calls) != before:
        raise SystemExit("the rerun called the real refund_order")
    if result.tool_dispositions.get("refund_order") != "blocked":
        raise SystemExit(f"expected refund_order blocked, got {result.tool_dispositions}")
    with open("production_trace.txt", "w") as fh:
        fh.write(first.trace_id)
    return 0


# ---------------------------------------------------------------------------


def drive(base: str, production_trace: str) -> int:
    from _capture189 import Capture
    from playwright.sync_api import sync_playwright

    cap = Capture("capture-replay-isolation", base)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = cap.new_page(browser)

        # -- 1. Traces: source badge, Hide experiments, Source box -------------
        print("▸ traces list")
        all_rows = cap.traces()
        replay_rows = [r for r in all_rows if r.get("source") == "replay"]
        cap.check(
            len(all_rows) == 3 and len(replay_rows) == 1,
            "3 traces, 1 of them a replay",
            str([(r["name"], r.get("source")) for r in all_rows]),
        )
        cap.goto(page, "/traces")
        cap.text(page, "3 traces", "Traces header counts 3")
        badges = page.get_by_test_id("trace-source-badge")
        cap.visible(badges.filter(has_text="replay"), "the rerun carries a 'replay' source badge")
        cap.check(badges.count() == 1, "only the rerun is badged", str(badges.count()))
        cap.shot(page, "189-replay-01-traces-source-badge")

        page.get_by_role("button", name="Hide experiments").click()
        cap.settle(page)
        cap.check(
            "exclude_experiments=true" in page.url, "Hide experiments is in the URL", page.url
        )
        cap.text(page, "2 traces", "Hide experiments: 2 traces")
        cap.hidden(page.get_by_test_id("trace-source-badge"), "no replay row left")
        cap.shot(page, "189-replay-02-traces-hide-experiments")

        cap.goto(page, "/traces?source=replay")
        cap.text(page, "1 trace", "source=replay: total counts the filtered trace")
        cap.check(page.get_by_label("Source").input_value() == "replay", "Source box reads replay")

        # -- 2. Production views leave the replay out, and say so ---------------
        print("▸ analytics / overview / agents")
        cap.goto(page, "/analytics", 1500)
        note = page.get_by_test_id("experiments-note")
        cap.visible(note, "Analytics says what it left out")
        cap.text(
            note,
            re.compile(r"Excludes 1 experiment trace .*replay 1"),
            "Analytics: Excludes 1 experiment trace (replay 1)",
        )
        cap.shot(page, "189-replay-03-analytics-excludes")
        page.get_by_label("Include experiments").first.click()
        cap.settle(page, 1000)
        cap.text(
            page.get_by_test_id("experiments-note"),
            re.compile(r"^Includes 1 experiment trace"),
            "Include experiments on: Includes 1 experiment trace",
        )
        page.get_by_label("Include experiments").first.click()
        cap.settle(page, 500)

        cap.goto(page, "/", 1200)
        cap.text(
            page.get_by_test_id("experiments-note"),
            re.compile(r"Excludes 1 experiment trace from the last 24h"),
            "Home: Excludes 1 experiment trace from the last 24h",
        )
        cap.goto(page, "/agents")
        cap.visible(
            page.get_by_test_id("production-only-note"), "Agents: counts production runs only"
        )

        # -- 3. Rerun from the UI: Block (default), then Stub --------------------
        print("▸ fork dialog and reruns")
        cap.goto(page, f"/traces/{production_trace}")
        cap.visible(
            page.get_by_role("link", name="Open in Replay"), "production trace offers Replay"
        )
        cap.goto(page, f"/traces/{production_trace}/replay", 1200)
        page.get_by_role("button", name="Fork here").click()
        dialog = page.get_by_role("dialog")
        cap.visible(
            dialog.get_by_role("heading", name="Fork and rerun with modifications"),
            "fork dialog opens",
        )
        dialog.locator("#fork-determinism").select_option("recorded")
        cap.enabled(dialog.locator("#fork-on-miss"), "on-miss applies once Recorded is picked")
        dialog.get_by_role("tab", name="Tools").click()
        row = dialog.get_by_test_id("fork-tool-refund_order")
        cap.visible(row, "Tools tab lists refund_order")
        cap.check(
            row.get_by_role("button", name="Block refund_order").get_attribute("aria-pressed")
            == "true",
            "refund_order starts blocked",
        )
        cap.disabled(
            row.get_by_role("button", name="Live refund_order"),
            "Live is disabled: this UI process has no implementation",
        )
        cap.text(
            row,
            "This UI process has no implementation of this tool.",
            "Live says why it is unavailable",
        )
        cap.text(dialog, "Tools: 1 blocked · 0 stubbed · 0 live")
        cap.shot(page, "189-replay-04-fork-dialog-tools")

        dialog.get_by_role("button", name="Rerun with modifications").click()
        cap.text(page, "Rerun complete", timeout=30_000)
        summary = page.get_by_test_id("rerun-tool-summary")
        cap.visible(
            summary.get_by_test_id("tool-disposition-refund_order").filter(has_text="blocked"),
            "chip: refund_order · blocked",
        )
        cap.text(
            summary,
            re.compile(r"The model called blocked tool\s+refund_order ×1"),
            "blocked-call warning names refund_order ×1",
        )
        cap.text(summary, "Local only: not counted in Analytics and not exported.")
        cap.text(summary, "LLM: recorded · tool policy: block")
        cap.visible(
            page.get_by_test_id("tool-diffs").get_by_text("Tool calls that changed"),
            "the diff lists the changed tool call",
        )
        cap.visible(page.get_by_test_id("tool-diff-refund_order"), "… for refund_order")
        page.get_by_role("heading", name="Rerun complete").scroll_into_view_if_needed()
        cap.shot(page, "189-replay-05-rerun-blocked")
        rerun_link = page.get_by_role("link", name=re.compile(r"^[0-9a-f]{16}…$")).first
        rerun_href = rerun_link.get_attribute("href") or ""

        # Stub this time.
        page.get_by_role("button", name="Fork here").click()
        dialog = page.get_by_role("dialog")
        dialog.locator("#fork-determinism").select_option("recorded")
        dialog.get_by_role("tab", name="Tools").click()
        row = dialog.get_by_test_id("fork-tool-refund_order")
        row.get_by_role("button", name="Stub refund_order").click()
        row.get_by_label("Stub response for refund_order").fill(
            '{"status": "refund queued (stub)"}'
        )
        cap.text(dialog, "Tools: 0 blocked · 1 stubbed · 0 live")
        dialog.get_by_role("button", name="Rerun with modifications").click()
        cap.visible(
            page.get_by_test_id("tool-disposition-refund_order").filter(has_text="stub"),
            "chip: refund_order · stub",
            timeout=30_000,
        )
        cap.hidden(
            page.get_by_test_id("rerun-tool-summary").get_by_role("alert"),
            "no blocked-call warning for a stubbed tool",
        )

        # -- 4. The rerun's trace links back to its source -------------------------
        print("▸ rerun trace detail")
        cap.goto(page, rerun_href, 1000)
        banner = page.get_by_test_id("replay-origin-banner")
        cap.visible(banner, "rerun trace shows the replay-origin banner")
        cap.visible(
            banner.get_by_role("link", name=production_trace[:8], exact=False),
            "banner links to the replayed trace",
        )
        cap.text(banner, "tool policy: block")
        cap.text(banner, "via ui")
        cap.text(page, "source: replay", "subtitle: source: replay")
        cap.shot(page, "189-replay-06-rerun-origin-banner")

        reruns = [r for r in cap.traces() if r.get("source") == "replay"]
        cap.check(len(reruns) == 3, "3 replay traces in all (1 Python, 2 UI)", str(len(reruns)))

        browser.close()
    return cap.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        raise SystemExit(seed())
    if sys.argv[1:2] == ["drive"]:
        with open(os.path.join(sys.argv[3], "production_trace.txt")) as fh:
            raise SystemExit(drive(sys.argv[2], fh.read().strip()))
    raise SystemExit("usage: capture_replay_isolation.py seed | drive <base-url> <seed-dir>")
