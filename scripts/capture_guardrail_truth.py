"""1.89.0 Phase 5 (guardrails tell the truth) — seed, then drive and assert.

Called by ``scripts/capture-guardrail-truth-screenshots.sh``:

    python scripts/capture_guardrail_truth.py seed
    python scripts/capture_guardrail_truth.py drive <base-url>

What it proves, on screen:

* guardrail events are recorded with tracing on and no ``FASTAIAGENT_UI_ENABLED``
  anywhere (the list used to be empty by default);
* each row says what the check did to the run (``effect``): blocked, rewrote,
  warned, observe-only; passes are hidden by default;
* the event page shows exactly the text the rule judged, the rule as it ran
  (the regex pattern from the stored rule snapshot), and what happened next;
* a run that completed after a ``warn`` is not a failed trace (run status is
  the root's), while a blocked run is.

Offline: the agent is a ``FunctionModel``; every rule is a local regex.
"""

from __future__ import annotations

import os
import sys

AGENT = "account-desk"
CARD = "4111-1111-1111-1111"
SSN = "123-45-6789"

REPLIES = {
    "card": "I can't help with that.",
    "ssn": f"The SSN on file is {SSN}. Mail it to ops@example.com.",
    "url": "See https://intranet.example.local/refunds for the refund form.",
    "ok": "Your order ships tomorrow.",
}

QUESTIONS = {
    "card": f"Charge my card {CARD} again please",
    "ssn": "What SSN do you have on file for me?",
    "url": "Where is the refund form?",
    "ok": "When does my order ship?",
}


def build_agent():  # type: ignore[no-untyped-def]
    from fastaiagent import Agent
    from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType
    from fastaiagent.testing import FunctionModel

    def responder(messages):  # type: ignore[no-untyped-def]
        question = next(str(m.content) for m in reversed(messages) if m.role == "user")
        for key, q in QUESTIONS.items():
            if question == q:
                return REPLIES[key]
        return REPLIES["ok"]

    def regex(name, position, pattern, action, **extra):  # type: ignore[no-untyped-def]
        blocking = extra.pop("blocking", True)
        return Guardrail(
            name=name,
            guardrail_type=GuardrailType.regex,
            position=position,
            config={"pattern": pattern, "should_match": False, **extra},
            action=action,
            blocking=blocking,
            severity="high" if action == "block" else "medium",
            description=f"{action} when the {position.value} matches {pattern}",
        )

    rails = [
        regex("no-card-numbers", GuardrailPosition.input, r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", "block"),
        regex(
            "mask-ssn",
            GuardrailPosition.output,
            r"\b\d{3}-\d{2}-\d{4}\b",
            "mask",
            mask_token="[SSN]",
        ),
        regex("warn-internal-url", GuardrailPosition.output, r"intranet\.", "warn"),
        regex("observe-emails", GuardrailPosition.output, r"[\w.]+@[\w.]+", "mask", blocking=False),
    ]
    return Agent(
        name=AGENT,
        system_prompt="You answer account questions.",
        llm=FunctionModel(responder, model="gpt-4o-mini"),
        guardrails=rails,
    )


def seed() -> int:
    from fastaiagent._internal.errors import GuardrailBlockedError

    assert not os.environ.get("FASTAIAGENT_UI_ENABLED"), "the point is that it is unset"
    agent = build_agent()
    for key, question in QUESTIONS.items():
        try:
            out = agent.run(question).output
            print(f"▸ {key:<4} → {out!r}")
        except GuardrailBlockedError as exc:
            print(f"▸ {key:<4} → BLOCKED: {exc}")
    return 0


# ---------------------------------------------------------------------------


def _event(cap, rule: str) -> dict:  # type: ignore[no-untyped-def]
    rows = cap.get(f"/api/guardrail-events?rule={rule}&hide_passes=true&page_size=50")["rows"]
    if not rows:
        raise SystemExit(f"no event for {rule}")
    return rows[0]


def drive(base: str) -> int:
    from _capture189 import Capture
    from playwright.sync_api import sync_playwright

    cap = Capture("capture-guardrail-truth", base)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = cap.new_page(browser)

        # -- 1. The list: recorded by default, effect per row, passes hidden -----
        print("▸ guardrail events list")
        fired = cap.get("/api/guardrail-events?hide_passes=true&page_size=100")
        everything = cap.get("/api/guardrail-events?hide_passes=false&page_size=100")
        cap.check(fired["total"] == 4, "4 checks did something", str(fired["total"]))
        cap.check(
            everything["total"] > fired["total"],
            "passes recorded too",
            f"{everything['total']} vs {fired['total']}",
        )
        cap.goto(page, "/guardrails")
        cap.text(page, "4 events (passes hidden)", "header: 4 events (passes hidden)")
        # The Action column is what the check did (action_taken); an
        # observe-only rule says it applied nothing.
        expect_rows = {
            "no-card-numbers": ("blocked", "blocked"),
            "mask-ssn": ("filtered", "masked"),
            "warn-internal-url": ("warned", "warned"),
            "observe-emails": ("warned", "masked · observe-only"),
        }
        for rule, (outcome, action) in expect_rows.items():
            row = page.locator("tr", has=page.get_by_role("link", name=rule, exact=True))
            cap.visible(row.get_by_text(outcome, exact=True), f"{rule}: outcome {outcome}")
            cap.visible(row.get_by_text(action, exact=True), f"{rule}: action {action}")
            cap.visible(row.get_by_text(AGENT, exact=True), f"{rule}: names agent {AGENT}")
        cap.shot(page, "189-guardrail-01-events-list")

        page.get_by_label("Hide passes").click()
        cap.settle(page)
        cap.text(
            page, f"{everything['total']} events", f"Hide passes off: {everything['total']} events"
        )
        cap.visible(page.get_by_text("passed", exact=True), "passes are listed once shown")
        cap.check("hide_passes=false" in page.url, "the filter is in the URL", page.url)

        # -- 2. Blocked input: judged text, rule snapshot, what happened ----------
        print("▸ blocked event")
        blocked = _event(cap, "no-card-numbers")
        cap.goto(page, f"/guardrail-events/{blocked['event_id']}", 1000)
        trigger = page.get_by_test_id("panel-trigger")
        cap.text(trigger, "exactly the text the rule judged")
        cap.text(trigger, CARD, "judged text holds the card number")
        cap.text(
            page.get_by_test_id("rule-action-chain"),
            "block → blocked → blocked",
            "action → taken → effect: block → blocked → blocked",
        )
        cap.text(page.get_by_test_id("panel-outcome"), "Run stopped here.")
        evidence = page.get_by_test_id("evidence-regex")
        cap.text(evidence, r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", "evidence: the pattern as it ran")
        cap.visible(page.get_by_label("Toggle guardrail event redaction"), "Mask secrets switch")
        cap.shot(page, "189-guardrail-02-blocked-event", full_page=True)

        # -- 3. Mask: the original text was judged, the rewrite is shown ----------
        print("▸ masked event")
        masked = _event(cap, "mask-ssn")
        cap.goto(page, f"/guardrail-events/{masked['event_id']}", 1000)
        cap.text(page.get_by_test_id("panel-trigger"), SSN, "judged text is the original reply")
        cap.text(
            page.get_by_test_id("panel-outcome"),
            "The guardrail rewrote the content before it continued.",
        )
        cap.text(page.get_by_test_id("panel-outcome"), "[SSN]", "after: the masked reply")
        cap.shot(page, "189-guardrail-03-masked-event", full_page=True)

        # -- 4. Observe-only: nothing applied -------------------------------------
        print("▸ observe-only event")
        observed = _event(cap, "observe-emails")
        cap.goto(page, f"/guardrail-events/{observed['event_id']}", 1000)
        cap.text(
            page.get_by_test_id("panel-outcome"),
            "Observe-only — would have masked; nothing applied.",
        )
        cap.text(page.get_by_test_id("panel-outcome"), "would have produced (not applied)")

        # -- 5. Run status is the root's ------------------------------------------
        print("▸ trace status")
        warned = _event(cap, "warn-internal-url")
        error_ids = {t["trace_id"] for t in cap.traces("status=ERROR")}
        cap.check(blocked["trace_id"] in error_ids, "the blocked run is an ERROR trace")
        cap.check(
            warned["trace_id"] not in error_ids,
            "the run that completed after a warn is not an ERROR trace",
        )
        cap.goto(page, f"/traces/{warned['trace_id']}", 1000)
        cap.visible(
            page.get_by_text("guardrail.warn-internal-url").first,
            "the warn trace shows the guardrail span",
        )
        cap.shot(page, "189-guardrail-04-warn-trace")
        cap.goto(page, "/traces?status=ERROR")
        cap.text(page, "1 trace", "Traces ?status=ERROR: only the blocked run")

        # -- 6. The agent's Guardrails tab -----------------------------------------
        print("▸ agent guardrails tab")
        cap.goto(page, f"/agents/{AGENT}", 1000)
        page.get_by_role("tab", name="Guardrails").click()
        cap.settle(page, 600)
        table = page.get_by_test_id("agent-guardrails")
        for rule in expect_rows:
            cap.visible(table.get_by_text(rule, exact=True), f"Guardrails tab lists {rule}")
        cap.shot(page, "189-guardrail-05-agent-guardrails-tab")

        browser.close()
    return cap.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        raise SystemExit(seed())
    if sys.argv[1:2] == ["drive"]:
        raise SystemExit(drive(sys.argv[2]))
    raise SystemExit("usage: capture_guardrail_truth.py seed | drive <base-url>")
