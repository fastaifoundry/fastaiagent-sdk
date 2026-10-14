"""1.89.0 Phase 9 (Playground, prompts, approvals, KB, traces) — seed, drive, assert.

Called by ``scripts/capture-playground-fixes-screenshots.sh``:

    python scripts/capture_playground_fixes.py seed
    python scripts/capture_playground_fixes.py drive <base-url>

What it proves, on screen:

* Playground: a failed call says why (kind, HTTP status, the provider's
  message with the key scrubbed, a hint); sampling rules for a typed model;
  a provider that needs an endpoint waits for one; unfilled variables are
  named before the run (Run anyway); a reply cut off at max_tokens says so;
  the version picker lists aliases and a run is attributed (or "modified");
  history restores a run;
* Prompts: alias chips, the Aliases card and the Compare versions panel;
* Traces: Playground and Retrieval badges, the Note card and the Noted filter;
* KB: a keyword-only KB offers keyword search only and says which type ran;
* Approvals: resolving an approval in the UI shows the Outcome card (decision,
  reason, final output, each run's trace) and the approval reads as resolved.

Offline: the Playground talks to a localhost SSE server (``tests/_sse_provider``)
scripted per run; the approval agent is a ``FunctionModel``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROMPT = "support-reply"
V1 = "Hi {{customer}}, thanks for writing about {{topic}}."
V2 = "Hello {{customer}}! About {{topic}}: we are on it and will reply within a day."
EXECUTION = "refund-ORD-77"
REASON = "Verified with the customer; within the refund policy."
LEAKED_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcd"

AGENT_APP = '''
from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, interrupt
from fastaiagent.testing import FunctionModel


def issue_refund(order_id: str, amount: int) -> str:
    """Refund an order. Amounts over 500 need a manager's approval."""
    if amount > 500:
        decision = interrupt(
            reason="manager_approval", context={"order_id": order_id, "amount": amount}
        )
        if not decision.approved:
            return f"refund for {order_id} rejected by a manager"
    return f"refunded {amount} on {order_id}"


def responder(messages):
    tool_results = [m for m in messages if str(getattr(m, "role", "")).endswith("tool")]
    if tool_results:
        return f"Update: {tool_results[-1].content}."
    return ("", [{"name": "issue_refund", "arguments": {"order_id": "ORD-77", "amount": 900}}])


agent = Agent(
    name="refund-approvals",
    system_prompt="You run the refund desk.",
    llm=FunctionModel(responder, model="gpt-4o-mini"),
    tools=[FunctionTool(name="issue_refund", fn=issue_refund)],
    checkpointer=SQLiteCheckpointer(),
)
'''


def seed() -> int:
    from fastaiagent.kb.local import LocalKB
    from fastaiagent.prompt.registry import PromptRegistry

    reg = PromptRegistry(path=os.environ["FASTAIAGENT_LOCAL_DB"])
    reg.register(PROMPT, V1, metadata={"owner": "support-team"})
    reg.register(PROMPT, V2, metadata={"owner": "support-team"})
    reg.set_alias(PROMPT, 1, "production")
    reg.set_alias(PROMPT, 2, "staging")
    print(f"▸ prompt {PROMPT}: v1 (production), v2 (staging)")

    kb = LocalKB(name="policies", path=os.environ["FASTAIAGENT_KB_DIR"], search_type="keyword")
    kb.add(
        "Refunds are processed within 7 business days of receiving the return.",
        metadata={"source": "refund-policy.md"},
    )
    kb.add(
        "Sale items are final sale and cannot be returned or refunded.",
        metadata={"source": "sale-policy.md"},
    )
    kb.add("Standard shipping is free on orders over $50.", metadata={"source": "shipping.md"})
    print("▸ keyword-only KB 'policies' with 3 documents")

    Path("agent_app.py").write_text(AGENT_APP)
    sys.path.insert(0, str(Path.cwd()))
    from agent_app import agent  # type: ignore[import-not-found]

    result = agent.run("Refund 900 on ORD-77, please", execution_id=EXECUTION)
    print(
        f"▸ approval run {EXECUTION}: status={result.status} "
        f"pending={result.pending_interrupt and result.pending_interrupt['reason']}"
    )
    if result.status != "paused":
        raise SystemExit("expected the refund to pause for approval")
    return 0


# ---------------------------------------------------------------------------


def drive(base: str) -> int:
    from _capture189 import Capture, pick_provider
    from playwright.sync_api import Page, sync_playwright

    from tests._sse_provider import (
        Reply,
        SSEProvider,
        openai_chunk,
        openai_stream,
        openai_usage_chunk,
    )

    cap = Capture("capture-playground-fixes", base)
    stub = SSEProvider().start()

    def connect(page: Page) -> None:
        page.get_by_role("button", name=re.compile(r"^Connection")).click()
        page.locator("#playground-base-url").fill(f"{stub.url}/v1")
        page.locator("#playground-api-key").fill("stub-token")

    def run(page: Page) -> None:
        page.get_by_role("button", name="Run", exact=True).click()

    def to_response(page: Page) -> None:
        page.get_by_text("// RESPONSE", exact=True).scroll_into_view_if_needed()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = cap.new_page(browser)

        # -- 1. A failed call says why --------------------------------------------
        print("▸ playground errors")
        cap.goto(page, "/playground")
        pick_provider(page, "openai")
        page.locator("#playground-model").fill("gpt-4o-mini")
        page.locator("#playground-template").fill("Say hello to {{name}}.")
        page.locator("#var-name").fill("Ada")
        connect(page)
        stub.script(
            Reply(
                status=401,
                body={
                    "error": {
                        "message": f"Incorrect API key provided: {LEAKED_KEY}. "
                        "You can find your API key "
                        "at https://platform.openai.com/account/api-keys.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        )
        run(page)
        err = page.get_by_test_id("run-error")
        cap.visible(err, "auth failure shows the error panel", timeout=20_000)
        cap.text(err, "Authentication failed (HTTP 401)")
        cap.text(err, "Incorrect API key provided", "the provider's own message is shown")
        cap.text(err, "[REDACTED]", "… with the key scrubbed")
        cap.absent(err, LEAKED_KEY, "the echoed key is not on the page")
        cap.text(err, "The provider rejected the credential for 'openai'.", "auth hint")
        cap.text(err, "Server log correlation id")
        to_response(page)
        cap.shot(page, "189-playground-01-auth-error")

        stub.script(
            Reply(
                status=404,
                body={
                    "error": {
                        "message": "The model `gpt-4o-mini-typo` does not exist "
                        "or you do not have access to it.",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        )
        page.locator("#playground-model").fill("gpt-4o-mini-typo")
        run(page)
        cap.text(err, "Model not found (HTTP 404)", timeout=20_000)
        cap.text(err, "The provider does not know model 'gpt-4o-mini-typo'", "not-found hint")

        # -- 2. Sampling rules follow the typed model --------------------------------
        print("▸ sampling rules")
        page.locator("#playground-model").fill("gpt-5-mini")
        cap.text(page, "(limited for this model)", "Parameters says the model is limited")
        page.get_by_role("button", name=re.compile(r"^Parameters")).click()
        cap.visible(
            page.get_by_test_id("param-temperature-rule"), "temperature: accepts only the default"
        )
        cap.text(
            page.get_by_test_id("param-temperature-rule"),
            "This model accepts only the default temperature (1).",
        )
        page.get_by_role("button", name=re.compile(r"^Parameters")).click()

        # -- 3. A provider that needs an endpoint waits for one ----------------------
        print("▸ custom provider needs an endpoint")
        page.locator("#playground-provider").click()
        cap.visible(
            page.get_by_role("option", name=re.compile(r"^custom.*needs endpoint")),
            "custom is marked (needs endpoint)",
        )
        page.get_by_role("option", name=re.compile(r"^custom")).click()
        page.locator("#playground-base-url").fill("")
        page.locator("#playground-model").fill("my-gateway-model")
        cap.text(
            page.get_by_test_id("run-blocked-reason"),
            "custom needs an Endpoint (Connection panel).",
        )
        cap.disabled(
            page.get_by_role("button", name="Run", exact=True), "Run waits for an endpoint"
        )
        cap.text(page, "custom has no default endpoint. Enter its base URL.")
        cap.shot(page, "189-playground-02-custom-needs-endpoint")
        page.locator("#playground-base-url").fill(f"{stub.url}/v1")
        page.locator("#playground-api-key").fill("stub-token")
        cap.enabled(
            page.get_by_role("button", name="Run", exact=True), "… and runs once it has one"
        )

        # -- 4. Prompt aliases, attribution, unfilled variables, truncation ----------
        print("▸ aliases, attribution, unfilled variables, truncation")
        cap.goto(page, f"/playground?prompt={PROMPT}&alias=production")
        cap.text(
            page.locator("#playground-version"),
            "production → v1",
            "?alias=production opens v1 through its alias",
        )
        page.locator("#playground-version").click()
        cap.visible(page.get_by_role("option", name="staging → v2"), "the picker lists aliases")
        page.keyboard.press("Escape")
        cap.check(page.locator("#playground-template").input_value() == V1, "template is v1")
        pick_provider(page, "openai")
        page.locator("#playground-model").fill("gpt-4o-mini")
        connect(page)
        page.locator("#var-customer").fill("Ada")
        run(page)
        warn = page.get_by_test_id("unfilled-warning")
        cap.text(warn, "Not filled: {{topic}}.", "unfilled {{topic}} is named before the run")
        cap.shot(page, "189-playground-03-unfilled-warning")
        stub.script(Reply(events=openai_stream("Hi Ada, thanks for writing", " about {{topic}}.")))
        warn.get_by_role("button", name="Run anyway").click()
        cap.text(
            page.get_by_test_id("sent-unfilled"),
            "Sent with unfilled variables: {{topic}}",
            timeout=20_000,
        )
        cap.text(page.get_by_test_id("prompt-attribution"), "attributed to support-reply v1")
        cap.visible(
            page.get_by_role("link", name="trace", exact=True),
            "the run links to its trace (it is in the UI's store)",
        )

        page.locator("#var-topic").fill("a late refund")
        page.locator("#playground-template").fill(V1 + " Keep it short.")
        page.get_by_role("button", name=re.compile(r"^Parameters")).click()
        page.locator("#param-max-tokens").fill("12")
        stub.script(
            Reply(
                events=[
                    (None, openai_chunk("Hi Ada, thanks for writing about a late refund. We")),
                    (None, openai_chunk(finish="length")),
                    (None, openai_usage_chunk(30, 12)),
                    (None, "[DONE]"),
                ]
            )
        )
        run(page)
        cap.text(
            page.get_by_test_id("truncated-banner"),
            "Cut off at max_tokens (12). Raise max_tokens to get the rest.",
            timeout=20_000,
        )
        cap.text(
            page.get_by_test_id("prompt-attribution"),
            "modified from support-reply",
            "an edited template is marked modified, not attributed",
        )
        cap.visible(
            page.get_by_test_id("history-truncated"), "history entry carries a truncated chip"
        )
        to_response(page)
        cap.shot(page, "189-playground-04-truncated-modified")

        history = page.locator(
            'li button[title="Restore this run\'s prompt, model and parameters"]'
        )
        cap.check(history.count() == 2, "two runs in history", str(history.count()))
        history.nth(1).click()
        cap.settle(page, 400)
        cap.check(
            page.locator("#playground-template").input_value() == V1,
            "selecting the older run restores its template",
        )

        # -- 5. Prompt editor: aliases and compare ------------------------------------
        print("▸ prompt editor aliases and compare")
        cap.goto(page, f"/prompts/{PROMPT}")
        chips = page.get_by_test_id("version-alias-chip")
        cap.visible(chips.filter(has_text="production"), "v1 carries the production chip")
        cap.visible(chips.filter(has_text="staging"), "v2 carries the staging chip")
        aliases = page.get_by_test_id("alias-list")
        cap.text(aliases, "production → v1")
        cap.text(aliases, "staging → v2")
        page.get_by_role("button", name="Compare versions").click()
        diff = page.get_by_test_id("prompt-diff")
        cap.visible(diff, "Compare versions panel shows the unified diff (v1 → v2)")
        cap.text(diff, re.compile(r"^-\s*Hi \{\{customer\}\}"), "diff: v1 line removed")
        cap.text(diff, re.compile(r"^\+\s*Hello \{\{customer\}\}"), "diff: v2 line added")
        cap.visible(page.get_by_test_id("prompt-side-1"), "both stored templates side by side")
        diff.scroll_into_view_if_needed()
        cap.shot(page, "189-playground-05-prompt-aliases-compare")
        cap.goto(page, f"/prompts/{PROMPT}?alias=staging")
        cap.text(page, "(editing v2)", "?alias=staging opens v2")

        # -- 6. Traces: badges, note, Noted filter -------------------------------------
        print("▸ traces: badges and notes")
        cap.goto(page, "/kb/policies")
        page.get_by_role("tab", name="Search playground").click()
        cap.text(
            page.get_by_test_id("kb-keyword-only"),
            "No chunk in this KB has an embedding, so only keyword search is available.",
        )
        options = page.locator("#search_type option").all_inner_texts()
        cap.check(options == ["keyword"], "search type offers keyword only", str(options))
        page.locator("#query").fill("refund")
        page.get_by_role("button", name="Run", exact=True).click()
        cap.text(
            page,
            re.compile(r"\d+ results? · search_type = keyword"),
            "the result says keyword search ran",
            timeout=20_000,
        )
        cap.shot(page, "189-playground-06-kb-keyword-search")

        cap.goto(page, "/traces")
        cap.visible(
            page.locator("tr", has_text="playground.run").get_by_text("playground").first,
            "Playground runs carry the playground runner badge",
        )
        kb_row = page.locator(
            "tr", has=page.get_by_test_id("trace-source-badge").filter(has_text="kb_search")
        )
        cap.visible(kb_row, "the KB search is a trace badged kb_search")
        cap.visible(
            kb_row.get_by_text("retrieval", exact=True), "… with the retrieval runner badge"
        )
        cap.shot(page, "189-playground-07-traces-badges")

        attributed = next(t for t in cap.traces() if t["name"] == "playground.run")
        cap.goto(page, f"/traces/{attributed['trace_id']}", 1000)
        card = page.get_by_test_id("trace-notes-card")
        cap.visible(card, "trace detail has a Note card")
        card.get_by_label("Trace note").fill("Reply ignored the topic: keep as an eval case.")
        cap.text(card.get_by_test_id("note-counter"), "46 / 20,000")
        card.get_by_role("button", name="Save note").click()
        cap.text(page, "Note saved", "toast: Note saved")
        cap.goto(page, "/traces?noted=true")
        cap.text(page, "1 trace", "Noted filter: 1 trace")
        cap.visible(page.get_by_test_id("trace-noted"), "the noted trace shows the note icon")
        cap.shot(page, "189-playground-08-noted-filter")

        # -- 7. Approvals: resolve in the UI, see the outcome ---------------------------
        print("▸ approvals and the execution outcome")
        cap.goto(page, f"/approvals/{EXECUTION}", 1000)
        cap.visible(page.get_by_text("Decision", exact=True), "the approval waits for a decision")
        page.locator("textarea").fill(REASON)
        page.get_by_role("button", name="Approve").click()
        cap.text(page.get_by_test_id("resumed-banner"), "Resumed (approved).", timeout=20_000)
        outcome = page.get_by_test_id("execution-outcome")
        cap.text(outcome.get_by_test_id("execution-resolution"), "Approved")
        cap.text(outcome, f"Reason: {REASON}", "Outcome card keeps the reason")
        cap.text(
            outcome.get_by_test_id("execution-final-output"),
            "Update: refunded 900 on ORD-77.",
            "Outcome card shows the final output",
        )
        runs = outcome.get_by_test_id("execution-runs").locator("li")
        cap.check(
            runs.count() == 2, "Outcome card lists both runs (paused + resume)", str(runs.count())
        )
        cap.visible(page.get_by_test_id("interrupt-resolved"), "the timeline says resolved")
        page.get_by_test_id("execution-outcome").scroll_into_view_if_needed()
        cap.shot(page, "189-playground-09-execution-outcome")

        cap.goto(page, f"/approvals/{EXECUTION}", 1000)
        resolved = page.get_by_test_id("approval-resolved")
        cap.text(resolved, "Approved", "the approval reads as resolved")
        cap.text(resolved, f"Reason: {REASON}")
        cap.text(page, "This approval has been resolved.")

        browser.close()
    stub.stop()
    return cap.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        raise SystemExit(seed())
    if sys.argv[1:2] == ["drive"]:
        raise SystemExit(drive(sys.argv[2]))
    raise SystemExit("usage: capture_playground_fixes.py seed | drive <base-url>")
