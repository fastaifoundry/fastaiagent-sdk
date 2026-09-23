# Managed governance (approval policies)

A **connected** agent can honor governance the platform admin defines centrally —
no policy code in the agent. When the agent is about to call a tool the admin has
flagged, the SDK asks the platform whether it may proceed; a high-stakes call
**pauses** and hands the decision to **your application**, which asks its own user
and resumes the run with the answer.

!!! info "Who approves: your application, not the console (since 1.74.0)"
    The plane never approves a runtime tool call. It distributes the policy,
    **records** every pause and every resolution, **reports** on them (pending
    count, age, outcome, who resolved it, time to resolve) and **flags** a pause
    that outlives its policy's timeout — it never expires or decides one. Your
    application is already talking to the person who started the run; the plane
    is not. The console's approvals pages are a read-only record of what your
    application decided.

This builds on [guardrails](index.md) (local, in-process checks) by adding the
**managed policy + pause/resume over the wire**. It needs [`fa.connect()`](../platform/index.md).
For connect-time enrollment and the opt-in **fail-closed** posture (refuse rather
than run ungoverned when the plane is unreachable), see
[Connected governance](../platform/connected-governance.md).

## How it works

```
connect() ──▶ GET /policy (cache)
   tool call ──▶ matches a cached approval policy?
        │ no  ─▶ run normally
        │ yes ─▶ POST /policy/decide
                    ├─ allow            ─▶ run
                    ├─ deny             ─▶ refuse (tell the model why)
                    └─ require_approval ─▶ POST /runs/{id}/pending + PAUSE (checkpoint)
                                              arun() returns status="paused"
                                              your app asks its user
                                              aresume(Resume(approved=…))
                                                 ├─ approved ─▶ run the tool
                                                 └─ rejected ─▶ refuse (tell the model)
```

1. **`connect()`** pulls and caches the policy (`GET /policy`). On a pull failure
   it keeps the last-known cache.
2. Before a tool call whose name matches a cached approval policy's `tool_pattern`,
   the SDK calls **`POST /policy/decide`**.
3. **`deny`** → the call is refused and the model is told why (the run continues).
4. **`require_approval`** → the SDK posts **`POST /runs/{id}/pending`** (so the
   plane can record the pause against its approval request) and pauses the agent
   (a real checkpoint). **`arun()` returns** the paused result to your code.
5. Your application resumes with **`aresume(...)`**. An approval runs the tool; a
   rejection **never runs it** — the model receives the refusal as the tool's
   result, exactly as for `deny`. The resolution is reported to the plane with the
   resolver you pass.

Only tools that match a configured policy incur a `/policy/decide` round-trip;
everything else runs untouched.

## Plane-authored guardrails (enforced at the edge)

The same `GET /policy` pull also carries **guardrails authored on the plane**. A
connected agent turns each distributed rule into a runtime guardrail and enforces
it *alongside* its local `guardrails=[...]` — at all four positions. An agent with
**no local guardrails** still blocks on a centrally-authored rule:

```python
import fastaiagent as fa
from fastaiagent import Agent, LLMClient

# An admin created a "block-ssn-output" regex guardrail in the console.
fa.connect(api_key="fa_k_...", target="https://app.fastaiagent.net")

agent = Agent(name="support", llm=LLMClient(provider="openai", model="gpt-4o"))
#                                    ^ no guardrails=[...] defined locally
agent.run("Confirm my record: name Dana, SSN 123-45-6789.")
# -> GuardrailBlockedError(guardrail_name="block-ssn-output")
```

- **No new check engine.** A rule is mapped onto the SDK's own
  `regex` / `schema` / `classifier` / `llm_judge` / `content_safety` /
  `groundedness` / `topic` / `pii` / `secrets` runners
  (`fastaiagent.guardrail.from_policy`), so plane rules
  enforce exactly like local ones — including the `on_error` fail policy. A
  `code` rule (a server-side callable the SDK doesn't have) is skipped rather
  than silently passing.
- **Positions are mapped, and a mismatch is logged.** The plane models a single
  `tool` phase; the SDK splits it into `tool_call` and `tool_result`, so a plane
  `tool` rule is enforced on the call. A `guardrail_type` the SDK does *not*
  recognise falls back to `output` and, **since 1.64.0, emits a `WARNING`**
  naming the rule and the unrecognised value. Watch for it: the rule still runs,
  but not where it was authored to run — a rule written to gate the user's
  prompt would be inspecting the model's reply instead, and the console would
  show a healthy control sitting over an ungated input.
- **The rule does what it says.** Each rule also carries an `action` — `block`,
  `warn`, `mask`, `override` or `reask` — so a "Mask PII in output" rule authored
  in the console **redacts** inside your process rather than blocking the run.
  It carries `severity` and `floor` too; neither changes enforcement. See
  [Actions, severity & floor](actions.md).
- **Scoping.** A rule attached to specific agents applies only to them; an
  unattached rule is domain-wide. Built guardrails are memoized by policy
  `version`, so an edit on the plane is picked up on the next pull.
- **Refresh.** `fa.refresh_policy()` re-pulls the policy mid-session to pick up a
  guardrail authored *after* `connect()`.
- **Local-only is unchanged.** With no connection there is no policy cache, so a
  local run enforces exactly its own `guardrails=[...]` and pays nothing.
- **Plane rules stay plane rules.** A reconstructed guardrail is marked
  `origin="plane"`, which keeps it out of the agent definition sent by
  `agent.push()` and stops it being enforced twice if you also pass it in
  `guardrails=[...]`. Both matter: the plane upserts pushed guardrails *by name*
  and attaches them to the pushing agent, so echoing a domain-wide rule back
  would narrow it to just that agent — silently dropping it for every other
  agent in the domain. You never need to pass plane rules in explicitly; the
  runtime injects them.

!!! warning "A `groundedness` rule needs its context"
    It is the one rule that reads a *pair*: an answer and the context the answer
    was supposed to use. An output guardrail only receives the answer, so supply
    the context with `fa.guardrail_context(context=docs)` around the run. Without
    it the rule **fails closed** — scoring an answer against nothing would block
    everything, which is worse than reporting that the rule could not run.

!!! tip "You don't need to pull them yourself"
    `plane_guardrails_for_agent(...)` exists for runtimes that aren't `fa.Agent`
    (see [Guardrails & evals without the runtime](../integrations/primitives-without-the-runtime.md)).
    A connected `fa.Agent` already enforces plane rules with no code at all.

Enforcement is always **local** — the runtime is the enforcement point (see
[Who actually blocks](index.md)). The plane *authors and distributes*; it never
reaches into a running agent.

## Enrolling an agent

Two things make an agent governable:

```python
import fastaiagent as fa
from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

fa.connect(api_key="fa_k_...", target="https://app.fastaiagent.net")

agent = Agent(
    name="banker",
    agent_id="<platform agent uuid>",        # (1) enroll: the id /policy/decide matches on
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
    checkpointer=SQLiteCheckpointer("agent.db"),  # (2) needed to pause/resume
)
```

- **`agent_id`** is the agent's **platform UUID** — it's sent to `/policy/decide`
  so the plane can match approval policies (and it validates the id). Without it,
  the agent isn't enrolled and the gate is a no-op. `agent.push()` registers the
  agent and sets it for you. An enrolled agent stays governed inside a `Swarm` or
  a `Supervisor` (before 1.74.0 their internal copies dropped the id, so a
  worker's policy-gated tool ran with no approval at all).
- A **`checkpointer`** is required so a paused run can be resumed. Without one,
  the pause cannot be saved and surfaces as an `InterruptSignal` exception (unless
  the agent runs inside a checkpointed `Chain`, which then owns the pause).

### Approving from your application

`arun()` returns the pause. Read the tool and its arguments from the pause, ask
your user, and resume with their answer and their identity:

```python
from fastaiagent import Resume

res = await agent.arun("Transfer $500 to Bob.", execution_id="run-42")
if res.status == "paused" and res.pending_interrupt["reason"] == "policy_approval_required":
    ctx = res.pending_interrupt["context"]
    tool, args = ctx["tool"], ctx["tool_input"]      # "transfer_funds", {"amount": 500, "to": "Bob"}

    approved = ask_my_user(f"Allow {tool} with {args}?")   # your UI, chat reply, ticket...

    res = await agent.aresume(
        res.execution_id,
        resume_value=Resume(approved=approved, metadata={"resolver": "alice@acme.com"}),
    )
```

- **The pause** (`res.pending_interrupt["context"]`) carries `tool`, `tool_input`
  (the arguments the model chose), `run_id` and the plane's `approval_request_id`.
- **The resume can happen later and elsewhere.** The run is checkpointed, so
  `aresume` works from another request or process that can reach the same
  checkpointer (or, when connected, the plane's replica). Keep the
  `execution_id`.
- **`approved=False` refuses the call.** The tool never runs; the model receives
  `Refused: governance approval denied for 'transfer_funds'` as the tool's result
  and continues, exactly as for a `deny` verdict.
- **Pass the resolver.** `Resume.metadata["resolver"]` is recorded on the plane's
  human-in-the-loop ledger as *who* decided — the evidence for human oversight
  (EU AI Act Article 14). Pass the identity of the person who answered. It is the
  only place that identity comes from; without it the ledger records the decision
  with no one attached.

!!! warning "`wait_for_approval=True` is deprecated"
    Before 1.74.0 `arun()` blocked by default, polling `GET /runs/{id}/pending` for
    a console decision. The plane no longer makes that decision, so the wait can only
    end in the deprecated console approve/deny — kept working for the plane's
    transition window — or at its 600 s ceiling. Passing `wait_for_approval=True`
    still works and emits a `DeprecationWarning`. **A rejection or an expired wait
    refuses the call**; it never runs the tool.

A runnable end-to-end example is in `examples/84_governed_agent.py`.

## Verified end-to-end

**Guardrail actions** (SDK 1.57.0, wire v1.9) are exercised against a live local
plane by `tests/e2e/test_connected_guardrail_actions_e2e.py`: a console-authored
"Mask PII in output" rule arrives over `/policy`, redacts a real agent reply
in-process, and lands a `filtered` event with a before/after diff — plus a
`content_safety` block naming the category that tripped, a `groundedness` rule
passing with context and failing closed without it, and `floor` surviving the
wire. The judges themselves run against a real model in
`tests/e2e/test_guardrail_actions_e2e.py`.

**Approvals** are exercised against a live local plane by
`tests/e2e/test_connected_approvals_e2e.py` (real `gpt-4o-mini`): a console-authored
approval policy pauses the run, the application rejects it, the tool **never runs**,
and the plane's ledger records `kind=approval`, `rejected` and the resolver the
application passed. The same paths — approve, reject, the deprecated wait expiring,
and the pause inside a `Chain`, `Swarm` and `Supervisor` — are pinned without a
plane in `tests/test_governance_approvals.py`.

## On the platform

The admin manages the rules on the **Approval Policies** page (the `tool_pattern`
set, and a `timeout_minutes` the plane uses to flag a pause as *overdue* — never to
expire or decide it). The approvals pages are a **read-only record**: every pause,
its outcome, who resolved it and how long it took, as reported by your
application through the SDK.

Each paused run is also a normal trace — the connected agent's run pushed to the
platform (here `agent.banker`):

![The connected agent's run trace](img/governance-trace.png)
