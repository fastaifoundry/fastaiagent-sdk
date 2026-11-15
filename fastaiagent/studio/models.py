"""Typed answers from the Agent Studio API (plane 0.20.0, wire v1.12).

Shapes follow the plane's ``docs/Agent_Studio_API.md``. Every model keeps the keys it
does not know (``extra="allow"``): the plane adds keys to a payload without a new
wire version, and a client must not fail on one. Read an unknown key with
``model.model_extra``.

The spec itself — what a coding tool sends to make a version — is a plain ``dict``
here. The plane owns its rules and checks it (``Studio.validate_spec``); the JSON
Schema it publishes is ``Studio.spec_schema().json_schema``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _null_is_absent(cls, data: Any) -> Any:
        """The plane sends ``null`` for a value it does not have (``updated_at`` on a new
        draft, ``note`` on a first version). Read it as absent, so the field's own default
        applies; a field that defaults to ``None`` reads exactly as before."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


# -- context and schema ----------------------------------------------------------------


class ProjectRef(_Model):
    id: str
    name: str = ""
    domain_id: str = ""


class Environment(_Model):
    tier: str = ""
    #: ``False`` on a production or acceptance install: every write answers
    #: ``environment_rule``. Reads (including the export) still work.
    studio_writes_allowed: bool = True


class LLMEndpoint(_Model):
    id: str
    name: str = ""
    provider: str = ""
    model: str = ""


class KnowledgeBaseRef(_Model):
    id: str
    name: str = ""


class SDKContract(_Model):
    """What the generated code targets — separate from the SDK that carries the commands."""

    sdk_pin: str = ""
    renderer_version: int = 0
    dependency_range: str = ""


class StudioContext(_Model):
    """``GET /context`` — what the project has. A caller reads this first."""

    project: ProjectRef
    environment: Environment = Field(default_factory=Environment)
    llm_endpoints: list[LLMEndpoint] = Field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseRef] = Field(default_factory=list)
    sdk: SDKContract = Field(default_factory=SDKContract)
    limits: dict[str, int] = Field(default_factory=dict)


class GuardrailFactory(_Model):
    factory: str
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class SpecSchema(_Model):
    """``GET /spec-schema`` — the JSON Schema of the spec, and what a schema cannot say."""

    spec_version: int = 1
    renderer_version: int = 0
    #: The JSON Schema document (the plane's key is ``schema``).
    json_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")
    guardrails: list[GuardrailFactory] = Field(default_factory=list)
    guardrails_that_warn: dict[str, str] = Field(default_factory=dict)
    guardrails_on_the_question_in_a_swarm: dict[str, str] = Field(default_factory=dict)
    tool_types: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    eval_criteria_keys: list[str] = Field(default_factory=list)
    limits: dict[str, int] = Field(default_factory=dict)


# -- drafts ----------------------------------------------------------------------------


class Step(_Model):
    key: str
    state: str = ""
    summary: str = ""


class NextAction(_Model):
    """``action`` is one of ``push_spec``, ``wait``, ``start_sandbox``, ``trial``,
    ``eval``, ``export``, ``none``."""

    action: str = "none"
    reason: str = ""
    hint: str = ""


class StepState(_Model):
    """Where a draft is and what comes next — the plane derives it, never stores it."""

    current: str = ""
    #: ``True`` when a person changed the draft's settings in the console after the
    #: current version was made. A coding tool that sees it stops and asks.
    stale: bool = False
    locked_until_built: bool = False
    building: dict[str, Any] | None = None
    steps: list[Step] = Field(default_factory=list)
    next: NextAction = Field(default_factory=NextAction)


class StepsSummary(_Model):
    current: str = ""
    stale: bool = False
    built: bool = False
    evals: dict[str, int] | None = None


class VersionSummary(_Model):
    id: str
    version: int = 0
    #: ``generate`` / ``refine`` (console) or ``spec_edit`` (this API, always).
    created_from: str = ""
    created_via: str = ""
    note: str | None = None
    #: ``pending`` or ``running`` while it is being made; ``completed`` or ``failed``.
    generation_status: str = ""
    error_message: str | None = None
    render_hash: str | None = None
    created_at: str = ""
    #: Seconds to wait before reading again; ``0`` means stop polling.
    retry_after_seconds: float | None = None


class Version(VersionSummary):
    """A version with its spec; ``rendered_files`` only when asked for (``include_files``)."""

    spec: dict[str, Any] | None = None
    rendered_files: dict[str, str] | None = None


class Fixture(_Model):
    id: str
    name: str = ""
    #: ``csv``, ``json`` or ``doc``.
    kind: str = ""
    source_filename: str | None = None
    rows: list[dict[str, Any]] | None = None
    raw_text: str | None = None
    raw_text_truncated: bool = False
    char_count: int = 0


class Draft(_Model):
    id: str
    project_id: str = ""
    domain_id: str = ""
    name: str = ""
    requirements_text: str = ""
    llm_endpoint_id: str | None = None
    #: ``drafting``, ``generating``, ``ready``, ``failed`` or ``exported``.
    status: str = ""
    #: The shape of the current version: ``single``, ``supervisor`` or ``swarm``.
    architecture: str | None = None
    harness: bool = False
    prompt_slug: str | None = None
    worker_prompt_slugs: dict[str, str] = Field(default_factory=dict)
    eval_dataset_id: str | None = None
    current_version_id: str | None = None
    created_via: str = ""
    console_url: str | None = None
    created_at: str = ""
    updated_at: str = ""
    steps_summary: StepsSummary | None = None


class DraftDetail(Draft):
    steps: StepState | None = None
    versions: list[VersionSummary] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)


# -- checking and making a version -----------------------------------------------------


class SpecProblem(_Model):
    """A rule the spec breaks. ``where`` names the field, e.g. ``workers[1].tools[0]``."""

    where: str
    message: str


class SpecWarning(_Model):
    """Does not stop a version being made. Read every one out to the person.

    ``code`` is ``guardrail_blocks_output``, ``guardrail_checks_question``,
    ``eval_value_unusable`` or ``eval_dataset_not_updated``.
    """

    where: str
    code: str = ""
    message: str


class SpecCheck(_Model):
    """``POST /spec/validate`` — problems and warnings; nothing is made."""

    ok: bool
    problems: list[SpecProblem] = Field(default_factory=list)
    warnings: list[SpecWarning] = Field(default_factory=list)


class VersionAccepted(_Model):
    """``POST /versions`` — the version is being made; poll it with ``wait_for_version``."""

    version_id: str
    version: int
    warnings: list[SpecWarning] = Field(default_factory=list)
    retry_after_seconds: float | None = None


class SpecChange(_Model):
    """A place in the spec, e.g. ``workers.escalation.system_prompt`` or ``tools.escalate``."""

    where: str
    #: ``added``, ``removed`` or ``changed``.
    kind: str


class FileChange(_Model):
    """A generated file, e.g. ``agent.py`` or ``evals/escalation.jsonl``."""

    path: str
    #: ``added``, ``removed`` or ``changed``.
    kind: str


class VersionDiff(_Model):
    """What changed between two versions: places in the spec, and rendered files."""

    from_version: int = Field(alias="from")
    to_version: int = Field(alias="to")
    spec_changes: list[SpecChange] = Field(default_factory=list)
    files: list[FileChange] = Field(default_factory=list)


# -- trying and evaluating -------------------------------------------------------------


class Sandbox(_Model):
    id: str = ""
    project_id: str = ""
    #: ``requested``, ``claimed``, ``ready`` or ``stopped``.
    status: str = ""
    runner_status: str | None = None
    last_trial_at: str | None = None
    retry_after_seconds: float | None = None


class MockCall(_Model):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    #: Whether a sample-data row (or a knowledge-base passage) came back.
    matched: bool = False


class CaseResult(_Model):
    name: str
    input: str = ""
    output: str | None = None
    #: ``None`` means a person has to review it (a ``behaviour`` case the word rules
    #: cannot score).
    passed: bool | None = None
    detail: str | None = None
    trace_id: str | None = None


class Trial(_Model):
    """A chat trial (``kind="chat"``) or an eval run (``kind="eval_preview"``)."""

    id: str
    draft_id: str = ""
    version_id: str = ""
    kind: str = ""
    worker: str | None = None
    input: str = ""
    #: ``queued``/``running``/``scoring`` while it takes time; ``completed`` or ``failed``.
    status: str = ""
    result: Any = None
    mock_calls: list[MockCall] = Field(default_factory=list)
    eval_results: list[CaseResult] | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    error: str | None = None
    created_at: str = ""
    retry_after_seconds: float | None = None

    @property
    def output(self) -> str | None:
        """The agent's answer for a chat trial, whatever shape ``result`` came in."""
        if isinstance(self.result, dict):
            value = self.result.get("output")
            return None if value is None else str(value)
        return None if self.result is None else str(self.result)


class EvalRuns(_Model):
    """``POST /evals/run`` — one run per scope; wait for each with ``wait_for_trial``."""

    runs: list[Trial] = Field(default_factory=list)


class EvalCase(_Model):
    name: str
    input: str = ""
    #: ``passed``, ``failed``, ``needs_review``, ``not_run`` or ``on_export``.
    status: str = ""
    expected_output: str | None = None
    criteria: dict[str, Any] | None = None
    output: str | None = None
    detail: str | None = None
    trace_id: str | None = None
    trial_id: str | None = None


class EvalGroup(_Model):
    #: A worker's name; ``"agent"`` for a single agent; ``"end_to_end"`` for a team's own
    #: cases, which run only from the exported code.
    scope: str
    on_export_only: bool = False
    cases: list[EvalCase] = Field(default_factory=list)


class EvalTotals(_Model):
    passed: int = 0
    failed: int = 0
    needs_review: int = 0
    not_run: int = 0
    runnable_here: int = 0
    on_export_only: int = 0


class EvalRollup(_Model):
    """``GET /evals`` — results for ONE version; an earlier version's pass is not carried."""

    version_id: str = ""
    version: int = 0
    totals: EvalTotals = Field(default_factory=EvalTotals)
    groups: list[EvalGroup] = Field(default_factory=list)


class DatasetLink(_Model):
    dataset_id: str


class Promoted(_Model):
    suite_id: str = ""
    dataset_id: str = ""
    cases: int = 0
