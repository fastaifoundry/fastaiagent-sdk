"""``fastaiagent studio`` — headless Agent Studio from a terminal or a coding tool.

A coding tool designs the agent with its own model and writes the design as a spec.
These commands send it to Studio, which checks it, renders the code, runs trials and
evals in its sandbox, keeps every version and exports. Nothing here calls Studio's
model: there is no ``plan``, ``build``, ``refine`` or ``reset`` (planning and building
from a description are console-only).

Conventions, the same as ``fastaiagent push`` and ``fastaiagent skills``:

* credentials are resolved as ``push`` does: ``--api-key`` / ``--target``, else
  ``FASTAIAGENT_API_KEY`` / ``FASTAIAGENT_TARGET``, else what ``fastaiagent connect`` saved;
* the answer goes to stdout — plain text, or with ``--json`` the plane's answer as JSON;
  notes and errors go to stderr;
* exit 0 means it happened; exit 1 means it did not (a refusal, a spec with problems, an
  eval that failed); exit 2 is a usage error;
* with ``--json`` a failure is also printed on stdout as ``{"error": {"code", "message",
  "hint", "status", "details"}}``, so a tool can read it without parsing stderr;
* nothing prompts unless a terminal is attached.

``<draft>`` is a draft id or its name.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    PlatformTierLimitError,
    StudioError,
)
from fastaiagent.studio import StudioClient
from fastaiagent.studio.models import Draft, EvalRollup, Trial, VersionSummary

studio_app = typer.Typer(
    name="studio",
    help="Build an agent through Agent Studio, headless: send a spec, try it, eval it, export it.",
    no_args_is_help=True,
)
spec_app = typer.Typer(
    name="spec",
    help="The design of an agent: its schema, check it, send it, read it back.",
    no_args_is_help=True,
)
studio_app.add_typer(spec_app, name="spec")

#: What to do about a refusal. Shown in the human output and as ``error.hint`` in JSON.
_HINTS: dict[str, str] = {
    "insufficient_scope": "Add the permission to the API key; the 'Agent Studio' preset has both "
    "studio:read and studio:write.",
    "feature_not_available": "This plan does not include Agent Studio.",
    "environment_rule": "Studio is off on this install (production rules). Do not retry.",
    "key_has_no_owner": "The person who made this key is gone. Make a new key.",
    "project_required": "This key is not tied to a project. Pass --project-id.",
    "name_taken": "Choose another name, or use the existing draft.",
    "no_llm_endpoint": "The project has no LLM endpoint. Add one in the console; nothing can be "
    "made without it.",
    "build_in_progress": "A version is still being made. Wait for it, then retry.",
    "no_version": "The draft has no completed version yet. Push a spec first.",
    "worker_required": "A team is tried one worker at a time. Name one with --worker.",
    "no_text_layer": "That file has no readable text (a scan). Read it yourself and pass the "
    "text with --data-text name=file.",
    "topology_promote_unsupported": "A team's cases cannot be promoted. Export instead.",
    "not_sdk_owned": "Run the exported code once with a key, so the agent registers, then promote.",
    "wait_timeout": "It is still going. Read the draft again with `studio status`.",
    "studio_not_available": "Use a plane at 0.20.0 or later.",
}

#: What an error says about itself, by class, when the plane gave no code.
_PLATFORM_CODES: tuple[tuple[type[PlatformError], str], ...] = (
    (PlatformAuthError, "auth_error"),
    (PlatformRateLimitError, "rate_limited"),
    (PlatformConnectionError, "connection_error"),
    (PlatformNotFoundError, "not_found"),
    (PlatformTierLimitError, "tier_limit"),
)


class _CliError(Exception):
    """A problem with how the command was used or what it was given (not the plane's)."""


# -- shared options and plumbing -----------------------------------------------------------


def _opt_json() -> Any:
    return typer.Option(False, "--json", help="Print the answer as JSON on stdout.")


def _opt_key() -> Any:
    return typer.Option(None, "--api-key", help="Override the saved API key.")


def _opt_target() -> Any:
    return typer.Option(None, "--target", help="Override the saved platform address.")


def _opt_project() -> Any:
    return typer.Option(
        None,
        "--project-id",
        help="Project id; only for a key that is not tied to one project.",
    )


def _client(api_key: str | None, target: str | None, project_id: str | None) -> StudioClient:
    from fastaiagent.cli.auth import _load_saved_credentials

    creds = _load_saved_credentials()
    key = api_key or creds.get("api_key")
    if not key:
        raise _CliError(
            "No API key. Run `fastaiagent connect` first, set FASTAIAGENT_API_KEY, "
            "or pass --api-key."
        )
    return StudioClient(api_key=key, target=target or creds.get("target"), project=project_id)


def _out(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _note(text: str) -> None:
    typer.echo(text, err=True)


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def _fail(
    as_json: bool,
    code: str,
    message: str,
    *,
    status: int = 0,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    hint = _HINTS.get(code)
    if as_json:
        error: dict[str, Any] = {"code": code, "message": message}
        if hint:
            error["hint"] = hint
        if status:
            error["status"] = status
        if details:
            error["details"] = details
        _out({"error": error})
    _note(f"error: {code}: {message}")
    for problem in (details or {}).get("problems", []):
        if isinstance(problem, dict):
            _note(f"  - {problem.get('where', '?')}: {problem.get('message', '')}")
    workers = (details or {}).get("workers")
    if workers:
        _note(f"  workers: {', '.join(str(w) for w in workers)}")
    if hint:
        _note(f"hint: {hint}")
    raise typer.Exit(code=1)


@contextmanager
def _guard(as_json: bool) -> Iterator[None]:
    """Turn every way a command can fail into one message, one JSON error and exit 1."""
    try:
        yield
    except typer.Exit:
        raise
    except StudioError as err:
        _fail(as_json, err.code, err.message, status=err.status, details=err.details)
    except PlatformError as err:
        code = next((c for cls, c in _PLATFORM_CODES if isinstance(err, cls)), "platform_error")
        _fail(as_json, code, str(err))
    except (_CliError, ValueError, OSError) as err:
        _fail(as_json, "invalid_input", str(err))


def _draft_id(studio: StudioClient, ref: str) -> str:
    """A draft's id from an id or a name."""
    from fastaiagent.studio.client import _UUID

    return ref if _UUID.match(ref) else studio.find_draft(ref).id


def _pick_version(versions: list[VersionSummary], number: int | None) -> str:
    """A version's id from its number; the newest completed one when no number is given."""
    if number is not None:
        for version in versions:
            if version.version == number:
                return version.id
        have = ", ".join(str(v.version) for v in versions) or "none"
        raise StudioError("not_found", f"No version {number} on this draft (it has: {have}).")
    done = [v for v in versions if v.generation_status == "completed"]
    if not done:
        raise StudioError("no_version", "The draft has no completed version yet.")
    return max(done, key=lambda v: v.version).id


def _version_id(studio: StudioClient, draft_id: str, number: int | None) -> str:
    return _pick_version(studio.list_versions(draft_id), number)


def _read_text(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def _read_spec(path: str) -> dict[str, Any]:
    try:
        spec = json.loads(_read_text(path))
    except json.JSONDecodeError as err:
        raise _CliError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(spec, dict):
        raise _CliError(f"{path} must hold one JSON object (the spec).")
    return spec


def _table(rows: list[list[str]], header: list[str]) -> str:
    grid = [header, *rows]
    widths = [max(len(str(r[i])) for r in grid) for i in range(len(header))]
    return "\n".join(
        "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in grid
    )


def _warnings_text(warnings: list[Any]) -> str:
    return "\n".join(f"  [{w.code}] {w.where}: {w.message}" for w in warnings)


# -- context, list, new, status ---------------------------------------------------------------


@studio_app.command("context")
def context(
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """What the project has: LLM endpoints, knowledge bases, guardrail names, the SDK contract.

    Read this first. With no LLM endpoint nothing can be made: tell the person to add one
    in the console.
    """
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        ctx = studio.context()
        schema = studio.spec_schema()
        if json_out:
            payload = _dump(ctx)
            payload.update(
                {
                    "spec_version": schema.spec_version,
                    "renderer_version": schema.renderer_version,
                    "architectures": schema.architectures,
                    "tool_types": schema.tool_types,
                    "eval_criteria_keys": schema.eval_criteria_keys,
                    "guardrails": [_dump(g) for g in schema.guardrails],
                    "guardrails_that_warn": schema.guardrails_that_warn,
                    "guardrails_on_the_question_in_a_swarm": (
                        schema.guardrails_on_the_question_in_a_swarm
                    ),
                }
            )
            _out(payload)
        else:
            lines = [
                f"Project   {ctx.project.name} ({ctx.project.id})",
                f"Install   {ctx.environment.tier}; "
                + ("writes allowed" if ctx.environment.studio_writes_allowed else "WRITES OFF"),
                "LLM endpoints:",
            ]
            lines += [
                f"  {e.id}  {e.name}  ({e.provider} {e.model})" for e in ctx.llm_endpoints
            ] or ["  none: add one in the console; nothing can be made without it"]
            lines.append("Knowledge bases:")
            lines += [f"  {k.id}  {k.name}" for k in ctx.knowledge_bases] or ["  none"]
            lines.append(
                f"SDK       generated code targets {ctx.sdk.dependency_range} "
                f"(renderer {ctx.sdk.renderer_version})"
            )
            lines.append(f"Limits    {json.dumps(ctx.limits)}")
            lines.append("Guardrails:")
            for g in schema.guardrails:
                args = ", ".join([*g.required, *(f"{o}?" for o in g.optional)])
                warn = "  [warns]" if g.factory in schema.guardrails_that_warn else ""
                lines.append(f"  {g.factory}({args}){warn}")
            lines.append(f"Tool types      {', '.join(schema.tool_types)}")
            lines.append(f"Architectures   {', '.join(schema.architectures)}")
            typer.echo("\n".join(lines))
        if not ctx.llm_endpoints:
            _note("note: this project has no LLM endpoint, so no version can be made.")
        if not ctx.environment.studio_writes_allowed:
            _note("note: Studio writes are off on this install (production rules).")


@studio_app.command("list")
def list_(
    status: str | None = typer.Option(None, "--status", help="Only drafts in this status."),
    query: str | None = typer.Option(None, "--query", "-q", help="Only names/briefs with this."),
    limit: int = typer.Option(50, "--limit", help="At most this many (200 max)."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """The project's drafts, each with the step it is on."""
    with _guard(json_out):
        drafts = _client(api_key, target, project_id).list_drafts(
            status=status, q=query, limit=limit
        )
        if json_out:
            _out([_dump(d) for d in drafts])
            return
        rows = [
            [
                d.id,
                d.name,
                d.status,
                d.architecture or "-",
                (d.steps_summary.current if d.steps_summary else "") or "-",
            ]
            for d in drafts
        ]
        typer.echo(
            _table(rows, ["ID", "NAME", "STATUS", "SHAPE", "STEP"]) if rows else "No drafts."
        )


@studio_app.command("new")
def new(
    name: str = typer.Option(..., "--name", help="The draft's name (unique in the project)."),
    brief: str = typer.Option(
        ..., "--brief", help="What the person asked for, in at least 10 characters."
    ),
    data: list[str] | None = typer.Option(
        None,
        "--data",
        help="A sample-data file (.csv .json .jsonl .md .txt .pdf .docx). Repeatable.",
    ),
    data_text: list[str] | None = typer.Option(
        None,
        "--data-text",
        help="name=file: text you read yourself (a scan or image) as sample data. Repeatable.",
    ),
    harness: bool = typer.Option(
        False, "--harness", help="Wire prompts, eval cases and the generated code to the plane."
    ),
    endpoint: str | None = typer.Option(
        None,
        "--endpoint",
        help="LLM endpoint id. Optional when the project has exactly one (see `studio context`).",
    ),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Make a draft and add its sample data. Then write a spec and `studio spec validate` it."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        texts: list[tuple[str, Path]] = []
        for entry in data_text or []:
            label, sep, filename = entry.partition("=")
            if not sep or not label or not filename:
                raise _CliError(f"--data-text wants name=file, got {entry!r}.")
            texts.append((label, Path(filename)))
        files = [Path(p) for p in data or []]
        for path in [*files, *(p for _, p in texts)]:
            if not path.is_file():
                raise _CliError(f"No such file: {path}")

        endpoint_id = endpoint
        if endpoint_id is None:
            endpoints = studio.context().llm_endpoints
            if not endpoints:
                raise StudioError("no_llm_endpoint", "The project has no LLM endpoint.")
            if len(endpoints) > 1:
                names = "; ".join(f"{e.id} ({e.name})" for e in endpoints)
                raise _CliError(
                    f"The project has {len(endpoints)} LLM endpoints; pass --endpoint: {names}"
                )
            endpoint_id = endpoints[0].id

        draft = studio.create_draft(name, brief, endpoint_id, use_harness=harness)
        fixtures: list[Any] = []
        problems: list[dict[str, str]] = []
        # One file per request, so a refusal names the file that caused it.
        for path in files:
            try:
                fixtures += studio.upload_fixtures(draft.id, [(path.name, path.read_bytes())])
            except StudioError as err:
                problems.append({"file": str(path), "code": err.code, "message": err.message})
        for label, path in texts:
            try:
                fixtures.append(
                    studio.add_text_fixture(
                        draft.id, label, path.read_text(encoding="utf-8"), source_filename=path.name
                    )
                )
            except StudioError as err:
                problems.append({"file": str(path), "code": err.code, "message": err.message})

        if json_out:
            payload: dict[str, Any] = {
                "draft": _dump(draft),
                "fixtures": [_dump(f) for f in fixtures],
            }
            if problems:
                payload["errors"] = problems
            _out(payload)
        else:
            typer.echo(f"Draft {draft.id}  {draft.name}")
            for fixture in fixtures:
                typer.echo(f"  sample data: {fixture.name} ({fixture.kind})")
            if draft.console_url:
                typer.echo(f"Console: {draft.console_url}")
            typer.echo(
                f"Next: write a spec, then `fastaiagent studio spec validate {draft.id} spec.json`"
            )
        for problem in problems:
            _note(f"not added: {problem['file']}: {problem['code']}: {problem['message']}")
        if problems:
            _note(f"The draft was made ({draft.id}); the files above were not added.")
            raise typer.Exit(code=1)


@studio_app.command("status")
def status(
    draft: str = typer.Argument(..., help="Draft id or name."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Where the draft is, and the next action.

    If it says stale, someone changed the draft in the console after the current version
    was made: stop and ask the person.
    """
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        detail = studio.get_draft(_draft_id(studio, draft))
        steps = detail.steps
        if json_out:
            payload = _dump(detail)
            payload.pop("messages", None)
            _out(payload)
        else:
            typer.echo(f"{detail.name}  ({detail.id})")
            typer.echo(f"Status {detail.status}; shape {detail.architecture or 'not built yet'}")
            if steps:
                for step in steps.steps:
                    typer.echo(f"  {step.key:<9} {step.state:<12} {step.summary}")
                nxt = steps.next
                typer.echo(f"Next: {nxt.action}" + (f" — {nxt.reason}" if nxt.reason else ""))
                if nxt.hint:
                    typer.echo(f"      {nxt.hint}")
        if steps and steps.stale:
            _note(
                "STALE: someone changed this draft's settings in the console after the current "
                "version was made. Stop and ask the person before changing anything."
            )


# -- spec -------------------------------------------------------------------------------------


@spec_app.command("schema")
def spec_schema(
    full: bool = typer.Option(
        False, "--full", help="Also print the guardrail names, tool types and limits."
    ),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Print the spec's JSON Schema (always JSON)."""
    with _guard(True if json_out else False):
        schema = _client(api_key, target, project_id).spec_schema()
        _out(_dump(schema) if full else schema.json_schema)


@spec_app.command("validate")
def spec_validate(
    draft: str = typer.Argument(..., help="Draft id or name."),
    file: str = typer.Argument(..., help="The spec, a JSON file ('-' reads stdin)."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Check a spec. Makes nothing. Exit 0 only when there are no problems.

    Warnings do not stop a version and do not change the exit code: read every one out to
    the person.
    """
    with _guard(json_out):
        spec = _read_spec(file)
        studio = _client(api_key, target, project_id)
        check = studio.validate_spec(_draft_id(studio, draft), spec)
        if json_out:
            _out(_dump(check))
        else:
            typer.echo("The spec checks clean." if check.ok else "The spec has problems:")
            for problem in check.problems:
                typer.echo(f"  - {problem.where}: {problem.message}")
            if check.warnings:
                typer.echo("Warnings (they do not stop a version; read them out):")
                typer.echo(_warnings_text(check.warnings))
        if check.problems or not check.ok:
            raise typer.Exit(code=1)


@spec_app.command("push")
def spec_push(
    draft: str = typer.Argument(..., help="Draft id or name."),
    file: str = typer.Argument(..., help="The spec, a JSON file ('-' reads stdin)."),
    note: str | None = typer.Option(None, "--note", help="Why this version (kept with it)."),
    wait: bool = typer.Option(False, "--wait", help="Wait until the version is made."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Make the next version from a whole spec. Never push a spec the person has not agreed to.

    A spec with problems makes nothing and exits 1. The old version is kept.
    """
    with _guard(json_out):
        spec = _read_spec(file)
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        accepted = studio.create_version(draft_id, spec, note=note)
        version = studio.wait_for_version(draft_id, accepted.version_id) if wait else None
        if version is not None and version.generation_status != "completed":
            # One document on stdout: the error carries what was accepted.
            _fail(
                json_out,
                "version_failed",
                version.error_message or "The version could not be made.",
                details={"version": accepted.version, "version_id": accepted.version_id},
            )
        if json_out:
            payload = _dump(accepted)
            if version is not None:
                payload["generation"] = _dump(version)
            _out(payload)
        else:
            state = f" — {version.generation_status}" if version else " — being made"
            typer.echo(f"Version {accepted.version} ({accepted.version_id}){state}")
            if accepted.warnings:
                typer.echo("Warnings (read them out to the person):")
                typer.echo(_warnings_text(accepted.warnings))


@spec_app.command("get")
def spec_get(
    draft: str = typer.Argument(..., help="Draft id or name."),
    version: int | None = typer.Option(
        None, "--version", "-v", help="Version number (default: current)."
    ),
    out: str | None = typer.Option(None, "--out", help="Write the spec to this file."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """The spec of a version, to edit and push again (always JSON)."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        found = studio.get_version(draft_id, _version_id(studio, draft_id, version))
        if found.spec is None:
            raise StudioError("not_found", f"Version {found.version} carries no spec.")
        text = json.dumps(found.spec, indent=2)
        if out:
            Path(out).write_text(text + "\n", encoding="utf-8")
            _note(f"spec of version {found.version} written to {out}")
        else:
            typer.echo(text)


# -- show, diff ---------------------------------------------------------------------------------


def _blueprint(spec: dict[str, Any]) -> dict[str, Any]:
    """The design at a glance, derived from the spec."""

    def tools(items: list[dict[str, Any]]) -> list[str]:
        return [f"{t.get('name')} ({t.get('type')})" for t in items]

    plan: dict[str, Any] = {
        "name": spec.get("name"),
        "architecture": spec.get("architecture"),
        "description": spec.get("description"),
        "guardrails": [g.get("factory") for g in spec.get("guardrails", [])],
        "eval_cases": len(spec.get("eval_cases", [])),
    }
    if spec.get("architecture") == "single":
        plan["tools"] = tools(spec.get("tools", []))
    else:
        plan["workers"] = [
            {
                "name": w.get("name"),
                "tools": tools(w.get("tools", [])),
                "eval_cases": len(w.get("eval_cases", [])),
            }
            for w in spec.get("workers", [])
        ]
    if spec.get("architecture") == "swarm":
        plan["entrypoint"] = spec.get("entrypoint")
        plan["handoffs"] = [
            f"{h.get('from_worker')} -> {h.get('to_worker')}" for h in spec.get("handoffs", [])
        ]
    return plan


@studio_app.command("show")
def show(
    draft: str = typer.Argument(..., help="Draft id or name."),
    version: int | None = typer.Option(
        None, "--version", "-v", help="Version number (default: current)."
    ),
    files: bool = typer.Option(False, "--files", help="List the generated files."),
    file: str | None = typer.Option(
        None, "--file", help="Print one generated file, e.g. agent.py."
    ),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """The design (the blueprint) of a version, or its generated code with --files / --file."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        found = studio.get_version(
            draft_id, _version_id(studio, draft_id, version), include_files=bool(files or file)
        )
        rendered = found.rendered_files or {}
        if file:
            if file not in rendered:
                have = ", ".join(sorted(rendered)) or "none"
                raise StudioError("not_found", f"No generated file {file!r} (there are: {have}).")
            if json_out:
                _out({"path": file, "content": rendered[file]})
            else:
                typer.echo(rendered[file], nl=not rendered[file].endswith("\n"))
            return
        if files:
            if json_out:
                _out({"version": found.version, "files": sorted(rendered)})
            else:
                typer.echo("\n".join(sorted(rendered)))
            return
        blueprint = _blueprint(found.spec or {})
        if json_out:
            summary = _dump(found)
            summary.pop("spec", None)
            _out({"version": summary, "blueprint": blueprint})
            return
        typer.echo(
            f"Version {found.version} ({found.generation_status}), made {found.created_at}"
            + (f" — {found.note}" if found.note else "")
        )
        typer.echo(json.dumps(blueprint, indent=2))


@studio_app.command("diff")
def diff(
    draft: str = typer.Argument(..., help="Draft id or name."),
    a: int = typer.Argument(..., help="The earlier version number."),
    b: int = typer.Argument(..., help="The later version number."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """What changed from version A to version B, by place in the spec and by file."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        versions = studio.list_versions(draft_id)  # once, for both numbers
        result = studio.diff_versions(
            draft_id, _pick_version(versions, b), _pick_version(versions, a)
        )
        if json_out:
            _out(_dump(result))
            return
        typer.echo(f"Version {a} -> {b}")
        typer.echo("Spec:" if result.spec_changes else "Spec: no changes")
        for spec_change in result.spec_changes:
            typer.echo(f"  {spec_change.kind:<8} {spec_change.where}")
        typer.echo("Files:" if result.files else "Files: no changes")
        for file_change in result.files:
            typer.echo(f"  {file_change.kind:<8} {file_change.path}")


# -- try, eval ---------------------------------------------------------------------------------

_GUARDRAIL_NOTE = (
    "note: guardrails are not applied in a sandbox trial; they apply in the exported code."
)


def _trial_text(trial: Trial) -> str:
    lines = [f"Trial {trial.status}" + (f" (worker: {trial.worker})" if trial.worker else "")]
    if trial.status == "failed":
        lines.append(f"Error: {trial.error or 'unknown'}")
    elif trial.output is not None:
        lines.append(f"Answer: {trial.output}")
    if trial.mock_calls:
        lines.append("Tool calls:")
        for call in trial.mock_calls:
            matched = "matched" if call.matched else "no match"
            lines.append(f"  {call.tool} {json.dumps(call.args)} -> {matched}")
    if trial.trace_url:
        lines.append(f"Trace: {trial.trace_url}")
    return "\n".join(lines)


@studio_app.command("try")
def try_(
    draft: str = typer.Argument(..., help="Draft id or name."),
    question: str = typer.Argument(..., help="What to ask the agent."),
    worker: str | None = typer.Option(None, "--worker", help="Required for a team: one worker."),
    version: int | None = typer.Option(
        None, "--version", "-v", help="Version number (default: current)."
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for the answer (the default; there is no command to read a trial later).",
    ),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Ask the agent one question in the sandbox (started if needed).

    A single agent is tried whole; a team one worker at a time. Tools answer from the
    draft's sample data, never from a live system. Exit 1 if the trial failed.
    """
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        version_id = _version_id(studio, draft_id, version) if version is not None else None
        trial = studio.run_trial(
            draft_id, question, worker=worker, version_id=version_id, wait=wait
        )
        if json_out:
            # ``result`` is the plane's own (a string, or {"output": …}); ``output`` is
            # the answer either way, so a tool never has to tell the two shapes apart.
            payload = _dump(trial)
            payload["output"] = trial.output
            _out(payload)
        else:
            typer.echo(_trial_text(trial))
        _note(_GUARDRAIL_NOTE)
        if trial.status == "failed":
            raise typer.Exit(code=1)


def _eval_text(runs: list[Trial], rollup: EvalRollup | None) -> str:
    lines: list[str] = []
    for run in runs:
        if run.status == "failed":
            lines.append(f"Run{f' ({run.worker})' if run.worker else ''} failed: {run.error}")
    if rollup is not None:
        for group in rollup.groups:
            lines.append(
                f"{group.scope}"
                + ("  (runs only from the exported code)" if group.on_export_only else "")
            )
            for case in group.cases:
                detail = f" — {case.detail}" if case.detail and case.status == "failed" else ""
                lines.append(f"  [{case.status}] {case.name}{detail}")
        t = rollup.totals
        lines.append(
            f"Version {rollup.version}: {t.passed} passed, {t.failed} failed, "
            f"{t.needs_review} need review, {t.not_run} not run, {t.on_export_only} on export only"
        )
    return "\n".join(lines)


@studio_app.command("eval")
def eval_(
    draft: str = typer.Argument(..., help="Draft id or name."),
    worker: str | None = typer.Option(None, "--worker", help="A team: only this worker's cases."),
    case: str | None = typer.Option(None, "--case", help="Only the case with this name."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for the scores (the default)."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Run the eval cases in the sandbox. Exit 0 only when none failed.

    Scoring is by word rules, not a judge: a case with only a `behaviour` reads "needs
    review", which is not a failure. A team's end-to-end cases run from the exported code.
    """
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        runs = studio.run_evals(draft_id, worker=worker, case=case, wait=wait)
        rollup = studio.get_evals(draft_id) if wait else None
        failed = any(r.status == "failed" for r in runs) or bool(rollup and rollup.totals.failed)
        if json_out:
            _out({"runs": [_dump(r) for r in runs], "rollup": _dump(rollup) if rollup else None})
        else:
            typer.echo(_eval_text(runs, rollup) or f"Started {len(runs)} run(s).")
        _note(_GUARDRAIL_NOTE)
        if failed:
            raise typer.Exit(code=1)


# -- export, promote, open, delete ---------------------------------------------------------------


@studio_app.command("export")
def export(
    draft: str = typer.Argument(..., help="Draft id or name."),
    out: str = typer.Option(".", "--out", help="Unpack the project into this folder."),
    version: int | None = typer.Option(
        None, "--version", "-v", help="Version number (default: current)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite files that already exist."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Unpack the project (code, tests, eval cases, data) into a folder; the code is then yours.

    Refuses to overwrite without --force, and writes nothing if it refuses. Run the
    exported tests, then run the agent with a key.
    """
    from fastaiagent.studio._export import unpack_export

    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        draft_id = _draft_id(studio, draft)
        version_id = _version_id(studio, draft_id, version) if version is not None else None
        with tempfile.TemporaryDirectory() as scratch:
            archive = studio.export(draft_id, Path(scratch) / "project.zip", version_id=version_id)
            written = unpack_export(archive, out, force=force)
        root = Path(out).resolve()
        # Forward slashes on every platform: a tool reads this JSON, and the zip names them so.
        relative = sorted(p.relative_to(root).as_posix() for p in written)
        folders = sorted({r.split("/", 1)[0] for r in relative})
        if json_out:
            _out({"out": str(root), "folders": folders, "files": relative})
        else:
            typer.echo(f"Exported {len(relative)} files to {root}")
            for folder in folders:
                typer.echo(f"  {folder}/")
            typer.echo("Next: run its tests (they run offline), then the agent with a key.")


@studio_app.command("promote")
def promote(
    draft: str = typer.Argument(..., help="Draft id or name."),
    agent: str = typer.Option(..., "--agent", help="The agent id the exported code registered."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Turn the eval cases into a real eval suite on an agent the exported code registered."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        result = studio.promote(_draft_id(studio, draft), agent)
        if json_out:
            _out(_dump(result))
        else:
            typer.echo(
                f"Promoted {result.cases} cases: suite {result.suite_id}, "
                f"dataset {result.dataset_id}"
            )


def _draft_url(draft: Draft) -> str:
    if not draft.console_url:
        raise StudioError("not_found", "The plane did not give a console address for this draft.")
    return draft.console_url


@studio_app.command("open")
def open_(
    draft: str = typer.Argument(..., help="Draft id or name."),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Print the console address of the draft (it does not launch a browser)."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        found = studio.get_draft(_draft_id(studio, draft))
        url = _draft_url(found)
        if json_out:
            _out({"id": found.id, "name": found.name, "console_url": url})
        else:
            typer.echo(url)


@studio_app.command("delete")
def delete(
    draft: str = typer.Argument(..., help="Draft id or name."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Do not ask first (asked only at a terminal)."
    ),
    json_out: bool = _opt_json(),
    api_key: str | None = _opt_key(),
    target: str | None = _opt_target(),
    project_id: str | None = _opt_project(),
) -> None:
    """Delete a draft and its versions. Cannot be undone."""
    with _guard(json_out):
        studio = _client(api_key, target, project_id)
        found = studio.find_draft(draft)
        if (
            not yes
            and sys.stdin.isatty()
            and not typer.confirm(f"Delete {found.name!r} ({found.id})?")
        ):
            _note("Not deleted.")
            raise typer.Exit(code=1)
        studio.delete_draft(found.id)
        if json_out:
            _out({"deleted": found.id, "name": found.name})
        else:
            typer.echo(f"Deleted {found.name} ({found.id})")
