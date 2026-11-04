"""A team member registers as itself, and keeps its plane identity (1.99.0).

A Supervisor or Swarm runs a fresh copy of each member every time. Members built
after ``connect()`` — the order a plane-prompt user has to use — register on their
first run, and that run is the copy's. Before 1.99.0 the push stamped ``agent_id``
on the copy and threw it away with it: the user's agent kept ``None``, every later
copy copied ``None``, and the member's tools were never gated by managed
governance again (live: an always-require-approval tool ran unapproved on every
run). The pushed definition was the copy's too, so a Swarm member showed the
swarm's ``handoff_to_*`` tools on the plane.

No mocking: a real HTTP server on a real socket answers the agent-push endpoint
the way the plane does, and the real ``connect()`` auto-registers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent._platform import push as push_mod
from fastaiagent.prompt import Prompt
from fastaiagent.testing.models import FunctionModel, TestModel
from tests.test_push_race import _Recorder, plane  # noqa: F401 — the plane fixture


@pytest.fixture
def auto(plane: _Recorder) -> Iterator[_Recorder]:  # noqa: F811
    plane.delay = 0.0
    host, port = plane.server_address[0], plane.server_address[1]
    push_mod.reset_registration_state()
    # An agent left over from another test would be pushed by connect() itself
    # and make the name look registered before the team ever ran — which hid the
    # bug these tests are for. Every agent here is built after connect().
    push_mod._agent_registry.clear()
    fa.connect(api_key="fa_k_test", target=f"http://{host}:{port}")
    try:
        yield plane
    finally:
        fa.disconnect()
        push_mod.reset_registration_state()


def _registered(name: str, timeout: float = 10.0) -> str | None:
    """Wait for the first-run registration (a daemon thread) to land."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        agent_id = push_mod.pushed_agent_id(name)
        if agent_id:
            return agent_id
        time.sleep(0.02)
    return None


def _lead_route(messages: list[Any]) -> Any:
    if messages[-1].role.value == "tool":
        return "done"
    return "", [{"name": "delegate_to_member", "arguments": {"task": "help"}}]


def _member() -> fa.Agent:
    prompt = Prompt(
        name="m", template="You help.", version=2, slug="member-prompt", source="platform"
    )
    return fa.Agent(name="member", system_prompt=prompt, llm=TestModel("ok"))


def _team(topology: str, member: fa.Agent) -> Any:
    if topology == "supervisor":
        return fa.Supervisor(
            name="lead", llm=FunctionModel(_lead_route), workers=[fa.Worker(member, role="member")]
        )
    entry = fa.Agent(
        name="entry",
        llm=FunctionModel(
            lambda m: ("", [{"name": "handoff_to_member", "arguments": {"reason": "x"}}])
        ),
    )
    return fa.Swarm(name="swarm", agents=[entry, member], entrypoint="entry")


@pytest.mark.parametrize("topology", ["supervisor", "swarm"])
def test_a_member_built_after_connect_keeps_its_plane_identity(
    topology: str, auto: _Recorder
) -> None:
    member = _member()
    team = _team(topology, member)

    team.run("hi")
    agent_id = _registered("member")
    assert agent_id, "the member was registered on its first run"
    assert member.agent_id == agent_id, "the id landed on the user's agent, not the copy"

    if topology == "supervisor":
        copy = team._build_worker_clone(team.workers[0])
    else:
        from fastaiagent.agent.swarm import SwarmState

        copy = team._active_agent("member", SwarmState())
    assert copy.agent_id == agent_id, "the next run's copy is enrolled from the start"


@pytest.mark.parametrize("topology", ["supervisor", "swarm"])
def test_the_pushed_definition_is_the_members_own(topology: str, auto: _Recorder) -> None:
    team = _team(topology, _member())

    team.run("hi")
    assert _registered("member")

    pushed = [p for p in auto.posts if p.get("name") == "member"]
    assert len(pushed) == 1
    assert [t["name"] for t in pushed[0]["tools"]] == [], "no team mechanics in it"
    assert pushed[0]["prompt_slug"] == "member-prompt"


def test_a_supervisor_lead_is_enrolled_from_its_second_run(auto: _Recorder) -> None:
    """The lead is rebuilt every run; once its name is registered, each new lead
    gets the id from the registration step its run opens with, before any tool
    runs."""
    team = _team("supervisor", _member())
    team.run("hi")
    lead_id = _registered("lead")
    assert lead_id

    next_lead = team._build_inner_agent()
    assert next_lead.agent_id is None
    push_mod.auto_register_async(next_lead)  # the first thing its run does
    assert next_lead.agent_id == lead_id
    assert len([p for p in auto.posts if p.get("name") == "lead"]) == 1


# ── streamed runs register too ─────────────────────────────────────────────


def _stream(runner: Any) -> None:
    import asyncio

    async def consume() -> None:
        async for _ in runner.astream("hi"):
            pass

    asyncio.run(consume())


@pytest.mark.parametrize("topology", ["agent", "supervisor", "swarm"])
def test_a_streamed_first_run_registers(topology: str, auto: _Recorder) -> None:
    """Only ``arun`` registered an agent on its first run. An agent built after
    ``connect()`` and only ever streamed — alone or in a team — never got an
    ``agent_id``, so managed governance never gated it."""
    member = _member()
    runner: Any = member if topology == "agent" else _team(topology, member)

    _stream(runner)

    agent_id = _registered("member")
    assert agent_id, "a streamed run registered nothing"
    assert member.agent_id == agent_id
