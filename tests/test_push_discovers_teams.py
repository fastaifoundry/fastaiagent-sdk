"""``fastaiagent push --module`` registers a team defined at module scope (1.99.0).

It found bare ``Agent`` objects only: a module holding a Supervisor and a Swarm
pushed their workers (when those happened to be module-level too) and never the
Supervisor's lead, so CI registration and first-run auto-registration left the
plane in different states.
"""

from __future__ import annotations

import types

import fastaiagent as fa
from fastaiagent.cli.push import _discover_agents
from fastaiagent.testing import TestModel


def _module() -> types.ModuleType:
    mod = types.ModuleType("team_mod")
    llm = TestModel()
    hidden = fa.Agent(name="hidden-worker", system_prompt="h", llm=llm)  # not module-level
    a = fa.Agent(name="a", system_prompt="a", llm=llm)
    b = fa.Agent(name="b", system_prompt="b", llm=llm)
    mod.a = a
    mod.lead = fa.Supervisor(
        name="lead", llm=llm, workers=[fa.Worker(a, role="a"), fa.Worker(hidden, role="h")]
    )
    mod.swarm = fa.Swarm(name="swarm", agents=[a, b], entrypoint="a")
    return mod


def test_a_module_level_team_is_registered_whole() -> None:
    found = _discover_agents(_module())

    names = [a.name for a in found]
    assert sorted(names) == ["a", "b", "hidden-worker", "lead"]
    assert len(names) == len(set(names)), "a shared member is pushed once"
    lead = next(a for a in found if a.name == "lead")
    assert sorted(t["name"] for t in lead.to_dict()["tools"]) == ["delegate_to_a", "delegate_to_h"]
    swarm_member = next(a for a in found if a.name == "b")
    assert swarm_member.to_dict()["tools"] == [], "no handoff_to_* tools in a member"


def test_a_routed_supervisor_has_no_lead_to_register() -> None:
    mod = types.ModuleType("routed_mod")
    llm = TestModel()
    mod.desk = fa.Supervisor(
        name="desk",
        llm=llm,
        workers=[
            fa.Worker(fa.Agent(name="x", llm=llm), role="x"),
            fa.Worker(fa.Agent(name="y", llm=llm), role="y"),
        ],
        routing="decisions",
        fallback_worker="y",
    )
    assert sorted(a.name for a in _discover_agents(mod)) == ["x", "y"]
