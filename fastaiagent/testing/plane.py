"""Offline stand-in for the guardrail rules a connected platform distributes (1.105.0).

A connected agent enforces rules its platform authored; an offline test has no platform,
so until now the only ways to test "what does my app do when a platform rule blocks?" were a
live platform or writing into the SDK's private connection state. Assistants did the latter:
a test that set ``fastaiagent.client._connection.policy_cache`` by hand and called that proof.

:func:`plane_rules` is the supported version. The rules go through the same cache and the
same builder as a real ``GET /public/v1/policy`` answer, so what a test sees is what a
connected run does with those rules.

It tests **your app's handling** of a platform rule. It does not test that your platform has
such a rule, or that a connected run receives it: that still needs a connected test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def plane_rules(rules: list[dict[str, Any]]) -> Iterator[None]:
    """Run the block as if a connected platform had distributed ``rules``.

    Each rule is a dict in the platform's policy shape::

        {"name": "no-codenames", "guardrail_type": "input",      # input | output | tool
         "implementation_type": "regex", "config": {"pattern": "bluejay"},
         "validation_mode": "blocking", "on_error": "block"}

    A rule with no ``agent_ids`` applies to every agent, as on the platform. One with
    ``agent_ids`` applies only to agents built with a matching ``Agent(agent_id=...)``.

    Raises:
        ValueError: a rule cannot be enforced from its configuration (for example a ``code``
            rule, whose logic lives on the platform). A connected run would skip that rule;
            a test that silently checked nothing would be worse than one that fails.

    The policy is process-wide, like the real one: do not nest this, and do not use it from
    tests that run in parallel threads. The previous policy is restored on exit.
    """
    from fastaiagent.client import _connection
    from fastaiagent.guardrail.from_policy import clear_cache, guardrail_from_policy_rule

    copied = [dict(rule) for rule in rules]
    for rule in copied:
        if guardrail_from_policy_rule(rule) is None:
            raise ValueError(
                f"plane_rules: rule {rule.get('name')!r} "
                f"(implementation_type={rule.get('implementation_type', 'code')!r}) cannot be "
                "enforced from its configuration, so a connected agent would skip it."
            )
    digest = hashlib.sha256(json.dumps(copied, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]
    previous = _connection.policy_cache
    _connection.policy_cache = {
        **(previous or {}),
        "version": f"testing-{digest}",
        "guardrail_rules": copied,
    }
    clear_cache()
    try:
        yield
    finally:
        _connection.policy_cache = previous
        clear_cache()
