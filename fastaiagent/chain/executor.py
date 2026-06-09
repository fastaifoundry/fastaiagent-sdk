"""Chain state-machine executor with cycles, parallel, HITL, and checkpointing."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any

from fastaiagent._internal.errors import (
    ChainCycleError,
    ChainError,
    ChainRoutingError,
)
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.chain.idempotent import _current_checkpointer
from fastaiagent.chain.interrupt import (
    InterruptSignal,
    Resume,
    _agent_path,
    _resume_value,
)
from fastaiagent.chain.interrupt import (
    _execution_id as _execution_id_var,
)
from fastaiagent.chain.node import Edge, NodeConfig, NodeType
from fastaiagent.chain.state import ChainState
from fastaiagent.checkpointers.protocol import Checkpointer, PendingInterrupt

logger = logging.getLogger(__name__)


def _render_template(template: str, context: dict[str, Any]) -> str:
    """Render {{path.to.value}} templates against context."""

    def replacer(match: re.Match[str]) -> str:
        path = match.group(1).strip()
        parts = path.split(".")
        value: Any = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, "")
            else:
                value = ""
                break
        if not isinstance(value, str):
            import json

            try:
                return json.dumps(value, default=str)
            except Exception:
                return str(value)
        return value

    return re.sub(r"\{\{(.+?)\}\}", replacer, template)


def _evaluate_condition(expression: str, context: dict[str, Any]) -> bool:
    """Evaluate a condition expression against context.

    Supports comparison operators (``==``, ``!=``, ``>``, ``<``, ``>=``,
    ``<=``) plus the string helpers ``contains`` and ``startswith``
    documented in ``docs/chains/index.md``. Values may be ``{{state.x}}``
    templates resolved against the chain context.
    """
    rendered = _render_template(expression, context)

    # Two-word string operators take precedence — split on the keyword and
    # compare the trimmed, unquoted sides. Quotes are tolerated so users
    # can write ``state.msg contains "late"``.
    def _unquote(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    for keyword, op in ((" contains ", "contains"), (" startswith ", "startswith")):
        if keyword in rendered:
            left, right = rendered.split(keyword, 1)
            haystack = _unquote(left)
            needle = _unquote(right)
            if op == "contains":
                return needle in haystack
            return haystack.startswith(needle)

    # Try simple comparisons: "value >= 0.8", "status == done"
    comparisons: list[tuple[str, Callable[[str, str], bool]]] = [
        (">=", lambda a, b: float(a) >= float(b)),
        ("<=", lambda a, b: float(a) <= float(b)),
        ("!=", lambda a, b: str(a).strip() != str(b).strip()),
        ("==", lambda a, b: str(a).strip() == str(b).strip()),
        (">", lambda a, b: float(a) > float(b)),
        ("<", lambda a, b: float(a) < float(b)),
    ]
    for op, fn in comparisons:
        if op in rendered:
            parts = rendered.split(op, 1)
            if len(parts) == 2:
                try:
                    return fn(parts[0].strip(), parts[1].strip())
                except (ValueError, TypeError):
                    return False

    # Fallback: truthy check
    return bool(rendered and rendered.lower() not in ("false", "0", "none", ""))


def _select_outgoing_edges(
    node: NodeConfig,
    result: Any,
    context: dict[str, Any],
    outgoing: list[Edge],
    *,
    strict: bool = False,
) -> list[Edge]:
    """Pick which non-cyclic outgoing edges to activate after a node runs.

    Selection rules (see ``docs/chains/spec.md``):

    * ``NodeType.condition`` nodes return ``{"matched": handle}``. The
      edge whose ``label == handle`` wins; otherwise an unconditional
      edge (or one labeled ``"default"``) is the fallback.
    * For every other node type:

      - If **all** outgoing edges have ``condition=None``, every edge
        fires. This preserves the existing fan-out behavior — chains
        without ``condition=`` keywords behave exactly as before.
      - If **any** outgoing edge has a ``condition`` set, the first
        conditional edge whose expression evaluates true (in declaration
        order) wins. Unconditional siblings act as the default fallback;
        the first one is taken if no condition matched.

    When ``strict=True`` (``Chain(..., strict_routing=True)``), the
    fall-through "no edge matched" case raises ``ChainRoutingError``
    instead of silently terminating the branch.

    Returns the list of edges to follow (possibly empty, which terminates
    that branch of the chain).
    """
    if not outgoing:
        return []

    if node.type == NodeType.condition:
        handle = ""
        if isinstance(result, dict):
            handle = str(result.get("matched", ""))
        for e in outgoing:
            if e.label and e.label == handle:
                return [e]
        # Fallback: explicit "default" label, then unlabeled edge.
        for e in outgoing:
            if e.label == "default":
                return [e]
        for e in outgoing:
            if not e.label:
                return [e]
        if strict:
            raise ChainRoutingError(
                f"Condition node '{node.id}' returned handle {handle!r} but no "
                f"outgoing edge labels it and no default edge exists. "
                f"Available labels: {[e.label for e in outgoing if e.label] or '(none)'}."
            )
        return []

    conditional = [e for e in outgoing if e.condition]
    if not conditional:
        # Pure fan-out — backwards-compatible with pre-routing chains.
        return list(outgoing)

    for e in conditional:
        if e.condition and _evaluate_condition(e.condition, context):
            return [e]

    unconditional = [e for e in outgoing if not e.condition]
    if unconditional:
        return [unconditional[0]]
    if strict:
        raise ChainRoutingError(
            f"Node '{node.id}' has {len(conditional)} conditional outgoing edge(s) "
            f"and no default — none matched. Add an unlabeled "
            f"chain.connect('{node.id}', '<fallback>') to silence this error, "
            f"or disable strict_routing."
        )
    return []


async def execute_chain(
    nodes: list[NodeConfig],
    edges: list[Edge],
    initial_state: dict[str, Any],
    state_schema: dict[str, Any] | None = None,
    checkpointer: Checkpointer | None = None,
    chain_name: str = "",
    execution_id: str | None = None,
    resume_from_node: str | None = None,
    hitl_handler: Any = None,
    resume_value: Resume | None = None,
    run_context: Any | None = None,
    strict_routing: bool = False,
) -> dict[str, Any]:
    """Execute a chain as a state machine.

    Returns a dict with:
        output: final output value
        final_state: state dict at completion
        execution_id: for resume
        node_results: dict of node_id -> result
        status: "completed" or "paused"
        pending_interrupt: dict (only when status == "paused")

    ``resume_value``, when provided, is injected into the ``_resume_value``
    ContextVar for the *first* node executed (the one being resumed). It is
    cleared before subsequent nodes run so a chain that pauses again works
    cleanly.

    ``run_context`` is an optional :class:`RunContext` propagated to every
    tool and agent node so dependency-injected tools work identically
    inside a Chain and inside an Agent.
    """
    execution_id = execution_id or str(uuid.uuid4())
    state = ChainState(initial_state)
    node_results: dict[str, Any] = {}
    iteration_counters: dict[str, int] = {}
    node_map = {n.id: n for n in nodes}

    # Make execution_id visible to interrupt() / @idempotent.
    exec_id_token = _execution_id_var.set(execution_id)
    # Make the active checkpointer reachable by ``@idempotent``.
    cp_token = _current_checkpointer.set(checkpointer)

    try:
        # Validate state against schema if provided
        if state_schema:
            state.validate(state_schema)

        # Determine execution order via topological sort (non-cyclic edges).
        # The order itself doesn't decide *which* nodes run — that is driven
        # by the ``activated`` set below, so conditional routing can skip
        # branches — but it still gives us a deterministic dependency-safe
        # walk: a node only runs after every predecessor that fed it has run.
        exec_order = _topological_sort(nodes, edges)

        # Outgoing non-cyclic edges per source, preserving declaration order
        # so routing tiebreaks ("first match wins") are deterministic.
        outgoing_by_source: dict[str, list[Edge]] = {n.id: [] for n in nodes}
        for e in edges:
            if not e.is_cyclic and e.source in outgoing_by_source:
                outgoing_by_source[e.source].append(e)

        # Nodes with no non-cyclic incoming edges are auto-activated entry
        # points — same set the previous topological loop implicitly started
        # from. Explicit ``NodeType.start`` nodes are entry points too.
        non_cyclic_in_degree: dict[str, int] = {n.id: 0 for n in nodes}
        for e in edges:
            if not e.is_cyclic and e.target in non_cyclic_in_degree:
                non_cyclic_in_degree[e.target] += 1
        activated: set[str] = {nid for nid, deg in non_cyclic_in_degree.items() if deg == 0}
        for n in nodes:
            if n.type == NodeType.start:
                activated.add(n.id)

        # If resuming, only the resume target is activated initially —
        # everything before it in ``exec_order`` is skipped.
        start_idx = 0
        if resume_from_node:
            for i, nid in enumerate(exec_order):
                if nid == resume_from_node:
                    start_idx = i
                    break
            activated = {resume_from_node}

        max_total_steps = 500
        step_count = 0

        for idx in range(start_idx, len(exec_order)):
            node_id = exec_order[idx]
            node = node_map.get(node_id)
            if node is None:
                continue

            # Skip nodes that no upstream branch routed to.
            if node_id not in activated:
                continue

            step_count += 1
            if step_count > max_total_steps:
                name = chain_name or "unnamed"
                raise ChainError(
                    f"Chain '{name}' exceeded maximum total steps "
                    f"({max_total_steps}). This usually means cycles "
                    f"are not terminating as expected.\n"
                    f"Options:\n"
                    f"  1. Review exit_condition on cyclic edges\n"
                    f"  2. Lower max_iterations on cycles\n"
                    f"  3. Split the chain into smaller sub-chains"
                )

            # Build context for this node
            context = {
                "input": initial_state,
                "state": state.data,
                "node_results": node_results,
            }

            # Execute the node, catching ``InterruptSignal`` from interrupt().
            # The first node of a resume sees ``_resume_value`` set; clear it
            # afterwards so a subsequent interrupt() call further down the
            # chain can suspend again.
            resume_token = None
            if idx == start_idx and resume_value is not None:
                resume_token = _resume_value.set(resume_value)
            try:
                result = await _execute_node(
                    node, context, state, hitl_handler, run_context=run_context
                )
            except InterruptSignal as sig:
                # Persist the suspension and bubble paused status up.
                ap = _agent_path.get()
                interrupt_ckpt = Checkpoint(
                    checkpoint_id=str(uuid.uuid4()),
                    chain_name=chain_name,
                    execution_id=execution_id,
                    node_id=node_id,
                    node_index=idx,
                    status="interrupted",
                    state_snapshot=state.snapshot(),
                    iteration_counters=iteration_counters,
                    interrupt_reason=sig.reason,
                    interrupt_context=sig.context,
                    agent_path=ap,
                )
                pending = PendingInterrupt(
                    execution_id=execution_id,
                    chain_name=chain_name,
                    node_id=node_id,
                    reason=sig.reason,
                    context=sig.context,
                    agent_path=ap,
                )
                if checkpointer is not None:
                    checkpointer.record_interrupt(interrupt_ckpt, pending)
                return {
                    "output": None,
                    "final_state": state.snapshot(),
                    "execution_id": execution_id,
                    "node_results": node_results,
                    "status": "paused",
                    "pending_interrupt": {
                        "reason": sig.reason,
                        "context": sig.context,
                        "node_id": node_id,
                        "agent_path": ap,
                    },
                }
            finally:
                if resume_token is not None:
                    _resume_value.reset(resume_token)

            node_results[node_id] = result

            # 2.4b (additive): the node's real output value is what an
            # output_schema validates and an output_key stores — unwrap the
            # ``{"output": ...}`` envelope tool/agent nodes return.
            out_value = result.get("output", result) if isinstance(result, dict) else result
            output_schema = node.config.get("output_schema")
            if output_schema:
                from fastaiagent.tool.schema import validate_schema

                violations = validate_schema(output_schema, out_value)
                if violations:
                    raise ChainError(
                        f"Node '{node_id}' output failed output_schema: "
                        + "; ".join(v.message for v in violations)
                    )

            # Update state with result. An explicit output_key stores the node's
            # output under that key; otherwise the legacy dict-merge /
            # ``_<node_id>_output`` auto-wrap is preserved unchanged.
            output_key = node.config.get("output_key")
            if output_key:
                state.set(output_key, out_value)
            elif isinstance(result, dict):
                state.update(result)
            elif result is not None:
                state.set(f"_{node_id}_output", result)

            # Validate state after update
            if state_schema:
                state.validate(state_schema)

            # Checkpoint
            if checkpointer is not None:
                checkpointer.put(
                    Checkpoint(
                        checkpoint_id=str(uuid.uuid4()),
                        chain_name=chain_name,
                        execution_id=execution_id,
                        node_id=node_id,
                        node_index=idx,
                        status="completed",
                        state_snapshot=state.snapshot(),
                        node_output=(
                            {"output": result} if not isinstance(result, dict) else result
                        ),
                        iteration_counters=iteration_counters,
                    )
                )

            # Activate the targets of every non-cyclic edge the routing
            # rules picked. Edges with no match are silently skipped so
            # the chain prunes dead branches instead of running them.
            #
            # ``context`` above was built *before* the node ran, and
            # ``ChainState.data`` returns a fresh copy — so condition
            # expressions like ``{{state.score}}`` would otherwise see
            # stale state. Refresh ``state`` (and ``node_results`` for
            # symmetry, even though it mutates in place) before edge
            # selection so a node can route on what it just wrote.
            routing_context = {
                "input": initial_state,
                "state": state.data,
                "node_results": node_results,
            }
            selected = _select_outgoing_edges(
                node,
                result,
                routing_context,
                outgoing_by_source.get(node_id, []),
                strict=strict_routing,
            )
            for sel in selected:
                activated.add(sel.target)

            # Handle cyclic edges from this node
            for edge in edges:
                if edge.source != node_id or not edge.is_cyclic:
                    continue

                counter_key = edge.cycle_config.get(
                    "iteration_counter_key", f"cycle_{edge.source}_{edge.target}"
                )
                max_iter = edge.cycle_config.get("max_iterations", 10)
                exit_condition = edge.cycle_config.get("exit_condition")

                iteration_counters.setdefault(counter_key, 0)
                iteration_counters[counter_key] += 1

                # Check exit condition
                if exit_condition and _evaluate_condition(exit_condition, context):
                    continue  # exit the cycle, proceed normally

                # Check max iterations
                if iteration_counters[counter_key] >= max_iter:
                    on_max = edge.cycle_config.get("on_max_reached", "error")
                    if on_max == "error":
                        src, tgt = edge.source, edge.target
                        raise ChainCycleError(
                            f"Cycle '{src}' -> '{tgt}' exceeded "
                            f"max_iterations ({max_iter}).\n"
                            f"Options:\n"
                            f"  1. Increase the limit: "
                            f"max_iterations={max_iter * 2}\n"
                            f"  2. Add an exit_condition\n"
                            f"  3. Set on_max_reached='continue'"
                        )
                    continue  # "continue" or "exit_to_node" — just proceed

                # Re-execute from the cycle target
                cycle_result = await execute_chain(
                    nodes=nodes,
                    edges=edges,
                    initial_state=state.snapshot(),
                    state_schema=state_schema,
                    checkpointer=checkpointer,
                    chain_name=chain_name,
                    execution_id=execution_id,
                    resume_from_node=edge.target,
                    hitl_handler=hitl_handler,
                    run_context=run_context,
                    strict_routing=strict_routing,
                )
                # If a node inside the cycle interrupted, bubble paused
                # status up — don't merge state from a partial run.
                if cycle_result.get("status") == "paused":
                    return cycle_result
                # Merge cycle results
                state = ChainState(cycle_result["final_state"])
                node_results.update(cycle_result["node_results"])
                break  # only follow one cyclic edge

        # Determine final output — pick the last node in topo order that
        # actually ran (conditional routing can prune the tail), falling
        # back to chain state when nothing executed.
        output: Any = state.data
        for nid in reversed(exec_order):
            if nid in node_results:
                output = node_results[nid]
                break

        return {
            "output": output,
            "final_state": state.snapshot(),
            "execution_id": execution_id,
            "node_results": node_results,
            "status": "completed",
            "pending_interrupt": None,
        }
    finally:
        _execution_id_var.reset(exec_id_token)
        _current_checkpointer.reset(cp_token)


async def _execute_node(
    node: NodeConfig,
    context: dict[str, Any],
    state: ChainState,
    hitl_handler: Any = None,
    *,
    run_context: Any | None = None,
) -> Any:
    """Execute a single node based on its type.

    ``run_context`` is the user-supplied :class:`RunContext` from
    :meth:`Chain.execute` / :meth:`Chain.aexecute`. Passing it through to
    ``agent.arun`` and ``tool.aexecute`` lets dependency-injected tools
    work identically inside a Chain and inside an Agent.
    """
    if node.type == NodeType.agent:
        if node.agent is None:
            return {"error": f"No agent attached to node '{node.id}'"}
        # Accept the same shapes ``Agent.run`` accepts so multimodal Chain
        # state can flow into a vision agent without the executor flattening
        # it. Dict inputs are still stringified for the legacy convention
        # (``chain.execute({"message": "..."})``).
        from fastaiagent.multimodal.image import Image as _MMImage
        from fastaiagent.multimodal.pdf import PDF as _MMPDF

        raw_input = context.get("input", "")
        if isinstance(raw_input, (str, _MMImage, _MMPDF, list)):
            agent_input = raw_input
        else:
            agent_input = str(raw_input)
        # Agent spans nest under the chain root span — see Chain.aexecute.
        result = await node.agent.arun(agent_input, context=run_context)
        return {"output": result.output, "tool_calls": result.tool_calls}

    elif node.type == NodeType.tool:
        if node.tool is None:
            return {"error": f"No tool attached to node '{node.id}'"}
        # Resolve arguments from config template
        args = {}
        for key, template in node.config.get("input_mapping", {}).items():
            if isinstance(template, str) and "{{" in template:
                args[key] = _render_template(template, context)
            else:
                args[key] = template
        # 2.4b: optional input-schema validation at the node boundary (additive).
        input_schema = node.config.get("input_schema")
        if input_schema:
            from fastaiagent.tool.schema import validate_schema

            violations = validate_schema(input_schema, args)
            if violations:
                raise ChainError(
                    f"Node '{node.id}' input failed input_schema: "
                    + "; ".join(v.message for v in violations)
                )
        result = await node.tool.aexecute(args, context=run_context)
        return {"output": result.output, "error": result.error}

    elif node.type == NodeType.condition:
        conditions = node.config.get("conditions", [])
        for cond in conditions:
            expr = cond.get("expression", "")
            if _evaluate_condition(expr, context):
                return {"matched": cond.get("handle", "default")}
        return {"matched": "default"}

    elif node.type == NodeType.transformer:
        template = node.config.get("template", "")
        return {"output": _render_template(template, context)}

    elif node.type == NodeType.parallel:
        # Execute child agents in parallel
        child_agents = node.config.get("agents", [])
        if not child_agents:
            return {"outputs": []}
        tasks = []
        from fastaiagent.multimodal.image import Image as _MMImage
        from fastaiagent.multimodal.pdf import PDF as _MMPDF

        raw_parallel_input = context.get("input", "")
        if isinstance(raw_parallel_input, (str, _MMImage, _MMPDF, list)):
            parallel_input = raw_parallel_input
        else:
            parallel_input = str(raw_parallel_input)
        for child in child_agents:
            if hasattr(child, "arun"):
                tasks.append(child.arun(parallel_input, context=run_context, trace=False))
        if not tasks:
            return {"outputs": []}

        mode = node.parallel_failure_mode

        if mode == "fail_fast":
            # asyncio.gather without return_exceptions raises the first
            # exception immediately and cancels in-flight siblings.
            try:
                results = await asyncio.gather(*tasks)
            except Exception as e:
                raise ChainError(
                    f"Parallel node '{node.id}' (fail_fast): child raised {type(e).__name__}: {e}"
                ) from e
            return {
                "outputs": [{"output": getattr(r, "output", str(r))} for r in results],
            }

        # ``continue`` (default) and ``any_success`` both collect every
        # result, then differ in how they react to all-failures.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                outputs.append({"error": str(r)})
            else:
                outputs.append({"output": getattr(r, "output", str(r))})

        if mode == "any_success":
            successes = [o for o in outputs if "error" not in o]
            if not successes:
                raise ChainError(
                    f"Parallel node '{node.id}' (any_success): every child failed "
                    f"({len(outputs)} children). Errors: "
                    f"{[o.get('error') for o in outputs]}"
                )
            return {"outputs": successes}

        return {"outputs": outputs}

    elif node.type == NodeType.hitl:
        if hitl_handler:
            approval = hitl_handler(node, context, state)
            return {"approved": approval}
        return {"approved": True, "message": "Auto-approved (no HITL handler)"}

    elif node.type in (NodeType.start, NodeType.end):
        return context.get("input", {})

    else:
        return {"error": f"Unknown node type: {node.type}"}


def _topological_sort(nodes: list[NodeConfig], edges: list[Edge]) -> list[str]:
    """Topological sort of nodes, ignoring cyclic edges."""
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}

    for e in edges:
        if not e.is_cyclic and e.source in adj:
            adj[e.source].append(e.target)
            in_degree[e.target] = in_degree.get(e.target, 0) + 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Add any remaining nodes not reached (isolated or in cycles)
    remaining = [n.id for n in nodes if n.id not in result]
    result.extend(remaining)

    return result
