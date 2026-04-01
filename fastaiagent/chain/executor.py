"""Chain state-machine executor with cycles, parallel, HITL, and checkpointing."""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any

from fastaiagent._internal.errors import (
    ChainCycleError,
    ChainError,
)
from fastaiagent.chain.checkpoint import CheckpointStore
from fastaiagent.chain.node import Edge, NodeConfig, NodeType
from fastaiagent.chain.state import ChainState


def _render_template(template: str, context: dict[str, Any]) -> str:
    """Render {{path.to.value}} templates against context."""

    def replacer(match: re.Match) -> str:
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
    """Evaluate a condition expression against context."""
    rendered = _render_template(expression, context)

    # Try simple comparisons: "value >= 0.8", "status == done"
    for op, fn in [
        (">=", lambda a, b: float(a) >= float(b)),
        ("<=", lambda a, b: float(a) <= float(b)),
        ("!=", lambda a, b: str(a).strip() != str(b).strip()),
        ("==", lambda a, b: str(a).strip() == str(b).strip()),
        (">", lambda a, b: float(a) > float(b)),
        ("<", lambda a, b: float(a) < float(b)),
    ]:
        if op in rendered:
            parts = rendered.split(op, 1)
            if len(parts) == 2:
                try:
                    return fn(parts[0].strip(), parts[1].strip())
                except (ValueError, TypeError):
                    return False

    # Fallback: truthy check
    return bool(rendered and rendered.lower() not in ("false", "0", "none", ""))


async def execute_chain(
    nodes: list[NodeConfig],
    edges: list[Edge],
    initial_state: dict[str, Any],
    state_schema: dict | None = None,
    checkpoint_store: CheckpointStore | None = None,
    chain_name: str = "",
    execution_id: str | None = None,
    resume_from_node: str | None = None,
    hitl_handler: Any = None,
) -> dict[str, Any]:
    """Execute a chain as a state machine.

    Returns a dict with:
        output: final output value
        final_state: state dict at completion
        execution_id: for resume
        node_results: dict of node_id -> result
    """
    execution_id = execution_id or str(uuid.uuid4())
    state = ChainState(initial_state)
    node_results: dict[str, Any] = {}
    iteration_counters: dict[str, int] = {}
    node_map = {n.id: n for n in nodes}

    # Validate state against schema if provided
    if state_schema:
        state.validate(state_schema)

    # Determine execution order via topological sort (non-cyclic edges)
    exec_order = _topological_sort(nodes, edges)

    # If resuming, skip to the resume point
    start_idx = 0
    if resume_from_node:
        for i, nid in enumerate(exec_order):
            if nid == resume_from_node:
                start_idx = i
                break

    max_total_steps = 500
    step_count = 0

    for idx in range(start_idx, len(exec_order)):
        node_id = exec_order[idx]
        node = node_map.get(node_id)
        if node is None:
            continue

        step_count += 1
        if step_count > max_total_steps:
            raise ChainError(f"Chain exceeded maximum total steps ({max_total_steps})")

        # Build context for this node
        context = {
            "input": initial_state,
            "state": state.data,
            "node_results": node_results,
        }

        # Execute the node
        result = await _execute_node(node, context, state, hitl_handler)
        node_results[node_id] = result

        # Update state with result
        if isinstance(result, dict):
            state.update(result)
        elif result is not None:
            state.set(f"_{node_id}_output", result)

        # Validate state after update
        if state_schema:
            state.validate(state_schema)

        # Checkpoint
        if checkpoint_store:
            checkpoint_store.save(
                chain_name=chain_name,
                execution_id=execution_id,
                node_id=node_id,
                node_index=idx,
                state_snapshot=state.snapshot(),
                node_output={"output": result} if not isinstance(result, dict) else result,
                iteration_counters=iteration_counters,
            )

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
                    raise ChainCycleError(
                        f"Cycle {edge.source} → {edge.target} exceeded "
                        f"max_iterations ({max_iter})"
                    )
                continue  # "continue" or "exit_to_node" — just proceed

            # Re-execute from the cycle target
            cycle_result = await execute_chain(
                nodes=nodes,
                edges=edges,
                initial_state=state.snapshot(),
                state_schema=state_schema,
                checkpoint_store=checkpoint_store,
                chain_name=chain_name,
                execution_id=execution_id,
                resume_from_node=edge.target,
                hitl_handler=hitl_handler,
            )
            # Merge cycle results
            state = ChainState(cycle_result["final_state"])
            node_results.update(cycle_result["node_results"])
            break  # only follow one cyclic edge

    # Determine final output
    output = node_results.get(exec_order[-1]) if exec_order else state.data

    return {
        "output": output,
        "final_state": state.snapshot(),
        "execution_id": execution_id,
        "node_results": node_results,
    }


async def _execute_node(
    node: NodeConfig,
    context: dict[str, Any],
    state: ChainState,
    hitl_handler: Any = None,
) -> Any:
    """Execute a single node based on its type."""
    if node.type == NodeType.agent:
        if node.agent is None:
            return {"error": f"No agent attached to node '{node.id}'"}
        result = await node.agent.arun(
            str(context.get("input", "")),
            trace=False,
        )
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
        result = await node.tool.aexecute(args)
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
        for child in child_agents:
            if hasattr(child, "arun"):
                tasks.append(child.arun(str(context.get("input", "")), trace=False))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            outputs = []
            for r in results:
                if isinstance(r, Exception):
                    outputs.append({"error": str(r)})
                else:
                    outputs.append({"output": r.output})
            return {"outputs": outputs}
        return {"outputs": []}

    elif node.type == NodeType.hitl:
        if hitl_handler:
            approval = hitl_handler(node, context, state)
            return {"approved": approval}
        return {"approved": True, "message": "Auto-approved (no HITL handler)"}

    elif node.type in (NodeType.start, NodeType.end):
        return context.get("input", {})

    else:
        return {"error": f"Unknown node type: {node.type}"}


def _topological_sort(
    nodes: list[NodeConfig], edges: list[Edge]
) -> list[str]:
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
