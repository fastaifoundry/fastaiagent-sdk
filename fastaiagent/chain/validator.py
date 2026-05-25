"""Chain validation and cycle detection."""

from __future__ import annotations

from fastaiagent.chain.node import Edge, NodeConfig, NodeType


def detect_cycles(nodes: list[NodeConfig], edges: list[Edge]) -> list[list[str]]:
    """Find all cycles in the chain graph. Returns list of node-id cycles."""
    adj: dict[str, list[str]] = {n.id: [] for n in nodes}
    for e in edges:
        if e.source in adj:
            adj[e.source].append(e.target)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path)
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        path.pop()
        rec_stack.discard(node)

    for node in adj:
        if node not in visited:
            dfs(node, [])

    return cycles


def validate_chain(nodes: list[NodeConfig], edges: list[Edge]) -> list[str]:
    """Validate chain structure. Returns list of error messages."""
    errors: list[str] = []
    node_ids = {n.id for n in nodes}

    if not nodes:
        errors.append("Chain has no nodes")
        return errors

    # Check edge targets exist
    for edge in edges:
        if edge.source not in node_ids:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids:
            errors.append(f"Edge target '{edge.target}' not found in nodes")

    # Check orphan nodes (no incoming or outgoing edges)
    sources = {e.source for e in edges}
    targets = {e.target for e in edges}
    connected = sources | targets
    for node in nodes:
        if len(nodes) > 1 and node.id not in connected:
            errors.append(f"Node '{node.id}' is orphaned (no edges)")

    # Check cyclic edges have max_iterations
    for edge in edges:
        if edge.is_cyclic:
            max_iter = edge.cycle_config.get("max_iterations")
            if not max_iter or max_iter < 1:
                errors.append(
                    f"Cyclic edge {edge.source} → {edge.target} must have max_iterations >= 1"
                )

    # Routing rules — mirror the selection logic in
    # ``executor._select_outgoing_edges`` so misconfigured chains surface
    # at validate() time instead of silently fanning out.
    node_by_id = {n.id: n for n in nodes}
    by_source: dict[str, list[Edge]] = {n.id: [] for n in nodes}
    for e in edges:
        if not e.is_cyclic and e.source in by_source:
            by_source[e.source].append(e)

    for source_id, outs in by_source.items():
        if not outs:
            continue
        source_node = node_by_id.get(source_id)

        if source_node is not None and source_node.type == NodeType.condition:
            # ``handle`` strings the node can return — every branch in its
            # conditions list plus an implicit "default".
            condition_specs = source_node.config.get("conditions", []) or []
            handles = {str(c.get("handle", "default")) for c in condition_specs}
            handles.add("default")
            covered: set[str] = {e.label for e in outs if e.label}
            uncovered = handles - covered - {"default"}
            if uncovered:
                errors.append(
                    f"Condition node '{source_id}' returns handle(s) "
                    f"{sorted(uncovered)} that no outgoing edge labels. "
                    f"Add chain.connect('{source_id}', '<target>', label='<handle>')."
                )
            if "default" not in covered and not any(not e.label for e in outs):
                errors.append(
                    f"Condition node '{source_id}' has no default edge. "
                    f"Add an unlabeled chain.connect('{source_id}', '<target>')."
                )
            continue

        # Non-condition sources: if any outgoing edge has a condition,
        # multiple unconditional siblings are ambiguous defaults.
        conditional = [e for e in outs if e.condition]
        unconditional = [e for e in outs if not e.condition]
        if conditional and len(unconditional) > 1:
            default_targets = ", ".join(e.target for e in unconditional)
            errors.append(
                f"Source '{source_id}' has {len(unconditional)} default "
                f"(unconditional) edges to [{default_targets}] alongside conditional "
                f"edges — at most one default is allowed."
            )

    return errors
