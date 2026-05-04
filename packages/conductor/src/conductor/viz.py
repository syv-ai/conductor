"""Render a ``CompiledGraph`` as a Mermaid flowchart.

Mermaid renders inline in GitHub markdown, on most documentation
sites, and in Jupyter via ``IPython.display.Markdown``. This module
takes a compiled graph and emits the equivalent flowchart text — no
external dependencies, no rendering done here.

Example::

    from conductor.viz import to_mermaid
    from IPython.display import Markdown

    Markdown("```mermaid\\n" + to_mermaid(compiled) + "\\n```")

Visual conventions:

- Regular IO node          → ``[rectangle]``
- Decision node            → ``{diamond}``
- Compound start           → ``[[subroutine]]``
- Managed body / end node  → ``[/parallelogram/]``
- Edge                     → solid arrow ``-->``
- Shared reference         → dashed arrow ``-.->``
- Decision-edge-guard CEL  → rendered between ``⟦…⟧`` on the edge label
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conductor.graph.compiler import CompiledGraph


# Mermaid keywords that cannot be used as bare node ids.
_RESERVED_IDS = frozenset({
    "end", "subgraph", "click", "style", "linkStyle", "classDef", "class",
    "default",
})


def _safe_id(node_id: str) -> str:
    """Coerce ``node_id`` into a valid Mermaid identifier.

    Replaces non-alphanumerics with ``_`` and prefixes ``n_`` if the
    result clashes with a reserved keyword or starts with a non-letter.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
    if not sanitized or not sanitized[:1].isalpha() or sanitized in _RESERVED_IDS:
        sanitized = f"n_{sanitized}"
    return sanitized


def _node_shape(compiled: "CompiledGraph", node_id: str) -> tuple[str, str]:
    node = compiled.node_map[node_id]
    if node_id in compiled.compound_nodes:
        return "[[", "]]"
    if node_id in compiled.managed_ids:
        return "[/", "/]"
    ndef = compiled.registry.get(node.type)
    if ndef is not None and getattr(ndef, "is_decision", False):
        return "{", "}"
    return "[", "]"


def _node_label(compiled: "CompiledGraph", node_id: str) -> str:
    node = compiled.node_map[node_id]
    title = node.node_label or node.type.split("@")[0]
    # Mermaid quotes need escaping. <br/> for line break.
    title = title.replace('"', "&quot;")
    return f"<b>{node_id}</b><br/>{title}"


def to_mermaid(compiled: "CompiledGraph", *, direction: str = "LR") -> str:
    """Render ``compiled`` as a Mermaid ``flowchart`` string.

    Args:
        compiled: A ``CompiledGraph`` returned by ``compile()``.
        direction: Mermaid layout direction — ``"LR"`` (default), ``"TB"``,
            ``"RL"``, or ``"BT"``.

    Returns:
        The flowchart definition — wrap it in a fenced ``mermaid`` code
        block to render it.
    """
    lines: list[str] = [f"flowchart {direction}"]

    # Nodes
    for node_id in compiled.execution_order:
        sid = _safe_id(node_id)
        open_, close_ = _node_shape(compiled, node_id)
        label = _node_label(compiled, node_id)
        lines.append(f"  {sid}{open_}\"{label}\"{close_}")

    # Build a guard-expression lookup so we can decorate decision edges.
    guard_by_edge: dict[str, str] = {}
    decision_guards = getattr(compiled, "decision_guards", None) or {}
    for guards in decision_guards.values():
        for g in guards:
            guard_by_edge[g.edge_id] = g.when.source if g.when else "else"

    # Edges (deduplicated by (source, target, target_handle))
    seen: set[tuple[str, str, str]] = set()
    for target_id, entries in compiled.incoming_map.items():
        for target_handle, source_id, source_handle, edge_id in entries:
            key = (source_id, target_id, target_handle or "")
            if key in seen:
                continue
            seen.add(key)
            label_parts: list[str] = []
            if source_handle and source_handle != "result":
                label_parts.append(source_handle)
            if target_handle and target_handle != "result":
                arrow = f"→{target_handle}" if label_parts else target_handle
                label_parts.append(arrow)
            guard = guard_by_edge.get(edge_id)
            if guard:
                label_parts.append(f"⟦{guard}⟧")
            label = " ".join(label_parts)
            if label:
                lines.append(
                    f"  {_safe_id(source_id)} -->|{label}| {_safe_id(target_id)}"
                )
            else:
                lines.append(f"  {_safe_id(source_id)} --> {_safe_id(target_id)}")

    # Shared references — dashed arrows
    for (target_id, target_handle), (source_id, _source_handle) in compiled.consume_map.items():
        label = target_handle if target_handle and target_handle != "result" else ""
        if label:
            lines.append(
                f"  {_safe_id(source_id)} -.->|{label}| {_safe_id(target_id)}"
            )
        else:
            lines.append(f"  {_safe_id(source_id)} -.-> {_safe_id(target_id)}")

    return "\n".join(lines)
