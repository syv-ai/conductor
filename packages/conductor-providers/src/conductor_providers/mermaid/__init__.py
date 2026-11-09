"""A compiled graph as a Mermaid flowchart: ``flowchart(compiled)``.

For a person reading a graph — in a notebook, a pull request, a bug
report — rather than for an editor, which draws from ``node`` and
``field`` itself. Everything here is read through those two and the
authored graph; nothing is worked out that compile did not already say.

One box per node the author drew, titled by its id and its type. A
node whose version is a graph is a subgraph holding its inner nodes under
their paths (``emb/up``), nested as deep as the graphs are embedded.
One arrow per ref on every connected input, labelled with how the reading
input receives it: ``per row of <index>``, nothing for one value,
``whole``, ``grouped by <index>``, ``gathered`` or ``merged``. A node with a fatal problem
carries the ``fault`` class — the one piece of styling — and an edge from
a node that is not there is not drawn; an arrow into a node compile could
not derive has no label, since how it receives is not known. The
problems themselves are ``compiled.problems``, not part of the picture.
Any graph draws, however broken. No layout: Mermaid lays it out.
"""

from __future__ import annotations

from typing import Any

from conductor import (
    CompiledGraph,
    From,
    GraphNode,
    ReceiveMode,
    Ref,
)

_FAULT = "classDef fault stroke:#c62828,stroke-width:2px"


def flowchart(compiled: CompiledGraph) -> str:
    """The flowchart, as Mermaid source ending in a newline."""
    faulty = {str(p.loc[0]) for p in compiled.problems if p.fatal}
    names: dict[str, str] = {}
    lines = ["flowchart LR"]
    for node in compiled.graph.nodes:
        lines += _node_lines(compiled, node, node.id, faulty, names, depth=1)
    for node_id in compiled.execution_order:
        node = compiled.node(node_id)
        plan = compiled.run_plan.nodes.get(node_id)
        for field in () if node.interface is None else node.interface.inputs:
            binding = node.graph_node.bindings.get(field.name)
            if not isinstance(binding, From) or node_id not in names:
                continue
            drawn = [source for source in binding.refs if source.node_id in names]
            if not drawn:
                continue
            label = None if plan is None else _label(plan.inputs[field.name])
            arrow = "-->" if label is None else f"-->|{label}|"
            lines += [f"    {names[source.node_id]} {arrow} {names[node_id]}" for source in drawn]
    if faulty:
        lines.append(f"    {_FAULT}")
    return "\n".join(lines) + "\n"


def _node_lines(
    compiled: CompiledGraph, node: GraphNode, node_id: str, faulty: set[str], names: dict[str, str], *, depth: int
) -> list[str]:
    """One node's lines: a box, or a subgraph of its inner nodes."""
    indent = "    " * depth
    name = names.setdefault(node_id, f"n{len(names)}")
    title = _quoted(f"{node_id} · {node.type}")
    fault = ":::fault" if node_id in faulty else ""
    found = compiled.node(node_id)
    if found.wiring is None or found.embedded_graph is None:
        return [f"{indent}{name}[{title}]{fault}"]  # a graph that couldn't be embedded has nothing inside
    lines = [f"{indent}subgraph {name} [{title}]"]
    for inner in found.version.graph.nodes:
        lines += _node_lines(compiled, inner, f"{node_id}/{inner.id}", faulty, names, depth=depth + 1)
    lines.append(f"{indent}end")
    if fault:
        lines.append(f"{indent}class {name} fault")
    return lines


def _label(input_plan: ReceiveMode) -> str | None:
    """How an arrow's reading input receives it, in a few words; ``None`` where there is nothing to say."""
    match input_plan.receive_mode:
        case "value_at_row":
            return f"per row of {input_plan.rows_of}"  # type: ignore[union-attr]
        case "all_rows_under_row":
            return f"grouped by {input_plan.rows_of}"  # type: ignore[union-attr]
        case "entire_value":
            return "whole"
        case "collected":
            return "gathered"
        case "merged":
            return "merged"
        case "same_value_every_row":
            return None


def _quoted(text: str) -> str:
    return '"' + text.replace('"', "#quot;") + '"'
