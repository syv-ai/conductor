"""A compiled graph as a Mermaid flowchart: ``flowchart(compiled)``.

For a person reading a graph — in a notebook, a pull request, a bug
report — rather than for an editor, which draws from ``node`` and
``field`` itself. Everything here is read through those two and the
authored graph; nothing is worked out that compile did not already say.

One box per node the author placed, titled by its id and its type. A
node whose version is a graph is a subgraph holding its inner nodes under
their expanded ids (``emb/up``), nested as deep as the placements are.
One arrow per ref on every connected input, labelled with how the reading
input receives it: ``per row of <index>`` (``Iterate``), nothing
(``Broadcast``), ``whole`` (``Whole``), ``grouped by <index>``
(``Group``), ``gathered`` (``Gather``). A node with a fatal problem
carries the ``fault`` class — the one piece of styling — and an edge from
a node that is not there is not drawn; an arrow into a node compile could
not derive has no label, since how it receives is not known. The
problems themselves are ``compiled.problems``, not part of the picture.
Any graph draws, however broken. No layout: Mermaid lays it out.
"""

from __future__ import annotations

from conductor import (
    Broadcast,
    CompiledGraph,
    From,
    Gather,
    GraphNode,
    Group,
    Iterate,
    Receive,
    Ref,
    Whole,
)

_FAULT = "classDef fault stroke:#c62828,stroke-width:2px"


def flowchart(compiled: CompiledGraph) -> str:
    """The flowchart, as Mermaid source ending in a newline."""
    faulty = {p.node_id for p in compiled.problems if p.fatal}
    names: dict[str, str] = {}
    lines = ["flowchart LR"]
    for node in compiled.graph.nodes:
        lines += _placed(compiled, node, node.id, faulty, names, depth=1)
    for node_id in compiled.execution_order:
        node = compiled.node(node_id)
        for field in node.interface.inputs:
            ref = Ref(node_id, field.name)
            binding = compiled.field(ref).binding
            if not isinstance(binding, From) or node_id not in names:
                continue
            drawn = [source for source in binding.refs if source.node_id in names]
            if not drawn:
                continue
            label = _label(compiled.field(ref).receives) if node.state == "ready" else None
            arrow = "-->" if label is None else f"-->|{label}|"
            lines += [f"    {names[source.node_id]} {arrow} {names[node_id]}" for source in drawn]
    if faulty:
        lines.append(f"    {_FAULT}")
    return "\n".join(lines) + "\n"


def _placed(
    compiled: CompiledGraph, node: GraphNode, expanded_id: str, faulty: set[str], names: dict[str, str], *, depth: int
) -> list[str]:
    """One placed node's lines: a box, or a subgraph of its inner nodes."""
    indent = "    " * depth
    name = names.setdefault(expanded_id, f"n{len(names)}")
    title = _quoted(f"{expanded_id} · {node.type}")
    fault = ":::fault" if expanded_id in faulty else ""
    placed = compiled.node(expanded_id)
    if placed.state == "resolution_failed" or placed.kind != "graph":
        return [f"{indent}{name}[{title}]{fault}"]
    lines = [f"{indent}subgraph {name} [{title}]"]
    for inner in placed.version.graph.nodes:
        lines += _placed(compiled, inner, f"{expanded_id}/{inner.id}", faulty, names, depth=depth + 1)
    lines.append(f"{indent}end")
    if fault:
        lines.append(f"{indent}class {name} fault")
    return lines


def _label(receives: Receive) -> str | None:
    """How an arrow's reading input receives it, in a few words; ``None`` for a broadcast."""
    match receives:
        case Iterate(index=index):
            return f"per row of {index.id}"
        case Group(index=index):
            return f"grouped by {index.id}"
        case Whole():
            return "whole"
        case Gather():
            return "gathered"
        case Broadcast():
            return None
    raise TypeError(f"no label for {receives!r}")


def _quoted(text: str) -> str:
    return '"' + text.replace('"', "#quot;") + '"'
