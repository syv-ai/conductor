"""``make_graph`` writes a graph record from Python: the nodes given, chained or joined by the edges given, with the
graph's inputs and outputs declared.

It is the one place that reads a place in a graph from dotted text: ``"lines.result"`` is the output ``result`` of
the node ``lines``, ``"shout.mark"`` the input ``mark`` of ``shout``, and ``"read.result.email"`` the field ``email``
of the record ``read`` returns. No name has a dot, so the text is cut at its first two dots and nothing is guessed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from conductor._internal.base import REQUIRED
from conductor._internal.gate import RefusedError, read_record, record_values
from conductor._internal.node import (
    FunctionDefinition,
    NodeDefinition,
    SubgraphDefinition,
    read_by_hooks,
)
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord, RecordGraphInput


def make_graph(
    *nodes: NodeDefinition | tuple[str, NodeDefinition],
    edges: Sequence[tuple[str, str]] | None = None,
    constants: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
    inputs: Mapping[str, str | Sequence[str]] | None = None,
    outputs: Mapping[str, str] | None = None,
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(lines, shout, join)                                # a chain
        make_graph(lines, shout, edges=[("lines.result", "shout.line")])
        make_graph(lines, shout, join, constants={"shout.mark": "?"})
        make_graph(read, send, edges=[("read.result.email", "send.to")])     # the field email of read's record
        make_graph(("a", shout), ("b", shout), edges=[("a.result", "b.line")])

    A node's id in the graph is the id given with it, ``("a", shout)``, or else its definition's id; a definition
    used more than once with no id given is ``shout-1``, ``shout-2``, and so on, skipping an id another node already
    has. An edge is ``(output, input)``, each written ``"node.name"``, and
    an output may be followed by one field of the record it returns. ``constants`` and ``defaults`` are keyed by
    ``"node.input"``: a constant is the input's value on every call, and a default makes the input a graph input,
    with that value when the caller gives none. Each value is written in the record as its JSON reads back.

    With no ``edges`` the nodes form a chain: the one required input of each node that no constant or default fills
    gets an edge from the one output of the node before it. With ``edges``, those are every edge of the graph.
    Nothing is matched by name.

    The record declares the graph's inputs and outputs. With no ``inputs``, each input of a node no edge leads into
    that has no constant (and that no hook of the node reads) is a graph input of the input's name. With no
    ``outputs``, each output of a node no edge leaves is a graph output of the output's name. Where two would share a
    name, name them: ``inputs`` maps a graph input's name to the inputs it feeds (``{"line": ["strip.line",
    "shout.line"]}``), and ``outputs`` a graph output's name to the output it is (``{"result": "pair.result"}``).
    The inputs a hook adds and the outputs a hook returns are known only when the graph is compiled, so a graph
    input or output on one is given by name.

    The record is written and not compiled: an input or output a node does not have, a value its input's type
    refuses, an edge between types that do not fit and a graph not finished are in
    ``Graph.compile(record, registry).problems``.

    Raises:
        InvalidCallError: Listing every fault in writing the record itself: a place naming a node not given
            (``node_not_in_graph``), a value with no JSON form or holding a set (``invalid_value``), a place that is
            not ``"node.name"``, a name the record refuses, an input given a value and an edge, an edge written
            twice, or one id given to two nodes (``invalid_record``), two graph inputs or outputs of one name (``ports_share_a_name``), and a chain
            that can't be drawn (``not_a_chain``).
    """
    problems: list[Problem] = []
    definitions = _by_id(nodes, problems)
    ids = list(definitions)
    entries: dict[str, dict[str, dict[str, Any]]] = {node_id: {} for node_id in ids}
    for key, value in _values_by_place("constants", constants, ids, problems).items():
        entries[key[0]][key[1]] = {"constant": value}
    given_defaults = _values_by_place("defaults", defaults, ids, problems)
    for (node_id, name), value in given_defaults.items():
        entries[node_id][name] = {**entries[node_id].get(name, {}), "graph_input": name, "default": value}  # beside a constant, the door refuses it
    if edges is None:
        edges = _chain_edges(ids, list(definitions.values()), entries, problems)
    for i, (source_text, target_text) in enumerate(edges):
        source, target = _place(source_text, ("edges", i, 0), ids, problems, field=True), _place(target_text, ("edges", i, 1), ids, problems)
        if source is not None and target is not None:
            entries[target[0]].setdefault(target[1], {}).setdefault("sources", []).append(dict(zip(("node", "output", "field"), source, strict=False)))
    led_into = {node_id for node_id, named in entries.items() if any("sources" in entry for entry in named.values())}
    if inputs is None:
        open_inputs = [(node_id, name) for node_id in ids if node_id not in led_into for name in _open_inputs(definitions[node_id]) if name not in entries[node_id]]
        inputs = {name: f"{node_id}.{name}" for name, node_id in _named_once("inputs", [*open_inputs, *given_defaults], problems).items()}
    for name, places in inputs.items():
        for text in [places] if isinstance(places, str) else places:
            place = _place(text, ("inputs", name), ids, problems)
            if place is not None:
                entries[place[0]][place[1]] = {**entries[place[0]].get(place[1], {}), "graph_input": name}
    if outputs is None:
        led_out = {source["node"] for named in entries.values() for entry in named.values() for source in entry.get("sources", ())}
        leaves = [(node_id, name) for node_id in ids if node_id not in led_out for name in _outputs_of(definitions[node_id])]
        outputs = {name: f"{node_id}.{name}" for name, node_id in _named_once("outputs", leaves, problems).items()}
    declared = {name: place for name, text in outputs.items() if (place := _place(text, ("outputs", name), ids, problems)) is not None}
    try:
        record = read_record({
            "nodes": {node_id: {"node": definition.id, "version": definition.version, "inputs": entries[node_id]} for node_id, definition in definitions.items()},
            "outputs": {name: {"node": node_id, "output": output} for name, (node_id, output) in declared.items()},
        })
    except RefusedError as refused:  # a name the record refuses, an input given a value and an edge, an edge given twice
        problems += refused.problems
    if problems:
        raise InvalidCallError(problems)
    return record


def _place(text: str, where: tuple[str | int, ...], ids: Sequence[str], problems: list[Problem], *, field: bool = False) -> tuple[str, ...] | None:
    """``text``, a place in the graph, as its parts: ``"shout.mark"`` gives ``("shout", "mark")``, and with ``field``,
    ``"read.result.email"`` gives ``("read", "result", "email")``.

    Returns:
        The parts. ``None``, with a problem added to ``problems``, when the text is not ``"node.name"`` (or, with
        ``field``, ``"node.name.field"``), at ``("record", *where)``, or names a node not given
        (``node_not_in_graph``).
    """
    parts = tuple(text.split(".", 2 if field else 1))
    if len(parts) < 2 or not all(parts):
        reason = f"{text!r} is not a place in a graph; write 'node.name'"
        problems.append(problem.invalid_record("record", *where, where=".".join(map(str, where)), reason=reason, error="not_node_and_name"))
        return None
    if parts[0] not in ids:
        problems.append(problem.node_not_in_graph(parts[0], nodes=list(ids)))
        return None
    return parts


def _values_by_place(kind: str, given: Mapping[str, Any], ids: Sequence[str], problems: list[Problem]) -> dict[tuple[str, str], Any]:
    """``given``, the ``constants`` or ``defaults`` (``kind``) of ``make_graph``, by node and input, each value as the
    record keeps it (``record_values``). A fault goes to ``problems``."""
    by_node: dict[str, dict[str, Any]] = {}
    for key, value in given.items():
        place = _place(key, (kind, key), ids, problems)
        if place is not None:
            by_node.setdefault(place[0], {})[place[1]] = value
    written: dict[tuple[str, str], Any] = {}
    for node_id, named in by_node.items():
        try:
            written |= {(node_id, name): value for name, value in record_values(node_id, named).items()}
        except RefusedError as refused:
            problems += refused.problems
    return written


def _open_inputs(definition: NodeDefinition) -> list[str]:
    """The inputs of ``definition`` a caller may give: a function's that no hook reads, a subgraph's graph inputs."""
    match definition:
        case FunctionDefinition():
            return [name for name in definition.inputs if name not in read_by_hooks(definition)]
        case SubgraphDefinition():
            return list(dict.fromkeys(
                entry.graph_input for node in definition.graph.nodes.values() for entry in node.inputs.values() if isinstance(entry, RecordGraphInput)
            ))


def _outputs_of(definition: NodeDefinition) -> list[str]:
    """The outputs of ``definition``: a function's, a subgraph's graph outputs."""
    match definition:
        case FunctionDefinition():
            return list(definition.outputs)
        case SubgraphDefinition():
            return list(definition.graph.outputs)


def _named_once(kind: str, places: Sequence[tuple[str, str]], problems: list[Problem]) -> dict[str, str]:
    """Each place's name, ``(node, name)``, as the node it is on, by the name; ``ports_share_a_name`` added to
    ``problems`` for a name two places share."""
    counts = Counter(name for _, name in places)
    for name, count in counts.items():
        if count > 1:
            problems.append(problem.ports_share_a_name(kind, name, name=name, places=[f"{node_id}.{n}" for node_id, n in places if n == name]))
    return {name: node_id for node_id, name in places if counts[name] == 1}


def _by_id(nodes: Sequence[NodeDefinition | tuple[str, NodeDefinition]], problems: list[Problem]) -> dict[str, NodeDefinition]:
    """Each node's definition by its id in the graph: the id given with it, or its definition's, numbered ``-1``,
    ``-2`` for a definition used more than once with no id given. One id given to two nodes goes to ``problems``."""
    given = [node[0] for node in nodes if isinstance(node, tuple)]
    bare = Counter(node.id for node in nodes if not isinstance(node, tuple))
    taken = {*given, *(definition_id for definition_id, count in bare.items() if count == 1)}
    numbered: Counter[str] = Counter()
    by_id: dict[str, NodeDefinition] = {}
    for node in nodes:
        node_id, definition = node if isinstance(node, tuple) else (node.id, node)
        while not isinstance(node, tuple) and bare[definition.id] > 1 and (node_id == definition.id or node_id in taken):
            numbered[definition.id] += 1
            node_id = f"{definition.id}-{numbered[definition.id]}"
        if node_id in by_id:
            reason = f"the id {node_id!r} is given to two nodes"
            problems.append(problem.invalid_record("record", "nodes", node_id, where=f"nodes.{node_id}", reason=reason, error="repeated_key"))
        taken.add(node_id)
        by_id[node_id] = definition
    return by_id


def _chain_edges(ids: Sequence[str], nodes: Sequence[NodeDefinition], entries: Mapping[str, Mapping[str, Any]], problems: list[Problem]) -> list[tuple[str, str]]:
    """The one edge from each node into the next, given that ``entries`` already holds each node's constants and
    defaults; ``not_a_chain`` added to ``problems`` for each pair without exactly one."""
    edges: list[tuple[str, str]] = []
    for index in range(1, len(nodes)):
        before_id, before, after_id, after = ids[index - 1], nodes[index - 1], ids[index], nodes[index]
        match before, after:
            case FunctionDefinition(), FunctionDefinition():
                outputs = list(before.outputs)
                open_inputs = [name for name, field in after.inputs.items() if field.default is REQUIRED and name not in entries[after_id]]
            case _:  # a subgraph's types are known only when it is compiled, so a chain can't be drawn through it
                outputs, open_inputs = [], []
        if len(outputs) != 1 or len(open_inputs) != 1:
            problems.append(problem.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs))
        else:
            edges.append((f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"))
    return edges
