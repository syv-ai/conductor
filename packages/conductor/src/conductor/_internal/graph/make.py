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

from pydantic import MISSING

from conductor._internal.base import REQUIRED, Address
from conductor._internal.gate import RefusedError, read_record, record_values
from conductor._internal.node import (
    FunctionDefinition,
    NodeDefinition,
    SubgraphDefinition,
    read_by_hooks,
)
from conductor._internal.problem import InvalidCallError, Problem, call_problems
from conductor._internal.record import GraphRecord, InputEntry, NodeRecord


def make_graph(
    *nodes: NodeDefinition | tuple[str, NodeDefinition],
    edges: Sequence[tuple[str, str]] | None = None,
    constants: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
    inputs: Mapping[str, str | Sequence[str]] = MappingProxyType({}),
    outputs: Mapping[str, str] = MappingProxyType({}),
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(lines, shout, join)                                # a chain
        make_graph(lines, shout, edges=[("lines.result", "shout.line")])
        make_graph(lines, shout, join, constants={"shout.mark": "?"})
        make_graph(read, send, edges=[("read.result.email", "send.to")])     # the field email of read's record
        make_graph(("a", shout), ("b", shout), edges=[("a.result", "b.line")])

    A node's id in the graph is the id given with it, ``("a", shout)``, or else its definition's id; a definition
    used more than once with no id given is ``shout-1``, ``shout-2``, and so on, skipping an id another node already
    has. An edge is ``(output, input)``, each written ``"node.name"``, and an output may be followed by one field of the
    record it returns. ``constants`` and ``defaults`` are keyed by ``"node.input"``: a constant is the input's value on
    every call, and a default makes the input a graph input, with that value when the caller gives none. Each value is
    written in the record as its JSON reads back.

    With no ``edges`` the nodes form a chain: the one required input of each node that no constant or default fills
    gets an edge from the one output of the node before it. With ``edges``, those are every edge of the graph.
    Nothing is matched by name.

    The record declares the graph's inputs and outputs. Each required input that gets nothing (no edge, no constant)
    is a graph input of the input's name, and so is each key of ``defaults``; each output of a node no edge leaves is a
    graph output of the output's name. ``inputs`` and ``outputs`` name some of them otherwise, on top of that:
    ``inputs={"who": "greet.name"}``, ``inputs={"line": ["strip.line", "shout.line"]}`` for one graph input that
    feeds two inputs, ``outputs={"long": "a.result"}``. Two graph inputs or outputs that would share a name, not
    joined by ``inputs``, are refused. The inputs a hook adds and the outputs a hook returns are known only when the
    graph is compiled, so a graph input or output on one is named in ``inputs`` or ``outputs``.

    The record is written and not compiled: an input or output a node does not have, a value its input's type
    refuses, an edge between types that do not fit and a graph not finished are in
    ``Graph.compile(record, registry).problems``.

    Raises:
        InvalidCallError: Listing every fault in writing the record itself: a place naming a node not given
            (``node_not_in_graph``), a place that is not ``"node.name"`` (``not_a_place``), a value with no JSON
            form or holding a set (``invalid_value``), two graph inputs or two graph outputs of one name
            (``graph_inputs_share_a_name``, ``graph_outputs_share_a_name``), what the record refuses (an input given
            a value and an edge, an edge written twice, one id given to two nodes, a name that is not one:
            ``invalid_record``), and a chain that can't be drawn (``not_a_chain``).
    """
    problems: list[Problem] = []
    definitions = _by_id(nodes, problems)
    ids = list(definitions)
    constant_of = _values_by_place("constants", constants, ids, problems)
    default_of = _values_by_place("defaults", defaults, ids, problems)
    if edges is None:
        edges = _chain_edges(definitions, constant_of.keys() | default_of.keys(), problems)
    sources_of: dict[tuple[str, str], list[Address]] = {}
    for i, (source_text, target_text) in enumerate(edges):
        source, target = _place(source_text, ("edges", i, 0), ids, problems, field=True), _place(target_text, ("edges", i, 1), ids, problems)
        if source is not None and target is not None:
            sources_of.setdefault((target[0], target[1]), []).append(Address(*source))
    graph_input_of = _graph_inputs(definitions, inputs, constant_of.keys() | sources_of.keys(), set(default_of), problems)
    entries: dict[str, dict[str, InputEntry]] = {node_id: {} for node_id in ids}
    for node_id, name in constant_of.keys() | default_of.keys() | sources_of.keys() | graph_input_of.keys():
        place = (node_id, name)
        entries[node_id][name] = InputEntry.model_construct(  # built unchecked: the record door checks it below, an entry of two kinds too
            constant=constant_of.get(place, MISSING), graph_input=graph_input_of.get(place, MISSING), default=default_of.get(place, MISSING),
            sources=tuple(sources_of[place]) if place in sources_of else MISSING,
        )
    declared = _graph_outputs(definitions, outputs, {source.node for found in sources_of.values() for source in found}, problems)
    written = GraphRecord.model_construct(
        nodes={node_id: NodeRecord.model_construct(node=definition.id, version=definition.version, inputs=entries[node_id]) for node_id, definition in definitions.items()},
        outputs=declared,
    )
    try:
        record = read_record(written.model_dump(mode="json"), written_by="make_graph's arguments")
    except RefusedError as refused:  # an input given a value and an edge, an edge given twice, a name that is not one
        problems += refused.problems
    if problems:
        raise InvalidCallError(problems)
    return record


def _place(text: str, where: tuple[str | int, ...], ids: Sequence[str], problems: list[Problem], *, field: bool = False) -> tuple[str, ...] | None:
    """``text``, a place in the graph, as its parts: ``"shout.mark"`` gives ``("shout", "mark")``, and with ``field``,
    ``"read.result.email"`` gives ``("read", "result", "email")``.

    Returns:
        The parts. ``None``, with a problem added to ``problems``, when the text is not ``"node.name"`` (or, with
        ``field``, ``"node.name.field"``; ``not_a_place`` at ``where``), or names a node not given
        (``node_not_in_graph``).
    """
    parts = tuple(text.split(".", 2 if field else 1))
    if len(parts) < 2 or not all(parts):
        problems.append(call_problems.not_a_place(*where, text=text))
        return None
    if parts[0] not in ids:
        problems.append(call_problems.node_not_in_graph(parts[0], nodes=list(ids)))
        return None
    return parts


def _values_by_place(kind: str, given: Mapping[str, Any], ids: Sequence[str], problems: list[Problem]) -> dict[tuple[str, str], Any]:
    """``given``, the ``constants`` or ``defaults`` (``kind``) of ``make_graph``, by place, each value as the record keeps
    it (``record_values``). A fault goes to ``problems``."""
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


def _graph_inputs(
    definitions: Mapping[str, NodeDefinition], named: Mapping[str, str | Sequence[str]], filled: set[tuple[str, str]], defaulted: set[tuple[str, str]], problems: list[Problem],
) -> dict[tuple[str, str], str]:
    """The graph input each input is, by place: those ``named`` gives, then, under the input's own name, every key of
    ``defaults`` (``defaulted``) and every required input that gets nothing else (not in ``filled``).
    ``graph_inputs_share_a_name`` goes to ``problems`` for a name two of the second kind share, or one of them shares
    with a name in ``named``."""
    graph_input_of: dict[tuple[str, str], str] = {}
    for name, texts in named.items():
        for text in [texts] if isinstance(texts, str) else texts:
            place = _place(text, ("inputs", name), list(definitions), problems)
            if place is not None:
                graph_input_of[(place[0], place[1])] = name
    open_inputs = [(node_id, name) for node_id, definition in definitions.items() for name in _required_inputs(definition) if (node_id, name) not in filled]
    automatic = [place for place in dict.fromkeys([*defaulted, *open_inputs]) if place not in graph_input_of]
    counts = Counter([name for _, name in automatic])
    for name, count in counts.items():
        if count > 1 or name in named:
            places = [f"{node_id}.{input_name}" for node_id, input_name in automatic if input_name == name]
            places += [f"{node_id}.{input_name}" for (node_id, input_name), given in graph_input_of.items() if given == name]
            problems.append(call_problems.graph_inputs_share_a_name("inputs", name, name=name, inputs=places))
    return {**graph_input_of, **{(node_id, name): name for node_id, name in automatic}}


def _graph_outputs(definitions: Mapping[str, NodeDefinition], named: Mapping[str, str], led_out: set[str], problems: list[Problem]) -> dict[str, Address]:
    """The graph outputs by name: those ``named`` gives, then every output of a node no edge leaves (not in
    ``led_out``), under the output's own name. ``graph_outputs_share_a_name`` goes to ``problems`` for a name two of the
    second kind share, or one of them shares with a name in ``named``."""
    declared: dict[str, Address] = {}
    for name, text in named.items():
        place = _place(text, ("outputs", name), list(definitions), problems, field=True)
        if place is not None:
            declared[name] = Address(*place)
    taken = {(address.node, address.output) for address in declared.values()}
    leaves = [(node_id, output) for node_id, definition in definitions.items() if node_id not in led_out for output in _outputs_of(definition) if (node_id, output) not in taken]
    counts = Counter(output for _, output in leaves)
    for name, count in counts.items():
        if count > 1 or name in declared:
            places = [f"{node_id}.{output}" for node_id, output in leaves if output == name] + ([str(declared[name])] if name in declared else [])
            problems.append(call_problems.graph_outputs_share_a_name("outputs", name, name=name, outputs=places))
    return {**declared, **{output: Address(node_id, output) for node_id, output in leaves if counts[output] == 1 and output not in declared}}


def _required_inputs(definition: NodeDefinition) -> list[str]:
    """The inputs of ``definition`` a caller must give a value for when nothing else does: a function's inputs with no
    default that no hook reads; a subgraph's graph inputs whose record gives no default."""
    match definition:
        case FunctionDefinition():
            hooked = read_by_hooks(definition)
            return [name for name, field in definition.inputs.items() if field.default is REQUIRED and name not in hooked]
        case SubgraphDefinition():
            names: dict[str, bool] = {}  # each graph input, and whether the record gives it a default
            for node in definition.graph.nodes.values():
                for entry in node.inputs.values():
                    if entry.graph_input is not MISSING:
                        names[entry.graph_input] = names.get(entry.graph_input, False) or entry.default is not MISSING
            return [name for name, defaulted in names.items() if not defaulted]


def _outputs_of(definition: NodeDefinition) -> list[str]:
    """The outputs of ``definition``: a function's, a subgraph's graph outputs."""
    match definition:
        case FunctionDefinition():
            return list(definition.outputs)
        case SubgraphDefinition():
            return list(definition.graph.outputs)


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
            problems.append(call_problems.invalid_record("record", "nodes", node_id, where=f"nodes.{node_id}", reason=reason, error="repeated_key", written_by="make_graph's arguments"))
        taken.add(node_id)
        by_id[node_id] = definition
    return by_id


def _chain_edges(definitions: Mapping[str, NodeDefinition], filled: set[tuple[str, str]], problems: list[Problem]) -> list[tuple[str, str]]:
    """The one edge from each node into the next, given the inputs that already get a constant or a default
    (``filled``); ``not_a_chain`` added to ``problems`` for each pair without exactly one."""
    edges: list[tuple[str, str]] = []
    chain = list(definitions.items())
    for (before_id, before), (after_id, after) in zip(chain, chain[1:], strict=False):
        match before, after:
            case FunctionDefinition(), FunctionDefinition():
                outputs = list(before.outputs)
                open_inputs = [name for name, field in after.inputs.items() if field.default is REQUIRED and (after_id, name) not in filled]
            case _:  # a subgraph's types are known only when it is compiled, so a chain can't be drawn through it
                outputs, open_inputs = [], []
        if len(outputs) != 1 or len(open_inputs) != 1:
            problems.append(call_problems.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs))
        else:
            edges.append((f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"))
    return edges
