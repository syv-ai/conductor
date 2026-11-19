"""``make_graph``, which writes a graph record from Python.

Writing a record by hand means spelling out every node, input entry and source. With ``make_graph`` you list the
nodes, and the edges if they aren't a simple chain. It fills in the rest: the edges of a chain, and the graph's
inputs and outputs.

This is the one place in conductor that reads a place in a graph from dotted text. ``"lines.result"`` is the output
``result`` of the node ``lines``. ``"shout.mark"`` is the input ``mark`` of ``shout``. ``"read.result.email"`` is
the field ``email`` of the value that ``read`` returns. No name may have a dot in it, so the text is cut at its
first two dots, and nothing has to be guessed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence, Set
from types import MappingProxyType
from typing import Any

from pydantic import MISSING, JsonValue

from conductor._internal.base import (
    FIELD_SEPARATOR,
    REQUIRED,
    Source,
    canonical_json,
    is_field,
    is_name,
    place_text,
)
from conductor._internal.node import FunctionDefinition, NodeDefinition
from conductor._internal.problem import (
    InvalidCallError,
    Problem,
    call_problems,
    conductor_error,
    shape_error,
)
from conductor._internal.record import (
    GraphRecord,
    InputEntry,
    NodeRecord,
    RecordValidator,
    WrittenValuesValidator,
)

from .validator import ArgumentsValidator


def make_graph(
    *nodes: NodeDefinition | tuple[str, NodeDefinition],
    edges: Sequence[tuple[str, str]] | None = None,
    constants: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
    inputs: Mapping[str, str | Sequence[str]] = MappingProxyType({}),
    outputs: Mapping[str, str] = MappingProxyType({}),
) -> GraphRecord:
    """Writes a graph record from Python, so you don't have to write its JSON by hand.

    You pass the nodes, and it uses each one at its own version. Here are a few ways to call it::

        make_graph(lines, shout, join)                                # a chain
        make_graph(lines, shout, edges=[("lines.result", "shout.line")])
        make_graph(lines, shout, join, constants={"shout.mark": "?"})
        make_graph(read, send, edges=[("read.result.email", "send.to")])     # the field email of read's result
        make_graph(("a", shout), ("b", shout), edges=[("a.result", "b.line")])

    Node ids. A node's id in the graph is the one given with it, as in ``("a", shout)``, or else its definition's
    id. A definition used more than once with no id given becomes ``shout-1``, ``shout-2`` and so on, skipping any
    id another node already has.

    Edges. An edge is ``(output, input)``, each written ``"node.name"``. The output may be followed by one field of
    the value it returns. With no ``edges``, the nodes form a chain. Each node's one required input that an edge
    can feed, and that no constant or default fills, gets an edge from the one output of the node before it. With
    ``edges``, those are all the edges. Nothing is ever matched by name.

    Constants and defaults. Both are keyed by ``"node.input"``. A constant is the input's value on every call. A
    default makes the input a graph input, with that value when the caller gives none. Each value is written into
    the record as JSON.

    Graph inputs and outputs. The record declares them for you. Each required input that gets nothing (no edge, no
    constant) becomes a graph input with the input's name, unless it's ``constant_only``. So does each key of
    ``defaults``. Each output of a node that no edge leaves becomes a graph output with the output's name. A node
    that any edge leaves gives none, not even an output no edge reads, such as a branch left unwired.
    ``inputs`` and ``outputs`` let you name some yourself, on top of those: ``inputs={"who": "greet.name"}``,
    ``inputs={"line": ["strip.line", "shout.line"]}`` for one graph input that feeds two inputs, or
    ``outputs={"long": "a.result"}``. An optional input named in ``inputs`` stays optional: its graph input takes
    the input's own default, or, for one graph input that feeds several, their default when they all declare the
    same one. Two graph inputs or outputs that would end up with one name are refused,
    unless ``inputs`` joins them. Inputs a hook adds and outputs a hook returns aren't known until the graph is
    compiled, so a graph input or output on one must be named in ``inputs`` or ``outputs``. A node with a
    ``compute_outputs`` hook that no edge leaves, and none of whose outputs ``outputs`` names, is refused.

    ``make_graph`` never wires a ``constant_only`` input or makes it a graph input by itself. If ``edges``,
    ``defaults`` or ``inputs`` do, they're written as given, and compiling the record reports
    ``input_takes_only_a_constant``.

    The record is written, not compiled. An input or output a node doesn't have, a value its type refuses, an edge
    between types that don't fit, or an unfinished graph shows up later, in
    ``Graph.compile(record, registry).problems``.

    Raises:
        InvalidCallError: ``invalid_argument``, one for each argument that isn't of the type its annotation says,
            such as ``outputs=None``, before anything else is checked. Otherwise it lists every fault in writing the
            record:
            ``node_not_in_graph``: a place names a node that wasn't given.
            ``invalid_place``: a place isn't written ``"node.name"``.
            ``invalid_value``: a value has no JSON form, or has a set in it.
            ``graph_inputs_share_a_name`` or ``graph_outputs_share_a_name``: two graph inputs or outputs have one
            name.
            ``invalid_record``: the record itself is refused. For example, an input is given both a value and an
            edge, an edge is written twice, one id is given to two nodes, or a name isn't a valid name.
            ``not_a_chain``: there are no ``edges``, and the nodes can't be chained. The automatic graph
            inputs, and the outputs of every node but the last, aren't made then, since they would follow from the
            gap, so a clash among them isn't listed; ``inputs``, ``outputs`` and ``defaults`` are still checked.
            An edge that can't be written, such as one that names a node that wasn't given, leaves out every
            automatic graph input and output the same way.
            ``hook_outputs_not_named``: no edge leaves a node whose ``compute_outputs`` hook gives its outputs,
            and ``outputs`` names none of them.
    """
    arguments = ArgumentsValidator()(
        {
            "nodes": nodes,
            "edges": edges,
            "constants": constants,
            "defaults": defaults,
            "inputs": inputs,
            "outputs": outputs,
        }
    )
    problems: list[Problem] = []
    definitions = _by_id(arguments["nodes"], problems)
    constant_of = _values_by_place("constants", arguments["constants"], definitions, problems)
    default_of = _values_by_place("defaults", arguments["defaults"], definitions, problems)
    # A chain with a gap, or an edge that can't be written, leaves inputs unwired and outputs unread. The automatic
    # graph inputs and outputs made from those would clash for that alone. So no automatic inputs are made then,
    # and automatic outputs only for the last node of a chain; inputs=, outputs= and defaults= are still checked.
    drawn = arguments["edges"]
    chained, gap = drawn is None, False
    if drawn is None:
        found_before_the_chain = len(problems)
        drawn = _chain_edges(definitions, constant_of.keys() | default_of.keys(), problems)
        gap = len(problems) > found_before_the_chain
    sources_of: dict[tuple[str, str], list[Source]] = {}
    for i, (source_text, target_text) in enumerate(drawn):
        source, target = (
            _source(source_text, ("edges", i, 0), definitions, problems),
            _place(target_text, ("edges", i, 1), definitions, problems),
        )
        if source is not None and target is not None:
            sources_of.setdefault((target[0], target[1]), []).append(source)
        else:
            gap = True
    graph_input_of = _graph_inputs(
        definitions,
        arguments["inputs"],
        constant_of.keys() | sources_of.keys(),
        default_of.keys(),
        gap,
        problems,
    )
    default_of |= _own_defaults(definitions, graph_input_of, arguments["inputs"], default_of.keys())
    given = dict.fromkeys([*constant_of, *default_of, *sources_of, *graph_input_of])
    declared_inputs = [
        (node_id, name)
        for node_id, definition in definitions.items()
        for name in definition.input_names
    ]
    declared_at = {place: position for position, place in enumerate(declared_inputs)}
    entries: dict[str, dict[str, InputEntry]] = {node_id: {} for node_id in definitions}
    # each node's inputs in the order the node declares them, then any it doesn't declare, such as one a hook adds,
    # in the order they are gathered here: constants, defaults, edges, then graph inputs
    for place in sorted(given, key=lambda place: declared_at.get(place, len(declared_at))):
        node_id, name = place
        entries[node_id][name] = (
            InputEntry.model_construct(  # built unchecked; the record validator below checks it, an entry of two kinds too
                constant=constant_of.get(place, MISSING),
                graph_input=graph_input_of.get(place, MISSING),
                default=default_of.get(place, MISSING),
                sources=tuple(sources_of[place]) if place in sources_of else MISSING,
            )
        )
    led_out = {source.node for found in sources_of.values() for source in found}
    # A chain leads out of every node but its last, so only the last one's outputs are the graph's. An edge that
    # can't be written may have led out of any node, so then no node's are.
    if gap:
        led_out |= set(list(definitions)[:-1] if chained else definitions)
    declared = _graph_outputs(definitions, arguments["outputs"], led_out, problems)
    written = GraphRecord.model_construct(
        nodes={
            node_id: NodeRecord.model_construct(
                node=definition.id, version=definition.version, inputs=entries[node_id]
            )
            for node_id, definition in definitions.items()
        },
        outputs=declared,
    )
    try:
        record = RecordValidator(written_by="make_graph's arguments")(
            written.model_dump(mode="json")
        )
    # such as an input given both a value and an edge, an edge given twice, or a name that isn't valid
    except InvalidCallError as refused:
        problems += refused.problems
    if problems:
        raise InvalidCallError(problems)
    return record


def _place(
    text: str,
    where: tuple[str | int, ...],
    definitions: Mapping[str, NodeDefinition],
    problems: list[Problem],
    *,
    field: bool = False,
) -> tuple[str, ...] | None:
    """Splits ``text``, a place in the graph, into its parts.

    ``"shout.mark"`` gives ``("shout", "mark")``. With ``field``, ``"read.result.email"`` gives
    ``("read", "result", "email")``. The node id and the name must each be a name (``is_name``), and a field must
    be one (``is_field``), so ``"greet.result~"`` isn't a place, since a name can't contain ``~``.

    This is where ``make_graph`` reads its dotted text, so it checks the parts here rather than in a validator: the
    record validator would see only the parts, and could not say which argument was written wrong.

    Returns:
        The parts, or ``None`` when the text isn't a valid place. Then a problem is added to ``problems``. It's
        ``invalid_place`` at ``where`` when the text isn't ``"node.name"`` (with ``field``, it may also be
        ``"node.name.field"``). It's ``node_not_in_graph`` when the text names a node that wasn't given.
    """
    parts = tuple(text.split(FIELD_SEPARATOR, 2 if field else 1))
    if len(parts) < 2 or not all(map(is_name, parts[:2])) or not all(map(is_field, parts[2:])):
        problems.append(call_problems.invalid_place(*where, text=text))
        return None
    if parts[0] not in definitions:
        problems.append(call_problems.node_not_in_graph(parts[0], nodes=list(definitions)))
        return None
    return parts


def _source(
    text: str,
    where: tuple[str | int, ...],
    definitions: Mapping[str, NodeDefinition],
    problems: list[Problem],
) -> Source | None:
    """Reads ``text`` as where an edge or a graph output gets its value, as ``_place`` reads it with a field.

    ``"read.result.email"`` gives ``Source("read", "result", "email")``. Returns ``None`` when ``_place`` refuses
    the text, with the problem added to ``problems``.
    """
    parts = _place(text, where, definitions, problems, field=True)
    return None if parts is None else Source(*parts)


def _values_by_place(
    kind: str,
    given: Mapping[str, Any],
    definitions: Mapping[str, NodeDefinition],
    problems: list[Problem],
) -> dict[tuple[str, str], Any]:
    """Returns the ``constants`` or ``defaults`` given to ``make_graph``, keyed by place.

    ``kind`` says which of the two ``given`` is. ``WrittenValuesValidator`` converts each value to the JSON the
    record stores. Faults go to ``problems``.
    """
    by_node: dict[str, dict[str, Any]] = {}
    for key, value in given.items():
        place = _place(key, (kind, key), definitions, problems)
        if place is not None:
            by_node.setdefault(place[0], {})[place[1]] = value
    written: dict[tuple[str, str], Any] = {}
    for node_id, named in by_node.items():
        try:
            written |= {
                (node_id, name): value
                for name, value in WrittenValuesValidator(node_id)(named).items()
            }
        except InvalidCallError as refused:
            problems += refused.problems
    return written


def _graph_inputs(
    definitions: Mapping[str, NodeDefinition],
    named: Mapping[str, Sequence[str]],
    filled: Set[tuple[str, str]],
    defaulted: Set[tuple[str, str]],
    gap: bool,
    problems: list[Problem],
) -> dict[tuple[str, str], str]:
    """Returns the graph input each input becomes, keyed by place.

    First come the ones ``named`` gives. Then, under the input's own name, every key of ``defaults``
    (``defaulted``), and every required input that gets nothing else (isn't in ``filled``), unless there is a
    ``gap``: a chain that can't be drawn or an edge that can't be written, which leaves such inputs unwired.

    If two of those automatic graph inputs share a name, or one shares a name with a graph input in ``named``,
    ``graph_inputs_share_a_name`` goes to ``problems``.
    """
    graph_input_of: dict[tuple[str, str], str] = {}
    for name, texts in named.items():
        for text in texts:
            place = _place(text, ("inputs", name), definitions, problems)
            if place is not None:
                graph_input_of[(place[0], place[1])] = name
    open_inputs = [
        (node_id, name)
        for node_id, definition in definitions.items()
        for name in definition.required_inputs
        if not gap and (node_id, name) not in filled
    ]
    automatic = [
        place for place in dict.fromkeys([*defaulted, *open_inputs]) if place not in graph_input_of
    ]
    counts = Counter([name for _, name in automatic])
    for name, count in counts.items():
        if count > 1 or name in named:
            places = [
                place_text((node_id,), input_name)
                for node_id, input_name in automatic
                if input_name == name
            ]
            places += [
                place_text((node_id,), input_name)
                for (node_id, input_name), given in graph_input_of.items()
                if given == name
            ]
            problems.append(
                call_problems.graph_inputs_share_a_name("inputs", name, name=name, inputs=places)
            )
    return {**graph_input_of, **{(node_id, name): name for node_id, name in automatic}}


def _own_defaults(
    definitions: Mapping[str, NodeDefinition],
    graph_input_of: Mapping[tuple[str, str], str],
    named: Mapping[str, Sequence[str]],
    defaulted: Set[tuple[str, str]],
) -> dict[tuple[str, str], JsonValue]:
    """Returns the defaults the graph inputs ``named`` in ``inputs=`` take from their node's own inputs, by place.

    Naming an input in ``inputs=`` gives it a graph input of another name, and keeps it as optional as it was.
    ``shout(text, mark: str = "!")`` with ``inputs={"m": "shout.mark"}`` has the graph input ``m``, with the
    default ``"!"``. A graph input that feeds several inputs takes their default when each of them declares the
    same one, and is required otherwise. One that ``defaults=`` gives a default on any of its inputs keeps that.
    """
    inputs_of: dict[str, list[tuple[str, str]]] = {}
    for place, name in graph_input_of.items():
        if name in named:
            inputs_of.setdefault(name, []).append(place)
    found: dict[tuple[str, str], JsonValue] = {}
    for places in inputs_of.values():
        own = [_own_default(definitions[node_id], input_name) for node_id, input_name in places]
        declared = [default for default in own if default is not MISSING]
        if (
            defaulted.isdisjoint(places)
            and len(declared) == len(places)
            and len({canonical_json(default) for default in declared}) == 1
        ):
            found |= {place: declared[0] for place in places}
    return found


def _own_default(definition: NodeDefinition, name: str) -> JsonValue | MISSING:
    """Returns the default that the input ``name`` of a node declares, as JSON, or ``MISSING`` when it has none.

    A function's input declares it in the function's signature. A subgraph's input is a graph input of its record,
    and its default is the one its record gives. An input a hook adds isn't known before the graph is compiled,
    so it has none here.
    """
    if isinstance(definition, FunctionDefinition):
        field = definition.inputs.get(name)
        return MISSING if field is None else field.default_json
    return next(
        (
            entry.default
            for node_record in definition.graph.nodes.values()
            for entry in node_record.inputs.values()
            if entry.graph_input == name and entry.default is not MISSING
        ),
        MISSING,
    )


def _graph_outputs(
    definitions: Mapping[str, NodeDefinition],
    named: Mapping[str, str],
    led_out: set[str],
    problems: list[Problem],
) -> dict[str, Source]:
    """Returns the graph outputs, by name.

    First come the ones ``named`` gives. Then every output of a node that no edge leaves (isn't in ``led_out``),
    under the output's own name.

    A node whose ``compute_outputs`` hook gives its outputs has none until the graph is compiled. So when no edge
    leaves it and ``named`` names none of its outputs, ``hook_outputs_not_named`` goes to ``problems``.

    If two of those automatic graph outputs share a name, or one shares a name with an output in ``named``,
    ``graph_outputs_share_a_name`` goes to ``problems``.
    """
    declared: dict[str, Source] = {}
    for name, text in named.items():
        source = _source(text, ("outputs", name), definitions, problems)
        if source is not None:
            declared[name] = source
    named_nodes = {source.node for source in declared.values()}
    for node_id, definition in definitions.items():
        if (
            isinstance(definition, FunctionDefinition)
            and definition.compute_outputs is not None
            and node_id not in led_out | named_nodes
        ):
            problems.append(call_problems.hook_outputs_not_named(node_id))
    taken = {(source.node, source.output) for source in declared.values()}
    leaves = [
        (node_id, output)
        for node_id, definition in definitions.items()
        if node_id not in led_out
        for output in definition.output_names
        if (node_id, output) not in taken
    ]
    counts = Counter(output for _, output in leaves)
    for name, count in counts.items():
        if count > 1 or name in declared:
            places = [
                place_text((node_id,), output) for node_id, output in leaves if output == name
            ] + ([str(declared[name])] if name in declared else [])
            problems.append(
                call_problems.graph_outputs_share_a_name("outputs", name, name=name, outputs=places)
            )
    return {
        **declared,
        **{
            output: Source(node_id, output)
            for node_id, output in leaves
            if counts[output] == 1 and output not in declared
        },
    }


def _by_id(
    nodes: Sequence[NodeDefinition | tuple[str, NodeDefinition]], problems: list[Problem]
) -> dict[str, NodeDefinition]:
    """Returns each node's definition, keyed by its id in the graph.

    The id is the one given with the node, or else its definition's id. A definition used more than once with no id
    given is numbered ``-1``, ``-2`` and so on. One id given to two nodes goes to ``problems``.
    """
    given = [node[0] for node in nodes if isinstance(node, tuple)]
    bare = Counter(node.id for node in nodes if not isinstance(node, tuple))
    taken = {*given, *(definition_id for definition_id, count in bare.items() if count == 1)}
    numbered: Counter[str] = Counter()
    by_id: dict[str, NodeDefinition] = {}
    for node in nodes:
        node_id, definition = node if isinstance(node, tuple) else (node.id, node)
        while (
            not isinstance(node, tuple)
            and bare[definition.id] > 1
            and (node_id == definition.id or node_id in taken)
        ):
            numbered[definition.id] += 1
            node_id = f"{definition.id}-{numbered[definition.id]}"
        if node_id in by_id:
            fault = shape_error(
                conductor_error(
                    "repeated_key", "the id '{id}' is given to two nodes", {"id": node_id}
                )
            )
            problems.append(
                call_problems.invalid_record(
                    "record",
                    "nodes",
                    node_id,
                    where=f"nodes.{node_id}",
                    reason=fault["msg"],
                    errors=[fault],
                    written_by="make_graph's arguments",
                )
            )
        taken.add(node_id)
        by_id[node_id] = definition
    return by_id


def _chain_edges(
    definitions: Mapping[str, NodeDefinition], filled: set[tuple[str, str]], problems: list[Problem]
) -> list[tuple[str, str]]:
    """Returns the edges that chain the nodes, one from each node into the next.

    ``filled`` is the inputs that already get a constant or a default. Between two nodes, the edge goes from the
    first node's one output into the second node's one required input that's left. A subgraph is chained the same
    way, by the graph inputs and outputs its record declares. If two nodes can't be joined
    that way, ``not_a_chain`` goes to ``problems``. ``not_a_chain_constant_only`` is used when the only required
    inputs left take only a constant.
    """
    edges: list[tuple[str, str]] = []
    chain = list(definitions.items())
    for (before_id, before), (after_id, after) in zip(chain, chain[1:], strict=False):
        outputs = list(before.output_names)
        open_inputs = [name for name in after.required_inputs if (after_id, name) not in filled]
        # a subgraph's inputs are graph inputs of its record, and a graph input never takes only a constant
        needs_constant = (
            [
                name
                for name, field in after.inputs.items()
                if field.param.constant_only
                and field.default is REQUIRED
                and (after_id, name) not in filled
            ]
            if isinstance(after, FunctionDefinition)
            else []
        )
        if len(outputs) == 1 and len(open_inputs) == 1:
            edges.append(
                (place_text((before_id,), outputs[0]), place_text((after_id,), open_inputs[0]))
            )
        elif not open_inputs and needs_constant:
            problems.append(
                call_problems.not_a_chain_constant_only(
                    after_id, before=before_id, inputs=needs_constant
                )
            )
        else:
            problems.append(
                call_problems.not_a_chain(
                    after_id, before=before_id, inputs=open_inputs, outputs=outputs
                )
            )
    return edges
