"""The flat graph: every function node of a compiled graph and of its subgraphs, in the order a run calls them.

The example used in this file. ``loud`` is the subgraph ``strip -> shout``, used as one node of an outer graph::

    lines(text: str) -> Series[str]     # called once; returns ["a", "b"]
    loud                                # wired to lines.result: its nodes are called once per line
        strip(line: str) -> str
        shout(line: str) -> str
    join(texts: Series[str]) -> str     # wired to loud.shout.result: called once, with ["A!", "B!"]

The compiled graph has three nodes: ``lines``, ``loud`` and ``join``. A subgraph is never called: only the
functions inside it are. So a run works on the flat graph, which ``flat_nodes(graph)`` makes once per graph. It
has the four function nodes ``lines``, ``loud/strip``, ``loud/shout`` and ``join`` (each a
``FlatFunctionNode``), and ``loud`` (a ``FlatSubgraph``), which has no outcomes of its own.

In the flat graph every edge and every loop names a function node:

- The edge into ``join.texts`` comes from ``loud.shout.result`` in the graph record. In the flat graph it reads
  ``result`` of ``loud/shout``. Both kinds of flat node answer ``source_of(output)`` with the function node that
  returns the output, so no code asks which kind of node an edge comes from.
- The input of ``loud/strip`` is a graph input inside the subgraph. In the flat graph it has the input the outer
  graph made for ``loud``'s input: one item of ``lines.result``.
- ``loud/strip`` and ``loud/shout`` are inside the loop over ``lines.result``, because ``loud`` is.

A flat node has nothing from a run in it: it is made once and shared by every run of the graph.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, assert_never

from conductor._internal.base import (
    REQUIRED,
    ItemIndexes,
    Loc,
    NodePath,
    node_path_text,
    split_address,
)
from conductor._internal.compiled.input import (
    CollectingInput,
    ConstantInput,
    EdgeInput,
    EdgeSource,
    GraphInput,
    Input,
    MissingInput,
    OutputAddress,
    OutputInLoops,
)
from conductor._internal.compiled.node import (
    CompiledFunctionNode,
    CompiledSubgraphNode,
)
from conductor._internal.field import Field
from conductor._internal.gate import EdgeDoor, demands_more
from conductor._internal.node import Policy
from conductor._internal.series import Series

if typing.TYPE_CHECKING:
    from conductor._internal.compiled.type import CompiledGraph


@dataclass(eq=False, slots=True, kw_only=True)
class _FlatNodeBase:
    """What both kinds of flat node have. Never made on its own: it is a ``FlatFunctionNode`` or a
    ``FlatSubgraph``.

    A subgraph is compiled once and may be used as a node in many places. A flat node is one node in one of
    those places. In the example, ``loud/shout`` is ``shout`` inside ``loud``.

    Attributes:
        parent: The subgraph the node is directly inside; ``None`` for a node of the graph itself.
        node_path: The ids of the subgraphs the node is inside, outermost first, then its own id:
            ``("loud", "shout")``.
        name: The node path as text, ``"loud/shout"``. A run keeps the node's outcomes under this name.
        loops: The loops the node is inside, outermost first: those of the subgraph it is inside, then its own.
            Each is the function node that returns the list looped over, and the name of that output. In the
            example, ``loud/shout`` has ``((lines, "result"),)``.
    """

    parent: FlatSubgraph | None
    node_path: NodePath
    name: str
    loops: tuple[tuple[FlatFunctionNode, str], ...]


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FlatFunctionNode(_FlatNodeBase):
    """A function node of the flat graph: what a run calls and keeps outcomes of.

    The arguments of one call come from three places: ``constants`` (known when the graph is compiled),
    ``graph_inputs`` (known when a run starts) and ``edge_inputs`` (read from the run's outcomes for each call).
    In every input here, a source names a function node by its ``name``.

    Attributes:
        compiled_node: The node as its own graph compiled it.
        policy: The node's policy, as its definition declares it.
        constants: Every input by name, in the order declared: the value of an input that is a constant, and
            ``None`` for every other input. The arguments of a call start as a copy of it.
        graph_inputs: The inputs the caller gives a value for: the input's name, and the key the run keeps the
            caller's value under (``"lines.text"``).
        edge_inputs: The inputs that edges lead to, in the order declared: the input's name, the compiled input, and
            whether the input accepts ``None``. A collecting input is here twice: first with an input that reads
            the list it collects as one value (so a list that is ``None``, skipped or not made yet is seen
            before any item is read), then as itself.
        lists_from_same_loop: The collecting inputs, grouped by the loop they collect: each input's name and
            whether it accepts ``None`` items. An item that is ``None`` for an input that does not accept it is
            left out of every list of its group, so the lists of one group have the same items.
        edge_door: Reads before each call the inputs whose type demands more than the outputs their edges read
            declare (``gate/edge.py``); ``None`` when no input does.
        function: What a call runs: the node's function, or for a node with an ``Asked`` parameter that function
            with the parameter filled (``AskedParameter.filling``), chosen here so a call makes no choice.
        looped_outputs: The ``Series`` outputs of the node that some edge reads one item at a time, by output
            name. A run holds each to ``max_items``. The value is the node and output that the reading node's
            graph names the list by: this node, or, when the reading node is outside a subgraph this node is
            inside, that subgraph and the output's address in it. The item-limit problem names the list that
            way.
        upstream: Each function node whose outcomes a call of this node reads, with a number ``n``: the call at
            ``item_indexes`` reads that node's outcome at ``item_indexes[:n]``, or, when that node is inside
            more than ``n`` loops, every outcome of it whose item indexes start with those. In the example,
            ``join`` has ``{loud/shout: 0, lines: 0}``: its one call reads every call of ``loud/shout``.
        downstream: Each function node that has this one in its ``upstream``, with the number it has there.
        position: The node's index in the order a run tries the function nodes in.
        all_upstream: Every function node upstream of this one, at any distance. ``None`` until
            ``all_upstream(node)`` works it out.
        display_inputs: The compiled inputs ``run.results`` reads each output with, by output name. Filled by
            ``display_inputs_of``.
    """

    compiled_node: CompiledFunctionNode
    policy: Policy
    constants: dict[str, Any]
    graph_inputs: tuple[tuple[str, str], ...]
    edge_inputs: tuple[tuple[str, EdgeInput | CollectingInput, bool], ...]
    lists_from_same_loop: tuple[tuple[tuple[str, bool], ...], ...]
    edge_door: EdgeDoor | None
    function: Callable[..., Any]
    looped_outputs: dict[str, tuple[FlatNode, str]] = dataclasses.field(default_factory=dict[str, tuple["FlatNode", str]])
    upstream: dict[FlatFunctionNode, int] = dataclasses.field(default_factory=dict["FlatFunctionNode", int])
    downstream: list[tuple[FlatFunctionNode, int]] = dataclasses.field(default_factory=list[tuple["FlatFunctionNode", int]])
    position: int = 0
    all_upstream: frozenset[FlatFunctionNode] | None = None
    display_inputs: dict[str, tuple[EdgeInput | CollectingInput, ...]] = dataclasses.field(default_factory=dict[str, tuple[EdgeInput | CollectingInput, ...]])

    def source_of(self, output: str) -> tuple[FlatFunctionNode, str]:
        """Returns the function node that returns ``output``, and the output's name there: this node and
        ``output``. A ``FlatSubgraph`` answers the same question with a node inside it."""
        return self, output


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FlatSubgraph(_FlatNodeBase):
    """A subgraph in the flat graph: ``loud`` in the example. It is never called and a run keeps no outcomes of
    it. What a run says about it (its results, that it started, that it is complete) is worked out from the
    nodes inside it.

    Attributes:
        compiled_node: The node as the outer graph compiled it.
        inputs: The compiled input of each input, by name. A graph input of the subgraph gets the input here under its
            name: in the example, ``loud/strip``'s ``line`` has the input ``inputs["line"]``.
        inner_nodes: The nodes of the subgraph as flat nodes, by their id in the subgraph. Empty until they are
            made, which is right after this node is.
    """

    compiled_node: CompiledSubgraphNode
    inputs: dict[str, Input]
    inner_nodes: dict[str, FlatNode] = dataclasses.field(default_factory=dict[str, "FlatNode"])

    def source_of(self, output: str) -> tuple[FlatFunctionNode, str]:
        """Returns the function node inside the subgraph that returns ``output``, and the output's name there.

        ``output`` is an address inside the subgraph. In the example, ``loud.source_of("shout.result")`` is
        ``(loud/shout, "result")``. When the node inside is itself a subgraph, the answer is a function node
        inside that one.
        """
        node_id, inner_output = split_address(output)
        return self.inner_nodes[node_id].source_of(inner_output)


#: A node of the flat graph: a function, or a subgraph.
FlatNode = FlatFunctionNode | FlatSubgraph


def all_upstream(node: FlatFunctionNode) -> frozenset[FlatFunctionNode]:
    """Returns every function node upstream of ``node``, at any distance. Worked out on the first call and kept
    on the node, as most nodes are never asked."""
    if node.all_upstream is None:
        seen: set[FlatFunctionNode] = set()
        stack = [node]
        while stack:
            for upstream_node in stack.pop().upstream:
                if upstream_node not in seen:
                    seen.add(upstream_node)
                    stack.append(upstream_node)
        node.all_upstream = frozenset(seen)
    return node.all_upstream


def outcome_loc(node: FlatNode, item_indexes: ItemIndexes) -> Loc:
    """Returns the loc of the call of ``node`` at ``item_indexes``: the ids of its node path, each followed by
    the indexes of the loops entered at that node.

    In the example, the call of ``loud/shout`` for the second line is ``("loud", 1, "shout")``: ``loud`` is
    inside the loop over ``lines.result``, and ``shout`` is inside no further loop.

    Args:
        node: The node.
        item_indexes: The item indexes of the call. With only the indexes of the loops the node's subgraph is
            inside, the loc ends at the node's id: ``("loud", 1, "shout")`` again.
    """
    if node.parent is None:
        return (node.compiled_node.id, *item_indexes)
    parent_depth = len(node.parent.loops)
    return (*outcome_loc(node.parent, item_indexes[:parent_depth]), node.compiled_node.id, *item_indexes[parent_depth:])


#: The input a host's read of an output stands in for: a ``Series`` input that accepts ``None`` items, so every
#: item is shown at its index, with ``None`` for an item that has no value.
_DISPLAY_FIELD = Field("shown", Series[Any | None], REQUIRED)  # type: ignore[misc]  # mypy takes no union as a type argument here


def display_inputs_of(node: FlatFunctionNode, output: str) -> tuple[EdgeInput | CollectingInput, ...]:
    """The compiled inputs ``run.results`` reads one output of a function node with.

    A host reads an output the way a node wired to it would, so the inputs are the ones such a node's input
    would have. Example: ``documents -> lines -> shout``, where ``lines`` is called once per document and
    ``shout`` once per line. ``shout`` is called inside two loops, so its output has two inputs: the first
    lists the documents, the second lists one document's lines and reads ``shout``'s return value for each.
    ``results`` then shows ``[["A", "B"], None, ["C"]]``: a list per document, ``None`` for a document
    whose lines were never made.

    Args:
        node: The function node that returns the output.
        output: The output's name.

    Returns:
        For a node called once, one input that reads the return value. For a node called per item, one input
        for each loop it is called in, outermost first; the input at index ``n`` is used for one item of each
        of the ``n`` loops outside it.
    """
    node_inputs = node.display_inputs.get(output)
    if node_inputs is None:
        source = EdgeSource(node.name, output, _loops_by_name(node.loops), None, node.compiled_node.outputs[output].type, item=False)
        functions = {source_node.name: source_node for source_node in (node, *(series_node for series_node, _ in node.loops))}
        per_loop = [_collecting_input(_DISPLAY_FIELD, looped_over, (source,), functions) for looped_over in node.loops]
        node_inputs = node.display_inputs[output] = tuple(per_loop) or (EdgeInput(_DISPLAY_FIELD, (source,), source.loops),)
    return node_inputs


def flat_nodes(graph: CompiledGraph) -> tuple[dict[str, FlatFunctionNode], dict[str, FlatNode]]:
    """Returns the flat graph of ``graph``: its function nodes by name, in the order a run tries them (each
    after the nodes upstream of it), and every flat node (function or subgraph) by its name."""
    functions: dict[str, FlatFunctionNode] = {}
    nodes_by_name: dict[str, FlatNode] = {}
    _add_flat_nodes(graph, None, functions, nodes_by_name)
    for node in functions.values():
        for upstream_node, shared_indexes in node.upstream.items():
            upstream_node.downstream.append((node, shared_indexes))
    return functions, nodes_by_name


def _add_flat_nodes(
    graph: CompiledGraph, parent: FlatSubgraph | None, functions: dict[str, FlatFunctionNode], nodes_by_name: dict[str, FlatNode],
) -> dict[str, FlatNode]:
    """Makes the flat nodes of ``graph``, and of each subgraph in it, and adds them to ``functions`` and
    ``nodes_by_name``.

    In the example, called with the outer graph it adds ``lines``, then ``loud`` (calling itself with the
    subgraph, which adds ``loud/strip`` and ``loud/shout``), then ``join``.

    Args:
        graph: The compiled graph: the one a run is of, or a subgraph of it.
        parent: The flat node of the subgraph that ``graph`` is; ``None`` for the graph a run is of.
        functions: The function nodes made so far, by name, in the order a run tries them. Added to.
        nodes_by_name: Every flat node made so far, by name. Added to.

    Returns:
        The flat nodes of ``graph`` itself, by their id in ``graph``.
    """
    parent_path = parent.node_path if parent is not None else ()
    outer_loops = parent.loops if parent is not None else ()
    nodes_by_id: dict[str, FlatNode] = {}
    for compiled_node in graph._nodes.values():
        inputs = {name: _flat_input(node_input, graph, parent, nodes_by_id, functions) for name, node_input in compiled_node.inputs.items()}
        loops = (*outer_loops, *[nodes_by_id[loop.node_id].source_of(loop.output) for loop in compiled_node.loops])
        node_path = (*parent_path, compiled_node.id)
        if isinstance(compiled_node, CompiledSubgraphNode):
            subgraph = FlatSubgraph(compiled_node=compiled_node, parent=parent, node_path=node_path, name=node_path_text(node_path), loops=loops, inputs=inputs)
            subgraph.inner_nodes = _add_flat_nodes(compiled_node.inner, subgraph, functions, nodes_by_name)
            nodes_by_id[compiled_node.id] = nodes_by_name[subgraph.name] = subgraph
            continue
        constants: dict[str, Any] = {}
        graph_inputs: list[tuple[str, str]] = []
        edge_inputs: list[tuple[str, EdgeInput | CollectingInput, bool]] = []
        upstream: dict[FlatFunctionNode, int] = {}
        collecting: dict[OutputInLoops, list[tuple[str, bool]]] = {}
        for name, node_input in inputs.items():
            constants[name] = None
            outputs_read: tuple[OutputInLoops, ...]
            outside_the_loop: list[tuple[FlatFunctionNode, int]] = []
            match node_input:
                case ConstantInput(value=value):
                    constants[name] = value
                    continue
                case GraphInput(address=address):
                    graph_inputs.append((name, address))
                    continue
                case MissingInput():
                    raise RuntimeError(f"'{node_path_text(node_path)}' has an input that gets no value: a graph with problems has no flat graph")
                case EdgeInput(sources=sources):
                    outputs_read = sources
                case CollectingInput(field=field, series=series, sources=sources):
                    outputs_read = (series, *sources)
                    series_node = functions[series.node_id]
                    looped_over = (series_node, series.output)
                    # what a node inside the loop reads from outside it can say that no item will have a value
                    read_inside = [read for inner in node_input.loop_body for read in inner.upstream.items()]
                    outside_the_loop = [(read, shared) for read, shared in read_inside if looped_over not in read.loops]
                    collecting.setdefault(series, []).append((name, field.accepts_none_items))
                    series_type = series_node.compiled_node.outputs[series.output].whole
                    whole_list = EdgeSource(series.node_id, series.output, series.loops, None, series_type, item=False)
                    # the list first, then (below) its items
                    edge_inputs.append((name, EdgeInput(field, (whole_list,), series.loops), field.whole.optional))
                case _:
                    assert_never(node_input)
            edge_inputs.append((name, node_input, node_input.field.whole.optional))
            depth = len(node_input.loops)
            for upstream_node, shared_indexes in (*[(functions[read.node_id], min(read.depth, depth)) for read in outputs_read], *outside_the_loop):
                upstream[upstream_node] = min(shared_indexes, upstream.get(upstream_node, shared_indexes))
        read_by_door = {name: node_input.field for name, node_input in inputs.items() if _demands_more(node_input)}
        definition = compiled_node.definition
        node = FlatFunctionNode(
            compiled_node=compiled_node, parent=parent, node_path=node_path, name=node_path_text(node_path), loops=loops, policy=compiled_node.definition.policy,
            constants=constants, graph_inputs=tuple(graph_inputs), edge_inputs=tuple(edge_inputs), lists_from_same_loop=tuple(map(tuple, collecting.values())),
            edge_door=EdgeDoor(read_by_door) if read_by_door else None,
            function=definition.function if definition.asked is None else definition.asked.filling(definition.function),
            upstream=upstream, position=len(functions),
        )
        if loops and min(node.upstream.values(), default=0) < len(loops):
            # it reads something made outside its innermost loop, which may exist before the list looped over does
            node.upstream.setdefault(loops[-1][0], len(loops) - 1)
        nodes_by_id[compiled_node.id] = nodes_by_name[node.name] = functions[node.name] = node
    return nodes_by_id


def _loops_by_name(loops: tuple[tuple[FlatFunctionNode, str], ...]) -> tuple[OutputAddress, ...]:
    """Returns ``loops`` as a compiled input of the flat graph names them: each list by the name of its function node."""
    return tuple([OutputAddress(series_node.name, output) for series_node, output in loops])


def _flat_input(
    node_input: Input, graph: CompiledGraph, parent: FlatSubgraph | None,
    nodes_by_id: Mapping[str, FlatNode], functions: Mapping[str, FlatFunctionNode],
) -> Input:
    """Returns the flat graph's input for an input that its own graph compiled as ``node_input``.

    Every source is changed to name the function node that returns the output, by its name, with the loops
    that function node is inside. In the example, the compiled input ``join.texts`` reads ``loud.shout.result``; the
    flat graph's reads ``result`` of ``loud/shout``. A graph input of a subgraph gets the input the outer graph
    made for it: ``loud/strip``'s ``line`` gets the input of ``loud``'s input ``line``.

    Args:
        node_input: The input as the node's own graph compiled it.
        graph: That graph.
        parent: The flat node of the subgraph that ``graph`` is; ``None`` for the graph a run is of.
        nodes_by_id: The flat nodes of ``graph`` made so far, by id.
        functions: Every function node made so far, by name.
    """

    def flat_source(source: EdgeSource, starts_loop: bool) -> EdgeSource:
        source_node, output = nodes_by_id[source.node_id].source_of(source.output)
        if starts_loop and source.item:
            source_node.looped_outputs.setdefault(output, (nodes_by_id[source.node_id], source.output))
        return EdgeSource(source_node.name, output, _loops_by_name(source_node.loops), source.record_field, source.type, item=source.item)

    match node_input:
        case ConstantInput() | MissingInput():
            return node_input
        case GraphInput(address=address):
            return node_input if parent is None else parent.inputs[graph._input_fields[address].name]
        case EdgeInput(field=field, sources=sources, as_list=as_list):
            flat_sources = tuple([flat_source(source, not as_list) for source in sources])
            return EdgeInput(field, flat_sources, max((source.loops_of_value for source in flat_sources), key=len), as_list=as_list)
        case CollectingInput(field=field, series=series, sources=sources):
            looped_over = nodes_by_id[series.node_id].source_of(series.output)
            return _collecting_input(field, looped_over, tuple([flat_source(source, False) for source in sources]), functions)
        case _:
            assert_never(node_input)


def _collecting_input(field: Field, looped_over: tuple[FlatFunctionNode, str], sources: tuple[EdgeSource, ...], functions: Mapping[str, FlatFunctionNode]) -> CollectingInput:
    """Returns the flat graph's input for an input that collects one value per item of the list
    ``looped_over``. Used for a node's own input and for what ``run.results`` shows.

    Args:
        field: The input's field.
        looped_over: The list, as the function node that returns it and the output's name.
        sources: What each edge reads for one item, each naming a function node by its name.
        functions: The function nodes the sources name, by name.
    """
    series_node, output = looped_over
    series = OutputInLoops(series_node.name, output, _loops_by_name(series_node.loops))
    loop_body = _loop_body_of(looped_over, [functions[source.node_id] for source in sources])
    return CollectingInput(field, series, sources, loop_body)


def _loop_body_of(series: tuple[FlatFunctionNode, str], sources: list[FlatFunctionNode]) -> tuple[FlatFunctionNode, ...]:
    """Returns the function nodes inside the loop over ``series`` that a collecting input's value depends on.

    In the example, ``join.texts`` collects the loop over ``lines.result`` and reads ``loud/shout``, which
    reads ``loud/strip``: the result is ``(loud/strip, loud/shout)``.

    Args:
        series: The list the input collects the loop over: the function node that returns it, and the output.
        sources: The function nodes the input's edges come from.

    Returns:
        The function nodes, in the order a run tries them: each after the nodes upstream of it.
    """
    loop_body: set[FlatFunctionNode] = set()
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.loops and inner not in loop_body:
            loop_body.add(inner)
            stack.extend(inner.upstream)
    return tuple(sorted(loop_body, key=lambda inner: inner.position))


def _demands_more(node_input: Input) -> bool:
    """Whether the input of ``node_input`` is read by its edge door before each call: an input that edges lead to, whose
    type demands more than the outputs they read declare (``demands_more``)."""
    match node_input:
        case ConstantInput() | GraphInput() | MissingInput():
            return False
        case EdgeInput(field=field, sources=sources, as_list=as_list):
            return demands_more(field, [source.type for source in sources], per_item=as_list)
        case CollectingInput(field=field, sources=sources):
            return demands_more(field, [source.type for source in sources], per_item=True)
        case _:
            assert_never(node_input)
