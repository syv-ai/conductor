"""The flat graph: every function node of a compiled graph and of its subgraphs, in the order a run tries them,
with what each call reads through edges.

The example used in this file. ``loud`` is the subgraph ``strip -> shout``, used as one node of an outer graph::

    lines(text: str) -> Series[str]     # called once; returns ["a", "b"]
    loud                                # wired to lines.result: its nodes are called once per line
        strip(line: str) -> str
        shout(line: str) -> str
    join(texts: Series[str]) -> str     # wired to loud.shout.result: called once, with ["A!", "B!"]

The compiled graph has three nodes: ``lines``, ``loud`` and ``join``. A subgraph is never called: only the
functions inside it are. So a run works on the flat graph, which is built on the first run of the graph and kept by
the ``Graph`` (``graph._flat``). It has the four function nodes ``lines``, ``loud/strip``, ``loud/shout`` and ``join`` (each a
``FlatFunctionNode``), and ``loud`` (a ``FlatSubgraph``), which has no outcomes of its own.

In the flat graph every edge and every loop names a function node:

- The edge into ``join.texts`` comes from ``loud.shout.result`` in the graph record. In the flat graph it reads
  ``result`` of the function node at ``("loud", "shout")``. Both kinds of flat node answer ``source_of(output)`` with the function node that
  returns the output, so no code asks which kind of node an edge comes from.
- The input of ``loud/strip`` is a graph input inside the subgraph. In the flat graph it has the input the outer
  graph made for ``loud``'s input: one item of ``lines.result``.
- ``loud/strip`` and ``loud/shout`` are inside the loop over ``lines.result``, because ``loud`` is.

A call reads an input that edges lead to through an ``EdgeInput`` (one value per call) or a ``CollectingArgument``
(one value per item of a loop, as a list), whose edges and loops the flat graph names as above: ``join.texts`` reads
``result`` of ``("loud", "shout")``. ``ResultReader`` (``arguments.py``) reads it from the run's outcomes, which a
run keeps by each function node's path.

A flat node has nothing from a run in it: it is made once and shared by every run of the graph.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

from conductor._internal.base import (
    REQUIRED,
    ItemIndexes,
    Loc,
    NodePath,
    place_text,
)
from conductor._internal.compiled import (
    CollectingInput,
    CompiledFunctionNode,
    CompiledGraph,
    CompiledSubgraphNode,
    ConstantInput,
    Edge,
    EdgeInput,
    GraphInput,
    Input,
    MissingInput,
)
from conductor._internal.field import Field
from conductor._internal.gate import EdgeDoor, demands_more
from conductor._internal.node import Policy
from conductor._internal.series import Series


@typing.final
@dataclass(frozen=True, slots=True)
class CollectingArgument:
    """How a call reads an input that collects a loop, one value per item, as a list: ``join.texts`` collects the
    loop over ``lines.result`` and reads ``shout.result`` for each item, so its one call gets ``["A!", "B!"]``.

    It is a class of its own, beside ``EdgeInput``, for one fact: the function nodes inside the loop, which say
    whether any item can have a value. Those are flat nodes, made once per use of a subgraph, and a compiled
    ``CollectingInput`` is made once per compiled graph, so it cannot hold them; a map of them by input name beside
    the node's inputs would be a second structure kept in step with the first.

    Attributes:
        input: The input as the flat graph names its list, loops and edges: the node's own field, or for a graph
            input of a subgraph the field of the subgraph node's input it is read as.
        loop_body: The function nodes inside the loop that the value depends on, each after the nodes upstream
            of it: ``(shout,)``.
    """

    input: CollectingInput
    loop_body: tuple[FlatFunctionNode, ...]

    @property
    def field(self) -> Field:
        """The input's field, so a call reads it from either kind of input the same way."""
        return self.input.field


@dataclass(eq=False, slots=True, kw_only=True)
class _FlatNodeBase:
    """What both kinds of flat node have. Never made on its own: it is a ``FlatFunctionNode`` or a
    ``FlatSubgraph``.

    A subgraph is compiled once and may be used as a node in many places. A flat node is one node in one of
    those places. In the example, ``loud/shout`` is ``shout`` inside ``loud``.

    Attributes:
        parent: The subgraph the node is directly inside; ``None`` for a node of the graph itself.
        node_path: The ids of the subgraphs the node is inside, outermost first, then its own id:
            ``("loud", "shout")``. A run keeps the node's outcomes under it.
        loops: The loops the node is inside, outermost first: those of the subgraph it is inside, then its own.
            Each is the path of the function node that returns the list looped over, and the name of that output.
            In the example, ``loud/shout`` has ``((("lines",), "result"),)``. The function node of the loop at
            index ``n`` is inside the first ``n`` loops.
    """

    parent: FlatSubgraph | None
    node_path: NodePath
    loops: tuple[tuple[NodePath, str], ...]


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FlatFunctionNode(_FlatNodeBase):
    """A function node of the flat graph: what a run calls and keeps outcomes of.

    The arguments of one call come from three places: ``constants`` (known when the graph is compiled),
    ``graph_inputs`` (known when a run starts) and ``edge_inputs`` (read from the run's outcomes for each call).
    In every input here, an edge names the function node it comes from by its path.

    Attributes:
        compiled_node: The node as its own graph compiled it.
        policy: The node's policy, as its definition declares it.
        constants: Every input by name, in the order declared: the value of an input that is a constant, and
            ``None`` for every other input. The arguments of a call start as a copy of it.
        graph_inputs: The inputs the caller gives a value for: the input's name, and the name of the graph input
            whose value it is (``"text"``).
        edge_inputs: The inputs that edges lead to, by name, in the order declared: an ``EdgeInput`` for an
            input that reads one value per call, a ``CollectingArgument`` for one that collects a loop.
        lists_from_same_loop: The collecting inputs, grouped by the loop they collect: each input's name and
            whether it accepts ``None`` items. An item that is ``None`` for an input that does not accept it is
            left out of every list of its group, so the lists of one group have the same items.
        edge_door: Reads before each call the inputs whose type demands more than the outputs their edges read
            declare (``gate/edge.py``); ``None`` when no input does.
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
        display_inputs: The arguments ``run.results`` reads each output with, by output name. Filled by
            ``display_inputs_of``.
    """

    compiled_node: CompiledFunctionNode
    policy: Policy
    constants: dict[str, Any]
    graph_inputs: tuple[tuple[str, str], ...]
    edge_inputs: dict[str, EdgeInput | CollectingArgument]
    lists_from_same_loop: tuple[tuple[tuple[str, bool], ...], ...]
    edge_door: EdgeDoor | None
    looped_outputs: dict[str, tuple[FlatNode, str]] = dataclasses.field(default_factory=dict[str, tuple["FlatNode", str]])
    upstream: dict[FlatFunctionNode, int] = dataclasses.field(default_factory=dict["FlatFunctionNode", int])
    downstream: list[tuple[FlatFunctionNode, int]] = dataclasses.field(default_factory=list[tuple["FlatFunctionNode", int]])
    position: int = 0
    all_upstream: frozenset[FlatFunctionNode] | None = None
    display_inputs: dict[str, tuple[EdgeInput | CollectingArgument, ...]] = dataclasses.field(default_factory=dict[str, tuple[EdgeInput | CollectingArgument, ...]])

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
        inner_nodes: The nodes of the subgraph as flat nodes, by their id in the subgraph. Empty until they are
            made, which is right after this node is.
    """

    compiled_node: CompiledSubgraphNode
    inner_nodes: dict[str, FlatNode] = dataclasses.field(default_factory=dict[str, "FlatNode"])

    def source_of(self, output: str) -> tuple[FlatFunctionNode, str]:
        """Returns the function node inside the subgraph that returns the graph output ``output``, and the output's
        name there.

        In the example, where ``loud``'s record declares the graph output ``result`` as ``shout.result``,
        ``loud.source_of("result")`` is ``(loud/shout, "result")``. When the node inside is itself a subgraph, the
        answer is a function node inside that one.
        """
        source = self.compiled_node.inner._outputs[output]
        return self.inner_nodes[source.node[0]].source_of(source.output)


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


#: The input a host's read of an output stands in for, compiled as if a node's: a ``Series`` input that accepts ``None`` items, so every
#: item is shown at its index, with ``None`` for an item that has no value.
_DISPLAY_FIELD = Field("shown", Series[Any | None], REQUIRED)  # type: ignore[misc]  # mypy takes no union as a type argument here


def display_inputs_of(node: FlatFunctionNode, output: str) -> tuple[EdgeInput | CollectingArgument, ...]:
    """The arguments ``run.results`` reads one output of a function node with.

    A host reads an output the way a node wired to it would, so the arguments are the ones such a node's input
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
        source = Edge(node.node_path, output, None, node.loops, node.compiled_node.outputs[output].type, series=None)
        per_loop = [
            CollectingArgument(CollectingInput(_DISPLAY_FIELD, loop, node.loops[:depth], (source,)), _loop_body_of(loop, [node]))
            for depth, loop in enumerate(node.loops)
        ]
        node_inputs = node.display_inputs[output] = tuple(per_loop) or (EdgeInput(_DISPLAY_FIELD, (source,), source.loops),)
    return node_inputs


@dataclass(frozen=True, eq=False)
class FlatGraph:
    """The flat graph of one compiled graph: what every run of it works on. ``FlatGraph.of(graph)`` builds it on the
    graph's first run, and the ``Graph`` keeps it (``graph._flat``). A first run on two threads at once may build two
    equal flat graphs; one is kept.

    Attributes:
        function_nodes: Every function node of the graph and of its subgraphs, at any depth, by node path
            (``("loud", "shout")``), in the order a run tries them: each after the nodes upstream of it.
        nodes: Every flat node, a function or a subgraph, by node path (``("loud",)``, ``("loud", "shout")``).
    """

    function_nodes: dict[NodePath, FlatFunctionNode]
    nodes: dict[NodePath, FlatNode]

    @functools.cached_property
    def threads(self) -> int:
        """How many calls can run on threads at once during one call of ``resume()``: the sum of the
        ``concurrency`` of every function node called on a thread."""
        return sum(node.policy.concurrency for node in self.function_nodes.values() if node.policy.threaded)

    @classmethod
    def of(cls, graph: CompiledGraph) -> FlatGraph:
        """Returns the flat graph of ``graph``, a graph with no problems."""
        functions: dict[NodePath, FlatFunctionNode] = {}
        nodes: dict[NodePath, FlatNode] = {}
        _add_flat_nodes(graph, None, None, functions, nodes)
        for node in functions.values():
            for upstream_node, shared_indexes in node.upstream.items():
                upstream_node.downstream.append((node, shared_indexes))
        return cls(functions, nodes)


def _add_flat_nodes(
    graph: CompiledGraph, parent: FlatSubgraph | None, outer_inputs: Mapping[str, ConstantInput | GraphInput | EdgeInput | CollectingArgument] | None,
    functions: dict[NodePath, FlatFunctionNode], nodes: dict[NodePath, FlatNode],
) -> dict[str, FlatNode]:
    """Makes the flat nodes of ``graph``, and of each subgraph in it, and adds them to ``functions`` and
    ``nodes``.

    In the example, called with the outer graph it adds ``lines``, then ``loud`` (calling itself with the
    subgraph, which adds ``loud/strip`` and ``loud/shout``), then ``join``.

    Args:
        graph: The compiled graph: the one a run is of, or a subgraph of it.
        parent: The flat node of the subgraph that ``graph`` is; ``None`` for the graph a run is of.
        outer_inputs: For a subgraph, how each input of its node gets its value, by name: a graph input of the
            subgraph is read the same way (``loud/strip``'s ``line`` as ``loud``'s ``line``). ``None`` for the graph
            a run is of, whose graph inputs the caller gives.
        functions: The function nodes made so far, by node path, in the order a run tries them. Added to.
        nodes: Every flat node made so far, by node path. Added to.

    Returns:
        The flat nodes of ``graph`` itself, by their id in ``graph``.
    """
    parent_path = parent.node_path if parent is not None else ()
    outer_loops = parent.loops if parent is not None else ()
    nodes_by_id: dict[str, FlatNode] = {}
    for compiled_node in graph._nodes.values():
        node_path = (*parent_path, compiled_node.id)
        inputs = {
            name: _flat_input(node_path, node_input, outer_inputs, outer_loops, nodes, functions) for name, node_input in compiled_node.inputs.items()
        }
        loops = _loops_in_flat_graph(compiled_node.loops, outer_loops, parent_path)
        if isinstance(compiled_node, CompiledSubgraphNode):
            subgraph = FlatSubgraph(compiled_node=compiled_node, parent=parent, node_path=node_path, loops=loops)
            subgraph.inner_nodes = _add_flat_nodes(compiled_node.inner, subgraph, inputs, functions, nodes)
            nodes_by_id[compiled_node.id] = nodes[node_path] = subgraph
            continue
        constants: dict[str, Any] = {}
        graph_inputs: list[tuple[str, str]] = []
        edge_inputs: dict[str, EdgeInput | CollectingArgument] = {}
        upstream: dict[FlatFunctionNode, int] = {}
        collecting: dict[tuple[NodePath, str], list[tuple[str, bool]]] = {}
        for name, node_input in inputs.items():
            constants[name] = None
            read: list[tuple[NodePath, int]]  # the node each value read comes from, and how many loops it is inside
            outside_the_loop: list[tuple[FlatFunctionNode, int]] = []
            match node_input:
                case ConstantInput(value=value):
                    constants[name] = value
                    continue
                case GraphInput(name=graph_input):
                    graph_inputs.append((name, graph_input))
                    continue
                case EdgeInput(sources=edges, loops=value_loops):
                    read = [(edge.node, edge.depth) for edge in edges]
                    depth = len(value_loops)
                case CollectingArgument(input=CollectingInput(field=field, series=series, loops=series_loops, sources=edges), loop_body=loop_body):
                    depth = len(series_loops)
                    read = [(series[0], depth), *[(edge.node, edge.depth) for edge in edges]]
                    # what a node inside the loop reads from outside it can say that no item will have a value
                    read_inside = [read for inner in loop_body for read in inner.upstream.items()]
                    outside_the_loop = [(read, shared) for read, shared in read_inside if series not in read.loops]
                    collecting.setdefault(series, []).append((name, field.accepts_none_items))
                case _:
                    assert_never(node_input)
            edge_inputs[name] = node_input
            for upstream_node, shared_indexes in (*[(functions[read_path], min(read_depth, depth)) for read_path, read_depth in read], *outside_the_loop):
                upstream[upstream_node] = min(shared_indexes, upstream.get(upstream_node, shared_indexes))
        if loops and min(upstream.values(), default=0) < len(loops):
            # it reads something made outside its innermost loop, which may exist before the list looped over does
            upstream.setdefault(functions[loops[-1][0]], len(loops) - 1)
        read_by_door = {name: edge_input.field for name, edge_input in edge_inputs.items() if _demands_more(edge_input)}
        node = FlatFunctionNode(
            compiled_node=compiled_node, parent=parent, node_path=node_path, loops=loops, policy=compiled_node.definition.policy,
            constants=constants, graph_inputs=tuple(graph_inputs), edge_inputs=edge_inputs, lists_from_same_loop=tuple(map(tuple, collecting.values())),
            edge_door=EdgeDoor(read_by_door) if read_by_door else None,
            upstream=upstream, position=len(functions),
        )
        nodes_by_id[compiled_node.id] = nodes[node_path] = functions[node_path] = node
    return nodes_by_id


def _loop_in_flat_graph(loop: tuple[NodePath, str], graph_path: NodePath) -> tuple[NodePath, str]:
    """Returns a loop of a compiled graph as the flat graph names it: the compiled graph names it by the path of the
    function node that returns the list, from that graph, and the flat graph by its full path.

    In the example, ``(("lines",), "result")`` stays itself. A loop of ``loud``'s own graph gets ``loud``'s path in
    front: one over ``(("split",), "result")`` would be ``(("loud", "split"), "result")``.

    Args:
        loop: The loop as the compiled graph names it, by node paths from that graph.
        graph_path: The path of the subgraph node that compiled graph is used as; ``()`` for the graph a run is of.
    """
    node_path, output = loop
    return (*graph_path, *node_path), output


def _loops_in_flat_graph(
    loops: tuple[tuple[NodePath, str], ...], outer_loops: tuple[tuple[NodePath, str], ...], graph_path: NodePath,
) -> tuple[tuple[NodePath, str], ...]:
    """Returns loops of a compiled graph (a node's, or an input's) as the flat graph names them: the loops of the
    subgraph node the graph is used as, then each of ``loops`` as ``_loop_in_flat_graph`` names it.

    In the example, ``loud/shout`` is inside no loop of ``loud``'s own graph, and ``loud`` is inside the loop over
    ``lines.result``, so its loops are ``((("lines",), "result"),)``.

    Args:
        loops: The loops as the compiled graph names them, outermost first.
        outer_loops: The flat loops of the subgraph node the compiled graph is used as; ``()`` for the graph a run is of.
        graph_path: The path of that subgraph node; ``()`` for the graph a run is of.
    """
    return (*outer_loops, *[_loop_in_flat_graph(loop, graph_path) for loop in loops])


def _flat_input(
    node_path: NodePath, node_input: Input, outer_inputs: Mapping[str, ConstantInput | GraphInput | EdgeInput | CollectingArgument] | None,
    outer_loops: tuple[tuple[NodePath, str], ...], nodes: Mapping[NodePath, FlatNode], functions: Mapping[NodePath, FlatFunctionNode],
) -> ConstantInput | GraphInput | EdgeInput | CollectingArgument:
    """Returns how a call reads the input that its own graph compiled as ``node_input``: a constant or a graph input
    as compiled, and an input that edges lead to as an ``EdgeInput`` or a ``CollectingArgument`` of the flat graph.

    Every edge is changed to name the function node that returns the output, by its path, with the loops
    that function node is inside, and so is every loop. In the example, the compiled input ``join.texts`` reads ``loud.shout.result``; the
    flat graph's input reads ``result`` of ``loud/shout``. A graph input of a subgraph is read as the input of
    the subgraph node is: ``loud/strip``'s ``line`` as ``loud``'s ``line``.

    Args:
        node_path: The node path of the node the input is of.
        node_input: The input as the node's own graph compiled it.
        outer_inputs: See ``_add_flat_nodes``.
        outer_loops: The loops of the subgraph node the node is inside, as the flat graph names them; ``()`` for a node
            of the graph a run is of.
        nodes: Every flat node made so far, by node path.
        functions: Every function node made so far, by node path.
    """

    graph_path = node_path[:-1]

    def flat_edge(edge: Edge, starts_loop: bool) -> Edge:
        compiled_node = nodes[(*graph_path, *edge.node)]
        source_node, output = compiled_node.source_of(edge.output)
        if starts_loop and edge.series is not None:
            source_node.looped_outputs.setdefault(output, (compiled_node, edge.output))
        series = None if edge.series is None else _loop_in_flat_graph(edge.series, graph_path)
        return Edge(source_node.node_path, output, edge.record_field, source_node.loops, edge.type, series=series)

    match node_input:
        case ConstantInput():
            return node_input
        case MissingInput(field=field):
            raise RuntimeError(f"'{place_text(node_path, field.name)}' gets no value: a graph with problems has no flat graph")
        case GraphInput(name=name):
            return node_input if outer_inputs is None else outer_inputs[name]
        case EdgeInput(field=field, sources=sources, as_list=as_list):
            edges = tuple([flat_edge(edge, not as_list) for edge in sources])
            return EdgeInput(field, edges, _loops_in_flat_graph(node_input.loops, outer_loops, graph_path), as_list)
        case CollectingInput(field=field, series=series, sources=sources):
            looped_over = _loop_in_flat_graph(series, graph_path)
            edges = tuple([flat_edge(edge, False) for edge in sources])
            loop_body = _loop_body_of(looped_over, [functions[edge.node] for edge in edges])
            return CollectingArgument(CollectingInput(field, looped_over, _loops_in_flat_graph(node_input.loops, outer_loops, graph_path), edges), loop_body)
        case _:
            assert_never(node_input)


def _loop_body_of(series: tuple[NodePath, str], sources: list[FlatFunctionNode]) -> tuple[FlatFunctionNode, ...]:
    """Returns the function nodes inside the loop over ``series`` that a collecting input's value depends on.

    In the example, ``join.texts`` collects the loop over ``lines.result`` and reads ``loud/shout``, which
    reads ``loud/strip``: the result is ``(loud/strip, loud/shout)``.

    Args:
        series: The list the input collects the loop over: the path of the function node that returns it, and the
            output.
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


def _demands_more(edge_input: EdgeInput | CollectingArgument) -> bool:
    """Whether ``edge_input`` is read by its node's edge door before each call: its input's type demands more than the
    outputs its edges read declare (``demands_more``)."""
    match edge_input:
        case EdgeInput(field=field, sources=edges, as_list=as_list):
            return demands_more(field, [edge.type for edge in edges], per_item=as_list)
        case CollectingArgument(input=CollectingInput(field=field, sources=edges)):
            return demands_more(field, [edge.type for edge in edges], per_item=True)
        case _:
            assert_never(edge_input)
