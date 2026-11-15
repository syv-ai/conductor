"""The run blueprint: how every run of a compiled graph plays out. It has every function node of the graph and of its
subgraphs, in the order a run tries them, with what each call reads through edges.

The example used in this file. ``loud`` is the subgraph ``strip -> shout``, used as one node of an outer graph::

    lines(text: str) -> Series[str]     # called once; returns ["a", "b"]
    loud                                # wired to lines.result: its nodes are called once per line
        strip(line: str) -> str
        shout(line: str) -> str
    join(texts: Series[str]) -> str     # wired to loud.result: called once, with ["A!", "B!"]

The compiled graph has three nodes: ``lines``, ``loud`` and ``join``. A subgraph is never called: only the
functions inside it are. So a run works on the blueprint, which is built on the first run of the graph and kept by
the ``Graph`` (``graph._blueprint``). It has the four function nodes ``lines``, ``loud/strip``, ``loud/shout`` and ``join`` (each a
``BlueprintFunctionNode``), and ``loud`` (a ``BlueprintSubgraphNode``), which has no outcomes of its own.

A compiled input names function nodes and loops by their path from the graph its node is in. A run keeps
outcomes by each function node's full path, which is the path of the subgraph node that graph is used as, then
the path from the graph (``function_node_in_run``):

- The edge into ``join.texts`` is wired to ``loud.result``, which the function node ``("loud", "shout")`` returns.
  ``join`` is in the graph a run is of, so that is also its full path.
- The edge into ``loud/shout``'s ``line`` reads ``("strip",)``, named in ``loud``'s graph: in a run, ``("loud",
  "strip")``.
- The input of ``loud/strip`` is a graph input inside the subgraph. A call reads it as ``loud``'s input, read
  where ``loud`` is: one item of ``lines.result``.
- ``loud/strip`` and ``loud/shout`` are inside the loop over ``lines.result``, because ``loud`` is. A blueprint
  node's ``loops`` are named by the function node's full path, as a run keeps outcomes.

``ResultReader`` (``arguments.py``) reads each input from the run's outcomes.

A blueprint node has nothing from a run in it: it is made once and shared by every run of the graph.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, assert_never

from conductor._internal.base import (
    ItemIndexes,
    Loc,
    NodePath,
    place_text,
)
from conductor._internal.graph import (
    CollectingInput,
    CompiledFunctionNode,
    CompiledSubgraphNode,
    ConstantInput,
    EdgeInput,
    Graph,
    GraphInput,
    MissingInput,
)
from conductor._internal.node import Policy

from .validator import EdgeValidator


@dataclass(eq=False, slots=True, kw_only=True)
class _BlueprintNodeBase:
    """What both kinds of blueprint node have. Never made on its own: it is a ``BlueprintFunctionNode`` or a
    ``BlueprintSubgraphNode``.

    A subgraph is compiled once and may be used as a node in many places. A blueprint node is one node in one of
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

    parent: BlueprintSubgraphNode | None
    node_path: NodePath
    loops: tuple[tuple[NodePath, str], ...]


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class BlueprintFunctionNode(_BlueprintNodeBase):
    """A function node of the blueprint: what a run calls and keeps outcomes of.

    The arguments of one call come from three places: ``constants`` (known when the graph is compiled),
    ``graph_inputs`` (known when a run starts) and ``edge_inputs`` (read from the run's outcomes for each call).
    A graph input of a subgraph is read as the input of the subgraph node of the same name, where that node is:
    ``loud/strip``'s ``line`` as ``loud``'s ``line``.

    Attributes:
        compiled_node: The node as its own graph compiled it.
        constants: Every input by name, in the order declared: the value of an input that is a constant, and
            ``None`` for every other input. The arguments of a call start as a copy of it.
        graph_inputs: The inputs the caller gives a value for: the input's name, and the name of the graph input
            whose value it is (``"text"``).
        edge_inputs: The inputs that edges lead to, by name, in the order declared: the path of the graph the
            input was compiled in, and the compiled input, an ``EdgeInput`` for one that reads one value per call or
            a ``CollectingInput`` for one that collects a loop. The path is that of the subgraph node the graph is
            used as, ``()`` for the graph a run is of: ``("loud",)`` for ``loud/shout``'s ``line``, and ``()`` for
            ``loud/strip``'s, which is read as ``loud``'s input.
        lists_from_same_loop: The collecting inputs, grouped by the loop they collect: each input's name and
            whether it accepts ``None`` items. An item that is ``None`` for an input that does not accept it is
            left out of every list of its group, so the lists of one group have the same items.
        edge_validator: Reads before each call the inputs whose type demands more than the outputs their edges read
            declare (``validator.py``); ``None`` when no input does.
        looped_outputs: The ``Series`` outputs of the node that some edge reads one item at a time, by output
            name. A run holds each to ``max_items``. The value is the node and output that the reading node's
            graph names the list by: this node, or, when the reading node is outside a subgraph this node is
            inside, that subgraph and the name of its output that gives the list. The item-limit problem names the list that
            way.
        upstream: Each function node whose outcomes a call of this node reads, with a number ``n``: the call at
            ``item_indexes`` reads that node's outcome at ``item_indexes[:n]``, or, when that node is inside
            more than ``n`` loops, every outcome of it whose item indexes start with those. In the example,
            ``join`` has ``{loud/shout: 0, lines: 0}``: its one call reads every call of ``loud/shout``.
        downstream: Each function node that has this one in its ``upstream``, with the number it has there.
        position: The node's index in the order a run tries the function nodes in.
        all_upstream: Every function node upstream of this one, at any distance. ``None`` until
            ``all_upstream(node)`` works it out.
    """

    compiled_node: CompiledFunctionNode
    constants: dict[str, Any]
    graph_inputs: tuple[tuple[str, str], ...]
    edge_inputs: dict[str, tuple[NodePath, EdgeInput | CollectingInput]]
    lists_from_same_loop: tuple[tuple[tuple[str, bool], ...], ...]
    edge_validator: EdgeValidator | None
    upstream: dict[BlueprintFunctionNode, int]
    position: int
    looped_outputs: dict[str, tuple[BlueprintNode, str]] = dataclasses.field(
        default_factory=dict[str, tuple["BlueprintNode", str]]
    )
    downstream: list[tuple[BlueprintFunctionNode, int]] = dataclasses.field(
        default_factory=list[tuple["BlueprintFunctionNode", int]]
    )
    all_upstream: frozenset[BlueprintFunctionNode] | None = None

    @property
    def policy(self) -> Policy:
        """The node's policy, as its definition declares it."""
        return self.compiled_node.definition.policy


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class BlueprintSubgraphNode(_BlueprintNodeBase):
    """A subgraph in the blueprint: ``loud`` in the example. It is never called and a run keeps no outcomes of
    it. What a run says about it (its results, that it started, that it is complete) is worked out from the
    nodes inside it.

    Attributes:
        compiled_node: The node as the outer graph compiled it.
        inner_nodes: The nodes of the subgraph as blueprint nodes, by their id in the subgraph. Empty until they are
            made, which is right after this node is.
    """

    compiled_node: CompiledSubgraphNode
    inner_nodes: dict[str, BlueprintNode] = dataclasses.field(
        default_factory=dict[str, "BlueprintNode"]
    )


#: A node of the blueprint: a function, or a subgraph.
BlueprintNode = BlueprintFunctionNode | BlueprintSubgraphNode


def all_upstream(node: BlueprintFunctionNode) -> frozenset[BlueprintFunctionNode]:
    """Returns every function node upstream of ``node``, at any distance. Worked out on the first call and kept
    on the node, as most nodes are never asked."""
    if node.all_upstream is None:
        seen: set[BlueprintFunctionNode] = set()
        stack = [node]
        while stack:
            for upstream_node in stack.pop().upstream:
                if upstream_node not in seen:
                    seen.add(upstream_node)
                    stack.append(upstream_node)
        node.all_upstream = frozenset(seen)
    return node.all_upstream


def outcome_loc(node: BlueprintNode, item_indexes: ItemIndexes) -> Loc:
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
    return (
        *outcome_loc(node.parent, item_indexes[:parent_depth]),
        node.compiled_node.id,
        *item_indexes[parent_depth:],
    )


@dataclass(frozen=True, eq=False)
class RunBlueprint:
    """The blueprint of how every run of one compiled graph plays out. ``RunBlueprint.of(graph)`` builds it on the
    graph's first run, and the ``Graph`` keeps it (``graph._blueprint``). A first run on two threads at once may build
    two equal blueprints; one is kept.

    Attributes:
        function_nodes: Every function node of the graph and of its subgraphs, at any depth, by node path
            (``("loud", "shout")``), in the order a run tries them: each after the nodes upstream of it.
        nodes: Every blueprint node, a function or a subgraph, by node path (``("loud",)``, ``("loud", "shout")``).
    """

    function_nodes: dict[NodePath, BlueprintFunctionNode]
    nodes: dict[NodePath, BlueprintNode]

    @functools.cached_property
    def threads(self) -> int:
        """How many calls can run on threads at once during one call of ``resume()``: the sum of
        ``policy.most_at_once`` over the nodes called on a thread, a node definition that sets ``concurrency``
        counted once however many places use it."""
        counted = {
            node.compiled_node.definition.running_calls
            if node.policy.concurrency is not None
            else node: node.policy.most_at_once
            for node in self.function_nodes.values()
            if node.policy.threaded
        }
        return sum(counted.values())

    @classmethod
    def of(cls, graph: Graph[Any]) -> RunBlueprint:
        """Returns the blueprint of ``graph``, a graph with no problems."""
        functions: dict[NodePath, BlueprintFunctionNode] = {}
        nodes: dict[NodePath, BlueprintNode] = {}
        _add_blueprint_nodes(graph, None, functions, nodes)
        for node in functions.values():
            for upstream_node, shared_indexes in node.upstream.items():
                upstream_node.downstream.append((node, shared_indexes))
        return cls(functions, nodes)


def _add_blueprint_nodes(
    graph: Graph[Any],
    parent: BlueprintSubgraphNode | None,
    functions: dict[NodePath, BlueprintFunctionNode],
    nodes: dict[NodePath, BlueprintNode],
) -> dict[str, BlueprintNode]:
    """Makes the blueprint nodes of ``graph``, and of each subgraph in it, and adds them to ``functions`` and
    ``nodes``.

    In the example, called with the outer graph it adds ``lines``, then ``loud`` (calling itself with the
    subgraph, which adds ``loud/strip`` and ``loud/shout``), then ``join``.

    Args:
        graph: The compiled graph: the one a run is of, or a subgraph of it.
        parent: The blueprint node of the subgraph that ``graph`` is; ``None`` for the graph a run is of.
        functions: The function nodes made so far, by node path, in the order a run tries them. Added to.
        nodes: Every blueprint node made so far, by node path. Added to.

    Returns:
        The blueprint nodes of ``graph`` itself, by their id in ``graph``.
    """
    graph_path = parent.node_path if parent is not None else ()
    outer_loops = parent.loops if parent is not None else ()
    graph_nodes: dict[str, BlueprintNode] = {}
    for compiled_node in graph._nodes.values():
        node_path = _path_in_run(graph_path, (compiled_node.id,))
        loops = (
            *outer_loops,
            *[
                (_path_in_run(graph_path, loop_node), output)
                for loop_node, output in compiled_node.loops
            ],
        )
        node: BlueprintNode
        if isinstance(compiled_node, CompiledSubgraphNode):
            node = BlueprintSubgraphNode(
                compiled_node=compiled_node, parent=parent, node_path=node_path, loops=loops
            )
            node.inner_nodes = _add_blueprint_nodes(compiled_node.inner, node, functions, nodes)
        else:
            node = functions[node_path] = _function_node(
                compiled_node, parent, node_path, loops, functions
            )
        graph_nodes[compiled_node.id] = nodes[node_path] = node
        _note_looped_outputs(node, functions, nodes)
    return graph_nodes


def _function_node(
    compiled_node: CompiledFunctionNode,
    parent: BlueprintSubgraphNode | None,
    node_path: NodePath,
    loops: tuple[tuple[NodePath, str], ...],
    functions: Mapping[NodePath, BlueprintFunctionNode],
) -> BlueprintFunctionNode:
    """Returns the blueprint node of a function node: where it is, what each call of it reads, and the function
    nodes whose outcomes those reads come from.

    Args:
        compiled_node: The node, as its graph compiled it.
        parent: The subgraph node it is directly inside; ``None`` for a node of the graph a run is of.
        node_path: Its full path.
        loops: Every loop it is inside, as a run names it.
        functions: Every function node made so far, by node path: those upstream of the node among them.
    """
    upstream: dict[BlueprintFunctionNode, int] = {}
    collecting: dict[tuple[NodePath, str], list[tuple[str, bool]]] = {}
    constants: dict[str, Any] = {}
    graph_inputs: list[tuple[str, str]] = []
    edge_inputs: dict[str, tuple[NodePath, EdgeInput | CollectingInput]] = {}
    for name, node_input in compiled_node.inputs.items():
        subgraph = parent  # the subgraph node whose graph ``node_input`` was compiled in
        while (
            isinstance(node_input, GraphInput) and subgraph is not None
        ):  # read as the subgraph node's input
            node_input, subgraph = subgraph.compiled_node.inputs[node_input.name], subgraph.parent
        graph_path, outer_loops = (
            (subgraph.node_path, subgraph.loops) if subgraph is not None else ((), ())
        )
        constants[name] = None
        read: list[
            tuple[BlueprintFunctionNode, int]
        ]  # the node each value read comes from, and how many loops it is inside
        outside_the_loop: list[tuple[BlueprintFunctionNode, int]] = []
        match node_input:
            case ConstantInput(value=value):
                constants[name] = value
                continue
            case GraphInput(name=graph_input):
                graph_inputs.append((name, graph_input))
                continue
            case MissingInput(field=field):
                raise RuntimeError(
                    f"'{place_text(node_path, field.name)}' gets no value: a graph with problems has no blueprint"
                )
            case EdgeInput(sources=edges, loops=value_loops):
                read = [
                    (source, len(source.loops))
                    for source in [
                        function_node_in_run(functions, graph_path, edge.node) for edge in edges
                    ]
                ]
                depth = len(outer_loops) + len(value_loops)
            case CollectingInput(field=field, series=series, sources=edges):
                series_node, sources = (
                    function_node_in_run(functions, graph_path, series[0]),
                    [function_node_in_run(functions, graph_path, edge.node) for edge in edges],
                )
                looped_over = series_node.node_path, series[1]
                depth = len(series_node.loops)
                read = [(series_node, depth), *[(source, len(source.loops)) for source in sources]]
                # what a node inside the loop reads from outside it can say that no item will have a value
                read_inside = [
                    read
                    for inner in loop_body_of(looped_over, sources)
                    for read in inner.upstream.items()
                ]
                outside_the_loop = [
                    (read, shared) for read, shared in read_inside if looped_over not in read.loops
                ]
                collecting.setdefault(looped_over, []).append((name, field.keeps_none_items))
            case _:
                assert_never(node_input)
        edge_inputs[name] = graph_path, node_input
        for upstream_node, shared_indexes in (
            *[(source, min(source_depth, depth)) for source, source_depth in read],
            *outside_the_loop,
        ):
            upstream[upstream_node] = min(
                shared_indexes, upstream.get(upstream_node, shared_indexes)
            )
    if loops and min(upstream.values(), default=0) < len(loops):
        # it reads something made outside its innermost loop, which may exist before the list looped over does
        upstream.setdefault(functions[loops[-1][0]], len(loops) - 1)
    return BlueprintFunctionNode(
        compiled_node=compiled_node,
        parent=parent,
        node_path=node_path,
        loops=loops,
        constants=constants,
        graph_inputs=tuple(graph_inputs),
        edge_inputs=edge_inputs,
        lists_from_same_loop=tuple(map(tuple, collecting.values())),
        edge_validator=EdgeValidator.of(
            {name: node_input for name, (_, node_input) in edge_inputs.items()}
        ),
        upstream=upstream,
        position=len(functions),
    )


def _path_in_run(graph_path: NodePath, node_path: NodePath) -> NodePath:
    """Returns ``node_path``, a path from a compiled graph, as a run names the node: by its full path. In the
    example, ``("strip",)`` in ``loud``'s graph is ``("loud", "strip")``.

    Args:
        graph_path: The path of the subgraph node the compiled graph is used as; ``()`` for the graph a run is of.
        node_path: The path from the compiled graph.
    """
    return (*graph_path, *node_path)


def function_node_in_run(
    function_nodes: Mapping[NodePath, BlueprintFunctionNode],
    graph_path: NodePath,
    node_path: NodePath,
) -> BlueprintFunctionNode:
    """Returns the function node that ``node_path``, a path from a compiled graph, names in a run. Every edge, loop and
    collected list of a compiled input names its function node this way. In the example, ``("shout",)`` in
    ``loud``'s graph is the function node ``loud/shout``.

    Args:
        function_nodes: The run's function nodes, by full path (``RunBlueprint.function_nodes``).
        graph_path: The path of the subgraph node the compiled graph is used as; ``()`` for the graph a run is of.
        node_path: The path from the compiled graph.
    """
    return function_nodes[_path_in_run(graph_path, node_path)]


def _note_looped_outputs(
    node: BlueprintNode,
    functions: Mapping[NodePath, BlueprintFunctionNode],
    nodes: Mapping[NodePath, BlueprintNode],
) -> None:
    """Adds each ``Series`` output that an input of ``node`` reads one item at a time to the ``looped_outputs`` of
    the function node that returns it, named as ``node``'s graph names it: by the node the edge is wired to."""
    graph_path = node.node_path[:-1]
    for node_input in node.compiled_node.inputs.values():
        match node_input:
            case EdgeInput(sources=edges, as_list=False):
                for edge in edges:
                    if edge.per_item:
                        wired_to = nodes[_path_in_run(graph_path, (edge.wired_to.node,))]
                        function_node_in_run(
                            functions, graph_path, edge.node
                        ).looped_outputs.setdefault(edge.output, (wired_to, edge.wired_to.output))
            case EdgeInput() | CollectingInput() | ConstantInput() | GraphInput() | MissingInput():
                pass
            case _:
                assert_never(node_input)


def loop_body_of(
    series: tuple[NodePath, str], sources: list[BlueprintFunctionNode]
) -> tuple[BlueprintFunctionNode, ...]:
    """Returns the function nodes inside the loop over ``series`` that a value read from ``sources`` depends on,
    each after the nodes upstream of it.

    In the example, ``join.texts`` collects the loop over ``lines.result`` and reads ``loud/shout``, which
    reads ``loud/strip``: the result is ``(loud/strip, loud/shout)``.

    Args:
        series: The list looped over, as a run names it: the path of the function node that returns it, and the
            output.
        sources: The function nodes the value is read from.
    """
    loop_body: set[BlueprintFunctionNode] = set()
    stack = list(sources)
    while stack:
        inner = stack.pop()
        if series in inner.loops and inner not in loop_body:
            loop_body.add(inner)
            stack.extend(inner.upstream)
    return tuple(sorted(loop_body, key=lambda inner: inner.position))
