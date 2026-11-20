"""The run blueprint: the plan every run of one graph follows.

A compiled graph can have subgraph nodes, but a subgraph is never called. Only the functions inside it are. So
when a graph's first run is made or read, conductor flattens it into a blueprint: every function node of the graph
and of its subgraphs, in the order a run tries them, with what each call reads. The ``Graph`` keeps the blueprint,
and every run of the graph shares it. Nothing in it belongs to one run.

The example in this file. ``loud`` is the subgraph ``strip -> shout``, used as one node of an outer graph::

    lines(text: str) -> Series[str]     # called once; returns ["a", "b"]
    loud                                # wired to lines.result: its nodes are called once per line
        strip(line: str) -> str
        shout(line: str) -> str
    join(texts: Series[str]) -> str     # wired to loud.result: called once, with ["A!", "B!"]

The compiled graph has three nodes: ``lines``, ``loud`` and ``join``. The blueprint has four function nodes:
``lines``, ``loud/strip``, ``loud/shout`` and ``join``, each a ``BlueprintFunctionNode``. It also has ``loud``
itself, as a ``BlueprintSubgraphNode`` with no outcomes of its own.

Inside a compiled graph, a node is named by its path from that graph. A run names it by its full path from the
outer graph instead, which is the subgraph node's path followed by the inner path. The blueprint names every node,
loop and input that way once, when it's built (``Edge.in_outer_graph``), so a run never asks which graph an input
was compiled in:

- ``join.texts`` is wired to ``loud.result``, which the function node ``("loud", "shout")`` returns. ``join`` is
  in the outer graph, so its path is already full.
- ``loud/shout``'s ``line`` reads ``("strip",)``, as ``loud``'s graph names it. In a run that is
  ``("loud", "strip")``.
- ``loud/strip``'s input is a graph input of the subgraph. A call gets it as ``loud``'s input, which is one item
  of ``lines.result``.
- ``loud/strip`` and ``loud/shout`` are inside the loop over ``lines.result``, because ``loud`` is. A blueprint
  node names its loops by full paths too.

``RunOutcomes`` in ``outcomes.py`` uses the blueprint to work out each call's arguments.
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
    Source,
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
    loops_in_outer_graph,
    path_in_outer_graph,
)
from conductor._internal.node import Policy

from .validator import EdgeValidator


@dataclass(eq=False, slots=True, kw_only=True)
class _BlueprintNodeBase:
    """What both kinds of blueprint node have.

    It is never made on its own: every blueprint node is a ``BlueprintFunctionNode`` or a ``BlueprintSubgraphNode``.
    A subgraph may be used as a node in many places. A blueprint node is one node in one of those places. In the
    example, ``loud/shout`` is ``shout`` inside ``loud``.

    Attributes:
        parent: The subgraph node this node is directly inside. ``None`` for a node of the outer graph.
        node_path: The ids of the subgraph nodes this node is inside, outermost first, then its own id, such as
            ``("loud", "shout")``. A run stores the node's outcomes under it.
        loops: The loops the node is inside, outermost first: those of its subgraph node, then its own. Each loop
            is the path of the function node that returns the list looped over, and that output's name. In the
            example, ``loud/shout`` has ``((("lines",), "result"),)``. The function node of the loop at index ``n``
            is itself inside the first ``n`` loops.
        off: Whether the node is switched off, by its own record or by a subgraph node it is inside
            (``NodeRecord.off``). A run skips each call of a function node switched off, and reads none of its
            inputs.
    """

    parent: BlueprintSubgraphNode | None
    node_path: NodePath
    loops: tuple[tuple[NodePath, str], ...]
    off: bool


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class BlueprintFunctionNode(_BlueprintNodeBase):
    """A function node in the blueprint: what a run calls and stores outcomes of.

    The arguments of one call come from three places. ``constants`` are known when the graph is compiled.
    ``graph_inputs`` are known when a run starts. ``edge_inputs`` are worked out from the run's outcomes for each
    call. A graph input of a subgraph is the input of the same name on the subgraph node: ``loud/strip``'s ``line``
    is ``loud``'s ``line``.

    Attributes:
        compiled_node: The node as its own graph compiled it.
        code_fingerprint: What a call's outputs depend on besides its arguments, as a short hash
            (``CompiledFunctionNode.code_fingerprint``), with the loops around the subgraph the node is in. An
            earlier call is reused, and a waiting id still matches, only while it stays the same.
        constants: Every input by name, in the order declared. An input that is a constant has its value, and
            every other input has ``None``. Each run copies the values once, when it is made
            (``RunOutcomes.constants``), and a call's arguments start as a copy of the run's dict.
        graph_inputs: The inputs whose value the caller gives: each input's name, and the name of its graph input
            (``"text"``).
        edge_inputs: The inputs that edges lead to, by name, in the order declared: an ``EdgeInput`` for one value
            per call, or a ``CollectingInput`` for one that collects a loop. Every node and loop in them is named by
            its full path. So ``loud/shout``'s ``line`` reads ``("loud", "strip")``. ``loud/strip``'s ``line`` is
            ``loud``'s input, which reads ``("lines",)``.
        lists_from_same_loop: The collecting inputs, grouped by the loop they collect. Each has its name and
            whether it takes ``None`` items. An item that is ``None`` for an input that doesn't take it is left out
            of every list in its group, so the lists of one group keep the same items.
        edge_validator: Checks, before each call, the inputs whose type is stricter than the outputs their edges
            come from (``validator.py``). ``None`` when no input is.
        looped_outputs: The node's ``Series`` outputs that some edge reads one item at a time, by output name. A run
            checks each against ``max_items``. The value is the output the reading edge is wired to: the node and
            output the reading node's graph knows the list by. That is this node, unless the reading node is outside
            a subgraph this node is inside. Then it is that subgraph node, and its output that gives the list. The
            ``item_limit`` problem names the list that way.
        upstream: Each function node whose outcomes a call of this node reads, with a number ``n``. The call at
            ``item_indexes`` reads that node's outcome at ``item_indexes[:n]``. If that node is inside more than
            ``n`` loops, the call reads every outcome whose item indexes start with those. In the example, ``join``
            has ``{loud/shout: 0, lines: 0}``: its one call reads every call of ``loud/shout``.
        downstream: Each function node that has this one in its ``upstream``, with its number there.
        position: The node's place in the order a run tries the function nodes in.
        all_upstream: Every function node upstream of this one, at any distance. ``None`` until ``all_upstream``
            works it out.
    """

    compiled_node: CompiledFunctionNode
    code_fingerprint: str
    constants: dict[str, Any]
    graph_inputs: tuple[tuple[str, str], ...]
    edge_inputs: dict[str, EdgeInput | CollectingInput]
    lists_from_same_loop: tuple[tuple[tuple[str, bool], ...], ...]
    edge_validator: EdgeValidator | None
    upstream: dict[BlueprintFunctionNode, int]
    position: int
    looped_outputs: dict[str, Source] = dataclasses.field(default_factory=dict[str, Source])
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
    """A subgraph node in the blueprint, such as ``loud`` in the example.

    It is never called, and a run stores no outcomes for it. What a run says about it, such as its results or that
    it is complete, comes from the nodes inside it.

    Attributes:
        compiled_node: The node as the outer graph compiled it.
        inner_nodes: The subgraph's nodes as blueprint nodes, by their id in the subgraph. Empty until they are
            made, which happens right after this node is.
    """

    compiled_node: CompiledSubgraphNode
    inner_nodes: dict[str, BlueprintNode] = dataclasses.field(
        default_factory=dict[str, "BlueprintNode"]
    )


#: A node of the blueprint: a function, or a subgraph.
BlueprintNode = BlueprintFunctionNode | BlueprintSubgraphNode


def all_upstream(node: BlueprintFunctionNode) -> frozenset[BlueprintFunctionNode]:
    """Returns every function node upstream of ``node``, at any distance.

    It is worked out on the first call and stored on the node, since most nodes are never asked.
    """
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
    """Returns the ``loc`` of the call of ``node`` at ``item_indexes``.

    A ``loc`` names a call the way a ``Problem`` does: each id of the node's path, followed by the item indexes of
    the loops that node adds. In the example, the call of ``loud/shout`` for the second line is
    ``("loud", 1, "shout")``. ``loud`` is inside the loop over ``lines.result``, and ``shout`` adds no loop of its
    own.

    Args:
        node: The node.
        item_indexes: The item indexes of the call. If they only cover the loops of the node's subgraph node, the
            ``loc`` ends at the node's id, as in ``("loud", 1, "shout")``.
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
    """The blueprint of one compiled graph, which every run of the graph follows.

    ``RunBlueprint.of(graph)`` builds it when the graph's first run is made (``graph.with_inputs``) or read
    (``Run.from_json``), and the ``Graph`` keeps it. If two threads do that at once, each may build one. They are
    equal, and one is kept.

    Attributes:
        function_nodes: Every function node of the graph and of its subgraphs, at any depth, by node path
            (``("loud", "shout")``). They are in the order a run tries them: each after the nodes upstream of it.
        nodes: Every blueprint node, function or subgraph, by node path (``("loud",)``, ``("loud", "shout")``).
    """

    function_nodes: dict[NodePath, BlueprintFunctionNode]
    nodes: dict[NodePath, BlueprintNode]

    @functools.cached_property
    def threads(self) -> int:
        """How many calls may run on threads at once during one ``resume()``.

        It is the sum of ``policy.most_at_once`` over the nodes called on a thread. A node definition that sets
        ``concurrency`` is counted once, however many places use it, since its limit is shared.
        """
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
        """Returns the blueprint of ``graph``, which must have no problems."""
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
    """Makes the blueprint nodes of ``graph`` and of each subgraph in it, and adds them to ``functions`` and ``nodes``.

    In the example, called with the outer graph, it adds ``lines``, then ``loud``, then ``join``. For ``loud`` it
    calls itself with the subgraph, which adds ``loud/strip`` and ``loud/shout``.

    Args:
        graph: The compiled graph: the outer one, or a subgraph inside it.
        parent: The blueprint node of the subgraph node ``graph`` is used as. ``None`` for the outer graph.
        functions: The function nodes made so far, by node path, in the order a run tries them. Added to.
        nodes: Every blueprint node made so far, by node path. Added to.

    Returns:
        The blueprint nodes of ``graph`` itself, by their id in ``graph``.
    """
    graph_path, outer_loops = (parent.node_path, parent.loops) if parent is not None else ((), ())
    graph_nodes: dict[str, BlueprintNode] = {}
    for compiled_node in graph._nodes.values():
        node_path = path_in_outer_graph(graph_path, (compiled_node.id,))
        loops = loops_in_outer_graph(graph_path, outer_loops, compiled_node.loops)
        off = compiled_node.off or (parent is not None and parent.off)
        node: BlueprintNode
        if isinstance(compiled_node, CompiledSubgraphNode):
            node = BlueprintSubgraphNode(
                compiled_node=compiled_node,
                parent=parent,
                node_path=node_path,
                loops=loops,
                off=off,
            )
            node.inner_nodes = _add_blueprint_nodes(compiled_node.inner, node, functions, nodes)
        else:
            node = functions[node_path] = _function_node(
                compiled_node, parent, node_path, loops, off, functions
            )
        graph_nodes[compiled_node.id] = nodes[node_path] = node
    return graph_nodes


def _function_node(
    compiled_node: CompiledFunctionNode,
    parent: BlueprintSubgraphNode | None,
    node_path: NodePath,
    loops: tuple[tuple[NodePath, str], ...],
    off: bool,
    functions: Mapping[NodePath, BlueprintFunctionNode],
) -> BlueprintFunctionNode:
    """Returns the blueprint node of one function node.

    It works out where the node is, what each call of it reads, and which function nodes those values come from.
    Each input that edges lead to is named by full paths here (``in_outer_graph``), once.

    A node switched off reads no input, since no call of it is made. Its graph inputs may be none of the graph's
    (``GraphRecord.inputs_offered``). It still waits for the list it loops over, which says how many calls it has.

    It also records each ``Series`` output that an input reads one item at a time. The output goes into the
    ``looped_outputs`` of the function node that returns it, named by the output the edge is wired to.

    Args:
        compiled_node: The node, as its graph compiled it.
        parent: The subgraph node it is directly inside. ``None`` for a node of the outer graph.
        node_path: Its full path.
        loops: Every loop it is inside, named by full paths.
        off: Whether it is switched off, by its own record or by a subgraph node it is inside.
        functions: Every function node made so far, by node path. The nodes upstream of this one are among them.
    """
    upstream: dict[BlueprintFunctionNode, int] = {}
    collecting: dict[tuple[NodePath, str], list[tuple[str, bool]]] = {}
    constants: dict[str, Any] = {}
    graph_inputs: list[tuple[str, str]] = []
    edge_inputs: dict[str, EdgeInput | CollectingInput] = {}
    for name, node_input in () if off else compiled_node.inputs.items():
        subgraph = parent  # the subgraph node whose graph ``node_input`` was compiled in
        while (
            isinstance(node_input, GraphInput) and subgraph is not None
        ):  # a graph input of a subgraph is the subgraph node's input
            node_input, subgraph = subgraph.compiled_node.inputs[node_input.name], subgraph.parent
        around = (subgraph.node_path, subgraph.loops) if subgraph is not None else ((), ())
        constants[name] = None
        read: list[
            tuple[BlueprintFunctionNode, int]
        ]  # the node each value comes from, and how many loops that node is inside
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
            case EdgeInput():
                node_input = node_input.in_outer_graph(*around)
                sources = [functions[edge.node] for edge in node_input.sources]
                read = [(source, len(source.loops)) for source in sources]
                depth = len(node_input.loops)
                for edge in node_input.sources:
                    if edge.per_item:
                        functions[edge.node].looped_outputs.setdefault(edge.output, edge.wired_to)
            case CollectingInput(field=field):
                node_input = node_input.in_outer_graph(*around)
                looped_over = node_input.series
                series_node = functions[looped_over[0]]
                sources = [functions[edge.node] for edge in node_input.sources]
                depth = len(series_node.loops)
                read = [(series_node, depth), *[(source, len(source.loops)) for source in sources]]
                # A value from outside the loop, read by a node inside it, can mean no item will have a value.
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
        edge_inputs[name] = node_input
        for upstream_node, shared_indexes in (
            *[(source, min(source_depth, depth)) for source, source_depth in read],
            *outside_the_loop,
        ):
            upstream[upstream_node] = min(
                shared_indexes, upstream.get(upstream_node, shared_indexes)
            )
    if loops and min(upstream.values(), default=0) < len(loops):
        # It reads a value made outside its innermost loop, which may exist before the list it loops over does,
        # so it waits for the node that returns that list too.
        upstream.setdefault(functions[loops[-1][0]], len(loops) - 1)
    return BlueprintFunctionNode(
        compiled_node=compiled_node,
        code_fingerprint=compiled_node.code_fingerprint(parent.loops if parent is not None else ()),
        parent=parent,
        node_path=node_path,
        loops=loops,
        off=off,
        constants=constants,
        graph_inputs=tuple(graph_inputs),
        edge_inputs=edge_inputs,
        lists_from_same_loop=tuple(map(tuple, collecting.values())),
        edge_validator=EdgeValidator.of(edge_inputs),
        upstream=upstream,
        position=len(functions),
    )


def loop_body_of(
    series: tuple[NodePath, str], sources: list[BlueprintFunctionNode]
) -> tuple[BlueprintFunctionNode, ...]:
    """Returns the function nodes inside the loop over ``series`` that a value read from ``sources`` depends on.

    They come in order, each after the nodes upstream of it. In the example, ``join.texts`` collects the loop over
    ``lines.result`` and reads ``loud/shout``, which reads ``loud/strip``. The result is
    ``(loud/strip, loud/shout)``.

    Args:
        series: The list looped over: the path of the function node that returns it, and the output's name.
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
