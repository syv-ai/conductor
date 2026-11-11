"""The nodes of a compiled graph where they stand: what a run holds units of, and how each of their inputs is read.

A graph loads once, alone, however many graphs embed it. What a run walks is every function of the graph and of
each graph embedded in it, each a ``FlatFunctionNode``, with its inputs planned where it stands: an output of an
subgraph is by definition an output of a function inside it, so a reader's plan names that output, and an open
input inside has the plan the subgraph's node has for it. A call of a subgraph therefore costs what
its functions cost, and its node (a ``FlatSubgraph``) holds no unit of its own: it is a name and a boundary.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conductor._internal.address import Loc, NodePath, node_name
from conductor._internal.compiled.input_plans import (
    EdgeSource,
    InputCollectsLoopResults,
    InputConstant,
    InputFromEdges,
    InputFromGraphInput,
    InputMissing,
    InputPlan,
    InputReadsOneValue,
    InputWrapsOneValue,
    ItemOfSeries,
    OutputAddress,
    OutputInLoops,
    SeriesAsList,
    WholeOutput,
)
from conductor._internal.compiled.nodes import (
    CompiledFunctionNode,
    CompiledSubgraphNode,
)
from conductor._internal.field import Field
from conductor._internal.node_definition import Policy
from conductor._internal.outcomes import ItemPath
from conductor._internal.sentinels import REQUIRED
from conductor._internal.series import Series

if typing.TYPE_CHECKING:
    from conductor._internal.compiled.type import CompiledGraph


@dataclass(eq=False, slots=True, kw_only=True)
class _FlatNodeBase:
    """A node where it stands among the graphs a run is of: a node of the graph itself, or of a graph embedded
    in it, at any depth. Never made on its own: it is a ``FlatFunctionNode`` or a ``FlatSubgraph``.

    A graph loads once and may be embedded in many places; this is a node of it in one of them. A node inside an
    subgraph runs once for each item its subgraph node runs for, and then once per item of its own scope,
    so the maps it runs in are the subgraph node's and then its own.

    Attributes:
        parent: The subgraph's node it is directly inside; ``None`` for a node of the graph itself.
        path: The ids of the subgraphs' nodes it is inside, outermost first, then its own id.
        name: The path as a run and a save name the node, ``e/ask``.
        loops: Every map it runs in, outermost first, each as the function whose ``Series`` output it is and
            that output: those of the subgraph's node it is inside, then those of its own scope. A loop
            over a ``Series`` a stored graph offers is the loop inside that graph (``_series_looped_over``).
    """

    parent: FlatSubgraph | None
    path: NodePath
    name: str
    loops: tuple[tuple[FlatFunctionNode, str], ...]


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FlatFunctionNode(_FlatNodeBase):
    """A function node of a graph with its subgraphs laid out flat. Built once per graph and shared by
    every run of it; it holds nothing from a run.

    The arguments of one call come from three places: ``constants`` (the values typed into the graph and
    the defaults, fixed when the graph is checked), the graph inputs (``graph_inputs``, fixed when
    a run starts), and the values other nodes returned (``edge_inputs``, read for each call).

    How each input gets its value is written for where the node stands: every output of a plan names a function
    by its ``name``, so a plan is read the same wherever its node stands. An open input of a node inside an
    subgraph has the plan that graph's node has for it, and an edge from a subgraph's output reads
    the output inside that holds it.

    Attributes:
        compiled_node: The node, as its graph loaded it.
        policy: The node's policy, as its definition declares it.
        constants: What the arguments of every call start as: each input by name, in the order declared,
            one that holds the same value on every call already holding it.
        graph_inputs: The inputs the caller gives: the name, and the address a run keeps the caller's value by.
        edge_inputs: What the node needs that edges bring, in the order declared: the input's name, a plan, and
            whether the input takes a blank. A gathering input is two of them, as it needs two things: its
            Series, read as the one value it is, and then what its edges bring for the items (the plan
            itself, which names the functions it gathers through, ``InputCollectsLoopResults.loop_body``).
        lists_from_same_loop: The gathering inputs that gather one Series, for each such Series: each input's name
            and whether its items may be blank. They receive the same items.
        checked: The inputs whose value is read by the input's type before a call, as the types the edges bring
            don't settle it.
        looped_outputs: The ``Series`` outputs some reader works through one item at a time where the node stands,
            which a run holds to ``max_items``, each with the node and the output its reader's graph names it
            by: its own, for a reader in its graph; for a reader of a subgraph's node around it, that
            node and the output it offers, so the ceiling is said of what that graph's author drew. A reader
            placed after the node adds to it, as its plan is written (``_plan_at_path``).
        reads_from: Every function whose units its own are made from, each with how many places of an item
            path of this node say which of that function's units: the unit at those places, and where that
            function runs deeper every unit under them. They are the outputs its inputs' plans read (``reads``
            of each), no deeper than the input sits; for a gathering input also what the functions it gathers
            through read outside the items, which can settle it for every item at once; and, for a node that
            reads anything from outside the innermost map it runs in, or reads nothing, the Series of that
            map, since what it reads may be there before its items are.
        readers: Each function whose ``reads_from`` holds this one, with the number held there.
        position: Its place in the order a leg tries the functions in.
        waits_on: Every function it waits on, however far up; filled when first asked for.
        display_plans: The input plans ``run.results`` reads each output with, by the output's name
            (``host_reads``); filled when first asked for.
    """

    compiled_node: CompiledFunctionNode
    policy: Policy
    constants: dict[str, Any]
    graph_inputs: tuple[tuple[str, str], ...]
    edge_inputs: tuple[tuple[str, InputFromEdges, bool], ...]
    lists_from_same_loop: tuple[tuple[tuple[str, bool], ...], ...]
    checked: tuple[str, ...]
    looped_outputs: dict[str, tuple[FlatNode, str]] = dataclasses.field(default_factory=dict[str, tuple["FlatNode", str]])
    reads_from: dict[FlatFunctionNode, int] = dataclasses.field(default_factory=dict["FlatFunctionNode", int])
    readers: list[tuple[FlatFunctionNode, int]] = dataclasses.field(default_factory=list[tuple["FlatFunctionNode", int]])
    position: int = 0
    waits_on: frozenset[FlatFunctionNode] | None = None
    display_plans: dict[str, tuple[InputReadsOneValue | InputCollectsLoopResults, ...]] = dataclasses.field(default_factory=dict[str, tuple[InputReadsOneValue | InputCollectsLoopResults, ...]])


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FlatSubgraph(_FlatNodeBase):
    """A placed subgraph where it stands among the graphs a run is of, made with the functions inside it
    when a ``Graph``'s nodes are first walked. It is a name and a boundary: it is never called and a run
    holds no unit of it. Its inputs are handed to the open inputs inside as their plans, and each output it
    offers is the output of a function inside. What a run says of it (its results, that it started, that it is
    complete) is read from the functions inside when asked.

    Attributes:
        compiled_node: The node, as its graph loaded it.
        inputs: How each input gets its value where the node stands, by name: the plan an open input inside
            is given for its own.
        nodes: The nodes of the graph inside, where they stand.
        output_sources: Each output it offers, by its address in the inner graph: the function inside that makes it, at
            any depth of subgraphs, and that function's output.
    """

    compiled_node: CompiledSubgraphNode
    inputs: dict[str, InputPlan]
    nodes: list[FlatNode] = dataclasses.field(default_factory=list["FlatNode"])
    output_sources: dict[str, tuple[FlatFunctionNode, str]] = dataclasses.field(default_factory=dict[str, tuple["FlatFunctionNode", str]])


#: A node where it stands among the graphs a run is of: a function, or a subgraph.
FlatNode = FlatFunctionNode | FlatSubgraph



def waits_on(node: FlatFunctionNode) -> frozenset[FlatFunctionNode]:
    """Every function ``node`` waits on, however far up; worked out when first asked, as most nodes never are."""
    if node.waits_on is None:
        seen: set[FlatFunctionNode] = set()
        stack = [node]
        while stack:
            for read in stack.pop().reads_from:
                if read not in seen:
                    seen.add(read)
                    stack.append(read)
        node.waits_on = frozenset(seen)
    return node.waits_on


def outcome_loc(node: FlatNode, key: ItemPath) -> Loc:
    """Where the instance ``key`` of ``node`` is: the node's id, then the places of its own scope's items, after
    the same for each subgraph's node it is inside, ``("e", 0, "ask", 2)``.

    Args:
        node: The node.
        key: The instance; one cut off after the places of the subgraph's node gives the node's id alone
            at the end, ``("e", 0, "ask")``.
    """
    if node.parent is None:
        return (node.compiled_node.id, *key)
    around = len(node.parent.loops)  # the places of the subgraph's node's own item
    return (*outcome_loc(node.parent, key[:around]), node.compiled_node.id, *key[around:])


#: What a host is shown of an output made per item is what an input of this type wired to it would be given:
#: every item in its place, a blank where an item has no value.
_DISPLAY_FIELD = Field("shown", Series[Any | None], REQUIRED)  # type: ignore[misc]  # mypy takes no union as a type argument here


def host_reads(node: FlatFunctionNode, output: str) -> tuple[InputReadsOneValue | InputCollectsLoopResults, ...]:
    """The input plans ``run.results`` reads one output of a function node with.

    A host reads an output the way a node wired to it would, so the plans are the ones such a node's input
    would have. Example: ``documents -> lines -> shout``, where ``lines`` is called once per document and
    ``shout`` once per line. ``shout`` is called inside two loops, so its output has two plans: the first
    lists the documents, the second lists one document's lines and reads ``shout``'s return value for each.
    ``results`` then shows ``[["A", "B"], None, ["C"]]``: a list per document, ``None`` for a document
    whose lines were never made.

    Args:
        node: The function node that returns the output.
        output: The output's name.

    Returns:
        For a node called once, one plan that reads the return value. For a node called per item, one plan
        for each loop it is called in, outermost first; the plan at index ``n`` is used for one item of each
        of the ``n`` loops outside it.
    """
    plans = node.display_plans.get(output)
    if plans is None:
        made = node.compiled_node.outputs[output]
        at = (SeriesAsList if made.many else WholeOutput)(node.name, output, _loops_by_name(node.loops), None, made.type)
        functions = {made_by.name: made_by for made_by in (node, *(series for series, _ in node.loops))}
        per_loop = [_gather_at_path(_DISPLAY_FIELD, looped_over, (at,), functions) for looped_over in node.loops]
        plans = node.display_plans[output] = tuple(per_loop) or (InputReadsOneValue(_DISPLAY_FIELD, (at,), at.loops),)
    return plans


def flat_nodes(graph: CompiledGraph) -> tuple[list[FlatFunctionNode], dict[str, FlatNode]]:
    """The nodes of ``graph`` and those of each graph embedded in it, where they stand: the functions in the
    order a leg tries them, each after the functions it reads, and every node by its name."""
    tree: list[FlatFunctionNode] = []
    node_at: dict[str, FlatNode] = {}
    _add_to_tree(graph, None, tree, node_at)
    for node in tree:
        for read, places in node.reads_from.items():
            read.readers.append((node, places))
    return tree, node_at


def _add_to_tree(graph: CompiledGraph, parent: FlatSubgraph | None, tree: list[FlatFunctionNode], node_at: dict[str, FlatNode]) -> dict[str, FlatNode]:
    """Add the functions of ``graph`` to ``tree``, each after the functions it reads, those of each graph
    embedded in it where that graph's node stands, and every node to ``node_at``.

    A plan of the compiled graph names the nodes of its own graph. Here each is written for where its node stands
    (``_plan_at_path``): the one place a subgraph's boundary is crossed, inwards for an open input and
    outwards for an output.

    Args:
        graph: The graph: the one a run is of, or one embedded in it.
        parent: The subgraph's node that ``graph`` is the inner graph of; ``None`` for the run's own.
        tree: The functions so far, added to.
        node_at: Every node so far by its name, added to.

    Returns:
        The nodes of ``graph`` where they stand, by id.
    """
    inside = parent.path if parent is not None else ()
    above = parent.loops if parent is not None else ()
    here: dict[str, FlatNode] = {}
    functions = typing.cast("dict[str, FlatFunctionNode]", node_at)  # an output of a plan at a path names a function
    for compiled_node in graph._nodes.values():
        inputs: dict[str, InputPlan] = {}
        for name, plan in compiled_node.inputs.items():
            if parent is not None and isinstance(plan, InputFromGraphInput):  # the subgraph's node is given it, as a caller would give it
                inputs[name] = parent.inputs[graph._input_fields[plan.address].name]
            else:
                inputs[name] = _plan_at_path(plan, here, inside, functions)
        loops = (*above, *[_series_looped_over(loop, inside, node_at) for loop in compiled_node.loops])
        path = (*inside, compiled_node.id)
        if isinstance(compiled_node, CompiledSubgraphNode):
            subgraph = FlatSubgraph(compiled_node=compiled_node, parent=parent, path=path, name=node_name(path), loops=loops, inputs=inputs)
            within = _add_to_tree(compiled_node.inner, subgraph, tree, node_at)
            subgraph.nodes = list(within.values())
            for address in compiled_node.inner.outputs:
                made_by, _, output = address.partition(".")
                last = within[made_by]
                subgraph.output_sources[address] = last.output_sources[output] if isinstance(last, FlatSubgraph) else (last, output)
            here[compiled_node.id] = node_at[subgraph.name] = subgraph
            continue
        constants: dict[str, Any] = {}
        graph_inputs: list[tuple[str, str]] = []
        edge_inputs: list[tuple[str, InputFromEdges, bool]] = []
        reads_from: dict[FlatFunctionNode, int] = {}
        gathering: dict[OutputInLoops, list[tuple[str, bool]]] = {}
        for name, plan in inputs.items():
            constants[name] = None
            match plan:
                case InputConstant(value=value):
                    constants[name] = value
                    continue
                case InputFromGraphInput(address=address):
                    graph_inputs.append((name, address))
                    continue
                case InputMissing():
                    raise RuntimeError(f"'{node_name(path)}' has an input with no value to give: a graph with problems is never walked")
            read_at = [(functions[at.node_id], min(at.depth, len(plan.loops))) for at in plan.reads]  # no deeper than the input sits
            if isinstance(plan, InputCollectsLoopResults):
                series = plan.series
                items_of = (functions[series.node_id], series.output)  # what the functions it gathers through read outside the items can settle it too
                read_at += [(read, places) for inner in plan.loop_body for read, places in inner.reads_from.items() if items_of not in read.loops]
                gathering.setdefault(series, []).append((name, plan.field.accepts_none_items))
                whole = WholeOutput(series.node_id, series.output, series.loops, None, functions[series.node_id].compiled_node.outputs[series.output].whole)
                edge_inputs.append((name, InputReadsOneValue(plan.field, (whole,), series.loops), plan.field.whole.optional))  # it needs its Series, and then its items
            edge_inputs.append((name, plan, plan.field.whole.optional))
            for read, places in read_at:
                reads_from[read] = min(places, reads_from.get(read, places))
        node = FlatFunctionNode(
            compiled_node=compiled_node, parent=parent, path=path, name=node_name(path), loops=loops, policy=compiled_node.definition.policy,
            constants=constants, graph_inputs=tuple(graph_inputs), edge_inputs=tuple(edge_inputs), lists_from_same_loop=tuple(map(tuple, gathering.values())),
            checked=tuple(name for name, plan in inputs.items() if _read_before_a_call(plan)), reads_from=reads_from,
        )
        if loops and min(node.reads_from.values(), default=0) < len(loops):  # what it reads may be there before its items are
            node.reads_from.setdefault(loops[-1][0], len(loops) - 1)
        here[compiled_node.id] = node_at[node.name] = node
        node.position = len(tree)
        tree.append(node)
    return here


def _made_by(port: OutputAddress, here: Mapping[str, FlatNode]) -> tuple[FlatFunctionNode, str]:
    """The function that returns the output ``port`` names, and the output's name there: the node itself, or
    for a stored graph's node the function inside it that returns what the node offers."""
    made_by = here[port.node_id]
    return made_by.output_sources[port.output] if isinstance(made_by, FlatSubgraph) else (made_by, port.output)


def _series_looped_over(loop: OutputAddress, inside: NodePath, node_at: Mapping[str, FlatNode]) -> tuple[FlatFunctionNode, str]:
    """The list one loop of a node's scope goes through, as the function that returns it and the output's name.

    A loop over a function's ``Series`` output is that output. A loop a stored graph's node names is a loop
    inside the stored graph. Example: ``e`` is the stored graph ``documents -> lines -> shout``, and a node
    outside is called once per item of ``e.documents.result``: that is ``result`` of ``e/documents``.

    Args:
        loop: The loop, as a graph's scope names it.
        inside: The path of the stored graph's node that graph is inside; empty for the run's own graph.
        node_at: Every node placed so far, by its name.
    """
    made_by = node_at[node_name((*inside, loop.node_id))]
    if isinstance(made_by, FlatSubgraph):
        inner_id, _, inner_output = loop.output.partition(".")
        return _series_looped_over(OutputAddress(inner_id, inner_output), made_by.path, node_at)
    return made_by, loop.output


def _loops_by_name(loops: tuple[tuple[FlatFunctionNode, str], ...]) -> tuple[OutputAddress, ...]:
    """``loops`` as the scope of a plan at a path, each Series named by the name of its function."""
    return tuple([OutputAddress(series.name, output) for series, output in loops])


def _plan_at_path(plan: InputPlan, here: Mapping[str, FlatNode], inside: NodePath, functions: Mapping[str, FlatFunctionNode]) -> InputPlan:
    """A plan of a compiled graph as the plan of its node where it stands: each output names the function that
    returns it by its ``name``, in the loops that function is called in.

    An output of a stored graph's node is the output of a function inside it: an edge from
    ``e.shout.result`` reads ``result`` of ``e/shout``.

    Args:
        plan: The plan, as its graph loaded it.
        here: The nodes of that graph placed so far, where they stand, by id.
        inside: The path of the stored graph's node that graph is inside; empty for the run's own graph.
        functions: Every function placed so far, by its name.
    """

    def at_path(source: EdgeSource, opened: bool) -> EdgeSource:
        made_by, output = _made_by(OutputAddress(source.node_id, source.output), here)
        brought = ItemOfSeries if made_by.compiled_node.outputs[output].many else WholeOutput
        if opened and brought is ItemOfSeries:
            made_by.looped_outputs.setdefault(output, (here[source.node_id], source.output))
        return brought(made_by.name, output, _loops_by_name(made_by.loops), source.record_field, source.type)

    match plan:
        case InputReadsOneValue(field=field, reads=reads):
            brings = tuple([at_path(source, True) for source in reads])
            return InputReadsOneValue(field, brings, max((source.read_in for source in brings), key=len))
        case InputWrapsOneValue(field=field, reads=reads):
            return InputWrapsOneValue(field, tuple([typing.cast(WholeOutput, at_path(source, False)) for source in reads]))
        case InputCollectsLoopResults(field=field, series=series, sources=sources):
            looped_over = _series_looped_over(OutputAddress(series.node_id, series.output), inside, functions)
            return _gather_at_path(field, looped_over, tuple([at_path(source, False) for source in sources]), functions)
        case _:
            return plan


def _gather_at_path(field: Field, looped_over: tuple[FlatFunctionNode, str], sources: tuple[EdgeSource, ...], functions: Mapping[str, FlatFunctionNode]) -> InputCollectsLoopResults:
    """The plan of an input that collects one value per item of the list ``looped_over``, where its node
    stands: the one way such a plan is made, for a node's own input and for what a host is shown.

    Args:
        field: The input's field.
        looped_over: The list, as the function that returns it and the output's name.
        sources: What each edge brings for an item, each naming a function by its name.
        functions: The functions the plan's outputs name, by name.
    """
    made_by, output = looped_over
    series = OutputInLoops(made_by.name, output, _loops_by_name(made_by.loops))
    through = _gathered_through(looped_over, [functions[at.node_id] for at in sources])
    return InputCollectsLoopResults(field, (series, *sources), through)


def _gathered_through(series: tuple[FlatFunctionNode, str], sources: list[FlatFunctionNode]) -> tuple[FlatFunctionNode, ...]:
    """Finds the functions made per item of ``series`` that a gathering input rests on, by what each one's
    units are made from (``reads_from``).

    Args:
        series: The Series the input gathers: the function that makes it, and its output.
        sources: The functions whose outputs the input reads for an item.

    Returns:
        The functions, in the order a leg tries them, each after the ones it reads.
    """
    through: set[FlatFunctionNode] = set()
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.loops and inner not in through:
            through.add(inner)
            stack.extend(inner.reads_from)
    return tuple(sorted(through, key=lambda inner: inner.position))


def _read_before_a_call(plan: InputPlan) -> bool:
    """Whether what edges bring an input is read by the input's type before each call: not when every value they
    bring is already one the field takes as it is, with no limit or choice to hold it to."""
    match plan:
        case InputReadsOneValue(field=field, sources=sources):
            return not all(field.whole.holds(s.type) for s in sources)
        case InputCollectsLoopResults(field=field, sources=sources) | InputWrapsOneValue(field=field, sources=sources):
            return field.whole.record is not None or not all(field.type.holds(s.type) for s in sources)
        case _:
            return False
