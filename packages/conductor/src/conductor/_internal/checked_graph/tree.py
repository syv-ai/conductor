"""The nodes of a checked graph where they stand: what a run holds units of, and how each of their inputs is read.

A graph loads once, alone, however many graphs embed it. What a run walks is every function of the graph and of
each graph embedded in it, each a ``FunctionAtPath``, with its inputs planned where it stands: an output of an
embedded graph is by definition an output of a function inside it, so a reader's plan names that output, and an open
input inside has the plan the embedded graph's node has for it. A call of an embedded graph therefore costs what
its functions cost, and its node (an ``EmbeddedGraphAtPath``) holds no unit of its own: it is a name and a boundary.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conductor._internal.address import Loc, NodePath, node_name
from conductor._internal.checked_graph.input_plans import (
    InputFedByEdges,
    InputFromCaller,
    InputGathersItems,
    InputGathersOneValue,
    InputHoldsValue,
    InputNotFed,
    InputPlan,
    InputReadsOneValue,
    ItemOfSeries,
    OutputAddress,
    OutputAtScope,
    Scope,
    SeriesAsList,
    Source,
    WholeOutput,
)
from conductor._internal.checked_graph.placed_node import PlacedEmbeddedGraph, PlacedFunction
from conductor._internal.field import Field
from conductor._internal.node_definition import Policy
from conductor._internal.sentinels import REQUIRED
from conductor._internal.series import Series
from conductor._internal.units import ItemPath

if typing.TYPE_CHECKING:
    from conductor._internal.checked_graph.type import CheckedGraph


@dataclass(eq=False, slots=True, kw_only=True)
class _AtPath:
    """A node where it stands among the graphs a run is of: a node of the graph itself, or of a graph embedded
    in it, at any depth. Never made on its own: it is a ``FunctionAtPath`` or an ``EmbeddedGraphAtPath``.

    A graph loads once and may be embedded in many places; this is a node of it in one of them. A node inside an
    embedded graph runs once for each item its embedding node runs for, and then once per item of its own scope,
    so the maps it runs in are the embedding node's and then its own.

    Attributes:
        embedding: The embedded graph's node it is directly inside; ``None`` for a node of the graph itself.
        path: The ids of the embedded graphs' nodes it is inside, outermost first, then its own id.
        name: The path as a run and a save name the node, ``e/ask``.
        maps: Every map it runs in, outermost first, each as the function whose ``Series`` output it is and
            that output: those of the embedded graph's node it is inside, then those of its own scope. A loop
            over a ``Series`` a stored graph offers is the loop inside that graph (``_series_looped_over``).
    """

    embedding: EmbeddedGraphAtPath | None
    path: NodePath
    name: str
    maps: Maps


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FunctionAtPath(_AtPath):
    """A function node where it stands: what a run holds units of and a leg calls.

    How each input gets its value is written for where the node stands: every output of a plan names a function
    by its ``name``, so a plan is read the same wherever its node stands. An open input of a node inside an
    embedded graph has the plan that graph's node has for it, and an edge from an embedded graph's output reads
    the output inside that holds it.

    Attributes:
        placed: The node, as its graph loaded it.
        policy: The node's policy, as its definition declares it.
        call_template: What the arguments of every call start as: each input by name, in the order declared,
            one that holds the same value on every call already holding it.
        from_caller: The inputs the caller gives: the name, and the address a run keeps the caller's value by.
        fed: What the node needs that edges bring, in the order declared: the input's name, a plan, and
            whether the input takes a blank. A gathering input is two of them, as it needs two things: its
            Series, read as the one value it is, and then what its edges bring for the items (the plan
            itself, which names the functions it gathers through, ``InputGathersItems.asked_for_any_item``).
        gathered_in_step: The gathering inputs that gather one Series, for each such Series: each input's name
            and whether its items may be blank. They receive the same items.
        checked: The inputs whose value is read by the input's type before a call, as the types the edges bring
            don't settle it.
        opens: The ``Series`` outputs some reader works through one item at a time where the node stands,
            which a run holds to ``max_items``, each with the node and the output its reader's graph names it
            by: its own, for a reader in its graph; for a reader of an embedded graph's node around it, that
            node and the output it offers, so the ceiling is said of what that graph's author drew. A reader
            placed after the node adds to it, as its plan is written (``_plan_at_path``).
        made_from: Every function whose units its own are made from, each with how many places of an item
            path of this node say which of that function's units: the unit at those places, and where that
            function runs deeper every unit under them. They are the outputs its inputs' plans read (``reads``
            of each), no deeper than the input sits; for a gathering input also what the functions it gathers
            through read outside the items, which can settle it for every item at once; and, for a node that
            reads anything from outside the innermost map it runs in, or reads nothing, the Series of that
            map, since what it reads may be there before its items are.
        readers: Each function whose ``made_from`` holds this one, with the number held there.
        position: Its place in the order a leg tries the functions in.
        waits_on: Every function it waits on, however far up; filled when first asked for.
        shown_as: The input plans ``run.results`` reads each output with, by the output's name
            (``host_reads``); filled when first asked for.
    """

    placed: PlacedFunction
    policy: Policy
    call_template: dict[str, Any]
    from_caller: tuple[tuple[str, str], ...]
    fed: tuple[tuple[str, InputFedByEdges, bool], ...]
    gathered_in_step: tuple[tuple[tuple[str, bool], ...], ...]
    checked: tuple[str, ...]
    opens: dict[str, tuple[NodeAtPath, str]] = dataclasses.field(default_factory=dict[str, tuple["NodeAtPath", str]])
    made_from: dict[FunctionAtPath, int] = dataclasses.field(default_factory=dict["FunctionAtPath", int])
    readers: list[tuple[FunctionAtPath, int]] = dataclasses.field(default_factory=list[tuple["FunctionAtPath", int]])
    position: int = 0
    waits_on: frozenset[FunctionAtPath] | None = None
    shown_as: dict[str, tuple[InputReadsOneValue | InputGathersItems, ...]] = dataclasses.field(default_factory=dict[str, tuple[InputReadsOneValue | InputGathersItems, ...]])


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class EmbeddedGraphAtPath(_AtPath):
    """A placed embedded graph where it stands among the graphs a run is of, made with the functions inside it
    when a ``Graph``'s nodes are first walked. It is a name and a boundary: it is never called and a run
    holds no unit of it. Its inputs are handed to the open inputs inside as their plans, and each output it
    offers is the output of a function inside. What a run says of it (its results, that it started, that it is
    complete) is read from the functions inside when asked.

    Attributes:
        placed: The node, as its graph loaded it.
        inputs: How each input gets its value where the node stands, by name: the plan an open input inside
            is given for its own.
        nodes: The nodes of the graph inside, where they stand.
        offers: Each output it offers, by its address in the inner graph: the function inside that makes it, at
            any depth of embedded graphs, and that function's output.
    """

    placed: PlacedEmbeddedGraph
    inputs: dict[str, InputPlan]
    nodes: list[NodeAtPath] = dataclasses.field(default_factory=list["NodeAtPath"])
    offers: dict[str, tuple[FunctionAtPath, str]] = dataclasses.field(default_factory=dict[str, tuple["FunctionAtPath", str]])


#: A node where it stands among the graphs a run is of: a function, or an embedded graph.
NodeAtPath = FunctionAtPath | EmbeddedGraphAtPath

#: The maps a node runs in, outermost first, each as the function and the output of its ``Series``.
Maps = tuple[tuple["FunctionAtPath", str], ...]


def waits_on(node: FunctionAtPath) -> frozenset[FunctionAtPath]:
    """Every function ``node`` waits on, however far up; worked out when first asked, as most nodes never are."""
    if node.waits_on is None:
        seen: set[FunctionAtPath] = set()
        stack = [node]
        while stack:
            for read in stack.pop().made_from:
                if read not in seen:
                    seen.add(read)
                    stack.append(read)
        node.waits_on = frozenset(seen)
    return node.waits_on


def unit_loc(node: NodeAtPath, key: ItemPath) -> Loc:
    """Where the instance ``key`` of ``node`` is: the node's id, then the places of its own scope's items, after
    the same for each embedded graph's node it is inside, ``("e", 0, "ask", 2)``.

    Args:
        node: The node.
        key: The instance; one cut off after the places of the embedded graph's node gives the node's id alone
            at the end, ``("e", 0, "ask")``.
    """
    if node.embedding is None:
        return (node.placed.id, *key)
    around = len(node.embedding.maps)  # the places of the embedded graph's node's own item
    return (*unit_loc(node.embedding, key[:around]), node.placed.id, *key[around:])


#: What a host is shown of an output made per item is what an input of this type wired to it would be given:
#: every item in its place, a blank where an item has no value.
_SHOWN = Field("shown", Series[Any | None], REQUIRED)  # type: ignore[misc]  # mypy takes no union as a type argument here


def host_reads(node: FunctionAtPath, output: str) -> tuple[InputReadsOneValue | InputGathersItems, ...]:
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
    plans = node.shown_as.get(output)
    if plans is None:
        made = node.placed.outputs[output]
        at = (SeriesAsList if made.many else WholeOutput)(node.name, output, _scope_at_path(node.maps), None, made.type)
        functions = {made_by.name: made_by for made_by in (node, *(series for series, _ in node.maps))}
        per_loop = [_gather_at_path(_SHOWN, looped_over, (at,), functions) for looped_over in node.maps]
        plans = node.shown_as[output] = tuple(per_loop) or (InputReadsOneValue(_SHOWN, (at,), at.scope),)
    return plans


def nodes_at_paths(graph: CheckedGraph) -> tuple[list[FunctionAtPath], dict[str, NodeAtPath]]:
    """The nodes of ``graph`` and those of each graph embedded in it, where they stand: the functions in the
    order a leg tries them, each after the functions it reads, and every node by its name."""
    tree: list[FunctionAtPath] = []
    node_at: dict[str, NodeAtPath] = {}
    _add_to_tree(graph, None, tree, node_at)
    for node in tree:
        for read, places in node.made_from.items():
            read.readers.append((node, places))
    return tree, node_at


def _add_to_tree(graph: CheckedGraph, embedding: EmbeddedGraphAtPath | None, tree: list[FunctionAtPath], node_at: dict[str, NodeAtPath]) -> dict[str, NodeAtPath]:
    """Add the functions of ``graph`` to ``tree``, each after the functions it reads, those of each graph
    embedded in it where that graph's node stands, and every node to ``node_at``.

    A plan of the checked graph names the nodes of its own graph. Here each is written for where its node stands
    (``_plan_at_path``): the one place an embedded graph's boundary is crossed, inwards for an open input and
    outwards for an output.

    Args:
        graph: The graph: the one a run is of, or one embedded in it.
        embedding: The embedded graph's node that ``graph`` is the inner graph of; ``None`` for the run's own.
        tree: The functions so far, added to.
        node_at: Every node so far by its name, added to.

    Returns:
        The nodes of ``graph`` where they stand, by id.
    """
    inside = embedding.path if embedding is not None else ()
    above = embedding.maps if embedding is not None else ()
    here: dict[str, NodeAtPath] = {}
    functions = typing.cast("dict[str, FunctionAtPath]", node_at)  # an output of a plan at a path names a function
    for placed in graph._nodes.values():
        inputs: dict[str, InputPlan] = {}
        for name, plan in placed.inputs.items():
            if embedding is not None and isinstance(plan, InputFromCaller):  # the embedded graph's node is given it, as a caller would give it
                inputs[name] = embedding.inputs[graph._input_fields[plan.address].name]
            else:
                inputs[name] = _plan_at_path(plan, here, inside, functions)
        maps = (*above, *[_series_looped_over(loop, inside, node_at) for loop in placed.scope])
        path = (*inside, placed.id)
        if isinstance(placed, PlacedEmbeddedGraph):
            embedded = EmbeddedGraphAtPath(placed=placed, embedding=embedding, path=path, name=node_name(path), maps=maps, inputs=inputs)
            within = _add_to_tree(placed.inner, embedded, tree, node_at)
            embedded.nodes = list(within.values())
            for address in placed.inner.outputs:
                made_by, _, output = address.partition(".")
                last = within[made_by]
                embedded.offers[address] = last.offers[output] if isinstance(last, EmbeddedGraphAtPath) else (last, output)
            here[placed.id] = node_at[embedded.name] = embedded
            continue
        call_template: dict[str, Any] = {}
        from_caller: list[tuple[str, str]] = []
        fed: list[tuple[str, InputFedByEdges, bool]] = []
        made_from: dict[FunctionAtPath, int] = {}
        gathering: dict[tuple[OutputAtScope, ...], list[tuple[str, bool]]] = {}
        for name, plan in inputs.items():
            call_template[name] = None
            match plan:
                case InputHoldsValue(value=value):
                    call_template[name] = value
                    continue
                case InputFromCaller(address=address):
                    from_caller.append((name, address))
                    continue
                case InputNotFed():
                    raise RuntimeError(f"'{node_name(path)}' has an input with no value to give: a graph with problems is never walked")
            read_at = [(functions[at.node_id], min(at.depth, len(plan.scope))) for at in plan.reads]  # no deeper than the input sits
            if isinstance(plan, InputGathersItems):
                series = plan.series
                items_of = (functions[series.node_id], series.output)  # what the functions it gathers through read outside the items can settle it too
                read_at += [(read, places) for inner in plan.asked_for_any_item for read, places in inner.made_from.items() if items_of not in read.maps]
                gathering.setdefault(plan.reads[: plan.levels], []).append((name, plan.field.keeps_blanks))
                whole = WholeOutput(series.node_id, series.output, series.scope, None, functions[series.node_id].placed.outputs[series.output].whole)
                fed.append((name, InputReadsOneValue(plan.field, (whole,), series.scope), plan.field.whole.optional))  # it needs its Series, and then its items
            fed.append((name, plan, plan.field.whole.optional))
            for read, places in read_at:
                made_from[read] = min(places, made_from.get(read, places))
        node = FunctionAtPath(
            placed=placed, embedding=embedding, path=path, name=node_name(path), maps=maps, policy=placed.definition.policy,
            call_template=call_template, from_caller=tuple(from_caller), fed=tuple(fed), gathered_in_step=tuple(map(tuple, gathering.values())),
            checked=tuple(name for name, plan in inputs.items() if _read_before_a_call(plan)), made_from=made_from,
        )
        if maps and min(node.made_from.values(), default=0) < len(maps):  # what it reads may be there before its items are
            node.made_from.setdefault(maps[-1][0], len(maps) - 1)
        here[placed.id] = node_at[node.name] = node
        node.position = len(tree)
        tree.append(node)
    return here


def _made_by(port: OutputAddress, here: Mapping[str, NodeAtPath]) -> tuple[FunctionAtPath, str]:
    """The function that returns the output ``port`` names, and the output's name there: the node itself, or
    for a stored graph's node the function inside it that returns what the node offers."""
    made_by = here[port.node_id]
    return made_by.offers[port.output] if isinstance(made_by, EmbeddedGraphAtPath) else (made_by, port.output)


def _series_looped_over(loop: OutputAddress, inside: NodePath, node_at: Mapping[str, NodeAtPath]) -> tuple[FunctionAtPath, str]:
    """The list one loop of a node's scope goes through, as the function that returns it and the output's name.

    A loop over a function's ``Series`` output is that output. A loop a stored graph's node names is a loop
    inside the stored graph. Example: ``e`` is the stored graph ``documents -> lines -> shout``, where
    ``shout`` is called once per line of each document.

    - ``e.documents.result`` is the list ``documents`` returns inside ``e``.
    - ``e.shout.result`` is what ``e`` offers: ``shout`` returns one value per line, so the loop over them is
      the loop ``shout`` is called in, over ``lines.result``.

    Args:
        loop: The loop, as a graph's scope names it.
        inside: The path of the stored graph's node that graph is inside; empty for the run's own graph.
        node_at: Every node placed so far, by its name.
    """
    made_by = node_at[node_name((*inside, loop.node_id))]
    if isinstance(made_by, EmbeddedGraphAtPath):
        inner_id, _, inner_output = loop.output.partition(".")
        return _series_looped_over(OutputAddress(inner_id, inner_output), made_by.path, node_at)
    return (made_by, loop.output) if made_by.placed.outputs[loop.output].many else made_by.maps[-1]


def _scope_at_path(maps: Maps) -> Scope:
    """``maps`` as the scope of a plan at a path, each Series named by the name of its function."""
    return tuple([OutputAddress(series.name, output) for series, output in maps])


def _plan_at_path(plan: InputPlan, here: Mapping[str, NodeAtPath], inside: NodePath, functions: Mapping[str, FunctionAtPath]) -> InputPlan:
    """A plan of a checked graph as the plan of its node where it stands: each output names the function that
    returns it by its ``name``, in the loops that function is called in.

    An output of a stored graph's node is the output of a function inside it. Example: ``e`` is the stored
    graph ``documents -> lines -> shout``. An edge from ``e.shout.result`` reads one item of the ``Series``
    that ``e`` offers, and that item is what one call of ``shout`` returned, so the plan reads ``e/shout``'s
    ``result`` whole.

    Args:
        plan: The plan, as its graph loaded it.
        here: The nodes of that graph placed so far, where they stand, by id.
        inside: The path of the stored graph's node that graph is inside; empty for the run's own graph.
        functions: Every function placed so far, by its name.
    """

    def at_path(source: Source, opened: bool) -> Source:
        made_by, output = _made_by(OutputAddress(source.node_id, source.output), here)
        brought = ItemOfSeries if made_by.placed.outputs[output].many else WholeOutput
        if opened and brought is ItemOfSeries:
            made_by.opens.setdefault(output, (here[source.node_id], source.output))
        return brought(made_by.name, output, _scope_at_path(made_by.maps), source.record_field, source.type)

    match plan:
        case InputReadsOneValue(field=field, reads=reads):
            brings = tuple([at_path(source, True) for source in reads])
            return InputReadsOneValue(field, brings, max((source.read_in for source in brings), key=len))
        case InputGathersOneValue(field=field, reads=reads):
            return InputGathersOneValue(field, tuple([typing.cast(WholeOutput, at_path(source, False)) for source in reads]))
        case InputGathersItems(field=field, series=series, sources=sources):
            looped_over = _series_looped_over(OutputAddress(series.node_id, series.output), inside, functions)
            return _gather_at_path(field, looped_over, tuple([at_path(source, False) for source in sources]), functions)
        case _:
            return plan


def _gather_at_path(field: Field, looped_over: tuple[FunctionAtPath, str], sources: tuple[Source, ...], functions: Mapping[str, FunctionAtPath]) -> InputGathersItems:
    """The plan of an input that collects one value per item of the list ``looped_over``, where its node
    stands: the one way such a plan is made, for a node's own input and for what a host is shown.

    Args:
        field: The input's field.
        looped_over: The list, as the function that returns it and the output's name.
        sources: What each edge brings for an item, each naming a function by its name.
        functions: The functions the plan's outputs name, by name.
    """
    made_by, output = looped_over
    series = OutputAtScope(made_by.name, output, _scope_at_path(made_by.maps))
    through = _gathered_through(looped_over, [functions[at.node_id] for at in sources])
    return InputGathersItems(field, (series, *sources), asked_for_any_item=through)


def _gathered_through(series: tuple[FunctionAtPath, str], sources: list[FunctionAtPath]) -> tuple[FunctionAtPath, ...]:
    """Finds the functions made per item of ``series`` that a gathering input rests on, by what each one's
    units are made from (``made_from``).

    Args:
        series: The Series the input gathers: the function that makes it, and its output.
        sources: The functions whose outputs the input reads for an item.

    Returns:
        The functions, in the order a leg tries them, each after the ones it reads.
    """
    through: set[FunctionAtPath] = set()
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.maps and inner not in through:
            through.add(inner)
            stack.extend(inner.made_from)
    return tuple(sorted(through, key=lambda inner: inner.position))


def _read_before_a_call(plan: InputPlan) -> bool:
    """Whether what edges bring an input is read by the input's type before each call: not when every value they
    bring is already one the field takes as it is, with no limit or choice to hold it to."""
    match plan:
        case InputReadsOneValue(field=field, sources=sources):
            return not all(field.whole.holds(s.type) for s in sources)
        case InputGathersItems(field=field, sources=sources) | InputGathersOneValue(field=field, sources=sources):
            return field.whole.record is not None or not all(field.type.holds(s.type) for s in sources)
        case _:
            return False
