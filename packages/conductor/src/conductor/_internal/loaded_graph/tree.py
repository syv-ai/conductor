"""The nodes of a loaded graph where they stand: what a run holds units of, and how each of their inputs is read.

A graph loads once, alone, however many graphs embed it. What a run walks is every function of the graph and of
each graph embedded in it, each a ``FunctionAtPath``, with its inputs planned where it stands: an output of an
embedded graph is by definition a cell of a function inside it, so a reader's plan names that cell, and an open
input inside has the plan the embedded graph's node has for it. A call of an embedded graph therefore costs what
its functions cost, and its node (a ``GraphAtPath``) holds no unit of its own: it is a name and a boundary.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conductor._internal.address import NodePath, Path, node_name
from conductor._internal.fields import REQUIRED, Field, Series
from conductor._internal.loaded_graph.input_plans import (
    Cell,
    Fed,
    Fixed,
    Gather,
    InputPlan,
    Item,
    Open,
    Port,
    Read,
    Scope,
    SeriesOfOne,
    Source,
    Unset,
    Value,
    WholeSeries,
)
from conductor._internal.loaded_graph.type import LoadedGraph, PlacedFunction, PlacedGraph
from conductor._internal.node_definition import Policy
from conductor._internal.units import ItemPath


@dataclass(eq=False, slots=True, kw_only=True)
class _AtPath:
    """A node where it stands among the graphs a run is of: a node of the graph itself, or of a graph embedded
    in it, at any depth. Never made on its own: it is a ``FunctionAtPath`` or a ``GraphAtPath``.

    A graph loads once and may be embedded in many places; this is a node of it in one of them. A node inside an
    embedded graph runs once for each item its embedding node runs for, and then once per item of its own scope,
    so the maps it runs in are the embedding node's and then its own.

    Attributes:
        embedding: The embedded graph's node it is directly inside; ``None`` for a node of the graph itself.
        path: The ids of the embedded graphs' nodes it is inside, outermost first, then its own id.
        name: The path as a run and a save name the node, ``e/ask``.
        maps: Every map it runs in, outermost first, each as the function whose ``Series`` output it is and
            that output: those of the embedded graph's node it is inside, then those of its own scope. A map
            over what an embedded graph offers is the map inside that makes it.
    """

    embedding: GraphAtPath | None
    path: NodePath
    name: str
    maps: Maps


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FunctionAtPath(_AtPath):
    """A function node where it stands: what a run holds units of and a leg calls.

    How each input gets its value is written for where the node stands: every cell of a plan names a function
    by its ``name``, so a plan is read the same wherever its node stands. An open input of a node inside an
    embedded graph has the plan that graph's node has for it, and an edge from an embedded graph's output reads
    the cell inside that holds it.

    Attributes:
        placed: The node, as its graph loaded it.
        policy: The node's policy, as its definition declares it.
        call_template: What the arguments of every call start as: each input by name, in the order declared,
            one that holds the same value on every call already holding it.
        from_caller: The inputs the caller gives: the name, and the address a run keeps the caller's value by.
        fed: What the node needs that edges bring, in the order declared: the input's name, a plan, and
            whether the input takes a blank. A gathering input is two of them, as it needs two things: its
            Series, read as the one value it is, and then what its edges bring for the items (the plan
            itself, which names the functions it gathers through, ``Gather.through``).
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
            function runs deeper every unit under them. They are the cells its inputs' plans read (``reads``
            of each), no deeper than the input sits; for a gathering input also what the functions it gathers
            through read outside the items, which can settle it for every item at once; and, for a node that
            reads anything from outside the innermost map it runs in, or reads nothing, the Series of that
            map, since what it reads may be there before its items are.
        readers: Each function whose ``made_from`` holds this one, with the number held there.
        position: Its place in the order a leg tries the functions in.
        waits_on: Every function it waits on, however far up; filled when first asked for.
        shown_as: How a host reads each output, by its name and the number of maps the host stands under
            (``host_reads``); filled when first asked for.
    """

    placed: PlacedFunction
    policy: Policy
    call_template: dict[str, Any]
    from_caller: tuple[tuple[str, str], ...]
    fed: tuple[tuple[str, Fed, bool], ...]
    gathered_in_step: tuple[tuple[tuple[str, bool], ...], ...]
    checked: tuple[str, ...]
    opens: dict[str, tuple[NodeAtPath, str]] = dataclasses.field(default_factory=dict[str, tuple["NodeAtPath", str]])
    made_from: dict[FunctionAtPath, int] = dataclasses.field(default_factory=dict["FunctionAtPath", int])
    readers: list[tuple[FunctionAtPath, int]] = dataclasses.field(default_factory=list[tuple["FunctionAtPath", int]])
    position: int = 0
    waits_on: frozenset[FunctionAtPath] | None = None
    shown_as: dict[tuple[str, int], Read | Gather] = dataclasses.field(default_factory=dict[tuple[str, int], Read | Gather])


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class GraphAtPath(_AtPath):
    """An embedded graph's node where it stands. It is a name and a boundary: it is never called and a run
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

    placed: PlacedGraph
    inputs: dict[str, InputPlan]
    nodes: list[NodeAtPath] = dataclasses.field(default_factory=list["NodeAtPath"])
    offers: dict[str, tuple[FunctionAtPath, str]] = dataclasses.field(default_factory=dict[str, tuple["FunctionAtPath", str]])


#: A node where it stands among the graphs a run is of: a function, or an embedded graph.
NodeAtPath = FunctionAtPath | GraphAtPath

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


def unit_loc(node: NodeAtPath, key: ItemPath) -> Path:
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


def host_reads(node: FunctionAtPath, output: str, under: int) -> Read | Gather:
    """How a host reads the output ``output`` of ``node`` from ``under`` maps down: the plan an input wired to
    it there would have. An output made once there is read as the one value it is; one made per item is
    gathered, through every map the host stands outside, as a ``Series`` input that keeps a blank in its place
    would gather it. So what a host is shown is what a node reading the output would be given.

    Args:
        node: The function that makes the output.
        output: The output's name.
        under: How many of the maps the node runs in the host stands inside: none for the run's own results,
            those of an embedded graph's node for what that node offers for one of its items.
    """
    plan = node.shown_as.get((output, under))
    if plan is None:
        spec = node.placed.outputs[output]
        cell = (WholeSeries if spec.many else Value)(node.name, output, _scope_at_path(node.maps), None, spec.type)
        if len(node.maps) == under:
            plan = Read(_SHOWN, (cell,), cell.scope)
        else:
            plan = _gather_at_path(_SHOWN, node.maps[under:], (cell,), {made_by.name: made_by for made_by in (node, *(series for series, _ in node.maps))})
        node.shown_as[(output, under)] = plan
    return plan


def nodes_at_paths(graph: LoadedGraph) -> tuple[list[FunctionAtPath], dict[str, NodeAtPath]]:
    """The nodes of ``graph`` and those of each graph embedded in it, where they stand: the functions in the
    order a leg tries them, each after the functions it reads, and every node by its name."""
    tree: list[FunctionAtPath] = []
    node_at: dict[str, NodeAtPath] = {}
    _add_to_tree(graph, None, tree, node_at)
    for node in tree:
        for read, places in node.made_from.items():
            read.readers.append((node, places))
    return tree, node_at


def _add_to_tree(graph: LoadedGraph, embedding: GraphAtPath | None, tree: list[FunctionAtPath], node_at: dict[str, NodeAtPath]) -> dict[str, NodeAtPath]:
    """Add the functions of ``graph`` to ``tree``, each after the functions it reads, those of each graph
    embedded in it where that graph's node stands, and every node to ``node_at``.

    A plan of the loaded graph names the nodes of its own graph. Here each is written for where its node stands
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
    functions = typing.cast("dict[str, FunctionAtPath]", node_at)  # a cell of a plan at a path names a function
    for placed in graph._nodes.values():
        inputs: dict[str, InputPlan] = {}
        for name, plan in placed.inputs.items():
            if embedding is not None and isinstance(plan, Open):  # the embedded graph's node is given it, as a caller would give it
                inputs[name] = embedding.inputs[graph._input_fields[plan.address].name]
            else:
                inputs[name] = _plan_at_path(plan, here, functions)
        maps = (*above, *(series for port in placed.scope for series in _maps_of(port, here)))
        path = (*inside, placed.id)
        if isinstance(placed, PlacedGraph):
            embedded = GraphAtPath(placed=placed, embedding=embedding, path=path, name=node_name(path), maps=maps, inputs=inputs)
            within = _add_to_tree(placed.inner, embedded, tree, node_at)
            embedded.nodes = list(within.values())
            for address in placed.inner.outputs:
                made_by, _, output = address.partition(".")
                last = within[made_by]
                embedded.offers[address] = last.offers[output] if isinstance(last, GraphAtPath) else (last, output)
            here[placed.id] = node_at[embedded.name] = embedded
            continue
        call_template: dict[str, Any] = {}
        from_caller: list[tuple[str, str]] = []
        fed: list[tuple[str, Fed, bool]] = []
        made_from: dict[FunctionAtPath, int] = {}
        gathering: dict[tuple[Cell, ...], list[tuple[str, bool]]] = {}
        for name, plan in inputs.items():
            call_template[name] = None
            match plan:
                case Fixed(value=value):
                    call_template[name] = value
                    continue
                case Open(address=address):
                    from_caller.append((name, address))
                    continue
                case Unset():
                    raise RuntimeError(f"'{node_name(path)}' has an input with no value to give: a graph with problems is never walked")
            read_at = [(functions[cell.node_id], min(cell.depth, len(plan.scope))) for cell in plan.reads]  # no deeper than the input sits
            if isinstance(plan, Gather):
                series = plan.series
                items_of = (functions[series.node_id], series.output)  # what the functions it gathers through read outside the items can settle it too
                read_at += [(read, places) for inner in plan.through for read, places in inner.made_from.items() if items_of not in read.maps]
                gathering.setdefault(plan.reads[: plan.levels], []).append((name, plan.spec.keeps_blanks))
                whole = Value(series.node_id, series.output, series.scope, None, functions[series.node_id].placed.outputs[series.output].whole)
                fed.append((name, Read(plan.spec, (whole,), series.scope), plan.spec.whole.optional))  # it needs its Series, and then its items
            fed.append((name, plan, plan.spec.whole.optional))
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


def _made_by(port: Port, here: Mapping[str, NodeAtPath]) -> tuple[FunctionAtPath, str, bool]:
    """The function that makes the output ``port`` names, and its output there: the node's own, or for an
    embedded graph's node the function inside that makes what it offers; with whether that function runs per
    item inside the embedded graph, so what is offered is every item's value."""
    made_by = here[port.node_id]
    if isinstance(made_by, GraphAtPath):
        last, output = made_by.offers[port.output]
        return last, output, len(last.maps) > len(made_by.maps)
    return made_by, port.output, False


def _maps_of(port: Port, here: Mapping[str, NodeAtPath]) -> Maps:
    """The maps the Series ``port`` names opens for what reads it per item: the Series itself; for one an
    embedded graph offers of an output made per item inside, the maps inside that it is made in."""
    last, output, per_item = _made_by(port, here)
    return last.maps[len(here[port.node_id].maps) :] if per_item else ((last, output),)


def _scope_at_path(maps: Maps) -> Scope:
    """``maps`` as the scope of a plan at a path, each Series named by the name of its function."""
    return tuple([Port(series.name, output) for series, output in maps])


def _plan_at_path(plan: InputPlan, here: Mapping[str, NodeAtPath], functions: Mapping[str, FunctionAtPath]) -> InputPlan:
    """A plan of a loaded graph as the plan of its node where it stands: each cell names the function that
    holds it by its ``name``, in the maps that function runs in.

    An output of an embedded graph is the output of a function inside it. One made once inside is that cell,
    read as the edge reads it. One made per item inside is offered as the ``Series`` of every item's value, so
    an item of it is the inner unit's own value (a ``Series`` there read as the list it is), and an input that
    gathers it gathers every map inside that it is made in.

    Args:
        plan: The plan, as its graph loaded it.
        here: The nodes of that graph placed so far, where they stand, by id.
        functions: Every function placed so far, by its name.
    """

    def at_path(source: Source, opened: bool) -> Source:
        last, output, per_item = _made_by(Port(source.node_id, source.output), here)
        spec = last.placed.outputs[output]
        brought: type[Source] = type(source) if not per_item else WholeSeries if spec.many else Value
        if opened and brought is Item:
            last.opens.setdefault(output, (here[source.node_id], source.output))
        return brought(last.name, output, _scope_at_path(last.maps), source.field, source.type)

    match plan:
        case Read(spec=spec, reads=reads):
            brings = tuple([at_path(source, True) for source in reads])
            return Read(spec, brings, max((source.read_in for source in brings), key=len))
        case SeriesOfOne(spec=spec, reads=reads):
            return SeriesOfOne(spec, tuple([typing.cast(Value, at_path(source, False)) for source in reads]))
        case Gather(spec=spec, series=series, sources=sources):
            return _gather_at_path(spec, _maps_of(Port(series.node_id, series.output), here), tuple([at_path(source, False) for source in sources]), functions)
        case _:
            return plan


def _gather_at_path(spec: Field, gathered: Maps, sources: tuple[Source, ...], functions: Mapping[str, FunctionAtPath]) -> Gather:
    """The plan of an input that gathers the items of ``gathered``, where its node stands: the one way such a
    plan is made, for a node's own input and for what a host is shown.

    Args:
        spec: The input's field.
        gathered: The maps gathered through, outermost first: one, but for an output an embedded graph offers
            of something made several maps deep inside it.
        sources: What each edge brings for an item, each naming a function by its name.
        functions: The functions the plan's cells name, by name.
    """
    levels = [Cell(made_by.name, output, _scope_at_path(made_by.maps)) for made_by, output in gathered]
    through = _gathered_through(gathered[0], [functions[cell.node_id] for cell in (*levels[1:], *sources)])
    return Gather(spec, (*levels, *sources), len(levels), through)


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
        case Read(spec=spec, sources=sources):
            return not all(spec.whole.holds(s.type) for s in sources)
        case Gather(spec=spec, sources=sources) | SeriesOfOne(spec=spec, sources=sources):
            return spec.whole.record is not None or not all(spec.type.holds(s.type) for s in sources)
        case _:
            return False
