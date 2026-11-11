"""``Graph``: a stored graph checked against the nodes it uses, ready to run; its placed nodes, and the tree a run walks.

A table is one value. An embedded graph is one node: its inputs are the inner
graph's inputs, its outputs the inner graph's outputs. It loads once, alone,
however many graphs embed it, and nothing of it is copied into the graph
around it. What a run walks is the tree those graphs make: every node of the
graph and of each graph embedded in it, each a ``NodeAtPath``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.address import NodePath
from conductor._internal.ask import output_questions
from conductor._internal.canonical_json import canonical_json
from conductor._internal.fields import REQUIRED, Field, FieldDescription, FieldValueError
from conductor._internal.graph_record import GraphRecord
from conductor._internal.loaded_graph.describe import (
    GraphDescription,
    GraphInput,
    GraphNodeDescription,
    PlacedFieldDescription,
    RunsOn,
    _feed,
    _runs_on,
)
from conductor._internal.loaded_graph.input_plans import (
    Cell,
    CellRead,
    Fed,
    Gather,
    InputPlan,
    Open,
    Scope,
    _read_before_a_call,
)
from conductor._internal.node_definition import FunctionDefinition, GraphDefinition, Policy
from conductor._internal.problem import InvalidCallError, Problem, make_problem

if TYPE_CHECKING:
    from conductor._internal.run import Run, SaveForm


@dataclass(frozen=True, kw_only=True)
class _Placed:
    """What every node of a loaded graph has, whether it is a function or an embedded graph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        scope: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
        opens: The ``Series`` outputs some reader works through one item at a time, which a run holds to
            ``max_items``. A reader placed after this node adds to it.
    """

    id: str
    scope: Scope
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]
    opens: set[str] = dataclasses.field(default_factory=set[str])

    @functools.cached_property
    def reads(self) -> frozenset[str]:
        """The nodes its inputs read."""
        return frozenset(source.node_id for plan in self.inputs.values() if isinstance(plan, Fed) for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.spec.type.hashes_plainly for plan in self.inputs.values())


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedFunction(_Placed):
    """A node of a loaded graph that is a function.

    Attributes:
        definition: The node as its author declared it.
    """

    definition: FunctionDefinition[..., Any]

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, its code, its fields and its scope.

        So a node redeclared under the same id and version, with other code, a field of another type, limit or
        choice, or one that runs in another scope, is called again by ``run.again``. Code is read from the function's own bytecode and
        constants, not from what it calls.
        """
        specs = (*(plan.spec for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.whole.reads_as) for spec in specs]
        definition = self.definition
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, self.scope))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.outputs)


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedGraph(_Placed):
    """A node of a loaded graph that is an embedded graph: the nodes of ``inner`` run where it stands.

    Its inputs are the inner graph's (``inner.inputs``) by their names there, and its outputs the inner graph's
    (``inner.outputs``) by their addresses there. It has no policy and no unit of it is reused whole: the nodes
    inside are called by their own policies, and a later run takes their earlier work node by node, as it would
    with the same nodes drawn in the graph itself.

    Attributes:
        definition: The node as its author declared it.
        inner: The embedded graph, loaded.
    """

    definition: GraphDefinition
    inner: Graph


#: One node of a loaded graph: a function, or an embedded graph.
PlacedNode = PlacedFunction | PlacedGraph


@dataclass(eq=False, slots=True, kw_only=True)
class _AtPath:
    """A node where it stands in the tree of graphs a run walks: a node of the graph itself, or of a graph
    embedded in it, at any depth. Never made on its own: it is a ``FunctionAtPath`` or a ``GraphAtPath``.

    A graph loads once and may be embedded in many places; this is a node of it in one of them. A node inside an
    embedded graph runs once for each item its embedding node runs for, and then once per item of its own scope,
    so the maps it runs in are the embedding node's and then its own.

    Attributes:
        embedding: The embedded graph's node it is directly inside; ``None`` for a node of the graph itself.
        inside: The ids of the embedded graphs' nodes it is inside, outermost first; ``()`` for a node of the
            graph itself.
        path: ``inside``, then its own id.
        maps: Every map it runs in, outermost first, each as the node and the output of its ``Series``: those of
            the embedded graph's node it is inside, then those of its own scope.
        base: How many of ``maps`` are the embedded graph's node's.
        inputs: How each input gets its value where the node stands, by name: its plan; the embedded graph's
            node in whose graph that plan reads, ``None`` for the node's own graph; and the cells the plan
            reads (its ``reads``), each with how many places of an instance of this node say which unit holds
            it. A run reads an input's cells from here and nowhere else, and ``made_from`` is made of the
            same. An open input of a node inside an embedded graph has what that graph's node has for it.
        made_from: Every node whose units its own are made from, each with how many places of an instance of this
            node say which of that node's units: the unit at those places, and where that node runs deeper every
            unit under them. They are the cells its inputs read (``inputs``); for a gathering
            input also what the nodes it gathers through read outside the items, which can settle it for
            every item at once; for an embedded graph's node, the cells it offers and the Series they are made
            per item of; and, for a node that reads anything from outside the innermost map it runs in, or
            reads nothing, the Series of that map, since what it reads may be there before its instances are.
        readers: Each node whose ``made_from`` holds this one, with the number held there.
        position: Its place in the order a leg tries the nodes of the tree in.
        waits_on: Every node it waits on, however far up; filled when first asked for.
    """

    embedding: GraphAtPath | None
    inside: NodePath
    path: NodePath
    maps: Maps
    base: int
    inputs: dict[str, tuple[InputPlan, GraphAtPath | None, tuple[CellRead, ...]]]
    made_from: dict[NodeAtPath, int] = dataclasses.field(default_factory=dict["NodeAtPath", int])
    readers: list[tuple[NodeAtPath, int]] = dataclasses.field(default_factory=list[tuple["NodeAtPath", int]])
    position: int = 0
    waits_on: frozenset[NodeAtPath] | None = None


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class FunctionAtPath(_AtPath):
    """A function node where it stands in the tree of graphs a run walks.

    Attributes:
        placed: The node, as its graph loaded it.
        policy: The node's policy, as its definition declares it.
        checked: The inputs whose value is read by the input's type before a call, as the types the edges bring
            don't settle it: the node's own, and the open ones an embedded graph's node is handed by such edges.
        gathered_through: For each gathering input, by name, the nodes made per item of its Series that what it
            gathers rests on, in the tree's order: what stands between the input and everything outside the
            items. A run asks them, in this order, whether any item could bring the input a value; none when
            the Series itself is all that has a say.
    """

    placed: PlacedFunction
    policy: Policy
    checked: tuple[str, ...]
    gathered_through: dict[str, tuple[NodeAtPath, ...]] = dataclasses.field(default_factory=dict[str, tuple["NodeAtPath", ...]])


@typing.final
@dataclass(eq=False, slots=True, kw_only=True)
class GraphAtPath(_AtPath):
    """An embedded graph's node where it stands in the tree of graphs a run walks. It is never called: its unit
    is what the outputs of the graph inside it came to.

    Attributes:
        placed: The node, as its graph loaded it.
        offers: Each output it offers, by its address in the inner graph: the cell of the inner graph that holds
            it, the maps that cell's node runs in, whether it is a ``Series``, and for an output made per item
            inside the Series, of the inner graph's own scope, that those items are of; ``None`` for an output
            made once inside.
        ends_silently: The nodes inside that nothing inside reads and that have no output. Its unit waits for
            theirs as for the outputs it offers, so it is done when the graph inside is.
    """

    placed: PlacedGraph
    offers: dict[str, tuple[Cell, Maps, bool, Cell | None]] = dataclasses.field(default_factory=dict)
    ends_silently: tuple[NodeAtPath, ...] = ()


#: A node where it stands in the tree of graphs a run walks: a function, or an embedded graph.
NodeAtPath = FunctionAtPath | GraphAtPath

#: The maps a node runs in, outermost first, each as the node and the output of its ``Series``.
Maps = tuple[tuple["NodeAtPath", str], ...]


def waits_on(node: NodeAtPath) -> frozenset[NodeAtPath]:
    """Every node ``node`` waits on, however far up; worked out when first asked, as most nodes never are."""
    if node.waits_on is None:
        seen: set[NodeAtPath] = set()
        stack = [node]
        while stack:
            for read in stack.pop().made_from:
                if read not in seen:
                    seen.add(read)
                    stack.append(read)
        node.waits_on = frozenset(seen)
    return node.waits_on


@dataclass(frozen=True, slots=True)
class InputField:
    """What a run keeps of a graph input.

    Attributes:
        name: Its name in ``graph.inputs``: the input's own, or its address where two inputs share a name.
        address: Where the caller's value is kept, ``node.input``.
        spec: Its field, whose default is what it holds when the caller gives nothing: the record's default, else
            the node's, and ``REQUIRED`` when the caller must give one.
    """

    name: str
    address: str
    spec: Field


@dataclass(frozen=True, eq=False)
class Graph:
    """A graph loaded against a registry: ``registry.load(record)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``.

    Attributes:
        record: The stored graph it was loaded from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node whose fields are known, by id, each after the nodes its inputs read. A graph with problems holds
    #: the nodes they are on too, with ``Unset`` for an input that gets nothing, and never runs.
    _nodes: dict[str, PlacedNode]
    #: Each graph input as a run keeps it, by address.
    _input_fields: dict[str, InputField]
    #: The nodes that read each node.
    _readers: dict[str, list[PlacedNode]]
    #: The nodes with a ``FromRun`` parameter, an embedded graph's too, as ``from_run_values`` reads them.
    _needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        return f"<Graph of {len(self.record.nodes)} node{'s' if len(self.record.nodes) != 1 else ''}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @functools.cached_property
    def outputs(self) -> list[str]:
        """The outputs of the nodes nothing reads, as addresses."""
        return [f"{p.id}.{o}" for p in self._nodes.values() if not self._readers[p.id] for o in p.outputs]

    @property
    def plan(self) -> dict[str, RunsOn]:
        """Where each node runs: ``once``, or ``per_item`` of the innermost Series of its scope."""
        return {placed.id: _runs_on(placed.scope) for placed in self._nodes.values()}

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for placed in self._nodes.values():
            for name, spec in placed.outputs.items():
                if spec.param.exclusive_group:
                    found.setdefault(placed.id, {}).setdefault(spec.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> GraphDescription:
        """The loaded graph as an editor draws it: every node of the record, with its fields and where it runs.

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node with a
        problem is drawn with its fields all the same, an input whose edge is refused fed ``unset``, and so is a
        node that reads it. A ``type_mismatch`` names the input in its ``loc``; a ``cycle`` and
        ``series_not_nested`` name the node. Only a node whose fields aren't known is there with ``loaded``
        false and no fields, its problems saying why: an unknown node or version, an id no address can say, a
        graph that embeds itself, a hook that failed or wasn't called for a value it reads, an embedded graph
        with problems of its own. An embedded graph is one node, with the fields it offers; one with problems
        isn't placed, and its problems are this graph's, each ``loc`` the embedding node's id and then the
        place inside it.
        """
        nodes: dict[str, GraphNodeDescription] = {}
        for record in self.record.nodes:
            own = [p for p in self.problems if p.loc[:1] == (record.id,)]
            placed = self._nodes.get(record.id)
            if placed is None:
                nodes[record.id] = GraphNodeDescription(node=record.node, version=record.version, loaded=False, problems=own)
                continue
            nodes[record.id] = GraphNodeDescription(
                node=record.node, version=record.version, loaded=True, problems=own, runs=_runs_on(placed.scope),
                inputs=[PlacedFieldDescription(field=plan.spec.description, feed=_feed(placed.scope, plan)) for plan in placed.inputs.values()],
                outputs=[spec.description for spec in placed.outputs.values()],
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: the record, but for each node's ``display``, which
        conductor never reads, and the fingerprint of every graph embedded in it.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        A saved run names it, and loading a save into a graph whose fingerprint differs is refused.
        """
        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        embedded = {p.id: p.inner.fingerprint for p in self._nodes.values() if isinstance(p, PlacedGraph)}
        return hashlib.blake2b(canonical_json([own, embedded]), digest_size=8).hexdigest()

    # -- running ----------------------------------------------------------------

    def with_inputs(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None, earlier: Run | None = None, **by_name: Any,
    ) -> Run:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Args:
            inputs: The inputs by name (``graph.inputs``) or by address (``node.input``), for names Python can't
                take as keywords (``{"a.text": ...}``).
            max_items: The most items any one Series a node works through may hold before the run stops.
            earlier: A run of this graph, or of an earlier version of it, that lends its work as ``run.again``
                does: a node with the same id, version, code, fields and scope is not called where it receives
                what it received there, so after an edit only the nodes it changes, and what they feed, run again.
            **by_name: The inputs, as keywords.

        Returns:
            The run, ``ready``.

        Raises:
            InvalidCallError: The graph has problems, or an input is missing, unknown or not of its type.
        """
        from conductor._internal.run import checked_leg, no_units, start_run

        if self.problems:
            raise InvalidCallError(self.problems)
        checked_leg(max_items=max_items)
        checked = self._checked_inputs(*self._given_by_address({**inputs, **by_name}))
        return start_run(self, checked, max_items, earlier._carried(into=self) if earlier is not None else {}, no_units(self))

    def run(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None,
        from_run: Mapping[type, Any] = types.MappingProxyType({}), timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote; a graph with problems refuses it with them.

        A save is read from its JSON text, a mapping written as JSON first, so each value is read as its type
        reads JSON.
        """
        from conductor._internal.run import run_from_json

        return run_from_json(self, saved)

    @functools.cached_property
    def _save_form(self) -> SaveForm:
        """How this graph's runs are read from JSON and written as it, made when the first one is."""
        from conductor._internal.run import save_form

        return save_form(self)

    @functools.cached_property
    def _tree(self) -> list[NodeAtPath]:
        """Every node of the graph and of each graph embedded in it, at any depth, each after the nodes it reads:
        the order a leg tries them in. An embedded graph's node comes after the nodes inside it."""
        tree: list[NodeAtPath] = []
        _add_to_tree(self, None, tree, {})
        for node in tree:
            for read, places in node.made_from.items():
                read.readers.append((node, places))
        return tree

    @functools.cached_property
    def _ids_inside(self) -> dict[NodePath, list[str]]:
        """The ids of the nodes of each graph of ``_tree``, by the path of the node that embeds it, ``()`` for
        the graph itself: what a run keeps units by."""
        ids: dict[NodePath, list[str]] = {}
        for node in self._tree:
            ids.setdefault(node.inside, []).append(node.placed.id)
            if type(node) is GraphAtPath:
                ids.setdefault(node.path, [])  # a graph of no nodes is a graph too
        return ids

    @functools.cached_property
    def _node_at(self) -> dict[NodePath, NodeAtPath]:
        """Each node of ``_tree``, by its path."""
        return {node.path: node for node in self._tree}

    @functools.cached_property
    def _functions(self) -> dict[NodePath, PlacedFunction]:
        """Every function node of ``_tree``, by its path."""
        return {node.path: node.placed for node in self._tree if isinstance(node, FunctionAtPath)}

    @functools.cached_property
    def _threads(self) -> int:
        """How many calls a leg over the graph can have on threads at once: the ``concurrency`` of every node
        called on a thread, those of each embedded graph with them."""
        return sum(policy.concurrency for placed in self._functions.values() if (policy := placed.definition.policy).threaded)

    def _given_by_address(self, given: Mapping[str, Any]) -> tuple[dict[str, Any], list[Problem]]:
        """What a caller gave by name or by address, by address; with a problem for each name the graph doesn't
        offer and each input given by both its name and its address."""
        fields = {**{f.address: f for f in self._input_fields.values()}, **{f.name: f for f in self._input_fields.values()}}
        by_address: dict[str, Any] = {}
        problems: list[Problem] = []
        for name, value in given.items():
            offered = fields.get(name)
            if offered is None:
                problems.append(make_problem("not_offered", "inputs", name, name=name, inputs=list(self.inputs)))
            elif offered.address in by_address:
                problems.append(make_problem("input_given_twice", "inputs", offered.name, name=offered.name, address=offered.address))
            else:
                by_address[offered.address] = value
        return by_address, problems

    def _checked_inputs(self, given: Mapping[str, Any], problems: list[Problem]) -> dict[str, Any]:
        """The value of every graph input, by address: the caller's, read by the input's field, else its default.

        Raises:
            InvalidCallError: With ``problems`` and then every input that is missing or not of its type.
        """
        problems, values = list(problems), dict[str, Any]()
        for address, offered in self._input_fields.items():
            if address not in given:
                if offered.spec.default is REQUIRED:
                    problems.append(make_problem("missing_input", "inputs", offered.name, name=offered.name))
                values[address] = offered.spec.default
                continue
            value, spec = given[address], offered.spec
            try:
                values[address] = spec.whole.read(value)
            except FieldValueError as refused:
                whole = all(not error["loc"] for error in refused.ctx.get("errors", ()))  # the list itself was refused, not an item of it
                if not spec.many and type(value) is list and whole:  # the one mistake whose fix is in the graph
                    problems.append(make_problem(
                        "invalid_input", "inputs", offered.name, name=offered.name, expected=spec.whole.id, length=len(value), hint="make_a_series",
                    ))
                else:
                    problems.append(make_problem("invalid_input", "inputs", offered.name, name=offered.name, reason=refused.reason, **refused.ctx))
        if problems:
            raise InvalidCallError(problems)
        return values


def _add_to_tree(graph: Graph, embedding: GraphAtPath | None, tree: list[NodeAtPath], beside: dict[NodePath, dict[str, NodeAtPath]]) -> None:
    """Add the nodes of ``graph`` to ``tree``, each after the nodes it reads, and those of each graph embedded
    in it before that graph's node.

    Args:
        graph: The graph: the one a run is of, or one embedded in it.
        embedding: The embedded graph's node that ``graph`` is the inner graph of; ``None`` for the run's own.
        tree: The nodes so far, added to.
        beside: The nodes of each graph placed so far by id, by the path of the node that embeds it, added to.
    """
    inside = embedding.path if embedding is not None else ()
    above = embedding.maps if embedding is not None else ()
    here = beside[inside] = {}
    for placed in graph._nodes.values():
        inputs: dict[str, tuple[InputPlan, GraphAtPath | None, tuple[CellRead, ...]]] = {}
        for name, plan in placed.inputs.items():
            if embedding is not None and isinstance(plan, Open):  # the embedded graph's node is given it, as a caller would give it
                handed, giver, cells = embedding.inputs[graph._input_fields[plan.address].name]
                inputs[name] = (handed, giver or embedding, cells)
            else:
                inputs[name] = (plan, None, tuple([(cell, len(above) + depth) for cell, depth in plan.reads]) if isinstance(plan, Fed) else ())
        maps = (*above, *((here[port.node_id], port.output) for port in placed.scope))
        if isinstance(placed, PlacedGraph):
            node: NodeAtPath = (embedded := GraphAtPath(
                placed=placed, embedding=embedding, inside=inside, path=(*inside, placed.id), maps=maps, base=len(above), inputs=inputs,
            ))
            _add_to_tree(placed.inner, embedded, tree, beside)
            within = beside[node.path]
            for address in placed.inner.outputs:
                last, output = within[address.partition(".")[0]], address.partition(".")[2]
                spec = last.placed.outputs[output]
                series = None
                if len(last.maps) > len(maps):  # made per item inside: of the first Series the inner graph makes on the way to it
                    maker, of = last.maps[len(maps)]
                    series = Cell(maker.placed.id, of, (), maker.placed.outputs[of].param.exclusive_group is not None)
                embedded.offers[address] = (Cell(last.placed.id, output, last.placed.scope, spec.param.exclusive_group is not None), last.maps, spec.many, series)
            embedded.ends_silently = tuple(within[last.id] for last in placed.inner._nodes.values() if not last.outputs and not placed.inner._readers[last.id])
            made_from = [(within[cell.node_id], made_in) for cell, made_in, _, _ in embedded.offers.values()] + [(last, last.maps) for last in embedded.ends_silently]
            for last, made_in in made_from:  # each unit of it is made from every unit inside, under its instance
                for read in (last, *(series for series, _ in made_in[len(maps):])):
                    node.made_from[read] = len(maps)
        else:
            node = FunctionAtPath(
                placed=placed, embedding=embedding, inside=inside, path=(*inside, placed.id), maps=maps, base=len(above), inputs=inputs,
                policy=placed.definition.policy, checked=tuple(name for name, (plan, _, _) in inputs.items() if _read_before_a_call(plan)),
            )
            for name, (plan, giver, cells) in inputs.items():
                within = here if giver is None else beside[giver.inside]
                read_at = [(within[cell.node_id], places) for cell, places in cells]
                if isinstance(plan, Gather):
                    (series, _), *edges = cells
                    node.gathered_through[name], outside = _gathered_through((within[series.node_id], series.output), [within[source.node_id] for source, _ in edges])
                    read_at += outside.items()
                for read, places in read_at:
                    node.made_from[read] = min(places, node.made_from.get(read, places))
        if maps and min(node.made_from.values(), default=0) < len(maps):  # what it reads may be there before its instances are
            node.made_from.setdefault(maps[-1][0], len(maps) - 1)
        here[placed.id] = node
        node.position = len(tree)
        tree.append(node)


def _gathered_through(series: tuple[NodeAtPath, str], sources: list[NodeAtPath]) -> tuple[tuple[NodeAtPath, ...], dict[NodeAtPath, int]]:
    """Finds the nodes made per item of ``series`` that a gathering input rests on, by what each one's units are
    made from (``made_from``), and what those nodes read outside the items.

    Args:
        series: The Series the input gathers: the node that makes it, and its output.
        sources: The nodes whose outputs the input's edges bring.

    Returns:
        The nodes, in the tree's order, each after the ones it reads; and each node outside the items that
        one of them reads, with how many places of an instance say which of its units. No nodes when all they
        read outside the items is the Series, from a node whose one output it is, and none of them has an
        output of an exclusive group: the input then comes to what the Series itself is.
    """
    made_by = series[0]
    through: set[NodeAtPath] = set()
    outside: dict[NodeAtPath, int] = {}
    only_the_series = len(made_by.placed.outputs) == 1
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.maps and inner not in through:
            through.add(inner)
            if any([spec.param.exclusive_group is not None for spec in inner.placed.outputs.values()]):
                only_the_series = False  # not called for a Series never made, it chooses no branch: a skip, not the Series' blank
            for read, places in inner.made_from.items():
                if series in read.maps:
                    stack.append(read)
                else:
                    outside[read] = min(places, outside.get(read, places))
                    only_the_series = only_the_series and read is made_by
    return (() if only_the_series else tuple(sorted(through, key=lambda inner: inner.position))), outside


def _code_text(code: Any) -> str:
    """A function's bytecode and constants as text that is the same in every process: a nested function by its own code."""
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in code.co_consts]}:{code.co_names}"


def _constant_text(constant: Any) -> str:
    """A constant as text that is the same in every process: a set's members sorted, as its order follows the hash seed."""
    if hasattr(constant, "co_code"):
        return _code_text(constant)
    if isinstance(constant, frozenset):
        return f"frozenset({sorted(_constant_text(c) for c in constant)})"
    if isinstance(constant, tuple):
        return f"({', '.join(_constant_text(c) for c in constant)},)"
    return repr(constant)
