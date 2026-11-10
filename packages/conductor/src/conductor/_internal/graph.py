"""``Graph``: a stored graph checked against the nodes it uses, ready to run.

Loading does everything once: embedded graphs are spliced in as plain nodes,
every edge and value is checked, and each node gets the rows it runs on. What
is wrong becomes a ``Problem`` with a place, never an exception.

Rows. Every value sits on a row of a table. The graph's own table has one
row, ``()``. An output of many values (``Series`` or ``Table``) makes a new
table under the row its node ran on; a row of it is its parent's key plus a
number, ``(0, 3)``. A table is named by the output that makes it,
``"split.result"``, and a level is the path of table names from the graph's
own row, so a level and a key always have the same length.

A node runs once per row of its level:

- an input of ``one`` value fed from deeper rows runs the node on those rows,
  and a value from rows above is copied down;
- a ``Series`` or ``Table`` input gathers the rows under the node's row, or
  under the row it shares with a sibling table; it chooses the level, one up
  from its source, only when no ``one`` input does. Several sources on one
  table are read in that table's row order;
- ``one``-value sources must lie on one path of the row tree. Rows of two
  different tables meet only through a join the author draws. Several edges
  into one ``one`` input must be alternatives: on opposite branches of one
  decision, so a row holds at most one of them.

A list given to an input that takes one value (``FieldSpec.per_item``), by
the caller or typed in, is a table of its items, and the node runs once per
item. A caller's list is a table of its own, so two of them meeting at one
node are rows of different tables (``lists_not_collected``) until a node
collects one; a ``Series`` input collects a caller's list whole, wherever its
node runs. A typed-in list is a table under each row its node runs on, a row
a node made, never another list's.
Which inputs a caller gives lists decides the rows, so a graph compiles once
for each such set, on first use.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import heapq
import inspect
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from conductor._internal.ask import output_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import (
    PROBLEM_CATALOGUE,
    InvalidCallError,
    NodeError,
    Problem,
    make_problem,
)
from conductor._internal.fields import (
    REQUIRED,
    FieldDescription,
    FieldSpec,
    Refused,
    accepts,
    already,
    given_as_items,
    is_a_number,
    read_field,
    read_given,
    takes_a_blank,
    type_id,
)
from conductor._internal.node_definition import NodeDefinition, is_node_id
from conductor._internal.record import GraphRecord, NodeRecord

if TYPE_CHECKING:
    from conductor._internal.run import Run

#: A path of table names from the graph's own row.
Level = tuple[str, ...]



@dataclass(frozen=True)
class Source:
    """Where one edge reads: a node's output (``node_id`` None for a graph input) at ``level``."""

    node_id: str | None
    output: str
    column: str | None
    level: Level
    #: One item of a list or table made at ``level``; its unit sits on the parent row.
    made: bool
    type: Any
    whole_table: bool = False
    #: The column it becomes in a table input.
    name: str = ""
    #: For a whole table, the columns its row type names; ``None`` when they are known only when it arrives.
    columns: Mapping[str, Any] | None = None
    #: For a gather, how many places of the node's row its rows are gathered under: the row both share.
    anchor: int = 0


@dataclass
class InputPlan:
    """How one input of a placed node gets its value: its sources, or a value typed in."""

    spec: FieldSpec
    sources: list[Source] = field(default_factory=list)
    value: Any = REQUIRED
    #: For a table input, the level whose rows it gathers.
    table_level: Level = ()
    #: How many places of the node's row the table's rows are gathered under: the row both share.
    table_anchor: int = 0
    #: For a table input fed several whole tables on opposite branches: it takes the one whose branch ran.
    alternatives: bool = False


@dataclass
class PlacedNode:
    """One node of a loaded graph: its definition as its hooks made it, and where it runs."""

    id: str
    definition: NodeDefinition[..., Any]
    record: NodeRecord
    level: Level = ()
    inputs: dict[str, InputPlan] = field(default_factory=dict)
    #: Each output of many values, and the level its rows are.
    makes: dict[str, Level] = field(default_factory=dict)
    #: The inputs whose value an edge brings is read by the input's type before a call, as its sources' types don't settle it.
    checked: tuple[str, ...] = ()

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, its code, its fields and its rows.

        So a node redeclared under the same id and version, with other code or a signature that takes a blank or
        runs on other rows, is new work on ``run.again``. Code is read from the function's own bytecode and
        constants, not from what it calls.
        """
        definition = self.definition
        fields = [(name, spec.shape, spec.optional, _type_text(spec.type)) for name, spec in (*definition.inputs.items(), *definition.outputs.items())]
        code = getattr(definition.function, "__code__", None)
        written = repr((definition.id, definition.version, _code_text(code) if code is not None else "", fields, self.level))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping (each is a table, or of plain classes), so what a unit received needs no key sorting to hash."""
        return all(
            spec.shape == "table" or spec.plain for spec in self.definition.inputs.values()
        )

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.definition)


class RunsOn(ConductorModel):
    """Where a node runs: ``once``, or ``per_row`` of the ``table`` named; ``msg`` says it in English."""

    mode: Literal["once", "per_row"]
    table: str | None = None
    msg: str


class Feed(ConductorModel):
    """How an input is fed on each row its node runs on; ``msg`` says it in English.

    A ``single`` value; ``per_row`` of its ``tables``, a value per row; ``copied_down`` from a table above the node's
    rows; or the ``whole_column`` of its ``tables`` under each of the node's rows, ``under`` each row of the table a
    sibling table shares with the node. In ``tables``, ``None`` is the graph's own row.
    """

    mode: Literal["single", "per_row", "copied_down", "whole_column"]
    tables: list[str | None] = []
    under: str | None = None
    msg: str


class PlacedFieldDescription(FieldDescription):
    """An input of a node in a loaded graph: its field, and how it is fed (``feed``) on each row its node runs on."""

    feed: Feed


class GraphNodeDescription(ConductorModel):
    """One node of a loaded graph as an editor draws it: the definition it uses, whether it ``loaded``, its own
    ``problems``, and once loaded where it ``runs`` and its fields as its hooks made them."""

    node: str
    version: int
    loaded: bool
    problems: list[Problem]
    runs: RunsOn | None = None
    inputs: list[PlacedFieldDescription] = []
    outputs: list[FieldDescription] = []


class GraphDescription(ConductorModel):
    """A loaded graph as an editor draws it: each node of the record by id, the graph's inputs and outputs, and every problem."""

    nodes: dict[str, GraphNodeDescription]
    inputs: dict[str, GraphInput]
    outputs: list[str]
    problems: list[Problem]


class GraphInput(FieldDescription):
    """An open input of a first node, a value of ``graph.inputs``: what a caller fills, at ``address`` (``node.input``).

    It is its node's field record, with the ``address`` and ``node_id`` a caller fills; ``name`` is the input's own.
    ``per_item`` says a list given here is many values, run a row per item, as the input takes one value of a type
    that takes no list; anything else given is one value. For an embedded graph's input, ``node_id`` is the node that
    embeds it and ``name`` the name it offers. ``default`` is what it holds when the caller gives nothing (the
    record's default, else the node's), as JSON, and ``None`` when it is ``required``.
    """

    address: str
    node_id: str
    per_item: bool


@dataclass(frozen=True, slots=True)
class InputField:
    """What a run keeps of a graph input: where it keeps the caller's value (the placed node's address), its name in
    ``graph.inputs`` and its address there, its field, its Python default, and whether this compile takes a list of
    items at it."""

    key: str
    name: str
    address: str
    spec: FieldSpec
    default: Any
    listed: bool


class Graph:
    """A graph loaded against a registry, with its own copy of the definitions it resolved.

    ``problems`` is everything wrong with it, and a graph with none runs:
    ``graph.run(**inputs)``. ``inputs`` are the open inputs of its first
    nodes, by name (``node.input`` where two share a name), and ``outputs`` the
    outputs of the nodes nothing reads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a graph is loaded with registry.load(record)")

    def _load(
        self, record: GraphRecord, definitions: Mapping[str, NodeDefinition[..., Any]], embedded_in: tuple[str, ...], alone: dict[Any, tuple[list[Problem], dict[str, tuple[str, str]], dict[str, str]]],
        listed: frozenset[str] | None,
    ) -> None:
        self.record = record
        self._embedded_in = embedded_in
        #: The open inputs, by the placed node's address, a caller gives a list of items at in a run of this compile;
        #: ``None`` while compiling for a run given nothing, where each input's default decides.
        self._listed: frozenset[str] | None = listed
        #: The compiles this graph made for other lists, by the inputs given lists. Each holds its own, never this
        #: one, so no compile holds a cycle.
        self._compiles: dict[frozenset[str], Graph] = {}
        #: What each embedded graph loaded alone found, by its definition and the graphs it sits in, shared by every graph
        #: this load makes, so each is loaded alone once however deep it nests.
        self._alone = alone
        self.problems: list[Problem] = []
        self._definitions = dict(definitions)
        #: What each embedding drawn in this record offers.
        self._offers: dict[str, tuple[dict[str, tuple[str, str]], dict[str, str]]] = {}
        #: Each embedding at any depth, by its spliced id: the record of the graph it embeds.
        self._embedded: dict[str, GraphRecord] = {}
        #: A placed field inside an embedding as the record holding that embedding names it: ``o/f/x.text`` -> ``o/f.x.text``.
        self._seen_as: dict[str, str] = {}
        placed_records, edges = self._splice(record, "", embedded_in)
        #: Each embedding's edges from outside it into it, which set the rows its nodes run on.
        self._entering = {e: [(a, b) for a, b in edges if b.startswith(e + "/") and not a.startswith(e + "/")] for e in self._embedded}
        #: An inner address as seen from outside: ``e/upper.text`` -> ``e.text``.
        self._outer: dict[str, str] = {}
        for e, (ins, outs) in self._offers.items():
            self._outer.update({f"{n}.{i}": f"{e}.{name}" for name, (n, i) in ins.items()})
            self._outer.update({address: f"{e}.{name}" for name, address in outs.items()})
        self._nodes: dict[str, PlacedNode] = {}
        self.inputs: dict[str, GraphInput] = {}
        #: Each table, by name, and the node and output that make it (``None`` for a graph input).
        self._makers: dict[str, tuple[str | None, str]] = {}
        #: Each list typed into an input, by its address, which names the table of its items.
        self._typed_lists: dict[str, list[Any]] = {}
        #: The given lists whose items go under each row of a table: a caller's under the graph's own row, by that table (``""`` for the graph's own row).
        self._lists_under: dict[str, list[str]] = {}
        self._graph_inputs: list[tuple[GraphInput, str, FieldSpec, Any, bool]] = []
        #: Each graph input as the run keeps it, by the address of the placed node's input.
        self._input_fields: dict[str, InputField] = {}
        #: Each first node's input a hook reads and nothing fills, by its placed address: a caller never fills it, and
        #: a record embedding this graph may type a value into it.
        self._fixed_open: list[str] = []
        self._position = {r.id: n for n, (r, _) in enumerate(placed_records)}
        self._readers: dict[str, list[PlacedNode]] = {r.id: [] for r, _ in placed_records}
        self._above: dict[str, frozenset[str]] = {}
        self._placed: dict[str, list[str]] = {}
        #: The placed nodes an embedding put in its place, at any depth.
        self._spliced: frozenset[str] = frozenset()
        #: Each node's branches: the ways decisions must go for it to hold a value, ``(node, group, output)``.
        self._branches: dict[str, frozenset[tuple[str, str, str]]] = {}
        self._compile(placed_records, edges)
        #: What a reader of this graph's results sees: each drawn node, an embedding as one, and the placed node behind each output it gives.
        self._readable: dict[str, dict[str, tuple[str, str]]] = {}
        for drawn in record.nodes:
            if drawn.id in self._offers:
                outs = self._offers[drawn.id][1]
                self._readable[drawn.id] = {name: (a.partition(".")[0], a.partition(".")[2]) for name, a in outs.items() if a.partition(".")[0] in self._nodes}
            elif drawn.id in self._nodes:
                self._readable[drawn.id] = {o: (drawn.id, o) for o in self._nodes[drawn.id].definition.outputs}
        if not embedded_in:
            self.problems = [self._seen_from_outside(p) for p in self.problems]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        return f"<Graph of {len(self.record.nodes)} node{'s' if len(self.record.nodes) != 1 else ''}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        return not self.problems

    @property
    def outputs(self) -> list[str]:
        """The outputs of the nodes nothing reads, as addresses."""
        return [address for address, _ in self._last_outputs]

    @functools.cached_property
    def _last_outputs(self) -> list[tuple[str, str]]:
        """Each output of the nodes nothing reads: its address in ``outputs``, beside the placed node's address."""
        read = {s.node_id for p in self._nodes.values() for i in p.inputs.values() for s in i.sources}
        return [(self._outward(f"{p.id}.{o}"), f"{p.id}.{o}") for p in self._nodes.values() if p.id not in read for o in p.definition.outputs]

    @functools.cached_property
    def _tables_seen(self) -> dict[str, str]:
        """Each table made inside an embedding that an output it offers sits on, by the first such output: how a reader
        outside names the rows it receives that output on."""
        seen: dict[str, str] = {}
        for e, (_, outs) in self._offers.items():
            for name, address in outs.items():
                placed = self._nodes.get(address.partition(".")[0])
                if placed is not None and placed.level and placed.level[-1].startswith(e + "/"):
                    seen.setdefault(placed.level[-1], f"{e}.{name}")
        return seen

    @property
    def plan(self) -> dict[str, RunsOn]:
        """Where each node runs: ``once``, or ``per_row`` of a table; an embedded graph where the outermost of the nodes
        behind its outputs runs, the rows it is fed on, as a table of its own is no table of this graph."""
        return dict(self._plan)

    @functools.cached_property
    def _plan(self) -> dict[str, RunsOn]:
        """``plan``, made once per load."""
        levels: dict[str, tuple[bool, Level]] = {}
        for p in self._nodes.values():
            node_id = self._outward(p.id)
            # inside an embedding, a node behind an output it offers decides over one behind none
            behind = node_id == p.id or any(self._outward(f"{p.id}.{o}") != node_id for o in p.definition.outputs)
            # an embedding runs on the rows it is fed on: a table made inside it, not a list given at an input it offers, is its own
            level = p.level if node_id == p.id else tuple(t for t in p.level if not t.startswith(node_id + "/") or t in self._outer)
            if node_id not in levels or (behind, -len(level)) > (levels[node_id][0], -len(levels[node_id][1])):
                levels[node_id] = (behind, level)
        plan: dict[str, RunsOn] = {}
        for node_id, (_, level) in levels.items():
            table = self._outward(level[-1], table=True) if level else None
            plan[node_id] = RunsOn(mode="per_row", table=table, msg=f"per row of {table}") if table else RunsOn(mode="once", msg="once")
        return plan

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for drawn, fields in self._readable.items():  # an embedding's by the outputs it offers
            shown = self._groups_shown[drawn]
            for name, (node_id, output) in fields.items():
                group = self._nodes[node_id].definition.outputs[output].param.exclusive_group
                if group:
                    found.setdefault(drawn, {}).setdefault(shown[node_id, group], []).append(name)
        return found

    @functools.cached_property
    def _groups_shown(self) -> dict[str, dict[tuple[str, str], str]]:
        """Each exclusive group behind a drawn node's outputs, by drawn node, then by its placed node and name, as a
        reader of the drawn node names it: by its own name, or, where two nodes inside an embedding hold a group of one
        name, by its node's place inside (``d1/answer``), as each is a decision of its own."""
        shown: dict[str, dict[tuple[str, str], str]] = {}
        for drawn, fields in self._readable.items():
            groups = {
                (node_id, group) for node_id, output in fields.values()
                if (group := self._nodes[node_id].definition.outputs[output].param.exclusive_group)
            }
            names = [group for _, group in groups]
            shown[drawn] = {(n, g): g if names.count(g) == 1 else f"{n[len(drawn) + 1:]}/{g}" for n, g in groups}
        return shown

    def describe(self) -> GraphDescription:
        """The loaded graph as an editor draws it: every node of the record, with its fields as its hooks made them and where it runs.

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node that didn't
        load (an unknown node, a broken edge, or one below it) is there with ``loaded`` false, its problems saying why;
        an embedded graph is one node, with the fields it offers.
        """
        nodes: dict[str, dict[str, Any]] = {}
        plan = self.plan
        for record in self.record.nodes:
            nodes[record.id] = {
                "node": record.node, "version": record.version, "loaded": record.id in self._nodes or record.id in plan,
                "problems": [p for p in self.problems if p.loc[:1] == (record.id,)],
            }
        for p in self._nodes.values():
            if p.id in nodes:
                nodes[p.id].update({
                    "runs": plan[p.id],
                    "inputs": [f.describe(PlacedFieldDescription, feed=self._feed(p, n)) for n, f in p.definition.inputs.items()],
                    "outputs": [f.describe() for f in p.definition.outputs.values()],
                })
        for e, (ins, _) in self._offers.items():
            if e in plan:
                shown = self._groups_shown[e]
                nodes[e].update({
                    "runs": plan[e],
                    "inputs": [
                        self._nodes[n].definition.inputs[i].describe(PlacedFieldDescription, name=name, feed=self._feed(self._nodes[n], i))
                        for name, (n, i) in ins.items() if n in self._nodes
                    ],
                    "outputs": [
                        self._nodes[n].definition.outputs[o].describe(name=name, exclusive_group=shown.get((n, self._nodes[n].definition.outputs[o].param.exclusive_group or "")))
                        for name, (n, o) in self._readable[e].items()
                    ],
                })
        return GraphDescription(
            nodes={node_id: GraphNodeDescription(**fields) for node_id, fields in nodes.items()},
            inputs=dict(self.inputs),
            outputs=self.outputs,
            problems=list(self.problems),
        )

    def _feed(self, placed: PlacedNode, name: str) -> Feed:
        """How one input is fed on each row its node runs on: a ``single`` value, ``copied_down`` from a table above,
        ``per_row`` of a table, or the ``whole_column`` of its tables, ``under`` a row of the table both share for a sibling table."""
        plan = placed.inputs[name]
        if not plan.sources:
            return Feed(mode="single", msg="one value")
        tables = list(dict.fromkeys(self._outward(s.level[-1], table=True) if s.level else None for s in plan.sources))
        named = ", ".join("the graph's own row" if t is None else t for t in tables)
        if plan.spec.shape != "one":
            beside = [s for s in plan.sources if 0 < s.anchor < len(placed.level) and s.anchor < len(s.level)]
            if beside:  # a sibling table, gathered under the row the two share
                under = self._outward(placed.level[beside[0].anchor - 1], table=True)
                return Feed(mode="whole_column", tables=tables, under=under, msg=f"whole column of {named} under each row of {under}")
            return Feed(mode="whole_column", tables=tables, msg=f"whole column of {named}")
        level = plan.sources[0].level
        if len(level) == len(placed.level):
            return Feed(mode="per_row", tables=tables, msg=f"per row of {named}") if level else Feed(mode="single", msg="one value")
        return Feed(mode="copied_down", tables=tables, msg=f"copied down from {named}")

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: each node's id, node, version, values, defaults
        and the edges, of this record and of every graph embedded in it, however deep.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        ``display`` is left out, as conductor never reads it. A saved run names it, and loading a save into a graph
        whose fingerprint differs is refused.
        """
        from conductor._internal.run import canonical_json

        include: Any = {"nodes": {"__all__": {"id", "node", "version", "values", "defaults"}}, "edges": True}
        records = [self.record, *(self._embedded[node_id] for node_id in sorted(self._embedded))]
        written = canonical_json([sorted(self._embedded), *(record.model_dump(mode="json", include=include) for record in records)])
        return hashlib.blake2b(written, digest_size=8).hexdigest()

    # -- running ----------------------------------------------------------------

    def with_inputs(
        self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, earlier: Run | None = None, **by_name: Any,
    ) -> Run:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Inputs are given by name (``graph.inputs``) or by address
        (``node.input``), as keywords or as one mapping for names Python can't
        take as keywords (``{"a.text": ...}``). ``max_rows`` is a ceiling on
        the rows any one table of the run may have. ``earlier``, a run of this
        graph or of an earlier version of it, lends its work as ``run.again``
        does: a node with the same id, version, code, fields and rows is not
        called on a row that receives what it received there, so after an
        edit only the nodes it changes, and what they feed, run again.

        A list given to an input with ``per_item`` is many values, a row per
        item, and the run's ``graph`` is this graph compiled for the lists
        given, so its ``plan`` says where each node ran.
        """
        from conductor._internal.run import checked_leg, start_run

        if self.problems:
            raise InvalidCallError(self.problems)
        checked_leg(max_rows=max_rows)
        by_key, problems = self._given_by_key({**(inputs or {}), **by_name})
        graph = self._compiled_for(self._listed_in(by_key))
        if graph.problems:
            raise InvalidCallError(graph.problems)
        run = start_run(graph, graph._checked_inputs(by_key, problems), max_rows)
        if earlier is not None:
            run._earlier = earlier._carried(into=graph)
        return run

    def calls_for(self, lengths: Mapping[str, int] | None = None, /, **by_name: int) -> dict[str, int | None]:
        """How many times each node is called in a run given lists of these lengths, before any run: ``graph.calls_for(file=11)``.

        Lengths are given by input name or address, as ``with_inputs`` takes
        values; an input not named gets one value, or its default. Each count
        is a most: a branch not taken calls nothing. A node that runs on rows
        another node makes counts ``None``, as how many is known only once that
        node has run. An embedded graph counts the calls of every node in it.
        """
        given = {**(lengths or {}), **by_name}
        bad = [name for name, n in given.items() if not is_a_number(type(n), int) or n < 0]
        if bad:
            raise InvalidCallError([make_problem("invalid_input", "inputs", name, name=name, reason="a length is a whole number of 0 or more") for name in bad])
        by_key, problems = self._given_by_key(given)
        if problems:
            raise InvalidCallError(problems)
        graph = self._compiled_for(frozenset(  # a length named is a list given; any other input lists as its default does
            key for key, f in self._input_fields.items() if (f.spec.per_item if key in by_key else given_as_items(f.spec, f.default))
        ))
        if graph.problems:
            raise InvalidCallError(graph.problems)
        sizes = graph._list_lengths({key: n for key, n in by_key.items() if graph._input_fields[key].listed})
        counts: dict[str, int | None] = {}
        for placed in graph._nodes.values():
            calls: int | None = 1
            for table in placed.level:  # a table a node makes has no size before the run
                size = sizes.get(table)
                calls = None if calls is None or size is None else calls * size
            drawn = graph._outward(placed.id)
            so_far = counts.get(drawn, 0)
            counts[drawn] = None if so_far is None or calls is None else so_far + calls
        return counts

    def run(
        self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, from_run: Mapping[type, Any] | None = None,
        timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_rows=max_rows, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote, compiled for the lists it was given; a graph with problems refuses it with them.

        A save is read from its JSON text, a mapping written as JSON first, so a float marked as not finite is in
        place before each value's type reads it.
        """
        from conductor._internal.run import run_from_json

        return run_from_json(self, saved)

    def _offered(self) -> tuple[list[Problem], dict[str, tuple[str, str]], dict[str, str]]:
        """What this graph offers where it is embedded: its problems, and its inputs and outputs by the name an
        embedding offers them under, beside the placed address behind each.

        They are this graph's own ``inputs`` and ``outputs``, as its hooks made them, and the open inputs its hooks
        read, which take a value typed on the embedding; each named by its field where that is unique and else
        ``node/field``, as a name holds no dot.
        """
        keys = [*((f.address, f.key) for f in self._input_fields.values()), *((self._outward(k), k) for k in self._fixed_open)]
        ins = [(address, (key.rpartition(".")[0], key.rpartition(".")[2])) for address, key in keys]
        outs = self._last_outputs

        def names(fields: list[tuple[str, Any]]) -> list[str]:
            fields_only = [address.partition(".")[2] for address, _ in fields]
            return [f if fields_only.count(f) == 1 else address.replace(".", "/", 1) for f, (address, _) in zip(fields_only, fields, strict=True)]

        return (
            self.problems,
            dict(zip(names(ins), (placed for _, placed in ins), strict=True)),
            dict(zip(names(outs), (placed for _, placed in outs), strict=True)),
        )

    def _placed_as(self, node_id: str) -> list[str]:
        """The placed nodes an id of this record names: the node itself, or every node an embedding put in its place."""
        found = self._placed.get(node_id)
        if found is None:
            found = self._placed[node_id] = [p for p in self._nodes if p == node_id or (p.startswith(node_id + "/") and node_id in self._offers)]
        return found

    def _above_of(self, node_id: str) -> frozenset[str]:
        """Every node ``node_id`` waits on, however far up."""
        found = self._above.get(node_id)
        if found is None:
            seen: set[str] = set()
            stack = [node_id]
            while stack:
                for plan in self._nodes[stack.pop()].inputs.values():
                    for s in plan.sources:
                        if s.node_id is not None and s.node_id not in seen:
                            seen.add(s.node_id)
                            stack.append(s.node_id)
            found = self._above[node_id] = frozenset(seen)
        return found

    def _compiled_for(self, listed: frozenset[str]) -> Graph:
        """This record compiled for a run given lists of items at ``listed``, by the placed node's address, and one value elsewhere."""
        if listed == self._listed:
            return self
        graph = self._compiles.get(listed)
        if graph is None:
            graph = self._compiles[listed] = load_graph(self.record, self._definitions, self._embedded_in, self._alone, listed=listed)
        return graph

    def _given_by_key(self, given: Mapping[str, Any]) -> tuple[dict[str, Any], list[Problem]]:
        """What ``given`` names by name or address, by the placed node's address; and each name the graph doesn't
        offer, and each input given by both its name and its address, as a problem."""
        fields = {**{f.address: f for f in self._input_fields.values()}, **{f.name: f for f in self._input_fields.values()}}
        by_key: dict[str, Any] = {}
        problems = []
        for name, value in given.items():
            offered = fields.get(name)
            if offered is None:
                problems.append(make_problem("not_offered", "inputs", name, name=name, inputs=list(self.inputs)))
            elif offered.key in by_key:
                problems.append(make_problem("input_given_twice", "inputs", offered.name, name=offered.name, address=offered.address))
            else:
                by_key[offered.key] = value
        return by_key, problems

    def _listed_in(self, by_key: Mapping[str, Any]) -> frozenset[str]:
        """The open inputs a run of ``by_key`` takes a list of items at: each given a list, or given nothing and its default a list."""
        return frozenset(
            key for key, f in self._input_fields.items()
            if given_as_items(f.spec, by_key[key] if key in by_key else f.default)
        )

    def _list_lengths(self, caller: Mapping[str, int]) -> dict[str, int]:
        """How many items each given list has, by its address: the caller's (``caller``), each typed in, and each default list not given."""
        lengths = {address: len(items) for address, items in self._typed_lists.items()}
        for key, f in self._input_fields.items():
            if f.listed:
                lengths[key] = caller[key] if key in caller else len(f.default)
        return lengths

    def _checked_inputs(self, by_key: Mapping[str, Any], problems: list[Problem]) -> dict[str, Any]:
        """The caller's inputs by address, each checked against its field, a list of items where this compile takes one;
        every fault at once, after the ``problems`` naming them found."""
        problems, values = list(problems), {}
        for key, offered in self._input_fields.items():
            name, spec = offered.name, offered.spec
            if key not in by_key:
                if offered.default is REQUIRED:
                    problems.append(make_problem("missing_input", "inputs", name, name=name))
                values[key] = offered.default
                continue
            value = by_key[key]
            if spec.shape == "many" and value is not None and not isinstance(value, (list, tuple, set, frozenset, range, pl.Series)):
                value = [value]  # one value where many are taken is a list of one
            try:
                values[key] = read_given(spec, value, items=offered.listed)
            except Refused as refused:
                problems.append(make_problem("invalid_input", "inputs", name, name=name, reason=refused.reason, **refused.ctx))
        if problems:
            raise InvalidCallError(problems)
        return values

    # -- embedding --------------------------------------------------------------

    def _splice(self, record: GraphRecord, prefix: str, stack: tuple[str, ...]) -> tuple[list[tuple[NodeRecord, NodeDefinition[..., Any] | None]], list[tuple[str, str]]]:
        """Resolve each node; an embedded graph's nodes take its place under ``<node>/`` ids."""
        out: list[tuple[NodeRecord, NodeDefinition[..., Any] | None]] = []
        edges = []
        refused = {n.id for n in record.nodes if not is_node_id(n.id)}
        for a, b in record.edges:
            if refused and (a.partition(".")[0] in refused or b.partition(".")[0] in refused):
                continue  # its own invalid_node_id says why
            inside = [address.partition(".")[0] for address in (a, b) if "/" in address.partition(".")[0]] if "/" in a or "/" in b else None
            if inside:  # no node of a record has a '/', so this names a node inside an embedded graph
                node_id, _, field = b.partition(".")
                self.problems.append(make_problem("edge_names_missing_node", prefix + node_id, field, node_id=inside[0], source=a, target=b))
                continue
            edges.append((prefix + a, prefix + b))
        renamed: dict[str, str] = {}  # "node.field" -> the inner node's "node.field"
        offered_here: dict[str, tuple[dict[str, tuple[str, str]], dict[str, str]]] = {}
        for node_record in record.nodes:
            if node_record.id in refused:
                self.problems.append(make_problem("invalid_node_id", prefix + node_record.id))
                continue  # never placed, so it can't stand in for a node an embedded graph puts there
            if prefix:  # an embedded graph's node, under its embedding's id
                node_record = node_record.model_copy(update={"id": prefix + node_record.id})
            definition = self._resolve(node_record)
            if definition is None or definition.graph is None:
                out.append((node_record, definition))
                continue
            if definition.id in stack:
                self.problems.append(make_problem("embeds_itself", node_record.id, node=definition.id))
                out.append((node_record, None))
                continue
            loaded_alone = (definition.id, definition.version, (*stack, definition.id))
            if loaded_alone not in self._alone:
                self._alone[loaded_alone] = load_graph(definition.graph, self._definitions, loaded_alone[2], self._alone, listed=frozenset())._offered()
            alone_problems, offered_ins, offered_outs = self._alone[loaded_alone]
            if alone_problems:
                inner_problems = [p.model_dump(mode="json") for p in alone_problems]
                self.problems.append(make_problem("embedded_graph_broken", node_record.id, graph=definition.id, count=len(inner_problems), problems=inner_problems))
                out.append((node_record, None))
                continue
            self._embedded[node_record.id] = definition.graph
            inner, inner_edges = self._splice(definition.graph, node_record.id + "/", (*stack, definition.id))
            ins = {name: (node_record.id + "/" + n, i) for name, (n, i) in offered_ins.items()}
            outs = {name: node_record.id + "/" + address for name, address in offered_outs.items()}
            for name, (inner_node, inner_input) in ins.items():
                self._seen_as[f"{inner_node}.{inner_input}"] = f"{node_record.id}.{name}"
            for name, address in outs.items():
                self._seen_as[address] = f"{node_record.id}.{name}"
            offered_here[node_record.id] = (ins, outs)
            if not prefix:
                self._offers[node_record.id] = (ins, outs)
            for name, (inner_node, inner_input) in ins.items():
                renamed[f"{node_record.id}.{name}"] = f"{inner_node}.{inner_input}"
            for name, address in outs.items():
                renamed[f"{node_record.id}.{name}"] = address
            typed: dict[str, dict[str, dict[str, Any]]] = {}
            for key in ("values", "defaults"):
                for name, value in getattr(node_record, key).items():
                    if name not in ins:
                        self.problems.append(make_problem("unknown_input", node_record.id, name, inputs=list(ins)))
                        continue
                    typed.setdefault(ins[name][0], {}).setdefault(key, {})[ins[name][1]] = value
            # What the outside gives an input it offers (a value, a default, an edge) takes the place of the inner default.
            drawn_id = node_record.id[len(prefix):]
            for _, target in record.edges:
                to_node, _, name = target.partition(".")
                if to_node == drawn_id and name in ins:
                    typed.setdefault(ins[name][0], {}).setdefault("fed", {})[ins[name][1]] = None
            inner = [(_given_from_outside(r, typed[r.id]) if r.id in typed else r, d) for r, d in inner]
            out.extend(inner)
            edges.extend(inner_edges)
        followed = []
        for a, b in edges:
            fa, fb = _follow(a, renamed), _follow(b, renamed)
            to_node, _, to_field = b.partition(".")
            if a.partition(".")[0] in offered_here and fa == a:
                outs = offered_here[a.partition(".")[0]][1]
                source_id = a.partition(".")[0]
                self.problems.append(make_problem("unknown_output", to_node, to_field, source=a, outputs=[f"{source_id}.{o}" for o in outs]))
                continue
            if to_node in offered_here and fb == b:
                ins = offered_here[to_node][0]
                self.problems.append(make_problem("unknown_input", to_node, to_field, inputs=list(ins)))
                continue
            followed.append((fa, fb))
        return out, followed

    def _resolve(self, node_record: NodeRecord) -> NodeDefinition[..., Any] | None:
        newest = self._definitions.get(node_record.node)
        if newest is None:
            self.problems.append(make_problem("unknown_node", node_record.id, node=node_record.node, choices=sorted(self._definitions)))
            return None
        if node_record.version != newest.version and node_record.version not in newest.versions:
            self.problems.append(make_problem("unknown_version", node_record.id, node=node_record.node, version=node_record.version, newest=newest.version))
            return None
        return newest.version_of(node_record.version)

    # -- compile ----------------------------------------------------------------

    def _compile(self, placed_records: list[tuple[NodeRecord, NodeDefinition[..., Any] | None]], edges: list[tuple[str, str]]) -> None:
        order = [r.id for r, _ in placed_records]
        known = {r.id: (r, d) for r, d in placed_records}
        if len(known) != len(placed_records):
            for node_id in sorted({i for i in order if order.count(i) > 1}):
                self.problems.append(make_problem("duplicate_node_id", node_id))
        into: dict[str, dict[str, list[str]]] = {i: {} for i in known}
        seen = set()
        for source, target in edges:
            node_id, _, name = target.partition(".")
            if (source, target) in seen:
                self.problems.append(make_problem("repeated_edge", node_id, name, source=source))
                continue
            seen.add((source, target))
            missing = [n for n in (source.partition(".")[0], node_id) if n not in known]
            if missing:
                self.problems.append(make_problem("edge_names_missing_node", node_id, name, node_id=missing[0], source=source, target=target))
            if node_id in known:  # the input is drawn as fed, whether or not its source exists
                into[node_id].setdefault(name, []).append(source)
        #: Each node's readers by the record's edges, before any is placed; only an embedded node's floor reads them, while compiling.
        self._fed: dict[str, list[str]] = {i: [] for i in known} if self._embedded else {}
        for node_id, fed in into.items() if self._embedded else ():
            for sources in fed.values():
                for source in sources:
                    if source.partition(".")[0] in self._fed:
                        self._fed[source.partition(".")[0]].append(node_id)
        for node_id in self._order(order, into):
            node_record, definition = known[node_id]
            if definition is not None:
                self._compile_node(node_record, definition, into[node_id])
        self._fed = {}
        offered = [(self._outside(p, key), key, spec, default, listed) for p, key, spec, default, listed in self._graph_inputs]
        self._graph_inputs = []
        bare = [p.name for p, *_ in offered]
        self.inputs = {}
        for p, key, spec, default, listed in offered:
            name = p.name if bare.count(p.name) == 1 else p.address
            self.inputs[name] = p
            self._input_fields[key] = InputField(key, name, p.address, spec, default, listed)
        if self._listed is None:
            self._listed = frozenset(key for key, f in self._input_fields.items() if f.listed)
        self._spliced = frozenset(n for n in self._nodes if "/" in n)
        #: The nodes with a ``FromRun`` parameter, as ``from_run_values`` reads them.
        self._needs_from_run = [(p.id, p.definition) for p in self._nodes.values() if p.definition.from_run]
        for placed in self._nodes.values():
            for source_id in {s.node_id for p in placed.inputs.values() for s in p.sources if s.node_id is not None}:
                self._readers[source_id].append(placed)

    def _order(self, order: list[str], into: dict[str, dict[str, list[str]]]) -> list[str]:
        """The nodes with each after what it reads, ties in drawing order; a node on a cycle is a problem."""
        position = {n: i for i, n in enumerate(order)}
        needs = {n: {s.partition(".")[0] for ss in into[n].values() for s in ss} & position.keys() for n in order}
        readers: dict[str, list[str]] = {n: [] for n in order}
        for n, sources in needs.items():
            for source in sources:
                readers[source].append(n)
        waiting = {n: len(sources) for n, sources in needs.items()}
        ready = [position[n] for n in order if not waiting[n]]
        heapq.heapify(ready)
        done = []
        while ready:
            n = order[heapq.heappop(ready)]
            done.append(n)
            for reader in readers[n]:
                waiting[reader] -= 1
                if not waiting[reader]:
                    heapq.heappush(ready, position[reader])
        stuck = {n for n in order if waiting[n]}
        for n in order:
            if n in stuck and _reaches(n, n, readers, stuck):  # on the cycle, not only below it
                self.problems.append(make_problem("cycle", n))
        return done

    def _compile_node(self, record: NodeRecord, definition: NodeDefinition[..., Any], edges: dict[str, list[str]]) -> None:
        if definition.compute_added_inputs is not None:
            added = self._hook(record, definition, definition.compute_added_inputs, edges)
            if added is None:
                return
            taken = sorted(added.keys() & (definition.inputs.keys() | definition.from_run.keys()))
            if taken:
                self.problems.append(make_problem("compute_hook_failed", record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            definition = dataclasses.replace(definition, inputs={**definition.inputs, **added})
        placed = PlacedNode(record.id, definition, record)
        broken = False
        inputs = list(definition.inputs)
        for name in sorted({*record.values, *record.defaults, *edges} - definition.inputs.keys()):
            self.problems.append(make_problem("unknown_input", placed.id, name, inputs=inputs))
        first_node = not any(name in definition.inputs for name in edges)
        read_at_load = _read_by_hooks(definition)
        for name, spec in definition.inputs.items():
            # an input an embedded graph offers is open to the caller whether or not edges from outside feed the others
            first = first_node or f"{record.id}.{name}" in self._seen_as
            plan = placed.inputs[name] = InputPlan(spec)
            if name in record.values and name in record.defaults:
                self.problems.append(make_problem("value_and_default", placed.id, name))
            if name in edges:
                if name in record.values or name in record.defaults:
                    self.problems.append(make_problem("value_and_edge", placed.id, name))
                for address in sorted(edges[name], key=self._gather_order):
                    source = self._source(address, placed.id, name)
                    if source is None:
                        broken = True
                    elif self._fits(spec, source, address, placed.id, name):
                        plan.sources.append(source)
            elif name in record.values and given_as_items(spec, record.values[name]):
                self._typed_list(placed, name, plan, record.values[name])
            elif name in record.values:
                plan.value = self._typed(placed.id, name, spec, record.values[name])
            elif first and name not in read_at_load:
                if name in record.defaults:
                    default = record.defaults[name]
                    plan.value = self._typed(placed.id, name, spec, default, items=given_as_items(spec, default))
                else:
                    plan.value = spec.default
                self._open(placed, name, plan)
            elif name in record.defaults:
                self.problems.append(make_problem("default_on_later_node", placed.id, name))
                plan.value = self._typed(placed.id, name, spec, record.defaults[name])
            elif spec.required and name not in read_at_load:  # a hook reading it says so, as hook_needs_value
                self.problems.append(make_problem("unbound_on_later_node", placed.id, name))
            else:
                plan.value = spec.default
                if first and name in read_at_load:
                    self._fixed_open.append(f"{placed.id}.{name}")
        if broken or not self._place(placed):
            return
        placed.checked = tuple(name for name, plan in placed.inputs.items() if any(not _settled(plan.spec, s) for s in plan.sources if s.node_id is not None))
        if definition.compute_outputs is not None:
            outputs = self._hook(record, definition, definition.compute_outputs, edges, placed)
            if outputs is None:
                return
            placed.definition = dataclasses.replace(definition, outputs=outputs)
        for name, spec in placed.definition.outputs.items():
            if spec.shape != "one":
                table = f"{placed.id}.{name}"
                placed.makes[name] = (*placed.level, table)
                self._makers[table] = (placed.id, name)
        if self._on_opposite_branches(placed):
            return
        self._nodes[placed.id] = placed

    def _gather_order(self, address: str) -> tuple[int, int, str]:
        """Where an edge into an input falls among several: its node's place in the record, then the output's in its node."""
        node_id, _, rest = address.partition(".")
        source = self._nodes.get(node_id)
        outputs = list(source.definition.outputs) if source is not None else []
        output = rest.partition(".")[0]
        return self._position.get(node_id, -1), outputs.index(output) if output in outputs else len(outputs), address

    def _hook(
        self, record: NodeRecord, definition: NodeDefinition[..., Any], hook: Any, edges: dict[str, list[str]], placed: PlacedNode | None = None,
    ) -> dict[str, FieldSpec] | None:
        """Call a hook with the node's typed values and arriving types: its answer read as fields, or a problem."""
        kwargs: dict[str, Any] = {}
        for name, parameter in inspect.signature(hook).parameters.items():
            if parameter.annotation in (type, "type"):
                plan = placed.inputs.get(name) if placed is not None else None
                kwargs[name] = _arriving(plan) if plan is not None and plan.sources else Any
            elif name in definition.inputs:
                spec = definition.inputs[name]
                raw = record.values.get(name, record.defaults.get(name, REQUIRED))
                if raw is REQUIRED and spec.default is REQUIRED:
                    self.problems.append(make_problem("hook_needs_value", record.id, name))
                    return None
                if raw is not REQUIRED and given_as_items(spec, raw):  # the input would take it as rows; the hook can't
                    problem = make_problem("invalid_value", record.id, name, reason="a hook works out this node's fields from one value; a list can't be typed in here")
                    if problem not in self.problems:  # both hooks of a node read it
                        self.problems.append(problem)
                    return None
                value = spec.default if raw is REQUIRED else self._typed(record.id, name, spec, raw)
                if value is REQUIRED:
                    return None  # a value with its own problem
                kwargs[name] = value
        try:
            fields = dict(hook(**kwargs))
        except NodeError as error:  # the hook's own account, with its own code and its ctx kept whole
            own = {"error_ctx": dict(error.ctx)} if error.ctx else {}
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=error.message, error=error.code or type(error).__name__, **own))
            return None
        except Exception as error:  # noqa: BLE001  the hook's failure is this node's problem
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None
        bad = [name for name in fields if not (isinstance(name, str) and name and not set(name) & set("./@"))]
        if bad:
            self.problems.extend(make_problem("parameter_name_invalid", record.id, str(name)) for name in bad)
            return None
        try:
            return {name: read_field(name, *_hint_and_default(hint), where=f"node {definition.id!r}, ") for name, hint in fields.items()}
        except Exception as error:  # noqa: BLE001  a hint conductor can't carry is the hook's failure
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None

    def _branch(self, source: Source) -> frozenset[tuple[str, str, str]]:
        """The ways decisions must go for ``source`` to hold a value: ``(node, exclusive_group, output)``."""
        if source.node_id is None:
            return frozenset()
        spec = self._nodes[source.node_id].definition.outputs[source.output]
        own = {(source.node_id, spec.param.exclusive_group, source.output)} if spec.param.exclusive_group else set()
        return self._branches[source.node_id] | own

    def _on_opposite_branches(self, placed: PlacedNode) -> bool:
        """Note the branches ``placed`` holds a value on, and refuse what never meets on a row.

        Several edges into one value must be alternatives, each on its own branch of one decision, or a row would
        have two values for it; and on rows of one table, or one's rows would have no place for the other's value. Two ``Series`` inputs whose sources never both have a value on a row would drop every row.
        """
        taken: set[tuple[str, str, str]] = set()
        for name, plan in placed.inputs.items():  # a skip on any one input skips the row
            if plan.spec.shape == "one" and plan.sources:
                branches = [self._branch(s) for s in plan.sources]
                if not all(_opposed(a, b) for i, a in enumerate(branches) for b in branches[i + 1:]):
                    self.problems.append(make_problem(
                        "several_edges_into_one_value", placed.id, name, sources=[_address(s) for s in plan.sources],
                    ))
                    return True
                levels = list(dict.fromkeys(s.level for s in plan.sources))
                if len(levels) > 1:  # the rows of the deeper one would have no place for the other's value
                    self.problems.append(make_problem(
                        "alternatives_on_different_rows", placed.id, name, tables=[_table(levels[0]), _table(levels[1])],
                    ))
                    return True
                taken |= frozenset.intersection(*branches)
        self._branches[placed.id] = frozenset(taken)
        series = [(n, [self._branch(s) for s in p.sources]) for n, p in placed.inputs.items() if p.spec.shape == "many" and p.sources and not takes_a_blank(p.spec.type)]
        for i, (a, on_a) in enumerate(series):
            for b, on_b in series[i + 1:]:
                if all(_opposed(x, y) for x in on_a for y in on_b):
                    self.problems.append(make_problem("series_inputs_never_share_a_row", placed.id, inputs=[a, b]))
                    return True
        return False

    def _typed(self, node_id: str, name: str, spec: FieldSpec, value: Any, *, items: bool = False) -> Any:
        """``value`` as ``spec`` reads it (a list of them with ``items``), or ``REQUIRED`` with the problem noted."""
        try:
            return read_given(spec, value, items=items)
        except Refused as refused:
            problem = make_problem("invalid_value", node_id, name, reason=refused.reason, **refused.ctx)
            if problem not in self.problems:  # a hook reads the value its node's input reads, and is told once
                self.problems.append(problem)
            return REQUIRED

    def _typed_list(self, placed: PlacedNode, name: str, plan: InputPlan, value: Any) -> None:
        """A list typed into an input that takes one value: a table of its items under each row the node runs on, which the node runs over."""
        items = self._typed(placed.id, name, plan.spec, value, items=True)
        if items is REQUIRED:
            return
        address = f"{placed.id}.{name}"
        self._typed_lists[address] = items
        self._makers[address] = (None, address)
        plan.sources.append(Source(None, address, None, (), True, plan.spec.type, name=name))

    def _open(self, placed: PlacedNode, name: str, plan: InputPlan) -> None:
        """An open input of a first node is the graph's: the caller fills it, by its address ``node.input``.

        Given a list of items in this compile, its rows are the list's own table.
        """
        address = f"{placed.id}.{name}"
        listed = address in self._listed if self._listed is not None else given_as_items(plan.spec, plan.value)
        level: Level = (address,) if listed else ()
        if listed:
            self._makers[address] = (None, address)
            self._lists_under.setdefault("", []).append(address)
        table = plan.spec.shape == "table"  # a caller's table, read to the input's columns at the call
        plan.sources.append(Source(None, address, None, level, listed, plan.spec.type, whole_table=table, name=name, columns=plan.spec.columns if table else None))
        required = plan.value is REQUIRED
        items = not required and given_as_items(plan.spec, plan.value)
        graph_input = plan.spec.describe(
            GraphInput, address=address, node_id=placed.id, per_item=plan.spec.per_item, required=required,
            default=None if required else plan.spec.list_adapter.dump_python(plan.value, mode="json") if items else _jsonable(plan.spec, plan.value),
        )
        self._graph_inputs.append((graph_input, address, plan.spec, plan.value, listed))

    def _source(self, address: str, node_id: str, name: str) -> Source | None:
        source_id, _, rest = address.partition(".")
        output, _, column = rest.partition(".")
        source = self._nodes.get(source_id)
        if source is None:  # a node that failed to load, or none; its own problem says why
            return None
        spec = source.definition.outputs.get(output)
        if spec is None or (column and spec.shape != "table") or (column and spec.columns is not None and column not in spec.columns):
            # What the edge could have named instead: the table's columns when it named a column of one, else the outputs.
            if column and spec is not None and spec.columns is not None:
                could = [f"{source_id}.{output}.{c}" for c in spec.columns]
            else:
                could = [f"{source_id}.{o}" for o in source.definition.outputs]
            self.problems.append(make_problem(
                "unknown_output", node_id, name, source=address, outputs=could, **({"column": column} if column else {}),
            ))
            return None
        if spec.shape == "one":
            return Source(source_id, output, None, source.level, False, spec.type, name=output)
        if spec.shape == "many":
            return Source(source_id, output, None, source.makes[output], True, spec.type, name=output)
        if column:
            return Source(source_id, output, column, source.makes[output], True, (spec.columns or {}).get(column, Any), name=column)
        return Source(source_id, output, None, source.makes[output], True, pl.DataFrame, whole_table=True, columns=spec.columns)

    def _fits(self, spec: FieldSpec, source: Source, address: str, node_id: str, name: str) -> bool:
        """Whether ``source`` may feed the input, or a ``type_mismatch``: a table input by each column it declares whose
        type the source knows, a table of unknown columns checked when it arrives."""
        if spec.shape == "table":
            given = (source.columns if source.whole_table else {source.name: source.type}) or {}
            for column, hint in (spec.columns or {}).items():
                if column in given and not accepts(hint, given[column]):
                    self.problems.append(make_problem(
                        "type_mismatch", node_id, name, source=address, got=type_id(given[column]), expected=type_id(hint), column=column,
                    ))
                    return False
            return True
        if source.whole_table or not accepts(spec.type, source.type):
            got = "Table" if source.whole_table else type_id(source.type)
            expected = type_id(spec.type)
            if spec.shape == "many":  # a gather: its items are checked, and it is said as the Series it reads
                got, expected = got if source.whole_table else f"Series[{got}]", f"Series[{expected}]"
            self.problems.append(make_problem("type_mismatch", node_id, name, source=address, got=got, expected=expected))
            return False
        return True

    def _place(self, placed: PlacedNode) -> bool:
        """Decide the rows ``placed`` runs on, or report one-value sources on rows of different tables.

        A ``Series`` or ``Table`` input gathers the rows of its source under the deepest row it shares with the
        node's, so a reduction of a sibling table (every keyword of a document, beside each of its invoices)
        gathers under their common parent.
        """
        typed = [(n, p) for n, p in placed.inputs.items() if p.sources and p.sources[0].output in self._typed_lists]
        ones = [s.level for p in placed.inputs.values() if p.spec.shape == "one" for s in p.sources if s.output not in self._typed_lists]
        # With no one-value input, a node runs one level above what its Series and Table inputs gather: each input
        # gathers at its deepest source (parents copied down), and the node sits above the shallowest input, so no
        # input receives a list of one.
        gathered = [_deepest([s.level for s in p.sources]) for p in placed.inputs.values() if p.spec.shape != "one" and p.sources]
        placed.level = _deepest(ones) if ones else min((level[:-1] for level in gathered), key=len, default=())
        floor = self._floor(placed)
        if not _under(floor, placed.level):  # an embedded graph runs on each row its embedding runs on
            if not _under(placed.level, floor):
                self._unrelated(placed, placed.level, floor)
                return False
            placed.level = floor
        for level in ones:
            if not (_under(placed.level, level) or _under(level, placed.level)):
                self._unrelated(placed, placed.level, level)
                return False
        for _, plan in typed:
            if not self._typed_list_placed(placed, plan.sources[0].output):
                return False
            plan.sources = [dataclasses.replace(plan.sources[0], level=placed.level)]
        for plan in placed.inputs.values():
            if plan.spec.shape != "one":
                plan.sources = [dataclasses.replace(s, anchor=shared_depth(placed.level, s.level)) for s in plan.sources]
                for s in plan.sources:
                    # a caller's or typed list is collected whole wherever the node runs; a node's table only where it relates
                    if not _gathers_from(placed.level, s.level) and self._makers[s.level[0]][0] is not None:
                        self._unrelated(placed, placed.level, s.level)
                        return False
            if plan.spec.shape == "table" and plan.sources and not self._table_fits(placed, plan):
                return False
        return True

    def _typed_list_placed(self, placed: PlacedNode, address: str) -> bool:
        """Put the items of a list typed into ``placed`` on rows: a table under each row the node would run on, which it
        runs over. Under the rows of another given list, typed in or the caller's, the two would meet uncollected,
        which is a problem: rows a node makes take a typed list under each of them, and a list never does."""
        given = [t for t in placed.level if self._makers[t][0] is None]
        if given:
            self._unrelated(placed, (given[-1],), (address,))
            return False
        self._lists_under.setdefault(placed.level[-1] if placed.level else "", []).append(address)
        placed.level = (*placed.level, address)
        return True

    def _table_fits(self, placed: PlacedNode, plan: InputPlan) -> bool:
        """Settle the rows a ``Table`` input combines its sources on; sources on two paths of the row tree are a problem."""
        plan.table_level = _deepest([s.level for s in plan.sources])
        plan.table_anchor = shared_depth(placed.level, plan.table_level)
        branches = [self._branch(s) for s in plan.sources]
        plan.alternatives = len(plan.sources) > 1 and all(s.whole_table for s in plan.sources) and all(
            _opposed(a, b) for i, a in enumerate(branches) for b in branches[i + 1:]
        )
        for s in plan.sources:
            if not plan.alternatives and not _under(s.level, plan.table_level):
                self._unrelated(placed, s.level, plan.table_level)
                return False
        names: list[str] = []
        for i, s in enumerate(plan.sources):
            if not s.whole_table and s.name in names:
                plan.sources[i] = dataclasses.replace(s, name=f"{s.name}_{s.node_id}")
            names.append(plan.sources[i].name)
        if plan.spec.columns is not None:
            each = [[s] for s in plan.sources] if plan.alternatives else [plan.sources]
            for sources in each:
                if any(s.whole_table and s.columns is None for s in sources):
                    continue
                given = {c for s in sources for c in ((s.columns or {}) if s.whole_table else (s.name,))}
                missing = sorted(plan.spec.columns.keys() - given)
                if missing:
                    name = next(n for n, p in placed.inputs.items() if p is plan)
                    self.problems.append(make_problem("columns_missing", placed.id, name, missing=missing, sources=[_address(s) for s in sources]))
                    return False
        return True

    def _floor(self, placed: PlacedNode) -> Level:
        """The rows an embedded node runs on at least: the rows its embedding enters on, by the entering edges it reads from.

        A node reads an entering edge when the edge lands on it or on a node above it inside the embedding. A node
        that reads none and makes rows takes the entering edges a node below it reads, so a table made inside lies
        under the outer row its reader receives the entering value on.
        """
        node_id = placed.id
        levels: list[Level] = []
        above: set[str] | None = None
        for e in self._embedded:
            if node_id.startswith(e + "/"):
                if above is None:
                    sources = {s.node_id for p in placed.inputs.values() for s in p.sources if s.node_id is not None}
                    above = {node_id, *sources, *(a for n in sources for a in self._above_of(n))}
                entering = [(address, target) for address, target in self._entering[e] if target.partition(".")[0] in above]
                if not entering and any(spec.shape != "one" for spec in placed.definition.outputs.values()):
                    below = self._below(node_id)
                    entering = [(address, target) for address, target in self._entering[e] if below & {target.partition(".")[0], *self._below(target.partition(".")[0])}]
                for address, target in entering:
                    source = self._nodes.get(address.partition(".")[0])
                    if source is None:
                        continue
                    level = self._level_of(source, address)
                    target_id, _, target_input = target.partition(".")
                    holder = placed if target_id == placed.id else self._nodes.get(target_id)
                    spec = holder.definition.inputs.get(target_input) if holder is not None else None  # its own copy, as its hooks made it
                    levels.append(level[:-1] if spec is not None and spec.shape != "one" and level else level)
        return _deepest(levels)

    def _below(self, node_id: str) -> set[str]:
        """Every node the record's edges feed from ``node_id``, however far down."""
        found: set[str] = set()
        stack = list(self._fed[node_id])
        while stack:
            n = stack.pop()
            if n not in found:
                found.add(n)
                stack.extend(self._fed[n])
        return found

    def _level_of(self, source: PlacedNode, address: str) -> Level:
        """The rows the value at ``address`` sits on: its node's, or the table one of its outputs makes."""
        output = address.partition(".")[2].partition(".")[0]
        return source.makes.get(output, source.level)

    def _outside(self, graph_input: GraphInput, key: str) -> GraphInput:
        """An input as a caller of this graph sees it: an embedded graph's by the node that embeds it."""
        outer = self._outward(key)
        if outer == key:
            return graph_input
        node_id, _, name = outer.partition(".")
        return graph_input.model_copy(update={"address": outer, "node_id": node_id, "name": name})

    def _shown_outputs(self, node_id: str) -> dict[str, str]:
        """Each output of a placed node an embedding offers under another name, by its own name: the name a person is asked by."""
        shown = {}
        for output in self._nodes[node_id].definition.outputs:
            outer = self._outward(f"{node_id}.{output}")
            if outer != f"{node_id}.{output}" and outer.partition(".")[2] != output:
                shown[output] = outer.partition(".")[2]
        return shown

    def _outward(self, name: str, *, table: bool = False) -> str:
        """A node id, address or table name the engine uses, as the author of this record reads it.

        A name that lies inside an embedded graph is said of the node that embeds it: an address the embedded
        graph offers by the embedding's address, any other name by the embedding's id, or with ``table`` by the
        table's own name. Every other name is as it is.
        """
        if name in self._outer:
            return self._outer[name]
        top, slash, _ = name.partition("/")
        if not slash or top not in self._offers:
            return name
        if table:
            return self._tables_seen.get(name) or name.rpartition("/")[2]
        return top

    def _seen_from_outside(self, problem: Problem) -> Problem:
        """A problem as the author of this record reads it: inside an embedded graph, on the node that embeds it."""
        if problem.type == "invalid_node_id" or not problem.loc or not isinstance(problem.loc[0], str) or problem.loc[0].split("/", 1)[0] not in self._embedded or "/" not in problem.loc[0]:
            return problem
        inner = problem.loc[0]
        field = problem.loc[1] if len(problem.loc) > 1 and isinstance(problem.loc[1], str) else None
        outer = self._outward(f"{inner}.{field}") if field else self._outward(inner)
        loc: tuple[str | int, ...] = tuple(outer.split(".", 1)) if "." in outer else (outer, *(problem.loc[1:] if field is None else ()))
        if problem.type not in PROBLEM_CATALOGUE:  # a node's own code: its message and ctx are its author's, and name no node
            return problem.model_copy(update={"loc": loc})
        ctx = dict(problem.ctx)
        for key in ("node_id", "source", "target"):
            if isinstance(ctx.get(key), str):
                ctx[key] = self._outward(ctx[key])
        if isinstance(ctx.get("sources"), list):
            ctx["sources"] = [self._outward(source) for source in ctx["sources"]]
        if isinstance(ctx.get("tables"), list):
            ctx["tables"] = [table if table is None else self._outward(table, table=True) for table in ctx["tables"]]
        if problem.type == "invalid_output" and isinstance(ctx.get("name"), str) and inner in self._nodes:  # the output as the embedding offers it
            ctx["name"] = self._shown_outputs(inner).get(ctx["name"], ctx["name"])
        return make_problem(problem.type, *loc, **ctx)

    def _unrelated(self, placed: PlacedNode, a: Level, b: Level) -> None:
        """Rows of two tables meet at ``placed``: ``lists_not_collected`` where one is a given list, which a node
        collects, else ``rows_not_related``, which a join brings together."""
        tables = [_table(a), _table(b)]
        given = any(t is not None and self._makers.get(t, ("",))[0] is None for t in tables)
        self.problems.append(make_problem("lists_not_collected" if given else "rows_not_related", placed.id, tables=tables))


def load_graph(
    record: GraphRecord, definitions: Mapping[str, NodeDefinition[..., Any]], embedded_in: tuple[str, ...] = (), alone: dict[Any, tuple[list[Problem], dict[str, tuple[str, str]], dict[str, str]]] | None = None,
    *, listed: frozenset[str] | None = None,
) -> Graph:
    """``record`` loaded against ``definitions``, the one way a ``Graph`` is made; ``embedded_in`` names the graphs it is
    loaded inside, and ``alone`` what the embedded graphs of this load found loaded alone.

    ``listed`` names the open inputs given lists of items. With no ``listed``, the graph is compiled as a run given nothing runs: with the lists its defaults hold, in one pass.
    """
    graph = Graph.__new__(Graph)
    graph._load(record, definitions, embedded_in, {} if alone is None else alone, listed)
    return graph


def _given_from_outside(record: NodeRecord, typed: Mapping[str, Mapping[str, Any]]) -> NodeRecord:
    """An embedded graph's node with what the record embedding it gives its offered inputs: values and defaults typed
    in, each replacing the inner default of its input, as an edge from outside does."""
    outside = {name for given in typed.values() for name in given}
    defaults = {name: value for name, value in record.defaults.items() if name not in outside}
    return record.model_copy(update={
        "values": {**record.values, **typed.get("values", {})},
        "defaults": {**defaults, **typed.get("defaults", {})},
    })


def _read_by_hooks(definition: NodeDefinition[..., Any]) -> frozenset[str]:
    """The inputs a node's hooks read as values: fixed when the graph loads, so never open to a caller."""
    hooks = (definition.compute_added_inputs, definition.compute_outputs)
    return frozenset(
        name for hook in hooks if hook is not None for name, parameter in inspect.signature(hook).parameters.items()
        if parameter.annotation not in (type, "type")
    )


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """What a hook gives for one field: a hint, or ``(hint, default)`` as pydantic's ``create_model`` takes one."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


def _follow(address: str, renamed: Mapping[str, str]) -> str:
    """Rewrite ``node.field[.column]`` to the inner node an embedded graph spliced in for it."""
    node_id, _, rest = address.partition(".")
    for cut in (rest, rest.partition(".")[0]):
        if cut and f"{node_id}.{cut}" in renamed:
            return renamed[f"{node_id}.{cut}"] + rest[len(cut):]
    return address


def _arriving(plan: InputPlan) -> Any:
    """The type a hook reads for what arrives on an input: a whole table as ``Table[R]`` with its columns, columns
    gathered into a ``Table`` input as ``Table[R]`` of those columns, else the value's type."""
    from conductor._internal.fields import Table

    source = plan.sources[0]
    if source.whole_table and source.columns is not None:
        return Table.__class_getitem__(typing.TypedDict("Arriving", dict(source.columns)))  # type: ignore[operator]
    if plan.spec.shape == "table" and not any(s.whole_table for s in plan.sources):
        return Table.__class_getitem__(typing.TypedDict("Arriving", {s.name: s.type for s in plan.sources}))  # type: ignore[operator]
    return source.type


def _type_text(hint: Any) -> str:
    """A type as text that is the same in every process and tells apart every two types a value is read by differently:
    a class by its module and name, a generic by its arguments (``list[int]``, ``list[str]``), a ``Literal`` by its
    values, a type alias by its name; the metadata of an ``Annotated`` left out."""
    if typing.get_origin(hint) is typing.Annotated:
        return _type_text(typing.get_args(hint)[0])
    if typing.get_origin(hint) is Literal:
        return f"Literal{[_constant_text(value) for value in typing.get_args(hint)]}"
    if isinstance(hint, typing.TypeAliasType):
        return f"{hint.__module__}.{hint.__name__}"
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if args:
        name = "Union" if origin in (typing.Union, types.UnionType) else _type_text(origin)
        return f"{name}[{', '.join(map(_type_text, args))}]"
    if isinstance(hint, type):
        return f"{hint.__module__}.{hint.__qualname__}"
    return repr(hint)


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


def _opposed(a: frozenset[tuple[str, str, str]], b: frozenset[tuple[str, str, str]]) -> bool:
    """Whether two sets of branches never hold together: one decision goes one way for ``a`` and another for ``b``."""
    return any(x[:2] == y[:2] and x[2] != y[2] for x in a for y in b)


def _address(source: Source) -> str:
    return source.output if source.node_id is None else ".".join(p for p in (source.node_id, source.output, source.column) if p)


def _settled(spec: FieldSpec, source: Source) -> bool:
    """Whether every value ``source`` brings is already one ``spec`` takes as it is, with no limit or choice to hold it to."""
    if spec.param.holds_a_value:
        return False
    if spec.shape == "table":
        if spec.columns is None:
            return True
        given = source.columns if source.whole_table else {source.name: source.type}
        return given is not None and all(c not in given or already(hint, given[c]) for c, hint in spec.columns.items())
    return already(spec.type, source.type)


def _reaches(start: str, goal: str, readers: Mapping[str, list[str]], within: set[str]) -> bool:
    """Whether ``goal`` is read, however far down, by what ``start`` feeds, staying ``within`` those nodes."""
    seen: set[str] = set()
    stack = list(readers[start])
    while stack:
        n = stack.pop()
        if n == goal:
            return True
        if n in within and n not in seen:
            seen.add(n)
            stack.extend(readers[n])
    return False


def _under(prefix: Level, level: Level) -> bool:
    return level[: len(prefix)] == prefix


def _gathers_from(level: Level, source: Level) -> bool:
    """Whether a node on ``level`` can gather ``source``'s rows: one is under the other, or they share a table above both."""
    return _under(level, source) or _under(source, level) or shared_depth(level, source) > 0


def shared_depth(a: Level, b: Level) -> int:
    """How many tables two levels share from the graph's own row down."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _deepest(levels: list[Level]) -> Level:
    return max(levels, key=len, default=())


def _table(level: Level) -> str | None:
    """The table a level's rows are, ``None`` for the graph's own row."""
    return level[-1] if level else None


def _jsonable(spec: FieldSpec, value: Any) -> Any:
    try:
        return spec.adapter.dump_python(value, mode="json")
    except Exception:  # noqa: BLE001  a value with no JSON form shows as nothing
        return None
