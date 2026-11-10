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
- a ``Series`` or ``Table`` input gathers the rows under the node's row; it
  chooses the level, one up from its source, only when no ``one`` input does;
- sources must lie on one path of the row tree. Rows of two different tables
  meet only through a join the author draws.
"""

from __future__ import annotations

import dataclasses
import heapq
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import ValidationError

from conductor._internal.errors import InvalidCallError, Problem, make_problem, value_free_errors
from conductor._internal.fields import (
    REQUIRED,
    FieldSpec,
    accepts,
    carried_list,
    read_field,
    type_id,
)
from conductor._internal.node_definition import NodeDefinition
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


@dataclass
class InputPlan:
    """How one input of a placed node gets its value: its sources, or a value typed in."""

    spec: FieldSpec
    sources: list[Source] = field(default_factory=list)
    value: Any = REQUIRED
    #: For a table input, the level whose rows it gathers.
    table_level: Level = ()


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


@dataclass(frozen=True)
class GraphInput:
    """An open input of a first node: what a caller fills, at ``address`` (``node.input``).

    For an embedded graph's input, ``node_id`` is the node that embeds it and ``input`` the name it offers.
    """

    address: str
    node_id: str
    input: str
    spec: FieldSpec
    #: The caller gives a list, a row per item.
    many: bool
    required: bool
    #: What it holds when the caller gives nothing: the record's default, else the node's.
    default: Any
    #: Where the run keeps the caller's value: the address of the placed node that takes it.
    key: str = ""

    def describe(self) -> dict[str, Any]:
        """The input as a form shows it, as JSON."""
        described = {"address": self.address, "node_id": self.node_id, "many": self.many, **self.spec.describe(), "name": self.input, "required": self.required}
        described.pop("default", None)
        if not self.required:
            described["default"] = _jsonable(self.spec, self.default)
        return described


class Graph:
    """A graph loaded against a registry, with its own copy of the definitions it resolved.

    ``problems`` is everything wrong with it, and a graph with none runs:
    ``graph.run(**inputs)``. ``inputs`` are the open inputs of its first
    nodes, by name (``node.input`` where two share a name), and ``outputs`` the
    outputs of the nodes nothing reads.
    """

    def __init__(self, record: GraphRecord, definitions: Mapping[str, NodeDefinition[..., Any]], _embedded_in: tuple[str, ...] = ()) -> None:
        self.record = record
        self.problems: list[Problem] = []
        self._definitions = dict(definitions)
        #: Each embedding, at any depth, by its spliced id; and for the ones drawn in this record, what they offer.
        self._embeddings: set[str] = set()
        self._offers: dict[str, tuple[dict[str, tuple[str, str]], dict[str, str]]] = {}
        placed_records, edges = self._splice(record, "", _embedded_in)
        #: Each embedding's edges from outside it into it, which set the rows its nodes run on.
        self._entering = {e: [(a, b) for a, b in edges if b.startswith(e + "/") and not a.startswith(e + "/")] for e in self._embeddings}
        #: An inner address as seen from outside: ``e/upper.text`` -> ``e.text``.
        self._outer: dict[str, str] = {}
        for e, (ins, outs) in self._offers.items():
            self._outer.update({f"{n}.{i}": f"{e}.{name}" for name, (n, i) in ins.items()})
            self._outer.update({address: f"{e}.{name}" for name, address in outs.items()})
        self._nodes: dict[str, PlacedNode] = {}
        self.inputs: dict[str, GraphInput] = {}
        #: Each table, by name, and the node and output that make it (``None`` for a graph input).
        self._makers: dict[str, tuple[str | None, str]] = {}
        self._graph_inputs: list[GraphInput] = []
        self._position = {r.id: n for n, (r, _) in enumerate(placed_records)}
        self._readers: dict[str, list[PlacedNode]] = {r.id: [] for r, _ in placed_records}
        self._above: dict[str, frozenset[str]] = {}
        #: Each node's branches: the ways decisions must go for it to hold a value, ``(node, group, output)``.
        self._branches: dict[str, frozenset[tuple[str, str, str]]] = {}
        self._records = {r.id: r for r, _ in placed_records}
        self._compile(placed_records, edges)
        if not _embedded_in:
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
        read = {s.node_id for p in self._nodes.values() for i in p.inputs.values() for s in i.sources}
        return [self._outer.get(f"{p.id}.{o}", f"{p.id}.{o}") for p in self._nodes.values() if p.id not in read for o in p.definition.outputs]

    @property
    def plan(self) -> dict[str, str]:
        """Where each node runs: ``once``, or ``per row of <table>``; an embedded graph where the nodes behind its outputs run."""
        plan: dict[str, str] = {}
        for p in self._nodes.values():
            node_id = self._outer_id(p.id)
            if node_id in plan and node_id != p.id and not any(self._outer.get(f"{p.id}.{o}") for o in p.definition.outputs):
                continue
            plan[node_id] = f"per row of {self._outer.get(p.level[-1], p.level[-1])}" if p.level else "once"
        return plan

    @property
    def decisions(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for placed in self._nodes.values():
            for name, spec in placed.definition.outputs.items():
                if spec.param.exclusive_group:
                    found.setdefault(placed.id, {}).setdefault(spec.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> dict[str, Any]:
        """The loaded graph as an editor draws it: every node of the record, with its fields as its hooks made them and where it runs.

        A node that didn't load (an unknown node, a broken edge, or one below it) is there with ``loaded`` false,
        its problems saying why; an embedded graph's nodes are there under ``<node>/`` ids.
        """
        nodes: dict[str, Any] = {}
        plan = self.plan
        for record in self.record.nodes:
            nodes[record.id] = {"node": record.node, "version": record.version, "loaded": record.id in self._nodes or record.id in plan}
        for p in self._nodes.values():
            if p.id in nodes:
                nodes[p.id].update({
                    "runs": plan[p.id],
                    "inputs": [f.describe() for f in p.definition.inputs.values()],
                    "outputs": [f.describe() for f in p.definition.outputs.values()],
                })
        for e, (ins, outs) in self._offers.items():
            if e in plan:
                nodes[e].update({
                    "runs": plan[e],
                    "inputs": [{**self._nodes[n].definition.inputs[i].describe(), "name": name} for name, (n, i) in ins.items() if n in self._nodes],
                    "outputs": [{**self._nodes[a.partition(".")[0]].definition.outputs[a.partition(".")[2]].describe(), "name": name}
                                for name, a in outs.items() if a.partition(".")[0] in self._nodes],
                })
        return {
            "nodes": nodes,
            "inputs": {name: i.describe() for name, i in self.inputs.items()},
            "outputs": self.outputs,
            "problems": [p.model_dump(mode="json") for p in self.problems],
        }

    # -- running ----------------------------------------------------------------

    def with_inputs(self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, **by_name: Any) -> Run:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Inputs are given by name (``graph.inputs``) or by address
        (``node.input``), as keywords or as one mapping for names Python can't
        take as keywords (``{"a.text": ...}``). ``max_rows`` is a ceiling on
        the rows any one table of the run may have.
        """
        from conductor._internal.run import Run, checked_max_rows

        if self.problems:
            raise InvalidCallError(self.problems)
        checked_max_rows(max_rows)
        return Run(self, self._checked_inputs({**(inputs or {}), **by_name}), max_rows=max_rows)

    def run(
        self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, from_run: Mapping[type, Any] | None = None,
        timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_rows=max_rows, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote."""
        from conductor._internal.run import run_from_json

        return run_from_json(self, saved)

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

    def _checked_inputs(self, given: Mapping[str, Any]) -> dict[str, Any]:
        """The caller's inputs by address, each checked against its field and ``many``; every fault at once."""
        problems, values = [], {}
        named = {graph_input.address: name for name, graph_input in self.inputs.items()}
        twice = [key for key in given if key in named and named[key] != key and named[key] in given]
        if twice:
            raise InvalidCallError([make_problem("input_given_twice", "inputs", named[key], name=named[key], address=key) for key in twice])
        given = {named.get(key, key): value for key, value in given.items()}
        for name in given.keys() - self.inputs.keys():
            problems.append(make_problem("not_offered", "inputs", name, name=name, inputs=", ".join(self.inputs) or "none"))
        for name, graph_input in self.inputs.items():
            if name not in given:
                if graph_input.required:
                    problems.append(make_problem("missing_input", "inputs", name, name=name))
                values[graph_input.key] = graph_input.default
                continue
            adapter = carried_list(graph_input.spec.type) if graph_input.many else graph_input.spec.adapter
            try:
                values[graph_input.key] = adapter.validate_python(given[name])
            except ValidationError as error:
                problems.append(make_problem("invalid_input", "inputs", name, name=name, reason=value_free_errors(error)[0]["msg"]))
        if problems:
            raise InvalidCallError(problems)
        return values

    # -- embedding --------------------------------------------------------------

    def _splice(self, record: GraphRecord, prefix: str, stack: tuple[str, ...]) -> tuple[list[tuple[NodeRecord, NodeDefinition[..., Any] | None]], list[tuple[str, str]]]:
        """Resolve each node; an embedded graph's nodes take its place under ``<node>/`` ids."""
        out: list[tuple[NodeRecord, NodeDefinition[..., Any] | None]] = []
        edges = []
        for a, b in record.edges:
            inside = [address.partition(".")[0] for address in (a, b) if "/" in address.partition(".")[0]]
            if inside:  # no node of a record has a '/', so this names a node inside an embedded graph
                node_id, _, field = b.partition(".")
                self.problems.append(make_problem("edge_names_missing_node", prefix + node_id, field, node_id=inside[0], source=a, target=b))
                continue
            edges.append((prefix + a, prefix + b))
        renamed: dict[str, str] = {}  # "node.field" -> the inner node's "node.field"
        offered_here: dict[str, tuple[dict[str, tuple[str, str]], dict[str, str]]] = {}
        for node_record in record.nodes:
            if any(mark in node_record.id for mark in "/.@"):
                self.problems.append(make_problem("invalid_node_id", prefix + node_record.id))
            node_record = node_record.model_copy(update={"id": prefix + node_record.id})
            definition = self._resolve(node_record)
            if definition is None or definition.graph is None:
                out.append((node_record, definition))
                continue
            if definition.id in stack:
                self.problems.append(make_problem("embeds_itself", node_record.id, node=definition.id))
                out.append((node_record, None))
                continue
            alone = Graph(definition.graph, self._definitions, (*stack, definition.id))
            if alone.problems:
                inner_problems = [p.model_dump(mode="json") for p in alone.problems]
                self.problems.append(make_problem("embedded_graph_broken", node_record.id, graph=definition.id, count=len(inner_problems), problems=inner_problems))
                out.append((node_record, None))
                continue
            inner, inner_edges = self._splice(definition.graph, node_record.id + "/", (*stack, definition.id))
            ins, outs = _interface(inner, inner_edges, node_record.id + "/")
            self._embeddings.add(node_record.id)
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
                        self.problems.append(make_problem("unknown_input", node_record.id, name, inputs=", ".join(ins) or "none"))
                        continue
                    typed.setdefault(ins[name][0], {}).setdefault(key, {})[ins[name][1]] = value
            inner = [
                (r.model_copy(update={k: {**getattr(r, k), **v} for k, v in typed[r.id].items()}) if r.id in typed else r, d)
                for r, d in inner
            ]
            out.extend(inner)
            edges.extend(inner_edges)
        followed = []
        for a, b in edges:
            fa, fb = _follow(a, renamed), _follow(b, renamed)
            to_node, _, to_field = b.partition(".")
            if a.partition(".")[0] in offered_here and fa == a:
                outs = offered_here[a.partition(".")[0]][1]
                self.problems.append(make_problem("unknown_output", to_node, to_field, source=a, outputs=", ".join(outs) or "none"))
                continue
            if to_node in offered_here and fb == b:
                ins = offered_here[to_node][0]
                self.problems.append(make_problem("unknown_input", to_node, to_field, inputs=", ".join(ins) or "none"))
                continue
            followed.append((fa, fb))
        return out, followed

    def _resolve(self, node_record: NodeRecord) -> NodeDefinition[..., Any] | None:
        newest = self._definitions.get(node_record.node)
        if newest is None:
            self.problems.append(make_problem("unknown_node", node_record.id, node=node_record.node, choices=", ".join(sorted(self._definitions)) or "none"))
            return None
        found = newest.version_of(node_record.version)
        if found is None:
            self.problems.append(make_problem("unknown_version", node_record.id, node=node_record.node, version=node_record.version, newest=newest.version))
        return found

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
        for node_id in self._order(order, into):
            node_record, definition = known[node_id]
            if definition is not None:
                self._compile_node(node_record, definition, into[node_id])
        offered = [self._outside(p) for p in self._graph_inputs]
        bare = [p.input for p in offered]
        self.inputs = {(p.input if bare.count(p.input) == 1 else p.address): p for p in offered}
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
            taken = sorted(added.keys() & definition.inputs.keys())
            if taken:
                self.problems.append(make_problem("compute_hook_failed", record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            definition = dataclasses.replace(definition, inputs={**definition.inputs, **added})
        placed = PlacedNode(record.id, definition, record)
        broken = False
        inputs = ", ".join(definition.inputs) or "none"
        for name in sorted({*record.values, *record.defaults, *record.many, *edges} - definition.inputs.keys()):
            self.problems.append(make_problem("unknown_input", placed.id, name, inputs=inputs))
        first = not any(name in definition.inputs for name in edges)
        for name, spec in definition.inputs.items():
            plan = placed.inputs[name] = InputPlan(spec)
            if name in record.values and name in record.defaults:
                self.problems.append(make_problem("value_and_default", placed.id, name))
            if name in record.many and name not in edges and (name in record.values or not first):
                self.problems.append(make_problem("many_not_open", placed.id, name))
            if name in edges:
                if name in record.values or name in record.defaults:
                    self.problems.append(make_problem("value_and_edge", placed.id, name))
                if name in record.many:
                    self.problems.append(make_problem("many_on_edge", placed.id, name))
                for address in sorted(edges[name], key=self._gather_order):
                    source = self._source(address, placed.id, name)
                    if source is None:
                        broken = True
                    elif self._fits(spec, source, address, placed.id, name):
                        plan.sources.append(source)
            elif name in record.values:
                plan.value = self._typed(placed.id, name, spec, record.values[name])
            elif first:
                many = name in record.many and spec.shape == "one"
                if name in record.defaults:
                    plan.value = self._typed(placed.id, name, spec, record.defaults[name], many)
                else:
                    plan.value = [] if many and not spec.required else spec.default
                self._open(placed, name, plan, record)
            elif name in record.defaults:
                self.problems.append(make_problem("default_on_later_node", placed.id, name))
                plan.value = self._typed(placed.id, name, spec, record.defaults[name])
            elif spec.required:
                self.problems.append(make_problem("unbound_on_later_node", placed.id, name))
            else:
                plan.value = spec.default
        if broken or not self._place(placed):
            return
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
                sources = placed.inputs[name].sources if placed is not None and name in placed.inputs else []
                kwargs[name] = _arriving(sources[0]) if sources else Any
            elif name in definition.inputs:
                spec = definition.inputs[name]
                raw = record.values.get(name, record.defaults.get(name, REQUIRED))
                if raw is REQUIRED and spec.default is REQUIRED:
                    self.problems.append(make_problem("hook_needs_value", record.id, name))
                    return None
                value = spec.default if raw is REQUIRED else self._typed(record.id, name, spec, raw)
                if value is REQUIRED:
                    return None  # a value with its own problem
                kwargs[name] = value
        try:
            fields = dict(hook(**kwargs))
        except Exception as error:  # noqa: BLE001  the hook's failure is this node's problem
            self.problems.append(make_problem("compute_hook_failed", record.id, reason=str(error), error=type(error).__name__))
            return None
        bad = [name for name in fields if not (isinstance(name, str) and name and not set(name) & set("./@"))]
        if bad:
            self.problems.extend(make_problem("parameter_name_invalid", record.id, str(name)) for name in bad)
            return None
        try:
            return {name: read_field(name, hint, where=f"node {definition.id!r}, ") for name, hint in fields.items()}
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
        """Note the branches ``placed`` holds a value on; refuse two Series inputs that never share a row."""
        taken: set[tuple[str, str, str]] = set()
        for plan in placed.inputs.values():  # a skip on any one input skips the row
            if plan.spec.shape == "one" and plan.sources:
                taken |= frozenset.intersection(*(self._branch(s) for s in plan.sources))
        self._branches[placed.id] = frozenset(taken)
        series = [
            (n, frozenset[tuple[str, str, str]]().union(*(self._branch(s) for s in p.sources)))
            for n, p in placed.inputs.items() if p.spec.shape == "many" and p.sources and not p.spec.optional
        ]
        for i, (a, on_a) in enumerate(series):
            for b, on_b in series[i + 1:]:
                if any(x[:2] == y[:2] and x[2] != y[2] for x in on_a for y in on_b):
                    self.problems.append(make_problem("series_inputs_never_share_a_row", placed.id, a=a, b=b))
                    return True
        return False

    def _typed(self, node_id: str, name: str, spec: FieldSpec, value: Any, many: bool = False) -> Any:
        """``value`` as ``spec`` reads it (a list of them when ``many``), or ``REQUIRED`` with the problem noted."""
        if value is None and spec.shape == "many" and spec.default is None:
            return None  # a Series that may be None
        try:
            if many:
                return carried_list(spec.type).validate_python(value)
            if spec.shape == "table":
                return value if isinstance(value, pl.DataFrame) else pl.DataFrame(value)
            return spec.adapter.validate_python(value)
        except ValidationError as error:
            self.problems.append(make_problem("invalid_value", node_id, name, reason=value_free_errors(error)[0]["msg"]))
            return REQUIRED

    def _open(self, placed: PlacedNode, name: str, plan: InputPlan, record: NodeRecord) -> None:
        """An open input of a first node is the graph's: the caller fills it, by its address ``node.input``."""
        address = f"{placed.id}.{name}"
        many = name in record.many
        if many and plan.spec.shape != "one":
            self.problems.append(make_problem("many_on_series", placed.id, name))
            many = False
        level: Level = (address,) if many else ()
        if many:
            self._makers[address] = (None, address)
        plan.sources.append(Source(None, address, None, level, many, plan.spec.type, name=name))
        self._graph_inputs.append(GraphInput(address, placed.id, name, plan.spec, many, plan.value is REQUIRED, plan.value, key=address))

    def _source(self, address: str, node_id: str, name: str) -> Source | None:
        source_id, _, rest = address.partition(".")
        output, _, column = rest.partition(".")
        source = self._nodes.get(source_id)
        if source is None:  # a node that failed to load, or none; its own problem says why
            return None
        spec = source.definition.outputs.get(output)
        if spec is None or (column and spec.shape != "table") or (column and spec.columns is not None and column not in spec.columns):
            self.problems.append(make_problem(
                "unknown_output", node_id, name, source=address, outputs=", ".join(source.definition.outputs) or "none", **({"column": column} if column else {}),
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
        if spec.shape == "table":
            return True
        if source.whole_table or not accepts(spec.type, source.type):
            got = "a table" if source.whole_table else type_id(source.type)
            self.problems.append(make_problem("type_mismatch", node_id, name, source=address, got=got, expected=type_id(spec.type)))
            return False
        return True

    def _place(self, placed: PlacedNode) -> bool:
        """Decide the rows ``placed`` runs on, or report sources on rows of different tables."""
        ones = [s.level for p in placed.inputs.values() if p.spec.shape == "one" for s in p.sources]
        gathers = [s for p in placed.inputs.values() if p.spec.shape != "one" for s in p.sources]
        # With no one-value input, a node runs one level above what its Series and Table inputs gather: each input
        # gathers at its deepest source (parents copied down), and the node sits above the shallowest input, so no
        # input receives a list of one.
        gathered = [_deepest([s.level for s in p.sources]) for p in placed.inputs.values() if p.spec.shape != "one" and p.sources]
        placed.level = _deepest(ones) if ones else min((level[:-1] for level in gathered), key=len, default=())
        floor = self._floor(placed.id)
        if not _under(floor, placed.level):  # an embedded graph runs on each row its embedding runs on
            if not _under(placed.level, floor):
                self._unrelated(placed, placed.level, floor)
                return False
            placed.level = floor
        for level in ones + [s.level for s in gathers]:
            if not (_under(placed.level, level) or _under(level, placed.level)):
                self._unrelated(placed, placed.level, level)
                return False
        for plan in placed.inputs.values():
            if plan.spec.shape == "table" and plan.sources:
                plan.table_level = _deepest([s.level for s in plan.sources])
                for s in plan.sources:
                    if not _under(s.level, plan.table_level):
                        self._unrelated(placed, s.level, plan.table_level)
                        return False
                names: list[str] = []
                for i, s in enumerate(plan.sources):
                    if not s.whole_table and s.name in names:
                        plan.sources[i] = dataclasses.replace(s, name=f"{s.name}_{s.node_id}")
                    names.append(plan.sources[i].name)
                if plan.spec.columns is not None and all(not s.whole_table or s.columns is not None for s in plan.sources):
                    given = {c for s in plan.sources for c in ((s.columns or {}) if s.whole_table else (s.name,))}
                    missing = sorted(plan.spec.columns.keys() - given)
                    if missing:
                        name = next(n for n, p in placed.inputs.items() if p is plan)
                        self.problems.append(make_problem(
                            "type_mismatch", placed.id, name, source=", ".join(f"{s.node_id}.{s.output}" for s in plan.sources),
                            got=f"the columns {sorted(given)}", expected=f"the columns {sorted(plan.spec.columns)}",
                        ))
                        return False
        return True

    def _floor(self, node_id: str) -> Level:
        """The rows an embedded node runs on at least: the rows its embedding runs on, set by the edges entering it."""
        levels: list[Level] = []
        for e in self._embeddings:
            if node_id.startswith(e + "/"):
                for address, target in self._entering[e]:
                    source = self._nodes.get(address.partition(".")[0])
                    if source is None:
                        continue
                    level = self._level_of(source, address)
                    target_id, _, target_input = target.partition(".")
                    target_record = self._records.get(target_id)
                    definition = self._definitions.get(target_record.node) if target_record is not None else None
                    spec = definition.version_of(target_record.version).inputs.get(target_input) if definition is not None and target_record is not None and definition.version_of(target_record.version) else None  # type: ignore[union-attr]
                    levels.append(level[:-1] if spec is not None and spec.shape != "one" and level else level)
        return _deepest(levels)

    def _level_of(self, source: PlacedNode, address: str) -> Level:
        """The rows the value at ``address`` sits on: its node's, or the table one of its outputs makes."""
        output = address.partition(".")[2].partition(".")[0]
        return source.makes.get(output, source.level)

    def _outside(self, graph_input: GraphInput) -> GraphInput:
        """An input as a caller of this graph sees it: an embedded graph's by the node that embeds it."""
        outer = self._outer.get(graph_input.key)
        if outer is None:
            return graph_input
        node_id, _, name = outer.partition(".")
        return dataclasses.replace(graph_input, address=outer, node_id=node_id, input=name)

    def _outer_id(self, node_id: str) -> str:
        """The node an author drew: an embedded node's embedding, else itself."""
        return node_id.split("/", 1)[0]

    def _seen_from_outside(self, problem: Problem) -> Problem:
        """A problem as the author of this record reads it: inside an embedded graph, on the node that embeds it."""
        if not problem.loc or not isinstance(problem.loc[0], str) or problem.loc[0].split("/", 1)[0] not in self._embeddings or "/" not in problem.loc[0]:
            return problem
        inner = problem.loc[0]
        field = problem.loc[1] if len(problem.loc) > 1 and isinstance(problem.loc[1], str) else None
        outer = self._outer.get(f"{inner}.{field}") if field else None
        loc: tuple[str | int, ...] = (outer.partition(".")[0], outer.partition(".")[2]) if outer else (self._outer_id(inner), *problem.loc[1:])
        if not outer and field is not None and len(problem.loc) > 1:
            loc = (self._outer_id(inner),)
        ctx = {**problem.ctx, **({"node_id": self._outer_id(str(problem.ctx["node_id"]))} if "node_id" in problem.ctx else {})}
        return make_problem(problem.type, *loc, **ctx)

    def _unrelated(self, placed: PlacedNode, a: Level, b: Level) -> None:
        self.problems.append(make_problem("rows_not_related", placed.id, a=_table(a), b=_table(b)))


def _interface(
    placed_records: list[tuple[NodeRecord, NodeDefinition[..., Any] | None]], edges: list[tuple[str, str]], prefix: str = "",
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """An embedded graph's inputs (the open inputs of its first nodes) and outputs (what nothing reads), by name.

    A name is the bare field name where it is unique, else ``node.field`` with the node's id as its own record says it.
    """
    fed = {b.partition(".")[0] for _, b in edges}
    read = {a.partition(".")[0] for a, _ in edges}
    ins = [(r.id, n) for r, d in placed_records if d is not None and r.id not in fed for n in d.inputs if n not in r.values]
    outs = [(r.id, n) for r, d in placed_records if d is not None and r.id not in read for n in d.outputs]
    bare_in, bare_out = [n for _, n in ins], [n for _, n in outs]
    return (
        {(n if bare_in.count(n) == 1 else f"{b.removeprefix(prefix)}.{n}"): (b, n) for b, n in ins},
        {(n if bare_out.count(n) == 1 else f"{b.removeprefix(prefix)}.{n}"): f"{b}.{n}" for b, n in outs},
    )


def _follow(address: str, renamed: Mapping[str, str]) -> str:
    """Rewrite ``node.field[.column]`` to the inner node an embedded graph spliced in for it."""
    node_id, _, rest = address.partition(".")
    for cut in (rest, rest.rpartition(".")[0]):
        if cut and f"{node_id}.{cut}" in renamed:
            return renamed[f"{node_id}.{cut}"] + rest[len(cut):]
    return address


def _arriving(source: Source) -> Any:
    """The type a hook reads for what arrives on an edge: a whole table as ``Table[R]`` with its columns, else the value's type."""
    if source.whole_table and source.columns is not None:
        from conductor._internal.fields import Table

        return Table.__class_getitem__(typing.TypedDict("Arriving", dict(source.columns)))  # type: ignore[operator]
    return source.type


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


def _deepest(levels: list[Level]) -> Level:
    return max(levels, key=len, default=())


def _table(level: Level) -> str:
    return level[-1] if level else "the graph's own row"


def _jsonable(spec: FieldSpec, value: Any) -> Any:
    try:
        return spec.adapter.dump_python(value, mode="json")
    except Exception:  # noqa: BLE001  a value with no JSON form shows as nothing
        return None
