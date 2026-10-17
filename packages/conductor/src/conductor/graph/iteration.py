"""One walk over the edges: what arrives on every field, whether each edge is allowed, and how each input receives.

Called by the compiler once every node's inputs are known. It walks the
nodes in execution order and, for each edge into each input, answers
three questions.

May this edge land? ``target.accepts(source)`` decides; a series is
judged by its element type. A refusal is the fatal ``type_mismatch``.

Does the node run once, or once per row? A series is a value with many
rows, and an index names where those rows come from. A scalar input is
an input that takes one value. A series arriving on a scalar input means
the node runs once per row of that index; we say the node runs per row
of the index. Its other scalar inputs are the same value every row, every
output becomes a series on the same index, and nodes downstream receive
a series and run per row in turn. Nothing is stored or marked to make this
happen; "receives a series" is the whole rule.

Do its inputs agree? A node fed series on several inputs needs their
indexes to be related: one must be the other, or a child of it (a child
index is made when a node splits each row into several). The node runs
once per row of the deepest one; a value on a shallower, parent index is
the same for every child row under it. Two indexes that are not related
are the fatal ``misaligned``.

Before those questions, inputs with no type of their own are typed from
their edges: a parameter declared ``Any`` takes the element type of what
arrives, and the node runs once per row if a series arrives; a
``**inputs`` parameter takes what arrives per edge — ``**inputs: Single``
receives each value whole, series and all, and never makes the node run
per row; ``**inputs: Series`` treats each as a series to reduce. Once
every input is typed, ``compute_outputs`` is asked with the arriving
types and the node's outputs are complete.

A ``Series[X]`` input receives a whole series; what it does depends on
the index that series is on. One series on a child index is a reduction:
the node runs once per parent row and receives the child rows under it.
One series on a root index is received whole, once. Anything else
feeding the input — several scalars, several series on different
indexes, a list set on the node, a default — is gathered onto a fresh index
that belongs to the input, and the node runs once.

Every one of those answers is written down as the input's receive record
(``conductor.graph.receive``): one row at a time on which index, broadcast,
whole, grouped at which depth, or gathered. The engine reads the record and decides
nothing of this again.

A scalar input holding a list set on the node (a multi-file upload, a list
typed by hand) is a series on an index of its own, under each row the node
otherwise runs on: questions set on a node that runs per document make a
questions index under the documents index, and the node runs once per
question per document. A node that otherwise runs once keeps its list on a
root index. Only one list may be set on a node that runs per row
(``lists_on_a_per_row_node``); two lists on a node that runs once are
``misaligned``, as two unrelated series would be.

Indexes are only named here; the engine adds rows to them later. Every
index belongs to the field that makes its rows and is named after it
(``_rows_of``): a series output ``split.lines`` sits on
``Index("split.lines")`` — a root when the node runs once, a child of the
node's own index when it runs per row — so two series outputs of one node
are two indexes, each with its own length. A gathering ``Series[X]`` input
sits on an index named after the input. The compiler reasons about which
index, never about how many rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from conductor.dtype import DType
from conductor.dtype_ref import description_of, name_of
from conductor.errors import Refuses
from conductor.graph.binding import From
from conductor.graph.embedding import SEPARATOR, address, as_drawn, under
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.receive import Broadcast, Gather, Merge, PerRow, Receive, RowsUnder, Whole
from conductor.graph.views import field_problems
from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.table import Table, column_type, columns_of, holds_tables

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph


@dataclass(frozen=True)
class Iteration:
    """What ``derive`` returns.

    ``runs_per_row_of``: for each node, the index it runs once per row of, or
    ``None`` when it runs once; for each embedded graph whose entering
    series agree, the index its inner nodes run per row of. A node has an
    entry exactly when the walk derived it. ``views``: each column a table
    output offers as a field, by address. ``types``: the type that travels on every
    field; ``indexes``: for a field carrying a series, where its rows come
    from (``None`` otherwise). ``receives``: for every input, how a unit
    receives its value. ``interfaces``: each node's inputs and outputs
    with the types the edges gave them — what ``CompiledGraph.node(...).interface``
    serves from then on. ``problems``: what went wrong.
    """

    runs_per_row_of: Mapping[str, Index | None]
    types: Mapping[Ref, Any]
    indexes: Mapping[Ref, Index | None]
    receives: Mapping[Ref, Receive]
    interfaces: Mapping[str, Interface]
    problems: tuple[Problem, ...]
    #: Every column of a table output offered as a field of its own, by its
    #: address (``csv.rows.email``): the table output and the column.
    views: Mapping[Ref, tuple[Ref, str]]
    #: Every table output, by address: the index its rows are on, which the
    #: table output owns and its columns are offered on.
    tables: Mapping[Ref, Index]


def derive(
    nodes: Sequence[GraphNode],
    interfaces: Mapping[str, Interface],
    versions: Mapping[str, Any],
    definitions: Mapping[str, type[NodeDefinition]],
    statics: Mapping[str, Mapping[str, Any]],
    listed: Mapping[str, frozenset[str]],
    graphs: frozenset[str],
) -> Iteration:
    """Walk ``nodes`` in execution order and record, for each, whether it
    runs once per row, what type every field carries, how every input
    receives, and its completed inputs and outputs.

    ``nodes`` holds only nodes whose edges all point at existing nodes
    (the compiler leaves the rest out). A node fed by a node that was
    left out or could not be resolved gets no entry in ``runs_per_row_of``,
    ``types``, ``indexes`` or ``receives`` — the broken source carries the
    problem, and nothing downstream of a fault is guessed at — but its
    inputs and outputs are still completed from what did arrive, so an
    editor can draw its handles.

    ``statics`` holds the values the author typed, by node and input, each
    read through its declared type; ``listed`` names, per node, the scalar
    inputs where the author typed many values (three files, a list of
    texts), which the node runs once per value of.

    ``compute_outputs`` is asked once per node, here, with ``arriving``
    (the type each connected input receives per call) and the values the
    author typed laid over the declaration's defaults. A hook that raises
    ``Refuses`` makes its code and message the node's one fatal problem.

    A node in ``graphs``, whose version is a compiled graph, is walked as
    one node with that graph's interface; ``Compilation.embed_graphs`` embeds
    the graph under the row the walk decides for it.
    """
    walk = _Walk(nodes, interfaces, versions, definitions, statics, listed, graphs)
    for node in nodes:
        walk.visit(node)
    return walk.result()


@dataclass(frozen=True, slots=True)
class _Carried:
    """What one field carries, while the walk works: its type, and the index
    its rows come from when it is a series (``None`` otherwise). Split into
    ``types`` and ``indexes`` when the walk ends."""

    dtype: Any
    index: Index | None


@dataclass
class _Arrivals:
    """What one node's inputs receive, gathered while its edges are read.

    ``arrivals``: per input, by address, what it carries — the type, and the index for
    a series. ``receives``: per input, how a unit receives it. ``arriving``:
    per connected input, the type the node sees per call (an element,
    where a series is sliced per row) — what ``compute_outputs`` is told.
    ``demands``: the indexes the node must run per row of, each with the
    input that demands it. ``bound``: the types the edges gave to inputs
    that had none of their own. ``broken`` is set when a source cannot be
    read and the walk stopped at that input.
    """

    arrivals: dict[Ref, _Carried] = dataclasses.field(default_factory=dict)
    receives: dict[Ref, Receive] = dataclasses.field(default_factory=dict)
    arriving: dict[str, Any] = dataclasses.field(default_factory=dict)
    demands: list[tuple[Index, Ref]] = dataclasses.field(default_factory=list)
    #: The inputs holding a list set on the node, each with its declared
    #: type; its rows go under the rows the node runs on, once those are known.
    listed: list[tuple[Ref, Any]] = dataclasses.field(default_factory=list)
    #: The ``Table`` inputs, each with its sources: how each is received is
    #: decided once the rows the node runs on are known.
    merging: list[tuple[Ref, Any, list[tuple[Ref, _Carried]]]] = dataclasses.field(default_factory=list)
    bound: dict[str, Any] = dataclasses.field(default_factory=dict)
    broken: bool = False


class _Walk:
    """The walk over the edges, one node at a time in execution order.

    ``visit`` reads what arrives on a node's inputs, decides whether it
    runs once per row, completes its inputs and outputs and records what
    every field carries; later nodes read those records as their sources.
    ``result`` freezes what was found into an ``Iteration``.
    """

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        interfaces: Mapping[str, Interface],
        versions: Mapping[str, Any],
        definitions: Mapping[str, type[NodeDefinition]],
        statics: Mapping[str, Mapping[str, Any]],
        listed: Mapping[str, frozenset[str]],
        graphs: frozenset[str],
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        #: Each node's inputs and outputs as the input hook answered them,
        #: before the walk types them.
        self.asked = interfaces
        self.versions = versions
        #: The class each node resolved to, which ``compute_outputs`` is asked of.
        self.definitions = definitions
        self.statics = statics
        self.listed = listed
        #: The nodes whose version is a compiled graph (``Layout.graphs``).
        self.graphs = graphs
        #: The index each visited node runs once per row of (``None``: once).
        self.runs_per_row_of: dict[str, Index | None] = {}
        #: What every output of every visited node carries: the sources a
        #: later node's edges may name. Only outputs are sources.
        self.carried: dict[Ref, _Carried] = {}
        #: What every input of every visited node carries.
        self.arrived: dict[Ref, _Carried] = {}
        #: Each column a table output offers as a field: the table output and the column.
        self.views: dict[Ref, tuple[Ref, str]] = {}
        #: Each table output: the index its rows are on.
        self.tables: dict[Ref, Index] = {}
        #: Where the rows of each output can come from, by index id: the
        #: index it sits on and every index above it, and for a table a node
        #: returns, the origins of every table it took in too, since rows keep
        #: their ids through a filter or a sort. ``None``: not known here (a
        #: field inside an embedded graph), which never warns.
        self.origins: dict[Ref, frozenset[str] | None] = {}
        #: The origins a table's rows carry, by the id of the index they are on.
        self.index_origins: dict[str, frozenset[str] | None] = {}
        #: How every input of every visited node receives its value.
        self.receives: dict[Ref, Receive] = {}
        #: Each visited node's inputs and outputs, completed: every type the
        #: edges gave them, and the outputs as ``compute_outputs`` answered.
        #: What ``Iteration.interfaces`` serves.
        self.completed: dict[str, Interface] = {}
        self.problems: list[Problem] = []

    def result(self) -> Iteration:
        fields = {**self.arrived, **self.carried}
        return Iteration(
            runs_per_row_of=self.runs_per_row_of,
            types={ref: c.dtype for ref, c in fields.items()},
            indexes={ref: c.index for ref, c in fields.items()},
            receives=self.receives,
            interfaces=self.completed,
            problems=tuple(self.problems),
            views=self.views,
            tables=self.tables,
        )

    def visit(self, node: GraphNode) -> None:
        """Read ``node``'s inputs, then either derive everything about it or,
        when a source could not be read, complete its fields from what did
        arrive and derive nothing."""
        arrived = self._read_inputs(node)
        if arrived.broken:
            self._complete_without_deriving(node, arrived)
        else:
            self._derive(node, arrived)

    def _read_inputs(self, node: GraphNode) -> _Arrivals:
        """What arrives on each input of ``node``, how each is received, and whether each edge is allowed.

        An input nothing feeds carries what the author typed (a scalar, or a
        series on an index of its own). A connected input is checked: every
        source must have been visited and have that output; a ``**inputs``
        parameter takes one edge whole; an input with no type of its own
        takes its first edge's; ``accepts`` judges every edge; a scalar input
        fed a series demands its index, a ``Series[X]`` input fed one series
        reduces on the parent index or receives the series whole, and fed
        anything else gathers onto a fresh index. Stops at the first input
        that cannot be read and says so in ``broken``.
        """
        interface = self.asked[node.id]
        found = _Arrivals()

        graphs = self.graphs if node.id in self.graphs else ()
        for inp in interface.inputs:
            ref = address(node.id, inp.name, graphs) if graphs else Ref(node.id, inp.name)
            binding = node.bindings.get(inp.name)
            if not isinstance(binding, From):
                if inp.name in self.listed[node.id]:
                    found.listed.append((ref, inp.dtype))
                    continue
                found.arrivals[ref], found.receives[ref] = self._originates(inp, ref)
                continue
            missing = [source for source in binding.refs if source not in self.carried]
            if missing:
                # A source that was resolved but has no such output: the ref names
                # an output it does not have, or an input. A source that was never
                # resolved carries its own problem and is not reported again here.
                self.problems.extend(
                    problem("unknown_ref_output", node.id, inp.name, source=str(source))
                    for source in missing
                    if source.node_id in self.runs_per_row_of or (self.graphs and as_drawn(source, self.graphs)[0] in self.runs_per_row_of)
                )
                found.broken = True
                return found
            sources = [self.carried[source] for source in binding.refs]
            if self._whole(node.id, inp):
                # A `**inputs` parameter takes what arrives, whole. It is passed
                # as a keyword argument, so its name must be an identifier.
                if not inp.name.isidentifier():
                    self.problems.append(problem("parameter_name_invalid", node.id, inp.name))
                    found.broken = True
                    return found
                if len(sources) > 1:
                    self.problems.append(problem("one_edge_per_parameter", node.id, inp.name))
                    found.broken = True
                    return found
                refused = self._refuses_whole(node.id, inp.name, sources[0].dtype)
                if refused is not None:
                    self.problems.append(refused)
                    found.broken = True
                    return found
                found.bound[inp.name] = sources[0].dtype
                found.arriving[inp.name] = sources[0].dtype
                found.arrivals[ref] = sources[0]
                found.receives[ref] = Whole()
                continue
            target = inp.dtype
            if isinstance(target, type) and issubclass(target, Table):
                # A table input takes tables and columns, merged on their rows' ids.
                for source_ref, source in zip(binding.refs, sources, strict=True):
                    self.problems.extend(self._admit_into_table(target, ref, source_ref, source))
                    if holds_tables(source.dtype) and source.index is not None:
                        found.demands.append((source.index, ref))  # a table per row: the node runs per row of it
                found.merging.append((ref, target, list(zip(binding.refs, sources, strict=True))))
                found.arriving[inp.name] = target
                continue
            if target is Any or target.element is Any:
                # An input with no type of its own takes the type of its first
                # edge; the other edges are checked against it.
                element = sources[0].dtype.element or sources[0].dtype
                target = Series[element] if getattr(inp.dtype, "element", None) is not None else element
                found.bound[inp.name] = target
            for source_ref, source in zip(binding.refs, sources, strict=True):
                self.problems.extend(self._admit(target, ref, source_ref, source))
            if target.element is None:
                if len(sources) > 1 and not self._one_index(sources):
                    self.problems.append(problem("union_needs_one_index", node.id, inp.name))
                    found.broken = True
                    return found
                arriving = sources[0]  # one ref, or a union of several on one index
                found.arriving[inp.name] = arriving.dtype.element or arriving.dtype
                if arriving.index is None:
                    found.receives[ref] = Broadcast()
                else:
                    found.receives[ref] = PerRow(arriving.index)
                    found.demands.append((arriving.index, ref))
            elif self._one_index(sources):
                arriving = sources[0]  # one series, or a union of several on one index
                found.arriving[inp.name] = arriving.dtype
                if arriving.index.parent is not None:
                    found.receives[ref] = RowsUnder(arriving.index, arriving.index.parent.depth)
                    found.demands.append((arriving.index.parent, ref))
                else:
                    found.receives[ref] = Whole()
            else:
                arriving = _Carried(target, Index(ref))
                found.arriving[inp.name] = target
                found.receives[ref] = Gather(arriving.index)
            found.arrivals[ref] = arriving
        return found

    def _derive(self, node: GraphNode, arrived: _Arrivals) -> None:
        """Every input read: complete the node's fields, ask its outputs,
        decide the index it runs per row of, and record what each field
        carries and how each input receives."""
        interface = self.asked[node.id]
        inputs = tuple(self._typed(inp, arrived.bound) for inp in interface.inputs)
        answered = self._outputs(node, arrived.arriving)
        if isinstance(answered, Problem):
            # The refusal is the node's problem: fatal, with no `no_outputs`
            # beside it, and nothing is derived past it.
            self.problems.append(answered)
            self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=())
            return
        outputs = answered
        self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=outputs)
        faults = field_problems(node.id, outputs, frozenset(i.name for i in inputs), untyped_ok=False)
        faults += [
            problem("series_of_tables", node.id, out.name) for out in outputs
            if (element := getattr(out.dtype, "element", None)) is not None and (element is Table or columns_of(element))
        ]
        if faults:
            # An output no edge could carry, or a name used twice: the node has
            # its interface, so an editor draws it, and nothing is derived
            # from it, so no reader is told the output is missing.
            self.problems.extend(faults)
            return
        if not outputs:
            self.problems.append(problem("no_outputs", node.id))
        runs_per_row_of, disagreeing = self._runs_per_row_of(arrived.demands)
        if disagreeing is not None:
            self.problems.append(self._misaligned(node.id, *disagreeing))
            return
        if len(arrived.listed) > 1:
            (first, _), (second, _) = arrived.listed[:2]
            if runs_per_row_of is not None:
                self.problems.append(problem("lists_on_a_per_row_node", node.id, second.field, a=str(first), b=str(second)))
            else:
                self.problems.append(self._misaligned(node.id, first, second))
            return
        if arrived.listed:
            # The list's rows go under each row the node runs on, and the node
            # runs once per value of the list in each of those rows.
            (ref, dtype), = arrived.listed
            own = self._rows_of(ref, runs_per_row_of)
            arrived.arrivals[ref], arrived.receives[ref] = _Carried(Series[dtype], own), PerRow(own)
            runs_per_row_of = own
        for ref, target, sources in arrived.merging:
            arrived.arrivals[ref] = _Carried(target, None) if runs_per_row_of is None else _Carried(Series[target], runs_per_row_of)
            arrived.receives[ref] = Merge(tuple((source_ref, self._part(source, runs_per_row_of)) for source_ref, source in sources))
            self._warn_no_rows_in_common(ref, [source_ref for source_ref, _ in sources])
            self._check_columns(ref, target, sources)
        self.runs_per_row_of[node.id] = runs_per_row_of
        self.arrived.update(arrived.arrivals)
        self.receives.update(arrived.receives)
        version = self.versions[node.id]
        names = {out.name for out in outputs}
        taken_in = self._origins_taken_in(arrived)
        if node.id in self.graphs:
            self._tables_inside(node.id, version, runs_per_row_of)
        for out in outputs:
            ref = address(node.id, out.name, self.graphs)
            if node.id in self.graphs:
                self.carried[ref] = self._carried_out_of_graph(node.id, version, out, runs_per_row_of)
                self._views_out_of_graph(node.id, version, out, runs_per_row_of)
            elif columns_of(out.dtype) or holds_tables(out.dtype):
                self._carry_table(node.id, ref, out.dtype, runs_per_row_of, names)
            elif out.dtype.element is not None:
                self.carried[ref] = _Carried(out.dtype, self._rows_of(ref, runs_per_row_of))
            elif runs_per_row_of is not None:
                self.carried[ref] = _Carried(Series[out.dtype], runs_per_row_of)
            else:
                self.carried[ref] = _Carried(out.dtype, None)
            self._note_origins(node.id, ref, taken_in)

    def _carry_table(self, node_id: str, ref: Ref, dtype: Any, runs_per_row_of: Index | None, names: set[str]) -> None:
        """A table output: one table per run of its node, and each of its
        columns offered as a field of its own (``rows.email``), a series on
        the table's rows, an index the table output owns. A node output already
        named like a column's field is ``view_name_taken``."""
        self.carried[ref] = _Carried(Series[dtype], runs_per_row_of) if runs_per_row_of is not None else _Carried(dtype, None)
        rows = self.tables[ref] = self._rows_of(ref, runs_per_row_of)
        for column, holds in columns_of(dtype):
            name = f"{ref.field}.{column}"
            if name in names:
                self.problems.append(problem("view_name_taken", node_id, name, table=ref.field, column=column))
                continue
            view = Ref(ref.node_id, name)
            self.carried[view] = _Carried(Series[holds], rows)
            self.views[view] = (ref, column)

    def _note_origins(self, node_id: str, ref: Ref, taken_in: frozenset[str] | None) -> None:
        """Record where the rows of output ``ref`` (and its columns) can come from."""
        if node_id in self.graphs:
            self.origins[ref] = None
            return
        carried = self.carried[ref]
        if ref in self.tables:
            rows = self.tables[ref]
            known = None if taken_in is None else taken_in | {rows.id}
            self.index_origins[rows.id] = known
            for view, (table, _) in self.views.items():
                if table == ref:
                    self.origins[view] = known
            self.origins[ref] = known
            return
        self.origins[ref] = self._origins_of(carried.index)

    def _origins_of(self, index: Index | None) -> frozenset[str] | None:
        """Where the ids of rows on ``index`` can come from: the index itself,
        whose rows get ids of their own, or, for a table's rows, the ids that
        table carried in too."""
        if index is None:
            return frozenset()
        return self.index_origins.get(index.id, frozenset({index.id}))

    def _tables_inside(self, node_id: str, graph: CompiledGraph, row: Index | None) -> None:
        """The rows of every table inside an embedded graph carry ids that came
        from wherever its inputs did, which this walk can't see: unknown."""
        for rows in graph._compilation.iteration.tables.values():
            lifted = under(rows, f"{node_id}{SEPARATOR}", row)
            if lifted is not None:
                self.index_origins[lifted.id] = None

    def _origins_taken_in(self, arrived: _Arrivals) -> frozenset[str] | None:
        """The origins of every table and column wired into this node's ``Table`` inputs: a table it
        returns may carry their rows' ids. ``None`` when any of them is unknown."""
        taken: set[str] = set()
        for _, _, sources in arrived.merging:
            for source_ref, _ in sources:
                known = self.origins.get(source_ref)
                if known is None:
                    return None
                taken |= known
        return frozenset(taken)

    def _warn_no_rows_in_common(self, ref: Ref, sources: list[Ref]) -> None:
        """A source of a ``Table`` input whose rows can't share an origin with any
        other source merges into blanks; the author is told while drawing. Nothing is refused."""
        found = [self.origins.get(source) for source in sources]
        known = [origins for origins in found if origins is not None]
        if len(sources) < 2 or len(known) < len(found):
            return
        rows = [self.tables.get(source) or self.carried[source].index for source in sources]
        above = [frozenset() if index is None else frozenset(step.id for step in index.lineage[:-1]) for index in rows]

        def meet(n: int, m: int) -> bool:
            """Can a row of source ``n`` meet one of source ``m``: by one id, or as its parent copied down?"""
            return bool(known[n] & known[m] or known[n] & above[m] or known[m] & above[n])

        for n, source in enumerate(sources):
            if not any(meet(n, m) for m in range(len(sources)) if m != n):
                self.problems.append(problem("no_rows_in_common", ref.node_id, ref.field, source=str(source)))
                return

    def _check_columns(self, ref: Ref, target: Any, sources: list[tuple[Ref, _Carried]]) -> None:
        """A ``Table[...]`` input names the columns it needs; when every source's
        columns are known (a bare table's are not), one nothing wired in has is
        ``table_column_missing``. A column a source holds but can't be read as
        the input wants fails the node when it runs."""
        if not columns_of(target):
            return
        available: set[str] = set()
        for source_ref, source in sources:
            if holds_tables(source.dtype):
                declared = columns_of(source.dtype.element or source.dtype)
                if not declared:
                    return
                available |= {name for name, _ in declared}
            else:
                view = self.views.get(source_ref)
                available.add(view[1] if view is not None else source_ref.field)
        missing = next((name for name, _ in columns_of(target) if name not in available), None)
        if missing is not None:
            self.problems.append(problem("table_column_missing", ref.node_id, ref.field, column=missing))

    def _admit_into_table(self, target: Any, ref: Ref, source_ref: Ref, source: _Carried) -> list[Problem]:
        """A ``Table`` input takes a table, or a column: a series of a type a
        table column holds. A single value or anything else is ``type_mismatch``."""
        element = source.dtype.element
        if holds_tables(source.dtype) or (element is not None and isinstance(element, type) and column_type(element) is not None):
            return []
        return self._admit(target, ref, source_ref, source) or [problem(
            "type_mismatch", ref.node_id, ref.field, source=str(source_ref), source_type=description_of(source.dtype),
            target_type=description_of(target), source_said=name_of(source.dtype), target_said=name_of(target),
        )]

    @staticmethod
    def _part(source: _Carried, runs_per_row_of: Index | None) -> PerRow | Broadcast | Whole | RowsUnder:
        """How a unit receives one source of a ``Table`` input: a table at its row
        or the one there is; a column's rows under its row; or all of them."""
        index = source.index
        if index is None:
            return Broadcast()
        lineage = set(runs_per_row_of.lineage) if runs_per_row_of is not None else set()
        if index in lineage:
            return PerRow(index)
        shared = [step for step in index.lineage[:-1] if step in lineage]
        if shared:
            return RowsUnder(index, shared[-1].depth)
        return Whole()

    def _views_out_of_graph(self, node_id: str, graph: CompiledGraph, out: Output, row: Index | None) -> None:
        """The columns a table output of a compiled graph offers, at the graph
        node, as they will stand once its nodes are in its place."""
        iteration = graph._compilation.iteration
        prefix = f"{node_id}{SEPARATOR}"
        if Ref(out.name) in iteration.tables:
            self.tables[address(node_id, out.name, self.graphs)] = under(iteration.tables[Ref(out.name)], prefix, row)
        for view, (table, column) in iteration.views.items():
            if str(table) == out.name:
                at = address(node_id, f"{out.name}.{column}", self.graphs)
                self.carried[at] = _Carried(iteration.types[view], under(iteration.indexes[view], prefix, row))
                self.views[at] = (address(node_id, out.name, self.graphs), column)
                self.origins[at] = None

    @staticmethod
    def _rows_of(ref: Ref, runs_per_row_of: Index | None) -> Index:
        """The index the rows an output makes sit on: named after the output
        (``split.lines``), under the row its node runs on. The one place an
        output's index is named; an input's own index is named after the input
        (``_originates``, a gathering ``Series[X]``)."""
        return Index(str(ref), parent=runs_per_row_of)

    @staticmethod
    def _carried_out_of_graph(node_id: str, graph: CompiledGraph, out: Output, row: Index | None) -> _Carried:
        """What an output of a compiled graph embedded as one node carries, as it
        will once the graph's nodes are in its place: the type and rows the
        output has inside the graph, its rows named
        under the graph node and hung under the ``row`` it runs on
        (``embedding.under``). Two outputs share rows only when one node inside
        makes the rows of both; an output that runs once inside is one value per
        ``row``, or one value."""
        ref, iteration = Ref(out.name), graph._compilation.iteration
        dtype, index = iteration.types[ref], iteration.indexes[ref]
        if index is None:
            return _Carried(Series[dtype], row) if row is not None else _Carried(dtype, None)
        return _Carried(dtype, under(index, f"{node_id}{SEPARATOR}", row))

    def _complete_without_deriving(self, node: GraphNode, arrived: _Arrivals) -> None:
        """A source was broken: complete the inputs and outputs from what
        arrived so an editor can draw the node, and derive nothing."""
        answered = self._outputs(node, arrived.arriving)
        if isinstance(answered, Problem):
            # The broken source carries the fault; a refusal about its missing
            # arrival would report the same fact twice.
            answered = ()
        inputs = tuple(self._typed(inp, arrived.bound) for inp in self.asked[node.id].inputs)
        self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=answered)
        # An output still ``Any`` here is one an edge would have typed; only a
        # name used twice or a type no edge can carry is the node's own fault.
        self.problems.extend(field_problems(node.id, answered, frozenset(i.name for i in inputs), untyped_ok=True))

    def _outputs(self, node: GraphNode, arriving: Mapping[str, Any]) -> tuple[Output, ...] | Problem:
        """Ask the node's ``compute_outputs`` now that it can be told what
        arrives, with the values the author typed laid over the
        declaration's defaults. An optional input is always there; a
        required one the author left empty, or connected, is not.

        A hook that cannot answer for these values and arrivals raises
        ``Refuses(code, message)``, and the refusal comes back as the node's
        one fatal ``Problem`` — the hook chose the code and the message,
        compile only records which node it belongs to.
        """
        version = self.versions[node.id]
        if node.id in self.graphs:
            return version.interface.outputs  # a compiled graph embedded as a node: it has no hooks
        declared = (*version.interface.inputs, *self.asked[node.id].inputs)
        values = {**{i.name: i.default for i in declared if i.optional}, **self.statics[node.id]}
        try:
            return self.definitions[node.id]().compute_outputs(version.interface.outputs, values, arriving)
        except Refuses as refusal:
            return Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node.id)

    def _whole(self, node_id: str, inp: Input) -> bool:
        """Is ``inp`` a ``**inputs: Single`` parameter — a connected name the
        version did not declare, received whole?"""
        version = self.versions[node_id]
        return version.interface.open == "single" and inp.name not in {i.name for i in version.interface.inputs}

    @staticmethod
    def _typed(field: Any, bound: Mapping[str, Any]) -> Any:
        """``field`` with the type its edge gave it, if an edge gave one."""
        if field.name in bound:
            return field.model_copy(update={"dtype": bound[field.name]})
        return field

    @staticmethod
    def _one_index(sources: Sequence[_Carried]) -> bool:
        """Do all sources carry a series on one and the same index? Then the edges together are one set of rows."""
        return all(s.index is not None for s in sources) and len({s.index for s in sources}) == 1

    @staticmethod
    def _refuses_whole(node_id: str, field: str, dtype: type[DType]) -> Problem | None:
        """A source type that says it cannot be handed over whole — a table whose
        columns nobody stated — is refused on the field, fatal, with the type's
        own message. Asked only where a node will *read* the value, a
        ``**inputs`` parameter; an ``Any`` input only passes the value on, and
        nothing is asked.
        """
        asked = dtype.element or dtype
        try:
            answered = asked.refuses_whole()
        except Refuses as refusal:
            return Problem(
                code=refusal.code, message=f"Field '{field}': {refusal.message}", fatal=True, node_id=node_id,
                field=field, details={"inner_message": refusal.message},
            )
        if answered is not None:
            raise TypeError(
                f"{asked.__name__}.refuses_whole() returned {answered!r}; a type refuses by raising Refuses "
                "and returns None otherwise"
            )
        return None

    @staticmethod
    def _originates(inp: Input, ref: Ref) -> tuple[_Carried, Receive]:
        """What a field carries when no edge feeds it, and how it is received:
        a scalar, broadcast; or, for a ``Series[X]`` input, a series on an
        index of the field's own, received whole. A list set on a scalar
        input is placed by ``_derive``, under the rows the node runs on.
        """
        if getattr(inp.dtype, "element", None) is not None:  # only a Series type has an element
            return _Carried(inp.dtype, Index(ref)), Whole()
        return _Carried(inp.dtype, None), Broadcast()

    @staticmethod
    def _admit(target: Any, ref: Ref, source_ref: Ref, source: _Carried) -> list[Problem]:
        """May what arrives land on ``ref``? ``target.accepts`` decides."""
        if not target.accepts(source.dtype):
            return [problem(
                "type_mismatch", ref.node_id, ref.field,
                source=str(source_ref),
                source_type=description_of(source.dtype),
                target_type=description_of(target),
                source_said=name_of(source.dtype),
                target_said=name_of(target),
            )]
        return []

    @staticmethod
    def _runs_per_row_of(demands: list[tuple[Index, Ref]]) -> tuple[Index | None, tuple[Ref, Ref] | None]:
        """The index a node runs once per row of, given the indexes its inputs
        demand — or the two fields whose indexes disagree.

        Every demanded index must lie on one line of descent: an index and the
        indexes opened from its rows. The deepest is the answer; a shallower
        one is an ancestor whose rows repeat down to it.
        """
        if not demands:
            return None, None
        deepest, deepest_ref = max(demands, key=lambda demand: demand[0].depth)
        lineage = set(deepest.lineage)
        for index, ref in demands:
            if index not in lineage:
                return None, (deepest_ref, ref)
        return deepest, None

    @staticmethod
    def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
        return problem("misaligned", node_id, a=str(a), b=str(b))
