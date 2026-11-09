"""What arrives on each input of one node, whether each edge may land, how each input receives, and on which rows the node runs.

May this edge land? ``target.accepts(source)``, element against element; a
refusal is the fatal ``type_mismatch``.

Does the node run once, or once per row? A column arriving on an input that
takes one value makes the node run once per row of it, and its outputs become
columns on the same rows; "receives a series" is the whole rule. Several
columns must sit on one line of descent (the deepest wins, a parent value is
copied down), or they are the fatal ``misaligned``.

A ``Series[X]`` input fed one column on a child level reduces: a call receives
all the values under its own row, however deep, flat. It doesn't choose the
row unless nothing else does; then the node runs per row of the level right
above. One column on a root is received whole, once. Anything else feeding it
(several scalars, unrelated columns, a list set on the node, a default) is
collected onto a level of the input's own.

A ``Table`` input merges everything wired in, matched on row ids; it never
makes the node run per row by itself, except a table arriving per row.

A ``**inputs: AsItArrives`` input takes what its edge carries, whole, and
never makes the node run per row. A parameter typed ``Any`` takes the type of
its first edge.

A list typed into an input that takes one value is a column of its own, under
each row the node otherwise runs on. Only one list may be set on a node that
runs per row (``lists_on_a_per_row_node``); two on a node that runs once are
``misaligned``.

Every answer is written as the input's receive record
(``conductor._internal.graph.receive``), which the engine reads and never decides again.
A level belongs to the field that makes its rows and is named after it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from conductor._internal.dtype import _ReceivedAsItArrives, description_of, is_untyped, name_of
from conductor._internal.errors import Problem, problem
from conductor._internal.field_records import Input
from conductor._internal.graph.binding import From
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.graph.receive import (
    Broadcast,
    EntireValue,
    Gather,
    Merge,
    PerRow,
    Receive,
    RowsUnder,
)
from conductor._internal.ref import Ref
from conductor._internal.series import Index, Series
from conductor._internal.table import Table, column_type, columns_of, holds_tables

if TYPE_CHECKING:
    from conductor._internal.graph.compile_graph import WalkSoFar
    from conductor._internal.graph.compiled_node import CompiledField


class Arrival(NamedTuple):
    """What one input carries: its type and, for a series, the level its rows are on."""

    dtype: Any
    rows: Index | None


@dataclass
class InputsRead:
    """What one node's inputs receive, gathered as its edges are read."""

    #: Per input name: what it carries, and how a call receives it.
    arrivals: dict[str, Arrival] = field(default_factory=dict)
    receives: dict[str, Receive] = field(default_factory=dict)
    #: Per connected input: the type one call sees, what ``compute_outputs`` is told.
    arriving: dict[str, Any] = field(default_factory=dict)
    #: The types edges gave to inputs that had none of their own.
    bound: dict[str, Any] = field(default_factory=dict)
    #: The rows the node must run per row of, each with the input that demands it.
    demands: list[tuple[Index, str]] = field(default_factory=list)
    #: The child levels ``Series[X]`` inputs read, placed once the node's row is known.
    reductions: list[tuple[Index, str]] = field(default_factory=list)
    #: The inputs holding a list set on the node, each with its declared type.
    listed: list[tuple[str, Any]] = field(default_factory=list)
    #: The ``Table`` inputs with their sources.
    merging: list[tuple[str, Any, list[tuple[Ref, CompiledField]]]] = field(default_factory=list)
    #: Every table and merged column taken in whole: a table the node returns may carry their ids.
    tables_taken_in: list[CompiledField] = field(default_factory=list)
    #: A source couldn't be read; the walk stopped at that input.
    broken: bool = False


def read_inputs(
    node: GraphNode, inputs: tuple[Input, ...], listed: frozenset[str], walk: WalkSoFar, problems: list[Problem],
) -> InputsRead:
    """What arrives on each of ``inputs``. Stops at the first input that can't be read and says so in ``broken``."""
    found = InputsRead()
    for inp in inputs:
        name = inp.name
        binding = node.bindings.get(name)
        if not isinstance(binding, From):
            if name in listed:
                found.listed.append((name, inp.dtype))
                continue
            found.arrivals[name], found.receives[name] = _originates(inp, node.id)
            continue
        feeding = [_source(source, walk) for source in binding.refs]
        if None in feeding:
            # A source that was worked out but has no such output is this input's
            # fault; one that wasn't carries its own problem already.
            problems.extend(
                problem("unknown_ref_output", node.id, name, source=str(ref_), outputs=_outputs_offered_by(ref_.node_id, walk))
                for ref_, source in zip(binding.refs, feeding, strict=True)
                if source is None and _source_node_is_wired(ref_, walk)
            )
            found.broken = True
            return found
        sources: list[CompiledField] = cast("list[CompiledField]", feeding)
        found.tables_taken_in.extend(source for source in sources if holds_tables(source.type))
        target = inp.dtype
        if target is _ReceivedAsItArrives:
            if not name.isidentifier():
                problems.append(problem("parameter_name_invalid", node.id, name))
                found.broken = True
                return found
            if len(sources) > 1:
                problems.append(problem("one_edge_per_parameter", node.id, name))
                found.broken = True
                return found
            found.bound[name] = found.arriving[name] = sources[0].type
            found.arrivals[name] = Arrival(sources[0].type, sources[0]._index)
            found.receives[name] = EntireValue()
            continue
        if _is_table_type(target):
            for source_ref, source in zip(binding.refs, sources, strict=True):
                problems.extend(_admit_into_table(target, node.id, name, source_ref, source))
                if holds_tables(source.type) and source._index is not None:
                    found.demands.append((source._index, name))
                elif not holds_tables(source.type):
                    found.tables_taken_in.append(source)
            found.merging.append((name, target, list(zip(binding.refs, sources, strict=True))))
            found.arriving[name] = target
            continue
        if is_untyped(target):
            element = sources[0].type.element or sources[0].type
            target = Series[element] if inp.dtype.element is not None else element
            found.bound[name] = target
        for source_ref, source in zip(binding.refs, sources, strict=True):
            problems.extend(_admit(target, node.id, name, source_ref, source))
        if target.element is None:
            if len(sources) > 1 and not _one_index(sources):
                problems.append(problem("union_needs_one_index", node.id, name))
                found.broken = True
                return found
            arriving = sources[0]
            found.arriving[name] = arriving.type.element or arriving.type
            if arriving._index is None:
                found.receives[name] = Broadcast()
            else:
                found.receives[name] = PerRow(arriving._index)
                found.demands.append((arriving._index, name))
            found.arrivals[name] = Arrival(arriving.type, arriving._index)
        elif _one_index(sources):
            arriving = sources[0]
            found.arriving[name] = arriving.type
            if arriving._index.parent is not None:
                found.reductions.append((arriving._index, name))
            else:
                found.receives[name] = EntireValue()
            found.arrivals[name] = Arrival(arriving.type, arriving._index)
        else:
            own = Index(Ref(node.id, name))
            found.arriving[name] = target
            found.receives[name] = Gather(own)
            found.arrivals[name] = Arrival(target, own)
    return found


@cache
def _is_table_type(dtype: Any) -> bool:
    return isinstance(dtype, type) and issubclass(dtype, Table)


def _source(ref: Ref, walk: WalkSoFar) -> CompiledField | None:
    """The output ``ref`` names, on a node above this one; ``None`` when no such output has been worked out."""
    source_id = ref.node_id
    node = walk.compiled.get(source_id)
    if node is None or node.wiring is None or source_id in walk.refused:
        return None
    found = node.wiring.fields.get(ref.field)
    return found if found is not None and found._is_output else None


def _outputs_offered_by(node_id: str, walk: WalkSoFar) -> str:
    """The names an edge may read from ``node_id``, for a sentence."""
    node = walk.compiled[node_id]
    if node.embedded_graph is not None:
        names = sorted(node.embedded_graph._for_embedding.outputs)
    else:
        names = sorted(name for name, field in node.wiring.fields.items() if field._is_output)  # type: ignore[union-attr]
    return ", ".join(names) or "none"


def _source_node_is_wired(ref: Ref, walk: WalkSoFar) -> bool:
    node = walk.compiled.get(ref.node_id)
    return node is not None and node.wiring is not None


def decide_rows(node: GraphNode, found: InputsRead, problems: list[Problem], row_origins: Any) -> tuple[bool, Index | None]:
    """The level ``node_id`` runs once per row of, with every input's receive record settled.

    ``False`` when its inputs disagree, with the fault in ``problems``."""
    node_id = node.id
    where = lambda name: Ref(node_id, name)  # noqa: E731
    demands = found.demands or _row_above_reductions(found.reductions)
    runs_on, disagreeing = _deepest(demands)
    if disagreeing is not None:
        problems.append(_misaligned(node_id, where(disagreeing[0]), where(disagreeing[1])))
        return False, None
    if not found.demands and any(runs_on not in index.lineage for index, _ in found.reductions):
        (_, first), (_, second) = found.reductions[:2]
        problems.append(_misaligned(node_id, where(first), where(second)))
        return False, None
    if len(found.listed) > 1:
        (first, _), (second, _) = found.listed[:2]
        if runs_on is not None:
            problems.append(problem("lists_on_a_per_row_node", node_id, second, a=str(where(first)), b=str(where(second))))
        else:
            problems.append(_misaligned(node_id, where(first), where(second)))
        return False, None
    if found.listed:
        (name, dtype), = found.listed
        own = Index(str(where(name)), parent=runs_on)
        found.arrivals[name], found.receives[name] = Arrival(Series[dtype], own), PerRow(own)
        runs_on = own
    for index, name in found.reductions:
        depth = _depth_shared_with_row(index, runs_on)
        if depth is None:
            problems.append(_misaligned(node_id, where(found.demands[0][1]), where(name)))
            return False, None
        found.receives[name] = RowsUnder(index, depth)
    for name, target, sources in found.merging:
        found.arrivals[name] = Arrival(target, None) if runs_on is None else Arrival(Series[target], runs_on)
        found.receives[name] = Merge(
            tuple((source_ref, _part(source, runs_on)) for source_ref, source in sources),
            frozenset(_copied_down(sources)),
        )
        problems.extend(row_origins.note_merge(node_id, name, sources))
        _check_columns(node_id, name, target, sources, problems)
    return True, runs_on


def _copied_down(sources: list[tuple[Ref, CompiledField]]) -> list[Ref]:
    """The sources whose rows sit above another source's: their values are copied down onto those rows."""
    rows = [(ref, source._makes_rows or source._index) for ref, source in sources]
    return [ref for ref, index in rows if index is not None and any(index != other and index in other.lineage for _, other in rows if other is not None)]


def _check_columns(node_id: str, name: str, target: Any, sources: list[tuple[Ref, CompiledField]], problems: list[Problem]) -> None:
    """A ``Table[...]`` input names the columns it needs; when every source's
    columns are known (a bare table's are not), one nothing wired in has is
    ``table_column_missing``. A column that can't be read as the input wants fails the node when it runs."""
    if not columns_of(target):
        return
    available: set[str] = set()
    for source_ref, source in sources:
        if holds_tables(source.type):
            declared = columns_of(source.type.element or source.type)
            if not declared:
                return
            available |= {column for column, _ in declared}
        else:
            available.add(source._view_of[1] if source._view_of is not None else source_ref.field)
    missing = next((column for column, _ in columns_of(target) if column not in available), None)
    if missing is not None:
        problems.append(problem("table_column_missing", node_id, name, column=missing))


def _admit_into_table(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> list[Problem]:
    """A ``Table`` input takes a table, or a column: a series of a type a
    table column holds. A single value or anything else is ``type_mismatch``."""
    element = source.type.element
    if holds_tables(source.type) or (element is not None and isinstance(element, type) and column_type(element) is not None):
        return []
    return _admit(target, node_id, name, source_ref, source) or [_type_mismatch(target, node_id, name, source_ref, source)]


def _admit(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> list[Problem]:
    """May what arrives land? The target's ``accepts`` decides, asked about one value of each."""
    if (target.element or target).accepts(source.type.element or source.type):
        return []
    return [_type_mismatch(target, node_id, name, source_ref, source)]


def _type_mismatch(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> Problem:
    return problem(
        "type_mismatch", node_id, name, source=str(source_ref), source_type=description_of(source.type),
        target_type=description_of(target), source_said=name_of(source.type), target_said=name_of(target),
    )


def _depth_shared_with_row(index: Index, runs_on: Index | None) -> int | None:
    """How far up ``index``'s rows are grouped for a node on ``runs_on``: the
    depth of the deepest level above them that the node's row sits on, ``None`` when none is."""
    lineage = () if runs_on is None else runs_on.lineage
    return next((step.depth for step in reversed(index.lineage[:-1]) if step in lineage), None)


def _part(source: CompiledField, runs_on: Index | None) -> PerRow | Broadcast | EntireValue | RowsUnder:
    """How a call receives one source of a ``Table`` input: a table at its row
    or the one there is; a column's rows under its row; or all of them."""
    index = source._index
    if index is None:
        return Broadcast()
    if runs_on is not None and index in runs_on.lineage:
        return PerRow(index)
    depth = _depth_shared_with_row(index, runs_on)
    return EntireValue() if depth is None else RowsUnder(index, depth)


def _one_index(sources: Sequence[CompiledField]) -> bool:
    """Do all sources carry a series on one and the same level? Then the edges together are one set of rows."""
    return all(s._index is not None for s in sources) and len({s._index for s in sources}) == 1


def _originates(inp: Input, node_id: str) -> tuple[Arrival, Receive]:
    """What an input carries when no edge feeds it: one value copied to every
    call, or for a ``Series[X]`` input a column of its own, received whole."""
    if inp.dtype.element is not None:
        return Arrival(inp.dtype, Index(Ref(node_id, inp.name))), EntireValue()
    return Arrival(inp.dtype, None), Broadcast()


def _row_above_reductions(reductions: list[tuple[Index, str]]) -> list[tuple[Index, str]]:
    """The one demand reductions make when nothing else does: the shallowest level right above a series."""
    if not reductions:
        return []
    index, name = min(reductions, key=lambda reduction: reduction[0].depth)
    return [(index.parent, name)]  # type: ignore[list-item]


def _deepest(demands: list[tuple[Index, str]]) -> tuple[Index | None, tuple[str, str] | None]:
    """The deepest level among ``demands`` if they lie on one line of descent, or the two fields that disagree."""
    if not demands:
        return None, None
    deepest, deepest_ref = max(demands, key=lambda demand: demand[0].depth)
    lineage = set(deepest.lineage)
    for index, ref in demands:
        if index not in lineage:
            return None, (deepest_ref, ref)
    return deepest, None


def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
    return problem("misaligned", node_id, a=str(a), b=str(b))
