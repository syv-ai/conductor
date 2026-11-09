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
makes the node run per row by itself, except a table arriving per row. What
it carries is the merged table's type, each column named as the merge will
name it, or what the input declares when it names columns.

A ``**inputs: AsItArrives`` input takes what its edge carries, whole, and
never makes the node run per row. A parameter typed ``Any`` takes the type of
its first edge.

A list typed into an input that takes one value is a column of its own, under
each row the node otherwise runs on, and an edge reads it like an output (``q.question``).
Only one list may be set on a node (``lists_on_a_per_row_node``).

Every answer is written as the input's receive record
(``conductor._internal.graph.receive_modes``), which the engine reads and never decides again.
A level belongs to the field that makes its rows and is named after it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from conductor._internal.dtype import (
    _ReceivedAsItArrives,
    description_of,
    is_untyped,
    type_name_for_sentence,
)
from conductor._internal.errors import Problem, make_problem
from conductor._internal.field_records import Input
from conductor._internal.graph.binding import From
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.graph.receive_modes import (
    Broadcast,
    EntireValue,
    Gather,
    Merge,
    PerRow,
    ReceiveMode,
    RowsUnder,
)
from conductor._internal.graph.row_origins import sources_taken_in_whole
from conductor._internal.ref import Ref
from conductor._internal.series import Index, Series
from conductor._internal.table import (
    Column,
    Table,
    columns_of,
    is_table_or_series_of_tables,
    polars_type_of,
    renames_in_merge,
)

if TYPE_CHECKING:
    from conductor._internal.graph.compile_graph import CompileProgress
    from conductor._internal.graph.compiled_node import CompiledField


class InputCarries(NamedTuple):
    """What one input carries: its type and, for a series, the level its rows are on."""

    dtype: Any
    index: Index | None


@dataclass
class WhatArrivesOnInputs:
    """What one node's inputs receive, gathered as its edges are read."""

    #: Per input name: what it carries, and how a call receives it.
    carries: dict[str, InputCarries] = field(default_factory=dict)
    receive_modes: dict[str, ReceiveMode] = field(default_factory=dict)
    #: Per connected input: the type one call sees, what ``compute_outputs`` is told.
    arriving_types: dict[str, Any] = field(default_factory=dict)
    #: The types edges gave to inputs that had none of their own.
    type_given_by_edge: dict[str, Any] = field(default_factory=dict)
    #: The rows the node must run per row of, each with the input that demands it.
    rows_required_by_inputs: list[tuple[Index, str]] = field(default_factory=list)
    #: The child levels ``Series[X]`` inputs read, placed once the node's row is known.
    child_indexes_read_by_series_inputs: list[tuple[Index, str]] = field(default_factory=list)
    #: The inputs holding a list set on the node, each with its declared type.
    lists_on_node: list[tuple[str, Any]] = field(default_factory=list)
    #: The ``Table`` inputs with the table type each carries and its sources.
    table_inputs_and_sources: list[tuple[str, Any, list[tuple[Ref, CompiledField]]]] = field(default_factory=list)
    #: Every table and merged column taken in whole: a table the node returns may carry their ids.
    tables_taken_in: list[CompiledField] = field(default_factory=list)
    #: A source couldn't be read; the walk stopped at that input.
    stopped_at_unreadable_source: bool = False


def read_what_arrives_on_inputs(
    node: GraphNode, inputs: tuple[Input, ...], inputs_given_a_list: frozenset[str], progress: CompileProgress, problems: list[Problem],
) -> WhatArrivesOnInputs:
    """What arrives on each of ``inputs``. Stops at the first input that can't be read and says so in ``stopped_at_unreadable_source``."""
    found = WhatArrivesOnInputs()
    for inp in inputs:
        name = inp.name
        binding = node.bindings.get(name)
        if not isinstance(binding, From):
            if name in inputs_given_a_list:
                found.lists_on_node.append((name, inp.dtype))
                continue
            found.carries[name], found.receive_modes[name] = _what_an_unwired_input_carries(inp, node.id)
            continue
        feeding = [_source_output(source, progress) for source in binding.refs]
        if None in feeding:
            # A source that was worked out but has no such output is this input's
            # fault; one that wasn't carries its own problem already.
            problems.extend(
                make_problem("unknown_ref_output", node.id, name, source=str(ref_), outputs=_outputs_offered_by(ref_.node_id, progress))
                for ref_, source in zip(binding.refs, feeding, strict=True)
                if source is None and _source_node_is_wired(ref_, progress)
            )
            found.stopped_at_unreadable_source = True
            return found
        sources: list[CompiledField] = cast("list[CompiledField]", feeding)
        target = inp.dtype
        found.tables_taken_in.extend(sources_taken_in_whole(target, sources))
        if target is _ReceivedAsItArrives:
            if not name.isidentifier():
                problems.append(make_problem("parameter_name_invalid", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            if len(sources) > 1:
                problems.append(make_problem("one_edge_per_parameter", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            found.type_given_by_edge[name] = found.arriving_types[name] = sources[0].type
            found.carries[name] = InputCarries(sources[0].type, sources[0]._index)
            found.receive_modes[name] = EntireValue()
            continue
        if _is_table_type(target):
            pairs = list(zip(binding.refs, sources, strict=True))
            problems_found = [problem for source_ref, source in pairs for problem in _type_mismatch_unless_table_or_column(target, node.id, name, source_ref, source)]
            problems.extend(problems_found)
            merged = None if problems_found else _merged_columns(pairs)
            found.rows_required_by_inputs.extend((source._index, name) for source in sources if is_table_or_series_of_tables(source.type) and source._index is not None)
            carried = _table_a_call_receives(target, merged)
            _check_columns(node.id, name, target, merged, problems)
            found.table_inputs_and_sources.append((name, carried, pairs))
            found.arriving_types[name] = carried
            continue
        if is_untyped(target):
            element = sources[0].type.element or sources[0].type
            target = Series[element] if inp.dtype.element is not None else element
            found.type_given_by_edge[name] = target
        for source_ref, source in zip(binding.refs, sources, strict=True):
            problems.extend(_type_mismatch_unless_accepted(target, node.id, name, source_ref, source))
        if target.element is None:
            if len(sources) > 1 and not _all_on_one_index(sources):
                problems.append(make_problem("union_needs_one_index", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            source = sources[0]
            found.arriving_types[name] = source.type.element or source.type
            if source._index is None:
                found.receive_modes[name] = Broadcast()
            else:
                found.receive_modes[name] = PerRow(source._index)
                found.rows_required_by_inputs.append((source._index, name))
            found.carries[name] = InputCarries(source.type, source._index)
        elif _all_on_one_index(sources):
            source = sources[0]
            found.arriving_types[name] = source.type
            if source._index.parent is not None:
                found.child_indexes_read_by_series_inputs.append((source._index, name))
            else:
                found.receive_modes[name] = EntireValue()
            found.carries[name] = InputCarries(source.type, source._index)
        else:
            input_own_index = Index(Ref(node.id, name))
            found.arriving_types[name] = target
            found.receive_modes[name] = Gather(input_own_index)
            found.carries[name] = InputCarries(target, input_own_index)
    return found


@cache
def _is_table_type(dtype: Any) -> bool:
    return isinstance(dtype, type) and issubclass(dtype, Table)


def _source_output(ref: Ref, progress: CompileProgress) -> CompiledField | None:
    """The output ``ref`` names, on a node above this one; ``None`` when no such output has been worked out."""
    source_id = ref.node_id
    node = progress.compiled.get(source_id)
    if node is None or node.wiring is None or source_id in progress.invalid_node_ids:
        return None
    found = node.wiring.fields.get(ref.field)
    return found if found is not None and (found._is_output or found._is_list_on_node) else None


def _outputs_offered_by(node_id: str, progress: CompileProgress) -> str:
    """The names an edge may read from ``node_id``, for a sentence."""
    node = progress.compiled[node_id]
    if node.embedded_graph is not None:
        names = sorted(node.embedded_graph._embeddable.outputs)
    else:
        names = sorted(name for name, field in node.wiring.fields.items() if field._is_output or field._is_list_on_node)  # type: ignore[union-attr]
    return ", ".join(names) or "none"


def _source_node_is_wired(ref: Ref, progress: CompileProgress) -> bool:
    node = progress.compiled.get(ref.node_id)
    return node is not None and node.wiring is not None


def decide_runs_per_row_of_and_receive_modes(node: GraphNode, found: WhatArrivesOnInputs, problems: list[Problem], row_origins: Any) -> tuple[bool, Index | None]:
    """The level ``node_id`` runs once per row of, with every input's receive record settled.

    ``False`` when its inputs disagree, with the fault in ``problems``."""
    node_id = node.id
    field_address_of = lambda name: Ref(node_id, name)  # noqa: E731
    required_rows = found.rows_required_by_inputs or _row_above_reductions(found.child_indexes_read_by_series_inputs)
    runs_per_row_of, disagreeing = _deepest_index_on_one_line_of_descent(required_rows)
    if disagreeing is not None:
        problems.append(_misaligned(node_id, field_address_of(disagreeing[0]), field_address_of(disagreeing[1])))
        return False, None
    if not found.rows_required_by_inputs and any(runs_per_row_of not in index.lineage for index, _ in found.child_indexes_read_by_series_inputs):
        (_, first), (_, second) = found.child_indexes_read_by_series_inputs[:2]
        problems.append(_misaligned(node_id, field_address_of(first), field_address_of(second)))
        return False, None
    if len(found.lists_on_node) > 1:
        (first, _), (second, _) = found.lists_on_node[:2]
        problems.append(make_problem("lists_on_a_per_row_node", node_id, second, a=str(field_address_of(first)), b=str(field_address_of(second))))
        return False, None
    if found.lists_on_node:
        (name, dtype), = found.lists_on_node
        input_own_index = Index(str(field_address_of(name)), parent=runs_per_row_of)
        found.carries[name], found.receive_modes[name] = InputCarries(Series[dtype], input_own_index), PerRow(input_own_index)
        runs_per_row_of = input_own_index
    for index, name in found.child_indexes_read_by_series_inputs:
        depth = _depth_shared_with_row(index, runs_per_row_of)
        if depth is None:
            problems.append(_misaligned(node_id, field_address_of(found.rows_required_by_inputs[0][1]), field_address_of(name)))
            return False, None
        found.receive_modes[name] = RowsUnder(index, depth)
    for name, carried, sources in found.table_inputs_and_sources:
        found.carries[name] = InputCarries(carried, None) if runs_per_row_of is None else InputCarries(Series[carried], runs_per_row_of)
        found.receive_modes[name] = Merge(
            tuple((source_ref, _receive_mode_of_merge_source(source, runs_per_row_of)) for source_ref, source in sources),
            frozenset(_copied_down(sources)),
        )
        problems.extend(row_origins.check_merge(node_id, name, sources))
    return True, runs_per_row_of


def _copied_down(sources: list[tuple[Ref, CompiledField]]) -> list[Ref]:
    """The sources whose rows sit above another source's: their values are copied down onto those rows."""
    rows = [(ref, source._makes_rows or source._index) for ref, source in sources]
    return [ref for ref, index in rows if index is not None and any(index != other and index in other.lineage for _, other in rows if other is not None)]


def _merged_columns(sources: list[tuple[Ref, CompiledField]]) -> tuple[Column, ...] | None:
    """The columns the merged table of ``sources`` has, named as the merge names them; ``None`` when a table among them names none."""
    columns_each: list[tuple[str, tuple[Column, ...]]] = []
    for source_ref, source in sources:
        if is_table_or_series_of_tables(source.type):
            columns = columns_of(source.type.element or source.type)
            if not columns:
                return None
        else:
            columns = ((source._title, source.type.element),)
        columns_each.append((source_ref.node_id, columns))
    renames = renames_in_merge((node_id, (name for name, _ in columns)) for node_id, columns in columns_each)
    return tuple((renamed.get(name, name), dtype) for (_, columns), renamed in zip(columns_each, renames, strict=True) for name, dtype in columns)


def _table_a_call_receives(target: Any, merged: tuple[Column, ...] | None) -> Any:
    """The table type a ``Table`` input carries, what a hook's ``Arriving`` is told: the merged table's, or what the input declares when it names columns, since those are what a call gets."""
    if columns_of(target) or merged is None:
        return target
    return target.with_columns(dict(merged))


def _check_columns(node_id: str, name: str, target: Any, merged: tuple[Column, ...] | None, problems: list[Problem]) -> None:
    """A ``Table[...]`` input names the columns it needs; when the merged table's columns are known, one it lacks is ``table_column_missing``."""
    if merged is None or not columns_of(target):
        return
    have = {column for column, _ in merged}
    missing = next((column for column, _ in columns_of(target) if column not in have), None)
    if missing is not None:
        problems.append(make_problem("table_column_missing", node_id, name, column=missing))


def _type_mismatch_unless_table_or_column(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> list[Problem]:
    """A ``Table`` input takes a table, or a column: a series of a type a
    table column holds. A single value or anything else is ``type_mismatch``."""
    element = source.type.element
    if is_table_or_series_of_tables(source.type) or (element is not None and isinstance(element, type) and polars_type_of(element) is not None):
        return []
    return _type_mismatch_unless_accepted(target, node_id, name, source_ref, source) or [_type_mismatch(target, node_id, name, source_ref, source)]


def _type_mismatch_unless_accepted(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> list[Problem]:
    """May what arrives land? The target's ``accepts`` decides, asked about one value of each."""
    if (target.element or target).accepts(source.type.element or source.type):
        return []
    return [_type_mismatch(target, node_id, name, source_ref, source)]


def _type_mismatch(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField) -> Problem:
    return make_problem(
        "type_mismatch", node_id, name, source=str(source_ref), source_type=description_of(source.type),
        target_type=description_of(target), source_said=type_name_for_sentence(source.type), target_said=type_name_for_sentence(target),
    )


def _depth_shared_with_row(index: Index, runs_per_row_of: Index | None) -> int | None:
    """How far up ``index``'s rows are grouped for a node on ``runs_per_row_of``: the
    depth of the deepest level above them that the node's row sits on, ``None`` when none is."""
    lineage = () if runs_per_row_of is None else runs_per_row_of.lineage
    return next((step.depth for step in reversed(index.lineage[:-1]) if step in lineage), None)


def _receive_mode_of_merge_source(source: CompiledField, runs_per_row_of: Index | None) -> PerRow | Broadcast | EntireValue | RowsUnder:
    """How a call receives one source of a ``Table`` input: a table at its row
    or the one there is; a column's rows under its row; or all of them."""
    index = source._index
    if index is None:
        return Broadcast()
    if runs_per_row_of is not None and index in runs_per_row_of.lineage:
        return PerRow(index)
    depth = _depth_shared_with_row(index, runs_per_row_of)
    return EntireValue() if depth is None else RowsUnder(index, depth)


def _all_on_one_index(sources: Sequence[CompiledField]) -> bool:
    """Do all sources carry a series on one and the same level? Then the edges together are one set of rows."""
    return all(source._index is not None for source in sources) and len({source._index for source in sources}) == 1


def _what_an_unwired_input_carries(inp: Input, node_id: str) -> tuple[InputCarries, ReceiveMode]:
    """What an input carries when no edge feeds it: one value copied to every
    call, or for a ``Series[X]`` input a column of its own, received whole."""
    if inp.dtype.element is not None:
        return InputCarries(inp.dtype, Index(Ref(node_id, inp.name))), EntireValue()
    return InputCarries(inp.dtype, None), Broadcast()


def _row_above_reductions(reductions: list[tuple[Index, str]]) -> list[tuple[Index, str]]:
    """The one demand reductions make when nothing else does: the shallowest level right above a series."""
    if not reductions:
        return []
    index, name = min(reductions, key=lambda child_index_and_input: child_index_and_input[0].depth)
    return [(index.parent, name)]  # type: ignore[list-item]


def _deepest_index_on_one_line_of_descent(required_rows: list[tuple[Index, str]]) -> tuple[Index | None, tuple[str, str] | None]:
    """The deepest level among ``rows_required_by_inputs`` if they lie on one line of descent, or the two fields that disagree."""
    if not required_rows:
        return None, None
    deepest, deepest_ref = max(required_rows, key=lambda required_row: required_row[0].depth)
    lineage = set(deepest.lineage)
    for index, ref in required_rows:
        if index not in lineage:
            return None, (deepest_ref, ref)
    return deepest, None


def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
    return make_problem("misaligned", node_id, a=str(a), b=str(b))
