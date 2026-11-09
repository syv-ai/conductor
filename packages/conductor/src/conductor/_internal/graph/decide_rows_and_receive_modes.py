"""What arrives on each input of one node, whether each edge may land, how each input receives, and on which rows the node runs.

May this edge land? ``target.accepts(source)``, element against element; ``False``
is the fatal ``type_mismatch``.

Does the node run once, or once per row? A column arriving on an input that
takes one value makes the node run once per row of it, and its outputs become
columns on the same rows; "receives a series" is the whole rule. Several
columns must sit on one line of descent (the deepest wins, a parent value is
copied down), or they are the fatal ``misaligned``.

A ``Series[X]`` input fed one column on a child index reduces: a call receives
all the values under its own row, however deep, flat. It doesn't choose the
row unless nothing else does; then the node runs per row of the index right
above. One column on a root is received whole, once. Anything else feeding it
(several scalars, unrelated columns, a list set on the node, a default) is
collected onto an index of the input's own.

A ``Table`` input merges everything wired in, matched on row ids; it never
makes the node run per row by itself, except a table arriving per row. Tables
that are alternatives (one ``exclusive_group``, or two branches of a decision)
are one place in the merge: the branch that ran is the table, and when they are
all there is, there is nothing to merge. What it carries is the merged table's
type, each column named as the merge will name it, or what the input declares
when it names columns.

A ``**inputs: AsItArrives`` input takes what its edge carries and never makes
the node run per row. A parameter typed ``Any`` takes the type of its first
edge.

Whatever an input receives under a row is cut at that row, as far as the row
and the source share one; sharing none, it is all of the source. A node that
runs once shares nothing, so it receives everything. The sources of a gather
or a merge are each read this way, on their own, once the node's row is known.

An input no edge feeds holds what compile already knows: a static or its
default, as a ``FixedValue``. A table wired alone into a ``Table`` input has
nothing to match with, so it is received as a plain ``Cell``, the same frame.

Inside an embedded graph a node compiles under the row the graph node runs on
(``runs_under``): it runs at least per row of it, and its outputs and rows
hang under it, as they would for the same nodes drawn there. A reduction over
a series made inside is therefore a reduction under that row. Whatever ran
once inside is a column on that row holding one value per run of the graph:
a ``Series[X]`` input over such columns gathers this run's values, and an
empty gather is an empty series.

A list set on an input that takes one value is a column of its own, under
each row the node otherwise runs on, and an edge reads it like an output (``q.question``).
Only one list may be set on a node (``lists_on_a_per_row_node``).

Every answer is written as the input's ``ReceiveMode``, which the engine
reads and never decides again. An index belongs to the field that makes its
rows and is named after it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from conductor._internal.dtype import (
    _ReceivedAsItArrives,
    is_untyped,
    type_name_for_sentence,
)
from conductor._internal.errors import Problem, make_problem
from conductor._internal.execution.values_of_run import (
    holds_part_of_a_table,
    holds_values,
    names_in_a_merge,
)
from conductor._internal.field_records import Input
from conductor._internal.graph.binding import From
from conductor._internal.graph.embedding import parent_graph_of_node_id
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.graph.output_conditions import can_never_both_hold
from conductor._internal.graph.receive_modes import (
    Cell,
    FixedValue,
    Gather,
    Merge,
    ReceiveMode,
    RowsUnder,
    cell_at_the_row,
    rows_under_the_row,
)
from conductor._internal.graph.row_origins import sources_taken_in_whole
from conductor._internal.index import Index
from conductor._internal.ref import NODE_PATH_SEPARATOR, Ref
from conductor._internal.table import (
    Column,
    columns_of,
    is_table_type,
    polars_type_of,
    renames_in_merge,
    table_with_columns,
)

if TYPE_CHECKING:
    from conductor._internal.graph.compile_graph import CompileProgress
    from conductor._internal.graph.compiled_node import CompiledField


class ReceivedAtTheRow(NamedTuple):
    """What one input ends up with on the node's row: its type, the index its rows are on for a series, and how a call receives it."""

    dtype: Any
    index: Index | None
    receive_mode: ReceiveMode


@dataclass(frozen=True, slots=True)
class ReadingOfInput:
    """How one input reads what arrives on it, what a call's row depends on, and how it is received once the row is known.

    One function per kind of input builds one of these, whole: everything about
    how that kind receives is in the one place. ``decide_runs_per_row_of`` reads
    only what the row depends on."""

    input: Input
    received_at_the_row: Callable[[Index | None], ReceivedAtTheRow]
    #: What ``compute_outputs`` is told about this input; ``None`` on one no edge feeds.
    arriving_type: Any = None
    #: What an edge gave an input that had no type of its own: the fields to set on it.
    given_by_edge: Mapping[str, Any] | None = None
    #: The indexes this input makes the node run per row of.
    requires_rows_of: tuple[Index, ...] = ()
    #: The child index a ``Series[X]`` input reduces, placed once the node's row is known.
    reduces: Index | None = None
    is_list_on_node: bool = False


#: One place in a merge: a source, or tables of which only one is ever at a row.
SourcesThatAreAlternatives = list[tuple[Ref, "CompiledField"]]


class WhatArrivesOnInputs(NamedTuple):
    """The reading of each input, as far as they could be read."""

    readings: tuple[ReadingOfInput, ...]
    #: Every table and merged column taken in whole: a table the node returns may carry their ids.
    tables_taken_in: tuple[CompiledField, ...]
    #: A source couldn't be read; the walk stopped at that input.
    stopped_at_unreadable_source: bool = False


def read_what_arrives_on_inputs(
    node: GraphNode, inputs: tuple[Input, ...], statics: Mapping[str, Any], inputs_given_a_list: frozenset[str], progress: CompileProgress, problems: list[Problem],
) -> WhatArrivesOnInputs:
    """The reading of each of ``inputs``, stopping at the first one that can't be read."""
    readings: list[ReadingOfInput] = []
    tables_taken_in: tuple[CompiledField, ...] = ()
    for input_ in inputs:
        name = input_.name
        binding = node.bindings.get(name)
        if not isinstance(binding, From):
            readings.append(_reading_of_a_list_on_the_node(input_, node.id) if name in inputs_given_a_list else _reading_of_a_value_set_on_it(input_, node.id, statics))
            continue
        feeding = [_source_output(source, node.id, progress) for source in binding.refs]
        if None in feeding:
            # A source that was worked out but has no such output is this input's
            # problem; one that wasn't carries its own problem already.
            problems.extend(
                make_problem("unknown_ref_output", node.id, name, source=str(ref_), outputs=_outputs_offered_by(ref_.node_id, progress))
                for ref_, source in zip(binding.refs, feeding, strict=True)
                if source is None and _source_node_is_wired(ref_, progress)
            )
            return WhatArrivesOnInputs(tuple(readings), tables_taken_in, True)
        sources: list[CompiledField] = cast("list[CompiledField]", feeding)
        tables_taken_in += tuple(sources_taken_in_whole(input_.dtype, sources))
        reading = _reading_of_edges(node.id, input_, binding.refs, sources, progress, problems)
        if reading is None:
            return WhatArrivesOnInputs(tuple(readings), tables_taken_in, True)
        readings.append(reading)
    return WhatArrivesOnInputs(tuple(readings), tables_taken_in)


def _reading_of_edges(
    node_id: str, input_: Input, refs: tuple[Ref, ...], sources: list[CompiledField], progress: CompileProgress, problems: list[Problem],
) -> ReadingOfInput | None:
    """The reading of an input with edges into it, by the kind of input it is; ``None`` when the input can't be read."""
    target = input_.dtype
    if target is _ReceivedAsItArrives:
        return _reading_of_as_it_arrives(node_id, input_, refs, sources, progress, problems)
    if is_table_type(target) and not input_.many:
        return _reading_of_a_table(node_id, input_, refs, sources, progress, problems)
    given_by_edge = None
    if is_untyped(target):
        target = sources[0].type
        given_by_edge = {"dtype": target}
    for source_ref, source in zip(refs, sources, strict=True):
        problems.extend(_type_mismatch_unless_accepted(target, node_id, input_.name, source_ref, source, target_many=input_.many))
    if not input_.many:
        if len(sources) > 1 and not _all_on_one_index(sources):
            problems.append(make_problem("union_needs_one_index", node_id, input_.name))
            return None
        return _reading_of_one_value(input_, refs, sources[0], given_by_edge)
    if _all_on_one_index(sources) and not _sits_on_the_row_of_a_graph_around_both(node_id, sources[0], progress):
        return _reading_of_a_series_on_one_index(input_, refs, sources[0], given_by_edge)
    return _reading_of_a_collected_series(node_id, input_, refs, sources, target, given_by_edge)


def _reading_of_a_value_set_on_it(input_: Input, node_id: str, statics: Mapping[str, Any]) -> ReadingOfInput:
    """An input no edge feeds holds one value copied to every call, or for a ``Series[X]`` input a column of its own, received whole."""
    value = statics.get(input_.name, input_.default)
    received = (
        ReceivedAtTheRow(input_.dtype, Index(Ref(node_id, input_.name)), FixedValue("entire_value", None if value is None else list(value)))
        if input_.many else ReceivedAtTheRow(input_.dtype, None, FixedValue("same_value_every_row", value))
    )
    return ReadingOfInput(input_, lambda _: received)


def _reading_of_a_list_on_the_node(input_: Input, node_id: str) -> ReadingOfInput:
    """A list set on an input that takes one value: a column of its own, on the row it makes."""
    own_ref = Ref(node_id, input_.name)
    return ReadingOfInput(input_, lambda row: ReceivedAtTheRow(input_.dtype, row, cell_at_the_row((own_ref,), row)), is_list_on_node=True)


def _reading_of_as_it_arrives(node_id: str, input_: Input, refs: tuple[Ref, ...], sources: list[CompiledField], progress: CompileProgress, problems: list[Problem]) -> ReadingOfInput | None:
    """``**x: AsItArrives`` takes what its one edge carries and never makes the node run per row."""
    name = input_.name
    if not name.isidentifier():
        problems.append(make_problem("parameter_name_invalid", node_id, name))
        return None
    if len(sources) > 1:
        problems.append(make_problem("one_edge_per_parameter", node_id, name))
        return None
    source = sources[0]
    one_value_per_run_of_a_graph = _sits_on_the_row_of_a_graph_around_both(node_id, source, progress)

    def received_at_the_row(row: Index | None) -> ReceivedAtTheRow:
        mode = cell_at_the_row(refs, source._index) if source._index is None or one_value_per_run_of_a_graph else _read_of_source_at_the_row(refs[0], source, row)
        return ReceivedAtTheRow(source.type, source._index, mode)

    return ReadingOfInput(input_, received_at_the_row, arriving_type=source.type, given_by_edge={"dtype": source.type, "many": source.many and not one_value_per_run_of_a_graph})


def _reading_of_a_table(node_id: str, input_: Input, refs: tuple[Ref, ...], sources: list[CompiledField], progress: CompileProgress, problems: list[Problem]) -> ReadingOfInput:
    """A ``Table`` input merges everything wired in; it runs per row only of a table that arrives per row."""
    name, target = input_.name, input_.dtype
    pairs = list(zip(refs, sources, strict=True))
    problems_found = [problem for source_ref, source in pairs for problem in _type_mismatch_unless_table_or_column(target, node_id, name, source_ref, source)]
    problems.extend(problems_found)
    places = _sources_grouped_into_alternatives(pairs)
    merged = None if problems_found else _merged_columns(places)
    carried = _table_a_call_receives(target, merged)
    _check_columns(node_id, name, target, merged, problems)

    def received_at_the_row(row: Index | None) -> ReceivedAtTheRow:
        problems.extend(progress.row_origins.check_merge(node_id, name, places))
        return ReceivedAtTheRow(carried, row, _receive_mode_of_table_input(places, row))

    return ReadingOfInput(input_, received_at_the_row, arriving_type=carried, requires_rows_of=tuple(source._index for source in sources if is_table_type(source.type) and source._index is not None))


def _reading_of_one_value(input_: Input, refs: tuple[Ref, ...], source: CompiledField, given_by_edge: Mapping[str, Any] | None) -> ReadingOfInput:
    """One value at the call's row; a series arriving makes the node run per row of it."""
    received = ReceivedAtTheRow(source.type, source._index, cell_at_the_row(refs, source._index))
    return ReadingOfInput(input_, lambda _: received, arriving_type=source.type, given_by_edge=given_by_edge, requires_rows_of=() if source._index is None else (source._index,))


def _reading_of_a_series_on_one_index(input_: Input, refs: tuple[Ref, ...], source: CompiledField, given_by_edge: Mapping[str, Any] | None) -> ReadingOfInput:
    """A ``Series[X]`` fed one column: a child index reduces under the call's row; on a root it is received whole."""
    index = source._index
    assert index is not None  # the kind is a series on one index
    if index.parent is None:
        received = ReceivedAtTheRow(source.type, index, rows_under_the_row(refs, index, 0))
        return ReadingOfInput(input_, lambda _: received, arriving_type=source.type, given_by_edge=given_by_edge)

    def received_at_the_row(row: Index | None) -> ReceivedAtTheRow:
        return ReceivedAtTheRow(source.type, index, rows_under_the_row(refs, index, _depth_shared_with_row(index.parent, row)))  # type: ignore[arg-type]  # checked once the row is known

    return ReadingOfInput(input_, received_at_the_row, arriving_type=source.type, given_by_edge=given_by_edge, reduces=index)


def _reading_of_a_collected_series(node_id: str, input_: Input, refs: tuple[Ref, ...], sources: list[CompiledField], target: Any, given_by_edge: Mapping[str, Any] | None) -> ReadingOfInput:
    """Anything else into a ``Series[X]`` input: each source read at the call's row on its own, collected onto an index of the input's own."""
    collected_onto = Index(Ref(node_id, input_.name))

    def received_at_the_row(row: Index | None) -> ReceivedAtTheRow:
        return ReceivedAtTheRow(target, collected_onto, Gather(tuple(_read_of_source_at_the_row(ref, source, row) for ref, source in zip(refs, sources, strict=True)), Ref(collected_onto.id)))

    return ReadingOfInput(input_, received_at_the_row, arriving_type=target, given_by_edge=given_by_edge)


def _sits_on_the_row_of_a_graph_around_both(node_id: str, source: CompiledField, progress: CompileProgress) -> bool:
    """Is ``source`` on the very row a graph around both it and this node runs on? Then it holds one value per run of that graph."""
    if NODE_PATH_SEPARATOR not in node_id or source._index is None:
        return False
    graph_id = parent_graph_of_node_id(source.ref.node_id)
    while graph_id is not None:
        if node_id.startswith(f"{graph_id}{NODE_PATH_SEPARATOR}") and progress.compiled[graph_id]._wiring._runs_per_row_of == source._index:  # type: ignore[union-attr]
            return True
        graph_id = parent_graph_of_node_id(graph_id)
    return False


def _source_output(ref: Ref, reader_id: str, progress: CompileProgress) -> CompiledField | None:
    """The output ``ref`` names, on a node above this one; ``None`` when no such output has been worked out.

    An id an author wrote with a ``/`` in it is no source; the same id on a node placed inside a graph is one."""
    source_id = ref.node_id
    node = progress.compiled.get(source_id)
    if node is None or node._wiring is None or (source_id in progress.invalid_node_ids and NODE_PATH_SEPARATOR not in reader_id):
        return None
    found = node._wiring.fields.get(ref.field)
    return found if found is not None and holds_values(found) else None


def _outputs_offered_by(node_id: str, progress: CompileProgress) -> str:
    """The names an edge may read from ``node_id``, for a sentence."""
    node = progress.compiled[node_id]
    if node.embedded_graph is not None:
        names = sorted(node.embedded_graph._embeddable.outputs)
    else:
        names = sorted(name for name, field in node._wiring.fields.items() if holds_values(field))  # type: ignore[union-attr]
    return ", ".join(names) or "none"


def _source_node_is_wired(ref: Ref, progress: CompileProgress) -> bool:
    node = progress.compiled.get(ref.node_id)
    return node is not None and node._wiring is not None


def decide_runs_per_row_of(node_id: str, readings: tuple[ReadingOfInput, ...], problems: list[Problem], runs_under: Index | None) -> tuple[bool, Index | None]:
    """The index the node runs once per row of. ``runs_under`` is the row of the graph it sits inside: the node runs at least
    per row of it. ``False`` when its inputs disagree, with the problem in ``problems``."""
    field_address_of = lambda name: Ref(node_id, name)  # noqa: E731
    rows_required = tuple((index, reading.input.name) for reading in readings for index in reading.requires_rows_of)
    reductions = [reading for reading in readings if reading.reduces is not None]
    runs_per_row_of, disagreeing = _deepest_index_on_one_line_of_descent(rows_required or _row_above_reductions(reductions))
    if disagreeing is not None:
        problems.append(_misaligned(node_id, field_address_of(disagreeing[0]), field_address_of(disagreeing[1])))
        return False, None
    if not rows_required and any(runs_per_row_of not in reading.reduces.lineage for reading in reductions):  # type: ignore[union-attr]  # kept above
        problems.append(_misaligned(node_id, field_address_of(reductions[0].input.name), field_address_of(reductions[1].input.name)))
        return False, None
    lists = [reading for reading in readings if reading.is_list_on_node]
    if len(lists) > 1:
        problems.append(make_problem("lists_on_a_per_row_node", node_id, lists[1].input.name, a=str(field_address_of(lists[0].input.name)), b=str(field_address_of(lists[1].input.name))))
        return False, None
    if runs_under is not None and (runs_per_row_of is None or runs_under.depth > runs_per_row_of.depth):
        runs_per_row_of = runs_under
    if lists:
        runs_per_row_of = Index(str(field_address_of(lists[0].input.name)), parent=runs_per_row_of)
    for reading in reductions:
        if _depth_shared_with_row(reading.reduces.parent, runs_per_row_of) is None:  # type: ignore[union-attr, arg-type]  # a reduction reads a child index
            problems.append(_misaligned(node_id, field_address_of(rows_required[0][1]), field_address_of(reading.input.name)))
            return False, None
    return True, runs_per_row_of


def series_inputs_that_never_share_a_row(node_id: str, inputs: Sequence[Input], receive_modes: Mapping[str, ReceiveMode], progress: CompileProgress) -> list[Problem]:
    """Two inputs of many values, neither taking a blank, reading opposite branches on one index:
    a row either has skipped is a blank the other way, so every row would leave them both and the node would get nothing."""
    taking_no_blank = [input_ for input_ in inputs if input_.many and not input_.takes_blank]
    if len(taking_no_blank) < 2:
        return []
    readers = [(input_.name, [field for ref in mode.sources if (field := _source_output(ref, node_id, progress)) is not None]) for input_ in taking_no_blank if isinstance(mode := receive_modes[input_.name], RowsUnder)]
    for position, (a, sources_of_a) in enumerate(readers):
        for b, sources_of_b in readers[position + 1:]:
            if all(_never_hold_a_value_on_one_row(left, right) for left in sources_of_a for right in sources_of_b):
                return [make_problem("series_inputs_never_share_a_row", node_id, a, a=a, b=b)]
    return []


def _receive_mode_of_table_input(places: list[SourcesThatAreAlternatives], runs_per_row_of: Index | None) -> Cell | Merge:
    """How a ``Table`` input receives what is wired into it: one table on its own, or the one of several alternatives that ran, has nothing to match with."""
    if len(places) == 1 and is_table_type(places[0][0][1].type):
        return cell_at_the_row(tuple(ref for ref, _ in places[0]), places[0][0][1]._index)
    return Merge(tuple(_read_of_merge_place(place, runs_per_row_of) for place in places), tuple(_copied_down([pair for place in places for pair in place])), tuple(tuple(sorted(carried)) for carried in _columns_carried_on(places)))


def _copied_down(sources: list[tuple[Ref, CompiledField]]) -> list[Ref]:
    """The sources whose rows sit above another source's: their values are copied down onto those rows."""
    rows = [(ref, source._held_as.rows_made or source._index) for ref, source in sources]
    return [ref for ref, index in rows if index is not None and any(index != other and index in other.lineage for _, other in rows if other is not None)]


def _sources_grouped_into_alternatives(pairs: list[tuple[Ref, CompiledField]]) -> list[SourcesThatAreAlternatives]:
    """The sources of a ``Table`` input, tables that are never at one row together in one place: the branch that ran is the table, with no merge."""
    places: list[SourcesThatAreAlternatives] = []
    for pair in pairs:
        place = next((place for place in places if all(_are_alternatives(pair[1], source) for _, source in place)), None)
        if place is None:
            places.append([pair])
        else:
            place.append(pair)
    return places


def _are_alternatives(left: CompiledField, right: CompiledField) -> bool:
    """Two tables on one row where only one ever has a value: two outputs of one ``exclusive_group``, or what hangs off two branches of one decision."""
    return is_table_type(left.type) and is_table_type(right.type) and _never_hold_a_value_on_one_row(left, right)


def _never_hold_a_value_on_one_row(left: CompiledField, right: CompiledField) -> bool:
    """Two fields on one index of which at most one ever has a value at a row: two outputs of one ``exclusive_group``, or what hangs off two branches of one decision."""
    if left._index != right._index:
        return False
    return can_never_both_hold(left._condition_with_rows, right._condition_with_rows)  # type: ignore[arg-type]  # a field without a condition holds nothing back


def _columns_carried_on(places: list[SourcesThatAreAlternatives]) -> list[frozenset[int]]:
    """Per place of a merge, the other places whose table its own table was made from: their columns of the same name are one column."""
    made_from: list[frozenset[int]] = []
    for position, place in enumerate(places):
        origins = frozenset[str]().union(*(source._row_origins or frozenset() for _, source in place))
        made_from.append(frozenset(
            other for other, other_place in enumerate(places)
            if other != position and all(holds_part_of_a_table(source) for _, source in (*place, *other_place))
            and any((source._held_as.rows_made or source._index).id in origins for _, source in other_place)  # type: ignore[union-attr]  # what is part of a table has rows
        ))
    return made_from


def _merged_columns(places: list[SourcesThatAreAlternatives]) -> tuple[Column, ...] | None:
    """The columns the merged table of ``places`` has, named as the merge names them; ``None`` when a table among them names none."""
    columns_each: list[tuple[str, tuple[Column, ...]]] = []
    for place in places:
        first = place[0][1]
        column_name, output_name = names_in_a_merge(first)
        if is_table_type(first.type):
            columns = _columns_of_alternatives(place)
            if columns is None:
                return None
        else:
            columns = ((column_name, first.type),)
        columns_each.append((output_name, columns))
    carries = _columns_carried_on(places)
    renames = renames_in_merge((output_name, (name for name, _ in columns), carried) for (output_name, columns), carried in zip(columns_each, carries, strict=True))
    merged: dict[str, tuple[Any, int]] = {}
    for position, ((_, columns), renamed) in enumerate(zip(columns_each, renames, strict=True)):
        for name, dtype in columns:
            final = renamed.get(name, name)
            if final not in merged or len(carries[position]) > len(carries[merged[final][1]]):
                merged[final] = (dtype, position)
    return tuple((name, dtype) for name, (dtype, _) in merged.items())


def _columns_of_alternatives(place: SourcesThatAreAlternatives) -> tuple[Column, ...] | None:
    """The columns every table of ``place`` holds alike, what a call can count on whichever ran; ``None`` when none is known to."""
    each = [columns_of(source.type) for _, source in place]
    return tuple(column for column in each[0] if all(column in other for other in each[1:])) or None


def _table_a_call_receives(target: Any, merged: tuple[Column, ...] | None) -> Any:
    """The table type a ``Table`` input carries, what a hook's ``ArrivingType`` is told: the merged table's, or what the input declares when it names columns, since those are what a call gets."""
    if columns_of(target) or merged is None:
        return target
    return table_with_columns(dict(merged), base=target)


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
    if is_table_type(source.type) or (source.many and polars_type_of(source.type) is not None):
        return []
    return _type_mismatch_unless_accepted(target, node_id, name, source_ref, source) or [_type_mismatch(target, node_id, name, source_ref, source)]


def _type_mismatch_unless_accepted(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField, *, target_many: bool = False) -> list[Problem]:
    """May what arrives land? The target's ``accepts`` decides, asked about one value of each."""
    if target.accepts(source.type):
        return []
    return [_type_mismatch(target, node_id, name, source_ref, source, target_many=target_many)]


def _type_mismatch(target: Any, node_id: str, name: str, source_ref: Ref, source: CompiledField, *, target_many: bool = False) -> Problem:
    return make_problem(
        "type_mismatch", node_id, name, source=str(source_ref), source_type=source.type.describe(), source_many=source.many,
        target_type=target.describe(), target_many=target_many, source_said=type_name_for_sentence(source.type, source.many), target_said=type_name_for_sentence(target, target_many),
    )


def _depth_shared_with_row(index: Index, runs_per_row_of: Index | None) -> int | None:
    """The depth of the deepest of ``index`` and the indexes above it that the node's row sits on, ``None`` when none is."""
    lineage = () if runs_per_row_of is None else runs_per_row_of.lineage
    return next((step.depth for step in reversed(index.lineage) if step in lineage), None)


def _read_of_source_at_the_row(ref: Ref, source: CompiledField, runs_per_row_of: Index | None) -> Cell | RowsUnder:
    """How a call on ``runs_per_row_of`` reads one source: the one value a source
    with no rows holds; else its rows, cut at the deepest row the two share
    (the source's own row when the call sits on or under it), all of them when they share none."""
    index = source._index
    if index is None:
        return cell_at_the_row((ref,), None)
    return rows_under_the_row((ref,), index, _depth_shared_with_row(index, runs_per_row_of) or 0)


def _read_of_merge_place(place: SourcesThatAreAlternatives, runs_per_row_of: Index | None) -> Cell | RowsUnder:
    """How a call reads one place of a ``Table`` input: a table is the one at its row, or the one there is, of the alternatives that ran; a column is read as any source."""
    ref, source = place[0]
    if is_table_type(source.type):
        assert source._index is None or (runs_per_row_of is not None and source._index in runs_per_row_of.lineage)  # a table arriving per row put the node on its row, or was misaligned
        return cell_at_the_row(tuple(ref for ref, _ in place), source._index)
    return _read_of_source_at_the_row(ref, source, runs_per_row_of)


def _all_on_one_index(sources: Sequence[CompiledField]) -> bool:
    """Do all sources carry a series on one and the same index? Then the edges together are one set of rows."""
    return all(source._index is not None for source in sources) and len({source._index for source in sources}) == 1


def _row_above_reductions(reductions: list[ReadingOfInput]) -> tuple[tuple[Index, str], ...]:
    """The one demand reductions make when nothing else does: the shallowest index right above a series."""
    if not reductions:
        return ()
    reading = min(reductions, key=lambda reading: reading.reduces.depth)  # type: ignore[union-attr]  # kept above
    return ((reading.reduces.parent, reading.input.name),)  # type: ignore[union-attr, return-value]


def _deepest_index_on_one_line_of_descent(required_rows: tuple[tuple[Index, str], ...]) -> tuple[Index | None, tuple[str, str] | None]:
    """The deepest index among ``rows_required_by_inputs`` if they lie on one line of descent, or the two fields that disagree."""
    if len(required_rows) <= 1:
        return (required_rows[0][0] if required_rows else None), None
    deepest, deepest_ref = max(required_rows, key=lambda required_row: required_row[0].depth)
    lineage = set(deepest.lineage)
    for index, ref in required_rows:
        if index not in lineage:
            return None, (deepest_ref, ref)
    return deepest, None


def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
    return make_problem("misaligned", node_id, a=str(a), b=str(b))
