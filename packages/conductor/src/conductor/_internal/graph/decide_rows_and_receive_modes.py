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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from conductor._internal.graph.output_conditions import (
    can_never_both_hold,
    no_decision_goes_two_ways,
)
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


class InputCarries(NamedTuple):
    """What one input carries: its type and, for a series, the index its rows are on."""

    dtype: Any
    index: Index | None


class SourcesReadAtTheRow(NamedTuple):
    """An input whose receive mode waits for the node's row, because each of its sources is read at that row."""

    input: Input
    sources: tuple[tuple[Ref, CompiledField], ...]
    #: The input's own index when it gathers; ``None`` when it reads one column's rows.
    collected_onto: Index | None


#: One place in a merge: a source, or tables of which only one is ever at a row.
SourcesThatAreAlternatives = list[tuple[Ref, "CompiledField"]]


@dataclass
class WhatArrivesOnInputs:
    """What one node's inputs receive, gathered as its edges are read."""

    #: Per input name: what it carries, and how a call receives it.
    carries: dict[str, InputCarries] = field(default_factory=dict)
    receive_modes: dict[str, ReceiveMode] = field(default_factory=dict)
    #: Per connected input: the type one call sees, what ``compute_outputs`` is told.
    arriving_types: dict[str, Any] = field(default_factory=dict)
    #: What edges gave to inputs that had no type of their own: the fields to set on each.
    given_by_edge: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The rows the node must run per row of, each with the input that demands it.
    rows_required_by_inputs: tuple[tuple[Index, str], ...] = ()
    #: The child indexes ``Series[X]`` inputs read, placed once the node's row is known.
    child_indexes_read_by_series_inputs: tuple[tuple[Index, Input, tuple[Ref, ...]], ...] = ()
    #: The inputs that don't choose the node's row and are cut by it, placed once it is known.
    sources_read_at_the_row: tuple[SourcesReadAtTheRow, ...] = ()
    #: The inputs holding a list set on the node.
    lists_on_node: tuple[Input, ...] = ()
    #: The ``Table`` inputs with the table type each carries and its sources, one group of alternatives per place in the merge.
    table_inputs_and_sources: tuple[tuple[Input, Any, list[SourcesThatAreAlternatives]], ...] = ()
    #: Every table and merged column taken in whole: a table the node returns may carry their ids.
    tables_taken_in: tuple[CompiledField, ...] = ()
    #: A source couldn't be read; the walk stopped at that input.
    stopped_at_unreadable_source: bool = False


def read_what_arrives_on_inputs(
    node: GraphNode, inputs: tuple[Input, ...], statics: Mapping[str, Any], inputs_given_a_list: frozenset[str], progress: CompileProgress, problems: list[Problem],
) -> WhatArrivesOnInputs:
    """What arrives on each of ``inputs``. Stops at the first input that can't be read and says so in ``stopped_at_unreadable_source``."""
    found = WhatArrivesOnInputs()
    for input_ in inputs:
        name = input_.name
        binding = node.bindings.get(name)
        if not isinstance(binding, From):
            if name in inputs_given_a_list:
                found.lists_on_node += (input_,)
                continue
            found.carries[name], found.receive_modes[name] = _what_an_unwired_input_carries(input_, node.id, statics)
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
            found.stopped_at_unreadable_source = True
            return found
        sources: list[CompiledField] = cast("list[CompiledField]", feeding)
        target = input_.dtype
        taken_in_whole = sources_taken_in_whole(target, sources)
        if taken_in_whole:
            found.tables_taken_in += tuple(taken_in_whole)
        if target is _ReceivedAsItArrives:
            if not name.isidentifier():
                problems.append(make_problem("parameter_name_invalid", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            if len(sources) > 1:
                problems.append(make_problem("one_edge_per_parameter", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            source = sources[0]
            one_value_per_run_of_a_graph = _sits_on_the_row_of_a_graph_around_both(node.id, source, progress)
            found.given_by_edge[name] = {"dtype": source.type, "many": source.many and not one_value_per_run_of_a_graph}
            found.arriving_types[name] = source.type
            found.carries[name] = InputCarries(source.type, source._index)
            if source._index is None or one_value_per_run_of_a_graph:
                found.receive_modes[name] = cell_at_the_row(binding.refs, source._index)
            else:
                found.sources_read_at_the_row += (SourcesReadAtTheRow(input_, ((binding.refs[0], source),), None),)
            continue
        if is_table_type(target) and not input_.many:
            pairs = list(zip(binding.refs, sources, strict=True))
            problems_found = [problem for source_ref, source in pairs for problem in _type_mismatch_unless_table_or_column(target, node.id, name, source_ref, source)]
            problems.extend(problems_found)
            places = _sources_grouped_into_alternatives(pairs)
            merged = None if problems_found else _merged_columns(places)
            found.rows_required_by_inputs += tuple((source._index, name) for source in sources if is_table_type(source.type) and source._index is not None)
            carried = _table_a_call_receives(target, merged)
            _check_columns(node.id, name, target, merged, problems)
            found.table_inputs_and_sources += ((input_, carried, places),)
            found.arriving_types[name] = carried
            continue
        if is_untyped(target):
            target = sources[0].type
            found.given_by_edge[name] = {"dtype": target}
        for source_ref, source in zip(binding.refs, sources, strict=True):
            problems.extend(_type_mismatch_unless_accepted(target, node.id, name, source_ref, source, target_many=input_.many))
        if not input_.many:
            if len(sources) > 1 and not _all_on_one_index(sources):
                problems.append(make_problem("union_needs_one_index", node.id, name))
                found.stopped_at_unreadable_source = True
                return found
            source = sources[0]
            found.arriving_types[name] = source.type
            found.receive_modes[name] = cell_at_the_row(binding.refs, source._index)
            if source._index is not None:
                found.rows_required_by_inputs += ((source._index, name),)
            found.carries[name] = InputCarries(source.type, source._index)
        elif _all_on_one_index(sources) and not _sits_on_the_row_of_a_graph_around_both(node.id, sources[0], progress):
            source = sources[0]
            found.arriving_types[name] = source.type
            if source._index.parent is not None:
                found.child_indexes_read_by_series_inputs += ((source._index, input_, binding.refs),)
            else:
                found.receive_modes[name] = rows_under_the_row(binding.refs, source._index, 0)
            found.carries[name] = InputCarries(source.type, source._index)
        else:
            input_own_index = Index(Ref(node.id, name))
            found.arriving_types[name] = target
            found.sources_read_at_the_row += (SourcesReadAtTheRow(input_, tuple(zip(binding.refs, sources, strict=True)), input_own_index),)
            found.carries[name] = InputCarries(target, input_own_index)
    return found


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


def decide_runs_per_row_of_and_receive_modes(node: GraphNode, found: WhatArrivesOnInputs, problems: list[Problem], row_origins: Any, runs_under: Index | None) -> tuple[bool, Index | None]:
    """The index ``node_id`` runs once per row of, with every input's receive mode settled.

    ``runs_under`` is the row of the graph it sits inside: the node runs at least
    per row of it. ``False`` when its inputs disagree, with the problem in ``problems``."""
    node_id = node.id
    field_address_of = lambda name: Ref(node_id, name)  # noqa: E731
    required_rows = found.rows_required_by_inputs or _row_above_reductions(found.child_indexes_read_by_series_inputs)
    runs_per_row_of, disagreeing = _deepest_index_on_one_line_of_descent(required_rows)
    if disagreeing is not None:
        problems.append(_misaligned(node_id, field_address_of(disagreeing[0]), field_address_of(disagreeing[1])))
        return False, None
    if not found.rows_required_by_inputs and any(runs_per_row_of not in index.lineage for index, _, _ in found.child_indexes_read_by_series_inputs):
        (_, first, _), (_, second, _) = found.child_indexes_read_by_series_inputs[:2]
        problems.append(_misaligned(node_id, field_address_of(first.name), field_address_of(second.name)))
        return False, None
    if len(found.lists_on_node) > 1:
        first, second = found.lists_on_node[:2]
        problems.append(make_problem("lists_on_a_per_row_node", node_id, second.name, a=str(field_address_of(first.name)), b=str(field_address_of(second.name))))
        return False, None
    if runs_under is not None and (runs_per_row_of is None or runs_under.depth > runs_per_row_of.depth):
        runs_per_row_of = runs_under
    if found.lists_on_node:
        (input_,) = found.lists_on_node
        own_ref = field_address_of(input_.name)
        input_own_index = Index(str(own_ref), parent=runs_per_row_of)
        found.carries[input_.name] = InputCarries(input_.dtype, input_own_index)
        found.receive_modes[input_.name] = cell_at_the_row((own_ref,), input_own_index)
        runs_per_row_of = input_own_index
    for index, input_, refs in found.child_indexes_read_by_series_inputs:
        depth = _depth_shared_with_row(index.parent, runs_per_row_of)  # type: ignore[arg-type]  # a reduction reads a child index
        if depth is None:
            problems.append(_misaligned(node_id, field_address_of(found.rows_required_by_inputs[0][1]), field_address_of(input_.name)))
            return False, None
        found.receive_modes[input_.name] = rows_under_the_row(refs, index, depth)
    for read in found.sources_read_at_the_row:
        found.receive_modes[read.input.name] = _receive_mode_read_at_the_row(read, runs_per_row_of)
    for input_, carried, sources in found.table_inputs_and_sources:
        found.carries[input_.name] = InputCarries(carried, runs_per_row_of)
        found.receive_modes[input_.name] = _receive_mode_of_table_input(sources, runs_per_row_of)
        problems.extend(row_origins.check_merge(node_id, input_.name, sources))
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


def _receive_mode_read_at_the_row(read: SourcesReadAtTheRow, runs_per_row_of: Index | None) -> Cell | RowsUnder | Gather:
    """How an input that doesn't choose the node's row receives its sources on that row: each source by its own read."""
    source_reads = tuple(_read_of_source_at_the_row(ref, source, runs_per_row_of) for ref, source in read.sources)
    if read.collected_onto is None:
        return source_reads[0]
    return Gather(source_reads, Ref(read.collected_onto.id))


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
    return not no_decision_goes_two_ways(left._branches_on_its_row | right._branches_on_its_row) or can_never_both_hold(left.condition, right.condition)  # type: ignore[arg-type]  # a field without a condition holds nothing back


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


def _what_an_unwired_input_carries(input_: Input, node_id: str, statics: Mapping[str, Any]) -> tuple[InputCarries, ReceiveMode]:
    """What an input carries when no edge feeds it: one value copied to every
    call, or for a ``Series[X]`` input a column of its own, received whole."""
    value = statics.get(input_.name, input_.default)
    if not input_.many:
        return InputCarries(input_.dtype, None), FixedValue("same_value_every_row", value)
    return InputCarries(input_.dtype, Index(Ref(node_id, input_.name))), FixedValue("entire_value", None if value is None else list(value))


def _row_above_reductions(reductions: tuple[tuple[Index, Input, tuple[Ref, ...]], ...]) -> tuple[tuple[Index, str], ...]:
    """The one demand reductions make when nothing else does: the shallowest index right above a series."""
    if not reductions:
        return ()
    index, input_, _ = min(reductions, key=lambda reduction: reduction[0].depth)
    return ((index.parent, input_.name),)  # type: ignore[return-value]


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
