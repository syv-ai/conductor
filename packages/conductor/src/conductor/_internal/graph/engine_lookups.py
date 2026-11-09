"""What the engine looks up on every step of a run: who reads a value when it is written, and what runs on each index.

Compile walks forward: a field knows what feeds it, never who reads it, since
its readers are compiled after it. The engine needs the other direction, so
this builds it once, on the first run, by turning every node's edges around.
A graph that is only compiled, or only embedded in another, never pays for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.receive_modes import ReceiveMode, reads_of_input
from conductor._internal.node_signature import Interface
from conductor._internal.ref import Ref
from conductor._internal.series import Index, Series


@dataclass(frozen=True)
class EngineLookups:
    """Every lookup the engine makes, by address or index id. Built once; only the engine reads it."""

    fields: dict[Ref, CompiledField]
    #: Per node: the index it runs once per row of, ``None`` when it runs once.
    runs_per_row_of: dict[str, Index | None]
    #: Per node: its inputs and outputs.
    interfaces: dict[str, Interface]
    #: Per node, how each of its inputs receives its value, by input name.
    receive_modes_of_node: dict[str, tuple[tuple[str, ReceiveMode], ...]]
    #: Per output, every input an edge from it feeds, in execution order.
    read_by: dict[Ref, tuple[Ref, ...]]
    #: Per index id: the nodes that run once per row of it.
    nodes_running_per_row_of_index: dict[str, tuple[str, ...]]
    #: Per index id: the outputs whose values sit on it, and a list set on a node whose own rows they are.
    outputs_on_index: dict[str, tuple[Ref, ...]]
    #: Per index id: the lists set on nodes whose rows are made under each of its rows.
    lists_on_nodes_under_index: dict[str, tuple[Ref, ...]]
    #: The field that makes each index's rows, by index id.
    field_making_rows_of_index: dict[str, CompiledField]
    #: Every column of a table output offered as a field: the table output and the column.
    table_column_fields: dict[Ref, tuple[Ref, str]]
    #: Per index whose rows are those of a table: the table output, whose stored table holds their ids.
    table_holding_row_ids_of_index: dict[str, Ref]
    #: The table outputs a merge matches against a table they were built from: a
    #: return where no row kept its id can never match (``row_ids_lost``).
    row_ids_must_survive: frozenset[Ref]
    #: Per node, one ``(writer node, depth)`` per edge it reads, in reading order: the
    #: writer's units at or under the reading unit's row cut to that depth are
    #: what the read waits for. An edge from a list set on a node, whose values are
    #: known from the start, waits on the node that makes its rows' parents, or on nothing.
    what_each_read_waits_for: dict[str, tuple[tuple[str, int], ...]]
    #: Per node that runs per row: the depths at which its finished units are
    #: counted, shorter than its rows; ``0`` always, others when something reads at that depth.
    depths_finished_units_are_counted_at: dict[str, tuple[int, ...]]
    #: Per node, the nodes that run on rows it makes, directly or through a list on
    #: a node under them: they may have finished the moment those rows are made, with none.
    nodes_finished_when_rows_made_by: dict[str, tuple[str, ...]]
    #: Per index id, the node whose units make its rows. ``None``: every row is known from the graph.
    node_making_rows_of_index: dict[str, str | None]
    #: Per index id: the indexes hanging under it.
    indexes_under_index: dict[str, tuple[str, ...]]
    #: Per field whose cell doesn't sit at its own row: where it is. ``None`` for a list
    #: of cells at the row above, a column name for a column of the table at the row above.
    cell_is_held_in: dict[Ref, str | None]
    #: Per list set on a node: its values.
    cells_of_list_on_node: dict[Ref, list[Any]]
    #: Per field, the depth of its own rows: ``0`` for a field with no index.
    depth_of_field: dict[Ref, int]


def build_engine_lookups(nodes: Iterable[CompiledNode], row_ids_must_survive: frozenset[Ref]) -> EngineLookups:
    """The plan for a finished compile, over its ready nodes in execution order.
    An index with entries and no owner is a bug in compile, and raises."""
    fields: dict[Ref, CompiledField] = {}
    runs_per_row_of: dict[str, Index | None] = {}
    interfaces: dict[str, Interface] = {}
    receive_modes_of_node: dict[str, list[tuple[str, ReceiveMode]]] = {}
    read_by: dict[Ref, list[Ref]] = {}
    read_at_depth: dict[str, list[tuple[Ref, int]]] = {}
    nodes_running_per_row_of_index: dict[str, list[str]] = {}
    outputs_on_index: dict[str, list[Ref]] = {}
    lists_on_nodes_under_index: dict[str, list[Ref]] = {}
    field_making_rows_of_index: dict[str, CompiledField] = {}
    table_column_fields: dict[Ref, tuple[Ref, str]] = {}
    cells_of_list_on_node: dict[Ref, list[Any]] = {}

    def register_index_owner(index_id: str, field: CompiledField) -> None:
        field_making_rows_of_index[index_id] = field
        nodes_running_per_row_of_index.setdefault(index_id, [])
        outputs_on_index.setdefault(index_id, [])
        lists_on_nodes_under_index.setdefault(index_id, [])

    for node in nodes:
        wiring = node.wiring
        if wiring is None or node.embedded_graph is not None or node.interface is None:
            continue
        node_receive_modes = receive_modes_of_node.setdefault(node.id, [])
        node_reads = read_at_depth.setdefault(node.id, [])
        runs_per_row_of[node.id] = wiring._runs_per_row_of
        interfaces[node.id] = node.interface
        if (index := wiring._runs_per_row_of) is not None:
            nodes_running_per_row_of_index.setdefault(index.id, []).append(node.id)
        for field in wiring.fields.values():
            ref = field.ref
            fields[ref] = field
            read_by.setdefault(ref, [])
            receive_mode = field._receive_mode
            if receive_mode is not None:
                node_receive_modes.append((ref.field, receive_mode))
                for source, depth in reads_of_input(receive_mode):
                    if source != ref:  # a list set on this input is read off the input itself: there is nothing to wait for
                        read_by.setdefault(source, []).append(ref)
                        node_reads.append((source, depth))
                if field._is_list_on_node and field._index is not None:
                    register_index_owner(field._index.id, field)
                    outputs_on_index[field._index.id].append(ref)
                    values = node.statics[ref.field]
                    cells_of_list_on_node[ref] = list(values.values if isinstance(values, Series) else values)
                    if field._index.parent is not None:
                        lists_on_nodes_under_index.setdefault(field._index.parent.id, []).append(ref)
                continue
            if field._column_of_table is not None:
                table_column_fields[ref] = field._column_of_table
            if field._makes_rows is not None:
                register_index_owner(field._makes_rows.id, field)
            if (index := field._index) is not None:
                outputs_on_index.setdefault(index.id, []).append(ref)
    for by_index_id in (nodes_running_per_row_of_index, outputs_on_index, lists_on_nodes_under_index):
        for index_id in by_index_id.keys() - field_making_rows_of_index.keys():
            raise RuntimeError(f"compile planned index {index_id!r}, which no field owns")
    index_ids_by_parent: dict[str, list[str]] = {index_id: [] for index_id in field_making_rows_of_index}
    node_making_rows_of_index: dict[str, str | None] = {}
    depth_of_field: dict[Ref, int] = {}
    for index_id, field in field_making_rows_of_index.items():
        index = field._makes_rows or field._index
        assert index is not None  # the field owns it
        if index.parent is not None:
            index_ids_by_parent.setdefault(index.parent.id, []).append(index_id)
        maker: CompiledField | None = field
        while maker is not None and maker._is_list_on_node and maker._index.parent is not None:  # type: ignore[union-attr]
            maker = field_making_rows_of_index.get(maker._index.parent.id)  # type: ignore[union-attr]
        node_making_rows_of_index[index_id] = None if maker is None or maker._is_list_on_node else maker.ref.node_id
    for ref, field in fields.items():
        depth_of_field[ref] = 0 if field._index is None else field._index.depth
    what_each_read_waits_for: dict[str, list[tuple[str, int]]] = {}
    depths_counted: dict[str, set[int]] = {node_id: {0} for node_id, index in runs_per_row_of.items() if index is not None}

    def count_finished_units_at(node_id: str, depth: int) -> None:
        """The depths a node's finished units are counted at, and those of the node making its rows, which decide they are all made."""
        index = runs_per_row_of[node_id]
        if index is None or depth >= index.depth or depth in depths_counted[node_id]:
            return
        depths_counted[node_id].add(depth)
        maker_node = node_making_rows_of_index[index.id]
        if maker_node is not None:
            count_finished_units_at(maker_node, depth)

    for node_id, node_reads_of_it in read_at_depth.items():
        waits: list[tuple[str, int]] = []
        for source, depth_read in node_reads_of_it:
            field = fields[source]
            writer = source.node_id if not field._is_list_on_node else node_making_rows_of_index[field._index.id]  # type: ignore[union-attr]
            if writer is None:
                continue
            waits.append((writer, depth_read))
            count_finished_units_at(writer, depth_read)
        what_each_read_waits_for[node_id] = waits
    nodes_finished_when_rows_made_by: dict[str, tuple[str, ...]] = {}
    index_ids_of_lists_under = {
        index_id: [fields[list_ref]._index.id for list_ref in lists]  # type: ignore[union-attr]
        for index_id, lists in lists_on_nodes_under_index.items()
    }

    def nodes_on_rows_and_lists_under(index_id: str) -> list[str]:
        return [*nodes_running_per_row_of_index[index_id], *(node_id for list_index in index_ids_of_lists_under[index_id] for node_id in nodes_on_rows_and_lists_under(list_index))]

    for node_id in runs_per_row_of:
        nodes_finished_when_rows_made_by[node_id] = tuple(
            on_rows
            for out in interfaces[node_id].outputs
            if (index := fields[Ref(node_id, out.name)]._makes_rows) is not None
            for on_rows in nodes_on_rows_and_lists_under(index.id)
        )
    table_holding_row_ids_of_index = {
        index_id: maker.ref for index_id, maker in field_making_rows_of_index.items() if maker._makes_rows is not None and not maker._unit_writes_many_cells
    }
    cell_is_held_in: dict[Ref, str | None] = {
        ref: None if field._column_of_table is None else field._column_of_table[1]
        for ref, field in fields.items()
        if field._column_of_table is not None or field._is_list_on_node or field._unit_writes_many_cells
    }
    return EngineLookups(
        fields, runs_per_row_of, interfaces, _lists_as_tuples(receive_modes_of_node), _lists_as_tuples(read_by), _lists_as_tuples(nodes_running_per_row_of_index), _lists_as_tuples(outputs_on_index), _lists_as_tuples(lists_on_nodes_under_index),
        field_making_rows_of_index, table_column_fields, table_holding_row_ids_of_index, row_ids_must_survive,
        _lists_as_tuples(what_each_read_waits_for),
        {node_id: tuple(sorted(depths)) for node_id, depths in depths_counted.items()},
        nodes_finished_when_rows_made_by, node_making_rows_of_index, _lists_as_tuples(index_ids_by_parent), cell_is_held_in, cells_of_list_on_node, depth_of_field,
    )


def _lists_as_tuples[K, V](by_index_id: dict[K, list[V]]) -> dict[K, tuple[V, ...]]:
    return {key: tuple(values) for key, values in by_index_id.items()}
