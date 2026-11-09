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

from conductor._internal.execution.values_of_run import is_held_at_its_own_row, is_list_on_node
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.receive_modes import RowsUnder, reads_of_input
from conductor._internal.index import Index
from conductor._internal.ref import Ref


@dataclass(frozen=True)
class EngineLookups:
    """Every lookup the engine makes, by address or index id. Built once; only the engine reads it."""

    fields: dict[Ref, CompiledField]
    #: Per node: the index it runs once per row of, ``None`` when it runs once.
    runs_per_row_of: dict[str, Index | None]
    #: Every node that runs as units, by id.
    nodes: dict[str, CompiledNode]
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
    #: Per node, the sources of its inputs of many values that read one index with another such input: columns of one table to keep in step.
    sources_of_series_inputs_on_one_index: dict[str, tuple[Ref, ...]]
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
    #: The lists set on nodes whose rows hang under no other row: written when a run starts.
    lists_on_nodes_at_the_top: tuple[Ref, ...]
    #: The fields a read finds at its own row: the one test on the path of every read.
    fields_held_at_their_own_row: frozenset[Ref]


def build_engine_lookups(nodes: Iterable[CompiledNode]) -> EngineLookups:
    """The plan for a runnable graph, over its nodes in execution order, every one a ready function.
    An index with entries and no owner is a bug in compile, and raises."""
    fields: dict[Ref, CompiledField] = {}
    runs_per_row_of: dict[str, Index | None] = {}
    nodes_by_id = {node.id: node for node in nodes}
    read_by: dict[Ref, list[Ref]] = {}
    read_at_depth: dict[str, list[tuple[Ref, int]]] = {}
    nodes_running_per_row_of_index: dict[str, list[str]] = {}
    outputs_on_index: dict[str, list[Ref]] = {}
    lists_on_nodes_under_index: dict[str, list[Ref]] = {}
    field_making_rows_of_index: dict[str, CompiledField] = {}
    lists_on_nodes_at_the_top: list[Ref] = []

    def register_index_owner(index_id: str, field: CompiledField) -> None:
        field_making_rows_of_index[index_id] = field
        nodes_running_per_row_of_index.setdefault(index_id, [])
        outputs_on_index.setdefault(index_id, [])
        lists_on_nodes_under_index.setdefault(index_id, [])

    for node in nodes_by_id.values():
        node_reads = read_at_depth.setdefault(node.id, [])
        runs_per_row_of[node.id] = node._runs_per_row_of
        if (index := node._runs_per_row_of) is not None:
            nodes_running_per_row_of_index.setdefault(index.id, []).append(node.id)
        for field in node._fields.values():  # type: ignore[union-attr]  # lookups hold only ready nodes
            ref = field.ref
            fields[ref] = field
            read_by.setdefault(ref, [])
            receive_mode = field._receive_mode
            if receive_mode is not None:
                for source, depth in reads_of_input(receive_mode):
                    if source != ref:  # a list set on this input is read off the input itself: there is nothing to wait for
                        read_by.setdefault(source, []).append(ref)
                        node_reads.append((source, depth))
                if is_list_on_node(field):
                    index = field._held_as.rows_made
                    assert index is not None  # a list on a node has its rows
                    register_index_owner(index.id, field)
                    outputs_on_index[index.id].append(ref)
                    if index.parent is None:
                        lists_on_nodes_at_the_top.append(ref)
                    else:
                        lists_on_nodes_under_index.setdefault(index.parent.id, []).append(ref)
                continue
            if (rows_made := field._held_as.rows_made) is not None:
                register_index_owner(rows_made.id, field)
            if (index := field._index) is not None:
                outputs_on_index.setdefault(index.id, []).append(ref)
    for by_index_id in (nodes_running_per_row_of_index, outputs_on_index, lists_on_nodes_under_index):
        for index_id in by_index_id.keys() - field_making_rows_of_index.keys():
            raise RuntimeError(f"compile planned index {index_id!r}, which no field owns")
    index_ids_by_parent: dict[str, list[str]] = {index_id: [] for index_id in field_making_rows_of_index}
    node_making_rows_of_index: dict[str, str | None] = {}
    for index_id, field in field_making_rows_of_index.items():
        index = field._held_as.rows_made
        assert index is not None  # the field owns it
        if index.parent is not None:
            index_ids_by_parent.setdefault(index.parent.id, []).append(index_id)
        maker: CompiledField | None = field
        while maker is not None and is_list_on_node(maker) and maker._index.parent is not None:  # type: ignore[union-attr]
            maker = field_making_rows_of_index.get(maker._index.parent.id)  # type: ignore[union-attr]
        node_making_rows_of_index[index_id] = None if maker is None or is_list_on_node(maker) else maker.ref.node_id
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
            writer = source.node_id if not is_list_on_node(field) else node_making_rows_of_index[field._index.id]  # type: ignore[union-attr]
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
            for out in nodes_by_id[node_id].interface.outputs  # type: ignore[union-attr]  # a node that runs has one
            if (index := fields[Ref(node_id, out.name)]._held_as.rows_made) is not None
            for on_rows in nodes_on_rows_and_lists_under(index.id)
        )
    return EngineLookups(
        fields, runs_per_row_of, nodes_by_id, _lists_as_tuples(read_by), _lists_as_tuples(nodes_running_per_row_of_index), _lists_as_tuples(outputs_on_index), _lists_as_tuples(lists_on_nodes_under_index),
        field_making_rows_of_index,
        {node_id: _sources_of_series_inputs_on_one_index(node._run_plan.inputs.values(), fields) for node_id, node in nodes_by_id.items()},
        _lists_as_tuples(what_each_read_waits_for),
        {node_id: tuple(sorted(depths)) for node_id, depths in depths_counted.items()},
        nodes_finished_when_rows_made_by, node_making_rows_of_index, _lists_as_tuples(index_ids_by_parent), tuple(lists_on_nodes_at_the_top),
        frozenset(ref for ref, field in fields.items() if is_held_at_its_own_row(field)),
    )


def _sources_of_series_inputs_on_one_index(receive_modes: Iterable[Any], fields: dict[Ref, CompiledField]) -> tuple[Ref, ...]:
    """The sources of every input read as rows of an index another such input also reads."""
    inputs_reading: dict[Index | None, list[RowsUnder]] = {}
    for receive_mode in receive_modes:
        if isinstance(receive_mode, RowsUnder):
            inputs_reading.setdefault(fields[receive_mode.sources[0]]._index, []).append(receive_mode)
    return tuple(source for readers in inputs_reading.values() if len(readers) > 1 for reader in readers for source in reader.sources)


def _lists_as_tuples[K, V](by_index_id: dict[K, list[V]]) -> dict[K, tuple[V, ...]]:
    return {key: tuple(values) for key, values in by_index_id.items()}
