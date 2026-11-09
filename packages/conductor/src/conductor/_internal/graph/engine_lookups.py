"""What the engine looks up on every step of a run: who reads a value when it is written, and what runs on each index.

Compile walks forward: a field knows what feeds it, never who reads it, since
its readers are compiled after it. The engine needs the other direction, so
this builds it once, on the first run, by turning every node's edges around.
A graph that is only compiled, or only embedded in another, never pays for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from conductor._internal.graph.binding import From
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.receive_modes import Merge, ReceiveMode
from conductor._internal.node_signature import Interface
from conductor._internal.ref import Ref
from conductor._internal.series import Index


@dataclass(frozen=True)
class EngineLookups:
    """Every lookup the engine makes, by address or index id. Built once; only the engine reads it."""

    fields: dict[Ref, CompiledField]
    #: Per node: the index it runs once per row of, ``None`` when it runs once.
    runs_per_row_of: dict[str, Index | None]
    #: Per node: its inputs and outputs.
    interfaces: dict[str, Interface]
    #: Per output, every input an edge from it feeds, in execution order, with how it receives the value.
    read_by: dict[Ref, tuple[tuple[Ref, ReceiveMode], ...]]
    #: Per node, every output an edge into it reads, one per edge ref.
    reads: dict[str, tuple[tuple[Ref, ReceiveMode], ...]]
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
    #: The table outputs a merge matches against a table they were built from: a
    #: return where no row kept its id can never match (``row_ids_lost``).
    row_ids_must_survive: frozenset[Ref]


def build_engine_lookups(nodes: Iterable[CompiledNode], row_ids_must_survive: frozenset[Ref]) -> EngineLookups:
    """The plan for a finished compile, over its ready nodes in execution order.
    An index with entries and no owner is a bug in compile, and raises."""
    fields: dict[Ref, CompiledField] = {}
    runs_per_row_of: dict[str, Index | None] = {}
    interfaces: dict[str, Interface] = {}
    reads: dict[str, list[tuple[Ref, ReceiveMode]]] = {}
    read_by: dict[Ref, list[tuple[Ref, ReceiveMode]]] = {}
    nodes_running_per_row_of_index: dict[str, list[str]] = {}
    outputs_on_index: dict[str, list[Ref]] = {}
    lists_on_nodes_under_index: dict[str, list[Ref]] = {}
    field_making_rows_of_index: dict[str, CompiledField] = {}
    table_column_fields: dict[Ref, tuple[Ref, str]] = {}

    def register_index_owner(index_id: str, field: CompiledField) -> None:
        field_making_rows_of_index[index_id] = field
        nodes_running_per_row_of_index.setdefault(index_id, [])
        outputs_on_index.setdefault(index_id, [])
        lists_on_nodes_under_index.setdefault(index_id, [])

    for node in nodes:
        wiring = node.wiring
        if wiring is None or node.embedded_graph is not None or node.interface is None:
            continue
        node_reads = reads.setdefault(node.id, [])
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
                if isinstance(field.binding, From):
                    receive_mode_of_each_source = dict(receive_mode.receive_mode_by_source) if isinstance(receive_mode, Merge) else {}
                    for source in field.binding.refs:
                        receive_mode = receive_mode_of_each_source.get(source, receive_mode)
                        read_by.setdefault(source, []).append((ref, receive_mode))
                        node_reads.append((source, receive_mode))
                if field._is_list_on_node and field._index is not None:
                    register_index_owner(field._index.id, field)
                    outputs_on_index[field._index.id].append(ref)
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
    return EngineLookups(
        fields, runs_per_row_of, interfaces, _lists_as_tuples(read_by), _lists_as_tuples(reads), _lists_as_tuples(nodes_running_per_row_of_index), _lists_as_tuples(outputs_on_index), _lists_as_tuples(lists_on_nodes_under_index),
        field_making_rows_of_index, table_column_fields, row_ids_must_survive,
    )


def _lists_as_tuples[K, V](by_index_id: dict[K, list[V]]) -> dict[K, tuple[V, ...]]:
    return {key: tuple(values) for key, values in by_index_id.items()}
