"""A compiled graph embedded as one node: its nodes, as they stand under the row that node runs on.

A host compiles each graph version once and hands the ``CompiledGraph`` to
conductor as the node's version. The outer compile wires the graph node as a
plain node with the graph's interface, which decides the row it runs on, and
this module then puts the graph's nodes in its place, under that row, so the
graph that embeds it reads exactly as if they had been drawn there. It asks
the embedded graph through one small record, ``EmbeddableGraph``.

Embedding, rule by rule:

1. Every index inside is renamed under the graph node (``emb/lines``), and
   its root hangs under the row the graph node runs on.
2. When the node runs per row, a value that ran once inside becomes a series
   on that row: ``Txt`` becomes ``Series[Txt]``, no index becomes the row,
   and an edge-fed ``Broadcast`` becomes ``PerRow``.
3. ``EntireValue`` becomes ``RowsUnder`` at the row's depth; a ``Gather`` whose
   every source carries one value becomes ``RowsUnder`` on the row, since
   per row the sources line up; a ``RowsUnder`` goes as many levels deeper
   as the row is deep.
4. An inner field the outer graph feeds (a crossing) takes what the outer
   wiring decided for the graph node's field of that address.

Refusing a graph that embeds itself is the host's job, as it builds the
versions: a compiled graph can't contain itself, but a host that swaps a
version in place can close a loop through a value typed on an embedded
graph, which compile then follows until Python's recursion limit.

``/`` separates the levels inside a node id, and ``.`` stays the address
separator, so a place in the run is a path and a field: ``approve/check.amount``.
Only embedding writes a ``/``: an id holding one is refused (``invalid_node_id``).

An embedded graph has an interface of its own. Each input and output it offers
goes by its bare name where that is unique on its side, by its address on a
clash (``offered_names``), and from outside those names are all there is: a
binding, a ``From``, a palette entry. A rename inside it breaks no caller.
Inner addresses stay for reading a run (``approve/check``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.graph.binding import Binding, From, Static
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode, Wiring
from conductor._internal.graph.conditions import (
    ALWAYS,
    Condition,
    condition_at_node,
    output_condition,
)
from conductor._internal.graph.receive import (
    Broadcast,
    EntireValue,
    Gather,
    Merge,
    PerRow,
    Receive,
    RowsUnder,
)
from conductor._internal.interface import Interface
from conductor._internal.node_decorator import NodeVersion
from conductor._internal.ref import Ref
from conductor._internal.series import Index, Series
from conductor._internal.table import columns_of, holds_tables

if TYPE_CHECKING:
    from conductor._internal.errors import Problem
    from conductor._internal.graph.compiled import CompiledGraph


SEPARATOR = "/"


def offered_names(addresses: Iterable[Ref]) -> dict[Ref, str]:
    """The name each of these fields goes by: its bare field name where that is unique among them, its address on a clash.

    The one rule for naming what a graph offers, used for wiring an embedded
    graph (``a.text`` is ``text``) and for ``with_inputs``."""
    addresses = list(addresses)
    bare = Counter(ref.field for ref in addresses)
    return {ref: ref.field if bare[ref.field] == 1 else str(ref) for ref in addresses}


def embedded_in(node_id: str) -> str | None:
    """The graph a node inside sits in, read off its id: ``approve/check`` for
    ``approve/check/amount``; ``None`` for an id with no ``/``."""
    outer, _, _ = node_id.rpartition(SEPARATOR)
    return outer or None


@dataclass(frozen=True)
class OfferedInput:
    """What an embedding graph needs to know about one input a graph offers."""

    #: Its address in that graph's own interface: the node its author drew, and the field there.
    drawn: Ref
    #: Where the value lands in the graph that runs, ``check/inner.amount`` through a nested graph.
    runs_at: Ref
    #: Its author typed a value on it, so nothing needs connecting where it's embedded.
    filled_inside: bool
    #: Its author typed a list on it, so the node holding it runs once per item.
    list_typed_inside: bool
    #: A hook of the node it lands on reads its value.
    read_by_a_hook: bool


@dataclass(frozen=True)
class EmbeddableGraph:
    """What a compiled graph tells a graph that embeds it (``CompiledGraph._for_embedding``)."""

    #: The interface the embedding graph sees: the same fields under ``offered_names``.
    interface: Interface
    #: The inputs it offers, by their offered names.
    inputs: Mapping[str, OfferedInput]
    #: The outputs it offers and each column of a table output, by offered name, each with where it is in the run.
    outputs: Mapping[str, Ref]
    #: Every fatal problem but a typed input it offers left empty, which a value from outside fills.
    faults: tuple[Problem, ...]
    #: Its nodes, in the order compile met them: sources before readers.
    nodes: Mapping[str, CompiledNode]
    #: Its tables that must keep their rows' ids.
    row_ids_must_survive: frozenset[Ref]


def read_embeddable(graph: CompiledGraph) -> EmbeddableGraph:
    """The record a graph that embeds ``graph`` asks."""
    nodes = graph._nodes

    def in_the_run(ref: Ref, side: str) -> Ref:
        """Where a field of ``graph``'s interface is in the run: on a node that is itself a graph, the place inside it."""
        embedded = nodes[ref.node_id].embedded_graph
        if embedded is None:
            return ref
        offered = embedded._for_embedding
        inside = offered.inputs[ref.field].runs_at if side == "input" else offered.outputs[ref.field]
        return Ref(f"{ref.node_id}{SEPARATOR}{inside.node_id}", inside.field)

    # One namespace for both sides: a field name is unique within the node this graph becomes.
    names = offered_names(Ref(field.name) for field in (*graph.interface.inputs, *graph.interface.outputs))
    input_names = {Ref(inp.name): names[Ref(inp.name)] for inp in graph.interface.inputs}
    inputs: dict[str, OfferedInput] = {}
    for ref in input_names:
        node = nodes[ref.node_id]
        field = None if node.wiring is None else node.wiring.fields.get(ref.field)
        inputs[input_names[ref]] = OfferedInput(
            drawn=ref, runs_at=in_the_run(ref, "input"),
            filled_inside=isinstance(node.graph_node.bindings.get(ref.field), Static),
            list_typed_inside=field is not None and field._listed,
            read_by_a_hook=isinstance(node.version, NodeVersion) and ref.field in node.version.hook_reads,
        )
    outputs: dict[str, Ref] = {}
    for out in graph.interface.outputs:
        name, at = names[Ref(out.name)], in_the_run(Ref(out.name), "output")
        outputs[name] = at
        outputs.update({f"{name}.{column}": Ref(at.node_id, f"{at.field}.{column}") for column, _ in columns_of(out.dtype)})
    interface = Interface(
        inputs=tuple(inp.model_copy(update={"name": input_names[Ref(inp.name)]}) for inp in graph.interface.inputs),
        outputs=tuple(out.model_copy(update={"name": names[Ref(out.name)]}) for out in graph.interface.outputs),
        needs=graph.interface.needs,
    )
    return EmbeddableGraph(
        interface=interface, inputs=inputs, outputs=outputs, faults=tuple(_faults_of(graph, nodes, inputs)), nodes=nodes,
        row_ids_must_survive=graph._row_ids_must_survive,
    )


def _faults_of(graph: CompiledGraph, nodes: Mapping[str, CompiledNode], offered: Mapping[str, OfferedInput]) -> list[Problem]:
    """A graph can be embedded only when nothing in it is fatal but an input it
    offers left empty, which a value from outside fills. An empty input the
    graph doesn't offer can never be filled, and one with no type of its own
    leaves its node unworked-out, so each stays a fault."""
    offered_addresses = {inp.drawn for inp in offered.values()}

    def fillable(found: Problem) -> bool:
        if found.type != "unbound_required":
            return False
        node = nodes.get(str(found.loc[0]))
        return node is not None and node.wiring is not None and Ref(str(found.loc[0]), str(found.loc[1])) in offered_addresses

    return [found for found in graph.problems if found.fatal and not fillable(found)]


@dataclass(frozen=True)
class Crossing:
    """What the outer wiring decided for one input of the graph node that an
    outer edge feeds: the type, rows and receipt the inner field takes."""

    type: Any
    index: Index | None
    receives: Receive


def lift_embedded_nodes(
    embeddable: EmbeddableGraph,
    node_id: str,
    edges: Mapping[Ref, From],
    values: Mapping[Ref, Any],
    row: Index | None,
    crossings: Mapping[Ref, Crossing],
    condition_above: Callable[[Ref], Condition],
) -> dict[str, CompiledNode]:
    """Every node of ``embeddable`` as it stands embedded at ``node_id``, running once per row of ``row``.

    Each of ``edges``, ``crossings`` and ``values`` is keyed by where it lands
    in the run: the edge connected there, what the outer wiring decided for
    it, and the value typed on the graph node, as if typed on the inner field.
    ``condition_above`` answers for a field outside the graph."""
    prefix = f"{node_id}{SEPARATOR}"
    inner_nodes = embeddable.nodes

    def inside(ref: Ref) -> Ref:
        node_part, _, field_part = ref.partition(".")
        return Ref._of_parts(prefix + node_part, field_part)

    edges_on: dict[str, dict[str, From]] = {}
    for ref, edge in edges.items():
        edges_on.setdefault(ref.node_id, {})[ref.field] = edge
    values_on: dict[str, dict[str, Any]] = {}
    for ref, value in values.items():
        values_on.setdefault(ref.node_id, {})[ref.field] = value

    lifted: dict[str, CompiledNode] = {}

    def condition_of(ref: Ref) -> Condition:
        node = lifted.get(ref.node_id)
        if node is None:
            return condition_above(ref)
        found = node.wiring.fields.get(ref.field) if node.wiring is not None else None
        return found.condition if found is not None and found.condition is not None else ALWAYS

    for inner_id, inner in inner_nodes.items():
        new_id = prefix + inner_id
        bindings: dict[str, Binding] = {
            name: (From(*(inside(ref) for ref in binding.refs)) if isinstance(binding, From) else binding)
            for name, binding in inner.graph_node.bindings.items()
        }
        crossed = edges_on.get(inner_id, ())
        bindings.update(crossed)
        statics = {name: value for name, value in inner.statics.items() if name not in crossed}
        for name, value in values_on.get(inner_id, {}).items():
            bindings[name], statics[name] = Static(value), value
        graph_node = inner.graph_node.model_copy(update={"id": new_id, "bindings": bindings, "locked": ()})
        assert inner.wiring is not None and inner.interface is not None  # an embeddable graph is wired throughout
        inner_row = inner.wiring._runs_per_row_of
        new_row = row if inner_row is None else under(inner_row, prefix, row)
        if inner.embedded_graph is not None:
            fields: dict[str, CompiledField] = {}
        else:
            at_node = condition_at_node(bindings, condition_of)
            runs_once = new_row is None
            outputs_of = {out.name: out for out in inner.interface.outputs}
            fields = {}
            for name, field in inner.wiring.fields.items():
                condition = None
                if field._view_of is not None:  # a column appears exactly when its table does
                    condition = fields[field._view_of[0].field].condition
                elif field._is_output:
                    condition = output_condition(new_id, outputs_of[name], at_node, runs_once)
                dtype, index, receives = _field_under(field, crossings, inner_nodes, prefix, row, inside)
                makes = None if field._makes_rows is None else under(field._makes_rows, prefix, row)
                fields[name] = CompiledField(
                    ref=Ref._of_parts(new_id, name), type=dtype, binding=bindings.get(name) if not field._is_output else None, condition=condition,
                    problems=(), _index=index, _receives=receives, _listed=field._listed and name not in crossed,
                    _makes_rows=makes, _view_of=None if field._view_of is None else (inside(field._view_of[0]), field._view_of[1]),
                    _takes_blank=field._takes_blank, _row_origins=None,
                )
        lifted[new_id] = CompiledNode(
            id=new_id, graph_node=graph_node, embedded_in=embedded_in(new_id), problems=(), cause=None,
            definition=inner.definition, version=inner.version, interface=inner.interface, statics=statics,
            wiring=Wiring(fields, new_row),
        )
    return lifted


def table_rows_inside(lifted: Mapping[str, CompiledNode]) -> list[str]:
    """The index ids of every table an embedded graph's nodes make: their rows' origins can't be seen from here."""
    return [
        field._makes_rows.id
        for node in lifted.values() for field in node.wiring.fields.values()
        if field._makes_rows is not None and holds_tables(field.type)
    ]


def _field_under(
    field: CompiledField, crossing_at: Mapping[Ref, Crossing], nodes: Mapping[str, CompiledNode], prefix: str,
    row: Index | None, inside: Callable[[Ref], Ref],
) -> tuple[Any, Index | None, Receive | None]:
    """An inner field's type, rows and receipt, as it stands embedded (rules 2-4)."""
    crossing = crossing_at.get(field.ref) if crossing_at else None
    if crossing is not None:
        return crossing.type, crossing.index, crossing.receives
    dtype, index, receipt = field.type, field._index, field._receives
    is_output = field._is_output
    edge_fed = isinstance(field.binding, From)
    once_per_row = row is not None and index is None and (is_output or edge_fed)
    new_type = Series[dtype] if once_per_row else dtype
    new_index = row if once_per_row else under(index, prefix, row)
    if is_output:
        return new_type, new_index, None
    if once_per_row and isinstance(receipt, Broadcast):
        return new_type, new_index, PerRow(row)
    if row is not None and isinstance(receipt, Gather) and edge_fed and all(_source_index(source, nodes) is None for source in field.binding.refs):
        return new_type, row, RowsUnder(row, row.depth)
    if isinstance(receipt, Merge):
        return new_type, new_index, Merge(
            tuple((inside(source), _part_under(part, _source_index(source, nodes), prefix, row)) for source, part in receipt.parts),
            frozenset(inside(source) for source in receipt.copied_down),
        )
    return new_type, new_index, _received_under(receipt, index, prefix, row)


def _source_index(source: Ref, nodes: Mapping[str, CompiledNode]) -> Index | None:
    return nodes[source.node_id].wiring.fields[source.field]._index


def under(index: Index | None, prefix: str, row: Index | None) -> Index | None:
    """An inner index as it stands embedded: each id prefixed, its root hung under ``row`` (rule 1)."""
    if index is None:
        return None
    parent = row
    for step in index.lineage:
        parent = Index(f"{prefix}{step.id}", parent=parent)
    return parent


def _part_under(part: Any, index: Index | None, prefix: str, row: Index | None) -> Any:
    """One source of an inner ``Table`` input as it stands embedded: a table the
    inner graph makes once is one per ``row`` when the graph runs per row."""
    if isinstance(part, Broadcast) and index is None and row is not None:
        return PerRow(row)
    return _received_under(part, index, prefix, row)


def _received_under(receipt: Receive, index: Index | None, prefix: str, row: Index | None) -> Receive:
    """An inner receipt as it stands embedded (rules 1 and 3)."""
    if isinstance(receipt, PerRow):
        return PerRow(under(receipt.index, prefix, row))
    if isinstance(receipt, RowsUnder):
        return RowsUnder(under(receipt.index, prefix, row), receipt.depth + (0 if row is None else row.depth))
    if isinstance(receipt, Gather):
        return Gather(under(receipt.index, prefix, row))
    if isinstance(receipt, EntireValue):
        return receipt if row is None else RowsUnder(under(index, prefix, row), row.depth)
    return receipt
