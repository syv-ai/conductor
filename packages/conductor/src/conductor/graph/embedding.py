"""A compiled graph placed as one node: its records, as they stand under the row that node runs on.

A host compiles each graph version once and hands the ``CompiledGraph`` to
conductor as the node's version. The outer compile walks the placed graph
as a plain node with the compiled graph's interface, which decides one
thing: the row the node runs on (``runs_per_row_of``), or none. This module
then lifts what the graph's own compile decided — its nodes, their fields'
types, rows and receipts, and the rows its nodes run on — under that row,
so the graph that places it reads exactly as if the graph's nodes had been
drawn in its place and walked there. Conditions aren't lifted: compile
works them out once the nodes are in place, as for any node. The compiler
lifts through it (``Compilation.place_graphs``); ``CompiledGraph`` reads
its address helpers.

The lift, rule by rule:

1. Every index inside is renamed under the placed node (``emb/lines``), and
   its root hangs under ``runs_per_row_of``.
2. When the node runs per row, a value that ran once inside becomes a series
   on that row: ``Txt`` becomes ``Series[Txt]``, no index becomes the row,
   and an edge-fed ``Broadcast`` becomes ``Iterate``.
3. ``Whole`` becomes ``Group`` at the row's depth; a ``Gather`` whose every
   source carries one value becomes ``Group`` on the row, since per row the
   sources line up; a ``Group`` goes as many levels deeper as the row is deep.
4. An inner field the outer graph feeds (a crossing) takes what the outer
   walk decided for the placed node's field of that address.
5. An output read outside already sits on the inner field's rows: the walk
   gave it the rows it has inside, named and hung the same way (rule 1),
   and keyed every field of the placed node by its address inside, so an
   index the walk named after one of them is already the inner field's.

Refusing a graph that places itself is the host's job, as it builds the
versions: a compiled graph can't contain itself, but a host that swaps a
version in place can close a loop through a value typed on a placed graph,
which compile then follows until Python's recursion limit.

``/`` separates the levels inside a node id, and ``.`` stays the address
separator, so an address is a path and a field: ``approve/check.amount``.
The first ``.`` always splits it, and the field is free text. Only the lift
writes a ``/``: an id holding one is refused (``invalid_node_id``), or it
could be read in a lifted node's place. So a lifted id is the path to its
node, and the graph it sits in is read off it (``embedded_in``).

An embedded graph's own field names are its addresses (``check.amount``), so
the field ``check.amount`` on the graph node ``approve`` is at
``approve/check.amount`` (``address``). Compile reads the author's spelling
of it, ``approve.check.amount``, once, when it lays out the graph that runs;
every step after it, and every reader of the compiled graph, sees only the
address.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor.graph.binding import Binding, From, Static
from conductor.graph.model import GraphNode
from conductor.graph.receive import Broadcast, Gather, Group, Iterate, Receive, Whole
from conductor.ref import Ref
from conductor.series import Index, Series

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph
    from conductor.interface import Interface
    from conductor.node import NodeDefinition


SEPARATOR = "/"


def address(node_id: str, name: str, graphs: Collection[str]) -> Ref:
    """Where field ``name`` of node ``node_id`` is, in the graph that runs.

    On a node whose version is a compiled graph (one of ``graphs``) the name
    is that graph's own address for the field, so the address is its path
    under the node: ``approve`` + ``check.amount`` is ``approve/check.amount``.
    On any other node it is ``node_id.name``. Compile asks it when it lays
    out edges, walks a graph node's fields and names the graph's interface;
    ``as_drawn`` is the other way round."""
    return Ref(f"{node_id}{SEPARATOR}{name}") if node_id in graphs else Ref(node_id, name)


def as_drawn(ref: Ref, graphs: Collection[str]) -> tuple[str, str]:
    """The node an address sits on in the graph its author drew, and the field name there.

    ``("approve", "check.amount")`` for ``approve/check.amount`` when
    ``approve`` is one of ``graphs``; any other address is its own node and
    field. ``address`` the other way round. Read only where the author's node
    is what is meant: ``with_inputs`` writing a value on it, and compile
    naming the node an edge reads from."""
    outer, _, inner = ref.node_id.partition(SEPARATOR)
    return (outer, str(Ref(inner, ref.field))) if inner and outer in graphs else (ref.node_id, ref.field)


def embedded_in(node_id: str) -> str | None:
    """The graph a lifted node sits in, read off its id — ``approve`` for
    ``approve/check``, ``approve/check`` for ``approve/check/amount`` — or
    ``None`` for an id with no ``/``."""
    outer, _, _ = node_id.rpartition(SEPARATOR)
    return outer or None


@dataclass(frozen=True)
class Crossing:
    """What the outer walk decided for one input of the placed node that an
    outer edge feeds: the type, rows and receipt the inner field takes."""

    type: Any
    index: Index | None
    receives: Receive


@dataclass(frozen=True)
class Placed:
    """A compiled graph's records under one placed node, keyed by the ids and
    refs the graph that places it uses (``emb/join``, ``emb/join.result``).
    ``Compilation.place_graphs`` merges each into its own records."""

    nodes: dict[str, GraphNode]
    order: tuple[str, ...]
    #: The graph nodes inside, nested ones included, by lifted id.
    graphs: frozenset[str]
    versions: dict[str, Any]
    definitions: dict[str, type[NodeDefinition]]
    interfaces: dict[str, Interface]
    statics: dict[str, dict[str, Any]]
    listed: dict[str, frozenset[str]]
    iterated: dict[str, Index | None]
    types: dict[Ref, Any]
    indexes: dict[Ref, Index | None]
    receives: dict[Ref, Receive]


def graph_as_placed(
    graph: CompiledGraph,
    node_id: str,
    edges: Mapping[str, From],
    values: Mapping[str, Any],
    runs_per_row_of: Index | None,
    crossings: Mapping[str, Crossing],
) -> Placed:
    """``graph``'s records as they stand placed at ``node_id``, running once per row of ``runs_per_row_of``.

    ``edges`` holds, by the placed node's input name (``holder.value``), the
    edge the outer graph connected there, and ``crossings`` what the outer
    walk decided for each. ``values`` holds, by input name, what the outer graph typed on
    the placed node; each lands on the inner field as if typed there. A
    value that decides something inside (a list, or a node whose hook reads
    it) is already in ``graph``: the compiler compiled it again with the
    value (``Compilation._as_placed``).
    """
    inner = graph._compilation
    expansion, iteration = inner.expansion, inner.iteration
    prefix = f"{node_id}{SEPARATOR}"

    def lifted(ref: Ref) -> Ref:
        return Ref(prefix + ref.node_id, ref.field)

    edge_at = {Ref(name): edge for name, edge in edges.items()}
    value_at = {Ref(name): value for name, value in values.items()}
    #: The inner fields the outer graph feeds: what the inner graph put there gives way.
    crossed: dict[str, set[str]] = {}
    for ref in edge_at:
        crossed.setdefault(ref.node_id, set()).add(ref.field)

    nodes: dict[str, GraphNode] = {}
    edge_fed: set[Ref] = set()
    for inner_id, inner_node in expansion.nodes.items():
        bindings: dict[str, Binding] = {}
        for name, binding in inner_node.bindings.items():
            if isinstance(binding, From):
                edge_fed.add(Ref(inner_id, name))
                bindings[name] = From(*(lifted(ref) for ref in binding.refs))
            else:
                bindings[name] = binding
        for ref, edge in edge_at.items():
            if ref.node_id == inner_id:
                bindings[ref.field] = edge
        for ref, value in value_at.items():
            if ref.node_id == inner_id:
                bindings[ref.field] = Static(value)
        nodes[prefix + inner_id] = inner_node.model_copy(update={"id": prefix + inner_id, "bindings": bindings, "locked": ()})

    structure = {
        "nodes": nodes,
        "order": tuple(prefix + inner_id for inner_id in expansion.order),
        "graphs": frozenset(prefix + inner_id for inner_id in expansion.graphs),
        "versions": {prefix + inner_id: version for inner_id, version in expansion.versions.items()},
        "definitions": {prefix + inner_id: definition for inner_id, definition in expansion.definitions.items()},
        "interfaces": {prefix + inner_id: interface for inner_id, interface in inner.interfaces.items()},
        "statics": {
            prefix + inner_id: {
                **{name: value for name, value in held.items() if name not in crossed.get(inner_id, ())},
                **{ref.field: value for ref, value in value_at.items() if ref.node_id == inner_id},
            }
            for inner_id, held in inner.statics.items()
        },
        "listed": {prefix + inner_id: held - crossed.get(inner_id, set()) for inner_id, held in inner.listed.items()},
    }
    crossing_at = {Ref(name): crossing for name, crossing in crossings.items()}

    iterated = {
        prefix + inner_id: runs_per_row_of if index is None else under(index, prefix, runs_per_row_of)
        for inner_id, index in iteration.iterated.items()
    }
    types: dict[Ref, Any] = {}
    indexes: dict[Ref, Index | None] = {}
    receives: dict[Ref, Receive] = {}
    per_row_of = runs_per_row_of
    for ref, dtype in iteration.types.items():
        new_ref = lifted(ref)
        crossing = crossing_at.get(ref)
        if crossing is not None:
            types[new_ref], indexes[new_ref], receives[new_ref] = crossing.type, crossing.index, crossing.receives
            continue
        index = iteration.indexes[ref]
        receipt = iteration.receives.get(ref)
        is_output = receipt is None
        once_per_row = per_row_of is not None and index is None and (is_output or ref in edge_fed)
        types[new_ref] = Series[dtype] if once_per_row else dtype
        indexes[new_ref] = per_row_of if once_per_row else under(index, prefix, per_row_of)
        if is_output:
            continue
        if once_per_row and isinstance(receipt, Broadcast):
            receives[new_ref] = Iterate(per_row_of)
        elif (
            per_row_of is not None and isinstance(receipt, Gather) and ref in edge_fed
            and all(iteration.indexes[source] is None for source in expansion.nodes[ref.node_id].bindings[ref.field].refs)
        ):
            indexes[new_ref], receives[new_ref] = per_row_of, Group(per_row_of, per_row_of.depth)
        else:
            receives[new_ref] = _received_under(receipt, index, prefix, per_row_of)

    return Placed(**structure, iterated=iterated, types=types, indexes=indexes, receives=receives)


def under(index: Index | None, prefix: str, row: Index | None) -> Index | None:
    """An inner index as it stands placed: each id prefixed, its root hung under ``row`` (rule 1).

    The walk names a placed graph's outputs this way before the lift, so
    the rows outer nodes read and the rows the lifted fields have are one
    answer.
    """
    if index is None:
        return None
    parent = row
    for step in index.lineage:
        parent = Index(f"{prefix}{step.id}", parent=parent)
    return parent


def _received_under(receipt: Receive, index: Index | None, prefix: str, row: Index | None) -> Receive:
    """An inner receipt as it stands placed (rules 1 and 3)."""
    if isinstance(receipt, Iterate):
        return Iterate(under(receipt.index, prefix, row))
    if isinstance(receipt, Group):
        return Group(under(receipt.index, prefix, row), receipt.depth + (0 if row is None else row.depth))
    if isinstance(receipt, Gather):
        return Gather(under(receipt.index, prefix, row))
    if isinstance(receipt, Whole):
        return receipt if row is None else Group(under(index, prefix, row), row.depth)
    return receipt
