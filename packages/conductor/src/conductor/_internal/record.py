"""``GraphRecord`` and ``NodeRecord``, the stored graph; ``make_graph`` writes one from Python.

A graph record is its nodes and its edges, and nothing else::

    {
        "nodes": [
            {"id": "read", "node": "read_file", "values": {"path": "in.txt"}},
            {"id": "up", "node": "upper"}
        ],
        "edges": [["read.text", "up.text"]]
    }

A node names the definition it uses (``node``) and the version (``version``,
1 when left out). ``values`` are typed into its inputs and fixed there: no
caller may change them. ``defaults`` fill an input that stays open, so a
caller may still give it another value. ``many`` names the inputs that take
a list, a row per item: an open input of a first node takes it from the
caller, and an input with a value is a list typed in, whose items are a table
under each row the node runs on, so the node runs once per item on each. ``display`` is the host's own:
conductor stores it and never reads it. An empty field is left out when the
record is saved, so a bare node is ``{"id": "up", "node": "upper"}``.

Edges are a set of ``[from, to]`` addresses, ``node.field`` or
``node.output.column`` for one column of a table, kept sorted, so two records
that differ only in the order of their edges are equal and save alike.
Several edges into one input on rows of one table gather in that table's row
order; on different tables, in the order their source nodes appear in
``nodes``, then in the order the source node declares its outputs.

``registry.load`` reads a record into a ``Graph``; what the record says that
can't be meant (a repeated edge, an edge naming a node the graph doesn't
have, an input set two ways) is a problem of that graph.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from conductor._internal.conductor_model import ConductorModel

if TYPE_CHECKING:
    from conductor._internal.node_definition import NodeDefinition


def _checked_address(address: str) -> str:
    """``address`` once it reads ``node.field``, or ``node.output.column``."""
    parts = address.split(".")
    if not 2 <= len(parts) <= 3 or not all(parts):
        raise ValueError(f"{address!r} is not an address; write 'node.field', or 'node.output.column' for a column of a table")
    return address


#: An address, ``"node.field"`` or ``"node.output.column"``.
Address = Annotated[str, AfterValidator(_checked_address)]


class NodeRecord(ConductorModel):
    """One node of a graph record: which definition it uses, and what is typed into its inputs.

    ``id`` names this use of the definition within the graph, and edges and
    addresses say it; ``node`` is the definition's id. The node's inputs and
    outputs are what loading answers (``graph.describe()``), not stored here.
    """

    id: str
    #: The id of the node definition this node uses.
    node: str
    version: int = 1
    #: Typed into these inputs and fixed: no caller may change them.
    values: Mapping[str, Any] = Field(default_factory=dict)
    #: Typed into these inputs, which stay open for a caller to change.
    defaults: Mapping[str, Any] = Field(default_factory=dict)
    #: The inputs that take a list, a row per item: from the caller on an open input, or typed in as the value.
    many: Annotated[AbstractSet[str], AfterValidator(frozenset)] = frozenset()
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _without_empty_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """The record as saved: an empty field and the version 1 are left out, ``many`` sorted."""
        written: dict[str, Any] = handler(self)
        if "many" in written:
            written["many"] = sorted(written["many"])
        return {key: value for key, value in written.items() if value not in ({}, [], ()) and not (key == "version" and value == 1)}


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes and its edges.

    What a graph takes and returns is derived when it is loaded, never
    stored: the open inputs of the nodes nothing feeds, and the outputs of the
    nodes nothing reads.
    """

    #: Any sequence going in; a tuple once built.
    nodes: Sequence[NodeRecord] = ()
    #: ``(from, to)`` pairs of addresses, ``("read.text", "up.text")``; sorted once built.
    edges: Sequence[tuple[Address, Address]] = ()

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
        return tuple(nodes)

    @field_validator("edges")
    @classmethod
    def _edges_sorted(cls, edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(edges))


def as_record(record: GraphRecord | str | bytes | Mapping[str, Any]) -> GraphRecord:
    """A record given as itself, as JSON text, or as a mapping read from JSON."""
    if isinstance(record, GraphRecord):
        return record
    if isinstance(record, (str, bytes)):
        return GraphRecord.model_validate_json(record)
    return GraphRecord.model_validate(record)


def make_graph(
    *nodes: NodeDefinition[..., Any],
    edges: Sequence[tuple[str, str]] | None = None,
    values: Mapping[str, Any] | None = None,
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(read_file, upper)                                  # a chain
        make_graph(read_file, upper, edges=[("read_file.result", "upper.text")])
        make_graph(read_file, upper, values={"read_file.path": "in.txt"})

    A node's id in the graph is its definition's id; a definition used more
    than once is ``upper-1``, ``upper-2``, and so on, skipping an id another
    node already has. ``values`` are keyed by those ids, ``"node.input"``,
    and fixed in the record.

    With no ``edges`` the nodes form a chain: each takes its one required
    input that no value fills from the one output of the node before it, and
    anything else raises ``TypeError`` here. With ``edges``, those are every
    edge of the graph. Nothing is matched by name.
    """
    used = Counter(definition.id for definition in nodes)
    taken = {definition.id for definition in nodes if used[definition.id] == 1}
    numbered: Counter[str] = Counter()
    ids: list[str] = []
    for definition in nodes:
        node_id = definition.id
        while used[definition.id] > 1 and (node_id == definition.id or node_id in taken):
            numbered[definition.id] += 1
            node_id = f"{definition.id}-{numbered[definition.id]}"
        taken.add(node_id)
        ids.append(node_id)
    values_of: dict[str, dict[str, Any]] = {}
    for address, value in (values or {}).items():
        node_id, _, name = address.partition(".")
        if node_id not in ids:
            raise TypeError(f"make_graph: values names {node_id!r}, which is not one of the nodes {', '.join(ids)}")
        values_of.setdefault(node_id, {})[name] = value
    if edges is None:
        edges = [_chain_edge(ids[index - 1], nodes[index - 1], ids[index], nodes[index], values_of.get(ids[index], {})) for index in range(1, len(nodes))]
    return GraphRecord(
        nodes=[
            NodeRecord(id=node_id, node=definition.id, version=definition.version, values=values_of.get(node_id, {}))
            for node_id, definition in zip(ids, nodes, strict=True)
        ],
        edges=list(edges),
    )


def _chain_edge(before_id: str, before: NodeDefinition[..., Any], after_id: str, after: NodeDefinition[..., Any], typed_in: Mapping[str, Any]) -> tuple[str, str]:
    """The one edge from ``before`` into ``after`` in a chain; ``TypeError`` when there isn't exactly one."""
    outputs = list(before.outputs)
    open_inputs = [name for name, spec in after.inputs.items() if spec.required and name not in typed_in]
    if len(outputs) != 1 or len(open_inputs) != 1:
        raise TypeError(
            f"make_graph: a chain takes {after_id!r}'s one required open input (it has {', '.join(open_inputs) or 'none'}) "
            f"from {before_id!r}'s one output (it has {', '.join(outputs) or 'none'}); pass edges="
        )
    return f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"
