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
caller may still give it another value. ``display`` is the host's own:
conductor stores it and never reads it. An empty field is left out when the
record is saved, so a bare node is ``{"id": "up", "node": "upper"}``.

Edges are a set of ``[from, to]`` addresses. Several edges into one input
gather in the order their source nodes appear in ``nodes``, not in the order
the list holds them, so a host may sort the list as it likes; only two
outputs of one node into one input keep the list's order between them.

``load_record`` is the one place the record is read: it gathers each node's
edges, values and defaults onto its inputs, which is what compile reads, and
reports what the record says that can't be meant (a repeated edge, an edge
into a node the graph doesn't have, an input set two ways).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import Problem, make_problem
from conductor._internal.graph.binding import Binding, From, Static
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.node_definition import NodeDefinition

#: An address ``"node.field"``, written as a string and read as a ``Ref``.
Address = Annotated[str, AfterValidator(Ref)]


class NodeRecord(ConductorModel):
    """One node of a graph record: which definition it uses, and what is typed into its inputs.

    ``id`` names this use of the definition within the graph, and edges and
    addresses say it; ``node`` is the definition's id. The node's actual
    inputs and outputs are what compile answers
    (``CompiledGraph.node(node_id).interface``), not stored here.
    """

    id: str
    #: The id of the node definition this node uses.
    node: str
    version: int = 1
    #: Typed into these inputs and fixed: no caller may change them.
    values: Mapping[str, Any] = Field(default_factory=dict)
    #: Typed into these inputs, which stay open for a caller to change.
    defaults: Mapping[str, Any] = Field(default_factory=dict)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _without_empty_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """The record as saved: an empty map and the version 1 are left out."""
        written: dict[str, Any] = handler(self)
        return {key: value for key, value in written.items() if value != {} and not (key == "version" and value == 1)}


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes and its edges.

    What a graph takes and returns is derived when it is compiled, never
    stored: the open inputs of the nodes nothing feeds, and the outputs of the
    nodes nothing reads.
    """

    #: Any sequence going in; a tuple once built.
    nodes: Sequence[NodeRecord] = ()
    #: ``(from, to)`` pairs of addresses, ``("read.text", "up.text")``.
    edges: Sequence[tuple[Address, Address]] = ()

    @field_validator("nodes", "edges")
    @classmethod
    def _as_tuple(cls, items: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(items)


@dataclass(frozen=True)
class LoadedNode:
    """A node of a record as compile reads it: its edges, values and defaults gathered onto its inputs."""

    id: str
    #: The id of the node definition it uses.
    node: str
    version: int
    #: One binding per input the record sets.
    bindings: Mapping[str, Binding]
    #: The inputs a value fixes, which no caller may fill.
    fixed: frozenset[str] = frozenset()


def load_record(record: GraphRecord) -> tuple[tuple[LoadedNode, ...], list[Problem]]:
    """Every node of ``record`` with its inputs bound, and what the record says that can't be meant.

    An edge drawn twice, an edge into a node the record doesn't have, and an
    input with both an edge and a value (or a value and a default) are each a
    problem; the edge is kept once and wins over a value.
    """
    position: dict[str, int] = {}
    for index, node in enumerate(record.nodes):
        position.setdefault(node.id, index)
    problems: list[Problem] = []
    sources_into: dict[str, dict[str, list[Ref]]] = {}
    seen: set[tuple[str, str]] = set()
    for source, target in record.edges:
        to = Ref(target)
        if (source, target) in seen:
            problems.append(make_problem("repeated_edge", to.node_id, to.field, source=source))
            continue
        seen.add((source, target))
        if to.node_id not in position:
            problems.append(make_problem("edge_into_unknown_node", to.node_id, to.field, source=source))
            continue
        sources_into.setdefault(to.node_id, {}).setdefault(to.field, []).append(Ref(source))
    loaded = []
    for node in record.nodes:
        bindings: dict[str, Binding] = {name: Static(value) for name, value in node.defaults.items()}
        for name, value in node.values.items():
            if name in node.defaults:
                problems.append(make_problem("value_and_default", node.id, name))
            bindings[name] = Static(value)
        for name, sources in sources_into.get(node.id, {}).items():
            if name in node.values or name in node.defaults:
                problems.append(make_problem("value_and_edge", node.id, name))
            bindings[name] = From(*sorted(sources, key=lambda ref: position.get(ref.node_id, len(position))))
        loaded.append(LoadedNode(id=node.id, node=node.node, version=node.version, bindings=bindings, fixed=frozenset(node.values)))
    return tuple(loaded), problems


def make_graph(
    *nodes: NodeDefinition[..., Any],
    edges: Sequence[tuple[str, str]] | None = None,
    values: Mapping[str, Any] | None = None,
) -> GraphRecord:
    """A graph record written from Python, each node at its current version::

        make_graph(read_file, upper)                                  # a chain
        make_graph(read_file, upper, edges=[("read_file.text", "upper.text")])
        make_graph(read_file, upper, values={"read_file.path": "in.txt"})

    A node's id in the graph is its definition's id; a definition used more
    than once is ``upper-1``, ``upper-2``, and so on. ``values`` are keyed by
    those ids, ``"node.input"``, and fixed in the record.

    With no ``edges`` the nodes form a chain: each takes its one required
    input that no value fills from the one output of the node before it, and
    anything else raises ``TypeError`` here. With ``edges``, those are every
    edge of the graph. Nothing is matched by name.
    """
    used = Counter(definition.id for definition in nodes)
    numbered: Counter[str] = Counter()
    ids: list[str] = []
    for definition in nodes:
        numbered[definition.id] += 1
        ids.append(f"{definition.id}-{numbered[definition.id]}" if used[definition.id] > 1 else definition.id)
    values_of: dict[str, dict[str, Any]] = {}
    for address, value in (values or {}).items():
        ref = Ref(address)
        if ref.node_id not in ids:
            raise TypeError(f"make_graph: values names {ref.node_id!r}, which is not one of the nodes {', '.join(ids)}")
        values_of.setdefault(ref.node_id, {})[ref.field] = value
    if edges is None:
        edges = [_chain_edge(ids[index - 1], nodes[index - 1], ids[index], nodes[index], values_of.get(ids[index], {})) for index in range(1, len(nodes))]
    return GraphRecord(
        nodes=[
            NodeRecord(id=node_id, node=definition.id, version=definition.current_version, values=values_of.get(node_id, {}))
            for node_id, definition in zip(ids, nodes, strict=True)
        ],
        edges=list(edges),
    )


def _chain_edge(before_id: str, before: NodeDefinition[..., Any], after_id: str, after: NodeDefinition[..., Any], typed_in: Mapping[str, Any]) -> tuple[str, str]:
    """The one edge from ``before`` into ``after`` in a chain; ``TypeError`` when there isn't exactly one."""
    outputs = [output.name for output in before.versions[before.current_version].interface.outputs]
    open_inputs = [
        input_.name for input_ in after.versions[after.current_version].interface.inputs
        if input_.required and input_.connectable and input_.name not in typed_in
    ]
    if len(outputs) != 1 or len(open_inputs) != 1:
        raise TypeError(
            f"make_graph: a chain takes {after_id!r}'s one required open input (it has {', '.join(open_inputs) or 'none'}) "
            f"from {before_id!r}'s one output (it has {', '.join(outputs) or 'none'}); pass edges="
        )
    return f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"
