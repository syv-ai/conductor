"""``GraphRecord`` and ``NodeRecord``, the stored graph; ``make_graph`` writes one from Python.

A graph record is its nodes and its edges, and a ``display`` of the host's own::

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
caller may still give it another value. A list given to an input that takes
one value, by the caller or typed in, is many values: the node runs once per
item. ``display`` is the host's own: conductor stores it and never reads it. An empty field is left out when the
record is saved, so a bare node is ``{"id": "up", "node": "upper"}``.

Edges are a list of ``[from, to]`` addresses, ``node.field`` or
``node.output.column`` for one column of a table, kept sorted, so two records
that differ only in the order of their edges are equal and save alike. An
edge listed twice stays twice, and loading reports it (``repeated_edge``).
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
from typing import TYPE_CHECKING, Annotated, Any

import polars as pl
from pydantic import (
    AfterValidator,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticSerializationError, to_jsonable_python

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, Problem, make_problem

if TYPE_CHECKING:
    from conductor._internal.node_definition import NodeDefinition


def _checked_address(address: str) -> str:
    """``address`` once it reads ``node.field``, or ``node.output.column``; a column's own name may hold dots."""
    node_id, _, rest = address.partition(".")
    field, dot, column = rest.partition(".")
    if not node_id or not field or (dot and not column):
        raise ValueError(f"{address!r} is not an address; write 'node.field', or 'node.output.column' for a column of a table")
    return address


#: An address, ``"node.field"`` or ``"node.output.column"``: node ids and field names hold no dot, so everything
#: after the second is the column's name (``"read.result.Amount in USD."``).
Address = Annotated[str, AfterValidator(_checked_address)]


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


class NodeRecord(ConductorModel):
    """One node of a graph record: which definition it uses, and what is typed into its inputs.

    ``id`` names this use of the definition within the graph, and edges and
    addresses say it; ``node`` is the definition's id. The node's inputs and
    outputs are what loading answers (``graph.describe()``), not stored here.
    """

    id: str
    #: The id of the node definition this node uses.
    node: str
    #: Left out of the saved record when it is 1, as is each field below when empty.
    version: int = Field(default=1, exclude_if=lambda version: version == 1)
    #: Typed into these inputs and fixed: no caller may change them.
    values: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: Typed into these inputs, which stay open for a caller to change.
    defaults: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)


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
    #: The host's own about the whole graph: a viewport, flags. Stored, never read here, and no part of the fingerprint.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
        return tuple(nodes)

    @field_validator("edges")
    @classmethod
    def _edges_sorted(cls, edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(edges))


def as_record(record: GraphRecord | str | bytes | Mapping[str, Any]) -> GraphRecord:
    """A record given as itself, as JSON text, or as a mapping read from JSON.

    What doesn't have a record's shape (text that isn't JSON, a key a record
    doesn't have, a field of the wrong type, an address that isn't one) is an
    ``InvalidCallError`` of ``invalid_record`` problems, one per fault, each at
    ``("record", *where)``. What has the shape and can't be meant is the
    loaded graph's problem, never an exception.
    """
    if isinstance(record, GraphRecord):
        return record
    try:
        if isinstance(record, (str, bytes)):
            return GraphRecord.model_validate_json(record)
        return GraphRecord.model_validate(record)
    except ValidationError as invalid:
        raise InvalidCallError([
            make_problem("invalid_record", "record", *error["loc"], at=".".join(map(str, error["loc"])) or "the record", reason=_said_of_shape(error), error=error["type"])
            for error in invalid.errors(include_url=False, include_input=False)
        ]) from None


def _said_of_shape(error: Any) -> str:
    """pydantic's sentence for a fault of shape; a validator's own sentence keeps only what it says after ``Value error,``."""
    return str(error["msg"]).removeprefix("Value error, ")


def typed_in(node_id: str, values: Mapping[str, Any]) -> tuple[dict[str, JsonValue], list[Problem]]:
    """``values`` typed into the node ``node_id`` as its record keeps them, JSON; and an ``invalid_value`` for each with no JSON form."""
    typed: dict[str, JsonValue] = {}
    problems: list[Problem] = []
    for name, value in values.items():
        try:
            typed[name] = to_jsonable_python(value)
        except PydanticSerializationError:
            reason = "a value typed into a graph is saved with it as JSON"
            if isinstance(value, pl.DataFrame):
                reason += "; give a table as a list of rows"
            problems.append(make_problem("invalid_value", node_id, name, reason=reason))
    return typed, problems


def make_graph(
    *nodes: NodeDefinition[..., Any],
    edges: Sequence[tuple[str, str]] | None = None,
    values: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> GraphRecord:
    """A graph record written from Python, each node at its version::

        make_graph(read_file, upper)                                  # a chain
        make_graph(read_file, upper, edges=[("read_file.result", "upper.text")])
        make_graph(read_file, upper, values={"read_file.path": "in.txt"})

    A node's id in the graph is its definition's id; a definition used more
    than once is ``upper-1``, ``upper-2``, and so on, skipping an id another
    node already has. ``values`` and ``defaults`` are keyed by ``"node.input"``
    addresses of those ids, as the record's fields of those names are: a value
    is fixed, and a default stays open to the caller. A list typed into an input
    that takes one value is many values, a row per item.

    With no ``edges`` the nodes form a chain: each takes its one required
    input that nothing typed in fills from the one output of the node before
    it. With ``edges``, those are every edge of the graph. Nothing is matched
    by name.

    The record is loaded against the definitions here, and every fault of what
    the call said (an address, a value, a type, an edge into a value or a
    setting) raises one ``InvalidCallError`` listing each; so does a chain it
    can't draw. A graph still half drawn (an input left open, a cycle) is
    written: ``graph.problems`` says that when it loads.
    """
    from conductor._internal.graph import load_graph  # which imports this module
    from conductor._internal.node_definition import NodeDefinition  # which imports this module

    for argument in nodes:
        if not isinstance(argument, NodeDefinition):
            raise TypeError(f"make_graph takes node definitions, made with @node; got {argument!r}")
    ids = _ids_of(nodes)
    known = dict(zip(ids, nodes, strict=True))
    problems: list[Problem] = []
    typed: dict[str, dict[str, dict[str, JsonValue]]] = {"values": {}, "defaults": {}}
    for kind, given in (("values", values or {}), ("defaults", defaults or {})):
        by_node: dict[str, dict[str, Any]] = {}
        for address, value in given.items():
            node_id, _, name = address.partition(".")
            if node_id not in known:
                problems.append(make_problem("node_not_in_graph", node_id, node_id=node_id, nodes=ids))
            else:
                by_node.setdefault(node_id, {})[name] = value
        for node_id, named in by_node.items():
            typed[kind][node_id], faults = typed_in(node_id, named)
            problems += faults
    if edges is None:
        edges = []
        for index in range(1, len(nodes)):
            filled = {**typed["values"].get(ids[index], {}), **typed["defaults"].get(ids[index], {})}
            edge = _chain_edge(ids[index - 1], nodes[index - 1], ids[index], nodes[index], filled)
            if isinstance(edge, Problem):
                problems.append(edge)
            else:
                edges.append(edge)
    else:
        for source, target in edges:
            try:
                _checked_address(source), _checked_address(target)
            except ValueError as wrong:
                problems.append(make_problem("invalid_record", "record", "edges", at=f"{source} -> {target}", reason=str(wrong), error="value_error"))
    if problems:
        raise InvalidCallError(problems)
    record = GraphRecord(
        nodes=[
            NodeRecord(
                id=node_id, node=definition.id, version=definition.version, values=typed["values"].get(node_id, {}),
                defaults=typed["defaults"].get(node_id, {}),
            )
            for node_id, definition in known.items()
        ],
        edges=list(edges),
    )
    found = [p for p in load_graph(record, {d.id: d for d in nodes}).problems if p.type in SAID_AT_THE_CALL]
    if found:
        raise InvalidCallError(found)
    return record


#: What loading finds that ``make_graph`` raises: a fault of what the call said (an address, a value, a type, an edge
#: where a value or a setting is), never a half-drawn graph's.
SAID_AT_THE_CALL = frozenset({
    "unknown_input", "unknown_output", "edge_names_missing_node", "invalid_value", "value_and_default", "value_and_edge",
    "setting_on_edge", "type_mismatch", "columns_missing", "repeated_edge",
})


def _ids_of(nodes: Sequence[NodeDefinition[..., Any]]) -> list[str]:
    """Each node's id in the graph: its definition's, numbered ``-1``, ``-2`` for one used more than once."""
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
    return ids


def _chain_edge(
    before_id: str, before: NodeDefinition[..., Any], after_id: str, after: NodeDefinition[..., Any], typed_in: Mapping[str, Any],
) -> tuple[str, str] | Problem:
    """The one edge from ``before`` into ``after`` in a chain, or the problem ``not_a_chain`` when there isn't exactly one."""
    outputs = list(before.outputs)
    open_inputs = [name for name, spec in after.inputs.items() if spec.required and not spec.param.setting and name not in typed_in]
    if len(outputs) != 1 or len(open_inputs) != 1:
        return make_problem("not_a_chain", after_id, before=before_id, inputs=open_inputs, outputs=outputs)
    return f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"
