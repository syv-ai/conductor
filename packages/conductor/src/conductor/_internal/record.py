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
caller may still give it another value. A list given to an input that takes
one value, by the caller or typed in, is many values: the node runs once per
item. ``display`` is the host's own: conductor stores it and never reads it. An empty field is left out when the
record is saved, so a bare node is ``{"id": "up", "node": "upper"}``.

Edges are a list of ``[from, to]`` addresses, ``node.field`` or
``node.output.field`` for one field of a record, kept sorted, so two records
that differ only in the order of their edges are equal and save alike. An
edge listed twice stays twice, and loading reports it (``repeated_edge``).
Several edges into one input are alternatives, opposite branches of one
decision, so nothing depends on the order of the list.

``registry.load`` reads a record into a ``Graph``; what the record says that
can't be meant (a repeated edge, an edge naming a node the graph doesn't
have, an input set two ways) is a problem of that graph.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any

import polars as pl
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticSerializationError, from_json, to_json

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.fields import REQUIRED, SET_HAS_NO_ORDER
from conductor._internal.problem import (
    InvalidCallError,
    Problem,
    invalid_record,
    make_problem,
    not_a_chain,
)

if TYPE_CHECKING:
    from conductor._internal.node_definition import NodeDefinition


def _checked_address(address: str) -> str:
    """``address`` once it reads ``node.field``, or ``node.output.field`` for one field of a record an output
    holds; that field's own name may hold dots."""
    node_id, _, rest = address.partition(".")
    field, dot, of_record = rest.partition(".")
    if not node_id or not field or (dot and not of_record):
        raise ValueError(f"{address!r} is not an address; write 'node.field', or 'node.output.field' for one field of a record")
    return address


#: An address, ``"node.field"`` or ``"node.output.field"`` for one field of a record an output holds: node ids and
#: a node's field names hold no dot, so everything after the second is the record's field (``"split.result.Amount in USD."``).
Address = Annotated[str, AfterValidator(_checked_address)]


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


#: A record holds JSON, which has no float that isn't finite: one is refused, so a record equals what its JSON reads back.
_JSON_ONLY = ConfigDict(allow_inf_nan=False)


class NodeRecord(ConductorModel):
    """One node of a graph record: which definition it uses, and what is typed into its inputs.

    ``id`` names this use of the definition within the graph, and edges and
    addresses say it; ``node`` is the definition's id. The node's inputs and
    outputs are what loading answers (``graph.describe()``), not stored here.
    """

    model_config = ConductorModel.model_config | _JSON_ONLY

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
    nodes nothing reads. What a host keeps about the whole graph (its name, a
    viewport) it stores beside the record.
    """

    model_config = ConductorModel.model_config | _JSON_ONLY

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
        raise InvalidCallError(_not_a_record(invalid)) from None


def _not_a_record(invalid: ValidationError) -> list[Problem]:
    """Gives an ``invalid_record`` problem for each fault pydantic found reading a record, at ``("record", *where)``."""
    return [
        invalid_record("record", *error["loc"], at=".".join(map(str, error["loc"])) or "the record", reason=error["msg"].removeprefix("Value error, "), error=error["type"])
        for error in invalid.errors(include_url=False, include_input=False)  # a validator's own sentence, without pydantic's prefix
    ]


def typed_in(node_id: str, values: Mapping[str, Any]) -> tuple[dict[str, JsonValue], list[Problem]]:
    """``values`` typed into the node ``node_id`` as its record keeps them, each as its JSON reads back (a date as its ISO
    text, a float that isn't finite as ``"NaN"`` or ``"Infinity"``); and an ``invalid_value`` for each with no JSON form,
    or holding a set, whose JSON would list it in another order in another process (``set_has_no_order``)."""
    typed: dict[str, JsonValue] = {}
    problems: list[Problem] = []
    for name, value in values.items():
        if _holds_a_set(value):
            errors: list[dict[str, JsonValue]] = [{"type": "set_has_no_order", "loc": [], "msg": SET_HAS_NO_ORDER}]
            problems.append(make_problem("invalid_value", node_id, name, reason=SET_HAS_NO_ORDER, errors=errors))
            continue
        try:
            typed[name] = from_json(to_json(value, inf_nan_mode="strings"))
        except PydanticSerializationError:
            reason = "a value typed into a graph is saved with it as JSON"
            if isinstance(value, pl.DataFrame):
                reason += "; give a table as a list of records"
            problems.append(make_problem("invalid_value", node_id, name, reason=reason))
    return typed, problems


def _holds_a_set(value: Any) -> bool:
    if isinstance(value, (set, frozenset)):
        return True
    if isinstance(value, dict):
        return any(map(_holds_a_set, value.values()))
    return isinstance(value, (list, tuple)) and any(map(_holds_a_set, value))


def make_graph(
    *nodes: NodeDefinition,
    edges: Sequence[tuple[str, str]] | None = None,
    values: Mapping[str, Any] = MappingProxyType({}),
    defaults: Mapping[str, Any] = MappingProxyType({}),
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
    that takes one value is refused: a node that returns a ``Series`` makes
    the items a graph runs per.

    With no ``edges`` the nodes form a chain: each takes its one required
    input that nothing typed in fills from the one output of the node before
    it. With ``edges``, those are every edge of the graph. Nothing is matched
    by name.

    The record is written and nothing is loaded: an input or output a node
    doesn't have, a value its field refuses, an edge between types that don't
    meet and a graph half drawn are in ``registry.load(record).problems``.

    Raises:
        InvalidCallError: Listing every fault in writing the record itself: a value or default addressed to
            a node not given (``node_not_in_graph``), a value with no JSON form or holding a set
            (``invalid_value``), an edge that is no address (``invalid_record``) and a chain that can't
            be drawn (``not_a_chain``).
    """
    ids = _ids_of(nodes)
    problems: list[Problem] = []
    typed_values: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    typed_defaults: dict[str, dict[str, JsonValue]] = {node_id: {} for node_id in ids}
    for given, typed in ((values, typed_values), (defaults, typed_defaults)):
        by_node: dict[str, dict[str, Any]] = {}
        for address, value in given.items():
            node_id, _, name = address.partition(".")
            if node_id not in typed:
                problems.append(make_problem("node_not_in_graph", node_id, node_id=node_id, nodes=ids))
            else:
                by_node.setdefault(node_id, {})[name] = value
        for node_id, named in by_node.items():
            typed[node_id], faults = typed_in(node_id, named)
            problems += faults
    if edges is None:
        edges, faults = _chain_edges(ids, nodes, typed_values, typed_defaults)
        problems += faults
    try:
        record = GraphRecord(
            nodes=[
                NodeRecord(id=node_id, node=definition.id, version=definition.version, values=typed_values[node_id], defaults=typed_defaults[node_id])
                for node_id, definition in zip(ids, nodes, strict=True)
            ],
            edges=edges,
        )
    except ValidationError as invalid:  # an edge that is no address
        problems += _not_a_record(invalid)
    if problems:
        raise InvalidCallError(problems)
    return record


def _ids_of(nodes: Sequence[NodeDefinition]) -> list[str]:
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


def _chain_edges(
    ids: Sequence[str], nodes: Sequence[NodeDefinition], typed_values: Mapping[str, Mapping[str, JsonValue]], typed_defaults: Mapping[str, Mapping[str, JsonValue]],
) -> tuple[list[tuple[str, str]], list[Problem]]:
    """The one edge from each node into the next, and ``not_a_chain`` for each pair without exactly one."""
    from conductor._internal.node_definition import FunctionDefinition  # which imports this module

    edges: list[tuple[str, str]] = []
    problems: list[Problem] = []
    for index in range(1, len(nodes)):
        before_id, before, after_id, after = ids[index - 1], nodes[index - 1], ids[index], nodes[index]
        filled = typed_values[after_id].keys() | typed_defaults[after_id].keys()
        match before, after:
            case FunctionDefinition(), FunctionDefinition():
                outputs = list(before.outputs)
                open_inputs = [name for name, spec in after.inputs.items() if spec.default is REQUIRED and name not in filled]
            case _:  # a stored graph's fields are known only when it loads, so a chain can't be drawn through it
                outputs, open_inputs = [], []
        if len(outputs) != 1 or len(open_inputs) != 1:
            problems.append(not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs))
        else:
            edges.append((f"{before_id}.{outputs[0]}", f"{after_id}.{open_inputs[0]}"))
    return edges, problems
