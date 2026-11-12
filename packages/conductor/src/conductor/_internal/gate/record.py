"""The record door: a graph record a host hands in, and the values ``make_graph`` writes into one.

Example: ``Graph.compile('{"nodes": [{"id": "shout", "node": "shout"}]}', registry)`` reads the text with
``read_record`` into a ``GraphRecord``. The text ``'{"nodes": [{"id": "shout"}]}'`` is refused with one
``invalid_record`` problem at ``("record", "nodes", 0, "node")``: the node names no definition. And
``make_graph(shout, constants={"shout.mark": {"!"}})`` is refused by ``record_values`` with ``invalid_value`` at
``("shout", "mark")``: a set has no order, so the record would list it in another order in another process.

``keyed_graph_record`` reads a record of the right shape for compiling: its nodes by id and the edges into each node
by input. On the way it finds what is wrong with the record's ids and edges (a node id with a ``.`` in it, two nodes
with one id, an edge listed twice, an edge naming a node the record lacks). Those faults, and what is wrong as a graph
(an edge into an input the node does not have, a constant its input's type refuses), are not refused: compiling says
so in ``graph.problems``, so an editor can show the graph while it is being drawn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import JsonValue, PlainValidator, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from conductor._internal.base import Address, is_name, plain_json
from conductor._internal.problem import Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord, NodeRecord
from conductor._internal.series import set_has_no_order
from conductor._internal.types import kind_of_value

from .refused import RefusedError


def read_record(record: GraphRecord | str | bytes) -> GraphRecord:
    """Returns ``record`` as a ``GraphRecord``: a ``GraphRecord`` as it is, JSON text read into one.

    A ``dict`` is not JSON text: it is refused as the text of a record would be, with one ``invalid_record`` problem at
    ``("record",)``. A host holding a ``dict`` passes ``GraphRecord.model_validate(data)``.

    Raises:
        RefusedError: The text is not JSON, or the record is not a record's shape (a key a record does not have, a
            field of the wrong type, an address that is not one): one ``invalid_record`` problem per fault, each at
            ``("record", *where)``.
    """
    if isinstance(record, GraphRecord):
        return record
    try:
        return GraphRecord.model_validate_json(record)
    except ValidationError as invalid:
        raise _refusal(invalid) from None


def record_from_parts(nodes: Sequence[NodeRecord], edges: Sequence[tuple[str, str]]) -> GraphRecord:
    """Returns the ``GraphRecord`` of ``nodes`` and ``edges`` a caller wrote in Python, refusing an edge that is no address.

    Raises:
        RefusedError: One ``invalid_record`` problem per fault, as ``read_record`` gives.
    """
    try:
        return GraphRecord.model_validate({"nodes": nodes, "edges": edges})
    except ValidationError as invalid:
        raise _refusal(invalid) from None


def _refusal(invalid: ValidationError, *under: str) -> RefusedError:
    """One ``invalid_record`` problem for each fault pydantic found in a graph record, or in its part ``under``
    (``"constants"``) that ``make_graph`` writes."""
    return RefusedError(
        problem.invalid_record(
            "record", *under, *error["loc"], where=".".join(map(str, (*under, *error["loc"]))) or "the record",
            reason=error["msg"].removeprefix("Value error, "), error=error["type"],  # a validator's own sentence, without pydantic's prefix
        )
        for error in invalid.errors()
    )


@dataclass(frozen=True, slots=True)
class KeyedGraphRecord:
    """A graph record read for compiling. For the README graph ``lines -> shout -> join``, ``nodes["shout"]`` is the
    node record of ``shout``, and ``edges_into["shout"]`` is ``{"line": [Address("lines", "result")]}``.

    Attributes:
        nodes: The nodes an edge can name, by id, in the record's order: each id that is a name (``is_name``), with
            the first node that has it.
        edges: Every edge of the record, in its order, as its source and its target.
        edges_into: For each of ``nodes``, by input name, the sources of the edges into that input, in the record's
            order, an edge listed twice once. An edge from a node the record lacks is kept, so its input is no graph
            input.
        problems: What is wrong with the record's ids (``invalid_node_id``, ``duplicate_node_id``), then with its
            edges, in their order (``repeated_edge``, ``edge_names_missing_node``).
    """

    nodes: dict[str, NodeRecord]
    edges: list[tuple[Address, Address]]
    edges_into: dict[str, dict[str, list[Address]]]
    problems: list[Problem]


def keyed_graph_record(record: GraphRecord) -> KeyedGraphRecord:
    """Returns ``record`` read for compiling, with what is wrong with its ids and edges. The record's validators have
    checked each edge's text; it is read into ``Address`` values here, and compiling and ``registry.upgrade`` read those."""
    nodes: dict[str, NodeRecord] = {}
    invalid: dict[str, None] = {}  # the ids that are not names, in the record's order
    twice: set[str] = set()
    for node in record.nodes:
        if not is_name(node.id):
            invalid[node.id] = None
        elif node.id in nodes:
            twice.add(node.id)
        else:
            nodes[node.id] = node
    problems = [problem.invalid_node_id(node_id) for node_id in invalid]
    problems += [problem.duplicate_node_id(node_id) for node_id in sorted(twice)]
    edges: list[tuple[Address, Address]] = []
    edges_into: dict[str, dict[str, list[Address]]] = {node_id: {} for node_id in nodes}
    previous: tuple[str, str] | None = None
    for source_text, target_text in record.edges:  # sorted, so an edge listed twice follows its first
        source, target = Address.parse(source_text), Address.parse_input(target_text)
        edges.append((source, target))
        if (source_text, target_text) == previous:
            problems.append(problem.repeated_edge(target.node, target.name, source=source_text))
            continue
        previous = source_text, target_text
        if source.node not in nodes or target.node not in nodes:
            missing = source.node if source.node not in nodes else target.node
            problems.append(problem.edge_names_missing_node(target.node, target.name, missing=missing, source=source_text, target=target_text))
        if target.node in nodes:  # kept even when the source does not exist, so the input is no graph input
            edges_into[target.node].setdefault(target.name, []).append(source)
    return KeyedGraphRecord(nodes, edges, edges_into, problems)


#: The constants or defaults ``make_graph`` is given, keyed by the addresses of inputs (``"shout.mark"``).
_VALUES_BY_INPUT: TypeAdapter[dict[Address, Any]] = TypeAdapter(dict[Annotated[Address, PlainValidator(Address.parse_input)], Any])


def values_by_input(kind: str, given: Mapping[str, Any]) -> dict[Address, Any]:
    """Returns ``given``, the ``constants`` or ``defaults`` (``kind``) of ``make_graph``, keyed by ``Address``:
    ``{"shout.mark": "?"}`` gives ``{Address("shout", "mark"): "?"}``. The values are not read here.

    Raises:
        RefusedError: One ``invalid_record`` problem for each key that is not the address of an input, at
            ``("record", kind, key, "[key]")``, pydantic's place for a key.
    """
    try:
        return _VALUES_BY_INPUT.validate_python(given)
    except ValidationError as invalid:
        raise _refusal(invalid, kind) from None


def record_values(node_id: str, values: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Returns ``values``, the constants or defaults given to the node ``node_id`` from Python, as the record keeps
    them: each as its JSON reads back. ``{"day": date(2026, 1, 2)}`` gives ``{"day": "2026-01-02"}``, and a float
    that is not finite is its text (``"NaN"``, ``"Infinity"``).

    Raises:
        RefusedError: One ``invalid_value`` problem, at ``(node_id, input)``, for each value with no JSON form or
            holding a set (``set_has_no_order``).
    """
    written: dict[str, JsonValue] = {}
    problems = []
    for name, value in values.items():
        if _holds_a_set(value):
            error = set_has_no_order()
            problems.append(problem.invalid_value(node_id, name, reason=error.message(), errors=[{"type": error.type, "loc": [], "msg": error.message()}]))
            continue
        try:
            written[name] = plain_json(value)
        except PydanticSerializationError:
            reason = "a value written in a graph record is saved with it as JSON" + kind_of_value(value).typed_in_advice
            problems.append(problem.invalid_value(node_id, name, reason=reason))
    if problems:
        raise RefusedError(problems)
    return written


def _holds_a_set(value: Any) -> bool:
    """Whether ``value`` is a set, or a list, tuple or dict with a set anywhere inside."""
    if isinstance(value, (set, frozenset)):
        return True
    if isinstance(value, dict):
        return any(map(_holds_a_set, value.values()))
    return isinstance(value, (list, tuple)) and any(map(_holds_a_set, value))
