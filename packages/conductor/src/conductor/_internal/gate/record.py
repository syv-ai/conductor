"""The record door: a graph record a host hands in, and the values ``make_graph`` writes into one.

Example: ``Graph.compile('{"nodes": [{"id": "shout", "node": "shout"}]}', registry)`` reads the text with
``read_record`` into a ``GraphRecord``. The text ``'{"nodes": [{"id": "shout"}]}'`` is refused with one
``invalid_record`` problem at ``("record", "nodes", 0, "node")``: the node names no definition. And
``make_graph(shout, constants={"shout.mark": {"!"}})`` is refused by ``record_values`` with ``invalid_value`` at
``("shout", "mark")``: a set has no order, so the record would list it in another order in another process.

Only the record's shape is read here. A record of the right shape can still be wrong as a graph (a node id with a
``.`` in it, an edge into an input the node does not have, a constant its input's type refuses): compiling it says
so in ``graph.problems``, so an editor can show the graph while it is being drawn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue, ValidationError
from pydantic_core import PydanticSerializationError

from conductor._internal.base import plain_json
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


def _refusal(invalid: ValidationError) -> RefusedError:
    """One ``invalid_record`` problem for each fault pydantic found in a graph record."""
    return RefusedError(
        problem.invalid_record("record", *error["loc"], where=".".join(map(str, error["loc"])) or "the record", reason=error["msg"].removeprefix("Value error, "), error=error["type"])
        for error in invalid.errors(include_url=False, include_input=False)  # a validator's own sentence, without pydantic's prefix
    )


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
