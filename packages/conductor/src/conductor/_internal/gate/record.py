"""The record door: a graph record a host hands in, and the values ``make_graph`` writes into one.

Example: ``Graph.compile('{"nodes": {"shout": {"node": "shout"}}}', registry)`` reads the text with ``read_record``
into a ``GraphRecord``. The text ``'{"nodes": {"shout": {}}}'`` is refused with one ``invalid_record`` problem at
``("record", "nodes", "shout", "node")``: the node names no definition. And ``make_graph(shout, constants={"shout.mark":
{"!"}})`` is refused by ``record_values`` with ``invalid_value`` at ``("shout", "mark")``: a set has no order, so the
record would list it in another order in another process.

What the record says that is wrong as a graph (an edge into an input the node does not have, an edge from a node the
record lacks, a constant its input's type refuses) is not refused: compiling says so in ``graph.problems``, so an
editor can show the graph while it is being drawn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue, ValidationError
from pydantic_core import PydanticSerializationError

from conductor._internal.base import plain_json
from conductor._internal.problem import call_problems, compile_problems
from conductor._internal.record import GraphRecord
from conductor._internal.series import set_has_no_order
from conductor._internal.types import kind_of_value

from .refused import RefusedError


def _each_key_once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The object of ``pairs``, the keys and values of one JSON object as the text has them.

    Raises:
        RefusedError: A key is in it twice, such as the id of two nodes, where a ``dict`` would keep only the last:
            one ``invalid_record`` problem, ``repeated_key``.
    """
    read = dict(pairs)
    if len(read) != len(pairs):
        keys = [key for key, _ in pairs]
        reason = f"the key {next(key for i, key in enumerate(keys) if key in keys[:i])!r} is written twice in one object"
        raise RefusedError([call_problems.invalid_record("record", where="the record", reason=reason, error="repeated_key")])
    return read


def read_json(text: str | bytes) -> Any:
    """Returns JSON text as plain data, with every object's keys seen before they become a dict: the one way a graph
    record's text is read, and a body that holds one.

    ``read_json('{"nodes": {"a": {"node": "upper"}, "a": {"node": "count"}}}')`` is refused: a dict would keep the
    second node ``a`` and lose the first without a word.

    Raises:
        RefusedError: The text is not JSON (``json_invalid``), or writes a key twice in one object (``repeated_key``):
            one ``invalid_record`` problem at ``("record",)``.
    """
    try:
        return json.loads(text, object_pairs_hook=_each_key_once)
    except ValueError as not_json:
        raise RefusedError([call_problems.invalid_record("record", where="the record", reason=f"it is not JSON: {not_json}", error="json_invalid")]) from None


def read_record(record: GraphRecord | str | bytes | Mapping[str, Any], *, written_by: str | None = None) -> GraphRecord:
    """Returns ``record`` as a ``GraphRecord``: a ``GraphRecord`` as it is; JSON text (read by ``read_json``), or the
    plain data ``make_graph`` and ``registry.upgrade`` write, read into one.

    Args:
        record: The record.
        written_by: What wrote it when it is not JSON a host sent, for the refusal to say what to fix:
            ``make_graph``'s arguments, or a node's ``Upgrade``.

    Raises:
        RefusedError: The text is not JSON or writes a key twice in one object, or the record is not a record's
            shape (a key a record does not have, an id or input that is not a name, an input given more than one of
            a constant, a graph input and sources, a source listed twice, a field on a graph output): one
            ``invalid_record`` problem per fault, each at ``("record", *where)``.
    """
    if isinstance(record, GraphRecord):
        return record
    data = read_json(record) if isinstance(record, (str, bytes)) else record
    try:
        return GraphRecord.model_validate(data)
    except ValidationError as invalid:
        raise RefusedError(
            call_problems.invalid_record("record", *error["loc"], where=".".join(map(str, error["loc"])) or "the record", reason=error["msg"], error=error["type"], written_by=written_by)
            for error in invalid.errors()
        ) from None


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
            problems.append(compile_problems.invalid_value(node_id, name, reason=error.message(), errors=[{"type": error.type, "loc": [], "msg": error.message()}]))
            continue
        try:
            written[name] = plain_json(value)
        except PydanticSerializationError:
            reason = "a value written in a graph record is saved with it as JSON" + kind_of_value(value).typed_in_advice
            problems.append(compile_problems.invalid_value(node_id, name, reason=reason))
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
