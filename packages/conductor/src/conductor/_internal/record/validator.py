"""Checking a graph record a host hands in, and the values Python code writes into one.

A host hands conductor a record as JSON text, as plain data or as a ``GraphRecord``. ``RecordValidator`` checks
that it has a record's shape and returns a ``GraphRecord``. Say a host compiles the text
``'{"nodes": {"shout": {}}}'``. The node doesn't say which node it uses, so the record is refused with one
``invalid_record`` problem at ``("record", "nodes", "shout", "node")``.

``WrittenValuesValidator`` checks the Python values that ``make_graph`` and ``registry.upgrade`` write into a
record, and converts them to JSON. A date becomes its ISO text. A set is refused, since it has no order and the
record could list it differently in another process.

Only the shape is checked here. A record can have the right shape and still be wrong as a graph, for example with
an edge from a node it doesn't have. That isn't refused: compiling lists it in ``graph.problems``, so an editor can
still show a graph while a person is drawing it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import polars as pl
from pydantic import JsonValue, ValidationError
from pydantic_core import PydanticSerializationError

from conductor._internal.base import plain_json
from conductor._internal.problem import (
    InvalidCallError,
    call_problems,
    compile_problems,
    refusal_errors,
)
from conductor._internal.validation import Validator, holds_a_set, set_has_no_order

from .graph import GraphRecord


def _each_key_once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Builds one JSON object from its keys and values, and refuses a key written twice.

    ``json.loads`` calls this for each object, with the pairs in the order the text has them. A plain ``dict``
    would quietly keep only the last value of a repeated key. In a record, that would lose a node with the same id.

    Raises:
        InvalidCallError: One ``invalid_record`` problem, ``repeated_key``, if a key is written twice.
    """
    read: dict[str, Any] = {}
    for key, value in pairs:
        if key in read:
            # The object may be a constant's value, so the problem doesn't quote the key.
            raise InvalidCallError(
                [
                    call_problems.invalid_record(
                        "record",
                        where="the record",
                        reason="a key is written twice in one object",
                        error="repeated_key",
                    )
                ]
            )
        read[key] = value
    return read


def read_json(text: str | bytes) -> Any:
    """Parses JSON text into plain data, and refuses a key written twice in one object.

    This is how conductor parses a graph record's text. Plain ``json.loads`` would keep only the last of two equal
    keys. So ``read_json('{"nodes": {"a": {"node": "upper"}, "a": {"node": "count"}}}')`` is refused, where
    ``json.loads`` would drop the first node ``a`` without a word.

    A host whose route takes a graph record in its body can parse the body with it,
    ``body = read_json(await request.body())``, and hand the data on to ``Graph.compile``.

    Raises:
        InvalidCallError: One ``invalid_record`` problem at ``("record",)``. Its error is ``json_invalid`` if the
            text isn't JSON (bytes that aren't UTF-8 included), or ``repeated_key`` if a key is written twice.
    """
    try:
        return json.loads(text, object_pairs_hook=_each_key_once)
    except InvalidCallError:  # a repeated key; also a ValueError, so let through first
        raise
    except ValueError as not_json:
        raise InvalidCallError(
            [
                call_problems.invalid_record(
                    "record",
                    where="the record",
                    reason=f"it isn't JSON ({not_json})",
                    error="json_invalid",
                )
            ]
        ) from None


class RecordValidator(Validator[GraphRecord | str | bytes | Mapping[str, Any], GraphRecord]):
    """A validator for a graph record a host hands in.

    It returns a ``GraphRecord``. A ``GraphRecord`` is returned as it is. JSON text is parsed with ``read_json``.
    Plain data, like what ``make_graph`` and ``registry.upgrade`` write, is checked as it is.

    Raises (when called):
        InvalidCallError: One ``invalid_record`` problem per fault, each at ``("record", *where)``. The faults
            include text that isn't JSON, a key written twice, a key a record doesn't have, an id that isn't a
            name, and an input given more than one kind of entry.
    """

    def __init__(self, written_by: str | None = None) -> None:
        """Builds the validator.

        Args:
            written_by: What wrote the record, when it isn't JSON a host sent: ``make_graph``'s arguments, or a
                node's ``Upgrade``. A refusal names it, so the reader knows what to fix.
        """
        self._written_by = written_by

    def __call__(self, record: GraphRecord | str | bytes | Mapping[str, Any]) -> GraphRecord:
        """Checks ``record`` and returns it as a ``GraphRecord``.

        Raises:
            InvalidCallError: See the class.
        """
        if isinstance(record, GraphRecord):
            return record
        data = read_json(record) if isinstance(record, (str, bytes)) else record
        try:
            return GraphRecord.model_validate(data)
        except ValidationError as invalid:
            raise InvalidCallError(
                call_problems.invalid_record(
                    "record",
                    *error["loc"],
                    where=".".join(map(str, error["loc"])) or "the record",
                    reason=error["msg"],
                    error=error["type"],
                    written_by=self._written_by,
                )
                for error in invalid.errors()
            ) from None


class WrittenValuesValidator(Validator[Mapping[str, Any], dict[str, JsonValue]]):
    """A validator for the constants or defaults Python code writes into a graph record for one node.

    It converts each value to the JSON the record stores. ``{"day": date(2026, 1, 2)}`` becomes
    ``{"day": "2026-01-02"}``. A float that isn't finite becomes its text, ``"NaN"`` or ``"Infinity"``.

    Raises (when called):
        InvalidCallError: One ``invalid_value`` problem at ``(node_id, input)`` for each value that can't be written
            as JSON, or that has a set in it (``set_has_no_order``).
    """

    def __init__(self, node_id: str) -> None:
        """Builds the validator for the values written for one node.

        Args:
            node_id: The node's id. The problems name it.
        """
        self._node_id = node_id

    def __call__(self, values: Mapping[str, Any]) -> dict[str, JsonValue]:
        """Checks ``values`` and returns them as JSON, by input name.

        Raises:
            InvalidCallError: See the class.
        """
        written: dict[str, JsonValue] = {}
        problems = []
        for name, value in values.items():
            if holds_a_set(value):
                error = set_has_no_order()
                problems.append(
                    compile_problems.invalid_value(
                        self._node_id, name, reason=error.message(), errors=refusal_errors(error)
                    )
                )
                continue
            try:
                written[name] = plain_json(value)
            except PydanticSerializationError:
                reason = "it can't be written as JSON, and a graph record is saved as JSON"
                if isinstance(value, pl.DataFrame):
                    reason += " (in a graph record, a table is written as a list of records)"
                problems.append(compile_problems.invalid_value(self._node_id, name, reason=reason))
        if problems:
            raise InvalidCallError(problems)
        return written
