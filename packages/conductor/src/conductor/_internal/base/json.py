"""How conductor writes values as JSON, whatever their type.

Every value conductor saves or sends follows the same few rules, and they are written once, here. JSON has no
number for ``inf`` or ``nan`` and no bytes, so conductor writes them as text (``JSON_CONFIG``). Two values are
compared and hashed by one fixed JSON text. And a value given from Python is checked exactly as if it had come in
as JSON, so it gives the same result as the same value sent over HTTP.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final, Literal, NoReturn

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic_core import (
    CoreConfig,
    PydanticCustomError,
    PydanticSerializationError,
    SchemaValidator,
    from_json,
    to_json,
    to_jsonable_python,
)


class _ReadOnlyConfig(dict[str, Any]):
    """A ``dict`` that refuses every change, so ``JSON_CONFIG`` can't be changed by a host.

    It is a ``dict`` and not a ``MappingProxyType``, because pydantic reads a config only from a ``dict``: a
    pydantic dataclass's ``config=`` and ``TypeAdapter(config=)`` ignore any other mapping.
    """

    def _refuse(self, *_: object, **__: object) -> NoReturn:
        raise TypeError(
            "JSON_CONFIG can't be changed; build your own config from it: ConfigDict(**JSON_CONFIG, ...)"
        )

    __setitem__ = __delitem__ = __ior__ = update = pop = popitem = clear = setdefault = _refuse  # type: ignore[assignment]


#: The pydantic config conductor writes and reads JSON by, and the one a host's own model, dataclass or ``TypedDict``
#: that holds ``bytes`` is built from to follow the same rule: ``ConfigDict(**JSON_CONFIG, ...)``, or
#: ``with_config(JSON_CONFIG)``. Every pydantic config in conductor that writes or reads JSON is built from it, and a
#: call that takes one of its settings, like pydantic's ``to_json``, reads that setting from it. So a value is
#: written one way wherever it goes. It can't be changed: a changed setting would split conductor's own writing in two.
#:
#: - ``inf``, ``-inf`` and ``nan`` are written as the texts ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``. A
#:   ``float`` reads each one back as the float, and a ``str`` keeps the text.
#: - ``bytes`` are written as URL-safe base64 text with its padding: ``b"\xff\xfe"`` is ``"__4="``. ``bytes`` reads
#:   base64 text back, standard or URL-safe, with or without its padding, and refuses any other text.
JSON_CONFIG: Final[ConfigDict] = _ReadOnlyConfig(  # type: ignore[assignment]
    ser_json_inf_nan="strings", ser_json_bytes="base64", val_json_bytes="base64"
)

#: ``JSON_CONFIG`` as pydantic-core takes it, for a check or a schema built from a core schema:
#: ``SchemaValidator(schema, config=JSON_CORE_CONFIG)``, and the ``config`` of a ``typed_dict_schema``, which reads
#: its fields by its own config and not by the one it sits in.
JSON_CORE_CONFIG: Final = CoreConfig(
    ser_json_inf_nan=JSON_CONFIG["ser_json_inf_nan"],
    ser_json_bytes=JSON_CONFIG["ser_json_bytes"],
    val_json_bytes=JSON_CONFIG["val_json_bytes"],
)


def plain_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> JsonValue:
    """Returns ``value`` as plain JSON data: what you get by writing it as JSON text and reading that back.

    The result has only dicts, lists, strings, numbers, booleans and ``None``. For example,
    ``plain_json({"day": date(2026, 1, 2), "sizes": (1, float("inf"))})`` returns
    ``{"day": "2026-01-02", "sizes": [1, "Infinity"]}``.

    Pydantic's ``to_jsonable_python`` would leave ``inf`` as a float, which JSON has no number for. So this writes
    the JSON text and reads it back, and ``inf`` comes out as the text ``"Infinity"`` (``JSON_CONFIG``), in a field
    of a host's own model too (``json_text``). ``bytes`` come out as their base64 text.

    Args:
        value: The value to convert.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead.

    Raises:
        PydanticSerializationError: A value can't be written as JSON and there is no ``fallback``, or ``fallback``
            raised.
    """
    plain: JsonValue = from_json(_written(value, JSON_CONFIG["ser_json_inf_nan"], fallback))
    return plain


def canonical_json(value: Any) -> str:
    """Returns one fixed JSON text for ``value``, so that two values can be compared and hashed by their text.

    The keys of every dict are sorted, and there are no spaces. So ``{"b": 1, "a": [2]}`` and
    ``{"a": [2], "b": 1}`` both give ``'{"a":[2],"b":1}'``. Two values are the same JSON when their texts are equal.

    ``value`` should already be plain data, as ``plain_json`` or pydantic's ``to_python(mode="json")`` returns it.
    The second leaves ``inf`` as a float, so a float that isn't finite is written as its text here too
    (``JSON_CONFIG``). That way ``float("inf")`` and ``"Infinity"`` give the same text.
    """
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except ValueError:  # inf or nan, which JSON has no number for
        return canonical_json(plain_json(value))


def validate_as_json(
    validator: SchemaValidator | TypeAdapter[Any],
    value: Any,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Checks a Python value strictly, the way ``validator`` checks the same value sent as JSON text.

    This is what makes a value given from Python and the same value sent over HTTP come out the same. A ``bool``
    takes ``True`` but refuses ``"yes"``. A ``date`` takes both ``date(2026, 1, 2)`` and ``"2026-01-02"``. A
    ``tuple`` takes a list.

    Pydantic has no call that checks a Python value by the rules for JSON input. ``validate_python(strict=True)``
    refuses a date given as text, and ``validate_strings`` takes ``"yes"`` as ``True``. So the value is written as
    JSON text first (``json_text``), and then the text is checked.

    Args:
        validator: Pydantic's validator for the type.
        value: The value to check.
        fallback: As ``json_text`` takes it.

    Returns:
        The value as ``validator`` returns it.

    Raises:
        PydanticSerializationError: As ``json_text`` raises it.
        PydanticCustomError: ``fallback`` refused the value.
        ValidationError: The type refuses the value.
    """
    return validator.validate_json(json_text(value, fallback), strict=True)


def json_text(value: Any, fallback: Callable[[Any], Any] | None = None) -> str:
    """Writes a Python value as the JSON text it is checked from (``validate_as_json``).

    ``inf`` and ``nan`` are written as the bare words ``Infinity`` and ``NaN``, which pydantic reads back as floats.
    ``bytes`` are written as their base64 text (``JSON_CONFIG``).

    Args:
        value: The value to write.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead, or raises
            ``PydanticCustomError`` to refuse the value. That error is raised from here unchanged.

    Raises:
        PydanticSerializationError: The value can't be written as JSON, and ``fallback`` gave nothing to write
            instead.
        PydanticCustomError: ``fallback`` refused the value.
    """
    return _written(value, "constants", fallback).decode()


def finite_json_text(value: Any, title: str, fallback: Callable[[Any], Any] | None = None) -> str:
    """Writes a Python value as its JSON text (``json_text``), refusing a number in it that isn't finite.

    This is the one rule that a number entering a graph is finite: a caller's input, an answer, a record's constant
    or default, a node's own default and a choice. ``json_text`` writes ``inf`` and ``nan`` as the bare words
    ``Infinity`` and ``NaN``, and pydantic's own parser refuses them with ``allow_inf_nan=False``, at any depth, so a
    value of any type, ``Any`` and a host's model included, is held to it. Their texts, ``"Infinity"`` and
    ``"NaN"``, are text, which it takes. What a node computes may hold them.

    Args:
        value: The value to write.
        title: The title of the refusal, such as the type's id.
        fallback: As ``json_text`` takes it.

    Raises:
        ValidationError: Pydantic's own ``finite_number``, as its float check raises it. The parser says only where
            in the text, so its ``loc`` is the value's own.
        PydanticSerializationError: As ``json_text`` raises it.
        PydanticCustomError: ``fallback`` refused the value.
    """
    text = json_text(value, fallback)
    try:
        from_json(text, allow_inf_nan=False)
    except ValueError:
        raise ValidationError.from_exception_data(
            title, [{"type": "finite_number", "loc": (), "input": value}]
        ) from None
    return text


def _written(
    value: Any,
    non_finite: Literal["null", "constants", "strings"],
    fallback: Callable[[Any], Any] | None,
) -> bytes:
    """Writes ``value`` as JSON text, with ``inf`` and ``nan`` written as ``non_finite`` says, wherever they are, and
    ``bytes`` as base64 (``JSON_CONFIG``).

    A pydantic model writes its own floats by its own config, whatever ``to_json`` is told, and its config writes
    ``inf`` as ``null`` by default. So a host's model with ``x=inf`` would be written as
    ``{"x": null}``, and a ``float | None`` field would read back as ``None``. The value is turned into Python's
    JSON values first (``to_jsonable_python``), which keep a float as it is, each model by its own serializer, and
    then written as text by one rule. A model writes its ``bytes`` by its own config too, so a host's model with a
    ``bytes`` field sets ``ser_json_bytes`` and ``val_json_bytes`` itself.

    What can't be written is refused as ``to_json`` refuses it, with ``PydanticSerializationError``: a cycle, a
    ``str`` with a lone surrogate, or a value ``fallback`` raises ``TypeError`` for. A ``PydanticCustomError`` that
    ``fallback`` raises to refuse a value is raised unchanged.
    """
    try:
        data = to_jsonable_python(
            value, fallback=fallback, bytes_mode=JSON_CONFIG["ser_json_bytes"]
        )
    except (PydanticCustomError, PydanticSerializationError):
        raise
    # what to_json raises as PydanticSerializationError, to_jsonable_python raises as it is
    except (TypeError, ValueError) as error:
        raise PydanticSerializationError(str(error)) from None
    return to_json(data, inf_nan_mode=non_finite)
