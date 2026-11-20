"""Where a pydantic core schema keeps the schemas inside it.

A core schema is a dict with a ``"type"``, like ``{"type": "list", "items_schema": {"type": "int"}}``. Some of its
keys hold the schemas of the values inside it: a list's items, a dict's keys and values, a union's choices, a
class's fields. Every other key, like a field's ``default``, its ``serialization`` or its ``metadata``, holds values,
not schemas. Conductor walks a type's schema to add its rules (``types/rules.py``), to check what a node declares
(``Type.check_declared``) and to see whether a saved value holds a model (``run/json/codec.py``). Each walk asks
this module which parts of a schema are schemas, so that rule is written once.

Pydantic writes a type that refers to itself, like a tree, as a ``definitions`` schema: the type's schema, with the
schemas it refers to beside it, each by its ``ref``. A ``definition-ref`` schema then names one of them by its
``schema_ref``. ``under_definitions`` and ``with_definitions`` take that wrapper off and put it back.
``under_all_definitions`` takes it off several forms of one value that are joined into one schema. ``looked_up``
finds the schema a reference names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from pydantic_core import CoreSchema, core_schema

#: The core schema types of a host's own class: a pydantic model, or a pydantic or plain dataclass. Pydantic checks
#: and writes their fields by the class's own config, not by the config of the schema they sit in.
HOSTS_OWN_CLASSES = frozenset({"model", "dataclass"})

#: The keys of a core schema whose values are schemas, besides each key that ends in ``_schema``, like
#: ``items_schema``: a wrapped schema, a chain's steps, a union's choices and a class's fields. ``definitions`` is not
#: one of them: what it declares is reached through each ``definition-ref`` to it.
_SCHEMA_KEYS = frozenset({"schema", "steps", "choices", "fields"})


def _holds_schemas(key: str) -> bool:
    """Whether the key of a core schema holds a schema, or a list or mapping of them."""
    return key in _SCHEMA_KEYS or key.endswith("_schema")


def schemas_in(schema: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yields each schema directly inside a core schema, with the name of the field it is, if it is one.

    For ``{"type": "list", "items_schema": {"type": "int"}}`` it yields ``({"type": "int"}, None)``. For a
    ``TypedDict`` with the field ``name: str``, it yields the field's ``typed-dict-field`` schema with ``"name"``. A
    model's and a dataclass's fields are named the same way. Each schema is yielded as it is: a
    ``definition-ref`` is not looked up, and the schemas a ``definitions`` schema declares are not yielded.

    It gives the field's name, not the key, because a walk that reads a schema names the place it found a fault by
    the field a person wrote (``Tags.tags``). ``with_schemas_in`` gives the key instead, because a walk that copies
    a schema decides by the key what to leave alone, like a dict's ``keys_schema``.
    """
    for key, value in schema.items():
        if not _holds_schemas(key):
            continue
        # most keys hold one schema, which is yielded here, without a generator for it
        if type(value) is dict and type(value.get("type")) is str:
            yield value, None
        else:
            yield from _found(value, is_fields=key == "fields")


def _found(value: Any, *, is_fields: bool) -> Iterator[tuple[dict[str, Any], str | None]]:
    """Yields each schema in the value of a key that holds schemas: a schema, or a list, tuple or mapping of them.

    A union's choice may be a pair of a schema and its label, and a function's arguments are dicts with a name and
    a schema, so a list's items and a mapping's values are searched the same way.

    Args:
        value: The value of one key that holds schemas.
        is_fields: Whether the key is ``fields``, so that each schema in it is a class's field.
    """
    if isinstance(value, (list, tuple)):
        for one in value:
            if is_fields and isinstance(one, dict):
                # a dataclass's fields are a list, each with its name on it; the dataclass lists their names as text
                yield one, one["name"]
            else:
                yield from _found(one, is_fields=False)
    elif isinstance(value, dict):
        if isinstance(value.get("type"), str):
            yield value, None
            return
        # schemas by name, not a schema, and one may be named "type"
        for name, one in value.items():
            if is_fields:
                yield one, name
            else:
                yield from _found(one, is_fields=False)


def with_schemas_in(
    schema: Mapping[str, Any], replace: Callable[[dict[str, Any], str], Any]
) -> dict[str, Any]:
    """Returns a copy of a core schema with each schema directly inside it replaced by ``replace``.

    The schemas are the ones ``schemas_in`` yields, in the same places. Every other key is kept as it is, and
    ``schema`` itself is left unchanged.

    It gives ``replace`` the key, not the field's name as ``schemas_in`` does, because a walk that copies a
    schema leaves some keys' schemas as they are, like a dict's ``keys_schema``, and never names a field.

    Args:
        schema: The core schema to copy.
        replace: Called with each schema inside ``schema`` and the key it is under, like ``"items_schema"``. It
            returns the schema to put in its place.
    """
    return {
        key: _replaced(value, key, replace) if _holds_schemas(key) else value
        for key, value in schema.items()
    }


def _replaced(value: Any, key: str, replace: Callable[[dict[str, Any], str], Any]) -> Any:
    """Returns the value of the key ``key`` with each schema in it replaced, found as ``_found`` finds them."""
    if isinstance(value, list):
        return [_replaced(one, key, replace) for one in value]
    if isinstance(value, tuple):
        return tuple(_replaced(one, key, replace) for one in value)
    if isinstance(value, dict):
        if isinstance(value.get("type"), str):
            return replace(value, key)
        return {name: _replaced(one, key, replace) for name, one in value.items()}
    return value


def under_definitions(
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Returns the schema inside a ``definitions`` schema, and the schemas it declares, by their ``ref``.

    Any other schema is returned as it is, with no definitions.
    """
    if schema["type"] == "definitions":
        return schema["schema"], {one["ref"]: one for one in schema["definitions"]}
    whole: dict[str, Any] = schema  # type: ignore[assignment]  # a core schema is a dict
    return whole, {}


def under_all_definitions[K](
    schemas: Mapping[K, Mapping[str, Any]],
) -> tuple[dict[K, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Returns each schema as ``under_definitions`` does, and the schemas they declare between them, by their ``ref``.

    It is for one schema built from the schemas of several forms of one value, like a ``Series`` of rows given as a
    list or as a table. Pydantic refuses a schema that declares one ``ref`` twice, and both forms declare a model
    that a row's cell holds. So the joined schema declares each one once, around the whole (``with_definitions``).

    A ``ref`` names a class, but not the config it was built under. A plain dataclass or a ``TypedDict`` takes the
    config it sits in, so one class built under two configs gives two schemas under one ``ref``. The forms joined
    here are built from one type under one config, so their schemas under a ``ref`` are the same. Two that differ
    are a fault in how conductor built them, and raise.

    Args:
        schemas: The schemas to join, each under its own key.

    Raises:
        AssertionError: Two of the schemas declare different schemas under one ``ref``.
    """
    inner: dict[K, dict[str, Any]] = {}
    definitions: dict[str, dict[str, Any]] = {}
    for key, schema in schemas.items():
        inner[key], declared = under_definitions(schema)
        for ref, definition in declared.items():
            if definitions.setdefault(ref, definition) != definition:
                raise AssertionError(f"two different schemas are declared under the ref {ref!r}")
    return inner, definitions


def looked_up(schema: Mapping[str, Any], definitions: Mapping[str, Any]) -> Any:
    """Returns the schema a ``definition-ref`` names, from ``definitions``, or any other schema as it is.

    It is ``None`` for a reference to a schema ``definitions`` doesn't declare.
    """
    if schema["type"] == "definition-ref":
        return definitions.get(schema["schema_ref"])
    return schema


def with_definitions(schema: CoreSchema, definitions: Mapping[str, Any]) -> CoreSchema:
    """Returns ``schema`` with ``definitions`` declared beside it, the way ``under_definitions`` found them.

    With no definitions, it is ``schema`` itself.
    """
    if not definitions:
        return schema
    return core_schema.definitions_schema(schema, list(definitions.values()))
