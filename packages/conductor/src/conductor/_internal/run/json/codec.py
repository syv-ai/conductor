"""Builds, for one graph, the pydantic model its saved runs are read and written by: the graph's codec.

``type.py`` declares every key of a saved run, with what is true of every graph. ``run_codec`` makes a subclass of
each of its classes for one graph, whose fields narrow those to what only the graph knows: the names of its nodes,
inputs and outputs, and the type of each value.
``saved_type`` is the type of an output's value on one call, and ``where_in`` names the place of an error in a
saved run as text.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import is_dataclass
from types import GenericAlias, MappingProxyType, NoneType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, GetCoreSchemaHandler, NonNegativeInt, create_model
from pydantic import Field as PydanticField
from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import place_text
from conductor._internal.field import Field
from conductor._internal.gate import numbers_refuse_bools
from conductor._internal.graph import Graph
from conductor._internal.types import Type, union_of

from ..blueprint import RunBlueprint
from .type import SavedOutcomes, SavedReusable, SavedRun


def saved_type(output: Field) -> Type:
    """The type of ``output``'s value on one call, as a saved run and ``run.outputs_to_json()`` write it.

    For ``shout.result``, a ``str``, it is ``Type(str | None)``: ``None`` where the call has no outputs. For a
    ``Series`` it is the form a run stores the list in (``SeriesItems``, or ``Records`` for a table), or ``None``.
    """
    return Type(union_of((output.whole.stored_hint, NoneType)))


class _SavedValue:
    """The hint a saved run declares a value of one type by: a graph input, an output's value on one call, or a
    person's answer. Pydantic reads the value by the type's own schema with ``numbers_refuse_bools`` put in. That rule
    holds for a Python value; read from a saved run's JSON, an ``int`` or a ``float`` still reads ``true`` as 1.

    ``inf`` is written as ``"Infinity"`` by the codec's config, which a ``float`` reads back as the float. A pydantic
    model or a dataclass writes its fields by its own config, whatever schema it is inside, and by default writes
    ``inf`` as ``null``, which a ``float`` field does not read back. So a type with one in it is written as
    ``Type.to_json`` writes it: first as Python's JSON values, which keep the float, then as text by the codec's
    config.

    The class is a hint, not ``GetPydanticSchema`` in ``Annotated``: pydantic moves the definitions of a schema a hint
    returns beside the whole schema, and leaves them in place for one that ``Annotated`` returns, where a reference
    from elsewhere cannot reach them.

    Attributes:
        schema: The core schema pydantic reads and writes the value by.
    """

    __slots__ = ("schema",)

    def __init__(self, value_type: Type) -> None:
        schema = numbers_refuse_bools(value_type.schema)
        if _holds_a_model(schema):
            as_json = core_schema.plain_serializer_function_ser_schema(functools.partial(value_type.serializer.to_python, mode="json"))
            if schema["type"] == "definitions":  # what pydantic defined beside the type stays beside it
                schema = core_schema.definitions_schema(core_schema.chain_schema([schema["schema"]], serialization=as_json), schema["definitions"])
            else:
                schema = core_schema.chain_schema([schema], serialization=as_json)
        self.schema = schema

    def __get_pydantic_core_schema__(self, _source: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:
        return self.schema


def _holds_a_model(schema: Any) -> bool:
    """Says whether a core schema has a pydantic model or a dataclass anywhere in it."""
    if isinstance(schema, dict):
        return schema.get("type") in ("model", "dataclass") or any(map(_holds_a_model, schema.values()))
    return isinstance(schema, list) and any(map(_holds_a_model, schema))


class _ExactKeys:
    """The hint of a JSON object that must have every key of ``required``, may have the keys of ``optional``, and may
    have no other key; each key's value is read by its own hint.

    ``_ExactKeys({}, {"lines": a, "shout": b})`` accepts ``{}`` and ``{"shout": ...}`` and refuses ``{"other": ...}``.
    Pydantic is asked for the schema of each distinct hint once, and every key with that hint is given that schema:
    a graph of 500 nodes has three such objects of 500 keys, nearly all with one hint.

    Attributes:
        required: The hint of each key the object must have.
        optional: The hint of each key the object may have.
    """

    __slots__ = ("optional", "required")

    def __init__(self, required: Mapping[str, Any], optional: Mapping[str, Any] = MappingProxyType({})) -> None:
        self.required, self.optional = required, optional

    def __get_pydantic_core_schema__(self, _source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        hints = {id(hint): hint for hint in (*self.required.values(), *self.optional.values())}  # each alive in required or optional
        schemas = {key: handler.generate_schema(hint) for key, hint in hints.items()}
        fields = {name: core_schema.typed_dict_field(schemas[id(hint)]) for name, hint in self.required.items()}
        fields.update((name, core_schema.typed_dict_field(schemas[id(hint)], required=False)) for name, hint in self.optional.items())
        return core_schema.typed_dict_schema(fields, extra_behavior="forbid")


def run_codec(
    fingerprint: str, inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], nodes: Iterable[str],
) -> type[SavedRun]:
    """Builds the codec of one graph: a subclass of ``SavedRun`` whose fields have the graph's own keys and types.

    For the README graph ``lines -> shout -> join``, it reads only a saved run whose ``graph`` is the graph's
    fingerprint, whose ``inputs`` has the one key ``text`` with a ``str``, and whose ``outcomes``,
    ``reusable`` and ``code`` have no key but ``lines``, ``shout`` and ``join``. Under ``shout``, ``outputs`` has
    the one key ``result``, a list of ``str | None``.

    Each node's outcomes are a subclass of ``SavedOutcomes`` with the node's outputs as the keys of ``outputs``,
    ``skipped`` and ``skipped_items``, and its reusable outcomes a subclass of ``SavedReusable`` whose ``outcomes``
    is that class; nodes whose outputs have the same names and types share both. The codec keeps them by node in
    ``outcomes_classes`` and ``reusable_classes``, and a run is written with them, so a run written and the run read
    back from it are of the same classes. Each class's name ends with the graph's fingerprint.

    Args:
        fingerprint: The graph's fingerprint.
        inputs: The type of each graph input, by name.
        outputs: The outputs of each function node, by the node's name in a saved run.
        nodes: The name in a saved run of every function node.
    """
    # by what the classes of a node depend on: for each output its name, its type and whether it is a Series
    of_same_outputs: dict[tuple[tuple[str, Type, bool], ...], tuple[type[SavedOutcomes], type[SavedReusable]]] = {}
    outcomes_classes, reusable_classes = {}, {}
    for node, node_outputs in outputs.items():
        value_types = {name: saved_type(output) for name, output in node_outputs.items()}
        same_outputs = tuple((name, value_types[name], output.is_series) for name, output in node_outputs.items())
        if same_outputs not in of_same_outputs:
            suffix = f"{len(of_same_outputs)}_{fingerprint}"
            fields_of_node: dict[str, Any] = {  # hints of several kinds: pydantic's create_model takes any hint
                "outputs": _ExactKeys({name: GenericAlias(list, (_SavedValue(value_type),)) for name, value_type in value_types.items()}),
                "skipped": (_ExactKeys({}, dict.fromkeys(node_outputs, list[NonNegativeInt])), PydanticField(default_factory=dict)),
                "skipped_items": (_ExactKeys({}, {name: dict[NonNegativeInt, list[NonNegativeInt]] for name, output in node_outputs.items() if output.is_series}), PydanticField(default_factory=dict)),
            }
            of_node = create_model(f"SavedOutcomes_{suffix}", __base__=SavedOutcomes, **fields_of_node)
            of_same_outputs[same_outputs] = of_node, create_model(f"SavedReusable_{suffix}", __base__=SavedReusable, outcomes=of_node)
        outcomes_classes[node], reusable_classes[node] = of_same_outputs[same_outputs]
    names = tuple(nodes)
    fields_of_run: dict[str, Any] = {  # as fields_of_node
        "graph": Literal[fingerprint],
        "inputs": _ExactKeys({name: _SavedValue(of_input) for name, of_input in inputs.items()}),
        "outcomes": _ExactKeys({}, outcomes_classes),
        "reusable": _ExactKeys({}, reusable_classes),
        "code": _ExactKeys({}, dict.fromkeys(outcomes_classes, str)),
        "complete": list[Literal[names]] if names else Annotated[list[str], PydanticField(max_length=0)],  # type: ignore[valid-type]  # a Literal of the node names, known once the graph is compiled
    }
    namespace = {"outcomes_classes": outcomes_classes, "reusable_classes": reusable_classes}
    return create_model(f"SavedRun_{fingerprint}", __base__=SavedRun, __namespace__=namespace, **fields_of_run)


def where_in(codec: type[SavedRun], loc: Sequence[str | int]) -> str:
    """Returns the place pydantic names in a saved run, ``loc``, as text, up to the value found there and no step
    inside the value, since a step inside it may be a key of the value's own dict; ``"the run"`` for an empty one.

    ``("outcomes", "g", "outputs", "result", 0, "counts", "ida")`` gives ``"outcomes.g.outputs.result.0"``: the
    value of ``g.result`` on call 0 is ``{"counts": {"ida": ...}}``. ``("inputs", "text")`` and
    ``("outcomes", "g", "states", 0)`` are kept whole. The steps are walked through the fields of ``codec``, the
    graph's codec: a value is wherever a field declares a value's hint (``_SavedValue``, ``Any``, ``str``, ...).
    """
    kept: list[str] = []
    hint: Any = codec
    for step in loc:
        hint = _inside(hint, step)
        if hint is None:
            break
        kept.append(str(step))
    return ".".join(kept) or "the run"


def _inside(hint: Any, step: str | int) -> Any:
    """Returns the hint that ``step`` leads to inside ``hint``, one of the hints of a codec's fields; ``None`` when
    ``hint`` is a value's. A step a class or an exact-keys object does not declare leads to ``Any``."""
    if get_origin(hint) in (Union, UnionType):  # an optional key, ``X | None``: what it is when it is there
        (hint,) = (member for member in get_args(hint) if member is not NoneType)
    if isinstance(hint, _ExactKeys):
        return {**hint.required, **hint.optional}.get(str(step), Any)
    origin, args = get_origin(hint) or hint, get_args(hint)
    if (fields_of_class := _fields_of(origin)) is not None:
        return fields_of_class.get(str(step), Any)
    if origin in (dict, Mapping):
        return args[1]
    if origin is list:
        return args[0]
    return None


def _fields_of(hint: Any) -> dict[str, Any] | None:
    """Returns the hint of each field of ``hint`` by name when it is a pydantic model or a dataclass (a codec's
    class, or ``Unanswered`` inside one), else ``None``."""
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        return {name: field.annotation for name, field in hint.model_fields.items()}
    if isinstance(hint, type) and is_dataclass(hint):
        return get_type_hints(hint)
    return None


def codec_of(graph: Graph[Any], blueprint: RunBlueprint) -> type[SavedRun]:
    """The model the runs of ``graph`` are read from JSON and written as JSON by. Built the first time one of its runs
    is read or written, and kept by the ``Graph`` (``graph._codec``).

    Args:
        graph: The graph.
        blueprint: Its blueprint.
    """
    return run_codec(
        graph.fingerprint,
        inputs={name: field.whole for name, field in graph._inputs.items()},
        outputs={place_text(path): node.compiled_node.outputs for path, node in blueprint.function_nodes.items()},
        nodes=[place_text(path) for path in blueprint.function_nodes],
    )
