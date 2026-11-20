"""The codec of a graph: the pydantic model its saved runs are written and read with.

``type.py`` declares every key of a saved run, with what holds for every graph. ``run_codec`` makes a subclass of
each class there for one graph, with fields narrowed to what only that graph knows: the names of its nodes,
inputs and outputs, and the type of each value. Reading a saved run through the codec checks all of that in
one pass.

``saved_type`` is the type an output's value is saved as, and ``where_in`` turns the place of an error in a
saved run into text.
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

from conductor._internal.base import (
    HOSTS_OWN_CLASSES,
    json_typed_dict_schema,
    place_text,
    schemas_in,
    under_definitions,
    with_definitions,
    with_schemas_in,
)
from conductor._internal.graph import Graph
from conductor._internal.port import Port
from conductor._internal.types import Type, union_of

from .type import SavedOutcomes, SavedReusable, SavedRun


def saved_type(output: Port) -> Type:
    """Returns the type ``output``'s value on one call is saved as, in a saved run and by ``run.outputs_to_json()``.

    For ``shout.result``, a ``str``, it is ``Type(str | None)``, with ``None`` for a call that has no outputs. For a
    ``Series`` it is how a run stores the list (``SeriesItems``, or ``SeriesRows`` for a table), or ``None``.
    """
    return Type(union_of((output.whole.stored_hint, NoneType)))


class _SavedValue:
    """The hint a saved run declares one value with: a graph input, or an output's value on one call.

    A saved run is read strictly, as a value that enters a graph is (``read_as_argument``, ``read_as_output``). A
    graph input is checked by its type's argument check (``Type.argument_schema``), and an output's value by its
    output check (``Type.output_schema``), the one a node's return gets. So ``true`` saved for an ``int`` is refused,
    and so is ``"Yes"`` saved for an output whose type a deploy made a ``bool``. ``_read_strictly`` says how.

    A pydantic model or a dataclass writes its fields by its own config, whatever schema it sits in. By default that
    writes ``inf`` as ``null``, which a ``float`` field can't read back. So a type with a model or dataclass inside is
    written the way ``Type.to_json`` writes it: first as Python's JSON values, which keep the float, then as text by
    the codec's config.

    This is a class used as the hint, rather than ``GetPydanticSchema`` inside ``Annotated``. Pydantic moves the
    definitions of a schema a hint returns up beside the whole schema. For one that ``Annotated`` returns, it leaves
    them in place, where a reference from elsewhere can't reach them.

    Attributes:
        schema: The core schema pydantic checks and writes the value with.
    """

    __slots__ = ("schema",)

    def __init__(self, value_type: Type, *, output: bool) -> None:
        """Builds the hint of a value of ``value_type``: an output's value when ``output``, else a graph input."""
        schema = _read_strictly(value_type.output_schema if output else value_type.argument_schema)
        if _holds_a_model(schema):
            as_json = core_schema.plain_serializer_function_ser_schema(
                functools.partial(value_type.serializer.to_python, mode="json")
            )
            # what pydantic defined beside the type stays beside it
            inner, definitions = under_definitions(schema)
            schema = with_definitions(
                core_schema.chain_schema([inner], serialization=as_json), definitions
            )
        self.schema = schema

    def __get_pydantic_core_schema__(
        self, _source: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return self.schema


#: The texts the codec's config writes a float that isn't finite as (``JSON_CONFIG``).
_NOT_FINITE = ["NaN", "Infinity", "-Infinity"]


def _read_strictly(schema: Mapping[str, Any]) -> Any:
    """Returns a copy of a value's core schema that a saved run reads strictly, and that reads a float that isn't
    finite back from its text.

    The codec's config is strict, and so is the config of each ``TypedDict`` inside the value here, since a
    ``TypedDict`` reads its fields by its own config. A host's own model or dataclass is left as it is: pydantic
    reads it by the validator its class keeps, so by its own config, as conductor's rules leave its fields to it
    (``with_rules``).

    Read strictly, a ``float`` takes only a number, but the codec writes ``inf`` as ``"Infinity"``. So each
    ``float`` here first reads a number strictly, or one of ``"NaN"``, ``"Infinity"`` and ``"-Infinity"`` laxly, as
    the float it stands for. Any other text is refused: ``"3"`` is no ``float``. Then the float is checked by the
    ``float`` as declared, so a limit names itself: ``"-Infinity"`` under ``ge=0`` is ``greater_than_equal``. A lax
    read is a weaker match than a strict one, so beside ``str`` in a union, ``"NaN"`` is the text, as a node
    returned it.

    A table's cells, and a ``Series`` of rows saved as a table, are read by the table's own reader, which doesn't
    read them strictly yet: once a deploy makes a column a ``bool``, a saved ``"Yes"`` there still reads as ``True``.
    """
    kind = schema["type"]
    if kind in HOSTS_OWN_CLASSES:
        return schema
    if kind == "float":
        # the chain stands for the float in the rest of the schema: it takes the float's ref and how it is written
        as_declared: Any = {key: value for key, value in schema.items() if key != "ref"}
        a_float = core_schema.union_schema(
            [
                core_schema.float_schema(),
                core_schema.chain_schema(
                    [
                        core_schema.literal_schema(_NOT_FINITE),
                        core_schema.float_schema(strict=False),
                    ]
                ),
            ]
        )
        return core_schema.chain_schema(
            [a_float, as_declared],
            ref=schema.get("ref"),
            serialization=schema.get("serialization", core_schema.simple_ser_schema("float")),
        )
    if kind == "definitions":
        inner, declared = under_definitions(schema)
        # the wrapper is copied, so any other key it has, like its metadata, is kept
        return {
            **schema,
            "schema": _read_strictly(inner),
            "definitions": [_read_strictly(one) for one in declared.values()],
        }
    walked = with_schemas_in(schema, lambda inside, _key: _read_strictly(inside))
    if "config" in walked:
        walked["config"] = {**walked["config"], "strict": True}
    return walked


def _holds_a_model(schema: CoreSchema) -> bool:
    """Returns whether a core schema has a pydantic model or a dataclass anywhere inside it.

    A value is written by its ``serialization`` schema where it has one, so a model there counts too, as in a
    ``PlainSerializer`` whose ``return_type`` is a model.
    """
    return (
        schema["type"] in HOSTS_OWN_CLASSES
        or any(_holds_a_model(one) for one, _ in schemas_in(schema))
        or any(map(_holds_a_model, under_definitions(schema)[1].values()))
        or ("serialization" in schema and _holds_a_model(schema["serialization"]))
    )


class _ExactKeys:
    """The hint of a JSON object with a fixed set of keys, each checked by its own hint.

    The object must have every key of ``required``, may have the keys of ``optional``, and may have no other key.
    ``_ExactKeys({}, {"lines": a, "shout": b})`` accepts ``{}`` and ``{"shout": ...}``, and refuses
    ``{"other": ...}``.

    Pydantic builds the schema of each distinct hint once, and every key with that hint shares it. That matters for
    size: a graph of 500 nodes has three such objects of 500 keys, nearly all with one hint.

    Attributes:
        required: The hint of each key the object must have.
        optional: The hint of each key the object may have.
    """

    __slots__ = ("optional", "required")

    def __init__(
        self, required: Mapping[str, Any], optional: Mapping[str, Any] = MappingProxyType({})
    ) -> None:
        self.required, self.optional = required, optional

    def __get_pydantic_core_schema__(
        self, _source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        hints = {
            id(hint): hint for hint in (*self.required.values(), *self.optional.values())
        }  # keyed by id, which is safe: each hint stays alive in required or optional
        schemas = {key: handler.generate_schema(hint) for key, hint in hints.items()}
        fields = {
            name: core_schema.typed_dict_field(schemas[id(hint)])
            for name, hint in self.required.items()
        }
        fields.update(
            (name, core_schema.typed_dict_field(schemas[id(hint)], required=False))
            for name, hint in self.optional.items()
        )
        schema = json_typed_dict_schema(fields, extra_behavior="forbid")
        # a typed dict reads its fields by its own config, and a saved run is read strictly
        schema["config"] = {**schema.get("config", {}), "strict": True}
        return schema


def run_codec(
    fingerprint: str,
    inputs: Mapping[str, Type],
    outputs: Mapping[str, Mapping[str, Port]],
    nodes: Iterable[str],
) -> type[SavedRun]:
    """Builds the codec of one graph: a subclass of ``SavedRun`` whose fields have the graph's own keys and types.

    For the README graph ``lines -> shout -> join``, the codec only accepts a saved run whose ``graph`` is this
    graph's fingerprint. Its ``inputs`` must have the one key ``text``, holding a ``str``. Its ``outcomes`` and
    ``reusable`` may have no keys but ``lines``, ``shout`` and ``join``. Under ``shout``, ``outputs`` has
    the one key ``result``, a list of ``str | None``.

    Each node gets a subclass of ``SavedOutcomes``, whose ``outputs``, ``skipped`` and ``skipped_items`` are keyed by
    the node's outputs. It also gets a subclass of ``SavedReusable`` with the same fields, and ``item_indexes``.
    Nodes whose outputs have the same names and types share both classes. The codec keeps them by node in
    ``outcomes_classes`` and ``reusable_classes``, and a run is written with them. So a run written and the run read
    back from it use the same classes. Each class's name ends with the graph's fingerprint.

    Args:
        fingerprint: The graph's fingerprint.
        inputs: The type of each graph input, by name.
        outputs: The outputs of each function node, by the node's name in a saved run.
        nodes: The name in a saved run of every function node.
    """
    # keyed by what a node's classes depend on: each output's name, its type, and whether it is a Series
    of_same_outputs: dict[
        tuple[tuple[str, Type, bool], ...], tuple[type[SavedOutcomes], type[SavedReusable]]
    ] = {}
    outcomes_classes, reusable_classes = {}, {}
    for node, node_outputs in outputs.items():
        value_types = {name: saved_type(output) for name, output in node_outputs.items()}
        same_outputs = tuple(
            (name, value_types[name], output.is_series) for name, output in node_outputs.items()
        )
        if same_outputs not in of_same_outputs:
            suffix = f"{len(of_same_outputs)}_{fingerprint}"
            fields_of_node: dict[
                str, Any
            ] = {  # hints of several kinds: pydantic's create_model takes any hint
                "outputs": _ExactKeys(
                    {
                        name: GenericAlias(list, (_SavedValue(value_type, output=True),))
                        for name, value_type in value_types.items()
                    }
                ),
                "skipped": (
                    _ExactKeys({}, dict.fromkeys(node_outputs, list[NonNegativeInt])),
                    PydanticField(default_factory=dict),
                ),
                "skipped_items": (
                    _ExactKeys(
                        {},
                        {
                            name: dict[NonNegativeInt, list[NonNegativeInt]]
                            for name, output in node_outputs.items()
                            if output.is_series
                        },
                    ),
                    PydanticField(default_factory=dict),
                ),
            }
            of_node = create_model(
                f"SavedOutcomes_{suffix}", __base__=SavedOutcomes, **fields_of_node
            )
            of_same_outputs[same_outputs] = (
                of_node,
                create_model(f"SavedReusable_{suffix}", __base__=SavedReusable, **fields_of_node),
            )
        outcomes_classes[node], reusable_classes[node] = of_same_outputs[same_outputs]
    names = tuple(nodes)
    # a Literal of the node names, known once the graph is compiled
    complete = list[Literal[names]] if names else Annotated[list[str], PydanticField(max_length=0)]  # type: ignore[valid-type]
    fields_of_run: dict[str, Any] = {  # hints of several kinds, as in fields_of_node
        "graph": Literal[fingerprint],
        "inputs": _ExactKeys(
            {name: _SavedValue(of_input, output=False) for name, of_input in inputs.items()}
        ),
        "outcomes": _ExactKeys({}, outcomes_classes),
        "reusable": _ExactKeys({}, reusable_classes),
        "complete": complete,
    }
    namespace = {"outcomes_classes": outcomes_classes, "reusable_classes": reusable_classes}
    return create_model(
        f"SavedRun_{fingerprint}", __base__=SavedRun, __namespace__=namespace, **fields_of_run
    )


def where_in(codec: type[SavedRun], loc: Sequence[str | int]) -> str:
    """Returns the place of an error in a saved run as text, such as ``"outcomes.g.outputs.result.0"``.

    ``loc`` is the place as pydantic gives it. The text stops at the value found there, and doesn't go into the
    value, since a step inside it may be a key of the value's own dict. So
    ``("outcomes", "g", "outputs", "result", 0, "counts", "ida")`` gives ``"outcomes.g.outputs.result.0"``, where
    the value of ``g.result`` on call 0 is ``{"counts": {"ida": ...}}``. ``("inputs", "text")`` and
    ``("outcomes", "g", "states", 0)`` are kept whole. An empty ``loc`` gives ``"the run"``.

    The steps are followed through the fields of ``codec``, the graph's codec. A value starts wherever a field
    declares a value's hint, such as ``_SavedValue``, ``Any`` or ``str``.
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
    """Returns the hint that ``step`` leads to inside ``hint``, or ``None`` when ``hint`` is a value's.

    ``hint`` is one of the hints of a codec's fields. A step that a class or an ``_ExactKeys`` object doesn't
    declare leads to ``Any``.
    """
    if get_origin(hint) in (
        Union,
        UnionType,
    ):  # an optional key, ``X | None``: follow ``X``
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
    """Returns the hint of each field of ``hint`` by name, if it is a pydantic model or a dataclass.

    It returns ``None`` for any other hint. A codec's classes are models, and ``Unanswered`` inside one is a dataclass.
    """
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        return {name: field.annotation for name, field in hint.model_fields.items()}
    if isinstance(hint, type) and is_dataclass(hint):
        return get_type_hints(hint)
    return None


def codec_of(graph: Graph[Any]) -> type[SavedRun]:
    """Returns the codec of ``graph``, the model its runs are written as JSON and read back with.

    It is built the first time one of the graph's runs is written or read, and the ``Graph`` keeps it.
    """
    blueprint = graph._blueprint
    return run_codec(
        graph.fingerprint,
        inputs={name: port.whole for name, port in graph._inputs.items()},
        outputs={
            place_text(path): node.compiled_node.outputs
            for path, node in blueprint.function_nodes.items()
        },
        nodes=[place_text(path) for path in blueprint.function_nodes],
    )
