"""Builds, for one graph, pydantic's validator and serializer for its saved runs: the graph's codec.

The keys of a saved run are declared once, by the classes of ``type.py``. ``run_codec`` takes pydantic's
schemas of those classes and puts in what only the graph knows: the names of its nodes, inputs and outputs, and
the type of each value.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple

from pydantic import NonNegativeInt, TypeAdapter
from pydantic_core import CoreConfig, CoreSchema, SchemaSerializer, SchemaValidator, core_schema

from conductor._internal.field import Field
from conductor._internal.outcomes import outputs_when_not_called, outputs_when_skipped
from conductor._internal.run.json.read import (
    outcomes_from_saved,
    reusable_from_saved,
    with_shared_tables,
)
from conductor._internal.run.json.type import (
    SavedOutcomes,
    SavedReusable,
    SavedRun,
)
from conductor._internal.value_type import Type

#: Pydantic's schema of each class of a saved run, which ``run_codec`` starts from.
_OUTCOMES, _REUSABLE, _RUN = (TypeAdapter(saved_class).core_schema for saved_class in (SavedOutcomes, SavedReusable, SavedRun))

#: The schema of one output's value under ``skipped`` (call numbers) and under ``skipped_items`` (by call number, indexes).
_CALL_NUMBERS = TypeAdapter(list[NonNegativeInt]).core_schema
_INDEXES_BY_CALL_NUMBER = TypeAdapter(dict[NonNegativeInt, list[NonNegativeInt]]).core_schema


class RunCodec(NamedTuple):
    """Pydantic's validator and serializer for the saved runs of one graph, with the types they were built from.

    Attributes:
        inputs: The type of each graph input, by address.
        outputs: The outputs of each function node, by the node's name in a saved run (``"shout"``, ``"loud/shout"``).
        shareable: The ``(node, output)`` pairs whose value may be a table that another output also has, in the
            order they are saved; writing looks only at these for a table to write once.
        validator: Reads the JSON text of a saved run into a ``SavedRun``.
        serializer: Writes a ``SavedRun`` as JSON text.
    """

    inputs: Mapping[str, Type]
    outputs: Mapping[str, Mapping[str, Field]]
    shareable: tuple[tuple[str, str], ...]
    validator: SchemaValidator
    serializer: SchemaSerializer


def run_codec(fingerprint: str, inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], nodes: Iterable[str]) -> RunCodec:
    """Builds the codec of one graph.

    For the README graph ``lines -> shout -> join``, the validator accepts only a saved run whose ``graph`` is
    the graph's fingerprint, whose ``inputs`` has the one key ``lines.text`` with a ``str``, and whose
    ``outcomes``, ``reusable`` and ``code`` have no key but ``lines``, ``shout`` and ``join``. Under ``shout``,
    ``outputs`` has the one key ``result``, a list of ``str | None``.

    While it validates, the validator also turns each node's ``SavedOutcomes`` into a list of outcomes
    (``outcomes_from_saved``) and each ``SavedReusable`` into a ``ReusableOutcomes`` (``reusable_from_saved``),
    so a saved run is walked once.

    The schema is put together from core schemas, not generated from a pydantic model made for the graph:
    generating a model costs more than reading a small run. Nodes with the same outputs share one schema. A
    graph builds its codec the first time one of its runs is saved or loaded.

    Args:
        fingerprint: The graph's fingerprint.
        inputs: The type of each graph input, by address.
        outputs: The outputs of each function node, by the node's name in a saved run.
        nodes: The name in a saved run of every function node.
    """
    shared: dict[str, CoreSchema] = {}
    outcomes_schema, reusable_schema = _inline_definitions(_OUTCOMES, shared), _inline_definitions(_REUSABLE, shared)
    # by what the schema of a node depends on: for each output its name, its type, whether it is a Series, and its value when not called
    references: dict[tuple[tuple[str, Type, bool, Any], ...], tuple[CoreSchema, CoreSchema]] = {}
    outcomes, reusable = {}, {}
    for node, node_outputs in outputs.items():
        skipped_outputs, not_called_outputs = outputs_when_skipped(node_outputs), outputs_when_not_called(node_outputs)
        same_outputs = tuple((name, output.saved_list_type, output.many, not_called_outputs[name]) for name, output in node_outputs.items())
        if same_outputs not in references:
            outcomes_ref, reusable_ref = f"outcomes:{len(references)}", f"reusable:{len(references)}"
            of_node = _with_field_schemas(
                outcomes_schema,
                outputs=_exact_keys_schema(
                    {name: _inline_definitions(output.saved_list_type.saved_schema, shared) for name, output in node_outputs.items()},
                ),
                skipped=_exact_keys_schema({}, dict.fromkeys(node_outputs, _CALL_NUMBERS)),
                skipped_items=_exact_keys_schema({}, {name: _INDEXES_BY_CALL_NUMBER for name, output in node_outputs.items() if output.many}),
            )
            shared[outcomes_ref] = core_schema.no_info_after_validator_function(
                functools.partial(outcomes_from_saved, skipped_outputs, not_called_outputs), of_node, ref=outcomes_ref,
            )
            reusable_of_node = _with_field_schemas(reusable_schema, outcomes=core_schema.definition_reference_schema(outcomes_ref))
            shared[reusable_ref] = core_schema.no_info_after_validator_function(reusable_from_saved, reusable_of_node, ref=reusable_ref)
            references[same_outputs] = core_schema.definition_reference_schema(outcomes_ref), core_schema.definition_reference_schema(reusable_ref)
        outcomes[node], reusable[node] = references[same_outputs]
    run = _with_field_schemas(
        _inline_definitions(_RUN, shared),
        graph=core_schema.literal_schema([fingerprint]),
        inputs=_exact_keys_schema({address: _inline_definitions(of_input.saved_schema, shared) for address, of_input in inputs.items()}),
        outcomes=_exact_keys_schema({}, outcomes),
        reusable=_exact_keys_schema({}, reusable),
        code=_exact_keys_schema({}, dict.fromkeys(outcomes, core_schema.str_schema())),
        complete=core_schema.list_schema(core_schema.literal_schema(names)) if (names := list(nodes)) else core_schema.list_schema(max_length=0),
    )
    shareable = tuple((node, name) for node, node_outputs in outputs.items() for name, output in node_outputs.items() if output.whole.may_write_once)
    if shareable:  # speed only: the extra pass is skipped when no output can be a shared table
        run = core_schema.no_info_after_validator_function(functools.partial(with_shared_tables, outputs), run)
    # a tagged union of one, so that a saved run of another graph is refused by its "graph" key before anything else is read
    schema = core_schema.definitions_schema(core_schema.tagged_union_schema({fingerprint: run}, discriminator="graph"), list(shared.values()))
    config = CoreConfig(ser_json_inf_nan="strings")
    return RunCodec(inputs, outputs, shareable, SchemaValidator(schema, config), SchemaSerializer(schema, config))


def _with_field_schemas(saved_class_schema: Any, **schemas: CoreSchema) -> CoreSchema:
    """Returns the schema of a class of a saved run with the schemas of the named fields replaced.

    ``_with_field_schemas(_RUN, graph=core_schema.literal_schema(["4c16"]))`` is the schema of ``SavedRun`` whose
    ``graph`` accepts only ``"4c16"``.

    Args:
        saved_class_schema: The schema pydantic made of ``SavedRun``, ``SavedOutcomes`` or ``SavedReusable``.
        **schemas: The schema to use for a field, by the field's name.

    Raises:
        KeyError: The class has no field of a name given.
    """
    fields = {declared["name"]: declared for declared in saved_class_schema["schema"]["fields"]}
    for name, schema in schemas.items():
        declared = fields[name]
        fields[name] = {**declared, "schema": {**declared["schema"], "schema": schema} if declared["schema"]["type"] == "default" else schema}
    replaced = {**saved_class_schema, "schema": {**saved_class_schema["schema"], "fields": list(fields.values())}}
    del replaced["ref"]  # pydantic's name for the schema of the class as declared, which this is not
    return replaced  # type: ignore[return-value]  # a dataclass schema, as the one given was


def _exact_keys_schema(required: Mapping[str, CoreSchema], optional: Mapping[str, CoreSchema] = MappingProxyType({})) -> CoreSchema:
    """Returns the schema of a JSON object that must have every key of ``required``, may have the keys of
    ``optional``, and may have no other key.

    ``_exact_keys_schema({}, {"lines": a, "shout": b})`` accepts ``{}`` and ``{"shout": ...}`` and refuses
    ``{"other": ...}``.
    """
    # speed only: each field is written out as a dict, since a graph makes three per node and pydantic's helper costs more than that
    fields: dict[str, Any] = {name: {"type": "typed-dict-field", "schema": schema} for name, schema in required.items()}
    fields.update((name, {"type": "typed-dict-field", "schema": schema, "required": False}) for name, schema in optional.items())
    return core_schema.typed_dict_schema(fields, extra_behavior="forbid")


def _inline_definitions(schema: CoreSchema, shared: dict[str, CoreSchema]) -> CoreSchema:
    """Returns ``schema`` ready to be put inside another schema, and moves its definitions to ``shared``.

    Pydantic wraps the schema of a type in a ``definitions`` schema when the type contains a model twice or a
    model that contains itself. One validator takes each definition once, at the top, so the definitions are
    collected in ``shared`` and the inner schema is returned. Any other schema is returned as it is.
    """
    if schema["type"] != "definitions":
        return schema
    # every definition has its ref
    shared.update((definition["ref"], definition) for definition in schema["definitions"])  # type: ignore[typeddict-item]
    return schema["schema"]
