"""How the runs of one graph are read from JSON and written as it: the records' own schemas, with what depends on the graph put in."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple

from pydantic import TypeAdapter
from pydantic_core import CoreConfig, CoreSchema, SchemaSerializer, SchemaValidator, core_schema

from conductor._internal.field import Field
from conductor._internal.outcomes import outputs_when_not_called, outputs_when_skipped
from conductor._internal.saved_run.read import outcomes_of, reusable_of, with_shared_tables
from conductor._internal.saved_run.type import (
    SavedOutcomes,
    SavedReusable,
    SavedRun,
    SkippedItems,
    SkippedSlots,
)
from conductor._internal.value_type import Type

#: Each record's own schema, as pydantic makes it from the record: what a save's schema is composed from.
_OUTCOMES, _REUSABLE, _RUN = (TypeAdapter(record).core_schema for record in (SavedOutcomes, SavedReusable, SavedRun))

#: The schemas of what ``SavedOutcomes`` holds for one output beside its column.
_SKIPPED_SLOTS, _SKIPPED_ITEMS = TypeAdapter(SkippedSlots).core_schema, TypeAdapter(SkippedItems).core_schema

#: The key of a save that names its graph, as ``SavedRun`` declares it.
_GRAPH = SavedRun.__dataclass_fields__["graph"].name  # a KeyError where this is read, should the field be renamed


class RunCodec(NamedTuple):
    """How the runs of one graph are read from JSON and written as it.

    Attributes:
        inputs: The type of each input of the graph, by address.
        outputs: The outputs of each function node of the graph and of the graphs embedded in it, by the node's
            name in a save.
        shareable: The ``(node, output)`` pairs whose stored value may be a table that another output also has,
            in the order a save writes them. A save looks only at these for a table to write once.
        validator: Reads a save into the ``SavedRun`` that ``_run_of`` makes a run of.
        serializer: Writes the ``SavedRun`` that ``Run._saved`` gives.
    """

    inputs: Mapping[str, Type]
    outputs: Mapping[str, Mapping[str, Field]]
    shareable: tuple[tuple[str, str], ...]
    validator: SchemaValidator
    serializer: SchemaSerializer


def save_form(fingerprint: str, inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], nodes: Iterable[str]) -> RunCodec:
    """Composes how the runs of one graph are read from JSON and written as it.

    What a save holds is declared once, by ``SavedRun``, ``SavedOutcomes`` and ``SavedReusable``, and their own
    schemas are what is written and read by. Only what depends on the graph is put in here: each input's and
    each output's own type, the nodes a save may name, and the graph's fingerprint. The parts are pydantic's own
    schemas of those types, put together as core schemas, since generating a model of a whole graph costs more
    than reading a small run. A graph's form is made the first time a run of it is saved or loaded.

    Args:
        fingerprint: The graph's fingerprint.
        inputs: The type of each input of the graph, by address.
        outputs: The outputs of each function node of the graph and of the graphs embedded in it, by the node's
            name in a save.
        nodes: Every function node of the graph and of the graphs embedded in it, by its name in a save.
    """
    shared: dict[str, CoreSchema] = {}
    outcomes_record, reusable_record = _inlined(_OUTCOMES, shared), _inlined(_REUSABLE, shared)
    blocks: dict[tuple[tuple[str, Type, bool, Any], ...], tuple[CoreSchema, CoreSchema]] = {}
    units, earlier = {}, {}
    for node, of_node in outputs.items():
        not_taken, not_called = outputs_when_skipped(of_node), outputs_when_not_called(of_node)
        columns = tuple((name, output.column, output.many, not_called[name]) for name, output in of_node.items())
        if columns not in blocks:  # nodes of the same outputs, which hold the same when they aren't called, share one schema
            block = len(blocks)
            of_block = _with_fields(
                outcomes_record,
                outputs=_closed({name: _inlined(column.saved_schema, shared) for name, column, _, _ in columns}),
                skipped=_closed({}, {name: _SKIPPED_SLOTS for name, _, _, _ in columns}),
                skipped_items=_closed({}, {name: _SKIPPED_ITEMS for name, _, many, _ in columns if many}),
            )
            shared[f"units:{block}"] = core_schema.no_info_after_validator_function(
                functools.partial(outcomes_of, not_taken, not_called), of_block, ref=f"units:{block}",
            )
            reusable_schema = _with_fields(reusable_record, units=core_schema.definition_reference_schema(f"units:{block}"))
            shared[f"earlier:{block}"] = core_schema.no_info_after_validator_function(reusable_of, reusable_schema, ref=f"earlier:{block}")
            blocks[columns] = core_schema.definition_reference_schema(f"units:{block}"), core_schema.definition_reference_schema(f"earlier:{block}")
        units[node], earlier[node] = blocks[columns]
    run = _with_fields(
        _inlined(_RUN, shared),
        graph=core_schema.literal_schema([fingerprint]),
        inputs=_closed({address: _inlined(whole.saved_schema, shared) for address, whole in inputs.items()}),
        units=_closed({}, units),
        earlier=_closed({}, earlier),
        code=_closed({}, dict.fromkeys(units, core_schema.str_schema())),
        complete=core_schema.list_schema(core_schema.literal_schema(names)) if (names := list(nodes)) else core_schema.list_schema(max_length=0),
    )
    shareable = tuple((node, name) for node, of_node in outputs.items() for name, output in of_node.items() if output.whole.may_write_once)
    if shareable:  # skip the extra pass when no output can be a shared table
        run = core_schema.no_info_after_validator_function(functools.partial(with_shared_tables, outputs), run)
    schema = core_schema.definitions_schema(core_schema.tagged_union_schema({fingerprint: run}, discriminator=_GRAPH), list(shared.values()))
    config = CoreConfig(ser_json_inf_nan="strings")
    return RunCodec(inputs, outputs, shareable, SchemaValidator(schema, config), SchemaSerializer(schema, config))


def _with_fields(record: Any, **schemas: CoreSchema) -> CoreSchema:
    """Gives a record's schema with the schema of each named field replaced by the one composed for a graph.

    Args:
        record: The schema pydantic made of a record of a save.
        **schemas: The schema of a field, by the field's name.

    Raises:
        KeyError: The record declares no field of a name given.
    """
    fields = {declared["name"]: declared for declared in record["schema"]["fields"]}
    for name, schema in schemas.items():
        declared = fields[name]
        fields[name] = {**declared, "schema": {**declared["schema"], "schema": schema} if declared["schema"]["type"] == "default" else schema}
    composed = {**record, "schema": {**record["schema"], "fields": list(fields.values())}}
    del composed["ref"]  # pydantic's name for the record as declared, which this no longer is
    return composed  # type: ignore[return-value]  # a dataclass schema, as the record's was


def _closed(required: Mapping[str, CoreSchema], optional: Mapping[str, CoreSchema] = MappingProxyType({})) -> CoreSchema:
    """Gives the schema of an object of exactly these keys, each of ``required`` there: the names a graph gives,
    of its inputs, its nodes and a node's outputs."""
    # each field is written out: a graph makes three for every node, and pydantic's own helper costs more than the field
    fields: dict[str, Any] = {name: {"type": "typed-dict-field", "schema": schema} for name, schema in required.items()}
    fields.update((name, {"type": "typed-dict-field", "schema": schema, "required": False}) for name, schema in optional.items())
    return core_schema.typed_dict_schema(fields, extra_behavior="forbid")


def _inlined(schema: CoreSchema, shared: dict[str, CoreSchema]) -> CoreSchema:
    """Gives ``schema`` as it stands inside another: what pydantic defined beside it (a model met twice, one that
    holds itself) moves to ``shared``, since one validator takes each definition once."""
    if schema["type"] != "definitions":
        return schema
    shared.update((definition["ref"], definition) for definition in schema["definitions"])  # type: ignore[typeddict-item]  # every definition has its ref
    return schema["schema"]
