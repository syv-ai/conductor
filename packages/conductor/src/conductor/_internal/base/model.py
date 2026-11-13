"""``ConductorModel``: the base class of every record a host saves or sends as JSON.

A graph record, a problem, an event, a question waiting for a person and a widget are each a frozen pydantic model
on this base. Each is written with pydantic's ``model_dump_json`` and read back with ``model_validate_json``. What
only the engine uses, such as a compiled graph or an outcome, is a dataclass instead: those are made far more often
than they are read from JSON, and a dataclass is faster to make.

Each subclass is checked once, when pydantic has finished it, for a JSON form of every field it writes: its JSON
schema in serialization mode is built, which pydantic refuses for a field of a class it can only check with
``isinstance``, such as a client or a ``polars.DataFrame``, or of a ``type[X]``. A field typed ``Any`` passes, since
pydantic cannot know what it will be given. A field with ``Field(exclude=True)`` is not written, so it is not checked.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticOmit
from pydantic_core.core_schema import IsSubclassSchema, ModelFieldsSchema, ModelSchema

from .json import NON_FINITE_FLOATS
from .text import comma_separated, qualified_name


class _JsonFormSchema(GenerateJsonSchema):
    """pydantic's JSON schema, which also refuses a ``type[X]`` and says which fields it refused.

    pydantic writes ``{}`` for a ``type[X]``, but a class has no JSON form.

    Attributes:
        refused: Each field the schema has no JSON form for, as ``"module.Model.field"``, the fields of a model inside
            the one asked for among them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.refused: list[str] = []
        self._models: list[type[BaseModel]] = []  # the model whose fields are being written, innermost last

    def is_subclass_schema(self, schema: IsSubclassSchema) -> JsonSchemaValue:
        return self.handle_invalid_for_json_schema(schema, f"core_schema.IsSubclassSchema ({schema['cls']})")

    def model_schema(self, schema: ModelSchema) -> JsonSchemaValue:
        self._models.append(schema["cls"])
        try:
            return super().model_schema(schema)
        finally:
            self._models.pop()

    def model_fields_schema(self, schema: ModelFieldsSchema) -> JsonSchemaValue:
        """The schema of a model's fields; when one has no JSON form, each such field is put in ``refused``."""
        try:
            return super().model_fields_schema(schema)
        except PydanticInvalidForJsonSchema:
            for name, field in schema["fields"].items():
                if not self.field_is_present(field):
                    continue
                try:
                    self.generate_inner(field)
                except PydanticOmit:
                    pass
                except PydanticInvalidForJsonSchema:
                    self.refused.append(f"{qualified_name(self._models[-1])}.{name}")
            raise


class ConductorModel(BaseModel):
    """A frozen pydantic model that a host writes as JSON and reads back. A field it does not declare is refused."""

    #: A float that is not finite is written as its text (``NON_FINITE_FLOATS``). A subclass that sets its own
    #: ``model_config`` keeps these three: pydantic merges a subclass's config into its base's.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan=NON_FINITE_FLOATS)

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        """Checks that every field the subclass writes has a JSON form, once pydantic has finished the class.

        pydantic finishes a class where its statement ends, or, for a field whose type is named before it is defined,
        when the class is rebuilt.

        Raises:
            pydantic.errors.PydanticInvalidForJsonSchema: A field's type has no JSON form. The message names the class
                and the field, and a field of a model inside it by that model.
        """
        schema = _JsonFormSchema()
        try:
            schema.generate(cls.__pydantic_core_schema__, mode="serialization")
        except PydanticInvalidForJsonSchema as error:
            raise PydanticInvalidForJsonSchema(
                f"{comma_separated(dict.fromkeys(schema.refused), empty=qualified_name(cls))} has a type with no JSON form, but {qualified_name(cls)} "
                "is a ConductorModel, which is written as JSON: give the field a type pydantic writes as JSON (text, a number, "
                f"a pydantic model, ...), or leave it out of the JSON with Field(exclude=True). pydantic says: {error.message.splitlines()[0]}"
            ) from error

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """The fields pydantic's repr shows, without those that have their default, as scikit-learn prints an estimator.

        A repr reads ``Textarea(rows=8)``, and ``Textarea()`` when ``rows`` has its default, 4.
        """
        fields = type(self).model_fields
        for name, value in super().__repr_args__():
            field = fields.get(name) if name is not None else None
            if field is not None and not field.is_required():
                default = field.get_default(call_default_factory=True)
                if value is default or (default is not None and value == default):  # a frame's == compares cell by cell, so ``is`` is tried first
                    continue
            yield name, value
