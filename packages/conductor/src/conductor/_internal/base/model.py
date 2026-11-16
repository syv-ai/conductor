"""``ConductorModel``, the base class of every record a host saves or sends as JSON.

A graph record, a problem, an event, a call waiting for a person and a widget are all ``ConductorModel``s. Each is
a frozen pydantic model, written with pydantic's ``model_dump_json`` and read back with ``model_validate_json``.

Things only the engine uses, such as a compiled graph or the outcome of a call, are plain dataclasses instead. The
engine makes far more of them than it ever reads from JSON, and a dataclass is quicker to make.

Since a ``ConductorModel`` is meant to be written as JSON, each subclass is checked once, as soon as pydantic has
built it: every field it writes must have a JSON form. A field whose class pydantic can only check with
``isinstance``, such as an HTTP client or a ``polars.DataFrame``, is refused, and so is a ``type[X]``. A field
typed ``Any`` passes, since pydantic can't know what it will be given. A field with ``Field(exclude=True)`` is never
written, so it isn't checked.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticOmit
from pydantic_core.core_schema import IsSubclassSchema, ModelFieldsSchema, ModelSchema

from conductor._internal.utils import comma_separated, qualified_name

from .json import NON_FINITE_FLOATS


class _JsonFormSchema(GenerateJsonSchema):
    """Pydantic's JSON schema generator, changed to refuse a ``type[X]`` and to note which fields it refuses.

    Pydantic writes ``{}`` as the schema of a ``type[X]``, but a class can't be written as JSON, so this refuses it.
    When a model's fields can't all be written, it tries each field on its own, so the error can name the ones at
    fault.

    Attributes:
        refused: Each field with no JSON form, as ``"module.Model.field"``. The fields of a model nested inside the
            one being checked are listed too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.refused: list[str] = []
        self._models: list[
            type[BaseModel]
        ] = []  # the model whose fields are being written, innermost last

    def is_subclass_schema(self, schema: IsSubclassSchema) -> JsonSchemaValue:
        return self.handle_invalid_for_json_schema(
            schema, f"core_schema.IsSubclassSchema ({schema['cls']})"
        )

    def model_schema(self, schema: ModelSchema) -> JsonSchemaValue:
        self._models.append(schema["cls"])
        try:
            return super().model_schema(schema)
        finally:
            self._models.pop()

    def model_fields_schema(self, schema: ModelFieldsSchema) -> JsonSchemaValue:
        """Returns the schema of a model's fields. If any field has no JSON form, notes each such field in
        ``refused`` and raises again."""
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
    """A frozen pydantic model that a host writes as JSON and reads back.

    It refuses any field it doesn't declare. Each subclass is checked when pydantic builds it, so that every field it
    writes has a JSON form.
    """

    #: ``inf`` and ``nan`` are written as text (``NON_FINITE_FLOATS``). A subclass that sets its own ``model_config``
    #: keeps these three settings, since pydantic merges a subclass's config into its base's.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan=NON_FINITE_FLOATS)

    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        """Checks that every field this subclass writes has a JSON form.

        Pydantic calls this once it has built the class. That is usually where the class statement ends. When a
        field's type is named before it is defined, it is when the class is rebuilt instead. The check builds the
        class's JSON schema for writing, which pydantic refuses for a field it can't write as JSON.

        Raises:
            pydantic.errors.PydanticInvalidForJsonSchema: A field's type has no JSON form. The message names the class
                and the field. A field of a model nested inside it is named by that model.
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
        """Returns the fields the repr shows, leaving out those that still have their default.

        This is how scikit-learn prints an estimator. A ``Textarea`` with 8 rows prints as ``Textarea(rows=8)``, and
        one with the default of 4 rows prints as ``Textarea()``.
        """
        fields = type(self).model_fields
        for name, value in super().__repr_args__():
            field = fields.get(name) if name is not None else None
            if field is not None and not field.is_required():
                default = field.get_default(call_default_factory=True)
                # Try ``is`` first: a frame's == compares cell by cell instead of giving True or False.
                if value is default or default is not None and value == default:
                    continue
            yield name, value
