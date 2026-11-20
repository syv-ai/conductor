"""``KindByKey``, which reads a union of one-field models as the model whose field is the one key a mapping has.

A graph record's input entry is one of three kinds: ``{"constant": "?"}``, ``{"graph_input": "text"}`` or
``{"sources": [...]}``. Each kind is a model with one field, and the field's name is the key that says which kind an
entry is. Annotating the union with ``KindByKey`` reads ``{"constant": "?"}`` as the constant's model, with no tag of
its own in the JSON.

pydantic's own discriminated union would pick the kind too, but it puts the tag in front of the place of every fault
inside the value, as ``inputs.text.sources.sources.0``. Reading the picked model directly keeps the place a fault has
in the JSON, ``inputs.text.sources.0``, and each kind's key is written once, as its field's name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import BaseModel, GetCoreSchemaHandler, ValidatorFunctionWrapHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.problem import conductor_error


@dataclass(frozen=True)
class KindByKey:
    """Annotates a union of models of one field each, so a mapping is read as the model whose field it has.

    ``Annotated[ConstantEntry | GraphInputEntry | SourcesEntry, KindByKey(...)]`` reads ``{"graph_input": "text"}``
    as a ``GraphInputEntry``. An instance of one of the models is taken as it is. The union's own schema still writes
    the value and describes it in a JSON schema.

    Attributes:
        no_kind: The code of the fault for a value that has none of the keys, or more than one.
        message: That fault's message.
    """

    no_kind: str
    message: str

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Returns the union's schema, read by the model the value's one key picks."""
        kinds: dict[str, type[BaseModel]] = {}
        for kind in get_args(source):
            [key] = kind.model_fields
            kinds[key] = kind

        def pick(value: Any, _: ValidatorFunctionWrapHandler) -> Any:
            """Reads ``value`` as the model its one key picks.

            Raises:
                PydanticCustomError: ``no_kind`` if the value has none of the keys, or more than one.
                ValidationError: The picked model's faults, each at its place in the value.
            """
            if isinstance(value, tuple(kinds.values())):
                return value
            keys = [key for key in kinds if key in value] if isinstance(value, Mapping) else []
            if len(keys) != 1:
                raise conductor_error(self.no_kind, self.message)
            return kinds[keys[0]].model_validate(value)

        return core_schema.no_info_wrap_validator_function(pick, handler(source))
