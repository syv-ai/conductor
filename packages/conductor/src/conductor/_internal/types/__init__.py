"""Type hints, as conductor reads them.

Every input and output of a node has a type hint, like ``int``, ``list[str]`` or
``Annotated[int, Param(ge=0)]``. Conductor needs more from a hint than Python gives. It needs a name to show a
person, a pydantic schema to check values with, a way to write a value as JSON, and a rule for whether an edge
may join two types. ``Type`` wraps one hint and answers all of these.

The rest of the package supports it. ``Param`` is what a node author adds to a hint: a title, limits like
``ge=0``, choices to pick from. ``Choice`` is one of those choices. ``@host_type`` is how a host gives a class of
its own a name. ``read_as_argument`` and ``read_as_output`` read a value that enters a graph by its type, strictly,
the way its JSON text would be read, and ``written_as_json`` writes one or refuses it. ``KindByKey`` marks a union of
one-field models, so a mapping is read as the model whose field is its one key. The full rules are in the reference: ``python -m conductor.about types``.
"""

from .choice import (
    Choice,
    Choices,
    check_one_of,
    choice_texts,
    choices_in_message,
    choices_schema,
    pick_texts,
)
from .fields import UnreadableFieldError, field_of
from .fit import fit
from .hint import (
    a_number,
    allows_extra,
    declares_fields,
    field_hints,
    hint_parts,
    is_alias,
    is_typed_dict,
    type_fields,
    union_members,
    union_of,
)
from .host_type import NAMES, NotBlank, host_type, is_declared
from .kind_by_key import KindByKey
from .param import SIZED, Param, as_written, split_hint
from .read import read_as_argument, read_as_output, written_as_json
from .type import Type, TypeRefusal, json_by_its_class

__all__ = [
    "NAMES",
    "SIZED",
    "Choice",
    "Choices",
    "KindByKey",
    "UnreadableFieldError",
    "NotBlank",
    "Param",
    "Type",
    "TypeRefusal",
    "a_number",
    "allows_extra",
    "as_written",
    "check_one_of",
    "choice_texts",
    "choices_in_message",
    "choices_schema",
    "declares_fields",
    "field_hints",
    "field_of",
    "fit",
    "hint_parts",
    "host_type",
    "is_alias",
    "is_declared",
    "is_typed_dict",
    "json_by_its_class",
    "pick_texts",
    "read_as_argument",
    "read_as_output",
    "split_hint",
    "type_fields",
    "union_members",
    "union_of",
    "written_as_json",
]
