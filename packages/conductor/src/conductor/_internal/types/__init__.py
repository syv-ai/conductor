"""Type hints, as conductor reads them.

Every input and output of a node has a type hint, like ``int``, ``list[str]`` or
``Annotated[int, Param(ge=0)]``. Conductor needs more from a hint than Python gives. It needs a name to show a
person, a pydantic schema to check values with, a way to write a value as JSON, and a rule for whether an edge
may join two types. ``Type`` wraps one hint and answers all of these.

The rest of the package supports it. ``Param`` is what a node author adds to a hint: a title, limits like
``ge=0``, choices to pick from. ``Choice`` is one of those choices. ``@host_type`` is how a host gives a class of
its own a name. The full rules are in the reference: ``python -m conductor.about types``.
"""

from .choice import (
    Choice,
    check_one_of,
    choice_texts,
    choices_in_message,
    choices_schema,
    pick_texts,
)
from .fields import NoSuchFieldError, field_of
from .fit import fit
from .hint import a_number, hint_parts, substituted, union_members, union_of
from .host_type import NAMES, NotBlank, host_type, is_declared
from .param import SIZED, Param, as_written, split_hint
from .type import Type, TypeRefusal, json_by_its_class

__all__ = [
    "NAMES",
    "SIZED",
    "Choice",
    "NoSuchFieldError",
    "NotBlank",
    "Param",
    "Type",
    "TypeRefusal",
    "a_number",
    "as_written",
    "check_one_of",
    "choice_texts",
    "choices_in_message",
    "choices_schema",
    "field_of",
    "fit",
    "hint_parts",
    "host_type",
    "is_declared",
    "json_by_its_class",
    "pick_texts",
    "split_hint",
    "substituted",
    "union_members",
    "union_of",
]
