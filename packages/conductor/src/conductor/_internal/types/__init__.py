"""Type hints, as conductor reads them.

Every input and output of a node has a type hint, like ``int``, ``list[str]`` or
``Annotated[int, Param(ge=0)]``. Conductor needs more from a hint than Python gives. It needs a name to show a
person, a pydantic schema to check values with, a way to write a value as JSON, and a rule for whether an edge
may join two types. ``Type`` wraps one hint and answers all of these.

The rest of the package supports it. ``Param`` is what a node author adds to a hint: a title, limits like
``ge=0``, choices to pick from. ``Choice`` is one of those choices. ``@host_type`` is how a host gives a class of
its own a name. The full rules are in the reference: ``python -m conductor.about types``.
"""

from .choice import Choice, check_one_of, choice_texts, pick_texts, read_choices
from .fit import fit
from .hint import a_number, union_members, union_of
from .host_type import NAMES, host_type
from .param import Param, merged_param, split_hint
from .record import NoSuchFieldError, field_of
from .type import Type, TypeRefusal

__all__ = [
    "NAMES",
    "Choice",
    "NoSuchFieldError",
    "Param",
    "Type",
    "TypeRefusal",
    "a_number",
    "check_one_of",
    "choice_texts",
    "field_of",
    "fit",
    "host_type",
    "merged_param",
    "pick_texts",
    "read_choices",
    "split_hint",
    "union_members",
    "union_of",
]
