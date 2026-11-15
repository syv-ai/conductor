"""The question about each output as a call asks it, and as a palette shows the output.

``assign(ticket: str) -> str`` raising ``Ask("Who?", default="bo")`` asks about ``result`` with ``default="bo"``,
so an answer may leave it out.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from conductor._internal.field import Field, FieldDescription
from conductor._internal.types import Choice


def as_asked(
    field: Field, choices: Mapping[str, tuple[Choice, ...]], defaults: Mapping[str, JsonValue]
) -> FieldDescription:
    """The question about the output ``field`` as one call asks it: the field's description (``Field.description``,
    which says what an answer that leaves it out gives), with the choices and the default the call's ``Ask`` set laid
    over it. The questions a host is shown and the answer validator both read it.

    ``assign(ticket: str) -> str`` asks ``result`` with ``required=True`` and no default; with ``Ask("Who?",
    default="bo")`` it asks with ``default="bo"``, ``required=False``. Choices the ``Ask`` sets limit the question
    as a type's choices do, so ``-> str | None`` asked with ``choices=["a"]`` must be answered. The ``Ask`` validator
    refuses a default for an output of an exclusive group: an answer that left it out would give it the default, and no
    other output of the group could be answered.

    Args:
        field: The output the answer fills.
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form.
        defaults: The answer the ``Ask`` proposed, by output name, in its JSON form.
    """
    name = field.name
    asked = field.description_with(choices[name]) if name in choices else field.description
    return (
        asked.model_copy(update={"default": defaults[name], "required": False})
        if name in defaults
        else asked
    )
