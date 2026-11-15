"""The questions a person is shown when a node asks.

A node that raises ``Ask`` is answered by filling in its outputs, and each output is one question. A node that
returns ``-> str`` has one output, ``result``, so it asks one question. A node that returns a dataclass with the
fields ``name`` and ``age`` asks two.

A question looks just like its output does in ``describe()``: a title, a type, a widget and so on. The ``Ask`` can
add two things on top. It can offer choices to pick from, and it can propose a default answer.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from conductor._internal.field import Field, FieldDescription
from conductor._internal.types import Choice


def as_asked(
    field: Field, choices: Mapping[str, tuple[Choice, ...]], defaults: Mapping[str, JsonValue]
) -> FieldDescription:
    """Returns the question about one output, as this call asks it.

    It starts from the output's own description, the one ``describe()`` shows. Then it adds what the call's ``Ask``
    set for this output. The ``Ask``'s choices replace the output's own. The ``Ask``'s default is filled in, and the
    question is then not required, since an answer may leave it out and get the default.

    For example, ``assign(ticket: str) -> str`` asks one question, ``result``. On its own it is required and has no
    default. If the node raises ``Ask("Who?", default="bo")``, the question has ``default="bo"`` and
    ``required=False``.

    What a host is shown and what the answer is checked against both come from here.

    Args:
        field: The output the question is about.
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form.
        defaults: The defaults the ``Ask`` proposed, by output name, in their JSON form.

    Returns:
        The output's description, with the ``Ask``'s choices and default added.
    """
    name = field.name
    asked = field.description_with(choices[name]) if name in choices else field.description
    return (
        asked.model_copy(update={"default": defaults[name], "required": False})
        if name in defaults
        else asked
    )
