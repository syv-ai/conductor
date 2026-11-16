"""The questions a person is shown when a node asks.

A node that raises ``Ask`` is answered by filling in its outputs, and each output is one question. A node that
returns ``-> str`` has one output, ``result``, so it asks one question. A node that returns a dataclass with the
fields ``name`` and ``age`` asks two.

A question looks just like its output does in ``describe()``: a title, a type, a widget and so on. The ``Ask`` can
add two things on top. It can offer choices to pick from, and it can propose a default answer. A node with one
output is answered with a value, so its one question is required, unless the output is in an exclusive group: an
output of a group is never required, even when it is the node's only output.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from conductor._internal.field import Field, FieldDescription
from conductor._internal.types import Choice


def as_asked(
    outputs: Mapping[str, Field],
    choices: Mapping[str, tuple[Choice, ...]],
    defaults: Mapping[str, JsonValue],
) -> dict[str, FieldDescription]:
    """Returns the questions one call asks, one per output of the node, by output name.

    Each starts from the output's own description, the one ``describe()`` shows. Then it adds what the call's
    ``Ask`` set for that output. The ``Ask``'s choices replace the output's own, and its default is filled in.

    A node with one output is answered with its one value, never with a mapping that could leave the question out.
    So its question is required, and a default is only the value a host starts the answer with. For example,
    ``assign(ticket: str) -> str`` raises ``Ask("Who?", default="bo")``. Its one question, ``result``, has
    ``default="bo"`` and ``required=True``. An answer of ``{}``, or ``None`` where the output doesn't take ``None``, is
    refused as ``invalid_answer``.

    An output in an exclusive group is never required, even when it is the node's only output.

    On a node with several outputs, a question left out of the mapping answer takes its default. So a question with
    a default from the ``Ask`` isn't required.

    What a host is shown and what the answer is checked against both come from here.

    Args:
        outputs: The node's outputs, by name.
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form.
        defaults: The defaults the ``Ask`` proposed, by output name, in their JSON form.

    Returns:
        Each output's description, with the ``Ask``'s choices and default added, in the order of the outputs.
    """
    questions: dict[str, FieldDescription] = {}
    for name, field in outputs.items():
        asked = field.description_with(choices[name]) if name in choices else field.description
        if name in defaults:
            asked = asked.model_copy(update={"default": defaults[name], "required": False})
        # An output in an exclusive group is never required, even alone: its record would be refused.
        if len(outputs) == 1 and field.param.exclusive_group is None:
            asked = asked.model_copy(update={"required": True})
        questions[name] = asked
    return questions
