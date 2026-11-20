"""The questions a person is shown when a node asks.

A node that raises ``Ask`` is answered by filling in its outputs, and each output is one question. A node that
returns ``-> str`` has one output, ``result``, so it asks one question. A node that returns a dataclass with the
fields ``name`` and ``age`` asks two.

A question looks just like its output does in ``describe()``: a title, a type, a widget and so on. The ``Ask`` can
add two things on top. It can offer choices to pick from, and it can propose a default answer. A node with one
output is answered with a value, so its one question is always required.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from conductor._internal.port import Port, PortDescription
from conductor._internal.types import Choice


def as_asked(
    outputs: Mapping[str, Port],
    choices: Mapping[str, tuple[Choice, ...]],
    defaults: Mapping[str, JsonValue],
    *,
    record: bool = False,
) -> dict[str, PortDescription]:
    """Returns the questions one call asks, one per output of the node, by output name.

    Each starts from the output's own description, the one ``describe()`` shows. Then it adds what the call's
    ``Ask`` set for that output. The ``Ask``'s choices replace the output's own, and its default is filled in.

    A node with one output is answered with its one value, never with a mapping that could leave the question out.
    So its question is required, and a default is only the value a host starts the answer with. For example,
    ``assign(ticket: str) -> str`` raises ``Ask("Who?", default="bo")``. Its one question, ``result``, has
    ``default="bo"`` and ``required=True``. An answer of ``{}``, or ``None`` where the output doesn't take ``None``, is
    refused as ``invalid_answer``. That holds for a branch too.

    On a node with several outputs, a question left out of the mapping answer takes its default. So a question with
    a default from the ``Ask`` isn't required.

    What a host is shown and what the answer is checked against both come from here.

    A question saved before a deploy may offer a choice or propose a default the output's type refuses now. The form
    a person answers (``run.waiting``) offers only what the type takes, and drops such a default, so an answer is
    read by the type now. A record of what was asked (``record``, for ``run.asked``) keeps them as they were shown.

    Args:
        outputs: The node's outputs, by name.
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form.
        defaults: The defaults the ``Ask`` proposed, by output name, in their JSON form.
        record: Keep the choices and defaults as they were asked, for a record, not a form to answer.

    Returns:
        Each output's description, with the ``Ask``'s choices and default added, in the order of the outputs.
    """
    questions: dict[str, PortDescription] = {}
    for name, port in outputs.items():
        asked = port.description_with(choices.get(name), as_asked=record)
        if name in defaults and (record or port.takes(defaults[name])):
            asked = asked.model_copy(update={"default": defaults[name], "required": False})
        if len(outputs) == 1:
            asked = asked.model_copy(update={"required": True})
        questions[name] = asked
    return questions
