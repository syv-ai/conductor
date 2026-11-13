"""The question about each output as a call asks it, and whether an answer must give it.

``assign(ticket: str) -> str`` raising ``Ask("Who?", default="bo")`` asks about ``result`` with ``default="bo"``,
so an answer may leave it out.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field, FieldDescription
from conductor._internal.types import Choice


def as_asked(
    field: Field, question: FieldDescription, choices: Mapping[str, tuple[Choice, ...]], defaults: Mapping[str, JsonValue]
) -> FieldDescription:
    """The question about ``field`` as one call asks it: the question its type sets, with the choices and default
    the call's ``Ask`` set on it, and ``required`` worked out from them. The questions a host is shown and the answer
    door both read it.

    ``Ask("Who?", default="bo")`` on ``assign(ticket: str) -> str`` asks ``result`` with ``default="bo"`` and
    ``required=False``. Where the ``Ask`` set nothing on ``field``, ``question`` itself is returned.

    Args:
        field: The output the answer fills.
        question: The question about it as its type sets it (``CompiledFunctionNode.questions``).
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form.
        defaults: The answer the ``Ask`` proposed, by output name, in its JSON form.
    """
    name = field.name
    if name not in choices and name not in defaults:
        return question
    limited_to = choices.get(name, question.choices)
    default = defaults.get(name, question.default)
    required = answer_required(field, defaults, limited=limited_to is not None)
    return question.model_copy(update={"choices": limited_to, "default": default, "required": required})


def answer_required(field: Field, defaults: Mapping[str, JsonValue], *, limited: bool) -> bool:
    """Whether an answer must give the question about ``field``. The answer door and the questions a host is shown
    both ask this.

    ``-> str`` must be answered; ``-> str | None`` may be left out, and is then ``None``, unless choices limit it.
    An answer needn't give a question that has a default (its field's, ``SKIPPED`` too, or one its ``Ask`` set) or
    is one of an exclusive group.

    Args:
        field: The field the answer fills.
        defaults: The answers the ``Ask`` proposed, by output name.
        limited: Whether choices, the type's or the ``Ask``'s, limit the answer.
    """
    defaulted = field.name in defaults or field.default is not REQUIRED
    may_be_none = not field.is_series and field.whole.optional and not limited
    return not defaulted and field.param.exclusive_group is None and not may_be_none
