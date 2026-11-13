"""What a call asked: ``Question``, kept with the call's outcome and saved with the run; each output's question as the
call asks it; and whether an answer must give a question.

``approve(line: str) -> bool`` raising ``Ask("approve b?")`` keeps ``Question(prompt="approve b?", inputs={"line":
"b"})``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue, SerializeAsAny

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field, FieldDescription
from conductor._internal.types import Choice


@dataclass(slots=True)
class Question:
    """What a node asked on one call, kept with the call's outcome while it waits and after it is answered, and saved
    with the run.

    A plain record, made for every call that asks, so it costs what a dict would; a save reads it back through
    pydantic. It is never changed once made.

    Attributes:
        prompt: What the person reads.
        inputs: What the node received, as JSON.
        choices: The choices the ``Ask`` offered, by output name, each id read by the output's type in its JSON form
            (``read_choices``); an output it set none on is not named.
        defaults: The answer the ``Ask`` proposed, by output name, in its JSON form; an output it proposed none for is
            not named.
    """

    prompt: str
    inputs: dict[str, Any]
    choices: Mapping[str, tuple[SerializeAsAny[Choice], ...]] = dataclasses.field(default_factory=dict)
    defaults: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)

    def as_asked(self, field: Field, question: FieldDescription) -> FieldDescription:
        """The question about ``field`` as this call asks it: the question its type sets, with the choices and default
        the ``Ask`` set on it, and ``required`` worked out from them. The questions a host is shown and the answer
        door both read it.

        ``Ask("Who?", default="bo")`` on ``assign(ticket: str) -> str`` asks ``result`` with ``default="bo"`` and
        ``required=False``. Where the ``Ask`` set nothing on ``field``, ``question`` itself is returned.

        Args:
            field: The output the answer fills.
            question: The question about it as its type sets it (``CompiledFunctionNode.questions``).
        """
        name = field.name
        if name not in self.choices and name not in self.defaults:
            return question
        choices = self.choices.get(name, question.choices)
        default = self.defaults.get(name, question.default)
        required = answer_required(field, self.defaults, limited=choices is not None)
        return question.model_copy(update={"choices": choices, "default": default, "required": required})


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
