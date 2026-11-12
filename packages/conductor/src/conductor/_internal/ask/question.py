"""What a call asked: ``Question``, kept with the call's outcome and saved with the run; ``Answer``, a person's answer
to an ``Asked`` parameter; and whether an answer must give a question.

``approve(line: str) -> bool`` raising ``Ask("approve b?")`` keeps ``Question(prompt="approve b?", inputs={"line":
"b"})``. ``field_questions`` makes the question about each field an answer fills from the field's type, before an
``Ask`` sets anything.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict

from pydantic import JsonValue

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field, FieldDescription


class QuestionOverrides(TypedDict, total=False):
    """What an ``Ask`` sets on one output's question, as JSON; a key is present only when the ``Ask`` set it.

    Attributes:
        choices: The choices offered, each its ``{"id", "title", ...}`` record with the id in its JSON form.
        default: The proposed answer in its JSON form.
    """

    choices: list[dict[str, Any]]
    default: JsonValue


#: The overrides of an ``Ask`` that set nothing, shared by every call such an ``Ask`` stops.
NOTHING_SET: Mapping[str, QuestionOverrides] = MappingProxyType({})


@dataclass(slots=True)
class Question:
    """What a node asked on one call, kept with the call's outcome while it waits and after it is answered, and saved
    with the run.

    A plain record, made for every call that asks, so it costs what a dict would; a save reads it back through
    pydantic. It is never changed once made.

    Attributes:
        prompt: What the person reads.
        inputs: What the node received, as JSON.
        overrides: What the ``Ask`` set on each question, by output name; empty when it set nothing.
    """

    prompt: str
    inputs: dict[str, Any]
    overrides: Mapping[str, QuestionOverrides] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    """A person's answer to the question of an ``Asked`` parameter, kept by the run until the node is called with it.

    ``run.answer(waiting, True)`` for ``approve`` keeps ``Answer(question, "ok", True)``, and the next ``resume()``
    calls ``approve`` with ``ok=True``. The call's outcome is then ``Answered``, with that question.

    Attributes:
        question: What the person was asked.
        parameter: The name of the parameter the answer fills.
        value: The answer, read by the parameter's type.
    """

    question: Question
    parameter: str
    value: Any


def field_questions(fields: Mapping[str, Field]) -> dict[str, FieldDescription]:
    """The question about each field an answer fills, as the field's type sets it, before an ``Ask`` sets anything.

    Args:
        fields: The fields the answer fills, by name: a node's outputs, or its ``Asked`` parameter.

    Returns:
        A question per field, by name, under the field's own label; a field of a returned dataclass with a default
        proposes it.
    """
    return {name: _question(field) for name, field in fields.items()}


def _question(field: Field) -> FieldDescription:
    """One field's question as its type sets it."""
    question = field.description
    return question.model_copy(update={"required": answer_required(field, {}, limited=question.choices is not None)})


def answer_required(field: Field, overrides: QuestionOverrides, *, limited: bool) -> bool:
    """Whether an answer must give the question about ``field``. The answer door and the questions a host is shown
    both ask this.

    ``-> str`` must be answered; ``-> str | None`` may be left out, and is then ``None``, unless choices limit it.
    An answer needn't give a question that has a default (its field's, ``SKIPPED`` too, or one its ``Ask`` set) or
    is one of an exclusive group.

    Args:
        field: The field the answer fills.
        overrides: What the ``Ask`` set on its question.
        limited: Whether choices, the type's or the ``Ask``'s, limit the answer.
    """
    defaulted = "default" in overrides or field.default is not REQUIRED
    may_be_none = not field.is_series and field.whole.optional and not limited
    return not defaulted and field.param.exclusive_group is None and not may_be_none
