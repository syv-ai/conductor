"""Asking a person: ``raise Ask(...)`` in a node or in the function of an ``Asked`` parameter, and the questions a host shows.

A function that raises ``Ask`` is answered with a value of its return type, and that return type says how to ask:
``-> str`` is a line of free text, ``-> bool`` a yes or no, and a frozen dataclass a form of one question per field.
How each question looks (its title, its widget, choices known when the node is written) is the type's, through
``Param``. ``Ask`` adds only what the run knows: the prompt, choices known only when the node runs, and a proposed
answer. Every return type's question is in the reference, ``python -m conductor.about asking-a-person``.

A node asks in one of two ways, and says which where it is declared. Its own function raises ``Ask``, and the answer
is its outputs: ``@node("reason", ..., asks=True)`` on ``def reason(ticket: str) -> str``. Or a parameter is
``Asked``, filled by a function that returns a value or raises ``Ask``, and the node is called once, with the value or
the answer: ``def approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str``, where
``approval(invoice: Invoice) -> bool``.

A host is shown a waiting call as a ``Waiting`` (``run/waiting.py``). The ``Ask`` is checked when it is raised, by the ``Ask`` door (``gate/raised.py``), and an answer before anything runs, by the answer
door (``gate/answer.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict

from pydantic import JsonValue

from conductor._internal.field import Field, FieldDescription
from conductor._internal.markers import REQUIRED


class Ask(BaseException):
    """Stops a call to ask a person; the return type of the function that raises it is what they answer.

    Nothing that reads the call runs until an answer arrives. A node that raises it in its own function says
    ``asks=True`` on its ``@node``, and the answer is its outputs: the node is not called again for it. A free-text
    question is the plain case, ``-> str`` and ``raise Ask(prompt)``::

        @node("reason", title="Reason", description="Asks why a ticket was closed.", asks=True)
        def reason(ticket: str) -> str:
            raise Ask(f"Why was {ticket} closed?")

        @node("assign", title="Assign", description="Asks who takes a ticket.", asks=True)
        def assign(ticket: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    The function of an ``Asked`` parameter raises it too, and the answer is that parameter's value. A raise from a
    node that declares neither fails the call as ``ask_not_declared``.

    ``Ask`` derives from ``BaseException``, so an ``except Exception`` in the function lets it through.

    Args:
        prompt: What the person reads.
        choices: The values the answer is limited to, known only when the node
            runs, each a value or a ``Choice`` to give it a title. For one
            question (an ``Asked`` parameter, or a node with one output), a
            list; a ``Series`` picks several of them. For a node with several
            outputs, which is answered by output name, a mapping by output name
            that names only the outputs it limits. Choices known when the node
            is written go on the type: a ``Literal``, or ``Param(choices=)``.
        default: The proposed answer, in the answer's shape: a value for one
            question, a mapping by output name for several, naming only the
            outputs it proposes for. A question with a default may be left out
            of an answer. Left out, nothing is proposed beyond the defaults of
            a returned dataclass's fields.

    Attributes:
        prompt: What the person reads.
        choices: The choices as given, ``None`` when none were.
        default: The proposed answer as given, ``REQUIRED`` when none was.
    """

    def __init__(self, prompt: str, *, choices: list[Any] | Mapping[str, list[Any]] | None = None, default: Any = REQUIRED) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.choices = choices
        self.default = default


@dataclass(frozen=True, slots=True)
class Asked:
    """Marks a node's parameter as filled by a function that returns its value or asks a person for it.

    Example: the parameter ``ok`` of ``approve`` is ``approval``'s return value, or a person's answer when
    ``approval`` raises ``Ask``::

        def approval(invoice: Invoice) -> bool:
            if invoice["amount"] <= 500:
                return True
            raise Ask(f"Pay {invoice['number']}?")

        @node("approve", title="Approve", description="Pays an invoice a person approves.")
        def approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str:
            return "paid" if ok else "held"

    The function's parameters are parameters of the node, by name: its inputs, and its context when the node takes
    one. It returns the parameter's type, and that type is what a person answers. No edge or constant reaches the
    parameter, so it is not an input of the node. A node has at most one, and a node with one does not also declare
    ``asks=True``.

    Attributes:
        function: The function that gives the parameter its value.
    """

    function: Callable[..., Any]


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
