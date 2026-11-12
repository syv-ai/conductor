"""Asking a person: ``raise Ask(...)`` in a node, and the questions a host shows.

The node's return type is what the person answers, and it says how to ask:
``-> str`` is a line of free text, ``-> bool`` a yes or no, and a frozen
dataclass a form of one question per field. How each question looks (its
title, its widget, choices known when the node is written) is the type's,
through ``Param``. ``Ask`` adds only what the run knows: the prompt, choices
known only when the node runs, and a proposed answer. Every return type's
question is in the reference, ``python -m conductor.about asking-a-person``.

A waiting unit is a ``Waiting``, and each of its ``questions`` is the field
record ``describe()`` gives an output, with what the ``Ask`` set: so a host
draws a question with the code that draws an input. The ``Ask`` is checked
when the node raises it, by the ``Ask`` door (``gate/raised.py``), and an
answer before anything runs, by the answer door (``gate/answer.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import pydantic_core
from pydantic import JsonValue

from conductor._internal.address import Loc, loc_text
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.field import Field, FieldDescription
from conductor._internal.hashing import canonical_json, short_hash
from conductor._internal.markers import REQUIRED

if TYPE_CHECKING:
    from conductor._internal import outcomes
    from conductor._internal.compiled import CompiledFunctionNode


class Ask(BaseException):
    """Stops a node's call to ask a person; the node's return type is what they answer.

    Nothing that reads the call runs until an answer arrives. The answer is the
    node's outputs, and the node is not called again for it. A
    free-text question is the plain case: ``-> str`` and ``raise Ask(prompt)``::

        @node("reason", title="Reason", description="Asks why a ticket was closed.")
        def reason(ticket: str) -> str:
            raise Ask(f"Why was {ticket} closed?")

        @node("assign", title="Assign", description="Asks who takes a ticket.")
        def assign(ticket: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    ``Ask`` derives from ``BaseException``, so an ``except Exception`` in the
    node lets it through.

    Args:
        prompt: What the person reads.
        choices: The values the answer is limited to, known only when the node
            runs, each a value or a ``Choice`` to give it a title. For a node
            with one output, a list; a ``Series`` output picks several of
            them. For a node with several outputs, which is answered by output
            name, a mapping by output name that names only the outputs it
            limits. Choices known when the node is written go on the type:
            a ``Literal``, or ``Param(choices=)``.
        default: The proposed answer, in the answer's shape: a value for a
            node with one output, a mapping by output name for several, naming
            only the outputs it proposes for. A question with a default may be
            left out of an answer. Left out, nothing is proposed beyond the
            defaults of a returned dataclass's fields.

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
    """What a node asked on one call, kept on its unit while it waits and after it is answered, and saved with the run.

    A plain record, made for every call that asks, so it costs what a dict would; a save reads it back through
    pydantic. Like the unit that holds it, it is never changed once made.

    Attributes:
        prompt: What the person reads.
        inputs: What the node received, as JSON.
        overrides: What the ``Ask`` set on each question, by output name; empty when it set nothing.
    """

    prompt: str
    inputs: dict[str, Any]
    overrides: Mapping[str, QuestionOverrides] = dataclasses.field(default_factory=dict)


class Waiting(ConductorModel):
    """A call of a node waiting for a person, as a host shows it.

    A node with one output is answered with its value, one with several with
    a mapping by output name.

    Attributes:
        id: What ``run.answer`` takes: the node's id, then ``@`` and each place
            of its item, then ``~`` and a fingerprint of the question asked
            there, ``"approve@1~3fa2c0d1"``. Inside a subgraph, the
            subgraph node and its item come first, ``"review@0/approve@1~3fa2c0d1"``.
            An answer for an id whose place has since come to hold another
            question is refused as ``answer_stale``.
        loc: Which unit it is, as a ``Problem``'s ``loc`` says it: the node
            that asks, then the place of its item in each Series the node
            runs per item of, ``("approve", 1)``; inside a subgraph,
            the subgraph node and its item first, ``("review", 0, "approve", 1)``.
        prompt: What the node's ``Ask`` said.
        questions: One per output: the ``FieldDescription`` its ``describe()``
            gives, with the ``choices`` and ``default`` the ``Ask`` set and
            ``required`` worked out from them.
        inputs: What the node received, as JSON.
    """

    id: str
    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


def output_questions(outputs: Mapping[str, Field]) -> dict[str, FieldDescription]:
    """Each output's question as the node's types set it, before an ``Ask`` sets anything.

    Args:
        outputs: The node's outputs, by name.

    Returns:
        A question per output, by name, under the output's own label, and a field of a returned dataclass with a
        default proposes it.
    """
    return {name: _question(field) for name, field in outputs.items()}


def _question(field: Field) -> FieldDescription:
    """One output's question as its type sets it."""
    question = field.description
    return question.model_copy(update={"required": answer_required(field, {}, limited=question.choices is not None)})


def waiting_questions(compiled_node: CompiledFunctionNode, overrides: Mapping[str, QuestionOverrides]) -> list[FieldDescription]:
    """A waiting unit's questions: each output's question with what its ``Ask`` set.

    Args:
        compiled_node: The node that asks.
        overrides: What the ``Ask`` set, by output name.

    Returns:
        A question per output, in the outputs' order.
    """
    outputs = compiled_node.outputs
    return [
        _overridden(outputs[name], question, overrides[name]) if name in overrides else question
        for name, question in compiled_node.questions.items()
    ]


def _overridden(field: Field, question: FieldDescription, overrides: QuestionOverrides) -> FieldDescription:
    """One output's question with what its ``Ask`` set, and whether it is ``required`` worked out again from that."""
    choices = overrides.get("choices", question.choices)
    return question.model_copy(update={**overrides, "required": answer_required(field, overrides, limited=choices is not None)})


def answer_required(field: Field, overrides: QuestionOverrides, *, limited: bool) -> bool:
    """Whether an answer must give the question about ``field``. The answer door and the questions a host is shown
    both ask this.

    ``-> str`` must be answered; ``-> str | None`` may be left out, and is then ``None``, unless choices limit it.
    An answer needn't give a question that has a default (its field's, ``SKIPPED`` too, or one its ``Ask`` set) or
    is one of an exclusive group.

    Args:
        field: The output.
        overrides: What the ``Ask`` set on its question.
        limited: Whether choices, the type's or the ``Ask``'s, limit the answer.
    """
    defaulted = "default" in overrides or field.default is not REQUIRED
    may_be_none = not field.is_series and field.whole.optional and not limited
    return not defaulted and field.param.exclusive_group is None and not may_be_none


def waiting_record(compiled_node: CompiledFunctionNode, loc: Loc, waiting: outcomes.Waiting) -> Waiting:
    """A waiting unit as a host shows it. Its id is where it is as text, then ``~`` and the fingerprint of the
    question asked there, as ``run.answer`` reads it.

    Args:
        compiled_node: The node that asks.
        loc: Where the unit is: the node and its item, behind those of each subgraph's node around it.
        unit: The unit.
    """
    question = waiting.question
    return Waiting(
        id=f"{loc_text(loc)}~{question_fingerprint(compiled_node, waiting)}", loc=loc, prompt=question.prompt,
        questions=waiting_questions(compiled_node, question.overrides), inputs=question.inputs,
    )


def question_fingerprint(compiled_node: CompiledFunctionNode, waiting: outcomes.Waiting) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when its place holds another question."""
    inputs = waiting.question.inputs
    values = [inputs[name] for name in sorted(inputs)]  # saved, so the names may come back from a store in any order
    hashed = [compiled_node.id, compiled_node.code_fingerprint, values]
    written = canonical_json(hashed) if {type(v) for v in values} & {dict, list} else pydantic_core.to_json(hashed)  # plain values have no keys to sort
    return short_hash(written, 4)
