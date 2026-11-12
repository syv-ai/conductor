"""A call waiting for a person, as a host is shown it: ``Waiting``, with the id ``run.answer`` takes.

Example: ``approve(line: str) -> bool`` raises ``Ask("approve b?")`` for the second line. ``run.waiting`` then lists
``Waiting(id="approve@1~3fa2c0d1", loc=("approve", 1), prompt="approve b?", questions=[...], inputs={"line": "b"})``.
Each of its ``questions`` is the field record ``describe()`` gives the field the answer fills, with what the ``Ask``
set, so a host draws a question with the code that draws an input.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pydantic_core

from conductor._internal.ask import QuestionOverrides, answer_required
from conductor._internal.base import ConductorModel, Loc, canonical_json, loc_text, short_hash
from conductor._internal.compiled import CompiledFunctionNode
from conductor._internal.field import Field, FieldDescription

from .outcome import Unanswered


class Waiting(ConductorModel):
    """A call of a node waiting for a person, as a host shows it.

    One question is answered with its value, several (a node with several outputs) with a mapping by question name.

    Attributes:
        id: What ``run.answer`` takes: the node's id, then ``@`` and each place
            of its item, then ``~`` and a fingerprint of the question asked
            there, ``"approve@1~3fa2c0d1"``. Inside a subgraph, the
            subgraph node and its item come first, ``"review@0/approve@1~3fa2c0d1"``.
            An answer for an id whose call now asks another question is
            refused as ``answer_stale``.
        loc: Which call it is, as a ``Problem``'s ``loc`` says it: the node
            that asks, then the place of its item in each Series the node
            runs per item of, ``("approve", 1)``; inside a subgraph,
            the subgraph node and its item first, ``("review", 0, "approve", 1)``.
        prompt: What the node's ``Ask`` said.
        questions: One per field the answer fills (each output of a node that
            asks in its own function, or its ``Asked`` parameter): the
            ``FieldDescription`` its ``describe()`` gives, with the ``choices``
            and ``default`` the ``Ask`` set and ``required`` worked out from them.
        inputs: What the node received, as JSON.
    """

    id: str
    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


def waiting_questions(compiled_node: CompiledFunctionNode, overrides: Mapping[str, QuestionOverrides]) -> list[FieldDescription]:
    """A waiting call's questions: the question about each field the answer fills, with what its ``Ask`` set.

    Args:
        compiled_node: The node that asks.
        overrides: What the ``Ask`` set, by field name.

    Returns:
        A question per field, in the fields' order.
    """
    fields = compiled_node.answer_fields
    return [
        _overridden(fields[name], question, overrides[name]) if name in overrides else question
        for name, question in compiled_node.questions.items()
    ]


def _overridden(field: Field, question: FieldDescription, overrides: QuestionOverrides) -> FieldDescription:
    """One field's question with what its ``Ask`` set, and whether it is ``required`` worked out again from that."""
    choices = overrides.get("choices", question.choices)
    return question.model_copy(update={**overrides, "required": answer_required(field, overrides, limited=choices is not None)})


def waiting_record(compiled_node: CompiledFunctionNode, loc: Loc, waiting: Unanswered) -> Waiting:
    """A waiting call as a host shows it. Its id is where the call is as text, then ``~`` and the fingerprint of the
    question asked there, as ``run.answer`` reads it.

    Args:
        compiled_node: The node that asks.
        loc: Where the call is: the node and its item, behind those of each subgraph's node around it.
        waiting: The call's outcome.
    """
    question = waiting.question
    return Waiting(
        id=f"{loc_text(loc)}~{question_fingerprint(compiled_node, waiting)}", loc=loc, prompt=question.prompt,
        questions=waiting_questions(compiled_node, question.overrides), inputs=question.inputs,
    )


def question_fingerprint(compiled_node: CompiledFunctionNode, waiting: Unanswered) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when
    its call asks another question. What the node received is all of its inputs, also for an ``Asked`` parameter
    whose function takes fewer: a change to any of them asks again."""
    inputs = waiting.question.inputs
    values = [inputs[name] for name in sorted(inputs)]  # saved, so the names may come back from a store in any order
    hashed = [compiled_node.id, compiled_node.code_fingerprint, values]
    written = canonical_json(hashed) if {type(v) for v in values} & {dict, list} else pydantic_core.to_json(hashed)  # plain values have no keys to sort
    return short_hash(written, 4)
