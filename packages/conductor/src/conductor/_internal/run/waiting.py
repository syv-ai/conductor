"""A call waiting for a person, as a host is shown it: ``Waiting``, with the id ``run.answer`` takes.

Example: ``approve(line: str) -> bool`` raises ``Ask("approve b?")`` for the second line. ``run.waiting`` then lists
``Waiting(id="approve@1~3fa2c0d1", loc=("approve", 1), prompt="approve b?", questions=[...], inputs={"line": "b"})``.
Each of its ``questions`` is the field record ``describe()`` gives the field the answer fills, with what the ``Ask``
set, so a host draws a question with the code that draws an input.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.base import ConductorModel, ItemIndexes, Loc, NodePath, json_hash
from conductor._internal.compiled import CompiledFunctionNode
from conductor._internal.field import FieldDescription
from conductor._internal.problem import InvalidCallError, call_problems

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
        questions: One per output of the node, which the answer fills: the
            ``FieldDescription`` its ``describe()`` gives, with the ``choices``
            and ``default`` the ``Ask`` set and ``required`` worked out from them.
        inputs: What the node received, as JSON.
    """

    id: str
    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


def waiting_record(compiled_node: CompiledFunctionNode, loc: Loc, waiting: Unanswered) -> Waiting:
    """A waiting call as a host shows it.

    Args:
        compiled_node: The node that asks.
        loc: Where the call is: the node and its item, behind those of each subgraph's node around it.
        waiting: The call's outcome.
    """
    asked, outputs = waiting.question, compiled_node.outputs
    return Waiting(
        id=write_waiting_id(loc, question_fingerprint(compiled_node, waiting)), loc=loc, prompt=asked.prompt,
        questions=[asked.as_asked(outputs[name], question) for name, question in compiled_node.questions.items()],
        inputs=asked.inputs,
    )


def question_fingerprint(compiled_node: CompiledFunctionNode, waiting: Unanswered) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when
    its call asks another question. What the node received is all of its inputs: a change to any of them asks
    again."""
    return json_hash([compiled_node.id, compiled_node.code_fingerprint, waiting.question.inputs], 4)


def write_waiting_id(loc: Loc, fingerprint: str | None) -> str:
    """The id of the call at ``loc`` waiting with the question whose fingerprint is ``fingerprint``.

    Its node ids are joined by ``/``, each followed by ``@`` and each of its item indexes, then ``~`` and the
    fingerprint: ``write_waiting_id(("review", 0, "approve", 1), "3fa2c0d1")`` returns
    ``"review@0/approve@1~3fa2c0d1"``. Without a fingerprint it is the call alone, ``"review@0/approve@1"``.
    ``read_waiting_id`` reads it back.
    """
    place = "".join(f"@{part}" if isinstance(part, int) else f"/{part}" for part in loc)[1:]
    return place if fingerprint is None else f"{place}~{fingerprint}"


def read_waiting_id(waiting_id: str, *, status: str) -> tuple[Loc, NodePath, ItemIndexes, str | None]:
    """The call a waiting id names, as ``write_waiting_id`` wrote it, and the fingerprint of its question.

    ``read_waiting_id("review@0/approve@1~3fa2c0d1", status="waiting")`` returns ``(("review", 0, "approve", 1),
    ("review", "approve"), (0, 1), "3fa2c0d1")``. An id without ``~`` and a fingerprint names the call alone, and
    its fingerprint is ``None``. Only the text ``write_waiting_id`` writes is read: ``"approve@01"`` names nothing.

    Args:
        waiting_id: The id, as a host gave it.
        status: The run's status, which a refusal names.

    Returns:
        The call's loc, its node path and its item indexes, and the fingerprint.

    Raises:
        InvalidCallError: A text after an ``@`` is not a whole number (``not_waiting``, at the loc read up to the
            node it follows), or the id is not the text ``write_waiting_id`` writes for what it names (``not_waiting``).
    """
    place, _, fingerprint = waiting_id.partition("~")
    loc: list[str | int] = []
    node_path: list[str] = []
    item_indexes: list[int] = []
    for part in place.split("/"):
        node_id, *indexes = part.split("@")
        node_path.append(node_id)
        try:
            read = [int(index) for index in indexes]
        except ValueError:
            raise InvalidCallError([call_problems.not_waiting(*loc, node_id, waiting_id=waiting_id, status=status)]) from None
        loc += [node_id, *read]
        item_indexes += read
    read_fingerprint = fingerprint or None
    if write_waiting_id(tuple(loc), read_fingerprint) != waiting_id:
        raise InvalidCallError([call_problems.not_waiting(*loc, waiting_id=waiting_id, status=status)])
    return tuple(loc), tuple(node_path), tuple(item_indexes), read_fingerprint
