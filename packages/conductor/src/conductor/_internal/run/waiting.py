"""``Waiting``, a call that is waiting for a person to answer, as a host is shown it.

When a node raises ``Ask``, its call stops and waits. The host lists those calls with ``run.waiting``, shows each
one to a person, and sends the answer back with ``run.answer(id, value)``. This module builds what the host is
shown, and writes and reads the ``id`` that ties an answer to its call.

Say ``approve(line: str) -> bool`` raises ``Ask("approve b?")`` for the second line. ``run.waiting`` then lists::

    Waiting(id="approve@1~3fa2c0d1", loc=("approve", 1), prompt="approve b?", questions=[...], inputs={"line": "b"})

Each of its ``questions`` looks just like the output it fills does in ``describe()``, with what the ``Ask`` added.
So a host can draw a question with the same code that draws an input.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_never

from conductor._internal.ask import as_asked
from conductor._internal.base import ConductorModel, ItemIndexes, Loc, NodePath, json_hash
from conductor._internal.field import FieldDescription
from conductor._internal.graph import CompiledFunctionNode
from conductor._internal.problem import InvalidCallError, call_problems

from .arguments import ResultReader
from .blueprint import BlueprintFunctionNode, outcome_loc
from .call import jsonable
from .outcome import NoneInput, NotReady, Skipped, Unanswered


class Waiting(ConductorModel):
    """A call of a node that is waiting for a person to answer, as a host shows it.

    A node with one output asks one question, and is answered with a plain value. A node with several outputs asks
    one question per output, and is answered with a mapping by question name.

    Attributes:
        id: What ``run.answer`` takes, such as ``"approve@1~3fa2c0d1"``. It is the node's id, then ``@`` and the
            index of each item the call is for, then ``~`` and a short fingerprint. The fingerprint is a hash of the
            node's id, its code and the call's inputs, not of the ``Ask``'s prompt, choices or default. Inside a
            subgraph, the subgraph node and its item come first: ``"review@0/approve@1~3fa2c0d1"``. If the call's
            inputs or the node's code change, it may ask something else, so the old id stops matching. An answer
            to it is refused with ``answer_stale``.
        loc: Which call it is, written the way a ``Problem``'s ``loc`` is: ``("approve", 1)``. It is the node
            that asks, then the index of its item in each Series the node is looped over. Inside a subgraph, the
            subgraph node and its item come first: ``("review", 0, "approve", 1)``.
        prompt: The text the node's ``Ask`` gave.
        questions: One question per output of the node, since the answer fills the outputs. Each is the
            ``FieldDescription`` that ``describe()`` gives the output, with the ``choices`` and ``default`` the
            ``Ask`` set. Its ``required`` and ``default`` say what happens if the answer leaves it out.
        inputs: The values the call was given, as JSON.
    """

    id: str
    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


def waiting_record(
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    waiting: Unanswered,
    reader: ResultReader,
) -> Waiting:
    """Builds the ``Waiting`` a host is shown for one waiting call.

    Args:
        node: The node that asks.
        item_indexes: Which call of the node waits.
        waiting: The call's outcome, with what the ``Ask`` said.
        reader: Looks up the run's outcomes, to find what the call was given.
    """
    compiled_node, inputs = node.compiled_node, received(node, item_indexes, reader)
    loc, outputs, choices, defaults = (
        outcome_loc(node, item_indexes),
        compiled_node.outputs,
        waiting.choices,
        waiting.defaults,
    )
    return Waiting(
        id=write_waiting_id(loc, question_fingerprint(compiled_node, inputs)),
        loc=loc,
        prompt=waiting.prompt,
        questions=list(as_asked(outputs, choices, defaults).values()),
        inputs=inputs,
    )


def received(
    node: BlueprintFunctionNode, item_indexes: ItemIndexes, reader: ResultReader
) -> dict[str, Any]:
    """Returns the arguments a waiting call was given, as JSON.

    The arguments aren't stored with the call. They are worked out again from the upstream outcomes, which is safe
    because those outcomes don't change while the call waits. For example, ``received(approve, (1,), reader)`` is
    ``{"line": "b"}``.

    Args:
        node: The node that asks.
        item_indexes: Which call of the node waits.
        reader: Looks up the run's outcomes.

    Raises:
        RuntimeError: The call comes out as skipped, not called or not ready, so it has no arguments. That's a bug:
            the call did ask, and the outcomes it was given haven't changed since.
    """
    match reader.arguments_for(node, item_indexes):
        case dict() as arguments:
            inputs: dict[str, Any] = jsonable(arguments)
            return inputs
        case Skipped() | NoneInput() | NotReady.NOT_READY as none:
            raise RuntimeError(
                f"the waiting call {outcome_loc(node, item_indexes)} reads as {none!r}, which a call that asked cannot"
            )
        case unexpected:
            assert_never(unexpected)


def question_fingerprint(compiled_node: CompiledFunctionNode, inputs: dict[str, Any]) -> str:
    """Returns a short fingerprint of a question: eight hex digits.

    It is a hash of the node that asks, its code, and the inputs the call was given (as ``received`` returns them).
    The fingerprint is part of the waiting id. If any of those change, the call may ask something else, so the old
    id stops matching. Every input counts, so a change to any one of them asks again.
    """
    return json_hash([compiled_node.id, compiled_node.code_fingerprint, inputs], 4)


def write_waiting_id(loc: Loc, fingerprint: str | None) -> str:
    """Returns the waiting id of the call at ``loc``, whose question has the fingerprint ``fingerprint``.

    The node ids are joined by ``/``. Each is followed by ``@`` and its item indexes, and the fingerprint comes last,
    after ``~``. So ``write_waiting_id(("review", 0, "approve", 1), "3fa2c0d1")`` returns
    ``"review@0/approve@1~3fa2c0d1"``. Without a fingerprint, the id names just the call: ``"review@0/approve@1"``.
    ``read_waiting_id`` turns the id back into its parts.
    """
    place = "".join(f"@{part}" if isinstance(part, int) else f"/{part}" for part in loc)[1:]
    return place if fingerprint is None else f"{place}~{fingerprint}"


def read_waiting_id(
    waiting_id: str, *, status: Callable[[], str]
) -> tuple[Loc, NodePath, ItemIndexes, str | None]:
    """Splits a waiting id back into the call it names and the fingerprint of its question.

    It undoes ``write_waiting_id``. For example, ``read_waiting_id("review@0/approve@1~3fa2c0d1", status=lambda:
    "waiting")`` returns ``(("review", 0, "approve", 1), ("review", "approve"), (0, 1), "3fa2c0d1")``. An id with no
    ``~`` part names just the call, and its fingerprint is ``None``.

    Only the exact text ``write_waiting_id`` would write is accepted. ``"approve@01"`` names nothing, since the
    index would be written ``1``.

    Args:
        waiting_id: The id, as a host gave it.
        status: Returns the run's status, for the refusal's message. It is only called when the id is refused.

    Returns:
        The call's loc, its node path, its item indexes, and the fingerprint.

    Raises:
        InvalidCallError: ``not_waiting`` if the id isn't one ``write_waiting_id`` could have written. When the
            text after an ``@`` isn't a whole number, the problem's loc ends at the node that index belongs to.
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
            raise InvalidCallError(
                [call_problems.not_waiting(*loc, node_id, waiting_id=waiting_id, status=status())]
            ) from None
        loc += [node_id, *read]
        item_indexes += read
    read_fingerprint = fingerprint or None
    if write_waiting_id(tuple(loc), read_fingerprint) != waiting_id:
        raise InvalidCallError(
            [call_problems.not_waiting(*loc, waiting_id=waiting_id, status=status())]
        )
    return tuple(loc), tuple(node_path), tuple(item_indexes), read_fingerprint
