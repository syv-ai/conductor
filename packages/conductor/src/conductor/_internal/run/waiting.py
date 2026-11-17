"""``Waiting``, a call that is waiting for a person to answer, as a host is shown it.

When a node raises ``Ask``, its call stops and waits. The host lists those calls with ``run.waiting``, shows each
one to a person, and sends the answer back with ``run.answer(id, value)``. This module builds what the host is
shown. ``base/place.py`` writes the ``id`` that ties an answer to its call, and ``WaitingIdValidator`` reads it.

Say ``approve(line: str) -> bool`` raises ``Ask("approve b?")`` for the second line. ``run.waiting`` then lists::

    Waiting(id="approve@1~3fa2c0d1", loc=("approve", 1), prompt="approve b?", questions=[...], inputs={"line": "b"})

Each of its ``questions`` looks just like the output it fills does in ``describe()``, with what the ``Ask`` added.
So a host can draw a question with the same code that draws an input.
"""

from __future__ import annotations

from typing import Any, assert_never

from conductor._internal.ask import as_asked
from conductor._internal.base import ConductorModel, ItemIndexes, write_waiting_id
from conductor._internal.field import FieldDescription

from .arguments import ResultReader
from .blueprint import BlueprintFunctionNode, outcome_loc
from .call import jsonable
from .outcome import NoneInput, NotReady, Skipped, Unanswered
from .reuse import call_fingerprint


class Waiting(ConductorModel):
    """A call of a node that is waiting for a person to answer, as a host shows it.

    A node with one output asks one question, and is answered with a plain value. A node with several outputs asks
    one question per output, and is answered with a mapping by question name.

    Attributes:
        id: What ``run.answer`` takes, such as ``"approve@1~3fa2c0d1"``. It is the node's id, then ``@`` and the
            index of each item the call is for, then ``~`` and the start of the call's fingerprint. That is the
            fingerprint reuse matches calls by: a hash of the node's code fingerprint
            (``BlueprintFunctionNode.code_fingerprint`` says what that covers), the call's inputs and its item in
            every loop it is inside, not of the ``Ask``'s prompt, choices or default. Inside a subgraph, the
            subgraph node and its item come first: ``"review@0/approve@1~3fa2c0d1"``. If any of those change, the
            call may ask something else, so the old id stops matching. An answer to it is refused with
            ``answer_stale``. A call whose inputs or items JSON can't write has no fingerprint, and its id is the
            call alone: ``"approve@1"``.
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
    loc, outputs, choices, defaults = (
        outcome_loc(node, item_indexes),
        node.compiled_node.outputs,
        waiting.choices,
        waiting.defaults,
    )
    return Waiting(
        id=write_waiting_id(loc, question_fingerprint(node, item_indexes, reader)),
        loc=loc,
        prompt=waiting.prompt,
        questions=list(as_asked(outputs, choices, defaults).values()),
        inputs=jsonable(_arguments(node, item_indexes, reader)),
    )


def question_fingerprint(
    node: BlueprintFunctionNode, item_indexes: ItemIndexes, reader: ResultReader
) -> str | None:
    """Returns the fingerprint in a waiting call's id: the first eight hex digits of its call fingerprint.

    ``call_fingerprint`` says what it covers. If any of that changes, the call may ask something else, so the old
    id stops matching. It is ``None`` when the call has no fingerprint, because a value it covers can't be written
    as JSON. Such a call's id names just the call, as ``write_waiting_id`` writes it.
    """
    fingerprint = call_fingerprint(
        node, item_indexes, _arguments(node, item_indexes, reader), reader.outcomes
    )
    return None if fingerprint is None else fingerprint[:8]


def _arguments(
    node: BlueprintFunctionNode, item_indexes: ItemIndexes, reader: ResultReader
) -> dict[str, Any]:
    """Returns the arguments a waiting call was given.

    The arguments aren't stored with the call. They are worked out again from the upstream outcomes, which is safe
    because those outcomes don't change while the call waits.

    Raises:
        RuntimeError: The call comes out as skipped, has a ``None`` input or is not ready, so it has no arguments. That's a bug:
            the call did ask, and the outcomes it was given haven't changed since.
    """
    match reader.arguments_for(node, item_indexes):
        case dict() as arguments:
            return arguments
        case Skipped() | NoneInput() | NotReady.NOT_READY as none:
            raise RuntimeError(
                f"the waiting call {outcome_loc(node, item_indexes)} reads as {none!r}, which a call that asked cannot"
            )
        case unexpected:
            assert_never(unexpected)
