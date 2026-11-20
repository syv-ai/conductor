"""``Waiting``, a call that is waiting for a person to answer, as a host is shown it, and ``Question``, what it shows.

When a node raises ``Ask``, its call stops and waits. The host lists those calls with ``run.waiting``, shows each
one to a person, and sends the answer back with ``run.with_answers(id, value)``. This module builds what the host is
shown. ``base/place.py`` writes the ``id`` that ties an answer to its call, and ``WaitingIdValidator`` reads it.

Say ``approve(line: str) -> bool`` raises ``Ask("approve b?")`` for the second line. ``run.waiting`` then lists::

    Waiting(id="approve@1~3fa2c0d1", loc=("approve", 1), prompt="approve b?", questions=[...], inputs={"line": "b"})

Each of its ``questions`` looks just like the output it fills does in ``describe()``, with what the ``Ask`` added.
So a host can draw a question with the same code that draws an input.
"""

from __future__ import annotations

from typing import Any, assert_never

from conductor._internal.ask import as_asked
from conductor._internal.base import ConductorModel, ItemIndexes, json_hash, write_waiting_id
from conductor._internal.field import FieldDescription

from .blueprint import BlueprintFunctionNode, outcome_loc
from .call import jsonable
from .outcome import NoneInput, NotReady, Skipped, Unanswered
from .outcomes import RunOutcomes


class Question(ConductorModel):
    """A call of a node that asked a person, as the person is shown it: where it is, what it asks, and its inputs.

    A node with one output asks one question, and is answered with a plain value. A node with several outputs asks
    one question per output, and is answered with a mapping by question name.

    ``Waiting`` is a ``Question`` a person can still answer, with the id an answer is sent to. In ``run.asked`` a
    ``Question`` has no id: the call was answered or declined already, so the record is never an answer's target.

    Attributes:
        loc: Which call it is, written the way a ``Problem``'s ``loc`` is: ``("approve", 1)``. It is the node
            that asks, then the index of its item in each Series the node is looped over. Inside a subgraph, the
            subgraph node and its item come first: ``("review", 0, "approve", 1)``.
        prompt: The text the node's ``Ask`` gave.
        questions: One question per output of the node, since the answer fills the outputs. Each is the
            ``FieldDescription`` that ``describe()`` gives the output, with the ``choices`` and ``default`` the
            ``Ask`` set. Its ``required`` and ``default`` say what happens if the answer leaves it out.
        inputs: The values the call was given, as JSON.
    """

    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


class Waiting(Question):
    """A call of a node that is waiting for a person to answer, as a host shows it: a ``Question`` with the id to
    answer it by.

    Attributes:
        id: What ``run.with_answers`` takes, such as ``"approve@1~3fa2c0d1"``. It is the node's id, then ``@`` and the
            index of each item the call is for, then ``~`` and a fingerprint of the call. That is a hash of the
            fingerprint reuse matches calls by, taken when the call was made, and of the node's code fingerprint now
            (``BlueprintFunctionNode.code_fingerprint`` says what that covers). So it covers the node's code, the
            call's inputs and its item in every loop it is inside, not the ``Ask``'s prompt, choices or default.
            Inside a subgraph, the subgraph node and its item come first: ``"review@0/approve@1~3fa2c0d1"``. If any
            of those change, the call may ask something else, so the old id stops matching. An answer to it is
            refused with ``answer_stale``. A call whose inputs or items JSON can't write has no fingerprint, and its
            id is the call alone: ``"approve@1"``.
    """

    id: str


def question_of(
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    asked: Unanswered,
    outcomes: RunOutcomes,
    *,
    record: bool = False,
) -> Question:
    """Builds the ``Question`` a person is shown for one call that asked.

    Args:
        node: The node that asks.
        item_indexes: Which call of the node asked.
        asked: What the call's ``Ask`` said.
        outcomes: The run's outcomes, to find what the call was given.
        record: Keep the choices and defaults exactly as asked, for ``run.asked``, though the output's type refuses
            some of them now. Without it, the question is a form to answer, with only what the type takes.
    """
    return Question(
        loc=outcome_loc(node, item_indexes),
        prompt=asked.prompt,
        questions=list(
            as_asked(
                node.compiled_node.outputs, asked.choices, asked.defaults, record=record
            ).values()
        ),
        inputs=jsonable(_arguments(node, item_indexes, outcomes)),
    )


def waiting_of(
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    waiting: Unanswered,
    outcomes: RunOutcomes,
) -> Waiting:
    """Builds the ``Waiting`` a host is shown for one waiting call: its ``Question``, and the id to answer it by.

    Args:
        node: The node that asks.
        item_indexes: Which call of the node waits.
        waiting: The call's outcome, with what the ``Ask`` said.
        outcomes: The run's outcomes, to find what the call was given.
    """
    question = question_of(node, item_indexes, waiting, outcomes)
    return Waiting(
        id=write_waiting_id(question.loc, question_fingerprint(node, waiting)),
        **{name: getattr(question, name) for name in Question.model_fields},
    )


def question_fingerprint(node: BlueprintFunctionNode, waiting: Unanswered) -> str | None:
    """Returns the fingerprint in a waiting call's id: eight hex digits hashed from the call's fingerprint and the
    node's code fingerprint now.

    The call's fingerprint (``call_fingerprint`` says what it covers) was taken when the call was made, with the
    code of that time. The code fingerprint is the node's code now, so after a deploy that changes the node's code,
    an id written before it stops matching even though the call was made before it. If anything the id covers
    changes, the call may ask something else. It is ``None`` when the call has no fingerprint, because a value it
    covers can't be written as JSON. Such a call's id names just the call, as ``write_waiting_id`` writes it.
    """
    if waiting.fingerprint is None:
        return None
    return json_hash([waiting.fingerprint, node.code_fingerprint], 4)


def _arguments(
    node: BlueprintFunctionNode, item_indexes: ItemIndexes, outcomes: RunOutcomes
) -> dict[str, Any]:
    """Returns the arguments a waiting call was given, for the ``inputs`` a host is shown.

    The arguments aren't stored with the call, only their fingerprint: storing them would write every upstream value
    the call read, a whole table too, a second time in each saved run. So they are worked out again from the
    upstream outcomes, which is safe because those outcomes don't change while the call waits.

    Raises:
        RuntimeError: The call comes out as skipped, has a ``None`` input or is not ready, so it has no arguments. That's a bug:
            the call did ask, and the outcomes it was given haven't changed since.
    """
    match outcomes.arguments_for(node, item_indexes):
        case dict() as arguments:
            return arguments
        case Skipped() | NoneInput() | NotReady.NOT_READY as none:
            raise RuntimeError(
                f"the waiting call {outcome_loc(node, item_indexes)} reads as {none!r}, which a call that asked cannot"
            )
        case unexpected:
            assert_never(unexpected)
