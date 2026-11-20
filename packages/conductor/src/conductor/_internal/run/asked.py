"""``Asked``, a call that asked a person and what they replied, as a host keeps it for its records.

A host that keeps runs as a record of who decided what lists ``run.asked``: for every call a person answered or
declined, the question as they were shown it and their reply. Say ``approve(line: str) -> bool`` asked about the
lines ``a`` and ``b``, a person answered ``True`` for ``a`` and declined ``b`` as "Over budget". ``run.asked`` then
lists::

    Asked(question=Question(loc=("approve", 0), prompt="Approve a?", ...), answer={"result": True}, reason=None)
    Asked(question=Question(loc=("approve", 1), prompt="Approve b?", ...), answer=None, reason="Over budget")

A record has no id to answer by: its call was answered or declined already, so it is never an answer's target.

An answered call keeps its question in its ``Done`` (``Done.asked``), and a declined one where the run says how it
ended (``EndedEarly.declined``, a ``Declined``), so both read the same from a run read back.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.base import ConductorModel

from .json import SavedRun, answer_to_json
from .outcome import Done
from .outcomes import RunOutcomes
from .state import EndedEarly, LastResume
from .waiting import Question, question_of


class Asked(ConductorModel):
    """A call of a node that asked a person, and what they replied: an answer, or a decline.

    Attributes:
        question: What the person was shown: the call's ``Question``, with the ``Ask``'s prompt, choices and
            default, and the call's inputs, where ``run.waiting`` showed a ``Waiting``. It has no id, since the call
            was answered or declined already: a record is never an answer's target.
        answer: The outputs the answer gave, by output name, as JSON, as a saved run writes them. A node of one
            output has its value under ``result``, and a branch the answer left out isn't named. ``None`` when the
            person declined.
        reason: The reason the person gave with a decline. ``None`` for an answer, and for a decline with no
            reason.
    """

    question: Question
    answer: dict[str, Any] | None
    reason: str | None


def asked_of(outcomes: RunOutcomes, last_resume: LastResume, codec: type[SavedRun]) -> list[Asked]:
    """Returns every call of a run that a person answered or declined, as ``run.asked`` lists them.

    The answered calls come in the graph's node order, the order a run tries its nodes, each after the nodes
    upstream of it, then in item order; not in the order the answers came. The declined call comes last, since it
    ended the run.

    Args:
        outcomes: The run's outcomes.
        last_resume: How the run's last ``resume()`` left it, which holds the call a decline said no to.
        codec: The graph's codec, which an answer's outputs are written by.
    """
    records = [
        Asked(
            question=question_of(node, item_indexes, outcome.asked, outcomes, record=True),
            answer=answer_to_json(codec, node, outcome),
            reason=None,
        )
        for node, item_indexes, outcome in outcomes.calls(Done)
        if outcome.asked is not None
    ]
    if isinstance(last_resume, EndedEarly) and (declined := last_resume.declined) is not None:
        node = outcomes.blueprint.function_nodes[declined.node_path]
        records.append(
            Asked(
                question=question_of(
                    node, declined.item_indexes, declined.question, outcomes, record=True
                ),
                answer=None,
                reason=declined.reason,
            )
        )
    return records
