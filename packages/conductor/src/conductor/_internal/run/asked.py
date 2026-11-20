"""``Asked``, a call that asked a person and what they replied, as a host keeps it for its records.

A host that keeps runs as a record of who decided what lists ``run.asked``: for every call a person answered or
declined, the question as they were shown it and their reply. Say ``approve(line: str) -> bool`` asked about the
lines ``a`` and ``b``, a person answered ``True`` for ``a`` and declined ``b`` as "Over budget". ``run.asked`` then
lists::

    Asked(question=Waiting(id="approve@0", loc=("approve", 0), prompt="Approve a?", ...),
          answer={"result": True}, reason=None)
    Asked(question=Waiting(id="approve@1", loc=("approve", 1), prompt="Approve b?", ...),
          answer=None, reason="Over budget")

An answered call keeps its question in its ``Done`` (``Done.asked``), and a declined one where the run says how it
ended (``EndedEarly.declined``, a ``Declined``), so both read the same from a run read back.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from conductor._internal.base import ConductorModel, ItemIndexes

from .blueprint import BlueprintFunctionNode
from .json import SavedRun, answer_to_json
from .outcome import Done, Unanswered
from .outcomes import RunOutcomes
from .state import EndedEarly, LastResume
from .waiting import Waiting, waiting_of


class Asked(ConductorModel):
    """A call of a node that asked a person, and what they replied: an answer, or a decline.

    Attributes:
        question: What the person was shown: the call as a ``Waiting``, with the ``Ask``'s prompt, choices and
            default, and the call's inputs. Its ``id`` is the call alone, such as ``"approve@1"``, with no ``~``
            part: no call here waits, so the id stays the same after a deploy and is never taken for a waiting id.
        answer: The outputs the answer gave, by output name, as JSON, as a saved run writes them. A node of one
            output has its value under ``result``, and a branch the answer left out isn't named. ``None`` when the
            person declined.
        reason: The reason the person gave with a decline. ``None`` for an answer, and for a decline with no
            reason.
    """

    question: Waiting
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
            question=_as_shown(outcomes, node, item_indexes, outcome.asked),
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
                question=_as_shown(outcomes, node, declined.item_indexes, declined.question),
                answer=None,
                reason=declined.reason,
            )
        )
    return records


def _as_shown(
    outcomes: RunOutcomes,
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    question: Unanswered,
) -> Waiting:
    """Returns the question as the person was shown it, with the call alone for its id.

    No call of the record waits, so its id has no ``~`` part: it names the call, stays the same after a deploy, and
    is never taken for a waiting id.
    """
    return waiting_of(node, item_indexes, dataclasses.replace(question, fingerprint=None), outcomes)
