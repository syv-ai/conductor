"""``Asked``, a call that asked a person and what they replied, as a host keeps it for its records.

A host that keeps runs as a record of who decided what lists ``run.asked``: for every call a person answered or
declined, the question as they were shown it and their reply. Say ``approve(line: str) -> bool`` asked about the
lines ``a`` and ``b``, a person answered ``True`` for ``a`` and declined ``b`` as "Over budget". ``run.asked`` then
lists::

    Asked(question=Waiting(id="approve@0~3fa2c0d1", loc=("approve", 0), prompt="Approve a?", ...),
          answer={"result": True}, reason=None)
    Asked(question=Waiting(id="approve@1", loc=("approve", 1), prompt="Approve b?", ...),
          answer=None, reason="Over budget")

An answered call keeps its question in its ``Done`` (``Done.asked``), and a declined one where the run says how it
ended (``EndedEarly.declined``), so both read the same from a run read back.
"""

from __future__ import annotations

from typing import Any, cast

from conductor._internal.base import ConductorModel, parse_waiting_id, write_waiting_id

from .json import SavedRun, answer_to_json
from .outcome import Done
from .outcomes import RunOutcomes
from .state import LastResume
from .waiting import Waiting, waiting_of


class Asked(ConductorModel):
    """A call of a node that asked a person, and what they replied: an answer, or a decline.

    Attributes:
        question: What the person was shown: the call as ``run.waiting`` listed it while it waited, with the
            ``Ask``'s prompt, choices and default, and the call's inputs. Its ``id`` is the one the call waited
            under, for the node's code now. A declined call's is the call alone, such as ``"approve@1"``, since it
            waits on nothing.
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

    The answered calls come in node order, then item order, the declined one last, since it ended the run.

    Args:
        outcomes: The run's outcomes.
        last_resume: How the run's last ``resume()`` left it, which holds the question a decline said no to.
        codec: The graph's codec, which an answer's outputs are written by.
    """
    records = [
        Asked(
            question=waiting_of(node, item_indexes, outcome.asked, outcomes),
            answer=answer_to_json(codec, node, outcome),
            reason=None,
        )
        for node, item_indexes, outcome in outcomes.calls(Done)
        if outcome.asked is not None
    ]
    if last_resume is not None and last_resume != "running" and last_resume.declined is not None:
        cause = last_resume.cause
        _, node_path, item_indexes, _ = parse_waiting_id(write_waiting_id(cause.loc, None))
        node = outcomes.blueprint.function_nodes[node_path]
        records.append(
            Asked(
                question=waiting_of(node, item_indexes, last_resume.declined, outcomes),
                answer=None,
                # answer_declined's reason is a str, and left out when the person gave none
                reason=cast("str | None", cause.ctx.get("reason")),
            )
        )
    return records
