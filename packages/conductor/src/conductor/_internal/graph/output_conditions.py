"""Under which condition does an output appear?

Outputs of one node that share a ``choice`` are exclusive alternatives:
exactly one is produced per run and the others carry ``SKIPPED``. The
engine does not need to know why a value was skipped; it only passes the
skip on. A caller reading a graph's outputs does need to know, and this
module is where it gets the answer, derived from the ``choice`` groups
in the nodes' interfaces and the edges — nothing else.

A ``Condition`` is a set of alternatives, each a set of decisions that
must all hold; an ``BranchTaken`` is one decision ("the ``choice`` group on node
N went to output O"). The output appears when any alternative holds::

    ALWAYS = frozenset({frozenset()})   # one empty alternative: always
    NEVER = frozenset()                  # no alternative at all

An input fed by several edges drops the skipped ones, so what hangs off
it appears when any one of its sources appeared — that is where a second
alternative comes from. An alternative naming one decision twice with different
outputs is dropped, since it can never hold.

Only a decision on a node that runs once holds anything back. A node that runs
once per row may take one branch on some rows and the
other on the rest, so each branch output is a series missing the rows
that went the other way. Nothing downstream is switched off, so such a
node contributes no atom.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from conductor._internal.field_records import Output
from conductor._internal.graph.binding import From
from conductor._internal.ref import Ref


@dataclass(frozen=True, slots=True)
class BranchTaken:
    """One way one decision went: output ``output`` of the ``choice`` group on ``node_id``.

    Made by ``output_condition`` for every output with a ``choice`` on a node
    that runs once, and only ever used inside a ``Condition``. The
    engine never sees one: it propagates ``SKIPPED`` without knowing which
    decision caused it.
    """

    node_id: str
    choice: str
    output: str


Condition = frozenset[frozenset[BranchTaken]]

ALWAYS: Condition = frozenset({frozenset()})
NEVER: Condition = frozenset()


def condition_node_runs_under(bindings: Mapping[str, Any], condition_of: Callable[[Ref], Condition]) -> Condition:
    """The condition under which a node runs at all: every connected input's
    holds, and an input's holds when any of its sources' does."""
    node_condition = ALWAYS
    for binding in bindings.values():
        if isinstance(binding, From):
            refs = binding.refs
            node_condition = _all_of(node_condition, condition_of(refs[0]) if len(refs) == 1 else _any_of(condition_of(ref) for ref in refs))
    return node_condition


def output_condition(node_id: str, out: Output, node_condition: Condition, runs_once: bool) -> Condition:
    """The condition of one output: its node's, and for a ``choice`` output on
    a node that runs once, that decision going this way."""
    if out.choice is None or not runs_once:
        return node_condition
    return _all_of(node_condition, frozenset({frozenset({BranchTaken(node_id, out.choice, out.name)})}))


def _any_of(conditions: Iterable[Condition]) -> Condition:
    """Holds when any of ``conditions`` does: their alternatives, pooled."""
    conditions = tuple(conditions)
    if len(conditions) == 1:
        return conditions[0]
    joined: set[frozenset[BranchTaken]] = set()
    for condition in conditions:
        joined |= condition
    return frozenset(joined)


def _all_of(left: Condition, right: Condition) -> Condition:
    """Holds when both ``left`` and ``right`` do: every alternative of one joined
    with every alternative of the other, the impossible ones dropped."""
    if left is ALWAYS:
        return right
    if right is ALWAYS:
        return left
    return frozenset(
        alternative
        for left_alternative in left
        for right_alternative in right
        if (alternative := left_alternative | right_alternative) is not None and no_decision_goes_two_ways(alternative)
    )


def no_decision_goes_two_ways(alternative: frozenset[BranchTaken]) -> bool:
    """No decision named twice with two different outputs."""
    taken: dict[tuple[str, str], str] = {}
    for branch_taken in alternative:
        if taken.setdefault((branch_taken.node_id, branch_taken.choice), branch_taken.output) != branch_taken.output:
            return False
    return True
