"""Under which condition does an output appear?

Outputs of one node that share an ``exclusive_group`` are exclusive alternatives:
exactly one is produced per run and the others carry ``SKIPPED``. The
engine does not need to know why a value was skipped; it only passes the
skip on. A caller reading a graph's outputs does need to know, and this
module is where it gets the answer, derived from the ``exclusive_group`` groups
in the nodes' interfaces and the edges — nothing else.

A ``Condition`` is a set of alternatives, each a set of decisions that
must all hold; an ``BranchTaken`` is one decision ("the ``exclusive_group`` group on node
N went to output O"). The output appears when any alternative holds::

    ALWAYS = frozenset({frozenset()})   # one empty alternative: always
    NEVER = frozenset()                  # no alternative at all

An input fed by several edges drops the skipped ones, so what hangs off
it appears when any one of its sources appeared — that is where a second
alternative comes from. An alternative naming one decision twice with different
outputs is dropped, since it can never hold.

A decision on a node that runs once holds back everything downstream. A node
that runs once per row may take one branch on some rows and the other on the
rest, so nothing downstream is switched off: its atom names the rows it is
decided on, and only a reader at or under those rows is held back by it. A
field's public ``condition`` is the run-once part, what a caller of a whole
run can read; the engine's answer about one row (can two fields hold a value
on one row?) reads the whole condition.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field

from conductor._internal.field_records import Output
from conductor._internal.graph.binding import From
from conductor._internal.ref import Ref


@dataclass(frozen=True, slots=True)
class BranchTaken:
    """One way one decision went: output ``output`` of the ``exclusive_group`` group on ``node_id``.

    Made by ``output_condition`` for every output with an ``exclusive_group``,
    and only ever used inside a ``Condition``. The engine never sees one: it
    propagates ``SKIPPED`` without knowing which decision caused it.
    """

    node_id: str
    exclusive_group: str
    output: str
    #: The index the decision is made once per row of; ``None`` when its node runs once.
    on_rows_of: Annotated[str | None, Field(exclude=True)] = field(default=None, repr=False)


Condition = frozenset[frozenset[BranchTaken]]

ALWAYS: Condition = frozenset({frozenset()})
NEVER: Condition = frozenset()

#: Past this many alternatives a condition stops following per-row decisions,
#: so what they would have kept apart compiles as if they weren't there.
_MOST_ALTERNATIVES_FOLLOWED = 32


def condition_for_reader_on_rows(condition: Condition, rows: frozenset[str]) -> Condition:
    """``condition`` as a reader on ``rows`` sees it: a per-row decision holds back only rows at or under its own, and a run-once decision every row."""
    return _keeping_only(condition, lambda branch: branch.on_rows_of is None or branch.on_rows_of in rows)


def run_once_condition(condition: Condition) -> Condition:
    """The run-once decisions of ``condition``, what a field's public ``condition`` says."""
    return _keeping_only(condition, lambda branch: branch.on_rows_of is None)


def _keeping_only(condition: Condition, keep: Callable[[BranchTaken], bool]) -> Condition:
    if condition is ALWAYS:
        return condition
    if all(keep(branch) for alternative in condition for branch in alternative):
        return condition
    return frozenset(frozenset(filter(keep, alternative)) for alternative in condition)


def condition_node_runs_under(bindings: Mapping[str, Any], condition_of: Callable[[Ref], Condition]) -> Condition:
    """The condition under which a node runs at all: every connected input's
    holds, and an input's holds when any of its sources' does."""
    node_condition = ALWAYS
    for binding in bindings.values():
        if isinstance(binding, From):
            refs = binding.refs
            node_condition = _all_of(node_condition, condition_of(refs[0]) if len(refs) == 1 else _any_of(condition_of(ref) for ref in refs))
    return node_condition


def output_condition(node_id: str, out: Output, node_condition: Condition, on_rows_of: str | None) -> Condition:
    """The condition of one output: its node's, and for an ``exclusive_group`` output that decision going this way."""
    if out.exclusive_group is None:
        return node_condition
    return _all_of(node_condition, frozenset({frozenset({BranchTaken(node_id, out.exclusive_group, out.name, on_rows_of)})}))


def _any_of(conditions: Iterable[Condition]) -> Condition:
    """Holds when any of ``conditions`` does: their alternatives, pooled."""
    conditions = tuple(conditions)
    if len(conditions) == 1:
        return conditions[0]
    joined: set[frozenset[BranchTaken]] = set()
    for condition in conditions:
        joined |= condition
    return _within_the_most_alternatives(frozenset(joined))


def _all_of(left: Condition, right: Condition) -> Condition:
    """Holds when both ``left`` and ``right`` do: every alternative of one joined
    with every alternative of the other, the impossible ones dropped."""
    if left is ALWAYS:
        return right
    if right is ALWAYS:
        return left
    return _within_the_most_alternatives(frozenset(
        alternative
        for left_alternative in left
        for right_alternative in right
        if (alternative := left_alternative | right_alternative) is not None and no_decision_goes_two_ways(alternative)
    ))


def _within_the_most_alternatives(condition: Condition) -> Condition:
    """``condition``, without its per-row decisions when it has more alternatives than compile follows: a chain of decisions merged per row doubles them at every step."""
    return run_once_condition(condition) if len(condition) > _MOST_ALTERNATIVES_FOLLOWED else condition


def can_never_both_hold(left: Condition, right: Condition) -> bool:
    """Is every way ``left`` can hold a way ``right`` can't? Then at most one of the two ever has a value."""
    return bool(left) and bool(right) and not any(no_decision_goes_two_ways(left_alternative | right_alternative) for left_alternative in left for right_alternative in right)


def no_decision_goes_two_ways(alternative: frozenset[BranchTaken]) -> bool:
    """No decision named twice with two different outputs."""
    taken: dict[tuple[str, str], str] = {}
    for branch_taken in alternative:
        if taken.setdefault((branch_taken.node_id, branch_taken.exclusive_group), branch_taken.output) != branch_taken.output:
            return False
    return True
