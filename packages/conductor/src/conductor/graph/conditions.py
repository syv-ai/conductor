"""Under which condition does an output appear?

Outputs of one node that share a ``choice`` are exclusive alternatives:
exactly one is produced per run and the others carry ``SKIPPED``. The
engine does not need to know why a value was skipped; it only passes the
skip on. A caller reading a graph's outputs does need to know, and this
module is where it gets the answer, derived from the ``choice`` groups
in the nodes' interfaces and the edges — nothing else.

A ``Condition`` is a set of alternatives, each a set of decisions that
must all hold; an ``Atom`` is one decision ("the ``choice`` group on node
N went to output O"). The output appears when any alternative holds::

    ALWAYS = frozenset({frozenset()})   # one empty alternative: no gate
    NEVER = frozenset()                  # no alternative at all

An input fed by several edges drops the skipped ones, so what hangs off
it appears when any one of its sources appeared — that is where a second
alternative comes from. An alternative naming one decision twice with different
outputs is dropped, since it can never hold.

Only a decision on a node that runs once gates anything. A node that runs
once per row may take one branch on some rows and the
other on the rest, so each branch output is a series missing the rows
that went the other way. Nothing downstream is switched off, so such a
node contributes no atom.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from conductor.field_records import Output
from conductor.graph.binding import From
from conductor.ref import Ref


@dataclass(frozen=True, slots=True)
class Atom:
    """One way one decision went: output ``output`` of the ``choice`` group on ``node_id``.

    Made by ``output_condition`` for every output with a ``choice`` on a node
    that runs once, and only ever used inside a ``Condition``. The
    engine never sees one: it propagates ``SKIPPED`` without knowing which
    decision caused it.
    """

    node_id: str
    choice: str
    output: str


Condition = frozenset[frozenset[Atom]]

ALWAYS: Condition = frozenset({frozenset()})
NEVER: Condition = frozenset()


def condition_at_node(bindings: Mapping[str, Any], condition_of: Callable[[Ref], Condition]) -> Condition:
    """The condition under which a node runs at all: every connected input's
    holds, and an input's holds when any of its sources' does."""
    at_node = ALWAYS
    for binding in bindings.values():
        if isinstance(binding, From):
            refs = binding.refs
            at_node = _all_of(at_node, condition_of(refs[0]) if len(refs) == 1 else _any_of(condition_of(ref) for ref in refs))
    return at_node


def output_condition(node_id: str, out: Output, at_node: Condition, runs_once: bool) -> Condition:
    """The condition of one output: its node's, and for a ``choice`` output on
    a node that runs once, that decision going this way."""
    if out.choice is None or not runs_once:
        return at_node
    return _all_of(at_node, frozenset({frozenset({Atom(node_id, out.choice, out.name)})}))


def _any_of(conditions: Iterable[Condition]) -> Condition:
    """Holds when any of ``conditions`` does: their alternatives, pooled."""
    conditions = tuple(conditions)
    if len(conditions) == 1:
        return conditions[0]
    joined: set[frozenset[Atom]] = set()
    for condition in conditions:
        joined |= condition
    return frozenset(joined)


def _all_of(a: Condition, b: Condition) -> Condition:
    """Holds when both ``a`` and ``b`` do: every alternative of one joined
    with every alternative of the other, the impossible ones dropped."""
    if a is ALWAYS:
        return b
    if b is ALWAYS:
        return a
    return frozenset(
        alternative
        for x in a
        for y in b
        if (alternative := x | y) is not None and _consistent(alternative)
    )


def _consistent(alternative: frozenset[Atom]) -> bool:
    """No decision named twice with two different outputs."""
    taken: dict[tuple[str, str], str] = {}
    for atom in alternative:
        if taken.setdefault((atom.node_id, atom.choice), atom.output) != atom.output:
            return False
    return True
