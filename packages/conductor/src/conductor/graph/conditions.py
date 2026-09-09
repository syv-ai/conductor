"""Under which condition does an output appear?

Outputs of one node that share a ``choice`` are exclusive alternatives:
exactly one is produced per run and the others carry ``SKIPPED``. The
engine does not need to know why a value was skipped; it only passes the
skip on. A caller reading a flow's outputs does need to know, and this
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
once per row (see ``iteration``) may take one branch on some rows and the
other on the rest, so each branch output is a series missing the rows
that went the other way. Nothing downstream is switched off, so such a
node contributes no atom.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from conductor.graph.binding import Edges
from conductor.graph.model import GraphNode
from conductor.interface import Interface
from conductor.ref import Ref
from conductor.series import Index


@dataclass(frozen=True, slots=True)
class Atom:
    """One way one decision went: output ``output`` of the ``choice`` group on ``node_id``.

    Made by ``conditions_of`` for every output with a ``choice`` on a node
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


def conditions_of(
    nodes: Sequence[GraphNode],
    interfaces: Mapping[str, Interface],
    iterated: Mapping[str, Index | None],
) -> dict[Ref, Condition]:
    """The condition of every output of every node in ``iterated`` — the nodes the edge walk resolved.

    Walks ``nodes`` in execution order. A node's own condition holds
    when every connected input's does, and an input's holds when any of
    its sources' does. An output with a ``choice`` on a node that runs
    once adds its own decision.
    """
    conditions: dict[Ref, Condition] = {}
    for node in nodes:
        if node.id not in iterated:
            continue
        at_node = ALWAYS
        for binding in node.bindings.values():
            if isinstance(binding, Edges) and binding.refs:
                at_node = _all_of(at_node, _any_of(conditions.get(ref, ALWAYS) for ref in binding.refs))
        for out in interfaces[node.id].outputs:
            gates = out.choice is not None and iterated[node.id] is None
            conditions[Ref(node.id, out.name)] = (
                _all_of(at_node, frozenset({frozenset({Atom(node.id, out.choice, out.name)})})) if gates else at_node
            )
    return conditions


def _any_of(conditions: Iterable[Condition]) -> Condition:
    """Holds when any of ``conditions`` does: their alternatives, pooled."""
    joined: set[frozenset[Atom]] = set()
    for condition in conditions:
        joined |= condition
    return frozenset(joined)


def _all_of(a: Condition, b: Condition) -> Condition:
    """Holds when both ``a`` and ``b`` do: every alternative of one joined
    with every alternative of the other, the impossible ones dropped."""
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
