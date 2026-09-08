"""Under which condition does an output appear?

Outputs of one node that share a ``choice`` are exclusive alternatives:
exactly one is produced per run and the others carry ``SKIPPED``. The
engine propagates the skip without knowing why. A caller reading a flow's
outputs does need to know why, and this module derives it from the
``choice`` groups in the rosters and the edges — nothing else.

A ``Condition`` is a boolean formula in disjunctive normal form: a set of
conjunctions, each a set of atoms ("the ``choice`` group on node N went
to output O")::

    ALWAYS = frozenset({frozenset()})   # one empty conjunction: no gate
    NEVER = frozenset()                  # no conjunction at all

An input fed by several edges drops the skipped ones, so what hangs off
it appears when *any* source did — that is where a disjunction comes
from. A conjunction naming one decision twice with different outputs is
dropped, since it can never hold.

Only a decision on a node that runs once gates anything. A node that runs
once per row (see ``lifting``) produces both branches as series with
gaps and picks rows instead; it contributes no atom.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from conductor.graph.binding import Edges
from conductor.graph.model import GraphNode
from conductor.metadata import Roster
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
    rosters: Mapping[str, Roster],
    lifted: Mapping[str, Index | None],
) -> dict[Ref, Condition]:
    """The condition of every output of every node in ``lifted`` — the nodes the edge walk resolved.

    Walks ``nodes`` in topological order. A node's own condition is the
    conjunction over its connected inputs, each input being the disjunction of
    its sources' conditions. An output with a ``choice`` on a node that
    runs once adds its own atom.
    """
    conditions: dict[Ref, Condition] = {}
    for node in nodes:
        if node.id not in lifted:
            continue
        at_node = ALWAYS
        for binding in node.bindings.values():
            if isinstance(binding, Edges) and binding.refs:
                at_node = _and(at_node, _or(conditions.get(ref, ALWAYS) for ref in binding.refs))
        for out in rosters[node.id].outputs:
            gates = out.choice is not None and lifted[node.id] is None
            conditions[Ref(node.id, out.name)] = (
                _and(at_node, frozenset({frozenset({Atom(node.id, out.choice, out.name)})})) if gates else at_node
            )
    return conditions


def _or(conditions: Iterable[Condition]) -> Condition:
    joined: set[frozenset[Atom]] = set()
    for condition in conditions:
        joined |= condition
    return frozenset(joined)


def _and(a: Condition, b: Condition) -> Condition:
    return frozenset(
        conjunction
        for x in a
        for y in b
        if (conjunction := x | y) is not None and _consistent(conjunction)
    )


def _consistent(conjunction: frozenset[Atom]) -> bool:
    """No decision named twice with two different outputs."""
    taken: dict[tuple[str, str], str] = {}
    for atom in conjunction:
        if taken.setdefault((atom.node_id, atom.choice), atom.output) != atom.output:
            return False
    return True
