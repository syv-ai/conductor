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
once per row (see ``iteration``) may take one branch on some rows and the
other on the rest, so each branch output is a series missing the rows
that went the other way. Nothing downstream is switched off, so such a
node contributes no atom.

A graph may be placed inside another, and there what arrives on one of
its inputs may itself be gated. So each input the graph offers stands for
a decision of its own, ``Atom(INPUT, address, address)``: "what was
connected to this input appeared". Compile records every output's
condition over those. A caller reading ``CompiledField.condition`` sees
every input holding; the graph that places this one swaps in, with
``substituted``, the condition of whatever it connected there.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from conductor.graph.binding import From
from conductor.graph.model import GraphNode
from conductor.interface import Interface
from conductor.ref import Ref
from conductor.series import Index


@dataclass(frozen=True, slots=True)
class Atom:
    """One way one decision went: output ``output`` of the ``choice`` group on ``node_id``.

    Made by ``conditions_of`` for every output with a ``choice`` on a node
    that runs once, and for every input the graph offers (``INPUT``), and
    only ever used inside a ``Condition``. The
    engine never sees one: it propagates ``SKIPPED`` without knowing which
    decision caused it.
    """

    node_id: str
    choice: str
    output: str


Condition = frozenset[frozenset[Atom]]

ALWAYS: Condition = frozenset({frozenset()})
NEVER: Condition = frozenset()

#: The node id of the decisions that stand for a graph's inputs. A node id
#: holding ``/`` is refused (``invalid_node_id``), so no node the author
#: placed can be mistaken for it.
INPUT = "/input"


def conditions_of(
    nodes: Sequence[GraphNode],
    interfaces: Mapping[str, Interface],
    iterated: Mapping[str, Index | None],
    placeholders: Mapping[Ref, str],
) -> dict[Ref, Condition]:
    """The condition of every output of every node in ``iterated`` — the nodes the edge walk resolved.

    Walks ``nodes`` in execution order. A node's own condition holds
    when every connected input's does, and an input's holds when any of
    its sources' does. An output with a ``choice`` on a node that runs
    once adds its own decision. ``placeholders`` names each input the graph
    offers, by the field it lands on, with its address in the graph's
    interface; each is one more input that must hold, ``Atom(INPUT, address, address)``.
    """
    offered: dict[str, list[str]] = {}
    for ref, address in placeholders.items():
        offered.setdefault(ref.node_id, []).append(address)
    conditions: dict[Ref, Condition] = {}
    for node in nodes:
        if node.id not in iterated:
            continue
        at_node = ALWAYS
        for binding in node.bindings.values():
            if isinstance(binding, From) and binding.refs:
                at_node = _all_of(at_node, _any_of(conditions.get(ref, ALWAYS) for ref in binding.refs))
        for address in offered.get(node.id, ()):
            at_node = _all_of(at_node, frozenset({frozenset({Atom(INPUT, address, address)})}))
        for out in interfaces[node.id].outputs:
            gates = out.choice is not None and iterated[node.id] is None
            conditions[Ref(node.id, out.name)] = (
                _all_of(at_node, frozenset({frozenset({Atom(node.id, out.choice, out.name)})})) if gates else at_node
            )
    return conditions


def substituted(condition: Condition, inputs: Mapping[str, Condition], prefix: str, per_row: bool) -> Condition:
    """``condition``, recorded inside a graph over its inputs, as it stands where the graph is placed.

    Each input's decision becomes ``inputs[address]``, the condition of
    what the outer graph connected there; an input nothing gates, or one
    missing from ``inputs``, always holds. Each of the graph's own
    decisions gets ``prefix`` in front of its node id, the placed node's
    ``"<id>/"``, or always holds when ``per_row``: placed on a node that
    runs once per row, the graph's decisions mask rows and gate nothing.
    With no inputs, no prefix and not per row, it is ``condition`` with
    every input holding, which is what ``CompiledField.condition`` answers.
    """
    result: Condition = NEVER
    for alternative in condition:
        joined = ALWAYS
        for atom in alternative:
            if atom.node_id == INPUT:
                part = inputs.get(atom.choice, ALWAYS)
            elif per_row:
                continue
            else:
                part = frozenset({frozenset({Atom(prefix + atom.node_id, atom.choice, atom.output)})})
            joined = _all_of(joined, part)
        result = _any_of((result, joined))
    return result


def _any_of(conditions: Iterable[Condition]) -> Condition:
    """Holds when any of ``conditions`` does: their alternatives, pooled."""
    pooled = list(conditions)
    if len(pooled) == 1:
        return pooled[0]
    joined: set[frozenset[Atom]] = set()
    for condition in pooled:
        joined |= condition
    return _without_longer_input_alternatives(frozenset(joined))


def _all_of(a: Condition, b: Condition) -> Condition:
    """Holds when both ``a`` and ``b`` do: every alternative of one joined
    with every alternative of the other, the impossible ones dropped.
    Joined with ``ALWAYS``, a condition stays as it is: every condition
    built here is already free of impossible alternatives."""
    if a == ALWAYS:
        return b
    if b == ALWAYS:
        return a
    return _without_longer_input_alternatives(frozenset(
        alternative
        for x in a
        for y in b
        if (alternative := x | y) is not None and _consistent(alternative)
    ))


def _without_longer_input_alternatives(condition: Condition) -> Condition:
    """``condition`` without each alternative that is another one plus input decisions only.

    "``a``, or ``a`` and input ``x``" holds exactly when ``a`` does, so the
    longer one says nothing. Without this, every input a graph offers
    would multiply alternatives the way a real decision does: two merges
    of eight inputs each, joined, give 36 alternatives where eight do.
    Only input decisions are dropped this way, so with every input holding
    the condition is what it was, and ``CompiledField.condition`` never
    changes."""
    if len(condition) < 2 or not any(atom.node_id == INPUT for alternative in condition for atom in alternative):
        return condition
    shortest_first = sorted(condition, key=len)
    kept: list[frozenset[Atom]] = []
    for alternative in shortest_first:
        if not any(
            shorter < alternative and all(atom.node_id == INPUT for atom in alternative - shorter) for shorter in kept
        ):
            kept.append(alternative)
    return frozenset(kept)


def _consistent(alternative: frozenset[Atom]) -> bool:
    """No decision named twice with two different outputs."""
    taken: dict[tuple[str, str], str] = {}
    for atom in alternative:
        if taken.setdefault((atom.node_id, atom.choice), atom.output) != atom.output:
            return False
    return True
