"""What is outside the items decides a gathering input only when it is certain.

A gathering input (a ``Series``, or a ``Table`` fed records) receives the items
of one Series. Before any item is asked, it asks what was settled outside the
items one thing: could any item bring a value here? The answers are an order:

    skipped  <  blank  <  could bring a value

A cell made outside the items is read as it stands. A node made per item is
the worst of its inputs, where an input that takes a blank receives it; an
input is the best of its alternative edges; an output of an exclusive group of
a node that isn't called is skipped, as no branch is chosen. A gathering input
is the worst of its Series and the best of its edges.

When no item could bring a value, the gathering node is treated as any node
reading from outside the items, with two items as with none: skipped for a
branch not taken, not called for a blank. When one could, each item answers
for itself.

The rule is asked twice. When the tree of a graph is built, ``gathered_through``
finds the nodes made per item that a gathering input rests on, and adds what
they read outside the items to what the gathering node reads, so the unit that
settles the answer makes the node ready. In a run, ``could_bring`` asks those
nodes, in order, and no other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from conductor._internal.fields import SKIPPED
from conductor._internal.graph import (
    Cell,
    Fixed,
    FunctionAtPath,
    Gather,
    GraphAtPath,
    Item,
    NodeAtPath,
    NodePath,
    Open,
)

#: An item could bring a value, so each item decides. The other answers are ``SKIPPED`` (none could, for a branch
#: not taken) and ``None`` (none could, for a blank); any other says what is outside isn't settled yet, and is
#: handed back as it was given.
COULD: Any = object()

#: Reads what was made outside the items, as it stands: ``stands(inside, base, read, through)`` for a cell, or for
#: the plan of a gathering input and the nodes it gathers through. It answers ``COULD`` for a value.
Stands = Callable[[NodePath, int, Any, tuple[NodeAtPath, ...]], Any]


def gathered_through(node: FunctionAtPath, series: tuple[NodeAtPath, str], sources: list[NodeAtPath]) -> tuple[NodeAtPath, ...]:
    """Finds the nodes made per item of ``series`` that a gathering input of ``node`` rests on, by each one's own
    ``reads``, and adds what they read outside the items to ``node.reads``.

    Args:
        node: The node that gathers.
        series: The Series the input gathers: the node that makes it, and its output.
        sources: The nodes whose outputs the input's edges bring.

    Returns:
        The nodes, in the tree's order: each after the ones it reads. None when all they read outside the
        items is the Series, from a node whose one output it is: then only the Series itself has a say, and
        a run reads that with no question asked.
    """
    made_by = series[0]
    through: set[NodeAtPath] = set()
    only_the_series = len(made_by.placed.outputs) == 1
    stack = sources
    while stack:
        inner = stack.pop()
        if series in inner.maps and inner not in through:
            through.add(inner)
            for read, places in inner.reads.items():
                if series in read.maps:
                    stack.append(read)
                else:
                    node.reads[read] = min(places, node.reads.get(read, places))
                    only_the_series = only_the_series and read is made_by
    return () if only_the_series else tuple(sorted(through, key=lambda inner: inner.position))


def could_bring(plan: Gather, through: tuple[NodeAtPath, ...], inside: NodePath, base: int, node_at: Mapping[NodePath, NodeAtPath], stands: Stands) -> Any:
    """Whether any item could bring the gathering input of ``plan`` a value, going by what is outside the items.

    Args:
        plan: The input's plan.
        through: The nodes the input gathers through (``gathered_through``).
        inside: The path of the embedded graph's node the plan was made in the graph of; ``()`` for the graph
            a run is of.
        base: How many places of an instance are of the items of that embedded graph's node.
        node_at: Each node of the tree, by its path.
        stands: Reads what was made outside the items.

    Returns:
        ``COULD``; ``SKIPPED`` when none could for a branch not taken; ``None`` when none could for a blank;
        what ``stands`` gave while something outside isn't settled.
    """
    asked: dict[NodeAtPath, Any] = {}
    for inner in through:
        asked[inner] = _could_come_to(inner, plan, asked, node_at, stands) if type(inner) is FunctionAtPath else COULD
    return _could_gather(inside, base, plan, plan, asked, node_at, stands)


def _could_gather(inside: NodePath, base: int, plan: Gather, gathered: Gather, asked: dict[NodeAtPath, Any], node_at: Mapping[NodePath, NodeAtPath], stands: Stands) -> Any:
    """What the gathering input of ``plan`` could come to for the items ``gathered`` gathers: the worst of its
    Series and the best of its edges, as it needs the Series and one of the edges. An input whose items take a
    blank receives the blank an edge brings."""
    brought = _best([_could_hold(inside, base, source, gathered, asked, node_at, stands) for source in plan.sources])
    if brought is None and plan.spec.keeps_blanks:
        brought = COULD
    return _worst([_could_hold(inside, base, plan.series, gathered, asked, node_at, stands), brought])


def _could_hold(inside: NodePath, base: int, cell: Cell, gathered: Gather, asked: dict[NodeAtPath, Any], node_at: Mapping[NodePath, NodeAtPath], stands: Stands) -> Any:
    """What ``cell`` could hold for an item of the Series ``gathered`` gathers: as it stands when it is made
    outside the items, and what its node could come to when it is made per item.

    Args:
        inside: The path of the embedded graph's node the cell's node is in the graph of.
        base: How many places of an instance are of the items of that embedded graph's node.
        cell: The cell.
        gathered: The plan whose items nobody knows.
        asked: What each node made per item could come to; a node not in it is made outside the items.
        node_at: Each node of the tree, by its path.
        stands: Reads what was made outside the items.
    """
    made_by = node_at[(*inside, cell.node_id)]
    if made_by not in asked:
        series = gathered.series
        if type(cell) is Item and cell.node_id == series.node_id and cell.output == series.output:
            return COULD  # an item of the Series itself
        return stands(inside, base, cell, ())
    if type(made_by) is GraphAtPath:  # an output of an embedded graph: the cell inside that holds it
        for address, offered, maps, _ in made_by.offers:
            if address == cell.output:
                if len(maps) == len(made_by.maps):
                    return _could_hold(made_by.path, len(maps), offered, gathered, asked, node_at, stands)
                return COULD  # made per item inside: a Series, whatever its items come to
    answer = asked[made_by]
    if answer is None and made_by.placed.outputs[cell.output].param.exclusive_group is not None:
        return SKIPPED  # its node isn't called, so no branch is chosen
    return answer


def _could_come_to(node: FunctionAtPath, gathered: Gather, asked: dict[NodeAtPath, Any], node_at: Mapping[NodePath, NodeAtPath], stands: Stands) -> Any:
    """What a node made per item could come to for an item nobody knows, by the rules a unit comes to anything:
    the worst of its inputs, an input that takes a blank receiving it. An input of it that gathers is asked the
    same of the same items when it is made per item too; one made outside them is read as it stands.

    Args:
        node: The node, one the input of ``gathered`` gathers through.
        gathered: The plan whose items nobody knows.
        asked: What each node before it could come to.
        node_at: Each node of the tree, by its path.
        stands: Reads what was made outside the items.

    Returns:
        ``COULD`` when it could be called; ``SKIPPED`` when an input comes only from branches not taken;
        ``None`` when an input that takes no blank gets one; else what isn't settled outside.
    """
    answers = []
    for name, (plan, giver) in node.inputs.items():
        kind = type(plan)
        if kind is Fixed or kind is Open:
            continue
        inside, base = (node.inside, node.base) if giver is None else (giver.inside, giver.base)
        if kind is not Gather:
            answer = _best([_could_hold(inside, base, source, gathered, asked, node_at, stands) for source in plan.sources])  # type: ignore[union-attr]
        elif node_at[(*inside, plan.series.node_id)] in asked:  # type: ignore[union-attr]
            answer = _could_gather(inside, base, plan, gathered, asked, node_at, stands)  # type: ignore[arg-type]
        else:
            answer = stands(inside, base, plan, node.gathered_through[name])
        answers.append(COULD if answer is None and plan.spec.whole.optional else answer)
    return _worst(answers)


def _worst(answers: list[Any]) -> Any:
    """What comes of things that are all needed: a branch not taken whatever else is there, then what isn't
    settled, then a blank."""
    worst = COULD
    for answer in answers:
        if answer is SKIPPED:
            return SKIPPED
        if answer is not COULD and (worst is COULD or worst is None):
            worst = answer
    return worst


def _best(answers: list[Any]) -> Any:
    """What comes of edges that are alternatives into one input: what isn't settled first; then one that could
    bring a value is enough; then a blank; skipped when every one is."""
    best = SKIPPED
    for answer in answers:
        if answer is not SKIPPED and answer is not None and answer is not COULD:
            return answer
        if answer is COULD or best is SKIPPED:
            best = answer
    return best
