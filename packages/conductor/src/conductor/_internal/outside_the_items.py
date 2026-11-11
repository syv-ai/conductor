"""What is outside the items decides a gathering input only when it is certain.

A gathering input (a ``Series``, or a ``Table`` fed records) receives the items
of one Series. Before any item is asked, it asks what was settled outside the
items one thing: could any item bring a value here? The answers are an order:

    skipped  <  blank  <  could bring a value

A cell made outside the items is read as it stands. A node made per item is
the worst of its inputs, where an input that takes a blank receives it; an
input is the best of its alternative edges; an output of an exclusive group of
a node that isn't called is skipped, as no branch is chosen. A gathering input
is the worst of its Series and the best of its edges, and an item of the
gathered Series is what the Series itself is: a Series that was never made
has no item, so a node per item that takes no blank isn't called for it.

When no item could bring a value, the gathering node is treated as any node
reading from outside the items, with two items as with none: skipped for a
branch not taken, not called for a blank. When one could, each item answers
for itself.

No answer is kept from load. The tree of a graph keeps which nodes to ask: for
each gathering input, the nodes made per item that it rests on
(``FunctionAtPath.gathered_through``), whose reads outside the items are among
what the gathering node is made from, so the unit that settles the answer
makes the node ready. In a run, ``could_bring`` asks those nodes, in order,
and no other, of the units the run holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

from conductor._internal.fields import SKIPPED
from conductor._internal.graph import (
    Cell,
    CellRead,
    Fixed,
    FunctionAtPath,
    Gather,
    GraphAtPath,
    Item,
    NodeAtPath,
    NodePath,
    Open,
    SeriesOfOne,
)

#: An item could bring a value, so each item decides. The other answers are ``SKIPPED`` (none could, for a branch
#: not taken) and ``None`` (none could, for a blank); any other says what is outside isn't settled yet, and is
#: handed back as it was given.
COULD: Any = object()

class Outside(Protocol):
    """Reads what was made outside the items, as it stands, for the instance a gathering input gathers in. Each
    answers ``COULD`` for a value."""

    def cell(self, inside: NodePath, depth: int, cell: Cell) -> Any:
        """What ``cell`` holds: the unit ``depth`` places of the instance name, in the graph inside ``inside``."""

    def gathered(self, inside: NodePath, plan: Gather, cells: tuple[CellRead, ...], through: tuple[NodeAtPath, ...]) -> Any:
        """What the gathering input of ``plan``, itself made outside the items, gathers from ``cells``."""


def could_bring(plan: Gather, cells: tuple[CellRead, ...], through: tuple[NodeAtPath, ...], inside: NodePath, node_at: Mapping[NodePath, NodeAtPath], outside: Outside) -> Any:
    """Whether any item could bring the gathering input of ``plan`` a value, going by what is outside the items.

    Args:
        plan: The input's plan.
        cells: The cells the input reads where its node stands: the Series, then what each edge brings.
        through: The nodes the input gathers through, as the tree keeps them.
        inside: The path of the embedded graph's node the plan was made in the graph of; ``()`` for the graph
            a run is of.
        node_at: Each node of the tree, by its path.
        outside: Reads what was made outside the items.

    Returns:
        ``COULD``; ``SKIPPED`` when none could for a branch not taken; ``None`` when none could for a blank;
        what ``outside`` gave while something there isn't settled.
    """
    series, _ = cells[0]
    asking = _Asking(node_at[(*inside, series.node_id)], series, {}, node_at, outside)
    for inner in through:
        asking.asked[inner] = _could_come_to(inner, asking) if type(inner) is FunctionAtPath else COULD
    return _could_gather(inside, plan, cells, asking)


class _Asking(NamedTuple):
    """One asking of what is outside the items of a Series.

    Attributes:
        made_by: The node that makes the Series whose items nobody knows.
        series: The Series, where it was made.
        asked: What each node made per item could come to, as far as asked; a node not in it is made outside
            the items.
        node_at: Each node of the tree, by its path.
        outside: Reads what was made outside the items.
    """

    made_by: NodeAtPath
    series: Cell
    asked: dict[NodeAtPath, Any]
    node_at: Mapping[NodePath, NodeAtPath]
    outside: Outside


def _could_gather(inside: NodePath, plan: Gather, cells: tuple[CellRead, ...], asking: _Asking) -> Any:
    """What the gathering input of ``plan`` could come to for an item nobody knows: the worst of its Series and
    the best of its edges, as it needs the Series and one of the edges. An input whose items take a blank
    receives the blank an edge brings."""
    (series, gathered_in), *edges = cells
    brought = _best([_could_hold(inside, depth, source, asking) for source, depth in edges])
    if brought is None and plan.spec.keeps_blanks:
        brought = COULD
    return _worst([_could_hold(inside, gathered_in, series, asking), brought])


def _could_hold(inside: NodePath, depth: int, cell: Cell, asking: _Asking) -> Any:
    """What ``cell`` could hold for an item nobody knows: as it stands when it is made outside the items, and
    what its node could come to when it is made per item.

    Args:
        inside: The path of the embedded graph's node the cell's node is in the graph of.
        depth: How many places of an instance say which unit of the cell's node holds it.
        cell: The cell.
        asking: The asking.
    """
    made_by = asking.node_at[(*inside, cell.node_id)]
    if made_by not in asking.asked:
        if type(cell) is Item and made_by is asking.made_by and cell.output == asking.series.output:
            cell = asking.series  # an item of the Series itself is what the Series is: one never made has none
        return asking.outside.cell(inside, depth, cell)
    if type(made_by) is GraphAtPath:  # an output of an embedded graph: the cell inside that holds it
        offered, maps, _, series = made_by.offers[cell.output]
        if series is not None:  # made per item inside: a Series, there whenever the Series its items are of is
            return _could_hold(made_by.path, len(made_by.maps), series, asking)
        return _could_hold(made_by.path, len(maps), offered, asking)
    answer = asking.asked[made_by]
    if answer is None and made_by.placed.outputs[cell.output].param.exclusive_group is not None:
        return SKIPPED  # its node isn't called, so no branch is chosen
    return answer


def _could_come_to(node: FunctionAtPath, asking: _Asking) -> Any:
    """What a node made per item could come to for an item nobody knows, by the rules a unit comes to anything:
    the worst of its inputs, an input that takes a blank receiving it. An input of it that gathers is asked the
    same of the same items when it is made per item too; one made outside them is read as it stands.

    Args:
        node: The node, one the gathering input gathers through.
        asking: The asking, holding what each node before this one could come to.

    Returns:
        ``COULD`` when it could be called; ``SKIPPED`` when an input comes only from branches not taken;
        ``None`` when an input that takes no blank gets one; else what isn't settled outside.
    """
    answers = []
    for name, (plan, giver, cells) in node.inputs.items():
        kind = type(plan)
        if kind is Fixed or kind is Open:
            continue
        inside = node.inside if giver is None else giver.inside
        if kind is not Gather:
            answer = _best([_could_hold(inside, depth, source, asking) for source, depth in cells])
        elif asking.node_at[(*inside, cells[0][0].node_id)] in asking.asked:
            answer = _could_gather(inside, plan, cells, asking)  # type: ignore[arg-type]
        else:
            answer = asking.outside.gathered(inside, plan, cells, node.gathered_through[name])  # type: ignore[arg-type]
        takes_a_blank = plan.spec.whole.optional or (kind is SeriesOfOne and plan.spec.keeps_blanks)
        answers.append(COULD if answer is None and takes_a_blank else answer)
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
