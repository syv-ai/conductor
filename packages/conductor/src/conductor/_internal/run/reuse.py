"""Reusing outcomes from a previous run, so a call that gets the same arguments as before isn't made again.

Take the README's graph, ``lines -> shout -> join``::

    run = graph.run({"text": "a\\nb\\nc"})    # shout is called for "a", "b" and "c"
    run = run.again({"text": "a\\nb\\nd"})    # shout is called for "d" only

In the second run, the calls of ``shout`` for ``"a"`` and ``"b"`` get the same arguments as before. So their
``Done`` outcomes are reused, and the function isn't called for them.

Two calls are matched by their call fingerprint (``call_fingerprint``): a short hash of the arguments and of the
node's code fingerprint. The code fingerprint covers the node's ``@node`` id and version, its code, the types of its
inputs and outputs, its loops, and whether it may ask. So when a function changes, it is called again.

A node that may ask a person (``@node(asks=True)``) is reused another way. Two of its calls with the same
arguments may have been answered differently. So each of its old outcomes is reused once, by the calls with
those arguments, in item order. A call takes the answer the person gave then, and a question still waiting goes
on waiting. The same holds for what the function returned without asking. A call beyond the outcomes there were
is made again.

``python -m conductor.about running-again`` has the full rules.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_core import PydanticSerializationError

from conductor._internal.base import ItemIndexes, NodePath, json_hash
from conductor._internal.table import hash_fallback

from .blueprint import BlueprintFunctionNode, RunBlueprint
from .outcome import Done, NotReady, Outcome, Unanswered
from .outcomes import RunOutcomes


@dataclass(slots=True)
class ReusableOutcomes:
    """The outcomes of a previous run that a run may reuse instead of calling a node, by node path.

    A run made from another gets them with ``from_previous``, and a saved run is read back with ``add``, one node at a
    time. ``entries`` gives a node's outcomes back, as a saved run writes them. As a run runs, the scheduler takes each
    outcome it reuses with ``take``. Once a node is complete, its outcomes are dropped with ``drop``.

    A node that never asks reuses an outcome two ways. The call with the same item indexes reuses its own outcome if
    it gets the same arguments, and any other call with those arguments reuses the first outcome with that
    fingerprint. A node that may ask reuses each outcome once, by fingerprint alone, in item order. The two never
    mix, because whether a node may ask is part of its code fingerprint. So an outcome is only reused by a node that
    asks the way the one that made it did.

    After the first run in the module docstring, the outcomes ``shout`` may reuse are
    ``{fp_a: {(0,): Done({"result": "A!"})}, fp_b: {(1,): ...}, fp_c: {(2,): ...}}``. Here ``fp_a`` is the call
    fingerprint of the call for ``"a"``.
    """

    # For a node that never asks: by call fingerprint, the outcomes of the calls with that fingerprint, each by its
    # call's item indexes. The first is reused by any call with the fingerprint, and each other one only by the call
    # with its item indexes.
    _reused_freely: dict[NodePath, dict[str, dict[ItemIndexes, Done]]] = field(
        default_factory=dict[NodePath, dict[str, dict[ItemIndexes, Done]]]
    )
    # For a node that may ask: by call fingerprint, the outcomes of the calls with that fingerprint, in item order,
    # each with its call's item indexes. Each is reused once, by the calls with that fingerprint in item order, and
    # then taken out.
    _reused_in_item_order: dict[
        NodePath, dict[str, list[tuple[ItemIndexes, Done | Unanswered]]]
    ] = field(
        default_factory=dict[NodePath, dict[str, list[tuple[ItemIndexes, "Done | Unanswered"]]]]
    )

    @classmethod
    def from_previous(
        cls, outcomes: RunOutcomes, previous: ReusableOutcomes, into: RunBlueprint
    ) -> ReusableOutcomes:
        """Returns what a run made from a previous run may reuse, given that run's outcomes and what it could reuse.

        For each node, that is its finished and waiting calls in item order, then its outcomes in ``previous``, which
        the previous run itself could have reused and didn't. ``add`` keeps those a call may reuse.

        ``into`` is the blueprint of the new run's graph, which may be the previous run's graph or an edited one. A
        node is only kept if ``into`` has it unchanged (the same ``code_fingerprint``). The other nodes' outcomes
        could never be reused there, and keeping them would only make the save bigger.
        """
        reusable = cls()
        for path, old_node in outcomes.blueprint.function_nodes.items():
            new_node = into.function_nodes.get(path)
            if new_node is not None and new_node.code_fingerprint == old_node.code_fingerprint:
                reusable.add(
                    new_node, itertools.chain(outcomes.of(path).items(), previous.entries(path))
                )
        return reusable

    def add(
        self, node: BlueprintFunctionNode, outcomes: Iterable[tuple[ItemIndexes, Outcome | None]]
    ) -> None:
        """Adds the outcomes of one node's calls that a later call of the node may reuse.

        Each outcome comes with the item indexes of the call that made it, in the order they are to be reused. Only
        a ``Done`` or ``Unanswered`` with a fingerprint is kept, since only a call that was made has one, and an
        ``Unanswered`` only for a node that may ask. Nothing is kept for a node whose ``Policy`` doesn't say
        ``reuse_when_inputs_match``.

        A node that may ask keeps every one of them, since each is reused once. A node that never asks keeps an
        outcome if it is the first with its fingerprint, which any call with that fingerprint reuses, or the first
        with its item indexes, which the call with those item indexes reuses. No call would reuse any other.

        Say ``shout`` ran on ``"a\\na"``, then on ``"b\\na"``, which reused the first run's outcome at ``(1,)``, and
        a third run is made from the second. The second run's own outcomes come first: ``(fp_b, (0,))`` and
        ``(fp_a, (1,))``. Then come the two it could reuse: ``(fp_a, (0,))``, which is dropped, since ``fp_a`` and
        ``(0,)`` both have an outcome already, and ``(fp_a, (1,))``, dropped too.

        Args:
            node: The node, in the blueprint of the run that may reuse the outcomes. It has none here yet.
            outcomes: The outcomes, each with its call's item indexes. ``None`` is a call with no outcome.
        """
        if not node.policy.reuse_when_inputs_match:
            return
        path = node.node_path
        if node.compiled_node.definition.asks:
            in_item_order: dict[str, list[tuple[ItemIndexes, Done | Unanswered]]] = {}
            for item_indexes, outcome in outcomes:
                if isinstance(outcome, (Done, Unanswered)) and outcome.fingerprint is not None:
                    in_item_order.setdefault(outcome.fingerprint, []).append(
                        (item_indexes, outcome)
                    )
            if in_item_order:
                self._reused_in_item_order[path] = in_item_order
            return
        reused_freely: dict[str, dict[ItemIndexes, Done]] = {}
        with_an_outcome: set[ItemIndexes] = set()
        for item_indexes, outcome in outcomes:
            if not isinstance(outcome, Done) or outcome.fingerprint is None:
                continue
            of_fingerprint = reused_freely.setdefault(outcome.fingerprint, {})
            if not of_fingerprint or item_indexes not in with_an_outcome:
                of_fingerprint[item_indexes] = outcome
            with_an_outcome.add(item_indexes)
        if reused_freely:
            self._reused_freely[path] = reused_freely

    def entries(self, path: NodePath) -> list[tuple[ItemIndexes, Done | Unanswered]]:
        """Returns the outcomes the node at ``path`` may reuse, each with its call's item indexes.

        They are grouped by fingerprint. The first with a fingerprint is the one any call with it reuses, and the
        outcomes of a node that may ask are in the order its calls take them. So ``add`` given them back keeps the
        same outcomes. After the first run in the module docstring, ``entries(("shout",))`` is
        ``[((0,), Done({"result": "A!"})), ((1,), ...), ((2,), ...)]``.
        """
        if (freely := self._reused_freely.get(path)) is not None:
            return [
                (item_indexes, done)
                for of_fingerprint in freely.values()
                for item_indexes, done in of_fingerprint.items()
            ]
        return [
            entry for left in self._reused_in_item_order.get(path, {}).values() for entry in left
        ]

    def take(
        self,
        node: BlueprintFunctionNode,
        item_indexes: ItemIndexes,
        fingerprint: str | None,
        in_turn: Callable[[], bool],
    ) -> Done | Unanswered | Literal[NotReady.NOT_ITS_TURN] | None:
        """Returns the outcome a call reuses instead of calling the function, or ``None`` if there is none.

        In the second run in the module docstring, the call ``(1,)`` of ``shout`` gets ``{"line": "b", "mark": "!"}``
        and reuses ``Done({"result": "B!"})``. The call ``(2,)`` gets ``"d"``, which no call got before, so the result
        is ``None`` and ``shout`` is called.

        For a node that never asks, it is the outcome with the call's fingerprint and item indexes, or else the
        first with its fingerprint. A node that may ask has each outcome reused once, in call order, and taken out
        once reused. If an outcome with outputs is left and an earlier call of the node hasn't been tried yet, the
        result is ``NOT_ITS_TURN``, and the call is tried again later. Waiting questions are all alike, so when only
        those are left, a call takes one at once without waiting for its turn.

        Args:
            node: The node.
            item_indexes: The item indexes of the call.
            fingerprint: The call's fingerprint (``call_fingerprint``). A call with none reuses nothing.
            in_turn: Returns whether every earlier call of the node has been tried.
        """
        path = node.node_path
        freely, in_item_order = self._reused_freely.get(path), self._reused_in_item_order.get(path)
        if (freely is None and in_item_order is None) or fingerprint is None:
            return None
        if freely is not None:
            of_fingerprint = freely.get(fingerprint)
            if of_fingerprint is None:
                return None
            return of_fingerprint.get(item_indexes, next(iter(of_fingerprint.values())))
        left = self._reused_in_item_order[path].get(fingerprint)
        if left is None:
            return None
        # An answer or a return goes to the first call in item order with the fingerprint.
        if any(isinstance(outcome, Done) for _, outcome in left) and not in_turn():
            return NotReady.NOT_ITS_TURN
        _, outcome = left.pop(0)
        if not left:  # drop empty entries, so a saved run writes nothing for them
            del self._reused_in_item_order[path][fingerprint]
            if not self._reused_in_item_order[path]:
                del self._reused_in_item_order[path]
        return outcome

    def drop(self, paths: Collection[NodePath]) -> None:
        """Drops every outcome the nodes at ``paths`` may reuse.

        A node that is complete has no call left to reuse them, and a forgotten node must be called again.
        """
        for by_node in (self._reused_freely, self._reused_in_item_order):
            for path in paths:
                by_node.pop(path, None)

    def copy(self) -> ReusableOutcomes:
        """Returns a copy to take outcomes from and drop nodes from, leaving this one as it is. The outcomes are shared."""
        return ReusableOutcomes(
            dict(self._reused_freely),
            {
                path: {fingerprint: list(left) for fingerprint, left in of_node.items()}
                for path, of_node in self._reused_in_item_order.items()
            },
        )


def call_fingerprint(
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    arguments: dict[str, Any],
    outcomes: RunOutcomes,
) -> str | None:
    """Returns the fingerprint of one call: a short hash of its node's code fingerprint and its arguments.

    Two calls with one fingerprint do the same, so one may reuse the other's outcome. For a node that may ask a
    person, the call's item in every loop it is inside counts too, since a person's answer belongs to its item.
    Take ``pages -> lines -> approve``, where two pages may have the same line. Each call of ``approve`` is told
    apart by its page, so each page keeps its own answers when the pages are reordered. An item counts even when the
    node doesn't take it: with ``pages -> size -> approve``, ``approve`` gets a page's length, and two pages of one
    length keep their own answers.

    The scheduler takes it once, when the call is made, and the call's ``Done`` or ``Unanswered`` keeps it. A
    waiting call's id is made from it (``question_fingerprint``). The hash sorts the keys of every dict, so a dict
    read back from a database with its keys in another order hashes the same as before it was saved.

    Args:
        node: The node.
        item_indexes: The item indexes of the call.
        arguments: The call's arguments, by input name.
        outcomes: The run's outcomes.

    Returns:
        The fingerprint, or ``None`` when an argument or an item can't be written as JSON. A call with no
        fingerprint is never reused.
    """
    hashed = [node.code_fingerprint, arguments]
    if node.compiled_node.definition.asks:
        # The node that returned each series has outputs here, since a call inside a loop is only made once its
        # series exists.
        hashed.append(
            [
                outcomes.of(series)[item_indexes[:depth]].outputs[output][item_indexes[depth]]  # type: ignore[union-attr]
                for depth, (series, output) in enumerate(node.loops)
            ]
        )
    try:
        return json_hash(hashed, 12, fallback=hash_fallback)
    except (TypeError, ValueError, PydanticSerializationError):
        return None
