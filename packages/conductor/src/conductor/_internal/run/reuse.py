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
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

from pydantic_core import PydanticSerializationError

from conductor._internal.base import ItemIndexes, NodePath, json_hash
from conductor._internal.table import hash_fallback

from .blueprint import BlueprintFunctionNode, RunBlueprint
from .outcome import Done, NoneInput, NotReady, Outcome, Skipped, Unanswered

if TYPE_CHECKING:
    from .type import Run


@dataclass(slots=True)
class ReusableOutcomes:
    """The outcomes of a previous run that a run may reuse instead of calling a node, by node path.

    A run made from another gets them with ``from_run``. As it runs, the scheduler takes each outcome it reuses with
    ``take``. Once a node is complete, its outcomes are dropped with ``drop``.

    A node that never asks reuses an outcome two ways. The call with the same item indexes reuses its own outcome if
    it gets the same arguments, and any other call with those arguments reuses the first outcome with that
    fingerprint. A node that may ask reuses each outcome once, by fingerprint alone, in item order. The two never
    mix, because whether a node may ask is part of its code fingerprint. So an outcome is only reused by a node that
    asks the way the one that made it did.

    After the first run in the module docstring, ``reused_freely[("shout",)]`` is
    ``{fp_a: {(0,): Done({"result": "A!"})}, fp_b: {(1,): ...}, fp_c: {(2,): ...}}``. Here ``fp_a`` is the call
    fingerprint of the call for ``"a"``.

    Attributes:
        reused_freely: For a node that never asks: by call fingerprint, the outcomes of the calls with that
            fingerprint, each by its call's item indexes. The first is reused by any call with the fingerprint, and
            each other one only by the call with its item indexes. ``from_run`` says which are kept.
        reused_in_item_order: For a node that may ask: by call fingerprint, the outcomes, ``Done`` or
            ``Unanswered``, of the calls with that fingerprint, in item order. Each is reused once, by the calls
            with that fingerprint in item order, and then taken out.
        code_changed: The nodes whose outcomes in this run were made by other code or another signature, as a
            saved run showed, with the code fingerprint they were made with. Their outcomes are never offered for
            reuse, and a save writes them under that old fingerprint.
    """

    reused_freely: dict[NodePath, dict[str, dict[ItemIndexes, Done]]] = field(
        default_factory=dict[NodePath, dict[str, dict[ItemIndexes, Done]]]
    )
    reused_in_item_order: dict[NodePath, dict[str, list[Done | Unanswered]]] = field(
        default_factory=dict[NodePath, dict[str, list["Done | Unanswered"]]]
    )
    code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])

    @classmethod
    def from_run(cls, run: Run[Any], into: RunBlueprint) -> ReusableOutcomes:
        """Returns what a run made from ``run`` may reuse.

        That is each node's finished and waiting calls in item order, then the outcomes ``run`` itself could have
        reused and didn't. A node that may ask keeps every one of them, since each is reused once. A node that never
        asks keeps an outcome if it is the first with its fingerprint, which any call with that fingerprint reuses,
        or the first with its item indexes, which the call with those item indexes reuses. No call would reuse any
        other: a call with the same item indexes finds the first outcome with them, and any other call the first
        with its fingerprint.

        Say ``shout`` ran on ``"a\\na"``, then on ``"b\\na"``, which reused the first run's outcome at ``(1,)``, and
        a third run is made from the second. The second run's own outcomes come first: ``(fp_b, (0,))`` and
        ``(fp_a, (1,))``. Then come the two it could reuse: ``(fp_a, (0,))``, which is dropped, since ``fp_a`` and
        ``(0,)`` both have an outcome already, and ``(fp_a, (1,))``, dropped too.

        ``into`` is the blueprint of the new run's graph, which may be ``run``'s graph or an edited one. A node is only
        kept if ``into`` has it unchanged (the same ``code_fingerprint``) and its ``Policy`` says
        ``reuse_when_inputs_match``. The other nodes' outcomes could never be reused there, and keeping them would only
        make the save bigger.
        """
        reusable, previous = cls(), run._reusable
        for path, old_node in run._blueprint.function_nodes.items():
            new_node = into.function_nodes.get(path)
            if (
                new_node is None
                or not new_node.policy.reuse_when_inputs_match
                or new_node.code_fingerprint != old_node.code_fingerprint
            ):
                continue
            called = () if path in previous.code_changed else _called(run, old_node)
            if new_node.compiled_node.definition.asks:
                in_item_order: dict[str, list[Done | Unanswered]] = {}
                for fingerprint, _, outcome in called:
                    in_item_order.setdefault(fingerprint, []).append(outcome)
                for fingerprint, left in previous.reused_in_item_order.get(path, {}).items():
                    in_item_order.setdefault(fingerprint, []).extend(left)
                if in_item_order:
                    reusable.reused_in_item_order[path] = in_item_order
                continue
            reused_freely: dict[str, dict[ItemIndexes, Done]] = {}
            with_an_outcome: set[ItemIndexes] = set()
            previous_freely = (
                (fingerprint, item_indexes, done)
                for fingerprint, of_fingerprint in previous.reused_freely.get(path, {}).items()
                for item_indexes, done in of_fingerprint.items()
            )
            # A node that never asks returns or fails, so each of its called outcomes is a Done.
            for fingerprint, item_indexes, done in cast(
                "Iterable[tuple[str, ItemIndexes, Done]]", itertools.chain(called, previous_freely)
            ):
                of_fingerprint = reused_freely.get(fingerprint)
                if of_fingerprint is None:
                    reused_freely[fingerprint] = {item_indexes: done}
                elif item_indexes not in with_an_outcome:
                    of_fingerprint[item_indexes] = done
                with_an_outcome.add(item_indexes)
            if reused_freely:
                reusable.reused_freely[path] = reused_freely
        return reusable

    def take(
        self,
        node: BlueprintFunctionNode,
        item_indexes: ItemIndexes,
        arguments: dict[str, Any],
        outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
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
            arguments: The arguments of the call.
            outcomes: The run's outcomes, by node path and then by item indexes.
            in_turn: Returns whether every earlier call of the node has been tried.
        """
        path = node.node_path
        freely, in_item_order = self.reused_freely.get(path), self.reused_in_item_order.get(path)
        if freely is None and in_item_order is None:
            return None
        fingerprint = call_fingerprint(node, item_indexes, arguments, outcomes)
        if fingerprint is None:
            return None
        if freely is not None:
            of_fingerprint = freely.get(fingerprint)
            if of_fingerprint is None:
                return None
            return of_fingerprint.get(item_indexes, next(iter(of_fingerprint.values())))
        left = self.reused_in_item_order[path].get(fingerprint)
        if left is None:
            return None
        # An answer or a return goes to the first call in item order with the fingerprint.
        if any(isinstance(outcome, Done) for outcome in left) and not in_turn():
            return NotReady.NOT_ITS_TURN
        outcome = left.pop(0)
        if not left:  # drop empty entries, so a saved run writes nothing for them
            del self.reused_in_item_order[path][fingerprint]
            if not self.reused_in_item_order[path]:
                del self.reused_in_item_order[path]
        return outcome

    def drop(self, paths: Collection[NodePath]) -> None:
        """Drops every outcome the nodes at ``paths`` may reuse.

        A node that is complete has no call left to reuse them, and a forgotten node must be called again.
        """
        for by_node in (self.reused_freely, self.reused_in_item_order):
            for path in paths:
                by_node.pop(path, None)

    def copy(self) -> ReusableOutcomes:
        """Returns a copy to take outcomes from and drop nodes from, leaving this one as it is. The outcomes are shared."""
        return ReusableOutcomes(
            dict(self.reused_freely),
            {
                path: {fingerprint: list(left) for fingerprint, left in of_node.items()}
                for path, of_node in self.reused_in_item_order.items()
            },
            self.code_changed,
        )


def _called(
    run: Run[Any], node: BlueprintFunctionNode
) -> Iterator[tuple[str, ItemIndexes, Done | Unanswered]]:
    """Yields the outcome of each call of ``node`` in ``run`` that called its function, in item order, with the call's
    fingerprint and item indexes. A call whose arguments can't be written as JSON has no fingerprint, and is left out."""
    for item_indexes, outcome in run._outcomes[node.node_path].items():
        match outcome:
            # The function wasn't called for the item, so there is nothing to reuse.
            case Skipped() | NoneInput():
                continue
            # The function was called, so the call's arguments all exist.
            case Done() | Unanswered():
                arguments = cast("dict[str, Any]", run._reader.arguments_for(node, item_indexes))
                fingerprint = call_fingerprint(node, item_indexes, arguments, run._outcomes)
                if fingerprint is not None:
                    yield fingerprint, item_indexes, outcome
            case _:
                assert_never(outcome)


def call_fingerprint(
    node: BlueprintFunctionNode,
    item_indexes: ItemIndexes,
    arguments: dict[str, Any],
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
) -> str | None:
    """Returns the fingerprint of one call: a short hash of its node's code fingerprint and its arguments.

    Two calls with one fingerprint do the same, so one may reuse the other's outcome. For a node that may ask a
    person, the call's item in every loop it is inside counts too, since a person's answer belongs to its item.
    Take ``pages -> lines -> approve``, where two pages may have the same line. Each call of ``approve`` is told
    apart by its page, so each page keeps its own answers when the pages are reordered. An item counts even when the
    node doesn't take it: with ``pages -> size -> approve``, ``approve`` gets a page's length, and two pages of one
    length keep their own answers.

    A waiting call's id ends with the first eight hex digits of its call fingerprint (``question_fingerprint``).
    The hash sorts the keys of every dict, so a dict read back from a database with its keys in another order
    hashes the same as before it was saved.

    Args:
        node: The node.
        item_indexes: The item indexes of the call.
        arguments: The call's arguments, by input name.
        outcomes: The run's outcomes, by node path and then by item indexes.

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
                outcomes[series][item_indexes[:depth]].outputs[output][item_indexes[depth]]  # type: ignore[union-attr]
                for depth, (series, output) in enumerate(node.loops)
            ]
        )
    try:
        return json_hash(hashed, 12, fallback=hash_fallback)
    except (TypeError, ValueError, PydanticSerializationError):
        return None
