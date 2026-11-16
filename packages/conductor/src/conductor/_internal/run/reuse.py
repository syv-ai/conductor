"""Reusing outcomes from a previous run, so a call that gets the same arguments as before isn't made again.

Take the README's graph, ``lines -> shout -> join``::

    run = graph.run(text="a\\nb\\nc")    # shout is called for "a", "b" and "c"
    run = run.again(text="a\\nb\\nd")    # shout is called for "d" only

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

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

from pydantic_core import PydanticSerializationError

from conductor._internal.base import ItemIndexes, NodePath, json_hash
from conductor._internal.types import untyped_to_hash

from .blueprint import BlueprintFunctionNode, RunBlueprint
from .outcome import Done, NoneInput, NotReady, Outcome, Skipped, Unanswered

if TYPE_CHECKING:
    from .type import Run


@dataclass(slots=True)
class ReusableOutcomes:
    """The outcomes of a previous run that a run may reuse instead of calling a node, by node path.

    A run made from another gets them with ``from_run``. As it runs, the scheduler takes each outcome it reuses with
    ``take``. Once a node is complete, its outcomes are dropped with ``drop``.

    After the first run in the module docstring, ``reused_freely_by_call[("shout",)]`` is
    ``{(0,): (fp_a, Done({"result": "A!"})), (1,): (fp_b, ...), (2,): (fp_c, ...)}``. Here ``fp_a`` is the
    call fingerprint of the call for ``"a"``.

    A node that never asks has its outcomes in ``reused_freely`` and ``reused_freely_by_call``. A node that may ask
    has them in ``reused_in_item_order``. The two never mix, because whether a node may ask is part of its code
    fingerprint. So an outcome is only reused by a node that asks the way the one that made it did.

    Attributes:
        reused_freely_by_call: For a node that never asks: each outcome, by the item indexes of its call, with that
            call's fingerprint. The call with the same item indexes reuses it if it gets the same arguments.
        reused_freely: For a node that never asks: each outcome, by its call's fingerprint. Any call with the same
            arguments reuses it. If several calls had the same arguments, the first in item order is kept.
        reused_in_item_order: For a node that may ask: each outcome, ``Done`` or ``Unanswered``, by its call's
            fingerprint, in call order. Each is reused once, by the calls with that fingerprint in item order, and
            then taken out.
        code_changed: The nodes whose outcomes in this run were made by other code or another signature, as a
            saved run showed, with the code fingerprint they were made with. Their outcomes are never offered for
            reuse, and a save writes them under that old fingerprint.
    """

    reused_freely_by_call: dict[NodePath, dict[ItemIndexes, tuple[str, Done]]] = field(
        default_factory=dict[NodePath, dict[ItemIndexes, tuple[str, Done]]]
    )
    reused_freely: dict[NodePath, dict[str, Done]] = field(
        default_factory=dict[NodePath, dict[str, Done]]
    )
    reused_in_item_order: dict[NodePath, dict[str, list[Done | Unanswered]]] = field(
        default_factory=dict[NodePath, dict[str, list["Done | Unanswered"]]]
    )
    code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])

    @classmethod
    def from_run(cls, run: Run[Any], into: RunBlueprint) -> ReusableOutcomes:
        """Returns what a run made from ``run`` may reuse.

        That is each node's finished and waiting calls in item order, then the outcomes ``run`` itself could have
        reused and didn't.

        ``into`` is the blueprint of the new run's graph, which may be ``run``'s graph or an edited one. A node is only
        kept if ``into`` has it unchanged (the same ``code_fingerprint``) and its ``Policy`` says
        ``reuse_when_inputs_match``. The other nodes' outcomes could never be reused there, and keeping them would only
        make the save bigger.
        """
        reusable, previous, reader = cls(), run._reusable, run._reader
        for path, old_node in run._blueprint.function_nodes.items():
            new_node = into.function_nodes.get(path)
            if (
                new_node is None
                or not new_node.policy.reuse_when_inputs_match
                or new_node.code_fingerprint != old_node.code_fingerprint
            ):
                continue
            if path not in previous.code_changed:
                for item_indexes, outcome in run._outcomes[path].items():
                    match outcome:
                        case (
                            Skipped() | NoneInput()
                        ):  # the function wasn't called for the item, so there is nothing to reuse
                            continue
                        case (
                            Done() | Unanswered()
                        ):  # the function was called, so the call's arguments all exist
                            arguments = cast(
                                "dict[str, Any]", reader.arguments_for(old_node, item_indexes)
                            )
                            fingerprint = call_fingerprint(
                                old_node, item_indexes, arguments, run._outcomes
                            )
                        case _:
                            assert_never(outcome)
                    if fingerprint is not None:
                        reusable.add(new_node, fingerprint, outcome, item_indexes=item_indexes)
            for fingerprint, done in previous.reused_freely.get(path, {}).items():
                reusable.add(new_node, fingerprint, done)
            for fingerprint, outcomes in previous.reused_in_item_order.get(path, {}).items():
                for outcome in outcomes:
                    reusable.add(new_node, fingerprint, outcome)
        return reusable

    def add(
        self,
        node: BlueprintFunctionNode,
        fingerprint: str,
        outcome: Done | Unanswered,
        *,
        item_indexes: ItemIndexes | None = None,
    ) -> None:
        """Adds one outcome ``node`` may reuse, after those already added.

        It goes into ``reused_in_item_order`` when the node may ask a person, and into ``reused_freely`` otherwise.

        Args:
            node: The node.
            fingerprint: The call fingerprint of the call the outcome came from.
            outcome: The outcome.
            item_indexes: For a node that never asks, the item indexes of the call, when they are known. The call with
                the same item indexes gets the outcome first.
        """
        path = node.node_path
        match outcome:
            case Done() if not node.compiled_node.definition.asks:
                self.reused_freely.setdefault(path, {}).setdefault(fingerprint, outcome)
                if item_indexes is not None:
                    self.reused_freely_by_call.setdefault(path, {})[item_indexes] = (
                        fingerprint,
                        outcome,
                    )
            case Done() | Unanswered():
                self.reused_in_item_order.setdefault(path, {}).setdefault(fingerprint, []).append(
                    outcome
                )

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

        For a node that never asks, it first looks for the outcome of the call with the same item indexes, then for
        any call's. A node that may ask has each outcome reused once, in call order, and taken out once reused. If an
        outcome with outputs is left and an earlier call of the node hasn't been tried yet, the result is
        ``NOT_ITS_TURN``, and the call is tried again later. Waiting questions are all alike, so when only those are
        left, a call takes one at once without waiting for its turn.

        Args:
            node: The node.
            item_indexes: The item indexes of the call.
            arguments: The arguments of the call.
            outcomes: The run's outcomes, by node path and then by item indexes.
            in_turn: Returns whether every earlier call of the node has been tried.
        """
        path = node.node_path
        freely, in_item_order = self.reused_freely.get(path), self.reused_in_item_order.get(path)
        if not freely and not in_item_order:
            return None
        fingerprint = call_fingerprint(node, item_indexes, arguments, outcomes)
        if fingerprint is None:
            return None
        if in_item_order:
            left = in_item_order.get(fingerprint)
            if left:
                for waiting_or_done in left:
                    match waiting_or_done:
                        case Done():  # an answer or a return, which goes to the first call in item order
                            if not in_turn():
                                return NotReady.NOT_ITS_TURN
                            break
                        case Unanswered():
                            pass
                        case _:
                            assert_never(waiting_or_done)
                outcome = left.pop(0)
                if not left:  # drop empty entries, so a saved run writes nothing for them
                    del in_item_order[fingerprint]
                    if not in_item_order:
                        del self.reused_in_item_order[path]
                return outcome
        same_call = self.reused_freely_by_call.get(path, {}).get(item_indexes)
        if same_call is not None and same_call[0] == fingerprint:
            return same_call[1]
        return freely.get(fingerprint) if freely else None

    def drop(self, paths: Collection[NodePath]) -> None:
        """Drops every outcome the nodes at ``paths`` may reuse.

        A node that is complete has no call left to reuse them, and a forgotten node must be called again.
        """
        for by_node in (self.reused_freely_by_call, self.reused_freely, self.reused_in_item_order):
            for path in paths:
                by_node.pop(path, None)

    def copy(self) -> ReusableOutcomes:
        """Returns a copy to take outcomes from and drop nodes from, leaving this one as it is. The outcomes are shared."""
        return ReusableOutcomes(
            dict(self.reused_freely_by_call),
            dict(self.reused_freely),
            {
                path: {fingerprint: list(left) for fingerprint, left in of_node.items()}
                for path, of_node in self.reused_in_item_order.items()
            },
            self.code_changed,
        )


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
        return json_hash(hashed, 12, fallback=untyped_to_hash)
    except (TypeError, ValueError, PydanticSerializationError):
        return None
