"""Reuse: taking an outcome from a previous run in place of a call that would get the same arguments.

The example graph is the README's, ``lines -> shout -> join``::

    run = graph.run(text="a\\nb\\nc")    # shout is called for "a", "b" and "c"
    run = run.again(text="a\\nb\\nd")    # shout is called for "d" only

In the second run the calls of ``shout`` for ``"a"`` and ``"b"`` get the same arguments as before, so their
``Done`` outcomes are reused and the function is not called. Arguments are compared by their **arguments
fingerprint**: a short hash of the arguments and of the node's code fingerprint (a hash of its version, its code,
the types of its inputs and outputs, and its loops), so a changed function is called again.

A node that may ask a person (``@node(asks=True)``) is reused another way: two of its calls with the same arguments
may have been answered differently, so each of its outcomes is reused once, by the calls with those arguments in
item order. A call takes the answer the person gave then, a question still waiting goes on waiting, and what the
function returned without asking is taken once too: a call with those arguments beyond the outcomes there were is
called again.
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
    """The outcomes of a previous run that a run may reuse in place of calling a node, by node path.

    A run going on from another takes them with ``from_run``; its scheduler takes each outcome it reuses with
    ``take`` and drops a node's with ``drop`` once the node is complete. For ``shout`` after the first run above,
    ``reused_freely_by_call[("shout",)]`` is ``{(0,): (fp_a, Done({"result": "A!"})), (1,): (fp_b, ...),
    (2,): (fp_c, ...)}``, where ``fp_a`` is the arguments fingerprint of the call for ``"a"``.

    A node that may not ask has its outcomes in ``reused_freely`` and ``reused_freely_by_call``, a node that may ask
    in ``reused_in_item_order``. The two never mix: whether a node may ask is part of its code fingerprint, so an
    outcome is only reused by a node that asks as the one that made it did.

    Attributes:
        reused_freely_by_call: For a node that may not ask, by node path, each outcome by the item indexes of its
            call, with the call's arguments fingerprint. Reused by the call with the same item indexes when its
            arguments are the same.
        reused_freely: For a node that may not ask, by node path, each outcome by its call's arguments fingerprint.
            Reused by any call with the same arguments. When several calls had the same arguments, the first in
            item order is kept.
        reused_in_item_order: For a node that may ask, by node path, each outcome, ``Done`` or ``Unanswered``, by its
            call's arguments fingerprint and the fingerprint of its outer items (see ``_outer_items_fingerprint``),
            in the order of the calls. Each is reused once, by the calls with those arguments in item order, and
            taken out.
        code_changed: For a node whose outcomes in the run a saved run made with other code or another signature,
            the code fingerprint they were made with. The run never passes them on to be reused, and saves them
            under that fingerprint.
    """

    reused_freely_by_call: dict[NodePath, dict[ItemIndexes, tuple[str, Done]]] = field(default_factory=dict[NodePath, dict[ItemIndexes, tuple[str, Done]]])
    reused_freely: dict[NodePath, dict[str, Done]] = field(default_factory=dict[NodePath, dict[str, Done]])
    reused_in_item_order: dict[NodePath, dict[tuple[str, str], list[Done | Unanswered]]] = field(
        default_factory=dict[NodePath, dict[tuple[str, str], list["Done | Unanswered"]]]
    )
    code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])

    @classmethod
    def from_run(cls, run: Run[Any], into: RunBlueprint) -> ReusableOutcomes:
        """What a run going on from ``run`` may reuse: each node's finished and waiting calls, in item order, then
        the outcomes ``run`` itself could reuse and did not.

        ``into`` is the blueprint of the new run's graph, ``run``'s or another's. Only the nodes it has as they are
        in ``run`` (by ``code_fingerprint``), and whose ``Policy`` there says ``reuse_when_inputs_match``, are kept:
        the others' outcomes could never be taken there, and kept they would only make its save larger.
        """
        reusable, previous, reader = cls(), run._reusable, run._reader
        for path, old_node in run._blueprint.function_nodes.items():
            new_node = into.function_nodes.get(path)
            if new_node is None or not new_node.policy.reuse_when_inputs_match or new_node.compiled_node.code_fingerprint != old_node.compiled_node.code_fingerprint:
                continue
            if path not in previous.code_changed:
                for item_indexes, outcome in run._outcomes[path].items():
                    match outcome:
                        case Skipped() | NoneInput():  # the node was not called for the item, so there is nothing to reuse
                            continue
                        case Done() | Unanswered():  # the node was called, so its arguments are settled
                            arguments = cast("dict[str, Any]", reader.arguments_for(old_node, item_indexes))
                            fingerprint = _arguments_fingerprint(old_node.compiled_node.code_fingerprint, arguments)
                        case _:
                            assert_never(outcome)
                    if fingerprint is not None:
                        outer_items = _outer_items_fingerprint(run._outcomes, old_node, item_indexes) if new_node.compiled_node.definition.asks else ""
                        reusable.add(new_node, fingerprint, outcome, item_indexes=item_indexes, outer_items=outer_items)
            for fingerprint, done in previous.reused_freely.get(path, {}).items():
                reusable.add(new_node, fingerprint, done)
            for (fingerprint, outer_items), outcomes in previous.reused_in_item_order.get(path, {}).items():
                for outcome in outcomes:
                    reusable.add(new_node, fingerprint, outcome, outer_items=outer_items)
        return reusable

    def add(
        self, node: BlueprintFunctionNode, fingerprint: str, outcome: Done | Unanswered, *, item_indexes: ItemIndexes | None = None,
        outer_items: str = "",
    ) -> None:
        """Add one outcome ``node`` may reuse, after those already added: to ``reused_in_item_order`` when the node
        may ask a person, else to ``reused_freely``.

        Args:
            node: The node.
            fingerprint: The arguments fingerprint of the call the outcome is of.
            outcome: The outcome.
            item_indexes: For a node that may not ask, the item indexes of the call, when they are known: the call
                with the same item indexes takes the outcome first.
            outer_items: For a node that may ask, the fingerprint of the outer items of the call.
        """
        path = node.node_path
        match outcome:
            case Done() if not node.compiled_node.definition.asks:
                self.reused_freely.setdefault(path, {}).setdefault(fingerprint, outcome)
                if item_indexes is not None:
                    self.reused_freely_by_call.setdefault(path, {})[item_indexes] = (fingerprint, outcome)
            case Done() | Unanswered():
                self.reused_in_item_order.setdefault(path, {}).setdefault((fingerprint, outer_items), []).append(outcome)

    def take(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
        in_turn: Callable[[], bool],
    ) -> Done | Unanswered | Literal[NotReady.NOT_ITS_TURN] | None:
        """The outcome a call reuses in place of calling the function, or ``None`` when there is none.

        In the second run above, the call ``(1,)`` of ``shout`` gets ``{"line": "b", "mark": "!"}`` and reuses
        ``Done({"result": "B!"})``; the call ``(2,)`` gets ``"d"``, which no call got before, so the result is ``None``
        and ``shout`` is called.

        For a node that may not ask, the outcome of the call with the same item indexes is looked for first, then one
        of any call. A node that may ask has its outcomes reused once each, in the order of the calls: an outcome
        reused is taken out, and while one with outputs is left and an earlier call of the node has not been tried
        yet, the result is ``NOT_ITS_TURN`` and the call is tried again later. Questions still waiting are alike, so
        a call that would reuse one takes it at once.

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
        fingerprint = _arguments_fingerprint(node.compiled_node.code_fingerprint, arguments)
        if fingerprint is None:
            return None
        if in_item_order:
            same_arguments = (fingerprint, _outer_items_fingerprint(outcomes, node, item_indexes))
            left = in_item_order.get(same_arguments)
            if left:
                if any(type(outcome) is Done for outcome in left) and not in_turn():
                    return NotReady.NOT_ITS_TURN
                outcome = left.pop(0)
                if not left:  # a node with no outcome left has no entry, so a saved run writes none
                    del in_item_order[same_arguments]
                    if not in_item_order:
                        del self.reused_in_item_order[path]
                return outcome
        same_call = self.reused_freely_by_call.get(path, {}).get(item_indexes)
        if same_call is not None and same_call[0] == fingerprint:
            return same_call[1]
        return freely.get(fingerprint) if freely else None

    def drop(self, paths: Collection[NodePath]) -> None:
        """Drop every outcome the nodes at ``paths`` may reuse: a complete node has no call left to take them, and a
        forgotten node must be called again."""
        for by_node in (self.reused_freely_by_call, self.reused_freely, self.reused_in_item_order):
            for path in [path for path in by_node if path in paths]:
                del by_node[path]

    def copy(self) -> ReusableOutcomes:
        """A copy to take outcomes from and drop nodes from without changing this one. The outcomes are shared."""
        return ReusableOutcomes(
            dict(self.reused_freely_by_call), dict(self.reused_freely),
            {path: {same_arguments: list(left) for same_arguments, left in of_node.items()} for path, of_node in self.reused_in_item_order.items()}, self.code_changed,
        )


def _outer_items_fingerprint(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode, item_indexes: ItemIndexes) -> str:
    """A fingerprint of the items of the outer loops of a call: every loop the node is inside but the innermost.

    It keeps the outcomes of a node that may ask, a person's answers among them, with the item they were given
    for. In ``pages -> lines -> approve``, two pages may have the same line; the outer-items fingerprint of each
    call of ``approve`` is a hash of its page, so a page keeps its own answers when the pages are reordered. The innermost item, the line, is one of the call's
    arguments, so it is not hashed again. A node inside fewer than two loops gets ``""``, and so does a call whose
    outer items have no JSON form.
    """
    # The outcome of the node that returned the series has outputs: a call inside a loop is made only after the series exists.
    above = [
        outcomes[series][item_indexes[:depth]].outputs[output][item_indexes[depth]]  # type: ignore[union-attr]
        for depth, (series, output) in enumerate(node.loops[:-1])
    ]
    return (_arguments_fingerprint("", {"items": above}) or "") if above else ""


def _arguments_fingerprint(code_fingerprint: str, arguments: dict[str, Any]) -> str | None:
    """A call's arguments and its node's code fingerprint as a short hash, or ``None`` when an argument has no JSON form.

    A call whose arguments have no fingerprint is never reused. A dict read back from a database may have its keys
    in another order; ``json_hash`` sorts them, so the dict hashes as it did before it was saved.

    Args:
        code_fingerprint: The node's code fingerprint.
        arguments: The call's arguments, by input name.
    """
    try:
        return json_hash([code_fingerprint, arguments], 12, fallback=untyped_to_hash)
    except (TypeError, ValueError, PydanticSerializationError):
        return None
