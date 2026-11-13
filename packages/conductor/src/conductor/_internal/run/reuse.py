"""Reuse: taking an outcome from a previous run in place of a call that would get the same arguments.

The example graph is the README's, ``lines -> shout -> join``::

    run = graph.run(text="a\\nb\\nc")    # shout is called for "a", "b" and "c"
    run = run.again(text="a\\nb\\nd")    # shout is called for "d" only

In the second run the calls of ``shout`` for ``"a"`` and ``"b"`` get the same arguments as before, so their
``Done`` outcomes are reused and the function is not called. Arguments are compared by their **arguments
fingerprint**: a short hash of the arguments and of the node's code fingerprint (a hash of its version, its code,
the types of its inputs and outputs, and its loops), so a changed function is called again.

A person's answer is reused the same way: when a call would ask the question it asked in the previous run, the
answer given then is its outcome, and a question still waiting is still waiting.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

from pydantic_core import PydanticSerializationError

from conductor._internal.base import ItemIndexes, NodePath, json_hash
from conductor._internal.types import untyped_to_hash

from .flat import FlatFunctionNode, FlatGraph
from .outcome import Answered, Done, NoneInput, NotReady, Outcome, Skipped, Unanswered

if TYPE_CHECKING:
    from .type import Run


@dataclass(slots=True)
class ReusableOutcomes:
    """The outcomes of a previous run that a run may reuse in place of calling a node, by node path.

    A run going on from another takes them with ``from_run``; its scheduler takes each outcome it reuses with
    ``take`` and drops a node's with ``drop`` once the node is complete. For ``shout`` after the first run above,
    ``done_by_call[("shout",)]`` is ``{(0,): (fp_a, Done({"result": "A!"})), (1,): (fp_b, ...), (2,): (fp_c, ...)}``,
    where ``fp_a`` is the arguments fingerprint of the call for ``"a"``.

    Attributes:
        done_by_call: By node path, each ``Done`` outcome by the item indexes of its call, with the call's arguments
            fingerprint. Reused by the call with the same item indexes when its arguments are the same.
        done: By node path, each ``Done`` outcome by its call's arguments fingerprint. Reused by any call with the
            same arguments. When several calls had the same arguments, the first in item order is kept.
        answers: By node path, each ``Answered`` and ``Unanswered`` outcome by its call's arguments fingerprint and
            the fingerprint of its outer items (see ``_outer_items_fingerprint``), in the order of the calls. Each
            is reused once, and taken out.
        code_changed: For a node whose outcomes in the run a saved run made with other code or another signature,
            the code fingerprint they were made with. The run never passes them on to be reused, and saves them
            under that fingerprint.
    """

    done_by_call: dict[NodePath, dict[ItemIndexes, tuple[str, Done]]] = field(default_factory=dict[NodePath, dict[ItemIndexes, tuple[str, Done]]])
    done: dict[NodePath, dict[str, Done]] = field(default_factory=dict[NodePath, dict[str, Done]])
    answers: dict[NodePath, dict[tuple[str, str], list[Answered | Unanswered]]] = field(
        default_factory=dict[NodePath, dict[tuple[str, str], list["Answered | Unanswered"]]]
    )
    code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])

    @classmethod
    def from_run(cls, run: Run[Any], into: FlatGraph) -> ReusableOutcomes:
        """What a run going on from ``run`` may reuse: each node's finished and waiting calls, in item order, then
        the outcomes ``run`` itself could reuse and did not.

        ``into`` is the flat graph of the new run's graph, ``run``'s or another's. Only the nodes it has as they are
        in ``run`` (by ``code_fingerprint``), and whose ``Policy`` there says ``reuse_when_inputs_match``, are kept:
        the others' outcomes could never be taken there, and kept they would only make its save larger.
        """
        reusable, previous, reader = cls(), run._reusable, run._reader
        for path, node in run._flat.function_nodes.items():
            there = into.function_nodes.get(path)
            if there is None or not there.policy.reuse_when_inputs_match or there.compiled_node.code_fingerprint != node.compiled_node.code_fingerprint:
                continue
            if path not in previous.code_changed:
                for item_indexes, outcome in run._outcomes[path].items():
                    match outcome:
                        case Skipped() | NoneInput():  # the node was not called for the item, so there is nothing to reuse
                            continue
                        case Done() | Answered() | Unanswered():  # the node was called, so its arguments are settled
                            arguments = cast("dict[str, Any]", reader.arguments_for(node, item_indexes))
                            fingerprint = _arguments_fingerprint(node.compiled_node.code_fingerprint, arguments)
                        case _:
                            assert_never(outcome)
                    if fingerprint is None:
                        continue
                    match outcome:
                        case Done():
                            reusable.add(path, fingerprint, outcome, item_indexes=item_indexes)
                        case Answered() | Unanswered():
                            reusable.add(path, fingerprint, outcome, outer_items=_outer_items_fingerprint(run._outcomes, node, item_indexes))
            for fingerprint, done in previous.done.get(path, {}).items():
                reusable.add(path, fingerprint, done)
            for (fingerprint, outer_items), answers in previous.answers.get(path, {}).items():
                for answer in answers:
                    reusable.add(path, fingerprint, answer, outer_items=outer_items)
        return reusable

    def add(
        self, path: NodePath, fingerprint: str, outcome: Done | Answered | Unanswered, *, item_indexes: ItemIndexes | None = None,
        outer_items: str = "",
    ) -> None:
        """Add one outcome the node at ``path`` may reuse, after those already added.

        Args:
            path: The node's path.
            fingerprint: The arguments fingerprint of the call the outcome is of.
            outcome: The outcome.
            item_indexes: For a ``Done`` outcome, the item indexes of its call, when they are known: the call with
                the same item indexes takes it first.
            outer_items: For an answer, the fingerprint of the outer items of its call.
        """
        match outcome:
            case Done():
                self.done.setdefault(path, {}).setdefault(fingerprint, outcome)
                if item_indexes is not None:
                    self.done_by_call.setdefault(path, {})[item_indexes] = (fingerprint, outcome)
            case Answered() | Unanswered():
                self.answers.setdefault(path, {}).setdefault((fingerprint, outer_items), []).append(outcome)

    def take(
        self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
        in_turn: Callable[[], bool],
    ) -> Done | Answered | Unanswered | Literal[NotReady.NOT_ITS_TURN] | None:
        """The outcome a call reuses in place of calling the function, or ``None`` when there is none.

        In the second run above, the call ``(1,)`` of ``shout`` gets ``{"line": "b", "mark": "!"}`` and reuses
        ``Done({"result": "B!"})``; the call ``(2,)`` gets ``"d"``, which no call got before, so the result is ``None``
        and ``shout`` is called.

        A person's answer is looked for first, then a ``Done`` outcome of the call with the same item indexes, then one
        of any call. Answers to the same question are reused once each, in the order of the calls: an answer reused is
        taken out, and when an earlier call of the node has not been tried yet, the result is ``NOT_ITS_TURN`` and the
        call is tried again later.

        Args:
            node: The node.
            item_indexes: The item indexes of the call.
            arguments: The arguments of the call.
            outcomes: The run's outcomes, by node path and then by item indexes.
            in_turn: Returns whether every earlier call of the node has been tried.
        """
        path = node.node_path
        done, answers = self.done.get(path), self.answers.get(path)
        if not done and not answers:
            return None
        fingerprint = _arguments_fingerprint(node.compiled_node.code_fingerprint, arguments)
        if fingerprint is None:
            return None
        if answers:
            question = (fingerprint, _outer_items_fingerprint(outcomes, node, item_indexes))
            asked = answers.get(question)
            if asked:
                if not in_turn():
                    return NotReady.NOT_ITS_TURN
                answer = asked.pop(0)
                if not asked:  # a node with no answer left has no entry, so a saved run writes none
                    del answers[question]
                    if not answers:
                        del self.answers[path]
                return answer
        same_call = self.done_by_call.get(path, {}).get(item_indexes)
        if same_call is not None and same_call[0] == fingerprint:
            return same_call[1]
        return done.get(fingerprint) if done else None

    def drop(self, paths: Collection[NodePath]) -> None:
        """Drop every outcome the nodes at ``paths`` may reuse: a complete node has no call left to take them, and a
        forgotten node must be called again."""
        for by_node in (self.done_by_call, self.done, self.answers):
            for path in [path for path in by_node if path in paths]:
                del by_node[path]

    def copy(self) -> ReusableOutcomes:
        """A copy to take outcomes from and drop nodes from without changing this one. The outcomes are shared."""
        return ReusableOutcomes(
            dict(self.done_by_call), dict(self.done),
            {path: {asked: list(left) for asked, left in of_node.items()} for path, of_node in self.answers.items()}, self.code_changed,
        )


def _outer_items_fingerprint(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: FlatFunctionNode, item_indexes: ItemIndexes) -> str:
    """A fingerprint of the items of the outer loops of a call: every loop the node is inside but the innermost.

    It keeps a person's answers with the item they were given for. In ``pages -> lines -> approve``, two pages may
    have the same line; the outer-items fingerprint of each call of ``approve`` is a hash of its page, so a page
    keeps its own answers when the pages are reordered. The innermost item, the line, is one of the call's
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
