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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pydantic_core
from pydantic_core import PydanticSerializationError

from conductor._internal.hashing import canonical_json, short_hash
from conductor._internal.outcomes import NotReady
from conductor._internal.value_type import untyped_to_hash

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode

from dataclasses import dataclass, field

from conductor._internal.address import ItemIndexes
from conductor._internal.outcomes import Answered, Done, Outcome, Waiting


@dataclass(slots=True)
class ReusableOutcomes:
    """The outcomes of one node in the previous run that a run may reuse.

    A run makes these from the previous run's outcomes, and takes answers out of them as it reuses them. For ``shout`` after the first run above,
    ``by_item_indexes`` is ``{(0,): (fp_a, Done({"result": "A!"})), (1,): (fp_b, ...), (2,): (fp_c, ...)}``, where
    ``fp_a`` is the arguments fingerprint of the call for ``"a"``.

    Attributes:
        by_item_indexes: Each ``Done`` outcome by the item indexes of its call, with the call's arguments
            fingerprint. Reused by the call with the same item indexes when its arguments are the same.
        by_fingerprint: Each ``Done`` outcome by its call's arguments fingerprint. Reused by any call with the same
            arguments. When several calls had the same arguments, the first in item order is kept.
        answers: Each ``Answered`` and ``Waiting`` outcome, by its call's arguments fingerprint and its outer-items
            fingerprint (see ``outer_items_fingerprint``), in the order of the calls. Each is reused once.
    """

    by_item_indexes: dict[ItemIndexes, tuple[str, Done]] = field(default_factory=dict[ItemIndexes, tuple[str, Done]])
    by_fingerprint: dict[str, Done] = field(default_factory=dict[str, Done])
    answers: dict[tuple[str, str], list[Answered | Waiting]] = field(default_factory=dict[tuple[str, str], list["Answered | Waiting"]])


def reused_outcome(
    previous: ReusableOutcomes, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any],
    outcomes: dict[str, dict[ItemIndexes, Outcome]], in_turn: Callable[[], bool],
) -> Done | Answered | Waiting | Literal[NotReady.NOT_ITS_TURN] | None:
    """The outcome a call reuses from the previous run in place of calling the function, or ``None`` when there is none.

    In the second run above, the call ``(1,)`` of ``shout`` gets ``{"line": "b", "mark": "!"}`` and reuses
    ``Done({"result": "B!"})``; the call ``(2,)`` gets ``"d"``, which no call got before, so the result is ``None``
    and ``shout`` is called.

    A person's answer is looked for first, then a ``Done`` outcome of the call with the same item indexes, then one
    of any call. Answers to the same question are reused once each, in the order of the calls: when an earlier call
    of the node has not been tried yet, the result is ``NOT_ITS_TURN`` and the call is tried again later.

    Args:
        previous: The node's reusable outcomes. An answer reused is taken out of them.
        node: The node.
        item_indexes: The item indexes of the call.
        arguments: The arguments of the call.
        outcomes: The run's outcomes, by node name and then by item indexes.
        in_turn: Returns whether every earlier call of the node has been tried.
    """
    compiled_node = node.compiled_node
    fingerprint = arguments_fingerprint(compiled_node.code_fingerprint, arguments, plain=compiled_node.plain_inputs)
    if fingerprint is None:
        return None
    if previous.answers:
        answers = previous.answers.get((fingerprint, outer_items_fingerprint(outcomes, node, item_indexes)))
        if answers:
            if not in_turn():
                return NotReady.NOT_ITS_TURN
            return answers.pop(0)
    previous_for_call = previous.by_item_indexes.get(item_indexes)
    if previous_for_call is not None and previous_for_call[0] == fingerprint:
        return previous_for_call[1]
    return previous.by_fingerprint.get(fingerprint)


def outer_items_fingerprint(outcomes: dict[str, dict[ItemIndexes, Outcome]], node: FlatFunctionNode, item_indexes: ItemIndexes) -> str:
    """A fingerprint of the items of the outer loops of a call: every loop the node is inside but the innermost.

    It keeps a person's answers with the item they were given for. In ``pages -> lines -> approve``, two pages may
    have the same line; the outer-items fingerprint of each call of ``approve`` is a hash of its page, so a page
    keeps its own answers when the pages are reordered. The innermost item, the line, is one of the call's
    arguments, so it is not hashed again. A node inside fewer than two loops gets ``""``, and so does a call whose
    outer items have no JSON form.
    """
    # The outcome of the node that returned the series has outputs: a call inside a loop is made only after the series exists.
    above = [
        outcomes[series.name][item_indexes[:depth]].outputs[output][item_indexes[depth]]  # type: ignore[union-attr]
        for depth, (series, output) in enumerate(node.loops[:-1])
    ]
    return (arguments_fingerprint("", {"items": above}) or "") if above else ""


def arguments_fingerprint(code_fingerprint: str, arguments: dict[str, Any], *, plain: bool = False) -> str | None:
    """A call's arguments and its node's code fingerprint as a short hash, or ``None`` when an argument has no JSON form.

    A call whose arguments have no fingerprint is never reused.

    Args:
        code_fingerprint: The node's code fingerprint.
        arguments: The call's arguments, by input name.
        plain: ``True`` when no input takes a mapping. The arguments are then hashed in their own order, which is
            the order of the node's inputs. A mapping, such as a dict read back from a database, may have its keys
            in another order, so otherwise the keys are sorted first.
    """
    try:
        written = (
            pydantic_core.to_json([code_fingerprint, arguments], fallback=untyped_to_hash) if plain
            else canonical_json([code_fingerprint, arguments], fallback=untyped_to_hash)
        )
    except (TypeError, ValueError, PydanticSerializationError):
        return None
    return short_hash(written, 12)
