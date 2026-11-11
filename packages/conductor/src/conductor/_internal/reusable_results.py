"""The work of an earlier run that a run may take in place of calling a node: what is kept of it, the
fingerprint of what a unit received, and which earlier unit a unit takes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pydantic_core

from conductor._internal.canonical_json import canonical_json
from conductor._internal.outcomes import NotReady
from conductor._internal.value_type import untyped_to_hash

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode

from dataclasses import dataclass, field

from conductor._internal.address import ItemIndexes
from conductor._internal.outcomes import Answered, Done, Outcome, Waiting


@dataclass(slots=True)
class ReusableResults:
    """The work of one node that a run may take from the run it went on from, in place of calling the node.

    A person's answer, or a question still waiting, is taken once, by the next unit in order that received the
    same under the same items above it. A node's own result is taken by the unit at the same place when it
    received the same there, and else by what was received, whichever unit made it.

    Attributes:
        at: Each result by the instance its unit stood in, with the fingerprint of what it received.
        computed: A node's results by the fingerprint of what the unit received.
        asked: A person's answers and the questions still waiting, by the fingerprint of what the unit received and
            of the items above it, in the order they stood.
    """

    at: dict[ItemIndexes, tuple[str, Done]] = field(default_factory=dict[ItemIndexes, tuple[str, Done]])
    computed: dict[str, Done] = field(default_factory=dict[str, Done])
    asked: dict[tuple[str, str], list[Answered | Waiting]] = field(default_factory=dict[tuple[str, str], list["Answered | Waiting"]])


def reused_outcome(
    earlier: ReusableResults, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], outcomes: dict[str, dict[ItemIndexes, Outcome]], in_turn: Callable[[], bool],
) -> Done | Answered | Waiting | Literal[NotReady.NOT_ITS_TURN] | None:
    """The unit of ``earlier`` that ``node`` takes in ``item_indexes`` in place of a call on ``arguments``; ``None`` when
    there is none.

    A person's answer, or a question still waiting, is taken once, in order: when one is there to take and
    ``in_turn`` says a unit before this one hasn't been tried yet, this is ``NOT_ITS_TURN``, tried again once it
    has. It is taken only under the same items above, so two documents' identical lines keep each document's
    answers when the documents move. A node's own result is the one of the unit at the same place, when that
    received the same, and else one any earlier unit computed from the same.

    Args:
        earlier: What the node may take: this run's own to take from.
        node: The node.
        item_indexes: The instance.
        arguments: What the unit received.
        units: Each function's units, for the items above this one.
        in_turn: Whether every unit of the node before this one has been tried.
    """
    compiled_node = node.compiled_node
    fingerprint = fingerprint_of(compiled_node.reuse_digest, arguments, plain=compiled_node.plain_inputs)
    if fingerprint is None:
        return None
    if earlier.asked:
        answers = earlier.asked.get((fingerprint, lineage(outcomes, node, item_indexes)))
        if answers:
            if not in_turn():
                return NotReady.NOT_ITS_TURN
            return answers.pop(0)
    there = earlier.at.get(item_indexes)
    if there is not None and there[0] == fingerprint:
        return there[1]
    return earlier.computed.get(fingerprint)


def lineage(outcomes: dict[str, dict[ItemIndexes, Outcome]], node: FlatFunctionNode, item_indexes: ItemIndexes) -> str:
    """A fingerprint of the item of each map above the innermost that the instance ``item_indexes`` of ``node`` sits
    in, those of the subgraphs' nodes around it included. A unit's innermost item is in what it
    received."""
    above = [
        outcomes[series.name][item_indexes[:depth]].outputs[output][item_indexes[depth]]  # type: ignore[union-attr]  # a unit under an item of a Series holds that Series
        for depth, (series, output) in enumerate(node.loops[:-1])
    ]
    return (fingerprint_of("", {"items": above}) or "") if above else ""


def fingerprint_of(digest: str, arguments: dict[str, Any], *, plain: bool = False) -> str | None:
    """What a unit of a node (its ``reuse_digest``) received, as a short hash; ``None`` for a value with no JSON form, which is never reused.

    ``plain`` says no input holds a mapping, so no key came from a store: the inputs' own order is the node's.
    """
    try:
        # No key of a plain node's inputs came from a store, so they need no canonical form: its inputs' order is the node's.
        written = pydantic_core.to_json([digest, arguments], fallback=untyped_to_hash) if plain else canonical_json([digest, arguments], fallback=untyped_to_hash)
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError):
        return None
    return hashlib.blake2b(written, digest_size=12).hexdigest()
