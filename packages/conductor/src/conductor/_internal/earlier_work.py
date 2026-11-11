"""The work of an earlier run that a run may take in place of calling a node: what is kept of it, the
fingerprint of what a unit received, and which earlier unit a unit takes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pydantic_core

from conductor._internal.canonical_json import canonical_json
from conductor._internal.fields import written_table
from conductor._internal.units import NOT_ITS_TURN, NotReady, Units

if TYPE_CHECKING:
    from conductor._internal.loaded_graph.tree import FunctionAtPath

from dataclasses import dataclass, field

from conductor._internal.address import ItemPath
from conductor._internal.units import Answered, Asking, Done


@dataclass(slots=True)
class EarlierWork:
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

    at: dict[ItemPath, tuple[str, Done]] = field(default_factory=dict[ItemPath, tuple[str, Done]])
    computed: dict[str, Done] = field(default_factory=dict[str, Done])
    asked: dict[tuple[str, str], list[Answered | Asking]] = field(default_factory=dict[tuple[str, str], list["Answered | Asking"]])


def taken(
    earlier: EarlierWork, node: FunctionAtPath, key: ItemPath, kwargs: dict[str, Any], units: Units, in_turn: Callable[[], bool],
) -> Done | Answered | Asking | Literal[NotReady.NOT_ITS_TURN] | None:
    """The unit of ``earlier`` that ``node`` takes in ``key`` in place of a call on ``kwargs``; ``None`` when
    there is none.

    A person's answer, or a question still waiting, is taken once, in order: when one is there to take and
    ``in_turn`` says a unit before this one hasn't been tried yet, this is ``NOT_ITS_TURN``, tried again once it
    has. It is taken only under the same items above, so two documents' identical lines keep each document's
    answers when the documents move. A node's own result is the one of the unit at the same place, when that
    received the same, and else one any earlier unit computed from the same.

    Args:
        earlier: What the node may take: this run's own to take from.
        node: The node.
        key: The instance.
        kwargs: What the unit received.
        units: Each function's units, for the items above this one.
        in_turn: Whether every unit of the node before this one has been tried.
    """
    placed = node.placed
    fingerprint = fingerprint_of(placed.reuse_digest, kwargs, plain=placed.plain_inputs)
    if fingerprint is None:
        return None
    if earlier.asked:
        answers = earlier.asked.get((fingerprint, lineage(units, node, key)))
        if answers:
            if not in_turn():
                return NOT_ITS_TURN
            return answers.pop(0)
    there = earlier.at.get(key)
    if there is not None and there[0] == fingerprint:
        return there[1]
    return earlier.computed.get(fingerprint)


def lineage(units: Units, node: FunctionAtPath, key: ItemPath) -> str:
    """A fingerprint of the item of each map above the innermost that the instance ``key`` of ``node`` sits
    in, those of the embedded graphs' nodes around it included. A unit's innermost item is in what it
    received."""
    above = [
        units[series.name][key[:depth]].out[output][key[depth]]  # type: ignore[union-attr]  # a unit under an item of a Series holds that Series
        for depth, (series, output) in enumerate(node.maps[:-1])
    ]
    return (fingerprint_of("", {"items": above}) or "") if above else ""


def fingerprint_of(digest: str, kwargs: dict[str, Any], *, plain: bool = False) -> str | None:
    """What a unit of a node (its ``reuse_digest``) received, as a short hash; ``None`` for a value with no JSON form, which is never reused.

    ``plain`` says no input holds a mapping, so no key came from a store: the inputs' own order is the node's.
    """
    try:
        # No key of a plain node's inputs came from a store, so they need no canonical form: its inputs' order is the node's.
        written = pydantic_core.to_json([digest, kwargs], fallback=written_table) if plain else canonical_json([digest, kwargs], fallback=written_table)
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError):
        return None
    return hashlib.blake2b(written, digest_size=12).hexdigest()
