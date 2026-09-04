"""Bindings — where an input's value comes from.

An author can do exactly two things to an input: draw a cable to it, or
type a value into it. A placement's ``bindings`` map holds one entry per
input the author touched::

    {
        "text":  Sources(refs=(Ref("reader", "text"),)),   # wired
        "limit": Static(value=200),                         # typed in
        # "language" absent: the input's declared default applies
    }

Because one input holds one binding, a wired cable and a typed value can
never both claim the same input. Absence is one state, not two.

There is deliberately no variant for "the caller supplies this": a run
supplies values for a flow's inputs beside the graph, never inside it, so
a stored flow stays runnable on its own. And there is no guard on a wire:
a branch is an output of the node that decides, and a branch not taken
carries ``SKIPPED``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

# ``Ref`` lives in ``conductor.ref``; re-exported here so the binding
# vocabulary reads in one place.
from conductor.ref import Ref


@dataclass(frozen=True)
class Sources:
    """The value comes from other nodes' outputs — the author drew a cable.

    ``refs`` is in operand order, and that order is the only order there
    is. Into a scalar input there is one ref; into a ``Series[X]`` input
    there may be several, and they are gathered into one series. There is
    no per-ref enable flag: muting an operand is deleting it.
    """

    refs: tuple[Ref, ...]


@dataclass(frozen=True)
class Static:
    """The author typed a value into the field.

    The compiler validates ``value`` against the input's declared type;
    ``static_values`` hands the typed values to the engine as one dict.
    Not a default: a default belongs to the node's declaration and applies
    when there is no binding at all.
    """

    value: Any


Binding: TypeAlias = Sources | Static


def static_values(bindings: Mapping[str, Binding]) -> dict[str, Any]:
    """The typed-in values as a plain dict, for the engine.

    Only ``Static`` contributes. An absent binding is left absent so the
    node's own default applies.
    """
    return {
        name: binding.value
        for name, binding in bindings.items()
        if isinstance(binding, Static)
    }


def many(value: Any) -> bool:
    """Is ``value`` a sequence of values, as a list widget or a multi-upload holds one?

    Text and bytes are one value each, not sequences. This is the one test
    for "a typed-in value that stands for many".
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
