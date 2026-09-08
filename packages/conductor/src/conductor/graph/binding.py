"""Where an input's value comes from.

An input of a node in a graph is bound in one of two ways: by an edge
from another node's output (``Edges``), or by a value the author typed
(``Static``). A node's ``bindings`` map holds one binding per input the
author touched; an input with no entry takes its declared default::

    {
        "text":  Edges(refs=(Ref("reader", "text"),)),   # an edge in
        "limit": Static(value=200),                       # typed in
        # "language" absent: the input's declared default applies
    }

Because one input holds one binding, an edge and a typed value can never
both claim the same input.

The editor writes both: connecting two handles writes an ``Edges``,
editing a control writes a ``Static``. Compile reads them: the
dependency map, the engine's maps and the graph's own inputs and outputs
are all derived from the bindings (``topology``, ``views``); nothing
stores an edge as a record of its own.

There is deliberately no third kind for "the caller supplies this": a
run supplies values for a graph's inputs beside the graph, never inside
it, so a stored graph stays runnable on its own. And there is no guard
on an edge: a branch is an output of the node that decides, and a branch
not taken carries ``SKIPPED``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

from conductor.ref import Ref


@dataclass(frozen=True)
class Edges:
    """The input's value arrives over one or more edges from other nodes' outputs.

    ``refs`` names those outputs in the order the author connected them,
    and that order is the only order there is. A scalar input has one ref;
    a ``Series[X]`` input may have several, gathered into one series. There
    is no per-ref enable flag: muting an operand is deleting it.

    Written by the editor when two handles are connected; read by compile.
    Its one sibling is ``Static``.
    """

    refs: tuple[Ref, ...]


@dataclass(frozen=True)
class Static:
    """The input's value is one the author typed into the control; the graph stores it.

    Static as in fixed in the graph, the same on every run; nothing to do
    with static typing. The compiler validates ``value`` against the
    input's declared type; ``static_values`` hands the typed values to the
    engine as one dict.

    Written by the editor when a person edits the control, or by a seed
    when the node is placed; read by compile. Not a default: a default
    belongs to the node's declaration and applies when there is no binding
    at all. Its one sibling is ``Edges``.
    """

    value: Any


Binding: TypeAlias = Edges | Static


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
