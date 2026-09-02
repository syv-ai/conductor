"""What is wrong with a graph, as data.

Two audiences, one list. An editor renders every problem, anchored to
whatever it is about; a run fails on the first fatal one. Because it is
data rather than an exception, the same compile serves both.

``code`` is stable and is what a frontend keys on. ``message`` is Danish and
for a person; nothing parses it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Problem"]


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a graph, said once.

    Compile is the only writer — a node has no ``Problem`` channel, and the
    engine has none either. Two readers, from one list: the editor renders
    every problem, anchored by ``node_id`` and optionally ``field`` to the
    thing it is about, while a run refuses to start on the first ``fatal``
    one. Constructed through the named makers below, never inline, so a code
    and its Danish message have one home.

    **Every problem is about a node**, so ``node_id`` is required and the
    anchor has two states, not three: a node, or one of that node's fields.
    A graph is only ever wrong somewhere — an empty flow is not broken, it is
    empty — and a nullable anchor buys a state nothing produces at the price
    of a ``None`` check in every reader.

    Nor is the anchor a ``Ref``. A ``Ref`` is an address that resolves, and
    half of these name a field that is not there — that *is* the problem
    (``stale_binding``, ``unknown_ref_output``, ``parameter_name_invalid``) —
    while ``misaligned`` is about a pair of addresses and anchors on neither.

    Its runtime counterpart is ``ErrorCause``: this one is about a graph that
    cannot run, that one about a run that went wrong. Not an exception —
    an editor mid-edit is allowed to be broken, and an exception would make
    compile refuse to answer at all.
    """

    #: Stable. The frontend keys on this and never on the message.
    code: str

    #: Danish, user-facing, and never parsed.
    message: str

    #: Whether this stops the flow running — one boolean, not a severity
    #: enum with a derived property. A misaligned pair of series is fatal; a
    #: binding on a name the roster lacks is not.
    fatal: bool

    #: Which node this is about. Required: every way a graph can be wrong is
    #: wrong at a placement, so there is no graph-level problem and no
    #: ``None`` for a reader to defend against.
    node_id: str

    #: Which of that node's fields, if it is about one. An input, an output
    #: handle, a template — hence ``field`` rather than ``input_name``: a
    #: problem is not always about something a value flows *into*.
    field: str | None = None
