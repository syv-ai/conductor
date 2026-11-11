"""The problems the engine modules still write from a table of sentences by code, beside the catalogue's functions.

``make_problem(code, *loc, **ctx)`` builds a ``Problem`` from ``PROBLEM_CATALOGUE``, the English for a code with the
``ctx`` keys as slots, and ``HINTED_CATALOGUE``, the English for a code said for the one mistake its ``ctx.hint``
names. ``PROBLEM_CODES`` is every code conductor itself writes, these and the catalogue's functions together, so a
host translating by code has one set to cover.

A problem the engine writes carries the English for its code; a code that is in neither table is a node's own, from a
``NodeError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from conductor._internal.problem.catalogue import PROBLEMS
from conductor._internal.problem.type import Problem

#: Every code conductor itself writes, one entry each: its English, with
#: the ``ctx`` keys as slots. A compile problem's slots also take
#: ``{node_id}`` and ``{field}``, the first two places of its ``loc``. A code
#: that is not here is a node's own, from a ``NodeError``. A host that
#: translates by code covers ``PROBLEM_CODES``.
PROBLEM_CATALOGUE: Mapping[str, str] = {
    # -- what loading finds in a graph -----------------------------------------
    # -- what a call is refused for ----------------------------------------------
    # -- why a node or a run stopped -------------------------------------------
}

#: The English of a problem whose ``ctx.hint`` names the fix, by its code and that hint, in place of the code's own
#: entry: the same code, said for the one mistake the hint is for.
HINTED_CATALOGUE: Mapping[tuple[str, str], str] = {
}

#: The codes a host translating by code has to cover.
PROBLEM_CODES: frozenset[str] = frozenset(PROBLEM_CATALOGUE) | PROBLEMS.codes


#: A value ``make_problem`` is given for a problem's ``ctx``: JSON, written by the code that made the problem. Read-only
#: where pydantic's ``JsonValue`` is a ``list``, so a caller's ``list[str]`` is one.
CtxValue: TypeAlias = "str | int | float | bool | None | Sequence[CtxValue] | Mapping[str, CtxValue]"


def make_problem(code: str, /, *loc: str | int, **ctx: CtxValue) -> Problem:
    """The problem ``code`` at ``loc``, its message made from ``ctx``, which the record keeps as it is given.

    ``ctx`` is conductor's own, typed as JSON where it is written, so the record is built without reading it again.

    A list in ``ctx`` stays a list, and the message writes it joined by
    commas (``none`` when empty); in a list of Series, ``None`` stands for a
    node that runs once. A ``hint`` in ``ctx`` picks the code's entry in ``HINTED_CATALOGUE``.
    A code not in ``PROBLEM_CATALOGUE``, or a slot its entry
    names that ``ctx`` does not fill, is a programming error and raises; only
    ``reason`` may be left out, the sentence a type's own constructor raised,
    which not every one gives.
    """
    said = {key: _said(value) for key, value in ctx.items()}
    reason = said.get("reason")
    if isinstance(reason, str):
        said["reason"] = reason.rstrip(".")  # the entry's own sentence goes on after it
    slots = {"reason": "", "node_id": loc[0] if loc else None, "field": loc[1] if len(loc) > 1 else None, **said}
    entry = HINTED_CATALOGUE[code, str(ctx["hint"])] if "hint" in ctx else PROBLEM_CATALOGUE[code]
    return Problem.model_construct(type=code, loc=loc, msg=entry.format_map(slots).rstrip(), ctx=ctx)


def _said(value: CtxValue) -> CtxValue:
    """A ``ctx`` value as its message writes it: a list joined by commas, ``None`` in it a node that runs once."""
    if not isinstance(value, (list, tuple)):
        return value
    return ", ".join("none (it runs once)" if item is None else str(item) for item in value) or "none"
