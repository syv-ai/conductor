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
    "compute_hook_failed": "The node could not work out its fields from what it was given: {reason}. Check what is typed into the node and what is wired to it.",
    "duplicate_node_id": "Two nodes have the id '{node_id}'; rename one.",
    "edge_names_missing_node": "The edge from '{source}' to '{target}' names '{node_id}', which is not a node of the graph; add that node or remove the edge.",
    "embeds_itself": "The node '{node}' is a graph that contains itself; embed a graph that does not contain this one.",
    "default_on_later_node": "'{field}' is not an open input of the graph, so a caller can't change its default; make it a value.",
    "hook_needs_value": "'{field}' must be typed in: the node works out its fields from it when the graph loads.",
    "invalid_node_id": "'{node_id}' cannot be a node id: '.', '/', '@' and '~' each mean something in an address or a waiting id; rename it without them.",
    "parameter_name_invalid": "'{field}' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field.",
    "repeated_edge": "The edge from '{source}' into '{field}' is drawn twice; remove one.",
    "alternatives_in_different_series": (
        "'{field}' takes alternatives that run per item of different Series, {series}, so a call for an item of one has no place "
        "for the other's value. Bring them to items of one Series first."
    ),
    "several_edges_into_one_input": (
        "More than one of the edges into '{field}', from {sources}, can hold a value on the same call. "
        "Feed it from opposite branches of one decision, or put a node between that has an input for each."
    ),
    "series_inputs_never_share_an_item": (
        "{inputs} read branches that never both have a value for an item, so no item would reach the node. "
        "Write Series[X | None] on them to keep every item, or take the branches through one input."
    ),
    "series_not_nested": (
        "'{node_id}' reads items of {series}, and neither is made inside the other, so nothing says which item goes with which. "
        "Make one inside the other (wire '{wire_from}' into '{wire_into}', the node that makes the other), or give one of them whole to a Series input."
    ),
    "unbound_on_later_node": "Nothing feeds '{field}', and only a first node takes inputs from the caller; draw an edge into it or type a value in.",
    "unknown_node": "There is no node '{node}'; the nodes are {choices}.",
    "unknown_output": "'{source}' is not something that node gives; it gives {outputs}.",
    "unknown_version": "'{node}' has no version {version}; the newest is {newest}.",
    "value_and_default": "Field '{field}' has both a value and a default; keep one.",
    # -- what a call is refused for ----------------------------------------------
    "invalid_answer": "That is not a valid answer: {reason}",
    "input_given_twice": "'{name}' is given twice, by its name and by its address '{address}'; give it once.",
    "invalid_max_items": "max_items is at least 1; it was {max_items}.",
    "invalid_checkpoint_every": "checkpoint_every is a number of seconds above 0; it was {checkpoint_every}.",
    "invalid_timeout": "timeout is a number of seconds above 0; it was {timeout}.",
    "missing_input": "'{name}' is required; give it in the call.",
    "node_not_in_graph": "'{node_id}' is not a node of this graph; it has {nodes}.",
    "run_not_complete": "The run is {status}, so it has no outputs yet; run.results has what it did. {reason}",
    "state_of_another_graph": "The saved run is of another graph ({saved}) than this one ({graph}); an edited graph starts a new run.",
    "state_unreadable": "The saved run does not fit this graph at '{at}': {reason}. Load it into the graph it was saved from.",
    "state_unwritable": "The run can't be written as JSON at '{at}': {reason}. Give that field a type pydantic writes as JSON.",
    # -- why a node or a run stopped -------------------------------------------
}

#: The English of a problem whose ``ctx.hint`` names the fix, by its code and that hint, in place of the code's own
#: entry: the same code, said for the one mistake the hint is for.
HINTED_CATALOGUE: Mapping[tuple[str, str], str] = {
    ("unknown_output", "split_first"): (
        "'{source}' names a field of a whole {whole}, which has no field to wire. Split it first: put a Split after it and wire "
        "the field from Split's result."
    ),
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
