"""``Problem`` — something wrong with a graph, as data rather than an exception.

Compiling a graph yields a list of problems. An editor shows every one,
pointing at the node (and field) it is about, so an author can be
mid-edit with a broken graph and still see what to fix; a run refuses to
start on the first ``fatal`` one. Because the same list serves both,
compile never raises for a problem in the graph::

    problem("unknown_ref_node", "letter", "template", source_node="ghost")

``code`` is stable and is what a frontend keys on. ``message`` is English
and for a person; ``details`` holds what the message was formatted from
(the source ref, the node type) under stable keys, so a host can say the
same thing in its own language from ``code`` and ``details`` without
parsing the message.

Every code compile emits is declared once, in ``CATALOGUE`` below, with
its message and whether it is fatal; ``problem`` builds one from a row.
The two problems whose code a *host* chose — a hook raising ``Refuses``,
a type refusing to be handed over whole — are built as a ``Problem``
directly where they arise, with the host's own text.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a graph.

    Every problem is about a node, so ``node_id`` is required; ``field``
    narrows it to one of that node's fields when the problem is about one.
    ``details`` carries the values the message names, so a host that
    translates by ``code`` has them::

        Problem(code="unknown_ref_node", message="Field 'text' is connected to 'a', which is not in the flow.",
                fatal=True, node_id="b", field="text", details={"source_node": "a"})

    There is no graph-level problem — an empty graph is not broken, it is
    empty — so readers never have to handle a missing anchor.

    The anchor is a plain node id and field name rather than a ``Ref``,
    because half of all problems are about a field that does *not* exist
    (a binding to a removed input, a reference to an unknown output), and
    a ``Ref`` is meant to resolve.

    A node failing while a graph *runs* is an exception carrying its own
    cause, not a ``Problem``: this record is about a graph that cannot run.
    """

    #: Stable identifier the frontend keys on.
    code: str

    #: For a person, in English. Never parsed; a host translates by ``code``.
    message: str

    #: Whether this stops the graph from running. A plain boolean rather than
    #: a severity level.
    fatal: bool

    #: The node the problem is about. Required.
    node_id: str

    #: The field on that node, when the problem is about one — an input, an
    #: output or a template alike, hence ``field`` rather than ``input_name``.
    field: str | None = None

    #: What the message was formatted from, under stable keys. Empty when
    #: the message names nothing beyond the anchor. (``dataclasses.field``
    #: by its module name: this record's own ``field`` shadows it here.)
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)


#: Every code compile itself emits: its message, with ``{node_id}``,
#: ``{field}`` and the ``details`` keys as slots, and whether it is fatal.
#: The non-fatal ones are states an author can leave a graph in and still
#: run it: a binding or a lock on a field the node no longer has (nothing
#: reads it), a node whose outputs wait on a value the author has not
#: typed yet. A host that translates problems by code covers exactly
#: these keys; a code a host's own hook raised is the host's and is not
#: here.
CATALOGUE: Mapping[str, tuple[str, bool]] = {
    "cycle": ("The node is part of a cycle, so the flow cannot run.", True),
    "duplicate_field_name": ("The node has two fields named '{field}'.", True),
    "duplicate_node_id": ("Two nodes have the id '{node_id}'.", True),
    "edge_into_closed_handle": ("Field '{field}' has no handle, so nothing can be connected to it.", True),
    "handle_needs_dtype": ("Field '{field}' has a handle but no type that can travel on an edge.", True),
    "invalid_static": ("The value in '{field}' cannot be read as the field's type. {reason}", True),
    "misaligned": ("'{a}' and '{b}' get their rows from different sources, so the node cannot run per row.", True),
    "no_outputs": ("The node has no fields to pass on, so nothing can be connected from it.", False),
    "one_edge_per_parameter": ("Parameter '{field}' takes one edge; an extra edge is an extra parameter.", True),
    "parameter_name_invalid": (
        "'{field}' cannot be a parameter name; use letters, digits and underscores, and start with a letter.", True,
    ),
    "stale_binding": ("Field '{field}' no longer exists on the node.", False),
    "type_mismatch": ("'{source}' is {source_said}, but '{node_id}.{field}' takes {target_said}.", True),
    "unbound_required": ("Nothing is connected to the field.", True),
    "union_needs_one_index": (
        "Field '{field}' has several edges; that only works when they are all rows of one table.", True,
    ),
    "unknown_locked_field": ("The lock on '{node_id}.{field}' points at a field the node does not have.", False),
    "unknown_node_type": ("Node type '{node_type}' does not exist.", True),
    "unknown_node_version": ("'{node_type}' has no version {version}.", True),
    "unknown_ref_node": ("Field '{field}' is connected to '{source_node}', which is not in the flow.", True),
    "unknown_ref_output": ("Field '{field}' is connected to '{source}', which is not an output of that node.", True),
}

#: The codes a host translating by code has to cover.
CODES: frozenset[str] = frozenset(CATALOGUE)


#: The one slot a caller may leave unfilled: the sentence a type's own
#: constructor raised, which not every constructor gives. Any other slot a
#: template names must be supplied, or ``problem`` raises where it is
#: called — a typo in a template or a forgotten keyword is a programming
#: error, not an empty message.
OPTIONAL_SLOTS: frozenset[str] = frozenset({"reason"})


class _Slots(dict[str, Any]):
    """Formatting slots: an optional slot the caller left out renders empty;
    any other missing slot raises."""

    def __missing__(self, key: str) -> str:
        if key in OPTIONAL_SLOTS:
            return ""
        raise KeyError(f"the message for this problem needs {key!r}")


def problem(code: str, node_id: str, field: str | None = None, **details: Any) -> Problem:
    """The problem ``code`` on ``node_id`` (and ``field``), its message
    formatted from ``details``, which the record keeps.

    A code not in ``CATALOGUE``, or a slot the template names that
    ``details`` does not fill (``OPTIONAL_SLOTS`` aside), is a programming
    error and raises.
    """
    template, fatal = CATALOGUE[code]
    message = template.format_map(_Slots(node_id=node_id, field=field, **details)).rstrip()
    return Problem(code=code, message=message, fatal=fatal, node_id=node_id, field=field, details=details)
