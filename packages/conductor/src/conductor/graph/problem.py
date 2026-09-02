"""``Problem`` — something wrong with a graph, as data rather than an exception.

Compiling a flow yields a list of problems. An editor shows every one,
pointing at the node (and field) it is about, so an author can be
mid-edit with a broken graph and still see what to fix; a run refuses to
start on the first ``fatal`` one. Because the same list serves both,
compile never raises for a problem in the graph::

    Problem(code="unknown_input", message="The node has no such field.",
            fatal=True, node_id="letter", field="template")

``code`` is stable and is what a frontend keys on. ``message`` is English
and for a person; ``details`` holds what the message was formatted from
(the field, the node type, the source ref) under stable keys, so a host
can say the same thing in its own language from ``code`` and ``details``
without parsing the message.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

__all__ = ["Problem"]


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a graph.

    Every problem is about a node, so ``node_id`` is required; ``field``
    narrows it to one of that node's fields when the problem is about one.
    ``details`` carries the values the message names, so a host that
    translates by ``code`` has them::

        Problem(code="unknown_ref_node", message="Field 'text' is wired to 'a', which is not in the flow.",
                fatal=True, node_id="b", field="text", details={"source_node": "a"})
    There is no graph-level problem — an empty flow is not broken, it is
    empty — so readers never have to handle a missing anchor.

    The anchor is a plain node id and field name rather than a ``Ref``,
    because half of all problems are about a field that does *not* exist
    (a binding to a removed input, a reference to an unknown output), and
    a ``Ref`` is meant to resolve.

    A node failing while a flow *runs* is an exception carrying its own
    cause, not a ``Problem``: this record is about a graph that cannot run.
    """

    #: Stable identifier the frontend keys on.
    code: str

    #: For a person, in English. Never parsed; a host translates by ``code``.
    message: str

    #: Whether this stops the flow from running. A plain boolean rather than
    #: a severity level.
    fatal: bool

    #: The node the problem is about. Required.
    node_id: str

    #: The field on that node, when the problem is about one — an input, an
    #: output or a template alike, hence ``field`` rather than ``input_name``.
    field: str | None = None

    #: What the message was formatted from, under stable keys. Empty when
    #: the message names nothing beyond the anchor.
    details: Mapping[str, Any] = dataclass_field(default_factory=dict)
