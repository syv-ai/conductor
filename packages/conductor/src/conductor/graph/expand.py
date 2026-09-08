"""The address rule between a placement and the nodes it expands to.

An embedded flow is one node in the editor — a placement — and many
nodes when it runs: its inner nodes, named ``placement/inner``. ``/``
separates namespace levels inside a node id (an editor never mints one)
and ``.`` stays the address separator, so an expanded address reads
``approve/check.amount``. ``expanded_ref`` reads a placement-side
address through to the inner field it names, as deep as the placements
go; a graph with no embedded flow has no placements, and every address
reads as itself.
"""

from __future__ import annotations

from conductor.ref import Ref

SEPARATOR = "/"


def expanded_ref(ref: Ref, placements: frozenset[str] | set[str]) -> Ref:
    """``Ref("approve", "check.amount")`` → ``Ref("approve/check", "amount")``, as deep as the placements go."""
    node_id, field = ref.node_id, ref.field
    while node_id in placements and "." in field:
        inner, field = field.split(".", 1)
        node_id = f"{node_id}{SEPARATOR}{inner}"
    return Ref(node_id, field)
