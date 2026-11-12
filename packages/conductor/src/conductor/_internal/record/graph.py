"""``GraphRecord``: a graph as a host stores it, its nodes and its edges, ``{"nodes": [...], "edges": [["lines.result", "shout.line"]]}``.

Edges are kept sorted, so two records that differ only in the order of their edges are equal and save alike. The
package's docstring has the whole record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, field_validator

from conductor._internal.base import ConductorModel, split_address
from conductor._internal.record.node import JSON_ONLY, NodeRecord


def _checked_address(address: str) -> str:
    """``address`` once it reads ``node.field``, or ``node.output.field`` for one field of a record an output
    holds; that field's own name may hold dots."""
    node_id, rest = split_address(address)
    field, dot, of_record = rest.partition(".")
    if not node_id or not field or (dot and not of_record):
        raise ValueError(f"{address!r} is not an address; write 'node.field', or 'node.output.field' for one field of a record")
    return address


#: An address, ``"node.field"`` or ``"node.output.field"`` for one field of a record an output holds: node ids and
#: a node's field names hold no dot, so everything after the second is the record's field (``"split.result.Amount in USD."``).
Address = Annotated[str, AfterValidator(_checked_address)]


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes and its edges.

    What a graph takes and returns is worked out when it is compiled, never stored: its graph inputs, the inputs
    with no edge on the nodes no edge leads into, and its graph outputs, the outputs of the nodes no edge leaves.
    What a host keeps about the whole graph (its name, a viewport) it stores beside the record.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    #: Any sequence going in; a tuple once built.
    nodes: Sequence[NodeRecord] = ()
    #: ``(from, to)`` pairs of addresses, ``("read.text", "up.text")``; sorted once built.
    edges: Sequence[tuple[Address, Address]] = ()

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
        return tuple(nodes)

    @field_validator("edges")
    @classmethod
    def _edges_sorted(cls, edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(edges))
