"""``GraphRecord``: a graph as a host stores it, its nodes and its edges, ``{"nodes": [...], "edges": [["lines.result", "shout.line"]]}``.

Edges are kept sorted, so two records that differ only in the order of their edges are equal and save alike. The
package's docstring has the whole record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, field_validator

from conductor._internal.base import Address, ConductorModel

from .node import JSON_ONLY, NodeRecord

#: Where an edge comes from, as a graph record writes it: an output, ``"lines.result"``, or one field of the record an
#: output returns, ``"read.result.email"``; text that ``Address.parse`` reads, kept as the text.
SourceText = Annotated[str, AfterValidator(lambda text: str(Address.parse(text)))]

#: Where an edge leads, as a graph record writes it: an input, ``"shout.line"``; text that ``Address.parse_input``
#: reads, kept as the text.
TargetText = Annotated[str, AfterValidator(lambda text: str(Address.parse_input(text)))]


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
    edges: Sequence[tuple[SourceText, TargetText]] = ()

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
        return tuple(nodes)

    @field_validator("edges")
    @classmethod
    def _edges_sorted(cls, edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(edges))
