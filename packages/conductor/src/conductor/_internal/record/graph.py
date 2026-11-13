"""``GraphRecord``: a graph as a host stores it, its nodes by id, ``{"nodes": {"shout": {"node": "shout"}}}``. The
package's docstring has a whole record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import MISSING, AfterValidator
from pydantic_core import PydanticCustomError

from conductor._internal.base import Address, ConductorModel, Name

from .node import JSON_ONLY, NodeRecord


def _whole_output(address: Address) -> Address:
    """Returns ``address`` when it names a whole output.

    Raises:
        PydanticCustomError: It names one field of an output (``output_has_field``): a graph output is a whole output.
    """
    if address.field is not None:
        raise PydanticCustomError("output_has_field", "a graph output is a whole output; give a node the field through an edge")
    return address


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes by id, each with what its inputs get, and the graph's outputs.

    A graph declares what it takes and returns, as a function declares its parameters and its return value: a graph
    input is an input of a node whose entry names it (``InputEntry(graph_input=...)``), and a graph output is an output of a
    node, by name. Used as a subgraph, these are the node's inputs and outputs. What a host keeps about the whole
    graph (its name, a viewport) it stores beside the record. The record keeps no order of its nodes: where order
    matters, nodes are taken by id.

    Attributes:
        nodes: Each node, by id.
        outputs: Each graph output, by name, as the output of a node it is: ``{"result": {"node": "join",
            "output": "result"}}``. Empty for a graph that returns nothing.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    nodes: Mapping[Name, NodeRecord] = {}
    outputs: Mapping[Name, Annotated[Address, AfterValidator(_whole_output)]] = {}

    @property
    def required_inputs(self) -> tuple[Name, ...]:
        """The graph inputs a caller must give: each one that no entry naming it gives a default.

        Two entries may name one graph input; a default on either is the graph input's default.
        """
        named = [(entry.graph_input, entry.default is not MISSING) for node in self.nodes.values() for entry in node.inputs.values() if entry.graph_input is not MISSING]
        defaulted = {name for name, has_default in named if has_default}
        return tuple(dict.fromkeys(name for name, _ in named if name not in defaulted))
