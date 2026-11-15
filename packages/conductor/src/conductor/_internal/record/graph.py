"""``GraphRecord``, a whole graph as a host saves it: its nodes by id, and its outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import MISSING, AfterValidator
from pydantic_core import PydanticCustomError

from conductor._internal.base import ConductorModel, Name, Source

from .node import JSON_ONLY, NodeRecord


def _whole_output(source: Source) -> Source:
    """Checks that a graph output is a whole output, not one field of it.

    An edge into a node may take one field of a record, but a graph output can't.

    Raises:
        PydanticCustomError: ``output_has_field`` if ``source`` names a field.
    """
    if source.field is not None:
        raise PydanticCustomError(
            "output_has_field",
            "a graph output can't name a field. Wire the field into a node with an edge, and use that node's output",
        )
    return source


class GraphRecord(ConductorModel):
    """A graph as a host saves it: its nodes by id, and the graph's outputs.

    A graph has inputs and outputs, much like a function has parameters and a return value. A graph input is
    declared by a node's input that names it, as in ``InputEntry(graph_input="text")``. A graph output is one
    node's output, listed under ``outputs``. When the graph is used as a subgraph, these become the subgraph node's
    inputs and outputs.

    Things a host keeps about the whole graph, like its name or a viewport, are saved beside the record, not in it.
    The nodes have no order. Where order matters, they are taken by id.

    Attributes:
        nodes: Each node, by id.
        outputs: Each graph output, by name, with the node output it is, as in ``{"result": {"node": "join",
            "output": "result"}}``. Empty for a graph that returns nothing.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    nodes: Mapping[Name, NodeRecord] = {}
    outputs: Mapping[Name, Annotated[Source, AfterValidator(_whole_output)]] = {}

    @property
    def input_names(self) -> tuple[Name, ...]:
        """The names of the graph inputs, in the order the nodes name them.

        Two inputs may name the same graph input. It is listed once.
        """
        return tuple(
            dict.fromkeys(
                entry.graph_input
                for node in self.nodes.values()
                for entry in node.inputs.values()
                if entry.graph_input is not MISSING
            )
        )

    @property
    def required_inputs(self) -> tuple[Name, ...]:
        """The names of the graph inputs a caller must give, because none of them has a default.

        When two inputs name the same graph input, a default on either one counts for both.
        """
        defaulted = {
            entry.graph_input
            for node in self.nodes.values()
            for entry in node.inputs.values()
            if entry.default is not MISSING
        }
        return tuple(name for name in self.input_names if name not in defaulted)
