"""``GraphRecord``, a whole graph as a host saves it: its graph inputs, its nodes by id, and its graph outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import MISSING, AfterValidator, JsonValue

from conductor._internal.base import ConductorModel, Name, Source
from conductor._internal.problem import conductor_error

from .node import CARRIED, JSON_ONLY, NodeRecord


def _whole_output(source: Source) -> Source:
    """Checks that a graph output is a whole output, not one field of it.

    An edge into a node may take one field of an output, but a graph output can't.

    Raises:
        PydanticCustomError: ``output_has_field`` if ``source`` names a field.
    """
    if source.field is not None:
        raise conductor_error(
            "output_has_field",
            "a graph output names a field, and a graph output can only be a whole output",
        )
    return source


class GraphInputRecord(ConductorModel):
    """One graph input a graph record declares: ``{}``, or ``{"default": "a"}`` for one with a default.

    A caller gives a graph input its value when a run starts. With a ``default``, the caller may leave it out.
    Without one, the caller must give it, whatever the defaults of the inputs it feeds.

    Attributes:
        default: The value the graph input has when the caller gives none.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    default: Annotated[JsonValue, CARRIED] | MISSING = MISSING


class GraphRecord(ConductorModel):
    """A graph as a host saves it: the graph's inputs, its nodes by id, and the graph's outputs.

    A graph has inputs and outputs, much like a function has parameters and a return value. A graph input is
    declared under ``inputs``, and a node's input takes its value by naming it, as in
    ``GraphInputEntry(graph_input="text")``. A graph output is one node's output, listed under ``outputs``. When the
    graph is used as a subgraph, these become the subgraph node's inputs and outputs.

    Things a host keeps about the whole graph, like its name or a viewport, are saved beside the record, not in it.
    The nodes have no order. Where order matters, they are taken by id.

    Attributes:
        inputs: Each graph input, by name, as in ``{"text": {}}``, or ``{"mark": {"default": "!"}}`` for one with a
            default. Empty for a graph that takes nothing.
        nodes: Each node, by id.
        outputs: Each graph output, by name, with the node output it is, as in ``{"result": {"node": "join",
            "output": "result"}}``. Empty for a graph that returns nothing.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    inputs: Mapping[Name, GraphInputRecord] = {}
    nodes: Mapping[Name, NodeRecord] = {}
    outputs: Mapping[Name, Annotated[Source, AfterValidator(_whole_output)]] = {}
