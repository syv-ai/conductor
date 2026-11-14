"""A graph: a graph record compiled against the node definitions it uses, which a host holds and starts runs of.

Code outside this package imports from here only. ``type.py`` has ``Graph``, and ``Graph.compile`` compiles a
record with ``compile.py``; ``node.py`` has its nodes; ``input.py`` how each input of a node gets its value;
``describe.py`` what a host is shown of a graph; ``make.py`` is ``make_graph``, a graph record written from Python.
The record a graph is compiled from is ``record/``.
"""

from .describe import (
    GraphDescription,
    InputSource,
)
from .input import (
    CollectingInput,
    ConstantInput,
    Edge,
    EdgeInput,
    GraphInput,
    Input,
    MissingInput,
)
from .make import make_graph
from .node import (
    CompiledFunctionNode,
    CompiledSubgraphNode,
    outputs_when_not_called,
    outputs_when_skipped,
)
from .type import Graph

__all__ = [
    "CollectingInput",
    "CompiledFunctionNode",
    "CompiledSubgraphNode",
    "ConstantInput",
    "Edge",
    "EdgeInput",
    "Graph",
    "GraphDescription",
    "GraphInput",
    "Input",
    "InputSource",
    "MissingInput",
    "make_graph",
    "outputs_when_not_called",
    "outputs_when_skipped",
]
