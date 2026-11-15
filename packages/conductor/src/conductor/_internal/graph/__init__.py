"""A graph: a graph record compiled against the node definitions it uses, which a host holds and starts runs of.

Code outside this package imports from here only. ``type.py`` has ``Graph``, and ``Graph.compile`` compiles a
record with ``compile.py``; ``node.py`` has its nodes; ``input.py`` how each input of a node gets its value;
``describe.py`` what a host is shown of a graph; ``make.py`` is ``make_graph``, a graph record written from Python;
``validator.py`` reads what a graph is given: a caller's graph inputs and a run's limits, and the constants of its
record.
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
)
from .type import Graph
from .validator import LimitsValidator

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
    "LimitsValidator",
    "MissingInput",
    "make_graph",
]
