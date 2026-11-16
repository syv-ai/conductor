"""Compiled graphs: a graph record checked against the nodes it uses, and ready to run.

A host stores a graph as a ``GraphRecord``. That's plain data: which nodes there are, and how they're wired.
Before it can run, the record is compiled against a registry of node definitions. Compiling checks every node and
edge. It also works out how the graph runs: where each input gets its value, and which nodes are called once per
item of a list. The result is a ``Graph``, which a host keeps and starts runs from.

The files:

- ``type.py``: ``Graph`` itself.
- ``compile.py``: the compiler behind ``Graph.compile``.
- ``node.py`` and ``input.py``: the compiled nodes, and where each of their inputs gets its value.
- ``describe.py``: what ``graph.describe()`` shows a host.
- ``make.py``: ``make_graph``, which writes a graph record from Python.
- ``validator.py``: the checks on what a graph is given, such as a caller's inputs and a run's limits.

Code outside this package imports from here only. The graph record itself is in ``record/``.
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
