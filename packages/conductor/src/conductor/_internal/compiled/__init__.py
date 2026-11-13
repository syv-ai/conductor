"""A graph record compiled against the node definitions it uses: compiling, what compiling makes, and what a host is shown.

Code outside this package imports from here only. ``compile.py`` has ``compile_graph``; ``type.py`` has
``CompiledGraph``; ``node.py`` its nodes; ``input.py`` how each input of a node gets its value;
``describe.py`` what a host is shown of a compiled graph.
"""

from .compile import compile_graph
from .describe import (
    GraphDescription,
    InputSource,
    RunsOn,
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
from .node import (
    CompiledFunctionNode,
    CompiledSubgraphNode,
    outputs_when_not_called,
    outputs_when_skipped,
)
from .type import CompiledGraph
