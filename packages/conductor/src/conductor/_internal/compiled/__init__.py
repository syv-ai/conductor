"""A graph record compiled against the node definitions it uses: compiling, what compiling makes, and what a host is shown.

Code outside this package imports from here only. ``compile.py`` has ``compile_graph``; ``type.py`` has
``CompiledGraph``; ``node.py`` its nodes; ``input.py`` how each input of a node gets its value;
``flat.py`` the flat graph, which a run works on; ``describe.py`` what a host is shown of a compiled graph.
"""

from .compile import compile_graph
from .describe import (
    GraphDescription,
    GraphInputDescription,
    InputSource,
    RunsOn,
)
from .flat import (
    FlatFunctionNode,
    FlatNode,
    FlatSubgraph,
    all_upstream,
    display_inputs_of,
    outcome_loc,
)
from .input import (
    CollectingInput,
    ConstantInput,
    Edge,
    EdgeInput,
    GraphInput,
)
from .node import (
    CompiledFunctionNode,
    outputs_when_not_called,
    outputs_when_skipped,
)
from .type import CompiledGraph
