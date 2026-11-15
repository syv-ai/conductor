"""Node definitions: a function, or a graph record, made a node.

``declare.py`` is ``@node`` and ``subgraph()``, which read a function's signature or take a graph record; ``type.py``
is what they make, ``FunctionDefinition`` and ``SubgraphDefinition``; ``policy.py`` is ``Policy`` and
``Deprecation``; ``running_calls.py`` is ``RunningCalls``, how many calls of a node are running in the process;
``describe.py`` is a node as an editor's palette shows it; ``validator.py`` reads what a node's code returns: what its
function returns (``ReturnValidator``) and what its hooks return (``HookValidator``).
"""

from .declare import node, subgraph
from .describe import NodeDescription
from .policy import Deprecation, Policy
from .running_calls import RunningCalls
from .type import (
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    Upgrade,
    gone_inputs,
    with_defaults,
)
from .validator import HookValidator, ReturnValidator

__all__ = [
    "Deprecation",
    "FunctionDefinition",
    "HookName",
    "HookValidator",
    "NodeDefinition",
    "NodeDescription",
    "Policy",
    "ReturnValidator",
    "RunningCalls",
    "SubgraphDefinition",
    "Upgrade",
    "gone_inputs",
    "node",
    "subgraph",
    "with_defaults",
]
