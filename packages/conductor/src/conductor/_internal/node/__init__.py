"""Node definitions: a function, or a graph record, made a node.

``declare.py`` is ``@node`` and ``subgraph()``, which read a function's signature or take a graph record; ``type.py``
is what they make, ``FunctionDefinition`` and ``SubgraphDefinition``; ``policy.py`` is ``Policy`` and
``Deprecation``; ``describe.py`` is a node as an editor's palette shows it.
"""

from conductor._internal.node.declare import node, subgraph
from conductor._internal.node.describe import (
    FunctionNodeDescription,
    NodeDescription,
    SubgraphDescription,
)
from conductor._internal.node.policy import Deprecation, Policy
from conductor._internal.node.type import (
    AskedParameter,
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    Upgrade,
    declared_inputs,
    read_by_hooks,
)
