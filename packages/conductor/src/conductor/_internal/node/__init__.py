"""Node definitions: a function, or a graph record, made a node.

``declare.py`` is ``@node`` and ``subgraph()``, which read a function's signature or take a graph record; ``type.py``
is what they make, ``FunctionDefinition`` and ``SubgraphDefinition``; ``policy.py`` is ``Policy`` and
``Deprecation``; ``describe.py`` is a node as an editor's palette shows it.
"""

from .declare import node, subgraph
from .describe import (
    FunctionNodeDescription,
    NodeDescription,
    SubgraphDescription,
)
from .policy import Deprecation, Policy
from .type import (
    AskedParameter,
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    Upgrade,
    declared_inputs,
    read_by_hooks,
)
