"""Nodes: the building blocks a graph is made of.

A node is a step in a graph. Most nodes are a plain Python function under ``@node``. Its typed parameters are the
node's inputs, and its return type gives its outputs::

    @node("shout", title="Shout", description="Shouts a line.")
    def shout(line: str, mark: str = "!") -> str:
        return line.upper() + mark

A node can also be a whole graph record, used as one step in another graph. That is a subgraph, made with
``subgraph()``.

The package has these parts:

- ``declare.py`` has ``@node`` and ``subgraph()``, which make a node definition.
- ``type.py`` has what they make, ``FunctionDefinition`` and ``SubgraphDefinition``, and ``Upgrade``, which moves a
  graph from one version of a node to the next.
- ``hook.py`` has ``Hook``, a function that works out some of a node's inputs or outputs when a graph is compiled.
- ``source.py`` reads the function's source once: its code as text, for the code fingerprint, and whether it
  raises ``Ask``.
- ``policy.py`` has ``Policy``, how the engine calls a node, and ``Deprecation``, a notice that it is going away.
- ``running_calls.py`` has ``RunningCalls``, which counts how many calls of a node are running at once.
- ``describe.py`` has ``NodeDescription``, a node as an editor's palette shows it.
- ``validator.py`` checks what a developer gives ``@node``, ``subgraph`` and ``NodeRegistry`` (``DeclarationValidator``),
  and what a node's code returns: ``ReturnValidator`` for the function, ``HookValidator`` for
  its hooks, and ``RewriteValidator`` for the rewrite of an ``Upgrade``.
"""

from .declare import node, subgraph
from .describe import NodeDescription
from .hook import Hook
from .policy import LONGEST_WAIT, Deprecation, Policy
from .running_calls import RunningCalls
from .type import (
    FunctionDefinition,
    NodeDefinition,
    SubgraphDefinition,
    Upgrade,
    gone_inputs,
    with_defaults,
)
from .validator import DeclarationValidator, HookValidator, ReturnValidator, RewriteValidator

__all__ = [
    "LONGEST_WAIT",
    "DeclarationValidator",
    "Deprecation",
    "FunctionDefinition",
    "Hook",
    "HookValidator",
    "NodeDefinition",
    "NodeDescription",
    "Policy",
    "ReturnValidator",
    "RewriteValidator",
    "RunningCalls",
    "SubgraphDefinition",
    "Upgrade",
    "gone_inputs",
    "node",
    "subgraph",
    "with_defaults",
]
