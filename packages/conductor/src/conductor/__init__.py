"""Conductor runs graphs of typed Python functions.

A node is a plain Python function under ``@node``. Its type hints say what it takes and what it returns. A graph is
data: a ``GraphRecord`` lists the nodes it uses and which outputs feed which inputs. A run is data too, so it can be
saved as JSON and picked up again later.

A node author writes nodes with ``@node``, and uses ``Param``, ``Series``, ``Table``, ``SKIPPED``, ``Ask`` and
``Choice`` inside them. A host puts the nodes in a ``NodeRegistry``, compiles a record into a ``Graph`` and runs it.
A ``Run`` can stop to wait for a person, take their answer and carry on.

Code outside conductor imports from five modules, and each lists its names in ``__all__``:

- ``conductor``, this module.
- ``conductor.widgets``, the controls an editor shows for an input.
- ``conductor.events``, what a run reports as it goes.
- ``conductor.testing``, one test for a whole registry.
- ``conductor.about``, the reference, also printed by ``python -m conductor.about``.

Everything in ``conductor._internal`` is private and may change.
"""

from typing import TYPE_CHECKING

from conductor._internal.ask import Ask
from conductor._internal.base import JSON_CONFIG, SKIPPED, Source
from conductor._internal.context import Context
from conductor._internal.field import FieldDescription
from conductor._internal.graph import (
    Graph,
    GraphDescription,
    InputSource,
)
from conductor._internal.make_graph import make_graph
from conductor._internal.node import (
    Deprecation,
    FunctionDefinition,
    NodeDefinition,
    NodeDescription,
    Policy,
    SubgraphDefinition,
    Upgrade,
    node,
    subgraph,
)
from conductor._internal.problem import (
    DECLARATION_PROBLEMS,
    PROBLEMS,
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
    ProblemCatalogue,
    ProblemKind,
)
from conductor._internal.record import (
    GraphInputRecord,
    GraphRecord,
    InputEntry,
    NodeRecord,
    read_json,
)
from conductor._internal.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor._internal.run import (
    CallCount,
    NoneInputCall,
    Run,
    RunProgress,
    RunStatus,
    Waiting,
)
from conductor._internal.series import Series
from conductor._internal.table import Table
from conductor._internal.types import Choice, Param, host_type, type_fields

# The reference (``python -m conductor.about``) lists the names in this order, so they are grouped by subject.
__all__ = [
    # Writing a node
    "node",
    "subgraph",
    "NodeDefinition",
    "FunctionDefinition",
    "SubgraphDefinition",
    "Upgrade",
    "Policy",
    "Deprecation",
    "Param",
    "Context",
    "Series",
    "Table",
    "host_type",
    "type_fields",
    # Inside a node
    "SKIPPED",
    "Ask",
    "Choice",
    "NodeError",
    "ExternalError",
    # The registry
    "NodeRegistry",
    # A graph
    "GraphRecord",
    "GraphInputRecord",
    "NodeRecord",
    "InputEntry",
    "Source",
    "JSON_CONFIG",
    "make_graph",
    "Graph",
    "read_json",
    # A run
    "Run",
    "Waiting",
    "NoneInputCall",
    "RunProgress",
    "CallCount",
    "RunStatus",
    # What an editor reads
    "FieldDescription",
    "NodeDescription",
    "RegistryDescription",
    "TypeDescription",
    "GraphDescription",
    "InputSource",
    # Errors
    "NodeDeclarationError",
    "InvalidCallError",
    "Problem",
    "PROBLEMS",
    "DECLARATION_PROBLEMS",
    "ProblemCatalogue",
    "ProblemKind",
]
