"""Conductor: typed Python functions as nodes, graphs of them as data, runs as JSON.

A node author writes ``@node`` functions with ``Param``, ``Series``, ``Table``,
``SKIPPED``, ``Ask`` and ``Choice``. A host collects them in a ``NodeRegistry``, loads a
``GraphRecord`` into a ``Graph``, and runs it: a ``Run`` pauses for a person,
takes answers, resumes and goes again, and is JSON throughout.

This module, ``conductor.widgets`` (the input controls), ``conductor.events``
(what a run says as it goes), ``conductor.testing`` (one test for a whole
registry) and ``conductor.about`` (this reference, ``python -m conductor.about``)
are everything a caller outside the library imports; each lists its names in
``__all__``.
Everything in ``conductor._internal`` is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._internal.ask import Ask
from conductor._internal.base import SKIPPED, Source
from conductor._internal.context import Context
from conductor._internal.field import FieldDescription
from conductor._internal.gate import read_json
from conductor._internal.graph import (
    Graph,
    GraphDescription,
    InputSource,
    make_graph,
)
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
from conductor._internal.record import GraphRecord, InputEntry, NodeRecord
from conductor._internal.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor._internal.run import (
    CallCount,
    NotCalled,
    Run,
    RunProgress,
    RunStatus,
    Waiting,
)
from conductor._internal.series import Series
from conductor._internal.table import Table
from conductor._internal.types import Choice, Param, host_type

# The reference lists these names in this order, so they are kept by subject.
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
    "NodeRecord",
    "InputEntry",
    "Source",
    "make_graph",
    "Graph",
    "read_json",
    # A run
    "Run",
    "Waiting",
    "NotCalled",
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
