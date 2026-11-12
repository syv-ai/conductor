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

from conductor._internal.ask import Ask, Waiting
from conductor._internal.compiled import GraphDescription, GraphInput, InputSource, RunsOn
from conductor._internal.context import Context
from conductor._internal.field import FieldDescription
from conductor._internal.graph import Graph
from conductor._internal.graph_record import GraphRecord, NodeRecord
from conductor._internal.make_graph import make_graph
from conductor._internal.markers import SKIPPED
from conductor._internal.node_definition import (
    Deprecation,
    FunctionDefinition,
    FunctionNodeDescription,
    NodeDefinition,
    NodeDescription,
    Policy,
    SubgraphDefinition,
    SubgraphDescription,
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
from conductor._internal.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor._internal.run import CallCount, NotCalled, Progress, Run
from conductor._internal.run.state import EndedEarly, Going, RunState, RunStatus
from conductor._internal.series import Series
from conductor._internal.table import Table
from conductor._internal.value_type import Choice, Param, host_type

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
    "make_graph",
    "Graph",
    # A run
    "Run",
    "Waiting",
    "NotCalled",
    "Progress",
    "CallCount",
    "RunState",
    "Going",
    "EndedEarly",
    "RunStatus",
    # What an editor reads
    "FieldDescription",
    "NodeDescription",
    "FunctionNodeDescription",
    "SubgraphDescription",
    "RegistryDescription",
    "TypeDescription",
    "GraphDescription",
    "GraphInput",
    "RunsOn",
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
