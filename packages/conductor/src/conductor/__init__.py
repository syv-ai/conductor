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
from conductor._internal.errors import (
    PROBLEM_CODES,
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
)
from conductor._internal.fields import (
    SKIPPED,
    Choice,
    FieldDescription,
    FromRun,
    NamedType,
    Param,
    Series,
    Table,
)
from conductor._internal.graph import Feed, Graph, GraphDescription, GraphInput, RunsOn
from conductor._internal.node_definition import (
    Deprecation,
    FunctionDefinition,
    FunctionNodeDescription,
    GraphDefinition,
    NodeDefinition,
    NodeDescription,
    Policy,
    StoredGraphDescription,
    Upgrade,
    node,
)
from conductor._internal.record import GraphRecord, NodeRecord, make_graph
from conductor._internal.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor._internal.run import EndedEarly, EndedStatus, Going, NotCalled, Run, RunState, RunStatus

# The reference lists these names in this order, so they are kept by subject.
__all__ = [
    # Writing a node
    "node",
    "NodeDefinition",
    "FunctionDefinition",
    "GraphDefinition",
    "Upgrade",
    "Policy",
    "Deprecation",
    "Param",
    "FromRun",
    "Series",
    "Table",
    "NamedType",
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
    "RunState",
    "Going",
    "EndedEarly",
    "EndedStatus",
    "RunStatus",
    # What an editor reads
    "FieldDescription",
    "NodeDescription",
    "FunctionNodeDescription",
    "StoredGraphDescription",
    "RegistryDescription",
    "TypeDescription",
    "GraphDescription",
    "GraphInput",
    "RunsOn",
    "Feed",
    # Errors
    "NodeDeclarationError",
    "InvalidCallError",
    "Problem",
    "PROBLEM_CODES",
]
