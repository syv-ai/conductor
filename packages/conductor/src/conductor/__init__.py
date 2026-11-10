"""Conductor: typed Python functions as nodes, graphs of them as data, runs as JSON.

A node author writes ``@node`` functions with ``Param``, ``Series``, ``Table``,
``SKIPPED``, ``Ask`` and ``Choice``. A host collects them in a ``NodeRegistry``, loads a
``GraphRecord`` into a ``Graph``, and runs it: a ``Run`` pauses for a person,
takes answers, resumes and goes again, and is JSON throughout.

This module, ``conductor.widgets`` (the input controls), ``conductor.events``
(what a run says as it goes) and ``conductor.testing`` (one test for a whole
registry) are everything a caller outside the library imports; each lists its
names in ``__all__``.
Everything in ``conductor._internal`` is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._internal.ask import Ask, Choice, Waiting
from conductor._internal.errors import (
    PROBLEM_CODES,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
)
from conductor._internal.fields import SKIPPED, FromRun, Param, Series
from conductor._internal.graph import Graph
from conductor._internal.node_definition import Deprecation, NodeDefinition, Policy, Upgrade, node
from conductor._internal.record import GraphRecord, NodeRecord, make_graph
from conductor._internal.registry import NodeRegistry
from conductor._internal.run import Run

if TYPE_CHECKING:
    # A node takes and returns frames; a type checker reads ``Table[R]`` as one that keeps its row type ``R``.
    from conductor._internal.table_as_a_type_checker_reads_it import Table
else:
    from conductor._internal.fields import Table

__all__ = [
    "PROBLEM_CODES",
    "SKIPPED",
    "Ask",
    "Choice",
    "Deprecation",
    "ExternalFailure",
    "FromRun",
    "Graph",
    "GraphRecord",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeDefinition",
    "NodeError",
    "NodeRecord",
    "NodeRegistry",
    "Param",
    "Policy",
    "Problem",
    "Run",
    "Series",
    "Table",
    "Upgrade",
    "Waiting",
    "make_graph",
    "node",
]
