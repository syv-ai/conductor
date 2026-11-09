"""Conductor: declare nodes, put them in a graph, compile it, run it.

This package is the main public door. With ``conductor.widgets`` (the
input controls) and ``conductor.events`` (what a run emits) it is
everything a caller outside the library imports; ``__all__`` lists it.
Everything in ``conductor._internal`` is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._internal.codec import to_wire
from conductor._internal.dtype import AsItArrives, DType
from conductor._internal.errors import (
    PROBLEM_CODES,
    CompileProblem,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeExecutionError,
    NodeValidationError,
    Problem,
)
from conductor._internal.execution.run_state import RunState
from conductor._internal.field_records import Input, Output, Param, Result
from conductor._internal.graph.binding import From, Static
from conductor._internal.graph.compiled import CompiledGraph
from conductor._internal.graph.conditions import ALWAYS, NEVER, Atom
from conductor._internal.graph.graph_records import FieldContent, Graph, GraphNode
from conductor._internal.graph.run_plan import InputPlan, RunPlan
from conductor._internal.graph.topology import dependencies_of
from conductor._internal.graph.upgrade import upgrade_node
from conductor._internal.interface import FromRun, Interface
from conductor._internal.node_decorator import (
    Arriving,
    Deprecation,
    NodeDefinition,
    NodeDescription,
    OlderVersion,
    Policy,
    Upgrade,
    node,
)
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry, TypeDescription
from conductor._internal.skip_and_ask import SKIPPED, AskPerson
from conductor._internal.table import ROW_ID

if TYPE_CHECKING:
    # A node takes and returns plain sequences and frames; a type checker reads ``Series[X]`` as one and ``Table[R]`` as the other.
    from collections.abc import Sequence as Series

    from conductor._internal.table_as_a_type_checker_reads_it import Table
else:
    from conductor._internal.series import Series
    from conductor._internal.table import Table

__all__ = [
    "ALWAYS",
    "NEVER",
    "PROBLEM_CODES",
    "ROW_ID",
    "SKIPPED",
    "Arriving",
    "AsItArrives",
    "AskPerson",
    "Atom",
    "CompileProblem",
    "CompiledGraph",
    "DType",
    "Deprecation",
    "ExternalFailure",
    "FieldContent",
    "From",
    "FromRun",
    "Graph",
    "GraphNode",
    "Input",
    "InvalidCallError",
    "InputPlan",
    "Interface",
    "NodeDeclarationError",
    "NodeDefinition",
    "NodeDescription",
    "NodeError",
    "NodeExecutionError",
    "NodeRegistry",
    "NodeValidationError",
    "Output",
    "OlderVersion",
    "Param",
    "Policy",
    "Problem",
    "Ref",
    "Result",
    "RunPlan",
    "RunState",
    "Series",
    "Static",
    "Table",
    "TypeDescription",
    "Upgrade",
    "dependencies_of",
    "node",
    "to_wire",
    "upgrade_node",
]
