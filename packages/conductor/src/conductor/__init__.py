"""Conductor: declare nodes, put them in a graph, compile it, run it.

This package is the main public door. With ``conductor.widgets`` (the
input controls) and ``conductor.events`` (what a run emits) it is
everything a caller outside the library imports; ``__all__`` lists it.
Every other module is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor.codec import to_wire
from conductor.dtype import AsItArrives, DType
from conductor.errors import (
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
from conductor.execution.run_state import RunState
from conductor.field_records import Input, Output, Param, Result
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.conditions import ALWAYS, NEVER, Atom
from conductor.graph.graph_records import FieldContent, Graph, GraphNode
from conductor.graph.run_plan import InputPlan, RunPlan
from conductor.graph.topology import dependencies_of
from conductor.graph.upgrade import upgrade_node
from conductor.interface import FromRun, Interface
from conductor.node_decorator import (
    Arriving,
    Deprecation,
    NodeDefinition,
    NodeDescription,
    OlderVersion,
    Policy,
    Upgrade,
    node,
)
from conductor.ref import Ref
from conductor.registry import NodeRegistry, TypeDescription
from conductor.skip_and_ask import SKIPPED, AskPerson
from conductor.table import ROW_ID, Table

if TYPE_CHECKING:
    # A node takes and returns plain sequences; a type checker reads ``Series[X]`` as one.
    from collections.abc import Sequence as Series
else:
    from conductor.series import Series

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
