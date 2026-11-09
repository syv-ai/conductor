"""Conductor: declare nodes, put them in a graph, compile it, run it.

This package is the main public door. With ``conductor.widgets`` (the
input controls) and ``conductor.events`` (what a run emits) it is
everything a caller outside the library imports; ``__all__`` lists it.
Everything in ``conductor._internal`` is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._internal.codec import to_jsonable
from conductor._internal.dtype import AsItArrives, DType, Series
from conductor._internal.errors import (
    PROBLEM_CODES,
    ComputeHookError,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
)
from conductor._internal.execution.run_state import (
    Answers,
    EarlierUnit,
    OutputColumn,
    Question,
    RunState,
    RunStatus,
    WaitingUnit,
    WorkOfNode,
)
from conductor._internal.field_records import Input, Output, Param, Result
from conductor._internal.graph.binding import OPEN, Binding, From, Static
from conductor._internal.graph.compiled_graph import CompiledGraph
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode, NodeStatus
from conductor._internal.graph.graph_description import (
    CompiledNodeDescription,
    FieldDescription,
    GraphDescription,
)
from conductor._internal.graph.graph_records import FieldContent, Graph, GraphNode
from conductor._internal.graph.output_conditions import ALWAYS, NEVER, BranchTaken, Condition
from conductor._internal.graph.receive_modes import ReceiveMode
from conductor._internal.graph.run_plan import NodeRunPlan, RunPlan
from conductor._internal.graph.topology import dependencies_of
from conductor._internal.graph.upgrade import upgrade_node
from conductor._internal.index import ValuesOnRows
from conductor._internal.node_definition import (
    ArrivingType,
    Deprecation,
    NodeDefinition,
    NodeDescription,
    NodeVersion,
    Policy,
    Upgrade,
    VersionDescription,
    node,
)
from conductor._internal.node_signature import FromRun, Interface
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor._internal.skip_and_ask import SKIPPED, AskPerson
from conductor._internal.table import ROW_ID_COLUMN, table_with_columns

if TYPE_CHECKING:
    # A node takes and returns frames; a type checker reads ``Table[R]`` as one.
    from conductor._internal.table_as_a_type_checker_reads_it import Table
else:
    from conductor._internal.table import Table

__all__ = [
    "ALWAYS",
    "NEVER",
    "OPEN",
    "PROBLEM_CODES",
    "ROW_ID_COLUMN",
    "SKIPPED",
    "Answers",
    "ArrivingType",
    "AsItArrives",
    "AskPerson",
    "Binding",
    "BranchTaken",
    "CompiledField",
    "CompiledGraph",
    "CompiledNode",
    "CompiledNodeDescription",
    "ComputeHookError",
    "Condition",
    "DType",
    "Deprecation",
    "ExternalFailure",
    "FieldContent",
    "FieldDescription",
    "From",
    "FromRun",
    "Graph",
    "GraphDescription",
    "GraphNode",
    "Input",
    "Interface",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeDefinition",
    "NodeDescription",
    "NodeError",
    "NodeRegistry",
    "NodeRunPlan",
    "NodeStatus",
    "NodeVersion",
    "Output",
    "EarlierUnit",
    "OutputColumn",
    "WorkOfNode",
    "Param",
    "Policy",
    "Problem",
    "Question",
    "ReceiveMode",
    "Ref",
    "RegistryDescription",
    "Result",
    "RunPlan",
    "RunState",
    "RunStatus",
    "Series",
    "Static",
    "Table",
    "TypeDescription",
    "Upgrade",
    "ValuesOnRows",
    "VersionDescription",
    "WaitingUnit",
    "dependencies_of",
    "node",
    "table_with_columns",
    "to_jsonable",
    "upgrade_node",
]
