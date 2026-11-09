"""Conductor: declare nodes, put them in a graph, compile it, run it.

This package is the main public door. With ``conductor.widgets`` (the
input controls) and ``conductor.events`` (what a run emits) it is
everything a caller outside the library imports; ``__all__`` lists it.
Every other module is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._sentinel import SKIPPED, AskPerson, is_skipped
from conductor.codec import from_wire, to_wire
from conductor.dtype import AsItArrives, DType
from conductor.errors import (
    PROBLEM_CODES,
    CompileProblem,
    ConductorError,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    Problem,
)
from conductor.execution.run_state import (
    Answers,
    OutputColumn,
    Question,
    RunState,
    RunStatus,
    WaitingUnit,
)
from conductor.graph.binding import Binding, From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.compiled_node import CompiledField, CompiledNode
from conductor.graph.conditions import ALWAYS, NEVER, Atom, Condition
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.graph.topology import dependencies_of
from conductor.graph.upgrade import upgrade_node
from conductor.interface import FromRun, Interface, model_of
from conductor.metadata import Input, Output, Param, Result
from conductor.node_decorator import (
    Arriving,
    Deprecation,
    NodeDefinition,
    NodeDescription,
    NodeVersion,
    OlderVersion,
    Policy,
    Upgrade,
    VersionDescription,
    node,
)
from conductor.ref import Ref
from conductor.registry import NodeRegistry, RegistryDescription, TypeDescription
from conductor.series import Index
from conductor.table import Table

if TYPE_CHECKING:
    # A node takes and returns plain sequences; a type checker reads ``Series[X]`` as one.
    from collections.abc import Sequence as Series
else:
    from conductor.series import Series

__all__ = [
    "ALWAYS",
    "NEVER",
    "PROBLEM_CODES",
    "SKIPPED",
    "Answers",
    "Arriving",
    "AsItArrives",
    "AskPerson",
    "Atom",
    "Binding",
    "CompileProblem",
    "CompiledField",
    "CompiledGraph",
    "CompiledNode",
    "Condition",
    "ConductorError",
    "DType",
    "Deprecation",
    "ExternalFailure",
    "FieldContent",
    "From",
    "FromRun",
    "Graph",
    "GraphNode",
    "Index",
    "Input",
    "InvalidCallError",
    "Interface",
    "NodeDeclarationError",
    "NodeDefinition",
    "NodeDescription",
    "NodeError",
    "NodeExecutionError",
    "NodeRegistry",
    "NodeTimeoutError",
    "NodeValidationError",
    "NodeVersion",
    "Output",
    "OlderVersion",
    "OutputColumn",
    "Param",
    "Policy",
    "Problem",
    "Question",
    "Ref",
    "RegistryDescription",
    "Result",
    "RunState",
    "RunStatus",
    "Series",
    "Static",
    "Table",
    "TypeDescription",
    "Upgrade",
    "VersionDescription",
    "WaitingUnit",
    "dependencies_of",
    "from_wire",
    "is_skipped",
    "model_of",
    "node",
    "to_wire",
    "upgrade_node",
]
