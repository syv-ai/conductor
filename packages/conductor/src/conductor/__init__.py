"""conductor — reusable graph execution engine.

The top-level package re-exports the surfaces most projects need. Deeper
internals (resolver, state, topology, etc.) stay in submodules.
"""

from conductor import errors, expr, widgets
from conductor._sentinel import SKIPPED
from conductor.compound import (
    FOR_EACH,
    SUBPROCESS,
    WHILE,
    ForEachNode,
    SubprocessNode,
    WhileNode,
    compute_for_each_end_outputs,
)
from conductor.compound.subprocess import SubprocessRegistry
from conductor.dtype import DType, Single, dtype_of, registered_dtypes
from conductor.dtype_ref import DTypeRef
from conductor.errors import (
    CompilationError,
    ConductorError,
    FlowExecutionError,
    FlowPausedError,
    HumanInputRequired,
    LoopRunawayError,
    NodeConnectionError,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    SignalRequired,
    SubprocessFailedError,
)
from conductor.execution.checkpoint import FlowCheckpoint
from conductor.execution.engine import execute, execute_sync, resume, resume_sync
from conductor.execution.retry import RetryConfig
from conductor.execution.store import FlowStore
from conductor.graph.compiler import CompiledGraph, compile
from conductor.graph.dynamic_inputs import resolve_graph_inputs
from conductor.graph.dynamic_outputs import resolve_graph_outputs
from conductor.graph.model import (
    Flow,
    FlowDependency,
    FlowTrigger,
    GraphEdge,
    GraphNode,
)
from conductor.interface import Interface, Provided
from conductor.metadata import Input, Output
from conductor.node import (
    Deprecation,
    GraphVersion,
    NodeDefinition,
    NodeDescription,
    NodeVersion,
    Policy,
    deprecated,
    upgrade,
    version,
)
from conductor.ref import Ref
from conductor.registry import NodeRegistry, runner_for
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import AnyWidget

__all__ = [
    # Registry + graph
    "NodeRegistry",
    "runner_for",
    "GraphNode",
    "GraphEdge",
    "Flow",
    "FlowDependency",
    "FlowTrigger",
    "compile",
    "CompiledGraph",
    "resolve_graph_inputs",
    "resolve_graph_outputs",
    # Execution
    "execute",
    "execute_sync",
    "resume",
    "resume_sync",
    "RetryConfig",
    "FlowStore",
    "FlowCheckpoint",
    "SKIPPED",
    # Compound nodes
    "ForEachNode",
    "FOR_EACH",
    "compute_for_each_end_outputs",
    "WhileNode",
    "WHILE",
    "SubprocessNode",
    "SUBPROCESS",
    "SubprocessRegistry",
    # The node contract
    "NodeDefinition",
    "NodeVersion",
    "GraphVersion",
    "Policy",
    "Deprecation",
    "NodeDescription",
    "Interface",
    "Provided",
    "Input",
    "Output",
    "AnyWidget",
    "version",
    "upgrade",
    "deprecated",
    # The type vocabulary
    "DType",
    "DTypeRef",
    "Single",
    "dtype_of",
    "registered_dtypes",
    "Ref",
    "Result",
    "Index",
    "Series",
    # Types / enums
    # Errors (most commonly raised from node code)
    "ConductorError",
    "CompilationError",
    "NodeError",
    "NodeValidationError",
    "NodeExecutionError",
    "NodeConnectionError",
    "NodeTimeoutError",
    "FlowExecutionError",
    "HumanInputRequired",
    "FlowPausedError",
    "SignalRequired",
    "LoopRunawayError",
    "SubprocessFailedError",
    # Submodules re-exported for namespace access (`conductor.widgets.Text`, etc.)
    "widgets",
    "errors",
    "expr",
]
