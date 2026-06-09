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
from conductor.graph.model import (
    Flow,
    FlowDependency,
    FlowTrigger,
    GraphEdge,
    GraphNode,
)
from conductor.node import BaseNode
from conductor.registry import NodeRegistry
from conductor.registry.definition import Actor
from conductor.types import NodeCategory, ResultFormat, WidgetType

__all__ = [
    # Registry + graph
    "NodeRegistry",
    "GraphNode",
    "GraphEdge",
    "Flow",
    "FlowDependency",
    "FlowTrigger",
    "BaseNode",
    "Actor",
    "compile",
    "CompiledGraph",
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
    # Types / enums
    "ResultFormat",
    "NodeCategory",
    "WidgetType",
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
