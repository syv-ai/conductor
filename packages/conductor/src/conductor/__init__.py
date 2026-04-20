"""conductor — reusable graph execution engine.

The top-level package re-exports the surfaces most projects need. Deeper
internals (resolver, state, topology, etc.) stay in submodules.
"""

from conductor import errors, widgets
from conductor._sentinel import SKIPPED
from conductor.compound.for_each import ForEachNode
from conductor.errors import (
    CompilationError,
    ConductorError,
    FlowExecutionError,
    FlowPausedError,
    HumanInputRequired,
    NodeConnectionError,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
)
from conductor.execution.checkpoint import FlowCheckpoint
from conductor.execution.engine import execute, execute_sync, resume_sync
from conductor.execution.retry import RetryConfig
from conductor.execution.store import FlowStore
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.node import BaseNode
from conductor.registry import NodeRegistry
from conductor.types import NodeCategory, ResultFormat, WidgetType

__all__ = [
    # Registry + graph
    "NodeRegistry",
    "GraphNode",
    "GraphEdge",
    "BaseNode",
    "compile",
    # Execution
    "execute",
    "execute_sync",
    "resume_sync",
    "RetryConfig",
    "FlowStore",
    "FlowCheckpoint",
    "SKIPPED",
    # Compound nodes
    "ForEachNode",
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
    # Submodules re-exported for namespace access (`conductor.widgets.Text`, etc.)
    "widgets",
    "errors",
]
