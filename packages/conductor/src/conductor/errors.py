"""Exception hierarchy for conductor.

All exceptions carry structured context (node_id, node_type) so errors
propagate upward with enough information for the host app to display
meaningful messages, log to observability tools, or route to error handlers.

Hierarchy:
    ConductorError                     # Base — catch-all for any engine error
    ├── CompilationError                # Graph structure is invalid
    │   ├── CycleDetectionError         # Graph contains a cycle
    │   └── TypeCheckError              # Edge type mismatch (strict mode)
    ├── NodeError                       # Something went wrong with a specific node
    │   ├── NodeValidationError         # Input validation failed (Pydantic)
    │   ├── NodeExecutionError          # Node function raised during execution
    │   ├── NodeTimeoutError            # Node exceeded its timeout
    │   └── NodeConnectionError         # External service / network failure inside a node
    ├── InputResolutionError            # Could not resolve inputs from edges
    ├── FlowExecutionError              # Flow-level failure (used by execute_sync)
    ├── FlowPausedError                 # Flow paused for human input (carries checkpoint)
    └── HumanInputRequired              # Signal raised by nodes to request human input
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Base
# =============================================================================


class ConductorError(Exception):
    """Base exception for all conductor errors."""


# =============================================================================
# Compilation errors
# =============================================================================


class CompilationError(ConductorError):
    """Raised when graph compilation fails (unknown types, invalid edges, etc.)."""


class CycleDetectionError(CompilationError):
    """Raised when a cycle is detected in the graph."""


class TypeCheckError(CompilationError):
    """Raised in strict_types mode when edge types are incompatible."""


# =============================================================================
# Node errors — always carry node_id and node_type for context
# =============================================================================


class NodeError(ConductorError):
    """Base for all node-level errors. Carries node context for upstream reporting.

    Attributes:
        node_id: Instance ID of the node that failed (e.g., "n3").
        node_type: Registry type of the node (e.g., "llm-chat@2").
        original: The original exception that caused this error, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        node_type: str | None = None,
        original: Exception | None = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.original = original
        super().__init__(message)


class NodeValidationError(NodeError):
    """Input validation failed (Pydantic rejected the inputs).

    Not retried — the inputs themselves are wrong, retrying won't help.
    """


class NodeExecutionError(NodeError):
    """The node function raised an exception during execution.

    Retried if retry is configured on the node or globally.
    """


class NodeTimeoutError(NodeError):
    """The node exceeded its execution timeout."""


class NodeConnectionError(NodeError):
    """An external service call inside a node failed (network, API, database).

    Raise this from node functions to distinguish transient failures
    (worth retrying) from logic errors (not worth retrying).

    Example:
        @registry.node("fetch-api", ..., max_retries=3)
        def fetch_api(url):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                raise NodeConnectionError(
                    f"API call failed: {e}",
                    node_id="auto",  # engine fills this in
                ) from e
    """


# =============================================================================
# Input resolution
# =============================================================================


class InputResolutionError(ConductorError):
    """Raised when node input resolution fails (missing source, bad handle)."""

    def __init__(self, message: str, *, node_id: str | None = None):
        self.node_id = node_id
        super().__init__(message)


# =============================================================================
# Flow-level errors
# =============================================================================


class FlowExecutionError(ConductorError):
    """Raised by execute_sync when a flow does not complete successfully.

    Attributes:
        node_id: The node that caused the failure, if applicable.
        node_error: The underlying NodeError, if the failure was node-level.
    """

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        node_error: NodeError | None = None,
    ):
        self.node_id = node_id
        self.node_error = node_error
        super().__init__(message)


# =============================================================================
# Human-in-the-loop
# =============================================================================


class HumanInputRequired(ConductorError):
    """Raised by a node to pause execution and request human input.

    The engine catches this, checkpoints state, and yields a flow_paused event.
    Execution can be resumed later via resume() with the human's response.
    """

    def __init__(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        node_id: str | None = None,
    ):
        self.prompt = prompt
        self.schema = schema
        self.node_id = node_id
        super().__init__(prompt)


class FlowPausedError(ConductorError):
    """Raised by execute_sync when a flow pauses for human input.

    Contains the checkpoint needed to resume execution later.
    """

    def __init__(self, checkpoint: dict[str, Any]):
        self.checkpoint = checkpoint
        super().__init__("Flow paused — human input required")


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# These map old names to new names so existing code doesn't break.
# They'll be removed in a future version.
NodeValidationException = NodeValidationError
NodeExecutionException = NodeExecutionError
FlowExecutionException = FlowExecutionError
FlowPausedException = FlowPausedError
