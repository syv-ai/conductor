"""Exception hierarchy for flowengine."""


class FlowEngineError(Exception):
    """Base exception for all flowengine errors."""


class CompilationError(FlowEngineError):
    """Raised when graph compilation fails (unknown types, invalid edges, etc.)."""


class CycleDetectionError(CompilationError):
    """Raised when a cycle is detected in the graph."""


class InputResolutionError(FlowEngineError):
    """Raised when node input resolution fails."""

    def __init__(self, message: str, *, node_id: str | None = None):
        self.node_id = node_id
        super().__init__(message)


class NodeValidationException(FlowEngineError):
    """Raised when node input validation fails (Pydantic)."""


class NodeExecutionException(FlowEngineError):
    """Raised when node execution fails."""


class FlowExecutionException(FlowEngineError):
    """Raised by execute_sync when a flow does not complete successfully."""


class HumanInputRequired(FlowEngineError):
    """Raised by a node to pause execution and request human input.

    The engine catches this, checkpoints state, and yields a flow_paused event.
    Execution can be resumed later via resume() with the human's response.

    Args:
        prompt: Human-readable description of what input is needed.
        schema: Optional dict describing the expected response shape
                (e.g., {"approved": "bool", "comment": "str"}).
        node_id: Set automatically by the engine — do not pass this.
    """

    def __init__(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        node_id: str | None = None,
    ):
        self.prompt = prompt
        self.schema = schema
        self.node_id = node_id
        super().__init__(prompt)


class FlowPausedException(FlowEngineError):
    """Raised by execute_sync when a flow pauses for human input.

    Contains the checkpoint needed to resume execution later.
    """

    def __init__(self, checkpoint: dict):
        self.checkpoint = checkpoint
        super().__init__("Flow paused — human input required")
