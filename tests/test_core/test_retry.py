"""Retry-classification tests.

The engine's retry loop must respect the per-error retryable/fatal
classification declared in :mod:`conductor.errors`. Pydantic-validation
failures are already covered in ``test_eager_and_retry.py``; this
module exercises the path where a node *itself* raises
:class:`~conductor.errors.NodeValidationError` (or another error with
``retryable=False``) — the engine must not retry, even when the node
declares ``max_retries`` greater than zero.
"""

from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import (
    FlowExecutionError,
    NodeExecutionError,
    NodeValidationError,
)
from conductor.execution.engine import execute_sync
from conductor.execution.retry import RetryConfig
from conductor.widgets import Output, Text


def test_validation_errors_not_retried():
    """A node that raises ``NodeValidationError`` runs exactly once,
    even when ``max_retries`` is generous and a global RetryConfig is
    configured. Validation errors are fatal — retrying just delays the
    failure and burns latency budget.
    """
    reg = NodeRegistry()
    call_count = 0

    @reg.node(
        "raise-validation", version=1, name="Raise Validation",
        description="Always raises NodeValidationError",
        max_retries=3, retry_delay=0.01,
    )
    def raise_validation(
        text: Annotated[str, Text(label="In")] = "",
    ) -> Annotated[str, Output(label="Out")]:
        nonlocal call_count
        call_count += 1
        raise NodeValidationError(
            "intentionally invalid input",
            node_id="n1",
            node_type="raise-validation@1",
        )

    compiled = compile(
        nodes=[GraphNode("n1", "raise-validation@1", {"text": "x"})],
        edges=[],
        registry=reg,
    )

    with pytest.raises(FlowExecutionError):
        execute_sync(compiled, retry=RetryConfig(max_retries=3, delay=0.01))

    # Exactly one invocation — no retries.
    assert call_count == 1, (
        f"expected NodeValidationError to be fatal (1 call), got {call_count}"
    )


def test_node_execution_error_with_retryable_false_not_retried():
    """An instance-level ``retryable=False`` override on
    :class:`NodeExecutionError` should also short-circuit the retry loop.
    """
    reg = NodeRegistry()
    call_count = 0

    class FatalExecutionError(NodeExecutionError):
        retryable = False

    @reg.node(
        "raise-fatal", version=1, name="Raise Fatal",
        description="Raises a fatal NodeExecutionError subclass",
        max_retries=5, retry_delay=0.01,
    )
    def raise_fatal(
        text: Annotated[str, Text(label="In")] = "",
    ) -> Annotated[str, Output(label="Out")]:
        nonlocal call_count
        call_count += 1
        raise FatalExecutionError(
            "fatal — do not retry",
            node_id="n1", node_type="raise-fatal@1",
        )

    compiled = compile(
        nodes=[GraphNode("n1", "raise-fatal@1", {"text": "x"})],
        edges=[],
        registry=reg,
    )

    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)

    assert call_count == 1, (
        f"expected fatal NodeExecutionError to skip retries (1 call), "
        f"got {call_count}"
    )


def test_default_node_execution_error_still_retries():
    """Sanity check: a vanilla ``NodeExecutionError`` (default
    ``retryable=True``) still triggers the retry loop the same way it
    used to. Guard against the new classification accidentally turning
    every error into a fatal one.
    """
    reg = NodeRegistry()
    call_count = 0

    @reg.node(
        "flaky-exec", version=1, name="Flaky Exec",
        description="Two NodeExecutionErrors, then succeeds",
        max_retries=3, retry_delay=0.01,
    )
    def flaky_exec(
        text: Annotated[str, Text(label="In")] = "",
    ) -> Annotated[str, Output(label="Out")]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise NodeExecutionError(
                f"transient {call_count}",
                node_id="n1", node_type="flaky-exec@1",
            )
        return "ok"

    compiled = compile(
        nodes=[GraphNode("n1", "flaky-exec@1", {"text": "x"})],
        edges=[],
        registry=reg,
    )

    results = execute_sync(compiled)
    assert results["n1"]["result"] == "ok"
    assert call_count == 3
