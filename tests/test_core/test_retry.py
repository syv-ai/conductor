"""Retry classification.

The engine's retry loop respects the per-error retryable/fatal
classification declared in :mod:`conductor.errors`. Pydantic-validation
failures are covered in ``test_eager_and_retry.py``; this module exercises
the path where a node *itself* raises
:class:`~conductor.errors.NodeValidationError` (or another error with
``retryable=False``) — the engine must not retry, even when the version's
``Policy`` grants retries.
"""

from typing import Annotated

import pytest
from conductor import GraphNode, NodeRegistry, compile
from conductor.dtype import DType
from conductor.errors import (
    FlowExecutionError,
    NodeExecutionError,
    NodeValidationError,
)
from conductor.execution.engine import execute_sync
from conductor.execution.retry import RetryConfig
from conductor.graph.binding import Static
from conductor.graph.model import Flow
from conductor.node import NodeDefinition, Policy, version
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "retry-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


def test_validation_errors_not_retried():
    """A node that raises ``NodeValidationError`` runs exactly once,
    even when its policy grants retries and a run-level RetryConfig is
    configured. Validation errors are fatal — retrying just delays the
    failure and burns latency budget.
    """
    reg = NodeRegistry()
    calls: list[int] = []

    class RaiseValidation(NodeDefinition):
        id = "raise-validation"
        title = "Raise Validation"
        description = "Always raises NodeValidationError"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=0.01))
        def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Out:
            calls.append(1)
            raise NodeValidationError(
                "intentionally invalid input",
                node_id="n1",
                node_type="raise-validation",
            )

    reg.register(RaiseValidation)
    compiled = compile(Flow(nodes=[GraphNode("n1", "raise-validation", 1, bindings={"text": Static(value="x")})]), reg)

    with pytest.raises(FlowExecutionError):
        execute_sync(compiled, retry=RetryConfig(max_retries=3, delay=0.01))

    # Exactly one invocation — no retries.
    assert len(calls) == 1, (
        f"expected NodeValidationError to be fatal (1 call), got {len(calls)}"
    )


def test_node_execution_error_with_retryable_false_not_retried():
    """An instance-level ``retryable=False`` override on
    :class:`NodeExecutionError` should also short-circuit the retry loop.
    """
    reg = NodeRegistry()
    calls: list[int] = []

    class FatalExecutionError(NodeExecutionError):
        retryable = False

    class RaiseFatal(NodeDefinition):
        id = "raise-fatal"
        title = "Raise Fatal"
        description = "Raises a fatal NodeExecutionError subclass"
        category = "test"

        @version(1, policy=Policy(retries=5, delay=0.01))
        def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Out:
            calls.append(1)
            raise FatalExecutionError(
                "fatal — do not retry",
                node_id="n1", node_type="raise-fatal",
            )

    reg.register(RaiseFatal)
    compiled = compile(Flow(nodes=[GraphNode("n1", "raise-fatal", 1, bindings={"text": Static(value="x")})]), reg)

    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)

    assert len(calls) == 1, (
        f"expected fatal NodeExecutionError to skip retries (1 call), "
        f"got {len(calls)}"
    )


def test_default_node_execution_error_still_retries():
    """Sanity check: a vanilla ``NodeExecutionError`` (default
    ``retryable=True``) triggers the retry loop. Guards against the
    classification turning every error into a fatal one.
    """
    reg = NodeRegistry()
    calls: list[int] = []

    class FlakyExec(NodeDefinition):
        id = "flaky-exec"
        title = "Flaky Exec"
        description = "Two NodeExecutionErrors, then succeeds"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=0.01))
        def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Out:
            calls.append(1)
            if len(calls) < 3:
                raise NodeExecutionError(
                    f"transient {len(calls)}",
                    node_id="n1", node_type="flaky-exec",
                )
            return Txt("ok")

    reg.register(FlakyExec)
    compiled = compile(Flow(nodes=[GraphNode("n1", "flaky-exec", 1, bindings={"text": Static(value="x")})]), reg)

    results = execute_sync(compiled)
    assert results["n1"]["result"] == "ok"
    assert len(calls) == 3
