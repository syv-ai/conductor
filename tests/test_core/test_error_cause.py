"""A failure carries its own explanation."""

import pytest
from conductor.errors import CompilationError, ErrorCause, NodeError, NodeExecutionError


def test_a_cause_names_a_code_a_message_and_its_details():
    cause = ErrorCause(
        code="rate_limited",
        message="The model rejected the call; try again shortly.",
        details={"retry_after": 30},
    )

    assert cause.code == "rate_limited"
    assert cause.details["retry_after"] == 30
    assert cause.row is None


def test_a_cause_on_an_iterating_node_names_the_row():
    """The row is its path on the iteration index — (2,) for the third
    document, (2, 0) for that document's first line."""
    cause = ErrorCause(code="x", message="y", row=(2, 0))

    assert cause.row == (2, 0)
    assert cause.details == {}


def test_a_cause_is_frozen():
    with pytest.raises(Exception):
        ErrorCause(code="x", message="y").code = "z"


def test_an_error_without_a_cause_is_still_valid():
    """Most failures are ordinary exceptions and stay that way."""
    assert NodeExecutionError("went wrong").cause is None


def test_compilation_error_carries_the_problems_it_refused_on():
    from conductor.graph.problem import Problem

    problem = Problem(code="cycle", message="m", fatal=True, node_id="a")

    assert CompilationError("no", problems=(problem,)).problems == (problem,)


def test_the_hierarchy_is_what_is_raised_and_nothing_else():
    import conductor.errors as errors

    for gone in (
        "CycleDetectionError", "TypeCheckError", "LoopRunawayError", "SubprocessFailedError",
        "InputResolutionError", "NodeValidationException", "NodeExecutionException",
        "FlowExecutionException", "FlowPausedException",
        "HumanInputRequired", "SignalRequired", "FlowPausedError",
    ):
        assert not hasattr(errors, gone), gone
    assert issubclass(NodeExecutionError, NodeError)


def test_a_pending_leg_is_not_an_error_but_the_sync_wrapper_hands_it_back_as_one():
    """A pause is a leg boundary, not a failure. `execute_sync` has no
    other channel, so it raises this — with the questions and the cells."""
    from conductor.errors import FlowPendingError

    pending = FlowPendingError([{"node_id": "ask", "row": None, "questions": ()}], {"cells": []})

    assert pending.pending[0]["node_id"] == "ask"
    assert pending.cells == {"cells": []}


def test_every_code_the_engine_emits_is_declared_once():
    """``conductor.errors.CODES`` is what a host translating causes by code
    has to cover, so it equals the literals at the emitting sites — the
    engine and the ledger. Read off the source with ``ast``, so the
    docstring example in ``errors.py`` does not count."""
    import ast
    from pathlib import Path

    import conductor.errors as errors
    import conductor.execution as execution
    from conductor.errors import CODES

    emitted: set[str] = set()
    for path in [Path(errors.__file__), *Path(execution.__file__).parent.glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "code" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        emitted.add(keyword.value.value)

    assert emitted == CODES
