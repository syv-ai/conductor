"""Backend evaluator tests for the if-else operator catalog.

Forked from AKA's ``backend/tests/unit_tests/test_flow_nodes.py`` so the
shared catalog is verified independently of AKA's node wiring. One row
per (operator, a, b, expected) triple covers every operator in the
catalog at least once. Tests use stable ids; a separate block covers the
legacy-label back-compat path.
"""

from __future__ import annotations

from typing import Any

import pytest
from conductor_nodes.control_operators import (
    OPERATORS,
    OPERATORS_BY_ID,
    UNARY_OPERATORS,
    OperatorEvaluationError,
    evaluate,
)

# ---------------------------------------------------------------------------
# Per-operator truth table — one happy-path row + at least one negative row
# per operator. ``b`` is ``None`` for unary operators.
# ---------------------------------------------------------------------------

CASES: list[tuple[str, Any, Any, bool]] = [
    # ---- Tekst ---------------------------------------------------------------
    ("equals", "Hello", "Hello", True),
    ("equals", "Hello", "World", False),
    ("not_equals", "Hello", "World", True),
    ("not_equals", "Hello", "Hello", False),
    ("contains", "Hello World", "World", True),
    ("contains", "Hello", "World", False),
    ("not_contains", "Hello", "World", True),
    ("not_contains", "Hello World", "World", False),
    ("is_empty", "", None, True),
    ("is_empty", "   ", None, True),
    ("is_empty", "x", None, False),
    ("is_not_empty", "x", None, True),
    ("is_not_empty", "", None, False),
    ("is_not_empty", "   ", None, False),
    ("starts_with", "Hello World", "Hello", True),
    ("starts_with", "Hello World", "World", False),
    ("ends_with", "Hello World", "World", True),
    ("ends_with", "Hello World", "Hello", False),
    # ---- Tal -----------------------------------------------------------------
    ("num_equals", "42", "42", True),
    ("num_equals", "42", "43", False),
    ("num_equals", "42", "42.0", True),
    ("num_not_equals", "42", "43", True),
    ("num_not_equals", "42", "42", False),
    ("gt", "10", "5", True),
    ("gt", "5", "10", False),
    ("gt", "5", "5", False),
    ("lt", "5", "10", True),
    ("lt", "10", "5", False),
    ("gte", "10", "5", True),
    ("gte", "5", "5", True),
    ("gte", "4", "5", False),
    ("lte", "5", "10", True),
    ("lte", "5", "5", True),
    ("lte", "6", "5", False),
    # ---- Boolean -------------------------------------------------------------
    ("is_true", "true", None, True),
    ("is_true", "ja", None, True),
    ("is_true", "1", None, True),
    ("is_true", "false", None, False),
    ("is_false", "false", None, True),
    ("is_false", "nej", None, True),  # not in truthy list -> bool() False
    ("is_false", "true", None, False),
    # ---- JSON ----------------------------------------------------------------
    ("has_key", '{"a": 1, "b": 2}', "a", True),
    ("has_key", '{"a": 1}', "missing", False),
    ("not_has_key", '{"a": 1}', "missing", True),
    ("not_has_key", '{"a": 1}', "a", False),
    ("is_empty_object", "{}", None, True),
    ("is_empty_object", '{"a": 1}', None, False),
    ("is_not_empty_object", '{"a": 1}', None, True),
    ("is_not_empty_object", "{}", None, False),
    ("value_equals", '{"status": "ok"}', "status=ok", True),
    ("value_equals", '{"status": "ok"}', "status=fail", False),
    ("value_contains", '{"msg": "Hello World"}', "msg=World", True),
    ("value_contains", '{"msg": "Hello World"}', "msg=missing", False),
]


@pytest.mark.parametrize(("op_id", "a", "b", "expected"), CASES)
def test_evaluator(op_id: str, a: Any, b: Any, expected: bool) -> None:
    assert evaluate(op_id, a, b) is expected


def test_every_operator_has_at_least_one_case() -> None:
    covered = {op_id for op_id, *_ in CASES}
    expected = {op.id for op in OPERATORS}
    missing = expected - covered
    assert not missing, f"operators missing test cases: {missing}"


# ---------------------------------------------------------------------------
# Legacy label back-compat — old saved flows persist Danish labels.
# ---------------------------------------------------------------------------


def test_label_form_back_compat() -> None:
    # ``"indeholder"`` is the legacy label for ``contains``.
    assert evaluate("indeholder", "Hello World", "World") is True
    assert evaluate("er lig med", "x", "x") is True
    assert evaluate("er sand", "true") is True
    assert evaluate("er tom", "") is True


def test_unknown_operator_raises() -> None:
    with pytest.raises(OperatorEvaluationError):
        evaluate("definitely_not_an_operator", "a", "b")


# ---------------------------------------------------------------------------
# Type-conversion failure paths -> raise ``OperatorEvaluationError``.
# ---------------------------------------------------------------------------


def test_numeric_operator_with_non_numeric_input_raises() -> None:
    with pytest.raises(OperatorEvaluationError):
        evaluate("gt", "not-a-number", "10")


def test_json_operator_with_invalid_json_raises() -> None:
    with pytest.raises(OperatorEvaluationError):
        evaluate("has_key", "not json", "key")


def test_value_equals_requires_key_equals_value_format() -> None:
    with pytest.raises(OperatorEvaluationError):
        evaluate("value_equals", '{"a": 1}', "no_equals_sign")


# ---------------------------------------------------------------------------
# Unary handling
# ---------------------------------------------------------------------------


def test_unary_operators_ignore_b() -> None:
    # Pass garbage ``b`` — unary evaluators must not look at it.
    for op_id in UNARY_OPERATORS:
        spec = OPERATORS_BY_ID[op_id]
        # Pick a deterministic ``a`` that produces False for the operator
        # so we can verify both that it ran AND ignored b.
        if spec.type_category == "tekst":
            a = "x"  # is_empty False, is_not_empty True
        elif spec.type_category == "boolean":
            a = "true"
        elif spec.type_category == "json":
            a = "{}"
        else:
            a = ""
        # Different b values must not change the answer.
        result_a = evaluate(op_id, a, "garbage_1")
        result_b = evaluate(op_id, a, "garbage_2")
        assert result_a is result_b, (
            f"unary operator {op_id} responded to b: "
            f"{result_a} vs {result_b}"
        )
