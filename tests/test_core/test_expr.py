"""Tests for the CEL expression engine."""

from __future__ import annotations

import pytest
from conductor.expr import (
    ExpressionParseError,
    ExpressionRuntimeError,
    evaluate,
    parse,
)

# ---------------------------------------------------------------------------
# Literals and arithmetic
# ---------------------------------------------------------------------------


def test_integer_literals() -> None:
    assert evaluate("42", {}) == 42
    assert evaluate("0", {}) == 0


def test_float_literals() -> None:
    assert evaluate("3.14", {}) == 3.14


def test_string_literals() -> None:
    assert evaluate("'hello'", {}) == "hello"
    assert evaluate('"world"', {}) == "world"
    assert evaluate("'with \\'quote\\''", {}) == "with 'quote'"


def test_boolean_and_null() -> None:
    assert evaluate("true", {}) is True
    assert evaluate("false", {}) is False
    assert evaluate("null", {}) is None
    assert evaluate("None", {}) is None


def test_arithmetic_precedence() -> None:
    assert evaluate("1 + 2 * 3", {}) == 7
    assert evaluate("(1 + 2) * 3", {}) == 9
    assert evaluate("10 / 2", {}) == 5
    assert evaluate("10 % 3", {}) == 1


def test_int_division_truncates_toward_zero() -> None:
    """CEL: int/int → int, truncated toward zero (not Python's floor)."""
    assert evaluate("5 / 2", {}) == 2
    assert evaluate("-5 / 2", {}) == -2  # not -3 (floor)
    assert evaluate("5 / -2", {}) == -2
    assert evaluate("-5 / -2", {}) == 2
    assert evaluate("6 / 2", {}) == 3
    # Mixed int/float stays float.
    assert evaluate("5 / 2.0", {}) == 2.5
    assert evaluate("5.0 / 2", {}) == 2.5


def test_unary() -> None:
    assert evaluate("-5", {}) == -5
    assert evaluate("!true", {}) is False
    assert evaluate("not true", {}) is False


def test_string_concatenation() -> None:
    assert evaluate('"hello" + " " + "world"', {}) == "hello world"


# ---------------------------------------------------------------------------
# Comparisons and logical ops
# ---------------------------------------------------------------------------


def test_comparisons() -> None:
    assert evaluate("1 < 2", {}) is True
    assert evaluate("2 <= 2", {}) is True
    assert evaluate("3 > 2", {}) is True
    assert evaluate("3 >= 3", {}) is True
    assert evaluate("1 == 1", {}) is True
    assert evaluate("1 != 2", {}) is True


def test_logical() -> None:
    assert evaluate("true && true", {}) is True
    assert evaluate("true && false", {}) is False
    assert evaluate("false || true", {}) is True
    assert evaluate("true and false", {}) is False
    assert evaluate("false or true", {}) is True


def test_short_circuit_and() -> None:
    # rhs would raise (missing x) but && short-circuits
    assert evaluate("false && x", {}) is False


def test_short_circuit_or() -> None:
    assert evaluate("true || x", {}) is True


def test_in_operator() -> None:
    assert evaluate("1 in [1, 2, 3]", {}) is True
    assert evaluate("4 in [1, 2, 3]", {}) is False
    assert evaluate('"k" in {"k": 1}', {}) is True


# ---------------------------------------------------------------------------
# Identifiers and context
# ---------------------------------------------------------------------------


def test_simple_ident() -> None:
    assert evaluate("x + 1", {"x": 5}) == 6


def test_attr_access() -> None:
    ctx = {"invoice": {"amount": 1500, "tier": "gold"}}
    assert evaluate("invoice.amount > 1000", ctx) is True
    assert evaluate('invoice.tier == "gold"', ctx) is True


def test_bracket_access() -> None:
    assert evaluate("a[0]", {"a": [1, 2, 3]}) == 1
    assert evaluate("m[\"k\"]", {"m": {"k": 42}}) == 42


def test_dollar_root() -> None:
    ctx = {"foo": {"bar": 42}}
    assert evaluate("$.foo.bar", ctx) == 42


def test_undefined_raises() -> None:
    with pytest.raises(ExpressionRuntimeError):
        evaluate("x", {})


# ---------------------------------------------------------------------------
# Ternary
# ---------------------------------------------------------------------------


def test_ternary() -> None:
    assert evaluate("x ? 1 : 2", {"x": True}) == 1
    assert evaluate("x ? 1 : 2", {"x": False}) == 2


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------


def test_size() -> None:
    assert evaluate('size("abc")', {}) == 3
    assert evaluate("size([1, 2, 3])", {}) == 3
    assert evaluate('size({"k": 1})', {}) == 1


def test_has() -> None:
    assert evaluate('has(m, "k")', {"m": {"k": 1}}) is True
    assert evaluate('has(m, "x")', {"m": {"k": 1}}) is False


def test_string_methods_and_functions() -> None:
    assert evaluate('"abc".contains("b")', {}) is True
    assert evaluate('"abc".startsWith("a")', {}) is True
    assert evaluate('"abc".endsWith("c")', {}) is True
    assert evaluate('"abc".matches("[a-z]+")', {}) is True
    assert evaluate('contains("abc", "b")', {}) is True


def test_type_conversions() -> None:
    assert evaluate("int(\"42\")", {}) == 42
    assert evaluate("double(\"3.14\")", {}) == 3.14
    assert evaluate("string(42)", {}) == "42"
    assert evaluate("bool(1)", {}) is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_parse_error() -> None:
    with pytest.raises(ExpressionParseError):
        parse("1 +")


def test_empty_expression() -> None:
    with pytest.raises(ExpressionParseError):
        parse("")


def test_divide_by_zero() -> None:
    with pytest.raises(ExpressionRuntimeError):
        evaluate("1 / 0", {})


def test_identifiers_introspection() -> None:
    assert parse("a.b + c").identifiers() == {"a", "c"}
    # Built-in function names don't count
    assert parse("size(x) > 0").identifiers() == {"x"}


# ---------------------------------------------------------------------------
# List and map literals
# ---------------------------------------------------------------------------


def test_list_literal() -> None:
    assert evaluate("[1, 2, 3]", {}) == [1, 2, 3]


def test_map_literal() -> None:
    assert evaluate('{"k": 1, "j": 2}', {}) == {"k": 1, "j": 2}


def test_nested_access() -> None:
    ctx = {"u": {"roles": ["admin", "user"]}}
    assert evaluate("u.roles[0]", ctx) == "admin"


# ---------------------------------------------------------------------------
# matches() — ReDoS mitigation via length caps
# ---------------------------------------------------------------------------


def test_matches_rejects_oversized_pattern() -> None:
    long_pattern = "a" * 300
    with pytest.raises(ExpressionRuntimeError, match="pattern exceeds"):
        evaluate(f'"abc".matches("{long_pattern}")', {})


def test_matches_rejects_oversized_input() -> None:
    # Pattern is fine, but the subject string is too long.
    huge = "a" * (64 * 1024 + 1)
    with pytest.raises(ExpressionRuntimeError, match="input exceeds"):
        evaluate("s.matches(\"a*\")", {"s": huge})


def test_matches_rejects_invalid_regex() -> None:
    with pytest.raises(ExpressionRuntimeError, match="Invalid regex"):
        evaluate('"abc".matches("[unclosed")', {})


# ---------------------------------------------------------------------------
# Sandbox — attribute access must not leak arbitrary Python methods/attrs
# ---------------------------------------------------------------------------


def test_sandbox_blocks_method_call_on_context_object() -> None:
    """Putting a real object in context must not let expressions invoke its methods."""

    class Service:
        def __init__(self) -> None:
            self.called = False

        def delete(self) -> str:
            self.called = True
            return "deleted"

    svc = Service()
    with pytest.raises(ExpressionRuntimeError):
        evaluate("svc.delete()", {"svc": svc})
    assert svc.called is False


def test_sandbox_blocks_attribute_access_on_context_object() -> None:
    class Obj:
        public = "secret"

    with pytest.raises(ExpressionRuntimeError):
        evaluate("o.public", {"o": Obj()})


def test_sandbox_blocks_dunder_escape_on_string() -> None:
    # Even primitives should not leak Python attributes.
    with pytest.raises(ExpressionRuntimeError):
        evaluate('"x".__class__', {})


def test_pydantic_model_field_access_allowed() -> None:
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    ctx = {"u": User(name="Alice", age=30)}
    assert evaluate("u.name", ctx) == "Alice"
    assert evaluate("u.age > 18", ctx) is True


def test_pydantic_model_method_call_blocked() -> None:
    from pydantic import BaseModel

    class User(BaseModel):
        name: str

    with pytest.raises(ExpressionRuntimeError):
        # model_dump is a real method, but not a declared field.
        evaluate("u.model_dump()", {"u": User(name="Alice")})


def test_dataclass_field_access_allowed() -> None:
    from dataclasses import dataclass

    @dataclass
    class Point:
        x: int
        y: int

    ctx = {"p": Point(x=3, y=4)}
    assert evaluate("p.x + p.y", ctx) == 7


def test_dataclass_method_call_blocked() -> None:
    from dataclasses import dataclass

    @dataclass
    class Thing:
        label: str

        def danger(self) -> str:
            return "pwned"

    with pytest.raises(ExpressionRuntimeError):
        evaluate("t.danger()", {"t": Thing(label="x")})
