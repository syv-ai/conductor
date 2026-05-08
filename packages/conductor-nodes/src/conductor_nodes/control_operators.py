"""Shared operator catalog for if-else style condition builders.

This module is the single source of truth for the ~22 operators that the
legacy AKA ``if-else@1`` node and the conductor port both consume. The
catalog is intentionally Python-flat: each ``OperatorSpec`` carries a
stable ``id``, a Danish UI ``label``, a ``type_category`` (tekst / tal /
boolean / json), an ``arity`` (1 unary | 2 binary), and an ``evaluator``
callable.

The JSON dump (``control_operators.json``) sits next to this file and is
mirrored into the frontend so the operator picker and the backend agree
on which operator is which.

Persistence migration
---------------------

Legacy flows persisted the Danish ``label`` (e.g. ``"indeholder"``) on
the condition row. New flows persist the stable ``id`` (e.g.
``"contains"``). For backward compatibility, ``evaluate()`` accepts
EITHER form — a frontend migration rewrites label-form to id-form on
read, but old saved flows may still send the label.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


# =============================================================================
# Type conversion helpers (mirror legacy backend/app/services/flows/nodes/control.py)
# =============================================================================


class OperatorEvaluationError(ValueError):
    """Raised by an operator evaluator when conversion or input is invalid.

    The if-else evaluator catches this and treats the row as ``False`` so
    the surrounding flow keeps running. Kept as ``ValueError`` subclass so
    legacy callers that catch the broader exception still work.
    """


def _to_number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise OperatorEvaluationError(
            f"Kan ikke konvertere '{value}' til tal"
        ) from None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "ja", "sand")


def _to_json(value: Any) -> dict[str, Any] | list[Any]:
    if isinstance(value, (dict, list)):
        return value
    try:
        result: dict[str, Any] | list[Any] = json.loads(value)
        return result
    except (TypeError, json.JSONDecodeError):
        snippet = str(value)
        snippet = snippet[:100] if len(snippet) > 100 else snippet
        raise OperatorEvaluationError(f"Ugyldig JSON: {snippet}") from None


def _json_key_value_check(
    value: str, compare: str, check_fn: Callable[[str, str], bool]
) -> bool:
    obj = _to_json(value)
    if not isinstance(obj, dict):
        raise OperatorEvaluationError("JSON skal være et objekt for nøgle-opslag")
    if "=" not in compare:
        raise OperatorEvaluationError("Format: 'nøgle=forventet_værdi'")
    key, expected = compare.split("=", 1)
    return check_fn(str(obj.get(key, "")), expected)


# =============================================================================
# Operator catalog
# =============================================================================


TypeCategory = Literal["tekst", "tal", "boolean", "json"]


@dataclass(frozen=True)
class OperatorSpec:
    """A single operator in the if-else builder catalog."""

    id: str
    label: str
    type_category: TypeCategory
    arity: Literal[1, 2]
    evaluator: Callable[[str, str], bool]


# Order matters: it's how the operator picker lists them in the UI.
OPERATORS: list[OperatorSpec] = [
    # ---- Tekst (8) -----------------------------------------------------------
    OperatorSpec(
        id="equals",
        label="er lig med",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: v == c,
    ),
    OperatorSpec(
        id="not_equals",
        label="er ikke lig med",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: v != c,
    ),
    OperatorSpec(
        id="contains",
        label="indeholder",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: c in v,
    ),
    OperatorSpec(
        id="not_contains",
        label="indeholder ikke",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: c not in v,
    ),
    OperatorSpec(
        id="is_empty",
        label="er tom",
        type_category="tekst",
        arity=1,
        evaluator=lambda v, c: not v or not v.strip(),
    ),
    OperatorSpec(
        id="is_not_empty",
        label="er ikke tom",
        type_category="tekst",
        arity=1,
        evaluator=lambda v, c: bool(v and v.strip()),
    ),
    OperatorSpec(
        id="starts_with",
        label="starter med",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: v.startswith(c),
    ),
    OperatorSpec(
        id="ends_with",
        label="slutter med",
        type_category="tekst",
        arity=2,
        evaluator=lambda v, c: v.endswith(c),
    ),
    # ---- Tal (6) -------------------------------------------------------------
    OperatorSpec(
        id="num_equals",
        label="er lig med (tal)",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) == _to_number(c or "0"),
    ),
    OperatorSpec(
        id="num_not_equals",
        label="er ikke lig med (tal)",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) != _to_number(c or "0"),
    ),
    OperatorSpec(
        id="gt",
        label="er større end",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) > _to_number(c or "0"),
    ),
    OperatorSpec(
        id="lt",
        label="er mindre end",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) < _to_number(c or "0"),
    ),
    OperatorSpec(
        id="gte",
        label="er større end eller lig",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) >= _to_number(c or "0"),
    ),
    OperatorSpec(
        id="lte",
        label="er mindre end eller lig",
        type_category="tal",
        arity=2,
        evaluator=lambda v, c: _to_number(v) <= _to_number(c or "0"),
    ),
    # ---- Boolean (2) ---------------------------------------------------------
    OperatorSpec(
        id="is_true",
        label="er sand",
        type_category="boolean",
        arity=1,
        evaluator=lambda v, c: _to_bool(v) is True,
    ),
    OperatorSpec(
        id="is_false",
        label="er falsk",
        type_category="boolean",
        arity=1,
        evaluator=lambda v, c: _to_bool(v) is False,
    ),
    # ---- JSON (6) ------------------------------------------------------------
    OperatorSpec(
        id="has_key",
        label="indeholder nøgle",
        type_category="json",
        arity=2,
        evaluator=lambda v, c: c in _to_json(v),
    ),
    OperatorSpec(
        id="not_has_key",
        label="indeholder ikke nøgle",
        type_category="json",
        arity=2,
        evaluator=lambda v, c: c not in _to_json(v),
    ),
    OperatorSpec(
        id="is_empty_object",
        label="er tom objekt",
        type_category="json",
        arity=1,
        evaluator=lambda v, c: len(_to_json(v)) == 0,
    ),
    OperatorSpec(
        id="is_not_empty_object",
        label="er ikke tom objekt",
        type_category="json",
        arity=1,
        evaluator=lambda v, c: len(_to_json(v)) > 0,
    ),
    OperatorSpec(
        id="value_equals",
        label="værdi er lig med",
        type_category="json",
        arity=2,
        evaluator=lambda v, c: _json_key_value_check(v, c, lambda a, b: a == b),
    ),
    OperatorSpec(
        id="value_contains",
        label="værdi indeholder",
        type_category="json",
        arity=2,
        evaluator=lambda v, c: _json_key_value_check(v, c, lambda a, b: b in a),
    ),
]


# Lookup tables --------------------------------------------------------------

OPERATORS_BY_ID: dict[str, OperatorSpec] = {op.id: op for op in OPERATORS}
OPERATORS_BY_LABEL: dict[str, OperatorSpec] = {op.label: op for op in OPERATORS}

OPERATORS_BY_TYPE: dict[str, list[str]] = {
    "tekst": [op.id for op in OPERATORS if op.type_category == "tekst"],
    "tal": [op.id for op in OPERATORS if op.type_category == "tal"],
    "boolean": [op.id for op in OPERATORS if op.type_category == "boolean"],
    "json": [op.id for op in OPERATORS if op.type_category == "json"],
}

UNARY_OPERATORS: set[str] = {op.id for op in OPERATORS if op.arity == 1}


def evaluate(op_id_or_label: str, a: Any, b: Any = None) -> bool:
    """Evaluate an operator by stable id OR legacy Danish label.

    Saved-flow back-compat: production data persists either form. New
    flows write the stable id; legacy flows write the label until the
    one-shot read-time migration rewrites them.

    Args:
        op_id_or_label: Stable id (``"contains"``) or legacy label
            (``"indeholder"``).
        a: Left operand. Coerced to ``str`` before passing to the
            evaluator (mirrors legacy behaviour).
        b: Right operand. Ignored for unary operators. Defaults to ``""``
            for binary operators when ``None`` is passed.

    Returns:
        Boolean result of the comparison.

    Raises:
        OperatorEvaluationError: If the operator id/label is unknown, or
            the evaluator's input cannot be parsed (bad JSON, non-numeric
            string for a numeric operator, etc.).
    """
    spec = OPERATORS_BY_ID.get(op_id_or_label) or OPERATORS_BY_LABEL.get(
        op_id_or_label
    )
    if spec is None:
        known = sorted({*OPERATORS_BY_ID, *OPERATORS_BY_LABEL})
        raise OperatorEvaluationError(
            f"Ukendt operator: '{op_id_or_label}'. "
            f"Gyldige operatorer: {', '.join(known)}"
        )

    a_str = "" if a is None else str(a)
    if spec.arity == 1:
        # Unary: ignore b entirely, mirror legacy by passing "" as second arg.
        return spec.evaluator(a_str, "")
    b_str = "" if b is None else str(b)
    return spec.evaluator(a_str, b_str)


__all__ = [
    "OperatorEvaluationError",
    "OperatorSpec",
    "OPERATORS",
    "OPERATORS_BY_ID",
    "OPERATORS_BY_LABEL",
    "OPERATORS_BY_TYPE",
    "UNARY_OPERATORS",
    "evaluate",
]
