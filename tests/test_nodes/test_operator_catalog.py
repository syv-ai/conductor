"""Parity test: ``control_operators.json`` matches the dataclass dump.

Failure means someone edited the JSON by hand or forgot to re-run
``python -m conductor_nodes.dump_operator_catalog`` after touching
``control_operators.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from conductor_nodes.control_operators import (
    OPERATORS,
    OPERATORS_BY_TYPE,
    UNARY_OPERATORS,
)
from conductor_nodes.dump_operator_catalog import serialize

JSON_PATH = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "conductor-nodes"
    / "src"
    / "conductor_nodes"
    / "control_operators.json"
)


def test_json_dump_matches_dataclass() -> None:
    on_disk = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert on_disk == serialize(), (
        "control_operators.json is stale — re-run "
        "`python -m conductor_nodes.dump_operator_catalog`."
    )


def test_catalog_has_at_least_twenty_operators() -> None:
    # Audit floor: legacy ``if-else@1`` shipped 22 operators; the conductor
    # port must keep at least 20 to avoid silently regressing flows.
    assert len(OPERATORS) >= 20


def test_operator_ids_are_unique() -> None:
    ids = [op.id for op in OPERATORS]
    assert len(ids) == len(set(ids)), "duplicate operator ids in OPERATORS"


def test_operator_labels_are_unique() -> None:
    labels = [op.label for op in OPERATORS]
    assert len(labels) == len(set(labels)), "duplicate operator labels in OPERATORS"


def test_operators_by_type_covers_every_operator() -> None:
    indexed = {op_id for ids in OPERATORS_BY_TYPE.values() for op_id in ids}
    assert indexed == {op.id for op in OPERATORS}


def test_unary_operators_match_arity_one() -> None:
    assert UNARY_OPERATORS == {op.id for op in OPERATORS if op.arity == 1}


def test_type_categories_are_valid() -> None:
    valid = {"tekst", "tal", "boolean", "json"}
    assert all(op.type_category in valid for op in OPERATORS)
