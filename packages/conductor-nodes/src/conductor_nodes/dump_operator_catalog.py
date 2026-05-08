"""Dump :data:`OPERATORS` to ``control_operators.json``.

Run from the repo root:

    uv run python -m conductor_nodes.dump_operator_catalog

The JSON file mirrors a strict subset of :class:`OperatorSpec` (no
``evaluator`` — that's Python-only). The frontend imports the same JSON
so the operator picker and the backend agree on which operators exist.

Parity is enforced by ``tests/test_nodes/test_operator_catalog.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from conductor_nodes.control_operators import OPERATORS

OUTPUT = Path(__file__).parent / "control_operators.json"


def serialize() -> list[dict[str, object]]:
    """Convert :data:`OPERATORS` to a JSON-serialisable list."""
    return [
        {
            "id": op.id,
            "label": op.label,
            "type_category": op.type_category,
            "arity": op.arity,
        }
        for op in OPERATORS
    ]


def main() -> None:
    OUTPUT.write_text(
        json.dumps(serialize(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(OPERATORS)} operators to {OUTPUT}")


if __name__ == "__main__":
    main()
