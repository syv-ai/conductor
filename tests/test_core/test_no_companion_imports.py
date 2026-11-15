"""The core's tests ship with the core alone, so no file of them imports conductor-nodes or conductor-providers,
at the top of the file or inside a function."""

from __future__ import annotations

import ast
from ast import AST, Import, ImportFrom
from pathlib import Path

HERE = Path(__file__).parent
COMPANIONS = {"conductor_nodes", "conductor_providers"}


def _imported(found: AST) -> list[str]:
    """The modules an import statement names; an ``import`` or a ``from`` that is not relative."""
    if isinstance(found, Import):
        return [alias.name for alias in found.names]
    if isinstance(found, ImportFrom) and found.level == 0 and found.module:
        return [found.module]
    return []


def test_no_core_test_imports_a_companion() -> None:
    found = [
        f"{path.relative_to(HERE)}:{statement.lineno} imports {module}"
        for path in sorted(HERE.rglob("*.py"))
        for statement in ast.walk(ast.parse(path.read_text(), str(path)))
        for module in _imported(statement)
        if module.split(".")[0] in COMPANIONS
    ]
    assert found == [], (
        "a core test ships without the companions; test them in tests/test_nodes or tests/test_providers"
    )
