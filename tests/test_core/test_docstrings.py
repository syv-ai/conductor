"""Every public class and module-level function of the three packages has a docstring.

A name with a leading underscore, an `@overload` and a body of `...` need none, and neither does a method or a
function defined inside another function. What a docstring says, and that it is in Google style, is read in review.
"""

from __future__ import annotations

import ast
from ast import AST, AsyncFunctionDef, ClassDef, Constant, Expr, FunctionDef
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_overload(definition: FunctionDef | AsyncFunctionDef | ClassDef) -> bool:
    """Whether the definition is decorated with `overload` or `typing.overload`."""
    return any(ast.unparse(decorator).split(".")[-1] == "overload" for decorator in definition.decorator_list)


def _is_stub(definition: FunctionDef | AsyncFunctionDef | ClassDef) -> bool:
    """Whether the definition's whole body is `...`."""
    body = definition.body
    return len(body) == 1 and isinstance(body[0], Expr) and isinstance(body[0].value, Constant) and body[0].value.value is Ellipsis


def missing_docstrings(path: Path) -> list[str]:
    """The definitions in one file that need a docstring and have none, as `path:line: name`.

    Args:
        path: A Python file.

    Returns:
        One line for each class at any depth outside a function, and each function at the module's top level, that is
        public, not an overload or a stub, and has no docstring.
    """
    found: list[str] = []

    def visit(node: AST, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FunctionDef | AsyncFunctionDef | ClassDef):
                is_class = isinstance(child, ClassDef)
                needs_one = (is_class or not in_class) and not child.name.startswith("_")
                if needs_one and not _is_overload(child) and not _is_stub(child) and ast.get_docstring(child) is None:
                    found.append(f"{path.relative_to(ROOT)}:{child.lineno}: {child.name}")
                if is_class:
                    visit(child, in_class=True)
            else:
                visit(child, in_class)

    visit(ast.parse(path.read_text()), in_class=False)
    return found


def test_every_public_definition_has_a_docstring() -> None:
    missing = [line for path in sorted(ROOT.glob("packages/*/src/**/*.py")) for line in missing_docstrings(path)]
    assert not missing, "no docstring:\n" + "\n".join(missing)
