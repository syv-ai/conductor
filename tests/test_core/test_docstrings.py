"""Every public class and top-level function in the three packages has a docstring.

Some definitions need none: a name with a leading underscore, an `@overload`, a body of just `...`, a method, and
a function defined inside another function. This test only checks that a docstring is there. Whether it says the
right thing, in Google style, is up to review.
"""

from __future__ import annotations

import ast
from ast import AST, AsyncFunctionDef, ClassDef, Constant, Expr, FunctionDef
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _is_overload(definition: FunctionDef | AsyncFunctionDef | ClassDef) -> bool:
    """Returns whether the definition is decorated with `overload` or `typing.overload`."""
    return any(
        ast.unparse(decorator).split(".")[-1] == "overload"
        for decorator in definition.decorator_list
    )


def _is_stub(definition: FunctionDef | AsyncFunctionDef | ClassDef) -> bool:
    """Returns whether the definition's whole body is `...`."""
    body = definition.body
    return (
        len(body) == 1
        and isinstance(body[0], Expr)
        and isinstance(body[0].value, Constant)
        and body[0].value.value is Ellipsis
    )


def missing_docstrings(path: Path) -> list[str]:
    """Lists the definitions in one file that need a docstring and have none.

    A definition needs one if it is public and isn't an overload or a stub. That covers every class outside a
    function, however deeply nested, and every function at the top of the module.

    Args:
        path: A Python file.

    Returns:
        One line per missing docstring, as `path:line: name`.
    """
    found: list[str] = []

    def visit(node: AST, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FunctionDef | AsyncFunctionDef | ClassDef):
                is_class = isinstance(child, ClassDef)
                needs_one = (is_class or not in_class) and not child.name.startswith("_")
                if (
                    needs_one
                    and not _is_overload(child)
                    and not _is_stub(child)
                    and ast.get_docstring(child) is None
                ):
                    found.append(f"{path.relative_to(ROOT)}:{child.lineno}: {child.name}")
                if is_class:
                    visit(child, in_class=True)
            else:
                visit(child, in_class)

    visit(ast.parse(path.read_text()), in_class=False)
    return found


def test_every_public_definition_has_a_docstring() -> None:
    missing = [
        line
        for path in sorted(ROOT.glob("packages/*/src/**/*.py"))
        for line in missing_docstrings(path)
    ]
    assert not missing, "no docstring:\n" + "\n".join(missing)
