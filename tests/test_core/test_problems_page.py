"""Every message conductor writes says what to do, and each example on ``docs/problems.md`` prints the code of its section."""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import re
import sys
import types

import pytest
from conductor import PROBLEMS

_ROOT = pathlib.Path(__file__).parents[2]
_PACKAGE = _ROOT / "packages" / "conductor" / "src" / "conductor"
_PAGE = (_ROOT / "docs" / "problems.md").read_text()
#: Messages that carry no fix of their own: a node's failure keeps the node's words, an outside failure has none a
#: person can act on, and an answer's ``reason`` is pydantic's account of what the answer lacks.
_NO_FIX = {"failed", "external_failed", "invalid_answer"}


def _template(message: ast.expr) -> str:
    """An f-string as its text with ``{slot}`` for each value, the ``{where}`` prefix naming the node left out."""
    if isinstance(message, ast.JoinedStr):
        text = "".join(str(p.value) if isinstance(p, ast.Constant) else "{" + ast.unparse(p.value) + "}" for p in message.values)  # type: ignore[attr-defined]
        return re.sub(r"^\{where\}", "", text)
    return str(ast.literal_eval(message))


def _declaration_messages() -> dict[str, list[str]]:
    """The messages the package raises each declaration code with; an error raised again under the code it had adds none."""
    found: dict[str, list[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for call in ast.walk(ast.parse(path.read_text())):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "NodeDeclarationError":
                code = next(k.value for k in call.keywords if k.arg == "code")
                if isinstance(code, ast.Constant):
                    found.setdefault(code.value, []).append(_template(call.args[0]))
    return found


def test_every_message_says_what_to_do() -> None:
    """A message carries its fix: a clause after what is wrong, which may be the alternatives to pick from."""
    sentences = [(kind.code, kind.sentence) for kind in PROBLEMS]
    without = [code for code, text in sentences if code not in _NO_FIX and not re.search(r"[;:.] \S", text.removesuffix(": {reason}"))]
    assert without == []


def test_every_declaration_message_says_what_to_do() -> None:
    """A declaration's message carries its fix too: a clause after what is wrong, past the name of the field it is about."""
    without = [
        (code, text) for code, texts in _declaration_messages().items() for text in texts
        if not re.search(r"[;,] \S", re.sub(r"^\{[^}]+\}: ", "", text))
    ]
    assert without == []


def test_each_example_prints_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    page = types.ModuleType("problems_page")  # a dataclass on the page looks its module up by name
    monkeypatch.setitem(sys.modules, page.__name__, page)
    blocks = [(match.start(), match.group(1)) for match in re.finditer(r"```python\n(.*?)```", _PAGE, re.S)]
    starts = [(match.start(), match.group(1)) for match in re.finditer(r"^### `([a-z_]+)`", _PAGE, re.M)]
    namespace = page.__dict__
    exec(compile(blocks[0][1], "docs/problems.md", "exec"), namespace)  # noqa: S102  the page's first block, as a reader runs it
    wrong = []
    for at, block in blocks[1:]:
        code = [name for start, name in starts if start < at][-1]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(block, f"docs/problems.md: {code}", "exec"), namespace)  # noqa: S102
        if code not in printed.getvalue():
            wrong.append((code, printed.getvalue()))
    assert wrong == []
