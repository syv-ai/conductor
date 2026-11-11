"""Every page's Python runs, and each line ``print(...)  # shown`` prints what the page shows.

Each page's blocks run in order as one module, in an empty directory. The comment may go on after what is printed
(``# {...}: it still runs version 1``), so it begins with the printed text, quotes read alike. Notebooks are run by
their own check.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import re
import sys
from typing import Any

import pytest

from tests.test_core.doc_examples import pages

_SHOWN = re.compile(r"^(\s*)print\((.*)\)\s+#\s?(.*)$")
_PAGES = pages()


def _checked_source(source: str) -> str:
    """``source`` with each shown print calling ``__shown`` with what it would print and what the page shows."""
    lines = []
    for number, line in enumerate(source.splitlines(), start=1):
        hit = _SHOWN.match(line)
        if hit:
            indent, arguments, claim = hit.groups()
            line = f"{indent}__shown({number}, {claim!r}, {arguments})"
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_a_page_runs_and_prints_what_it_shows(name: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = _PAGES[name]
    printed: list[tuple[int, str, str]] = []

    def shown(number: int, claim: str, *arguments: Any, sep: str = " ") -> None:
        out = io.StringIO()
        print(*arguments, sep=sep, file=out)
        printed.append((number, claim, out.getvalue().rstrip("\n")))

    module = type(sys)(name)
    module.__dict__["__shown"] = shown
    monkeypatch.setitem(sys.modules, name, module)
    code = compile(_checked_source(source), name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with contextlib.redirect_stdout(io.StringIO()):
        ran = eval(code, module.__dict__)  # noqa: S307  the page's own code, as a reader runs it
        if inspect.iscoroutine(ran):
            asyncio.run(ran)

    lines = source.splitlines()
    differ = [
        f"line {number}: {lines[number - 1].strip()}\n  shows:  {claim}\n  prints: {got}"
        for number, claim, got in printed
        if not claim.replace('"', "'").startswith(got.replace('"', "'"))
    ]
    assert not differ, "\n".join(differ)
