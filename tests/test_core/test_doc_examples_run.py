"""The Python on every docs page runs, and its printed output matches what the page shows.

A page shows output with a comment after a print, like ``print(run.outputs)  # {'result': 'HI'}``. This test runs
each page's blocks in order, as one module, in an empty directory. It then checks that each such print printed
what its comment says.

The comment may carry on after the output, as in ``# {...}: it still runs version 1``. So the test only checks
that the comment starts with what was printed. Single and double quotes count as the same.

The notebooks have a check of their own.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import linecache
import re
import sys
from io import StringIO
from typing import Any

import pytest
from pytest import MonkeyPatch

from tests.test_core.doc_examples import pages

_SHOWN = re.compile(r"^(\s*)print\((.*)\)\s+#\s?(.*)$")
_PAGES = pages()


def _checked_source(source: str) -> str:
    """Returns ``source`` with each commented print turned into a call to ``__shown``.

    The call gets the line number, the comment, and what the print would have printed. That way the test can
    compare the two after the page has run.
    """
    lines = []
    for number, line in enumerate(source.splitlines(), start=1):
        hit = _SHOWN.match(line)
        if hit:
            indent, arguments, claim = hit.groups()
            line = f"{indent}__shown({number}, {claim!r}, {arguments})"
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_a_page_runs_and_prints_what_it_shows(
    name: str, tmp_path: Any, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = _PAGES[name]
    printed: list[tuple[int, str, str]] = []

    def shown(number: int, claim: str, *arguments: Any, sep: str = " ") -> None:
        out = StringIO()
        print(*arguments, sep=sep, file=out)
        printed.append((number, claim, out.getvalue().rstrip("\n")))

    module = type(sys)(name)
    module.__dict__["__shown"] = shown
    monkeypatch.setitem(sys.modules, name, module)
    checked = _checked_source(source)
    # The page's source can be read, as a file's can, so @node reads a function on it as it reads one in a file.
    monkeypatch.setitem(
        linecache.cache, name, (len(checked), None, checked.splitlines(keepends=True), name)
    )
    code = compile(checked, name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with contextlib.redirect_stdout(StringIO()):
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
