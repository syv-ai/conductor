"""The Python on every illustrated page runs, and prints what the page shows it prints.

The pages are HTML in ``docs/explainers/``. ``tests.test_core.doc_examples.explainer_samples`` reads each page's
samples, the setup it hides, and the output it shows after each. This test runs one page's Python in order, in one
module, in an empty directory, and compares what each piece printed with the output shown after it.

A piece that raises prints the exception's last line, ``TypeError: ...``, as a traceback ends, and the page goes
on. So a page shows a refusal as the output of the code that is refused. A shown output is all of the printed
lines, unless the page marks it as some of them, in order. Single and double quotes count as the same.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import linecache
import sys
import traceback
from io import StringIO
from typing import Any

import pytest
from pytest import MonkeyPatch

from tests.test_core.doc_examples import ExplainerSample, explainer_samples

_PAGES = explainer_samples()


def _lines(text: str) -> list[str]:
    return [line.rstrip().replace('"', "'") for line in text.strip("\n").splitlines()]


def _shows(sample: ExplainerSample, printed: str) -> bool:
    got = _lines(printed)
    shown = [line for block in sample.shown for line in _lines(block)]
    if not sample.in_order:
        return shown == got
    rest = iter(got)
    return all(any(line == other for other in rest) for line in shown if line)


def _run(sample: ExplainerSample, name: str, module: Any, monkeypatch: MonkeyPatch) -> str:
    where = f"{name}, line {sample.line}"
    # The piece's source can be read, as a file's can, so @node reads a function in it as it reads one in a file.
    monkeypatch.setitem(
        linecache.cache,
        where,
        (len(sample.code), None, sample.code.splitlines(keepends=True), where),
    )
    code = compile(sample.code, where, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    out = StringIO()
    with contextlib.redirect_stdout(out):
        try:
            ran = eval(code, module.__dict__)  # noqa: S307  the page's own code, as a reader runs it
            if inspect.iscoroutine(ran):
                asyncio.run(ran)
        except Exception as error:  # noqa: BLE001  the page shows the refusal as the piece's output
            print(traceback.format_exception_only(error)[-1].rstrip("\n"))
    return out.getvalue()


@pytest.mark.parametrize("name", sorted(_PAGES))
def test_an_illustrated_page_runs_and_prints_what_it_shows(
    name: str, tmp_path: Any, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    module = type(sys)(name)
    monkeypatch.setitem(sys.modules, name, module)
    differ = []
    for sample in _PAGES[name]:
        printed = _run(sample, name, module, monkeypatch)
        if sample.shown and not _shows(sample, printed):
            shown = "\n".join(sample.shown)
            differ.append(f"line {sample.line}:\n  shows:\n{shown}\n  prints:\n{printed}")
        elif not sample.shown and printed.strip():
            differ.append(f"line {sample.line}: prints output the page doesn't show:\n{printed}")
    assert not differ, "\n".join(differ)
