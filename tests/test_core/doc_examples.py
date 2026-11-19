"""Collects the Python examples from conductor's docs, so tests can run and type-check them.

A reader copies these examples, so they have to work. Each document becomes one module of Python source.

The documents are the pages and the notebooks. The pages are the README files, ``docs/``, ``skills/``,
``about/llms.txt`` and the problems page. A page's module is its ``python`` blocks, one after another. A notebook's
module is its code cells.

A few documents need some help to run:

- The problems page is taken as ``python -m conductor.about problems`` prints it, with each problem's example.
- A notebook cell may use ``await`` at the top level. Such a cell is wrapped in a coroutine of its own.
- ``llms.txt`` uses conductor's public names without importing them, so its module starts with a star import of
  ``conductor`` and ``conductor.widgets``.

The illustrated pages in ``docs/explainers/`` are HTML, and each shows what its samples print, so they are read
apart, by ``explainer_samples``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from conductor.about import problems_page

ROOT = Path(__file__).parents[2]
#: The module name of the problems page.
PROBLEMS_PAGE = "doc_problems"
_STAR_IMPORTS = (
    "from typing import Annotated\nfrom conductor import *\nfrom conductor.widgets import *\n"
)


def pages() -> dict[str, str]:
    """Returns the Python of each page, keyed by a module name made from the page's path."""
    found: dict[str, str] = {}
    markdown = [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("skills/*/*.md"),
        *ROOT.glob("packages/*/README.md"),
    ]
    for path in sorted(markdown):
        blocks = [
            _doctest_free(block)
            for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        ]
        if blocks:
            found[_module_name(path)] = "\n".join(blocks)
    llms = ROOT / "packages" / "conductor" / "src" / "conductor" / "about" / "llms.txt"
    found[_module_name(llms)] = _STAR_IMPORTS + "\n".join(
        _doctest_free(b) for b in re.findall(r"```python\n(.*?)```", llms.read_text(), re.S)
    )
    found[PROBLEMS_PAGE] = "\n".join(re.findall(r"```python\n(.*?)```", problems_page(), re.S))
    return found


def documents() -> dict[str, str]:
    """Returns the Python of every page and every notebook, keyed by module name."""
    found = pages()
    for path in sorted(ROOT.glob("examples/*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        found[_module_name(path)] = "\n".join(
            _awaitable(source, n) for n, source in enumerate(code)
        )
    return found


def _awaitable(cell: str, number: int) -> str:
    """Wraps a notebook cell that uses ``await`` at the top level in a coroutine, so it is valid Python.

    A notebook allows a top-level ``await``, but a module doesn't. A cell without one is returned as it is.
    """
    if not re.search(r"\bawait\b|\basync (for|with)\b", cell):
        return cell
    return f"async def _cell_{number}() -> None:\n" + "\n".join(
        f"    {line}" if line else "" for line in cell.splitlines()
    )


def _doctest_free(block: str) -> str:
    """Turns a block written as an interactive session (``>>> ...``) into plain code.

    Only the ``>>>`` and ``...`` lines are kept, without their prompts. The printed output is dropped.
    """
    if not block.lstrip().startswith(">>>"):
        return block
    return "\n".join(line[4:] for line in block.splitlines() if line.startswith((">>> ", "... ")))


def _module_name(path: Path) -> str:
    return "doc_" + re.sub(r"\W", "_", str(path.relative_to(ROOT)))


@dataclass
class ExplainerSample:
    """One piece of Python on an illustrated page, and what the page shows it prints.

    Attributes:
        code: The Python, as a reader sees it, or the setup the page hides.
        line: The line of the page the Python starts on.
        shown: Each block of output the page shows after the Python, before the next Python. Empty when it shows
            none.
        in_order: True when a shown block is only some of the printed lines, in order (``data-check="lines"``).
            Otherwise it is all of them.
    """

    code: str
    line: int
    shown: list[str] = field(default_factory=list)
    in_order: bool = False


class _ExplainerReader(HTMLParser):
    """Reads an illustrated page's Python and its shown output, in the order the page has them.

    - ``<pre data-lang="py">`` is a sample. ``data-run="no"`` leaves out one that isn't a whole program.
    - ``<script type="text/x-python">`` is setup a sample needs and the page doesn't show, such as its imports.
    - An element whose class has ``out`` is what the Python before it prints. ``data-check="lines"`` on it says
      it shows only some of the printed lines.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.samples: list[ExplainerSample] = []
        self._open: str | None = None
        self._depth = 0
        self._text: list[str] = []
        self._line = 0
        self._lines_only = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._open:
            self._depth += tag == self._open
            return
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if tag == "pre" and attr.get("data-lang") == "py" and attr.get("data-run") != "no":
            self._start("code", tag)
        elif tag == "script" and attr.get("type") == "text/x-python":
            self._start("code", tag)
        elif "out" in classes:
            self._lines_only = attr.get("data-check") == "lines"
            self._start("out", tag)

    def _start(self, kind: str, tag: str) -> None:
        self._kind, self._open, self._depth, self._text = kind, tag, 1, []
        self._line = self.getpos()[0]

    def handle_endtag(self, tag: str) -> None:
        if tag != self._open:
            return
        self._depth -= 1
        if self._depth:
            return
        text = "".join(self._text).strip("\n")
        self._open = None
        if not text.strip():
            return  # filled in by the page's script
        if self._kind == "code":
            self.samples.append(ExplainerSample(text, self._line))
        elif self.samples:
            self.samples[-1].shown.append(text)
            self.samples[-1].in_order |= self._lines_only
        else:
            raise ValueError(f"line {self._line}: shown output with no Python before it")

    def handle_data(self, data: str) -> None:
        if self._open:
            self._text.append(data)


def explainer_samples() -> dict[str, list[ExplainerSample]]:
    """Returns the Python of each illustrated page in ``docs/explainers/``, keyed by a module name."""
    found: dict[str, list[ExplainerSample]] = {}
    for path in sorted(ROOT.glob("docs/explainers/*.html")):
        reader = _ExplainerReader()
        reader.feed(path.read_text())
        found[_module_name(path)] = reader.samples
    return found
