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
"""

from __future__ import annotations

import json
import re
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
