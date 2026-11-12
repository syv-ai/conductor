"""The Python a reader copies from conductor's documents, each document as one module, for the tests that check it.

The documents are the README files, ``docs/``, ``skills/``, ``about/llms.txt`` and the problems page (the pages), and the
notebooks. The problems page is read as ``python -m conductor.about problems`` prints it, each problem's example in it.
Each page is its ``python`` blocks in order; a notebook is its code cells, a cell that awaits at the top read as a
coroutine. ``llms.txt`` names conductor's public names without importing them, so it is read after a star import of
``conductor`` and ``conductor.widgets``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from conductor.about import problems_page

ROOT = Path(__file__).parents[2]
#: The module name the problems page runs as.
PROBLEMS_PAGE = "doc_problems"
_STAR_IMPORTS = "from typing import Annotated\nfrom conductor import *\nfrom conductor.widgets import *\n"


def pages() -> dict[str, str]:
    """Each page's Python as one module, by a module name made from its path."""
    found: dict[str, str] = {}
    markdown = [*ROOT.glob("*.md"), *ROOT.glob("docs/*.md"), *ROOT.glob("skills/*/*.md"), *ROOT.glob("packages/*/README.md")]
    for path in sorted(markdown):
        blocks = [_doctest_free(block) for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.S)]
        if blocks:
            found[_module_name(path)] = "\n".join(blocks)
    llms = ROOT / "packages" / "conductor" / "src" / "conductor" / "about" / "llms.txt"
    found[_module_name(llms)] = _STAR_IMPORTS + "\n".join(_doctest_free(b) for b in re.findall(r"```python\n(.*?)```", llms.read_text(), re.S))
    found[PROBLEMS_PAGE] = "\n".join(re.findall(r"```python\n(.*?)```", problems_page(), re.S))
    return found


def documents() -> dict[str, str]:
    """The pages and the notebooks, each as one module."""
    found = pages()
    for path in sorted(ROOT.glob("examples/*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        found[_module_name(path)] = "\n".join(_awaitable(source, n) for n, source in enumerate(code))
    return found


def _awaitable(cell: str, number: int) -> str:
    """A notebook cell that awaits at the top, as a notebook allows, as the body of a coroutine of its own."""
    if not re.search(r"\bawait\b|\basync (for|with)\b", cell):
        return cell
    return f"async def _cell_{number}() -> None:\n" + "\n".join(f"    {line}" if line else "" for line in cell.splitlines())


def _doctest_free(block: str) -> str:
    """A block written as a session (``>>> ...``) as the code it runs."""
    if not block.lstrip().startswith(">>>"):
        return block
    return "\n".join(line[4:] for line in block.splitlines() if line.startswith((">>> ", "... ")))


def _module_name(path: Path) -> str:
    return "doc_" + re.sub(r"\W", "_", str(path.relative_to(ROOT)))
