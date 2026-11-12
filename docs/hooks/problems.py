"""The docs build's own step: the problems page, and the router's problems on the page on serving runs.

Neither is a file in ``docs/``: the problems page is ``python -m conductor.about problems``, printed from the
functions that declare each kind of problem, and a ``<!-- problems: ... -->`` line on any page is replaced by the
kinds of that catalogue. ``mkdocs.yml`` names this file under ``hooks``.
"""

from __future__ import annotations

from typing import Any

from conductor.about import fill_placeholders, problems_page
from mkdocs.structure.files import File, Files


def on_files(files: Files, config: Any) -> Files:
    """Adds ``problems.md``, as ``python -m conductor.about problems`` prints it."""
    files.append(File.generated(config, "problems.md", content=problems_page()))
    return files


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Replaces each placeholder line of a page by what it stands for."""
    return fill_placeholders(markdown)
