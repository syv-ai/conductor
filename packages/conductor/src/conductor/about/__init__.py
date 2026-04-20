"""Runnable library context — meant to be consumed by AI agents or humans.

Usage (CLI):

    python -m conductor.about                  # full reference
    python -m conductor.about sections         # list section slugs
    python -m conductor.about scheduling       # a single section (prefix match ok)

Usage (programmatic):

    from conductor.about import get_content, list_sections, get_section

The text served is ``conductor/about/llms.txt`` — the canonical reference
text shipped inside the package. Downstream projects get it for free via
pip/uv install without needing repo access.
"""

from __future__ import annotations

import re


def _load_text() -> str:
    """Read the packaged ``llms.txt``."""
    from importlib.resources import files

    resource = files("conductor.about").joinpath("llms.txt")
    if resource.is_file():
        return resource.read_text(encoding="utf-8")

    raise RuntimeError(
        "conductor.about could not locate llms.txt inside the package. "
        "This indicates a broken install; please reinstall conductor."
    )


_HEADING = re.compile(r"^(##+) (.+)$", re.MULTILINE)


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _parse_sections(text: str) -> dict[str, str]:
    """Parse H2 and H3 headings into sections.

    A section runs from its heading up to the next heading of the **same or
    higher level** (fewer ``#``), so "Retry" (H3 inside Core Concepts) ends
    at the next H3 or when Core Concepts ends — not when a later H2 starts.
    """
    matches = list(_HEADING.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        level = len(m.group(1))        # 2 for ##, 3 for ###
        start = m.start()
        end = len(text)
        for nxt in matches[i + 1:]:
            if len(nxt.group(1)) <= level:
                end = nxt.start()
                break
        out[_slug(m.group(2))] = text[start:end].rstrip() + "\n"
    return out


def get_content() -> str:
    """Return the full reference text."""
    return _load_text()


def list_sections() -> list[str]:
    """Return the slugs of every top-level (``##``) section, in document order."""
    return list(_parse_sections(_load_text()).keys())


def get_section(name: str) -> str | None:
    """Return one section by slug. Accepts a prefix/substring match."""
    sections = _parse_sections(_load_text())
    if name in sections:
        return sections[name]
    for slug, body in sections.items():
        if name.lower() in slug:
            return body
    return None


__all__ = ["get_content", "list_sections", "get_section"]
