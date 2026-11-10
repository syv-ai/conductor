"""Conductor's reference, shipped in the package: every rule on one page, ``conductor/about/llms.txt``.

From a shell::

    python -m conductor.about                  # the whole reference
    python -m conductor.about sections         # the section slugs
    python -m conductor.about rows             # one section, by slug or a part of one

From Python::

    from conductor.about import get_content, list_sections, get_section
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


def _parse_sections(text: str) -> dict[str, str]:
    """Each ``##`` and ``###`` section by its slug.

    A section runs from its heading to the next heading of the same level or
    a higher one (fewer ``#``), so "Retries and Timeouts", a ``###`` inside
    "Core Concepts", ends at the next ``###`` or where "Core Concepts" ends.
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
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
        out[slug] = text[start:end].rstrip() + "\n"
    return out


def get_content() -> str:
    """The whole reference."""
    return _load_text()


def list_sections() -> list[str]:
    """The slug of every ``##`` and ``###`` section, in the order the reference has them."""
    return list(_parse_sections(_load_text()).keys())


def get_section(name: str) -> str | None:
    """One section by its slug, or the first whose slug contains ``name``; ``None`` when none does."""
    sections = _parse_sections(_load_text())
    if name in sections:
        return sections[name]
    for slug, body in sections.items():
        if name.lower() in slug:
            return body
    return None


__all__ = ["get_content", "list_sections", "get_section"]
