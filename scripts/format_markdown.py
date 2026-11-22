#!/usr/bin/env python3
"""Format Markdown files with a fixed parser and a registry of rules.

The parser splits a file into blocks once; each registered rule rewrites the
block list in turn. A new rule is one ``@register`` class: the parser, the
runner and the other rules stay as they are.

A rewrite that would change the file's word sequence is refused, so a rule
can only move whitespace, never text.

    python3 scripts/format_markdown.py FILE...          # rewrite in place
    python3 scripts/format_markdown.py --check FILE...  # list, change nothing
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TypeVar


class Kind(Enum):
    BLANK = "blank"
    FRONT_MATTER = "front_matter"
    FENCED_CODE = "fenced_code"
    INDENTED_CODE = "indented_code"
    TABLE = "table"
    HEADING = "heading"
    THEMATIC_BREAK = "thematic_break"
    HTML = "html"
    BLOCK_QUOTE = "block_quote"
    LINK_DEFINITION = "link_definition"
    ADMONITION_MARKER = "admonition_marker"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True)
class Block:
    kind: Kind
    lines: tuple[str, ...]


FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])(\s+|$)")
ATX_HEADING = re.compile(r"^\s*#{1,6}(\s|$)")
SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
THEMATIC_BREAK = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
LINK_DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s")
ADMONITION = re.compile(r"^\s*:::")
BLOCK_QUOTE = re.compile(r"^\s*>")
HTML_START = re.compile(r"^\s*<")


def _indent(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip())


def _is_table_start(lines: list[str], i: int) -> bool:
    if lines[i].lstrip().startswith("|"):
        return True
    return (
        "|" in lines[i]
        and i + 1 < len(lines)
        and "-" in lines[i + 1]
        and TABLE_DELIMITER.match(lines[i + 1]) is not None
    )


def _starts_block(lines: list[str], i: int, in_paragraph: bool) -> bool:
    """Whether line ``i`` opens a block of its own rather than continuing text."""
    line = lines[i]
    if not line.strip():
        return True
    if (
        FENCE.match(line)
        or ATX_HEADING.match(line)
        or THEMATIC_BREAK.match(line)
        or BLOCK_QUOTE.match(line)
        or ADMONITION.match(line)
        or HTML_START.match(line)
        or LINK_DEFINITION.match(line)
        or _is_table_start(lines, i)
    ):
        return True
    marker = LIST_MARKER.match(line)
    if marker is None:
        return False
    # CommonMark: only "1." starts an ordered list mid-paragraph, so a wrapped
    # line beginning "2026. " stays prose.
    number = marker.group(2)[:-1]
    return not (in_paragraph and number.isdigit() and number != "1")


def parse(text: str) -> list[Block]:
    """Split Markdown into blocks. Joining every block's lines gives ``text`` back."""
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    # Column where the innermost open list item's text starts; indented code
    # inside an item sits four columns past it.
    list_content_column: int | None = None
    previous_blank = True

    def take(kind: Kind, end: int) -> None:
        nonlocal i
        blocks.append(Block(kind, tuple(lines[i:end])))
        i = end

    if lines[0].strip() == "---":
        end = next((j for j in range(1, len(lines)) if lines[j].strip() == "---"), None)
        if end is not None:
            take(Kind.FRONT_MATTER, end + 1)

    while i < len(lines):
        line = lines[i]
        indent = _indent(line)
        blank_before = previous_blank
        previous_blank = not line.strip()

        if not line.strip():
            take(Kind.BLANK, i + 1)
            continue
        if list_content_column is not None and indent == 0 and blank_before:
            list_content_column = None

        code_column = (list_content_column or 0) + 4
        if blank_before and indent >= code_column:
            end = i + 1
            while end < len(lines) and (
                not lines[end].strip() or _indent(lines[end]) >= code_column
            ):
                end += 1
            while end > i + 1 and not lines[end - 1].strip():
                end -= 1  # trailing blanks belong to what follows
            take(Kind.INDENTED_CODE, end)
            continue

        fence = FENCE.match(line)
        if fence:
            char, length = fence.group(1)[0], len(fence.group(1))
            closing = re.compile(rf"^\s*{re.escape(char)}{{{length},}}\s*$")
            end = next(
                (j for j in range(i + 1, len(lines)) if closing.match(lines[j])),
                len(lines) - 1,
            )
            take(Kind.FENCED_CODE, end + 1)
            continue

        if line.lstrip().startswith("<!--"):
            end = next((j for j in range(i, len(lines)) if "-->" in lines[j]), len(lines) - 1)
            take(Kind.HTML, end + 1)
            continue

        single_line_kinds = (
            (ATX_HEADING, Kind.HEADING),
            (THEMATIC_BREAK, Kind.THEMATIC_BREAK),
            (ADMONITION, Kind.ADMONITION_MARKER),
            (LINK_DEFINITION, Kind.LINK_DEFINITION),
        )
        kind = next((k for pattern, k in single_line_kinds if pattern.match(line)), None)
        if kind is not None:
            take(kind, i + 1)
            continue

        # These run to the next blank line.
        if HTML_START.match(line) or BLOCK_QUOTE.match(line) or _is_table_start(lines, i):
            if HTML_START.match(line):
                kind = Kind.HTML
            elif BLOCK_QUOTE.match(line):
                kind = Kind.BLOCK_QUOTE
            else:
                kind = Kind.TABLE
            end = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
            take(kind, end)
            continue

        marker = LIST_MARKER.match(line)
        if marker:
            list_content_column = len(marker.group(0).expandtabs(4))
            end = i + 1
            while end < len(lines) and not _starts_block(lines, end, in_paragraph=False):
                end += 1
            take(Kind.LIST_ITEM, end)
            continue

        end = i + 1
        while end < len(lines) and not _starts_block(lines, end, in_paragraph=True):
            if SETEXT_UNDERLINE.match(lines[end]):
                break
            end += 1
        if end < len(lines) and SETEXT_UNDERLINE.match(lines[end]):
            take(Kind.HEADING, end + 1)  # setext: the underline makes it all a heading
            continue
        take(Kind.PARAGRAPH, end)
    return blocks


def render(blocks: list[Block]) -> str:
    return "\n".join(line for block in blocks for line in block.lines)


class Rule(ABC):
    """One formatting rule: takes the file's blocks, returns them rewritten."""

    @abstractmethod
    def apply(self, blocks: list[Block]) -> list[Block]: ...


RULES: list[Rule] = []


R = TypeVar("R", bound=type[Rule])


def register(rule_class: R) -> R:
    """Add a rule to the run; rules run in the order they are registered."""
    RULES.append(rule_class())
    return rule_class


def _is_hard_break(line: str) -> bool:
    return line.endswith("  ") or line.endswith("\\")


@register
class UnwrapProse(Rule):
    """One line per paragraph and per list item: wrapped lines render jumbled
    on screens of another width. A hard break (two trailing spaces or a
    backslash) is kept."""

    def apply(self, blocks: list[Block]) -> list[Block]:
        return [
            replace(block, lines=self._join(block.lines))
            if block.kind in (Kind.PARAGRAPH, Kind.LIST_ITEM)
            else block
            for block in blocks
        ]

    @staticmethod
    def _join(lines: tuple[str, ...]) -> tuple[str, ...]:
        joined = [lines[0]]
        for line in lines[1:]:
            if _is_hard_break(joined[-1]):
                joined.append(line)
            else:
                joined[-1] = f"{joined[-1].rstrip()} {line.strip()}"
        return tuple(joined)


def format_text(text: str) -> str:
    blocks = parse(text)
    for rule in RULES:
        blocks = rule.apply(blocks)
    return render(blocks)


class WordsChangedError(Exception):
    """A rewrite would add, drop or reorder words."""


def format_file(path: Path, check: bool) -> bool:
    """Format one file; return whether it changed (or would, with ``check``)."""
    with path.open(encoding="utf-8", newline="") as f:
        old = f.read()
    new = format_text(old)
    if new.split() != old.split():
        raise WordsChangedError(str(path))
    if new == old:
        return False
    if not check:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(new)
    return True


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [Path(arg) for arg in argv if arg != "--check"]
    changed: list[Path] = []
    refused: list[Path] = []
    for path in paths:
        try:
            if format_file(path, check):
                changed.append(path)
        except WordsChangedError:
            refused.append(path)

    for path in refused:
        print(f"{path}: formatting would change its words; left as it was.")
    if changed:
        verb = "would reformat" if check else "reformatted"
        print(f"Markdown formatting {verb} {len(changed)} file(s):")
        for path in changed:
            print(f"  {path}")
        if not check:
            print("Stage them again and commit.")
    return 1 if changed or refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
