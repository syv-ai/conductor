"""Regex nodes (``regex-match``, ``regex-replace``, ``regex-extract``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

from conductor.widgets import Checkbox, Output, Text, Textarea

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register every regex node on the supplied registry."""

    @registry.node(
        "regex-match", version=1, name="Regex Match",
        description="True if the pattern matches anywhere in the text",
    )
    def match(
        text: Annotated[str, Textarea(label="Text")],
        pattern: Annotated[str, Text(label="Pattern")],
        ignore_case: Annotated[bool, Checkbox(label="Ignore case")] = False,
    ) -> Annotated[bool, Output(label="Matched")]:
        flags = re.IGNORECASE if ignore_case else 0
        return re.search(pattern, text, flags=flags) is not None

    @registry.node(
        "regex-replace", version=1, name="Regex Replace",
        description="Replaces all pattern matches with `replacement`",
    )
    def replace(
        text: Annotated[str, Textarea(label="Text")],
        pattern: Annotated[str, Text(label="Pattern")],
        replacement: Annotated[str, Text(label="Replace with")] = "",
        ignore_case: Annotated[bool, Checkbox(label="Ignore case")] = False,
    ) -> Annotated[str, Output(label="Result")]:
        flags = re.IGNORECASE if ignore_case else 0
        return re.sub(pattern, replacement, text, flags=flags)

    @registry.node(
        "regex-extract", version=1, name="Regex Extract",
        description="Returns a list of all matches (or the first group of each, if present)",
    )
    def extract(
        text: Annotated[str, Textarea(label="Text")],
        pattern: Annotated[str, Text(label="Pattern")],
        ignore_case: Annotated[bool, Checkbox(label="Ignore case")] = False,
    ) -> Annotated[list[str], Output(label="Matches")]:
        flags = re.IGNORECASE if ignore_case else 0
        compiled = re.compile(pattern, flags=flags)
        if compiled.groups:
            return [m.group(1) for m in compiled.finditer(text)]
        return compiled.findall(text)
