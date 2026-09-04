"""Regex nodes (``regex-match``, ``regex-replace``, ``regex-extract``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import Switch, Textarea
from conductor.widgets import Text as TextWidget

from conductor_nodes.types import Flag, StdlibNode, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


class Match(StdlibNode):
    id = "regex-match"
    title = "Regex Match"
    description = "True if the pattern matches anywhere in the text"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        pattern: Annotated[Text, TextWidget(title="Pattern")],
        ignore_case: Annotated[Flag, Switch(title="Ignore case")] = Flag(False),
    ) -> Annotated[Flag, Result(title="Matched")]:
        flags = re.IGNORECASE if ignore_case else 0
        return Flag(re.search(pattern, text, flags=flags) is not None)


class ReplaceAll(StdlibNode):
    id = "regex-replace"
    title = "Regex Replace"
    description = "Replaces all pattern matches with `replacement`"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        pattern: Annotated[Text, TextWidget(title="Pattern")],
        replacement: Annotated[Text, TextWidget(title="Replace with")] = Text(""),
        ignore_case: Annotated[Flag, Switch(title="Ignore case")] = Flag(False),
    ) -> Annotated[Text, Result(title="Result")]:
        flags = re.IGNORECASE if ignore_case else 0
        return Text(re.sub(pattern, replacement, text, flags=flags))


class Extract(StdlibNode):
    id = "regex-extract"
    title = "Regex Extract"
    description = "Every match (or the first group of each, if the pattern has groups)"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        pattern: Annotated[Text, TextWidget(title="Pattern")],
        ignore_case: Annotated[Flag, Switch(title="Ignore case")] = Flag(False),
    ) -> Annotated[Series[Text], Result(title="Matches")]:
        flags = re.IGNORECASE if ignore_case else 0
        compiled = re.compile(pattern, flags=flags)
        if compiled.groups:
            return [Text(m.group(1)) for m in compiled.finditer(text)]
        return [Text(m) for m in compiled.findall(text)]


NODES = (Match, ReplaceAll, Extract)


def register(registry: "NodeRegistry") -> None:
    """Register every regex node on the supplied registry."""
    for node_cls in NODES:
        registry.register(node_cls)
