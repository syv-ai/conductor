"""Text-manipulation nodes (``text-uppercase``, ``text-lowercase``, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import List, Switch, Textarea
from conductor.widgets import Text as TextWidget

from conductor_nodes.types import Flag, Number, StdlibNode, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


class Uppercase(StdlibNode):
    id = "text-uppercase"
    title = "Uppercase"
    description = "Returns the text in uppercase"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Annotated[Text, Result(title="Uppercased")]:
        return Text(text.upper())


class Lowercase(StdlibNode):
    id = "text-lowercase"
    title = "Lowercase"
    description = "Returns the text in lowercase"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Annotated[Text, Result(title="Lowercased")]:
        return Text(text.lower())


class Trim(StdlibNode):
    id = "text-trim"
    title = "Trim"
    description = "Strips leading/trailing whitespace"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Annotated[Text, Result(title="Trimmed")]:
        return Text(text.strip())


class Length(StdlibNode):
    id = "text-length"
    title = "Length"
    description = "Character count of the text"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Annotated[Number, Result(title="Length")]:
        return Number(len(text))


class Concat(StdlibNode):
    id = "text-concat"
    title = "Concat"
    description = "Concatenates two strings with an optional separator"
    category = "text"

    def run(
        self,
        a: Annotated[Text, TextWidget(title="A")],
        b: Annotated[Text, TextWidget(title="B")],
        separator: Annotated[Text, TextWidget(title="Separator")] = Text(""),
    ) -> Annotated[Text, Result(title="Result")]:
        return Text(f"{a}{separator}{b}")


class Replace(StdlibNode):
    id = "text-replace"
    title = "Replace"
    description = "Replaces every occurrence of a substring with another"
    category = "text"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        needle: Annotated[Text, TextWidget(title="Find")],
        replacement: Annotated[Text, TextWidget(title="Replace with")] = Text(""),
    ) -> Annotated[Text, Result(title="Result")]:
        return Text(text.replace(needle, replacement))


class Contains(StdlibNode):
    id = "text-contains"
    title = "Contains"
    description = "True if `needle` appears in `text`"
    category = "text"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        needle: Annotated[Text, TextWidget(title="Needle")],
        case_sensitive: Annotated[Flag, Switch(title="Case sensitive")] = Flag(True),
    ) -> Annotated[Flag, Result(title="Contains")]:
        if case_sensitive:
            return Flag(needle in text)
        return Flag(needle.lower() in text.lower())


class Split(StdlibNode):
    id = "text-split"
    title = "Split"
    description = "Splits text on a separator into a series of parts"
    category = "text"

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        separator: Annotated[Text, TextWidget(title="Separator")] = Text(","),
    ) -> Annotated[Series[Text], Result(title="Parts")]:
        return [Text(part) for part in text.split(separator)]


class Join(StdlibNode):
    id = "text-join"
    title = "Join"
    description = "Joins a series of strings with a separator"
    category = "text"

    def run(
        self,
        parts: Annotated[
            Series[Text], List(title="Parts")
        ],
        separator: Annotated[Text, TextWidget(title="Separator")] = Text(", "),
    ) -> Annotated[Text, Result(title="Joined")]:
        return Text(separator.join(parts))


class Reverse(StdlibNode):
    id = "text-reverse"
    title = "Reverse"
    description = "Returns the text reversed"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Text")]
    ) -> Annotated[Text, Result(title="Reversed")]:
        return Text(text[::-1])


NODES = (Uppercase, Lowercase, Trim, Length, Concat, Replace, Contains, Split, Join, Reverse)


def register(registry: "NodeRegistry") -> None:
    """Register every text node on the supplied registry."""
    for node_cls in NODES:
        registry.register(node_cls)
