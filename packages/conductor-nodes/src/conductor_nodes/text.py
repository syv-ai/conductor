"""Text-manipulation nodes (``text-uppercase``, ``text-lowercase``, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.widgets import Checkbox, Output, Text, Textarea

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register every text node on the supplied registry."""

    @registry.node("text-uppercase", version=1, name="Uppercase", description="Returns the text in uppercase")
    def uppercase(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[str, Output(label="Uppercased")]:
        return text.upper()

    @registry.node("text-lowercase", version=1, name="Lowercase", description="Returns the text in lowercase")
    def lowercase(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[str, Output(label="Lowercased")]:
        return text.lower()

    @registry.node("text-trim", version=1, name="Trim", description="Strips leading/trailing whitespace")
    def trim(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[str, Output(label="Trimmed")]:
        return text.strip()

    @registry.node("text-length", version=1, name="Length", description="Character count of the text")
    def length(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[int, Output(label="Length")]:
        return len(text)

    @registry.node("text-concat", version=1, name="Concat", description="Concatenates two strings with an optional separator")
    def concat(
        a: Annotated[str, Text(label="A")],
        b: Annotated[str, Text(label="B")],
        separator: Annotated[str, Text(label="Separator")] = "",
    ) -> Annotated[str, Output(label="Result")]:
        return f"{a}{separator}{b}"

    @registry.node(
        "text-replace", version=1, name="Replace",
        description="Replaces every occurrence of a substring with another",
    )
    def replace(
        text: Annotated[str, Textarea(label="Text")],
        needle: Annotated[str, Text(label="Find")],
        replacement: Annotated[str, Text(label="Replace with")] = "",
    ) -> Annotated[str, Output(label="Result")]:
        return text.replace(needle, replacement)

    @registry.node("text-contains", version=1, name="Contains", description="True if `needle` appears in `text`")
    def contains(
        text: Annotated[str, Textarea(label="Text")],
        needle: Annotated[str, Text(label="Needle")],
        case_sensitive: Annotated[bool, Checkbox(label="Case sensitive")] = True,
    ) -> Annotated[bool, Output(label="Contains")]:
        if case_sensitive:
            return needle in text
        return needle.lower() in text.lower()

    @registry.node("text-split", version=1, name="Split", description="Splits text on a separator into a list")
    def split(
        text: Annotated[str, Textarea(label="Text")],
        separator: Annotated[str, Text(label="Separator")] = ",",
    ) -> Annotated[list[str], Output(label="Parts")]:
        return text.split(separator)

    @registry.node(
        "text-join", version=1, name="Join",
        description="Joins a list of strings with a separator",
    )
    def join(
        parts: Annotated[list[str], Text(label="Parts")],
        separator: Annotated[str, Text(label="Separator")] = ", ",
    ) -> Annotated[str, Output(label="Joined")]:
        return separator.join(str(p) for p in parts)

    @registry.node("text-reverse", version=1, name="Reverse", description="Returns the text reversed")
    def reverse(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[str, Output(label="Reversed")]:
        return text[::-1]
