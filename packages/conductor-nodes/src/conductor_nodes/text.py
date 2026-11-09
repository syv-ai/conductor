"""Text-manipulation nodes (``text-uppercase``, ``text-lowercase``, etc.)."""

from __future__ import annotations

from typing import Annotated

from conductor import Param, Result, Series
from conductor.widgets import ListWidget, Switch, Textarea, TextWidget

from conductor_nodes.types import Flag, Number, Text, stdlib_node


@stdlib_node("text-uppercase", title="Uppercase", description="Returns the text in uppercase", category="text")
def uppercase(text: Annotated[Text, Param(title="Text", widget=Textarea())]
) -> Annotated[Text, Result(title="Uppercased")]:
    return Text(text.upper())


@stdlib_node("text-lowercase", title="Lowercase", description="Returns the text in lowercase", category="text")
def lowercase(text: Annotated[Text, Param(title="Text", widget=Textarea())]
) -> Annotated[Text, Result(title="Lowercased")]:
    return Text(text.lower())


@stdlib_node("text-trim", title="Trim", description="Strips leading/trailing whitespace", category="text")
def trim(text: Annotated[Text, Param(title="Text", widget=Textarea())]
) -> Annotated[Text, Result(title="Trimmed")]:
    return Text(text.strip())


@stdlib_node("text-length", title="Length", description="Character count of the text", category="text")
def length(text: Annotated[Text, Param(title="Text", widget=Textarea())]
) -> Annotated[Number, Result(title="Length")]:
    return Number(len(text))


@stdlib_node("text-concat", title="Concat", description="Concatenates two strings with an optional separator", category="text")
def concat(
    a: Annotated[Text, Param(title="A", widget=TextWidget())],
    b: Annotated[Text, Param(title="B", widget=TextWidget())],
    separator: Annotated[Text, Param(title="Separator", widget=TextWidget())] = Text(""),
) -> Annotated[Text, Result(title="Result")]:
    return Text(f"{a}{separator}{b}")


@stdlib_node("text-replace", title="Replace", description="Replaces every occurrence of a substring with another", category="text")
def replace_node(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    needle: Annotated[Text, Param(title="Find", widget=TextWidget())],
    replacement: Annotated[Text, Param(title="Replace with", widget=TextWidget())] = Text(""),
) -> Annotated[Text, Result(title="Result")]:
    return Text(text.replace(needle, replacement))


@stdlib_node("text-contains", title="Contains", description="True if `needle` appears in `text`", category="text")
def contains(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    needle: Annotated[Text, Param(title="Needle", widget=TextWidget())],
    case_sensitive: Annotated[Flag, Param(title="Case sensitive", widget=Switch())] = Flag(True),
) -> Annotated[Flag, Result(title="Contains")]:
    if case_sensitive:
        return Flag(needle in text)
    return Flag(needle.lower() in text.lower())


@stdlib_node("text-split", title="Split", description="Splits text on a separator into a series of parts", category="text")
def split_node(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    separator: Annotated[Text, Param(title="Separator", widget=TextWidget())] = Text(","),
) -> Annotated[Series[Text], Result(title="Parts")]:
    return [Text(part) for part in text.split(separator)]


@stdlib_node("text-join", title="Join", description="Joins a series of strings with a separator", category="text")
def join_node(
    parts: Annotated[
        Series[Text], Param(title="Parts", widget=ListWidget())
    ],
    separator: Annotated[Text, Param(title="Separator", widget=TextWidget())] = Text(", "),
) -> Annotated[Text, Result(title="Joined")]:
    return Text(separator.join(parts))


@stdlib_node("text-reverse", title="Reverse", description="Returns the text reversed", category="text")
def reverse(text: Annotated[Text, Param(title="Text", widget=Textarea())]
) -> Annotated[Text, Result(title="Reversed")]:
    return Text(text[::-1])


#: Every node this module defines, in palette order.
NODES = (uppercase, lowercase, trim, length, concat, replace_node, contains, split_node, join_node, reverse)
