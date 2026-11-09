"""str-manipulation nodes (``text-uppercase``, ``text-lowercase``, etc.)."""

from __future__ import annotations

from typing import Annotated

from conductor import Param, Result, Series
from conductor.widgets import ListWidget, Switch, Textarea, TextWidget

from conductor_nodes.types import stdlib_node


@stdlib_node("text-uppercase", title="Uppercase", description="Returns the text in uppercase", category="text")
def uppercase(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Annotated[str, Result(title="Uppercased")]:
    return text.upper()


@stdlib_node("text-lowercase", title="Lowercase", description="Returns the text in lowercase", category="text")
def lowercase(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Annotated[str, Result(title="Lowercased")]:
    return text.lower()


@stdlib_node("text-trim", title="Trim", description="Strips leading/trailing whitespace", category="text")
def trim(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Annotated[str, Result(title="Trimmed")]:
    return text.strip()


@stdlib_node("text-length", title="Length", description="Character count of the text", category="text")
def length(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Annotated[float, Result(title="Length")]:
    return float(len(text))


@stdlib_node("text-concat", title="Concat", description="Concatenates two strings with an optional separator", category="text")
def concat(
    a: Annotated[str, Param(title="A", widget=TextWidget())],
    b: Annotated[str, Param(title="B", widget=TextWidget())],
    separator: Annotated[str, Param(title="Separator", widget=TextWidget())] = "",
) -> Annotated[str, Result(title="Result")]:
    return f"{a}{separator}{b}"


@stdlib_node("text-replace", title="Replace", description="Replaces every occurrence of a substring with another", category="text")
def replace_node(
    text: Annotated[str, Param(title="Text", widget=Textarea())],
    needle: Annotated[str, Param(title="Find", widget=TextWidget())],
    replacement: Annotated[str, Param(title="Replace with", widget=TextWidget())] = "",
) -> Annotated[str, Result(title="Result")]:
    return text.replace(needle, replacement)


@stdlib_node("text-contains", title="Contains", description="True if `needle` appears in `text`", category="text")
def contains(
    text: Annotated[str, Param(title="Text", widget=Textarea())],
    needle: Annotated[str, Param(title="Needle", widget=TextWidget())],
    case_sensitive: Annotated[bool, Param(title="Case sensitive", widget=Switch())] = True,
) -> Annotated[bool, Result(title="Contains")]:
    if case_sensitive:
        return bool(needle in text)
    return bool(needle.lower() in text.lower())


@stdlib_node("text-split", title="Split", description="Splits text on a separator into a series of parts", category="text")
def split_node(
    text: Annotated[str, Param(title="Text", widget=Textarea())],
    separator: Annotated[str, Param(title="Separator", widget=TextWidget())] = ",",
) -> Annotated[Series[str], Result(title="Parts")]:
    return [part for part in text.split(separator)]


@stdlib_node("text-join", title="Join", description="Joins a series of strings with a separator", category="text")
def join_node(
    parts: Annotated[
        Series[str], Param(title="Parts", widget=ListWidget())
    ],
    separator: Annotated[str, Param(title="Separator", widget=TextWidget())] = ", ",
) -> Annotated[str, Result(title="Joined")]:
    return separator.join(parts)


@stdlib_node("text-reverse", title="Reverse", description="Returns the text reversed", category="text")
def reverse(text: Annotated[str, Param(title="Text", widget=Textarea())]
) -> Annotated[str, Result(title="Reversed")]:
    return text[::-1]


#: Every node this module defines, in palette order.
NODES = (uppercase, lowercase, trim, length, concat, replace_node, contains, split_node, join_node, reverse)
