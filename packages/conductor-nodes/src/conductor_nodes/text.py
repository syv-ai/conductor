"""Text nodes: ``text-uppercase``, ``text-split``, ``text-join`` and the rest."""

from typing import Annotated

from conductor import NodeRegistry, Param, Series
from conductor.widgets import ListInput, TextInput

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="text")


@registry.node("text-uppercase", title="Uppercase", description="Makes every letter in the text a capital letter.")
def uppercase(text: LongText) -> Annotated[str, Param(title="Uppercased")]:
    """Returns the text in uppercase."""
    return text.upper()


@registry.node("text-lowercase", title="Lowercase", description="Makes every letter in the text a small letter.")
def lowercase(text: LongText) -> Annotated[str, Param(title="Lowercased")]:
    """Returns the text in lowercase."""
    return text.lower()


@registry.node("text-trim", title="Trim", description="Removes spaces and blank lines from the start and end of the text.")
def trim(text: LongText) -> Annotated[str, Param(title="Trimmed")]:
    """Strips whitespace from both ends of the text."""
    return text.strip()


@registry.node("text-length", title="Length", description="Counts how many characters the text has.")
def length(text: LongText) -> Annotated[int, Param(title="Length")]:
    """Counts the characters of the text."""
    return len(text)


@registry.node("text-concat", title="Join two texts", description="Puts two texts together, with a separator between them if you want one.")
def concat(a: Line, b: Line, separator: Line = "") -> str:
    """Joins two texts, with a separator between them."""
    return f"{a}{separator}{b}"


@registry.node("text-replace", title="Replace", description="Replaces every place a piece of text appears with another text.")
def replace(
    text: LongText,
    needle: Annotated[str, Param(title="Find", widget=TextInput())],
    replacement: Annotated[str, Param(title="Replace with", widget=TextInput())] = "",
) -> str:
    """Replaces every occurrence of a text with another."""
    return text.replace(needle, replacement)


@registry.node("text-contains", title="Contains", description="Tells whether the text contains another piece of text.")
def contains(text: LongText, needle: Line, case_sensitive: Flag = True) -> Annotated[bool, Param(title="Contains")]:
    """Whether `needle` appears in the text."""
    if case_sensitive:
        return needle in text
    return needle.lower() in text.lower()


@registry.node("text-split", title="Split", description="Splits the text wherever the separator appears, one row for each part.")
def split(text: LongText, separator: Annotated[Line, Param(min_length=1)] = ",") -> Annotated[Series[str], Param(title="Parts")]:
    """Splits the text on a separator, a row per part."""
    return text.split(separator)


@registry.node("text-join", title="Join", description="Joins all the parts into one text, with a separator between them.")
def join(parts: Annotated[Series[str], Param(widget=ListInput())], separator: Line = ", ") -> Annotated[str, Param(title="Joined")]:
    """Joins every part into one text, with a separator between them."""
    return separator.join(parts)


@registry.node("text-reverse", title="Reverse", description="Writes the text backwards.")
def reverse(text: LongText) -> Annotated[str, Param(title="Reversed")]:
    """Returns the text backwards."""
    return text[::-1]
