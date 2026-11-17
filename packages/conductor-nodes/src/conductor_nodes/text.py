"""Nodes for text: ``text-uppercase``, ``text-split``, ``text-join``, ``text-list`` and the rest."""

from typing import Annotated, Any

from conductor import NodeRegistry, Param, Policy, Series
from conductor.widgets import Textarea

registry = NodeRegistry(category="text")


@registry.node(
    "text-uppercase",
    title="Uppercase",
    description="Makes every letter in the text a capital letter.",
    policy=Policy(costs_nothing=True),
)
def uppercase(
    text: Annotated[str, Param(widget=Textarea())],
) -> Annotated[str, Param(title="Uppercased")]:
    """Returns the text in uppercase."""
    return text.upper()


@registry.node(
    "text-lowercase",
    title="Lowercase",
    description="Makes every letter in the text a small letter.",
    policy=Policy(costs_nothing=True),
)
def lowercase(
    text: Annotated[str, Param(widget=Textarea())],
) -> Annotated[str, Param(title="Lowercased")]:
    """Returns the text in lowercase."""
    return text.lower()


@registry.node(
    "text-trim",
    title="Trim",
    description="Removes spaces and blank lines from the start and end of the text.",
    policy=Policy(costs_nothing=True),
)
def trim(text: Annotated[str, Param(widget=Textarea())]) -> Annotated[str, Param(title="Trimmed")]:
    """Removes spaces, tabs and blank lines from both ends of the text."""
    return text.strip()


@registry.node(
    "text-length",
    title="Length",
    description="Counts how many characters the text has.",
    policy=Policy(costs_nothing=True),
)
def length(text: Annotated[str, Param(widget=Textarea())]) -> Annotated[int, Param(title="Length")]:
    """Counts the characters of the text."""
    return len(text)


@registry.node(
    "text-concat",
    title="Join two texts",
    description="Puts two texts together, with a separator between them if you want one.",
    policy=Policy(costs_nothing=True),
)
def concat(a: str, b: str, separator: str = "") -> str:
    """Puts ``a`` and ``b`` together, with ``separator`` between them."""
    return f"{a}{separator}{b}"


@registry.node(
    "text-replace",
    title="Replace",
    description="Replaces every place a piece of text appears with another text.",
    policy=Policy(costs_nothing=True),
)
def replace(
    text: Annotated[str, Param(widget=Textarea())],
    needle: Annotated[str, Param(title="Find")],
    replacement: Annotated[str, Param(title="Replace with")] = "",
) -> str:
    """Replaces every place ``needle`` appears in the text with ``replacement``."""
    return text.replace(needle, replacement)


@registry.node(
    "text-contains",
    title="Contains",
    description="Tells whether the text contains another piece of text.",
    policy=Policy(costs_nothing=True),
)
def contains(
    text: Annotated[str, Param(widget=Textarea())], needle: str, case_sensitive: bool = True
) -> Annotated[bool, Param(title="Contains")]:
    """Whether ``needle`` appears in the text. Set ``case_sensitive`` to false to ignore upper and lower case."""
    if case_sensitive:
        return needle in text
    return needle.lower() in text.lower()


@registry.node(
    "text-split",
    title="Split text",
    description="Splits the text wherever the separator appears, so what follows runs once for each part.",
    policy=Policy(costs_nothing=True),
)
def split(
    text: Annotated[str, Param(widget=Textarea())],
    separator: Annotated[str, Param(min_length=1)] = "\n",
) -> Annotated[Series[str], Param(title="Parts")]:
    """Splits the text wherever ``separator`` appears, into a ``Series`` of parts.

    The separator is a new line unless you give another. ``"a\\nb"`` gives ``["a", "b"]``, and an empty text gives
    ``[]``.
    """
    return text.split(separator) if text else []


@registry.node(
    "text-list",
    title="Text list",
    description="A list of texts that you enter. Wire a list of items to 'for each' to get the whole list once for each item.",
    policy=Policy(costs_nothing=True),
)
def text_list(
    items: list[str], for_each: Any = None
) -> Annotated[Series[str], Param(title="Texts")]:
    """Returns ``items``, the texts written in the graph, as a ``Series``.

    The node never looks at ``for_each``. It is there so you can put the node inside a loop. Wire a ``Series`` into
    it, and the node runs once for each item, returning the whole list each time. Say ``items`` is ``["q1", "q2"]``
    and two documents are wired to ``for_each``. Then each document gets both questions, ``["q1", "q2"]``.
    """
    return items


@registry.node(
    "text-join",
    title="Join",
    description="Joins all the parts into one text, with a separator between them.",
    policy=Policy(costs_nothing=True),
)
def join(parts: Series[str], separator: str = ", ") -> Annotated[str, Param(title="Joined")]:
    """Joins every part into one text, with ``separator`` between them."""
    return separator.join(parts)


@registry.node(
    "text-reverse",
    title="Reverse",
    description="Writes the text backwards.",
    policy=Policy(costs_nothing=True),
)
def reverse(
    text: Annotated[str, Param(widget=Textarea())],
) -> Annotated[str, Param(title="Reversed")]:
    """Returns the text backwards."""
    return text[::-1]
