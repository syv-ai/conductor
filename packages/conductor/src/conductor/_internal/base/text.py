"""Small pieces of English that messages and reprs are written with."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def comma_separated(items: Iterable[object], empty: str = "none") -> str:
    """The items joined by commas, for a sentence: ``comma_separated(["a", "b"])`` returns ``"a, b"``.

    With no items it returns ``empty``, so "the outputs are none" is written where "the outputs are " would be.
    """
    return ", ".join(str(item) for item in items) or empty


def plural(count: int, word: str) -> str:
    """The count and the word, with an ``s`` unless the count is 1.

    ``plural(1, "node")`` returns ``"1 node"`` and ``plural(0, "node")`` returns ``"0 nodes"``.
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def qualified_name(named: type | Callable[..., object]) -> str:
    """Where a class or a function is written: its module and its name there, ``"my_nodes.text.shout"``."""
    return f"{named.__module__}.{named.__qualname__}"
