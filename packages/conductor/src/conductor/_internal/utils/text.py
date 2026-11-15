"""Small helpers for writing English in conductor's messages and reprs, such as ``"a, b"`` or ``"3 nodes"``."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def comma_separated(items: Iterable[object], empty: str = "none") -> str:
    """Joins the items with commas, for use in a sentence.

    ``comma_separated(["a", "b"])`` returns ``"a, b"``. With no items it returns ``empty`` instead, so a message
    reads "the outputs are none" rather than "the outputs are ".
    """
    return ", ".join(str(item) for item in items) or empty


def plural(count: int, word: str) -> str:
    """Returns the count and the word, adding an ``s`` unless the count is 1.

    ``plural(1, "node")`` returns ``"1 node"``, and ``plural(0, "node")`` returns ``"0 nodes"``.
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def qualified_name(named: type | Callable[..., object]) -> str:
    """Returns the full name of a class or function: its module, a dot, and its name inside the module.

    For a function ``shout`` in ``my_nodes/text.py``, that is ``"my_nodes.text.shout"``. A method's name includes
    its class, as in ``"my_nodes.text.Shouter.shout"``.
    """
    return f"{named.__module__}.{named.__qualname__}"
