"""Nodes for regular expressions: ``regex-match``, ``regex-replace`` and ``regex-extract``.

Some patterns make Python's ``re`` run practically forever. ``(a+)+$`` on a long run of a's is one. These nodes use
the ``regex`` package instead, which handles those patterns fine.

Every search also has a timeout, as a second guard. It has to be inside the call: a search that holds the GIL
can't be stopped from outside.

A node fails with one of three codes:

- ``pattern_timeout`` if the search takes longer than ``TIMEOUT``.
- ``invalid_pattern`` if the pattern isn't a valid regular expression.
- ``invalid_replacement`` if a replacement refers to a group the pattern doesn't have.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import regex
from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import Textarea
from regex import Pattern
from regex import error as RegexError

registry = NodeRegistry(category="regex")

#: How many seconds one search or replace may take before the node fails.
TIMEOUT = 2.0


@contextmanager
def _as_node_error(invalid: str = "invalid_pattern", what: str = "pattern") -> Iterator[None]:
    """Turns a timeout or a ``regex.error`` inside the ``with`` block into a ``NodeError``.

    A timeout becomes ``pattern_timeout``. A ``regex.error`` becomes ``invalid``.

    Args:
        invalid: The code to fail with for a ``regex.error``.
        what: What wasn't valid, ``"pattern"`` or ``"replacement"``, for the message.
    """
    try:
        yield
    except TimeoutError:
        raise NodeError(
            "The pattern took too long on this text. Try a simpler pattern.", code="pattern_timeout"
        ) from None
    except RegexError as error:
        raise NodeError(f"The {what} is not valid: {error}.", code=invalid) from None


def _compiled(pattern: str, ignore_case: bool) -> Pattern[str]:
    """Compiles the pattern.

    Raises:
        NodeError: ``invalid_pattern`` if the pattern isn't a valid regular expression.
    """
    with _as_node_error():
        return regex.compile(pattern, flags=regex.IGNORECASE if ignore_case else 0)


@registry.node(
    "regex-match",
    title="Matches pattern",
    description="Tells whether a pattern is found anywhere in the text.",
)
def match(
    text: Annotated[str, Param(widget=Textarea())], pattern: str, ignore_case: bool = False
) -> Annotated[bool, Param(title="Matched")]:
    """Whether the pattern is found anywhere in the text."""
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error():
        return compiled.search(text, timeout=TIMEOUT) is not None


@registry.node(
    "regex-replace",
    title="Replace pattern",
    description="Replaces everything in the text that matches a pattern.",
)
def replace_all(
    text: Annotated[str, Param(widget=Textarea())],
    pattern: str,
    replacement: Annotated[str, Param(title="Replace with")] = "",
    ignore_case: bool = False,
) -> str:
    """Replaces every match of the pattern with ``replacement``.

    ``replacement`` may refer to the pattern's groups, such as ``\\1``.
    """
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error("invalid_replacement", "replacement"):
        return compiled.sub(replacement, text, timeout=TIMEOUT)


@registry.node(
    "regex-extract",
    title="Find pattern",
    description="Finds every piece of the text that matches a pattern, so what follows runs once for each match.",
)
def extract(
    text: Annotated[str, Param(widget=Textarea())], pattern: str, ignore_case: bool = False
) -> Annotated[Series[str], Param(title="Matches")]:
    """Returns every match, as a ``Series``.

    If the pattern has groups, each item is what the first group matched, not the whole match. On ``"a1b22"``, the
    pattern ``\\d+`` gives ``["1", "22"]``, and ``([a-z])\\d`` gives ``["a", "b"]``. A group that matched nothing,
    such as ``(x)?``, gives ``""``.
    """
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error():  # finditer is lazy, so a timeout is raised while the list is built
        return [
            m.groups(default="")[0] if compiled.groups else m.group(0)
            for m in compiled.finditer(text, timeout=TIMEOUT)
        ]
