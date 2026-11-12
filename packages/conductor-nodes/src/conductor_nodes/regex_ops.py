"""Regular-expression nodes: ``regex-match``, ``regex-replace`` and ``regex-extract``.

They run on the ``regex`` package rather than ``re``: its engine doesn't
backtrack on the patterns that freeze ``re`` (``(a+)+$`` over a long run of
a's), and every call has a timeout as a second guard. A call that holds the GIL
can't be stopped from outside, so the guard is inside it. A call fails with the
code ``pattern_timeout`` when its pattern takes longer than ``TIMEOUT``,
``invalid_pattern`` when the pattern doesn't compile, and ``invalid_replacement``
when a replacement refers to a group the pattern doesn't have.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import regex
from conductor import NodeError, NodeRegistry, Param, Series
from regex import Pattern
from regex import error as RegexError

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="regex")

#: Seconds one match, replace or search may take before the node fails.
TIMEOUT = 2.0


@contextmanager
def _as_node_error(invalid: str = "invalid_pattern", what: str = "pattern") -> Iterator[None]:
    """Turns a ``TimeoutError`` or a ``regex.error`` raised inside the ``with`` block into a ``NodeError``.

    Args:
        invalid: The code of the ``NodeError`` for a ``regex.error``.
        what: The word for what is not valid in its message, ``"pattern"`` or ``"replacement"``.
    """
    try:
        yield
    except TimeoutError:
        raise NodeError("The pattern took too long.", code="pattern_timeout") from None
    except RegexError as error:
        raise NodeError(f"The {what} is not valid: {error}.", code=invalid) from None


def _compiled(pattern: str, ignore_case: bool) -> Pattern[str]:
    """Returns the compiled pattern; a pattern that doesn't compile raises ``NodeError`` (``invalid_pattern``)."""
    with _as_node_error():
        return regex.compile(pattern, flags=regex.IGNORECASE if ignore_case else 0)


@registry.node("regex-match", title="Matches pattern", description="Tells whether a pattern is found anywhere in the text.")
def match(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[bool, Param(title="Matched")]:
    """Whether the pattern matches anywhere in the text."""
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error():
        return compiled.search(text, timeout=TIMEOUT) is not None


@registry.node("regex-replace", title="Replace pattern", description="Replaces everything in the text that matches a pattern.")
def replace_all(
    text: LongText,
    pattern: Line,
    replacement: Annotated[Line, Param(title="Replace with")] = "",
    ignore_case: Flag = False,
) -> str:
    """Replaces every match of the pattern."""
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error("invalid_replacement", "replacement"):
        return compiled.sub(replacement, text, timeout=TIMEOUT)


@registry.node("regex-extract", title="Find pattern", description="Finds every piece of the text that matches a pattern, one item for each.")
def extract(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[Series[str], Param(title="Matches")]:
    """Returns every match as a ``Series``, or the first group of each match when the pattern has groups.

    ``"a1b22"`` with ``\\d+`` gives ``["1", "22"]``; with ``([a-z])\\d`` it gives ``["a", "b"]``. A group that did not
    match, such as ``(x)?``, gives ``""``.
    """
    compiled = _compiled(pattern, ignore_case)
    with _as_node_error():  # ``finditer`` is lazy, so the timeout is raised while the list is built
        return [m.groups(default="")[0] if compiled.groups else m.group(0) for m in compiled.finditer(text, timeout=TIMEOUT)]
