"""Regular-expression nodes: ``regex-match``, ``regex-replace`` and ``regex-extract``.

They run on the ``regex`` package rather than ``re``: its engine doesn't
backtrack on the patterns that freeze ``re`` (``(a+)+$`` over a long run of
a's), and every call has a timeout as a second guard. A call that holds the GIL
can't be stopped from outside, so the guard is inside it. A pattern that takes
longer than ``TIMEOUT`` fails its node as ``pattern_timeout``, one that
doesn't compile as ``invalid_pattern``, and a replacement that refers to a group
the pattern lacks as ``invalid_replacement``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

import regex
from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import TextInput

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="regex")

#: Seconds one match, replace or search may take before the node fails.
TIMEOUT = 2.0


@contextmanager
def _pattern_faults(invalid: str = "invalid_pattern", what: str = "pattern") -> Iterator[None]:
    """Turns a call that ran out of ``TIMEOUT`` and a ``regex.error`` into the node's own faults.

    Args:
        invalid: The code of a ``regex.error`` raised inside the block.
        what: What the block's ``regex.error`` is about, as its message names it.
    """
    try:
        yield
    except TimeoutError:
        raise NodeError("The pattern took too long.", code="pattern_timeout") from None
    except regex.error as error:
        raise NodeError(f"The {what} is not valid: {error}.", code=invalid) from None


def _compiled(pattern: str, ignore_case: bool) -> regex.Pattern[str]:
    with _pattern_faults():
        return regex.compile(pattern, flags=regex.IGNORECASE if ignore_case else 0)


@registry.node("regex-match")
def match(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[bool, Param(title="Matched")]:
    """Whether the pattern matches anywhere in the text."""
    compiled = _compiled(pattern, ignore_case)
    with _pattern_faults():
        return compiled.search(text, timeout=TIMEOUT) is not None


@registry.node("regex-replace")
def replace_all(
    text: LongText,
    pattern: Line,
    replacement: Annotated[str, Param(title="Replace with", widget=TextInput())] = "",
    ignore_case: Flag = False,
) -> str:
    """Replaces every match of the pattern."""
    compiled = _compiled(pattern, ignore_case)
    with _pattern_faults("invalid_replacement", "replacement"):
        return compiled.sub(replacement, text, timeout=TIMEOUT)


@registry.node("regex-extract")
def extract(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[Series[str], Param(title="Matches")]:
    """Every match, a row each, or the first group of each when the pattern has groups; a group that took no part is ``""``."""
    compiled = _compiled(pattern, ignore_case)
    with _pattern_faults():  # ``finditer`` is lazy and the timeout fires while it is walked
        return [m.groups(default="")[0] if compiled.groups else m.group(0) for m in compiled.finditer(text, timeout=TIMEOUT)]
