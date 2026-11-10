"""Regular-expression nodes: ``regex-match``, ``regex-replace`` and ``regex-extract``.

They run on the ``regex`` package rather than ``re``: its engine doesn't
backtrack on the patterns that freeze ``re`` (``(a+)+$`` over a long run of
a's), and every call has a timeout as a second guard. A call that holds the GIL
can't be stopped from outside, so the guard is inside it. A pattern that takes
longer than ``TIMEOUT`` fails its node as ``pattern_timeout``, one that
doesn't compile as ``invalid_pattern``, and a replacement that refers to a group
the pattern lacks as ``invalid_replacement``.
"""

from collections.abc import Callable
from typing import Annotated

import regex
from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import TextInput

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="regex")

#: Seconds one match, replace or search may take before the node fails.
TIMEOUT = 2.0


def _timed[R](operation: Callable[[float], R]) -> R:
    """``operation(TIMEOUT)``, a call given the timeout, a pattern's faults as the node's own."""
    try:
        return operation(TIMEOUT)
    except TimeoutError:
        raise NodeError("The pattern took too long.", code="pattern_timeout") from None
    except regex.error as error:
        raise NodeError(f"The pattern is not valid: {error}.", code="invalid_pattern") from None


def _replaced(compiled: regex.Pattern[str], replacement: str, text: str, timeout: float) -> str:
    """``compiled.sub(replacement, text)``, a replacement's own fault apart from the pattern's."""
    try:
        return compiled.sub(replacement, text, timeout=timeout)
    except regex.error as error:
        raise NodeError(f"The replacement is not valid: {error}.", code="invalid_replacement") from None


def _flags(ignore_case: bool) -> int:
    return regex.IGNORECASE if ignore_case else 0


@registry.node("regex-match")
def match(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[bool, Param(title="Matched")]:
    """Whether the pattern matches anywhere in the text."""
    return _timed(lambda timeout: regex.search(pattern, text, flags=_flags(ignore_case), timeout=timeout)) is not None


@registry.node("regex-replace")
def replace_all(
    text: LongText,
    pattern: Line,
    replacement: Annotated[str, Param(title="Replace with", widget=TextInput())] = "",
    ignore_case: Flag = False,
) -> str:
    """Replaces every match of the pattern."""
    compiled = _timed(lambda timeout: regex.compile(pattern, flags=_flags(ignore_case)))
    return _timed(lambda timeout: _replaced(compiled, replacement, text, timeout))


@registry.node("regex-extract")
def extract(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[Series[str], Param(title="Matches")]:
    """Every match, a row each, or the first group of each when the pattern has groups; a group that took no part is ``""``."""
    compiled = _timed(lambda timeout: regex.compile(pattern, flags=_flags(ignore_case)))
    # ``finditer`` is lazy and the timeout fires while it is walked, so the walk happens inside ``_timed``.
    found = _timed(lambda timeout: list(compiled.finditer(text, timeout=timeout)))
    return [m.groups(default="")[0] if compiled.groups else m.group(0) for m in found]
