"""Regular-expression nodes: ``regex-match``, ``regex-replace`` and ``regex-extract``.

They run on the ``regex`` package rather than ``re``: its engine doesn't
backtrack on the patterns that freeze ``re`` (``(a+)+$`` over a long run of
a's), and every call has a timeout as a second guard. A call that holds the GIL
can't be stopped from outside, so the guard is inside it. A pattern that takes
longer than ``TIMEOUT`` fails its node as ``pattern_timeout``, and one that
doesn't compile as ``invalid_pattern``.
"""

from collections.abc import Callable
from typing import Annotated, Any

import regex
from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import TextInput

from conductor_nodes._inputs import Flag, Line, LongText

registry = NodeRegistry(category="regex")

#: Seconds one match, replace or search may take before the node fails.
TIMEOUT = 2.0


def _timed(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``operation(*args, **kwargs)`` under the timeout, a pattern's faults as the node's own."""
    try:
        return operation(*args, timeout=TIMEOUT, **kwargs)
    except TimeoutError:
        raise NodeError("The pattern took too long.", code="pattern_timeout") from None
    except regex.error as error:
        raise NodeError(f"The pattern is not valid: {error}.", code="invalid_pattern") from None


def _flags(ignore_case: bool) -> int:
    return regex.IGNORECASE if ignore_case else 0


@registry.node("regex-match")
def match(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[bool, Param(title="Matched")]:
    """Whether the pattern matches anywhere in the text."""
    return _timed(regex.search, pattern, text, flags=_flags(ignore_case)) is not None


@registry.node("regex-replace")
def replace_all(
    text: LongText,
    pattern: Line,
    replacement: Annotated[str, Param(title="Replace with", widget=TextInput())] = "",
    ignore_case: Flag = False,
) -> str:
    """Replaces every match of the pattern."""
    replaced: str = _timed(regex.sub, pattern, replacement, text, flags=_flags(ignore_case))
    return replaced


@registry.node("regex-extract")
def extract(text: LongText, pattern: Line, ignore_case: Flag = False) -> Annotated[Series[str], Param(title="Matches")]:
    """Every match, a row each, or the first group of each when the pattern has groups."""
    compiled = _timed(lambda timeout: regex.compile(pattern, flags=_flags(ignore_case)))
    # ``finditer`` is lazy and the timeout fires while it is walked, so the walk happens inside ``_timed``.
    found = _timed(lambda timeout: list(compiled.finditer(text, timeout=timeout)))
    return [m.group(1) if compiled.groups else m.group(0) for m in found]
