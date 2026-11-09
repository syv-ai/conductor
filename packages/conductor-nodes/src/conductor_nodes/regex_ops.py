"""Regex nodes (``regex-match``, ``regex-replace``, ``regex-extract``).

They run on the ``regex`` package rather than ``re``: its engine does not
backtrack on the patterns that freeze ``re`` (``(a+)+$`` over a long run
of a's), and every call carries a timeout as a second guard. A run that
holds the GIL cannot be interrupted from outside, so the guard has to be
inside the call. A pattern that takes longer than ``timeout`` fails its
node with a sentence meant for people, code ``pattern_timeout``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import regex
from conductor import NodeError, Param, Result, Series
from conductor.widgets import Switch, Textarea, TextWidget

from conductor_nodes.types import Flag, Text, stdlib_node

TOO_SLOW = "The pattern took too long."


#: Seconds a single match, replace or search may take before the node fails.
TIMEOUT = 2.0


def _timed(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``operation(*args, **kwargs)`` under the timeout; too slow is the node's ``pattern_timeout``."""
    try:
        return operation(*args, timeout=TIMEOUT, **kwargs)
    except TimeoutError as slow:
        raise NodeError(TOO_SLOW, code="pattern_timeout") from slow


@stdlib_node("regex-match", title="Regex Match", description="True if the pattern matches anywhere in the text", category="regex")
def match(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
    ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
) -> Annotated[Flag, Result(title="Matched")]:
    flags = regex.IGNORECASE if ignore_case else 0
    return Flag(_timed(regex.search, pattern, text, flags=flags) is not None)


@stdlib_node("regex-replace", title="Regex Replace", description="Replaces all pattern matches with `replacement`", category="regex")
def replace_all(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
    replacement: Annotated[Text, Param(title="Replace with", widget=TextWidget())] = Text(""),
    ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
) -> Annotated[Text, Result(title="Result")]:
    flags = regex.IGNORECASE if ignore_case else 0
    return Text(_timed(regex.sub, pattern, replacement, text, flags=flags))


@stdlib_node("regex-extract", title="Regex Extract", description="Every match (or the first group of each, if the pattern has groups)", category="regex")
def extract(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
    ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
) -> Annotated[Series[Text], Result(title="Matches")]:
    flags = regex.IGNORECASE if ignore_case else 0
    compiled = regex.compile(pattern, flags=flags)
    if compiled.groups:
        # ``finditer`` is lazy and the timeout fires while it is walked, so
        # the walk happens inside ``_timed``.
        matches = _timed(lambda timeout: list(compiled.finditer(text, timeout=timeout)))
        return [Text(m.group(1)) for m in matches]
    return [Text(m) for m in _timed(compiled.findall, text)]


#: Every node this module defines, in palette order.
NODES = (match, replace_all, extract)
