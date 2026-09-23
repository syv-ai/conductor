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
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import regex
from conductor.errors import ErrorCause, NodeError
from conductor.metadata import Param, Result
from conductor.series import Series
from conductor.widgets import Switch, Textarea, TextWidget

from conductor_nodes.types import Flag, StdlibNode, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry

TOO_SLOW = "The pattern took too long."


class PatternNode(StdlibNode):
    """The base of the three regex nodes: how long a pattern may run, and the one way they call ``regex``."""

    #: Seconds a single match, replace or search may take before the node fails.
    timeout: ClassVar[float] = 2.0

    def _timed(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """``operation(*args, **kwargs)`` under the timeout; too slow is the node's ``pattern_timeout``."""
        try:
            return operation(*args, timeout=self.timeout, **kwargs)
        except TimeoutError as slow:
            raise NodeError(TOO_SLOW, cause=ErrorCause(code="pattern_timeout", message=TOO_SLOW)) from slow


class Match(PatternNode):
    id = "regex-match"
    title = "Regex Match"
    description = "True if the pattern matches anywhere in the text"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
        ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
    ) -> Annotated[Flag, Result(title="Matched")]:
        flags = regex.IGNORECASE if ignore_case else 0
        return Flag(self._timed(regex.search, pattern, text, flags=flags) is not None)


class ReplaceAll(PatternNode):
    id = "regex-replace"
    title = "Regex Replace"
    description = "Replaces all pattern matches with `replacement`"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
        replacement: Annotated[Text, Param(title="Replace with", widget=TextWidget())] = Text(""),
        ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
    ) -> Annotated[Text, Result(title="Result")]:
        flags = regex.IGNORECASE if ignore_case else 0
        return Text(self._timed(regex.sub, pattern, replacement, text, flags=flags))


class Extract(PatternNode):
    id = "regex-extract"
    title = "Regex Extract"
    description = "Every match (or the first group of each, if the pattern has groups)"
    category = "regex"

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        pattern: Annotated[Text, Param(title="Pattern", widget=TextWidget())],
        ignore_case: Annotated[Flag, Param(title="Ignore case", widget=Switch())] = Flag(False),
    ) -> Annotated[Series[Text], Result(title="Matches")]:
        flags = regex.IGNORECASE if ignore_case else 0
        compiled = regex.compile(pattern, flags=flags)
        if compiled.groups:
            # ``finditer`` is lazy and the timeout fires while it is walked, so
            # the walk happens inside ``_timed``.
            matches = self._timed(lambda timeout: list(compiled.finditer(text, timeout=timeout)))
            return [Text(m.group(1)) for m in matches]
        return [Text(m) for m in self._timed(compiled.findall, text)]


def register(registry: "NodeRegistry") -> None:
    """Register every regex node on the supplied registry."""
    for node_cls in (Match, ReplaceAll, Extract):
        registry.register(node_cls)
