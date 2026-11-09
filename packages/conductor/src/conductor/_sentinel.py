"""The two values a node's ``run`` may return that are not results: ``SKIPPED`` and ``AskPerson``.

The engine acts on the returned value itself. Nothing on the node class
declares that it may skip or ask.

``SKIPPED`` on an output means the node did not take that branch::

    def run(self, text: Txt, short: Boolean) -> Branches:
        return Branches(if_true=text if short else SKIPPED,
                        if_false=SKIPPED if short else text)

Nothing that reads a skipped output runs. When the node runs once per row
of a series (a value with many rows), the skip applies to that row alone
and the series it produces is sparse.

``AskPerson``, returned where a result would be, means a person must answer
before the graph can continue. The node's declared outputs are what is asked:
an answer becomes the node's outputs and ``run`` is not called again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conductor.widgets import Widget


class _SkippedType:
    """The type of ``SKIPPED``: a falsy singleton whose repr is ``SKIPPED``."""

    _instance: _SkippedType | None = None

    def __new__(cls) -> _SkippedType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIPPED"

    def __bool__(self) -> bool:
        return False


SKIPPED = _SkippedType()


@dataclass(frozen=True)
class AskPerson:
    """What ``run`` returns in place of a result when a person must answer first.

    The questions are the node's outputs, with the name, type and title the
    return annotation already gives them, so ``-> Approved | AskPerson`` is the
    only thing a class says about asking. ``prompt`` is text for the person;
    when ``None`` the host shows the node's own title and description.
    ``defaults`` proposes an answer, by output name, and a question with one may
    be left blank. ``widgets`` insists on a control, by output name; otherwise
    the host picks one from the type::

        def run(self, proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())]) -> Out | AskPerson:
            return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})

    Its sibling is ``SKIPPED``, the other value that is not a result.
    """

    prompt: str | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    widgets: Mapping[str, Widget] = field(default_factory=dict)


def is_skipped(value: Any) -> bool:
    return value is SKIPPED
