"""The two values a node's ``run`` may return that are not results: ``SKIPPED`` and ``Asks``.

The engine acts on the returned value itself. Nothing on the node class
declares that it may skip or ask.

``SKIPPED`` on an output means the node did not take that branch::

    def run(self, text: Txt, short: Boolean) -> Branches:
        return Branches(if_true=text if short else SKIPPED,
                        if_false=SKIPPED if short else text)

Nothing that reads a skipped output runs. When the node runs once per row
of a series (a value with many rows), the skip applies to that row alone
and the series it produces is sparse.

``Asks``, returned where a result would be, means a person must answer
before the graph can continue. It carries the questions, one ``Input`` (the
record that describes one field a node takes) per value the person
supplies — or none, and the node's declared outputs are the questions.
The engine reports them and ends the leg pending — a leg is one call of
``execute``, and a run takes several when a person must answer in
between. The answers reach the next leg as the node's outputs through
``execute(cache=...)``, so ``run`` is not called again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conductor.metadata import Input


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
class Asks:
    """A request for a person's input, returned by ``run`` in place of a result.

    ``questions`` are the fields the person fills in, one ``Input`` per
    output of the node; the engine re-keys them by address (``node.field``)
    when it reports them. ``Asks()`` with no questions asks for the node's
    declared outputs as they are — name, type, title, description, and
    no widget, so the host picks the control from the type. A node writes
    its own questions when it has a default to offer or a control to
    insist on. ``prompt`` is optional text for the person; when ``None``
    the host shows the node's own title and description::

        def run(self, proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | Asks:
            return Asks(questions=(
                Input(name="result", dtype=Txt, title="Answer",
                      widget=Textarea(), default=proposal, optional=True),
            ))

    The engine is the only reader: ``is_asking`` on the returned value
    parks the unit — one run of one node, on one row when the node runs
    per row — through ``Ledger.pend``, and the leg ends ``graph_pending``
    once nothing else can run; ``conductor.run`` returns that ending like any other.
    Its sibling is ``SKIPPED``, the other value that
    is not a result. A ``run`` that may ask says so only in its return
    annotation, ``-> X | Asks``.
    """

    questions: tuple[Input, ...] = ()
    prompt: str | None = None


def is_skipped(value: Any) -> bool:
    return value is SKIPPED


def is_asking(value: Any) -> bool:
    return isinstance(value, Asks)
