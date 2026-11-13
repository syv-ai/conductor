"""``Ask``, which a node raises to ask a person.

``raise Ask("Why was T-12 closed?")`` in ``reason(ticket: str) -> str``, declared with ``asks=True``, stops the call
until a person answers with a ``str``. The package's docstring says how a node asks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor._internal.base import REQUIRED


class Ask(BaseException):
    """Stops a call to ask a person; the return type of the function that raises it is what they answer.

    Nothing that reads the call runs until an answer arrives. A node that raises it in its own function says
    ``asks=True`` on its ``@node``, and the answer is its outputs: the node is not called again for it. A free-text
    question is the plain case, ``-> str`` and ``raise Ask(prompt)``::

        @node("reason", title="Reason", description="Asks why a ticket was closed.", asks=True)
        def reason(ticket: str) -> str:
            raise Ask(f"Why was {ticket} closed?")

        @node("assign", title="Assign", description="Asks who takes a ticket.", asks=True)
        def assign(ticket: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    A raise from a node that does not say ``asks=True`` fails the call as ``ask_not_declared``.

    To ask only sometimes and use the answer in another node, give the question a node of its own and wire its output
    on: ``approval(invoice) -> bool`` returns ``True`` for a small invoice and raises ``Ask`` for a large one, and
    ``pay(invoice, ok: bool)`` reads its ``result``. A host that offers the two as one node makes them a
    ``subgraph``.

    ``Ask`` derives from ``BaseException``, so an ``except Exception`` in the function lets it through.

    Args:
        prompt: What the person reads.
        choices: The values the answer is limited to, known only when the node
            runs, each a value or a ``Choice`` to give it a title. For one
            question (a node with one output), a list; a ``Series`` picks
            several of them. For a node with several
            outputs, which is answered by output name, a mapping by output name
            that names only the outputs it limits. Choices known when the node
            is written go on the type: a ``Literal``, or ``Param(choices=)``.
        default: The proposed answer, in the answer's shape: a value for one
            question, a mapping by output name for several, naming only the
            outputs it proposes for. A question with a default may be left out
            of an answer. Left out, nothing is proposed beyond the defaults of
            a returned dataclass's fields.

    Attributes:
        prompt: What the person reads.
        choices: The choices as given, ``None`` when none were.
        default: The proposed answer as given, ``REQUIRED`` when none was.
    """

    def __init__(self, prompt: str, *, choices: list[Any] | Mapping[str, list[Any]] | None = None, default: Any = REQUIRED) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.choices = choices
        self.default = default
