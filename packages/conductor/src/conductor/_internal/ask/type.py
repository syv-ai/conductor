"""``Ask``, which a node raises to ask a person, and ``Asked``, a parameter filled by a function that may ask.

``raise Ask("Why was T-12 closed?")`` in ``reason(ticket: str) -> str``, declared with ``asks=True``, stops the call
until a person answers with a ``str``. The package's docstring says how a node asks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

    The function of an ``Asked`` parameter raises it too, and the answer is that parameter's value. A raise from a
    node that declares neither fails the call as ``ask_not_declared``.

    ``Ask`` derives from ``BaseException``, so an ``except Exception`` in the function lets it through.

    Args:
        prompt: What the person reads.
        choices: The values the answer is limited to, known only when the node
            runs, each a value or a ``Choice`` to give it a title. For one
            question (an ``Asked`` parameter, or a node with one output), a
            list; a ``Series`` picks several of them. For a node with several
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


@dataclass(frozen=True, slots=True)
class Asked:
    """The metadata of a node's parameter that is filled by a function that returns its value or asks a person for it.

    Example: the parameter ``ok`` of ``approve`` is ``approval``'s return value, or a person's answer when
    ``approval`` raises ``Ask``::

        def approval(invoice: Invoice) -> bool:
            if invoice["amount"] <= 500:
                return True
            raise Ask(f"Pay {invoice['number']}?")

        @node("approve", title="Approve", description="Pays an invoice a person approves.")
        def approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str:
            return "paid" if ok else "held"

    The function's parameters are parameters of the node, by name: its inputs, and its context when the node takes
    one. It returns the parameter's type, and that type is what a person answers. No edge or constant reaches the
    parameter, so it is not an input of the node. A node has at most one, and a node with one does not also declare
    ``asks=True``.

    Attributes:
        function: The function that gives the parameter its value.
    """

    function: Callable[..., Any]
