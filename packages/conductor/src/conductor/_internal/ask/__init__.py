"""Asking a person for a value while a graph runs.

Some nodes can't finish on their own. They need a person to approve something, pick an option or type a reason.
Such a node raises ``Ask`` with a prompt, and its call stops there to wait. When the person answers, the answer
becomes the node's output, just as if the function had returned it.

The node's return type decides what kind of answer it takes. ``-> bool`` asks for a yes or no, ``-> str`` for a
line of text, and a frozen dataclass for a small form with one question per field. Here is a node that asks for a
yes or no::

    @node("approve", title="Approve", description="Asks a person to approve an invoice.", asks=True)
    def approve(invoice: str) -> bool:
        raise Ask(f"Approve {invoice}?")

``asks=True`` tells conductor up front that the node may stop for a person. A host sees each call that waits as a
``Waiting``, and gives the answer with ``run.answer``.

Conductor checks both ends. It checks the ``Ask`` when the node raises it, and the answer when the person gives it.
The full rules are in the reference: ``python -m conductor.about asking-a-person``.
"""

from .question import as_asked
from .type import Ask
from .validator import AnswerValidator, AskValidator, DeclineValidator

__all__ = ["AnswerValidator", "Ask", "AskValidator", "DeclineValidator", "as_asked"]
