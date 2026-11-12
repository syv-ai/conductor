"""Asking a person: ``raise Ask(...)`` in a node or in the function of an ``Asked`` parameter, and the questions a host shows.

A function that raises ``Ask`` is answered with a value of its return type, and that return type says how to ask:
``-> str`` is a line of free text, ``-> bool`` a yes or no, and a frozen dataclass a form of one question per field.
How each question looks (its title, its widget, choices known when the node is written) is the type's, through
``Param``. ``Ask`` adds only what the run knows: the prompt, choices known only when the node runs, and a proposed
answer. Every return type's question is in the reference, ``python -m conductor.about asking-a-person``.

A node asks in one of two ways, and says which where it is declared. Its own function raises ``Ask``, and the answer
is its outputs: ``@node("reason", ..., asks=True)`` on ``def reason(ticket: str) -> str``. Or a parameter is
``Asked``, filled by a function that returns a value or raises ``Ask``, and the node is called once, with the value or
the answer: ``def approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str``, where
``approval(invoice: Invoice) -> bool``.

A host is shown a waiting call as a ``Waiting`` (``run/waiting.py``). The ``Ask`` is checked when it is raised, by the ``Ask`` door (``gate/raised.py``), and an answer before anything runs, by the answer
door (``gate/answer.py``).
"""

from .question import (
    NOTHING_SET,
    Answer,
    Question,
    QuestionOverrides,
    answer_required,
    field_questions,
)
from .type import Ask, Asked
