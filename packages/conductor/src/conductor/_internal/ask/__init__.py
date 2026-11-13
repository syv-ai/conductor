"""Asking a person: ``raise Ask(...)`` in a node, and the questions a host shows.

A node that raises ``Ask`` is answered with a value of its return type, and that return type says how to ask:
``-> str`` is a line of free text, ``-> bool`` a yes or no, and a frozen dataclass a form of one question per field.
How each question looks (its title, its widget, choices known when the node is written) is the type's, through
``Param``. ``Ask`` adds only what the run knows: the prompt, choices known only when the node runs, and a proposed
answer. Every return type's question is in the reference, ``python -m conductor.about asking-a-person``.

A node that may ask says so where it is declared, and the answer is its outputs: ``@node("reason", ..., asks=True)``
on ``def reason(ticket: str) -> str``. A node that needs a person's answer to go on reads it from such a node's
output, as it reads any other value.

A host is shown a waiting call as a ``Waiting`` (``run/waiting.py``). The ``Ask`` is checked when it is raised, by the ``Ask`` door (``gate/raised.py``), and an answer before anything runs, by the answer
door (``gate/answer.py``).
"""

from .question import (
    Question,
    answer_required,
)
from .type import Ask
