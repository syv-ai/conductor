"""``Ask``, the exception a node raises when it needs a person to answer.

Most nodes work out their outputs and return them. A node that needs a person raises ``Ask`` instead. Its call
stops and waits, and the person's answer becomes the node's outputs. The class below shows how a node uses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor._internal.base import REQUIRED


class Ask(BaseException):
    """Raised by a node to ask a person for a value.

    The node's return type says what kind of answer it takes. ``-> str`` asks for a line of text, ``-> bool`` for a
    yes or no, ``-> Literal["ida", "bo"]`` for one of those values, and a frozen dataclass for a form with one
    question per field. The person's answer becomes the node's outputs, so the node isn't called again for it.
    Nothing that reads those outputs runs until then.

    A node that may raise ``Ask`` says so with ``asks=True`` on its ``@node``. If it raises ``Ask`` without that,
    the call fails with ``ask_not_declared``.

    Here are three nodes that ask. The first takes any text. The second lets the person pick one of two people,
    named on its return type. The third offers the members of a team that comes in on an input::

        @node("reason", title="Reason", description="Asks why a ticket was closed.", asks=True)
        def reason(ticket: str) -> str:
            raise Ask(f"Why was {ticket} closed?")

        @node("assign", title="Assign", description="Asks who takes a ticket.", asks=True)
        def assign(ticket: str) -> Literal["ida", "bo"]:
            raise Ask(f"Who takes {ticket}?")

        @node("pick", title="Pick", description="Asks which member of a team takes a ticket.", asks=True)
        def pick(ticket: str, team: list[str]) -> str:
            raise Ask(f"Who takes {ticket}?", choices=team)

    Sometimes a node should ask only now and then. Give the question a node of its own, and wire its output to the
    node that needs the answer. For example, ``approval(invoice) -> bool`` returns ``True`` for a small invoice and
    raises ``Ask`` for a large one. Then ``pay(invoice, ok: bool)`` reads ``approval.result``. A host that wants to
    offer the two as one node puts them in a ``subgraph``.

    ``Ask`` is a ``BaseException``, not an ``Exception``. That way an ``except Exception`` in the node doesn't catch
    it by mistake.

    Args:
        prompt: The text the person reads, such as ``"Who takes T-12?"``.
        choices: The values the person may pick from, when they come from the run's data, such as a list on an
            input or in the context. Choices you know when you write the node are its contract, so they go on its
            return type instead: a ``Literal``, or ``Param(choices=)``. Each is a value, or a ``Choice`` to give
            it a title. For a node with one output, pass a list. For a node with several, pass a mapping from
            output name to list, naming only the outputs you want to limit. On a ``Series`` output, the person may
            pick several.
        default: A proposed answer. It has the answer's shape: a value for a node with one output, or a mapping by
            output name for several, naming only the outputs it proposes for. On a node with several outputs, a
            question with a default may be left out of the answer. A node with one output is still answered with a
            value, so its question stays required, and the default is the value a host fills the answer in with.
            Without a default, the only defaults are those of a returned dataclass's fields.

    Attributes:
        prompt: The text the person reads.
        choices: The choices as given, or ``None`` when none were.
        default: The proposed answer as given, or ``REQUIRED`` when none was.
    """

    def __init__(
        self,
        prompt: str,
        *,
        choices: list[Any] | Mapping[str, list[Any]] | None = None,
        default: Any = REQUIRED,
    ) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.choices = choices
        self.default = default
