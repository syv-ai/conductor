"""``Choice``: one option a person can pick, with an id and a title."""

from __future__ import annotations

from typing import Any

from conductor._internal.base import ConductorModel


class Choice(ConductorModel):
    """One option a person may pick: ``Choice("ida", "Ida B.")``.

    A host subclasses it to carry more about an option; the fields it adds reach the question, and a dropdown,
    as written.

    Attributes:
        id: The value an answer picking it holds.
        title: What a person reads for it.
    """

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        """Makes a choice from its id and title given in that order.

        Args:
            id: The value an answer picking it holds.
            title: What a person reads for it.
            **more: The fields a host's subclass adds.
        """
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)
