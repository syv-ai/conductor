"""``Choice``: one option a person can pick, with an id and a title."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from conductor._internal.base import ConductorModel


class Choice(ConductorModel):
    """One option a person may pick: ``Choice("ida", "Ida B.")``.

    A host gives it more fields to carry more about an option, as keywords or on a subclass of its own; they reach
    the question, and a dropdown, as written, and a saved run or a description reads them back.

    Attributes:
        id: The value an answer picking it holds.
        title: What a person reads for it.
    """

    model_config = ConfigDict(extra="allow")

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        """Makes a choice from its id and title given in that order.

        Args:
            id: The value an answer picking it holds.
            title: What a person reads for it.
            **more: The other fields a host gives it.
        """
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)
