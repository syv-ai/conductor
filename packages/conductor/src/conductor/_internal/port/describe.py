"""``PortDescription``, one input or output of a node as JSON, for an editor to show."""

from __future__ import annotations

from typing import Self

from pydantic import MISSING, JsonValue, model_validator

from conductor._internal.base import ConductorModel
from conductor._internal.problem import conductor_error
from conductor._internal.types import Choice


class PortDescription(ConductorModel):
    """One input or output of a node, as JSON for an editor to show.

    An editor shows a node's inputs and outputs this way. A node that raises ``Ask`` asks its questions this way
    too, one per output. The JSON it writes can be read back into the same description.

    ``required`` says whether the port can be left out, and ``default`` is the value an editor starts it with.
    There are four cases:

    - It has a ``default``, such as ``"default": "a"`` or ``"default": null``, and ``"required": false``. Left out,
      it gets that value, and ``null`` means ``None``.
    - It has a ``default`` and ``"required": true``. It can't be left out, and the default is only the value it
      starts with. That is the one question of a node with one output, whose ``Ask`` proposed a default.
    - It has no ``default`` key and ``"required": true``. It can't be left out, like ``text: str``.
    - It has no ``default`` key and ``"required": false``. It is a branch, an output whose default is ``SKIPPED``.
      Left out, it is skipped.

    In Python, a ``default`` that isn't there is pydantic's ``MISSING``. A description whose parts contradict each
    other is refused: ``loops`` on a port that isn't ``is_series``.

    Attributes:
        name: The parameter's name, or the output's.
        title: The name a person reads, from ``Param(title=)``. ``None`` when the port declares none.
        description: A longer help text.
        is_series: Whether the value is a list the graph can loop over. That is an output declared ``Series[T]``, an
            input that collects every item of a loop, or an output of a subgraph that comes from inside a loop.
        type: The id of the type of one value. For a ``Series``, that is one item's type. For an output of a
            subgraph from inside loops, it is what the node inside returns for one item. A table's type id names its
            columns.
        type_variables: The names of the type variables the hint leaves open, for an edge to fill in. Two ports
            that share a name have the same type once it is filled in.
        json_schema: The JSON schema of one value, the same value ``type`` names. A table's schema lists its
            columns. A port with ``choices`` has their ids as its ``enum``.
        required: Whether the port can't be left out. An input with no default is required. So is an output an
            answer must give: ``-> str`` is required, while ``-> str | None`` may be left out and is then ``None``.
        default: The value an editor starts the port with, as JSON, which the port also gets when it is left out
            and isn't ``required``. ``MISSING``, with no key in the JSON, when there is none.
        choices: The values the port takes, when it names them. They come from a ``Literal``, from
            ``Param(choices=)`` or, on a question, from the ``Ask``. Each is a ``Choice`` with every keyword a host
            gave it.
        widget: The control a person edits the port with, as JSON.
        constant_only: Whether an input takes only a constant. An editor then offers it no edge and no graph input.
        loops: For an output of a subgraph that comes from inside loops: the lists it loops over, outermost first,
            each written as its source inside the subgraph. If the node inside returns a ``Series``, that list comes
            last. Empty for every other port.

            Say a subgraph loops over ``documents``, then over the ``lines`` of each one, and ``shout`` returns one
            ``str`` per line. Its output has ``loops`` ``["documents.result", "lines.result"]``, and its value is a
            list of lists of ``str``. The value has one level of list per entry in ``loops``. With no loops, it is
            one list if ``is_series``, else a single value.

            Two outputs whose loops start the same come from the same loops. A node outside that reads both is called
            once per item, with the two values for that item.
    """

    name: str
    title: str | None = None
    description: str | None = None
    is_series: bool
    type: str
    type_variables: list[str] = []
    json_schema: dict[str, object]
    required: bool
    default: JsonValue | MISSING = MISSING
    choices: tuple[Choice, ...] | None = None
    widget: dict[str, object] | None = None
    constant_only: bool = False
    loops: list[str] = []

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        """Refuses a description whose parts contradict each other, each case with its own error type."""
        if self.loops and not self.is_series:
            raise conductor_error(
                "loops_without_series",
                "a port with loops must be a list (is_series), with one level of list per loop",
            )
        return self
