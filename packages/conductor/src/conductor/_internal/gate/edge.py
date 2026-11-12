"""The edge door: a value an edge gives an input whose type demands more than the edge's source declares.

Example: ``letters(text: str) -> int`` is wired to ``small(n: Annotated[int, Param(le=3)])``. Compiling knows that
an ``int`` reaches ``small.n``, but not whether it is at most 3, so each call of ``small`` first reads ``n`` by
its type. ``letters`` returning 12 fails that call with ``invalid_edge_value``. An edge from ``shout.result: str``
into ``join.texts: Series[str]`` needs no such read, and a node none of whose inputs needs one has no edge door.

Which inputs are read is decided once, when the graph is compiled (``demands_more``); the door is built then and
called before each call of the node.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from conductor._internal.field import Field
from conductor._internal.gate.refused import RefusedError
from conductor._internal.gate.rules import exact_classes, validator_of
from conductor._internal.markers import SKIPPED
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import Type


def demands_more(field: Field, sources: Iterable[Type], *, per_item: bool) -> bool:
    """Whether the input ``field`` demands more of a value than the types of the outputs its edges read declare,
    so that each call reads the value by the input's type first.

    Every value of ``str`` is a value of ``str | None`` as it is, so an edge from ``-> str`` into ``line: str | None``
    demands nothing more; an edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does one from
    ``-> Any``.

    Args:
        field: The input.
        sources: The types of the outputs its edges read.
        per_item: Whether each source gives the input one item of its value: an input that collects a loop, or
            that takes a list wired to an output of one value. A table input then always reads its rows, since
            each row is checked against the table's columns.
    """
    if per_item:
        return field.whole.record is not None or not all(field.type.holds(source) for source in sources)
    return not all(field.whole.holds(source) for source in sources)


class EdgeDoor:
    """Reads the arguments of a node's call whose inputs ``demands_more`` chose, each by the input's type.

    Built once per node when the graph is compiled. Called with a call's arguments, it returns them with those
    inputs read; a value already exactly of one of the type's plain classes is kept as it is, and so is
    ``SKIPPED``.

    Raises (when called):
        RefusedError: An input's type refuses its value (``invalid_edge_value``, which names the input and quotes
            nothing of the value). The first input refused is reported.
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the door for ``inputs``, the inputs to read, by name."""
        self._inputs = tuple((name, field.whole, exact_classes(field.whole), validator_of(field.whole).validate_python) for name, field in inputs.items())

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Returns a copy of ``arguments`` with each of the door's inputs read by its type.

        Raises:
            RefusedError: An input's type refuses its value (``invalid_edge_value``).
        """
        read = dict(arguments)
        for name, whole, exact, validate in self._inputs:
            value = read[name]
            if value is SKIPPED or type(value) in exact:
                continue
            try:
                read[name] = validate(value, strict=False)
            except ValidationError as error:
                refused = whole.refusal(error.errors(include_url=False, include_input=False))
                raise RefusedError([problem.invalid_edge_value(name=name, reason=refused.reason, errors=refused.errors, column=refused.column)]) from None
        return read
