"""The edge validator: a value an edge gives an input whose type demands more than the edge's source declares.

Example: ``letters(text: str) -> int`` is wired to ``small(n: Annotated[int, Param(le=3)])``. Compiling knows that
an ``int`` reaches ``small.n``, but not whether it is at most 3, so each call of ``small`` first reads ``n`` by
its type. ``letters`` returning 12 fails that call with ``invalid_edge_value``. An edge from ``shout.result: str``
into ``join.texts: Series[str]`` needs no such read, and a node none of whose inputs needs one has no edge validator.

Which inputs are read is decided once, when the run blueprint is built (``demands_more``); the validator is built then
and called before each call of the node.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from conductor._internal.base import SKIPPED
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.types import Type
from conductor._internal.validation import exact_classes, schema_validator_of


def demands_more(field: Field, sources: Iterable[Type], *, sources_are_items: bool) -> bool:
    """Whether the input ``field`` demands more of a value than the types of the outputs its edges read declare,
    so that each call reads the value by the input's type first.

    Every value of ``str`` is a value of ``str | None`` as it is, so an edge from ``-> str`` into ``line: str | None``
    demands nothing more; an edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does one from
    ``-> Any``.

    Args:
        field: The input.
        sources: The types of the outputs its edges read.
        sources_are_items: Whether each source gives the input one item of its value: an input that collects a loop, or
            that takes a list wired to an output of one value. A table input then always reads its rows, since
            each row is checked against the table's columns.
    """
    if sources_are_items:
        return field.whole.record is not None or not all(
            field.type.holds(source) for source in sources
        )
    return not all(field.whole.holds(source) for source in sources)


class EdgeValidator:
    """Reads the arguments of a node's call whose inputs ``demands_more`` chose, each by the input's type.

    Built once per node when the graph is compiled. Called with a call's arguments, it returns them with those
    inputs read; a value already exactly of one of the type's plain classes is kept as it is, and so is
    ``SKIPPED``.

    Raises (when called):
        InvalidCallError: An input's type refuses its value (``invalid_edge_value``, which names the input and quotes
            nothing of the value). The first input refused is reported.
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the validator for ``inputs``, the inputs to read, by name."""
        self._inputs = tuple(
            (
                name,
                field.whole,
                exact_classes(field.whole),
                schema_validator_of(field.whole).validate_python,
            )
            for name, field in inputs.items()
        )

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Returns a copy of ``arguments`` with each of the validator's inputs read by its type.

        Raises:
            InvalidCallError: An input's type refuses its value (``invalid_edge_value``).
        """
        read = dict(arguments)
        for name, whole, exact, validate in self._inputs:
            value = read[name]
            if value is SKIPPED or type(value) in exact:
                continue
            try:
                read[name] = validate(value, strict=False)
            except ValidationError as error:
                refused = whole.refusal(error)
                raise InvalidCallError(
                    [
                        run_problems.invalid_edge_value(
                            name=name,
                            reason=refused.reason,
                            errors=refused.errors,
                            column=refused.column,
                        )
                    ]
                ) from None
        return read
