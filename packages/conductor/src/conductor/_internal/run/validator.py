"""The edge validator: a value an edge gives an input whose type demands more than the edge's source declares.

Example: ``letters(text: str) -> int`` is wired to ``small(n: Annotated[int, Param(le=3)])``. Compiling knows that
an ``int`` reaches ``small.n``, but not whether it is at most 3, so each call of ``small`` first reads ``n`` by
its type. ``letters`` returning 12 fails that call with ``invalid_edge_value``. An edge from ``shout.result: str``
into ``join.texts: Series[str]`` needs no such read, and a node none of whose inputs needs one has no edge validator.

Which inputs are read is decided once, when the run blueprint is built (``EdgeValidator.of``); the validator is built
then and called before each call of the node.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, assert_never

from pydantic import ValidationError

from conductor._internal.base import SKIPPED
from conductor._internal.field import Field
from conductor._internal.graph import CollectingInput, EdgeInput
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.validation import Validator, exact_classes, schema_validator_of


def _demands_more(node_input: EdgeInput | CollectingInput) -> bool:
    """Whether the input ``node_input`` demands more of a value than the types of the outputs its edges read declare,
    so that its node's edge validator reads the value by the input's type before each call.

    Every value of ``str`` is a value of ``str | None`` as it is, so an edge from ``-> str`` into ``line: str | None``
    demands nothing more; an edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does one from
    ``-> Any``. When each source gives the input one item of its value (an input that collects a loop, or that takes
    a list wired to an output of one value), each item is held to the input's item type, and a table input always
    reads its rows, since each row is checked against the table's columns.
    """
    match node_input:
        case EdgeInput(field=field, sources=edges, as_list=as_list):
            sources_are_items = as_list
        case CollectingInput(field=field, sources=edges):
            sources_are_items = True
        case _:
            assert_never(node_input)
    sources = [edge.type for edge in edges]
    if sources_are_items:
        return field.whole.record is not None or not all(
            field.type.holds(source) for source in sources
        )
    return not all(field.whole.holds(source) for source in sources)


class EdgeValidator(Validator[dict[str, Any], dict[str, Any]]):
    """Reads the arguments of a node's call whose inputs demand more than the outputs their edges read declare, each
    by the input's type.

    Built once per node with the run blueprint, on a graph's first run, by ``EdgeValidator.of``. Called with a call's arguments, it returns them with those
    inputs read; a value already exactly of one of the type's plain classes is kept as it is, and so is
    ``SKIPPED``.

    Raises (when called):
        InvalidCallError: An input's type refuses its value (``invalid_edge_value``, which names the input and quotes
            nothing of the value). The first input refused is reported.
    """

    @classmethod
    def of(cls, inputs: Mapping[str, EdgeInput | CollectingInput]) -> EdgeValidator | None:
        """Returns the edge validator of a node whose inputs read through edges are ``inputs``, by name: one that
        reads each input whose type demands more than the outputs its edges read declare. ``None`` when the source of
        every edge already satisfies its input, so the node's calls read nothing first.
        """
        read = {
            name: node_input.field
            for name, node_input in inputs.items()
            if _demands_more(node_input)
        }
        return cls(read) if read else None

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
