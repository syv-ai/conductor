"""The edge validator, which checks a value an edge brings to an input when the graph's types can't vouch for it.

When a graph is compiled, each edge is checked: the output's type must fit the input's type. Often that's enough
to know every value will be fine. Sometimes it isn't, and then the value is checked at run time, before the call.

Say ``letters(text: str) -> int`` is wired to ``small(n: Annotated[int, Param(le=3)])``. Compiling knows an
``int`` reaches ``small.n``, but not whether it is at most 3. So before each call of ``small``, the edge validator
checks ``n`` against its type. If ``letters`` returns 12, that call of ``small`` fails with ``invalid_edge_value``.

An edge from ``shout.result: str`` into ``join.texts: Series[str]`` needs no such check. A node none of whose inputs
needs one has no edge validator at all.

Compiling checks that each edge's types fit. Which inputs still need a check at run time is worked out once, when
the run blueprint is built, by ``EdgeValidator.of``. The validator is built then, and called before each call of the
node.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, assert_never

from pydantic import ValidationError

from conductor._internal.base import SKIPPED
from conductor._internal.field import Field
from conductor._internal.graph import CollectingInput, EdgeInput
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.validation import Validator


def _demands_more(node_input: EdgeInput | CollectingInput) -> bool:
    """Whether an input's type asks for more than the outputs wired to it promise.

    If it does, the node's edge validator checks the input's value before each call.

    Every ``str`` is already a valid ``str | None``, so an edge from ``-> str`` into ``line: str | None`` needs no
    check. An edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does an edge from ``-> Any``.

    An input that collects a loop gets one item of its value from each edge, not the whole value. Then each output
    is compared with the input's item type. An input that collects records into a table is always checked, since
    each row has to be checked against the table's columns.
    """
    field, sources = node_input.field, [edge.type for edge in node_input.sources]
    match node_input:
        case EdgeInput():
            return not all(field.whole.holds(source) for source in sources)
        case CollectingInput():
            return field.whole.record is not None or not all(
                field.type.holds(source) for source in sources
            )
        case _:
            assert_never(node_input)


class EdgeValidator(Validator[dict[str, Any], dict[str, Any]]):
    """A validator for the arguments of a node's call, for the inputs the graph's types can't vouch for.

    It checks each of those inputs against the input's type, and returns the arguments with the checked values. A
    value whose class is exactly one of the type's plain classes is kept as it is, and so is ``SKIPPED``.

    ``EdgeValidator.of`` builds one for each node that needs it. That happens once, with the run blueprint, when the
    graph's first run is made or read.

    Raises (when called):
        InvalidCallError: ``invalid_edge_value`` for the first input whose type refuses its value. The problem
            names the input, and quotes nothing of the value.
    """

    @classmethod
    def of(cls, inputs: Mapping[str, EdgeInput | CollectingInput]) -> EdgeValidator | None:
        """Returns the edge validator for a node, or ``None`` if it needs none.

        The validator checks each input whose type asks for more than the outputs wired to it promise. If every
        edge's output already fits its input, nothing needs checking, and this returns ``None``.

        Args:
            inputs: The node's inputs that get their values through edges, by name.
        """
        read = {
            name: node_input.field
            for name, node_input in inputs.items()
            if _demands_more(node_input)
        }
        return cls(read) if read else None

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the validator for ``inputs``, the inputs to check, by name."""
        self._inputs = tuple(
            (
                name,
                field.whole,
                field.whole.exact_classes,
                field.whole.argument_validator.validate_python,
            )
            for name, field in inputs.items()
        )

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Returns a copy of ``arguments``, with each of the validator's inputs checked against its type.

        Raises:
            InvalidCallError: See the class.
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
