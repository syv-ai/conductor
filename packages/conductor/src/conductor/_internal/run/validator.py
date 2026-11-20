"""The validators of what a run is given: the values an edge brings, the id of a waiting call, and the node ids to
forget.

``EdgeValidator`` checks a value an edge brings to an input when the graph's types can't vouch for it.
``WaitingIdValidator`` reads the id a host names a waiting call by, in ``run.with_answers``, ``run.with_decline``,
``graph.answers_as_json`` and ``Run.declines_as_json``.
``ForgetValidator`` checks the node ids a host gives ``run.with_forgotten``.

When a graph is compiled, each edge is checked: the output's type must fit the input's type. Often that's enough
to know every value will be fine. Sometimes it isn't, and then the value is checked at run time, before the call.

Say ``letters(text: str) -> int`` is wired to ``small(n: Annotated[int, Param(le=3)])``. Compiling knows an
``int`` reaches ``small.n``, but not whether it is at most 3. So before each call of ``small``, the edge validator
checks ``n`` against its type. If ``letters`` returns 12, that call of ``small`` fails with ``invalid_edge_value``.

An edge from ``shout.result: str`` into ``join.texts: Series[str]`` needs no such check. A node none of whose inputs
needs one has no edge validator at all.

A ``Table[R]`` input that collects rows is the one input always read by the edge validator, since the rows are made
into a table there. Its rows are checked against ``R`` only where an output wired to it declares less than ``R``,
as an item is for a ``Series`` input. Rows from ``-> Series[R]`` aren't checked again, as a list or as a frame with
exactly the polars types ``R``'s own frame has; the table's own checks still hold: its limits, and that a saved run
keeps each cell (``with_rows_checked``).

Compiling checks that each edge's types fit. Which inputs still need a check at run time is worked out once, when
the run blueprint is built, by ``EdgeValidator.of``. The validator is built then, and called before each call of the
node.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, assert_never

from pydantic import StrictStr, TypeAdapter, ValidationError

from conductor._internal.base import (
    SKIPPED,
    ItemIndexes,
    Loc,
    NodePath,
    Validator,
    parse_waiting_id,
)
from conductor._internal.graph import CollectingInput, EdgeInput
from conductor._internal.port import Port
from conductor._internal.problem import InvalidCallError, call_problems, run_problems, shape_error
from conductor._internal.table import is_bare_table_row, with_rows_checked
from conductor._internal.types import Type


def _demands_more(node_input: EdgeInput | CollectingInput) -> bool:
    """Whether an input's type asks for more than the outputs wired to it promise.

    If it does, the node's edge validator checks the input's value before each call.

    Every ``str`` is already a valid ``str | None``, so an edge from ``-> str`` into ``line: str | None`` needs no
    check. An edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does an edge from ``-> Any``.

    An input that collects a loop gets one item of its value from each edge, not the whole value. Then each output
    is compared with the input's item type, which for a table is its row type.
    """
    port, sources = node_input.port, [edge.type for edge in node_input.sources]
    match node_input:
        case EdgeInput():
            return not all(port.whole.holds(source) for source in sources)
        case CollectingInput():
            item = port.type if port.whole.row is None else port.whole.row
            return not all(item.holds(source) for source in sources)
        case _:
            assert_never(node_input)


class EdgeValidator(Validator[dict[str, Any], dict[str, Any]]):
    """A validator for the arguments of a node's call, for the inputs the graph's types can't vouch for and the
    table inputs that collect rows.

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

        The validator checks each input whose type asks for more than the outputs wired to it promise. A table
        input that collects rows is read too, since its rows are made into a table there. Where its row type names
        columns and every output wired to it declares that row type, it is read by its type for rows already that row
        type's values (``with_rows_checked``), so the cells aren't checked again. A bare table's rows are read as they come, since their keys must be text. If no input
        needs any of this, this returns ``None``.

        Args:
            inputs: The node's inputs that get their values through edges, by name.
        """
        read: dict[str, tuple[Port, Type]] = {}
        for name, node_input in inputs.items():
            whole = node_input.port.whole
            if _demands_more(node_input):
                read[name] = (node_input.port, whole)
            elif isinstance(node_input, CollectingInput) and (row := whole.row) is not None:
                read[name] = (
                    node_input.port,
                    whole if is_bare_table_row(row.hint) else Type(with_rows_checked(whole.hint)),
                )
        return cls(read) if read else None

    def __init__(self, inputs: Mapping[str, tuple[Port, Type]]) -> None:
        """Builds the validator for ``inputs``: by name, each input to check and the type its value is checked by,
        which for a table of rows its sources already checked is the table for rows already its row type's values
        (``with_rows_checked``)."""
        self._inputs = tuple(
            (
                name,
                port.whole,
                port.whole.exact_classes,
                checked_by.argument_validator.validate_python,
            )
            for name, (port, checked_by) in inputs.items()
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
                raise InvalidCallError(
                    [run_problems.invalid_edge_value(input=name, **whole.refusal(error))]
                ) from None
        return read


#: Text, as a waiting id must be: a ``str`` or a subclass of it, never a number.
_TEXT = TypeAdapter(StrictStr)


class WaitingIdValidator(Validator[object, tuple[Loc, NodePath, ItemIndexes, str | None]]):
    """A validator for the id a host names a waiting call by: in ``run.with_answers``, ``run.with_decline``,
    ``graph.answers_as_json`` and ``Run.declines_as_json``.

    The id must be text. Then it is split as ``write_waiting_id`` wrote it: ``"review@0/approve@1~3fa2c0d1"`` gives
    the loc ``("review", 0, "approve", 1)``, the node path ``("review", "approve")``, the item indexes ``(0, 1)`` and
    the fingerprint ``"3fa2c0d1"``. An id with no ``~`` part names just the call, and its fingerprint is ``None``.
    Text that isn't such an id, such as ``"approve@01"`` or ``""``, is split as far as ``parse_waiting_id`` reads it;
    the run finds no call waiting under it (``not_waiting``).

    Raises (when called):
        InvalidCallError: ``invalid_argument`` if the id isn't text. The fault's ``loc`` is ``[]`` for the waiting
            call given alone, and ``["[key]"]`` for a key of a mapping, which isn't quoted.
    """

    def __init__(self, *, function: str, argument: str, keyed: bool) -> None:
        """Builds the validator for the ids one function is given.

        Args:
            function: The function, such as ``run.with_decline``, for the refusal's message.
            argument: The argument the id is given in, as the function's signature names it.
            keyed: Whether the ids are the keys of a mapping, rather than one id given alone.
        """
        self._function, self._argument = function, argument
        self._loc = ["[key]"] if keyed else []

    def __call__(self, waiting_id: object) -> tuple[Loc, NodePath, ItemIndexes, str | None]:
        """Returns the call's loc, its node path, its item indexes, and the fingerprint of its question.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            text = _TEXT.validate_python(waiting_id)
        except ValidationError as error:
            errors = [shape_error(fault, self._loc) for fault in error.errors()]
            raise InvalidCallError(
                [
                    call_problems.invalid_argument(
                        self._argument,
                        function=self._function,
                        reason=errors[0]["msg"],
                        errors=errors,
                    )
                ]
            ) from None
        return parse_waiting_id(text)


class ForgetValidator(Validator[tuple[str, ...], tuple[str, ...]]):
    """A validator for the node ids a host gives ``run.with_forgotten``.

    Each must be the id of a node of the graph itself. A node inside a subgraph is forgotten with its subgraph node,
    so ``"loud"`` is taken and ``"loud/shout"`` is refused.

    Raises (when called):
        InvalidCallError: ``node_not_in_graph`` for each id the graph has no node for, in the order given.
    """

    def __init__(self, nodes: Collection[str]) -> None:
        """Builds the validator for a graph whose nodes have the ids ``nodes``, in the graph's order."""
        self._nodes = nodes

    def __call__(self, node_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Returns ``node_ids``, each the id of a node of the graph.

        Raises:
            InvalidCallError: See the class.
        """
        unknown = [node_id for node_id in node_ids if node_id not in self._nodes]
        if unknown:
            raise InvalidCallError(
                [
                    call_problems.node_not_in_graph(node_id, nodes=list(self._nodes))
                    for node_id in unknown
                ]
            )
        return node_ids
