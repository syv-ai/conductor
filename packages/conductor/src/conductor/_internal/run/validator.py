"""The validators of what a run is given: the values an edge brings, the id of a waiting call, and the node ids to
forget.

``EdgeValidator`` checks a value an edge brings to an input when the graph's types can't vouch for it.
``WaitingIdValidator`` reads the id a host names a waiting call by in ``run.with_answers`` and ``run.with_decline``.
``ForgetValidator`` checks the node ids a host gives ``run.with_forgotten``.

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

from collections.abc import Callable, Collection, Mapping
from typing import Any, assert_never

from pydantic import ValidationError

from conductor._internal.base import (
    SKIPPED,
    ItemIndexes,
    Loc,
    NodePath,
    parse_waiting_id,
    write_waiting_id,
)
from conductor._internal.field import Field
from conductor._internal.graph import CollectingInput, EdgeInput
from conductor._internal.problem import InvalidCallError, call_problems, run_problems
from conductor._internal.validation import Validator


def _demands_more(node_input: EdgeInput | CollectingInput) -> bool:
    """Whether an input's type asks for more than the outputs wired to it promise.

    If it does, the node's edge validator checks the input's value before each call.

    Every ``str`` is already a valid ``str | None``, so an edge from ``-> str`` into ``line: str | None`` needs no
    check. An edge from ``-> int`` into ``Annotated[int, Param(le=3)]`` does, and so does an edge from ``-> Any``.

    An input that collects a loop gets one item of its value from each edge, not the whole value. Then each output
    is compared with the input's item type. An input that collects rows into a table is always checked, since
    each row has to be checked against the table's columns.
    """
    field, sources = node_input.field, [edge.type for edge in node_input.sources]
    match node_input:
        case EdgeInput():
            return not all(field.whole.holds(source) for source in sources)
        case CollectingInput():
            return field.whole.row is not None or not all(
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
                raise InvalidCallError(
                    [run_problems.invalid_edge_value(input=name, **whole.refusal(error))]
                ) from None
        return read


class WaitingIdValidator(Validator[str, tuple[Loc, NodePath, ItemIndexes, str | None]]):
    """A validator for the id a host names a waiting call by, in ``run.with_answers`` and ``run.with_decline``.

    It undoes ``write_waiting_id``: ``"review@0/approve@1~3fa2c0d1"`` gives the loc
    ``("review", 0, "approve", 1)``, the node path ``("review", "approve")``, the item indexes ``(0, 1)`` and the
    fingerprint ``"3fa2c0d1"``. An id with no ``~`` part names just the call, and its fingerprint is ``None``.

    Only the exact text ``write_waiting_id`` would write is taken. ``"approve@01"`` names nothing, since the index
    would be written ``1``, and ``"review//approve"`` names nothing, since a node id can't be empty. An id that names
    no node, such as ``""``, reads as the loc ``()``; the run finds no call there.

    Raises (when called):
        InvalidCallError: ``not_waiting`` if the id isn't one ``write_waiting_id`` could have written. Its loc is
            as far as ``parse_waiting_id`` reads one.
    """

    def __init__(self, *, status: Callable[[], str]) -> None:
        """Builds the validator for the ids a run is given.

        Args:
            status: Returns the run's status, for the refusal's message. It is only called when an id is refused.
        """
        self._status = status

    def __call__(self, waiting_id: str) -> tuple[Loc, NodePath, ItemIndexes, str | None]:
        """Returns the call's loc, its node path, its item indexes, and the fingerprint of its question.

        Raises:
            InvalidCallError: See the class.
        """
        loc, fingerprint = parse_waiting_id(waiting_id)
        if write_waiting_id(loc, fingerprint) != waiting_id:
            raise InvalidCallError(
                [call_problems.not_waiting(*loc, waiting_id=waiting_id, status=self._status())]
            )
        node_path = tuple(part for part in loc if isinstance(part, str))
        item_indexes = tuple(part for part in loc if isinstance(part, int))
        return loc, node_path, item_indexes, fingerprint


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
