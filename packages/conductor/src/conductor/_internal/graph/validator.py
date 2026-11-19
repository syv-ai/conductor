"""The validators for what a graph is given.

There are three:

- ``InputsValidator`` checks the graph inputs a caller gives to start a run.
- ``LimitsValidator`` checks a run's limits.
- ``ConstantValidator`` checks a constant, or a graph input's default, written in the graph record.

Inputs. Take ``make_graph(shout, defaults={"shout.mark": "!"})``, a graph of one node. It has the graph inputs
``line``, which is required, and ``mark``, which defaults to ``"!"``. ``graph.run({"line": "b"})`` gives the validator ``{"line": "b"}``, and it returns
``{"line": "b", "mark": "!"}``. That's every graph input, with its default where the caller gave none. Given
``{"line": 3, "colour": "red"}``, it raises ``InvalidCallError`` with both faults at once: ``unknown_graph_input`` for
``colour``, then ``invalid_input`` for ``line``.

Limits. These are ``max_items``, ``timeout`` and ``checkpoint_every``. They're the same for every graph, so
``LimitsValidator`` is built with no arguments. A yes or no isn't a number, so ``max_items=True`` is refused, and so
is ``max_items=2.5``.

Constants. These are checked by their input's type when the graph is compiled. Take ``shout(line: str, mark: str =
"!")`` with ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}``. It gets ``"?"`` on every call. The
constant ``3`` is refused with ``invalid_value`` at ``("shout", "mark")``. The problem says what the type refused,
without quoting the value. A record stores JSON, so a value is read strictly, as a caller's graph input is:
``"2026-01-02"`` is a ``date`` for an input that takes one, and ``"3"`` is no ``int``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast

from pydantic import JsonValue, ValidationError
from pydantic_core import SchemaValidator, core_schema

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field
from conductor._internal.problem import (
    InvalidCallError,
    Problem,
    call_problems,
    compile_problems,
)
from conductor._internal.types import Type
from conductor._internal.utils import comma_separated
from conductor._internal.validation import Validator, read_as_json


class InputsValidator(Validator[Mapping[Any, Any], dict[str, Any]]):
    """The validator for the graph inputs a caller gives, by name.

    It returns the value of every graph input. One is built per graph. An input with no default must be given, and
    nothing else may be.

    Each value is read by its input's type strictly, the way its JSON text would be, as a person's answer is
    (``read_as_json``). So a value given from Python and the same value sent over HTTP read the same, and so does
    the value a saved run reads back. ``3`` is an ``int``, but ``"3"`` and ``3.0`` are refused. ``True`` is a
    ``bool``, but ``"yes"`` and ``1`` are refused. A ``date`` may be given as a ``date`` or as its ISO text, and a
    model as its class or as a mapping of its fields. A value JSON can't write, such as an object held in memory,
    is refused: what a node needs from the host that isn't data goes in the run's context. So is a set, which has
    no order and so no one JSON text: give a list.

    A frame is read as it is where the input's type is a table, and is refused anywhere else, as the reader says.

    Raises (when called):
        InvalidCallError: Lists every fault, in this order. First, each name the caller gave that the graph doesn't
            offer (``unknown_graph_input``), in the order given. Then each graph input, in the graph's order, that's
            missing (``missing_input``) or whose value its type refuses (``invalid_input``), or
            ``invalid_input_list_for_one_value`` for a list given to an input whose type takes no list.
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the validator for a graph's inputs.

        Args:
            inputs: Each graph input by name, as its field. The field's default is used when the caller gives none,
                and is ``REQUIRED`` when there is none.
        """
        self._inputs = dict(inputs)

    def __call__(self, given: Mapping[Any, Any]) -> dict[str, Any]:
        """Checks ``given`` and returns the value of every graph input, by name, in the graph's order.

        Each value is the caller's, read by the input's type, or else the input's default.

        Args:
            given: What the caller gave, each value under its input's name.

        Raises:
            InvalidCallError: See the class.
        """
        problems = [
            call_problems.unknown_graph_input(
                "inputs", name, name=str(name), inputs=list(self._inputs)
            )
            for name in given
            if name not in self._inputs
        ]
        values: dict[str, Any] = {}
        for name, field in self._inputs.items():
            if name in given:
                try:
                    values[name] = read_as_json(
                        field.whole, field.whole.argument_validator, given[name]
                    )
                except ValidationError as error:
                    problems.append(_refused(field, given[name], error))
            elif field.default is REQUIRED:
                problems.append(call_problems.missing_input("inputs", name, name=name))
            else:
                values[name] = field.default  # the default itself, never a copy
        if problems:
            raise InvalidCallError(problems)
        return values


def _refused(field: Field, value: Any, error: ValidationError) -> Problem:
    """Returns the problem of a graph input whose value its type refuses."""
    if type(value) is list and not field.is_series and field.whole.refuses_as_a_list(value):
        return call_problems.invalid_input_list_for_one_value(
            "inputs", field.name, name=field.name, expected=field.whole.id, length=len(value)
        )  # the one mistake whose fix is in the graph, not in the value
    return call_problems.invalid_input(
        "inputs", field.name, name=field.name, **field.whole.refusal(error)
    )


class Limits(TypedDict, total=False):
    """A run's limits, by name.

    This is what a caller gives ``LimitsValidator``, and what it returns. A limit left out, or given as ``None``, is
    one the caller didn't set. What the validator returns has all three, with ``None`` for each one not set.
    """

    max_items: int | None
    timeout: float | None
    checkpoint_every: float | None


#: Checks the limits a caller gives a run. A limit left out is ``None``, which isn't checked.
_LIMITS = SchemaValidator(
    core_schema.typed_dict_schema(
        {
            "max_items": core_schema.typed_dict_field(
                core_schema.nullable_schema(core_schema.int_schema(strict=True, ge=1))
            ),
            "timeout": core_schema.typed_dict_field(
                core_schema.nullable_schema(
                    core_schema.float_schema(strict=True, gt=0, allow_inf_nan=False)
                )
            ),
            "checkpoint_every": core_schema.typed_dict_field(
                core_schema.nullable_schema(
                    core_schema.float_schema(strict=True, gt=0, allow_inf_nan=False)
                )
            ),
        }
    )
)


class LimitsValidator(Validator[Limits, Limits]):
    """The validator for the limits a caller gives a run.

    ``max_items`` must be a whole number of at least 1. ``timeout`` and ``checkpoint_every`` are seconds, finite and
    above 0. A yes or no is none of these. So ``LimitsValidator()({"max_items": 0})`` raises.

    It returns ``timeout`` and ``checkpoint_every`` as a ``float``, so ``timeout=Decimal("1.5")`` becomes ``1.5``.

    Raises (when called):
        InvalidCallError: One ``invalid_limit`` for each limit refused, in the order above, with pydantic's errors
            and not the value. For ``timeout`` and ``checkpoint_every``, it's ``invalid_limit_seconds``.
    """

    def __call__(self, limits: Limits) -> Limits:
        """Checks ``limits`` and returns them, by name. A limit left out, or ``None``, isn't set.

        Raises:
            InvalidCallError: See the class.
            TypeError: ``limits`` names a limit that doesn't exist, such as ``max_item``. That's a bug in
                conductor's own call, which mypy catches too.
        """
        if unknown := sorted(limits.keys() - Limits.__annotations__.keys()):
            raise TypeError(
                f"a run has no limit {unknown[0]!r}; its limits are "
                f"{comma_separated(Limits.__annotations__)}"
            )
        given = {
            "max_items": limits.get("max_items"),
            "timeout": limits.get("timeout"),
            "checkpoint_every": limits.get("checkpoint_every"),
        }
        try:
            return cast(Limits, _LIMITS.validate_python(given))
        except ValidationError as error:
            refused = dict.fromkeys(str(e["loc"][0]) for e in error.errors())
            raise InvalidCallError(
                [
                    (
                        call_problems.invalid_limit
                        if name == "max_items"
                        else call_problems.invalid_limit_seconds
                    )(name, errors=Type(Any).refusal(error, (name,))["errors"])
                    for name in refused
                ]
            ) from None


class ConstantValidator(Validator[JsonValue, Any]):
    """The validator for a value the graph record gives one input of a node.

    The value is a constant, or a graph input's default. The record stores it as JSON, and this validator reads it
    by the input's type strictly, as a caller's graph input is read (``read_as_json``). So ``"3"`` is refused for an
    ``int``, as ``graph.run({"n": "3"})`` is, and ``"2026-01-02"`` is a ``date``.

    Raises (when called):
        InvalidCallError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """

    def __init__(self, node_id: str, field: Field) -> None:
        """Builds the validator for the input ``field`` of the node ``node_id``."""
        self._node_id = node_id
        self._field = field

    def __call__(self, value: JsonValue) -> Any:
        """Checks ``value`` and returns it read by the input's type.

        Raises:
            InvalidCallError: See the class.
        """
        whole, name = self._field.whole, self._field.name
        try:
            return read_as_json(whole, whole.argument_validator, value)
        except ValidationError as error:
            found = compile_problems.invalid_value(self._node_id, name, **whole.refusal(error))
        raise InvalidCallError([found])
