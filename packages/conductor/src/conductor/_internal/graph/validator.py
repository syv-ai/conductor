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
without quoting the value. A record stores JSON, so a value is converted the way its JSON would be:
``"2026-01-02"`` is a ``date`` for an input that takes one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast

from pydantic import JsonValue, ValidationError
from pydantic_core import ErrorDetails, SchemaValidator, core_schema
from pydantic_core.core_schema import TypedDictField

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, Problem, call_problems, compile_problems
from conductor._internal.types import Type
from conductor._internal.utils import comma_separated
from conductor._internal.validation import Validator


class InputsValidator(Validator[Mapping[Any, Any], dict[str, Any]]):
    """The validator for the graph inputs a caller gives, by name.

    It returns the value of every graph input. One is built per graph, and inside it one pydantic validator checks
    all the inputs. An input with no default must be given, and nothing else may be. Each value is checked by its
    input's type the way pydantic checks a Python value, so ``"3"`` for an ``int`` becomes ``3``.

    Raises (when called):
        InvalidCallError: Lists every fault, in this order. First, each name the caller gave that the graph doesn't
            offer (``unknown_graph_input``), in the order given. Then each graph input, in the graph's order, that's
            missing (``missing_input``) or whose value its type refuses (``invalid_input``), or
            ``invalid_input_make_a_series`` for a list given to an input whose type takes no list.
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the validator for a graph's inputs.

        Args:
            inputs: Each graph input by name, as its field. The field's default is used when the caller gives none,
                and is ``REQUIRED`` when there is none.
        """
        self._inputs = dict(inputs)
        # For each input that isn't a Series, each option of its type without its limits or choices. Kept here so
        # its validator is built only once, the first time a list given there is refused.
        self._bare_options = {
            name: tuple(Type(option) for option in field.whole.options)
            for name, field in inputs.items()
            if not field.is_series
        }
        self._validator = SchemaValidator(
            core_schema.typed_dict_schema(
                {name: _input_schema(field) for name, field in inputs.items()},
                extra_behavior="forbid",
            )
        )

    def __call__(self, given: Mapping[Any, Any]) -> dict[str, Any]:
        """Checks ``given`` and returns the value of every graph input, by name, in the graph's order.

        Each value is the caller's, converted by the input's type, or else the input's default.

        Args:
            given: What the caller gave, each value under its input's name.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            values: dict[str, Any] = self._validator.validate_python(given)
        except ValidationError as error:
            raise InvalidCallError(self._problems(given, error)) from None
        return values

    def _problems(self, given: Mapping[Any, Any], error: ValidationError) -> list[Problem]:
        """Turns pydantic's ``error`` for ``given`` into problems.

        Names the graph doesn't take come first, in the order given. Then come the faults of each input, in the
        graph's order.
        """
        names: list[Problem] = []
        first_faults: dict[str, ErrorDetails] = {}
        for fault in error.errors():
            key = fault["loc"][0]
            if (
                fault["type"] in ("extra_forbidden", "invalid_key") and len(fault["loc"]) == 1
            ):  # a key inside a value, such as a record's, is a fault of that input
                names.append(
                    call_problems.unknown_graph_input(
                        "inputs", key, name=str(key), inputs=list(self._inputs)
                    )
                )
            else:
                first_faults.setdefault(str(key), fault)
        return names + [
            self._input_problem(name, given, error, first_faults[name])
            for name in self._inputs
            if name in first_faults
        ]

    def _input_problem(
        self, name: str, given: Mapping[Any, Any], error: ValidationError, first: ErrorDetails
    ) -> Problem:
        """Returns the problem of one graph input whose value is missing or refused by its type.

        ``first`` is the first fault in pydantic's ``error`` that's about this input.
        """
        field = self._inputs[name]
        key, *inside = first["loc"]
        # the input itself is missing, not a field inside its value
        if first["type"] == "missing" and not inside:
            return call_problems.missing_input("inputs", name, name=name)
        value = given[name]
        if (
            type(value) is list
            and name in self._bare_options
            and all(_refused_as_a_list(option, value) for option in self._bare_options[name])
        ):
            return call_problems.invalid_input_make_a_series(
                "inputs", name, name=name, expected=field.whole.id, length=len(value)
            )  # the one mistake whose fix is in the graph, not in the value
        refused = field.whole.refusal(error, under=(key,))
        return call_problems.invalid_input(
            "inputs",
            name,
            name=name,
            reason=refused.reason,
            errors=refused.errors,
            column=refused.column,
        )


def _refused_as_a_list(option: Type, value: list[Any]) -> bool:
    """Whether ``option`` refuses the list ``value`` as a whole, rather than because of an item in it.

    This tells whether a list given to an input should have been a loop over its items instead. ``option`` is one
    option of the input's type, without its limits and choices.

    ``str`` refuses ``["a", "b"]`` as a whole. ``list[str]`` takes ``[]``, and
    ``Annotated[list[str], Param(min_length=1)]`` refuses it only by its limit. ``list[int]`` refuses ``["x"]``
    because of its item. So for ``int | list[int]`` given ``["x"]``, a loop isn't the fix.
    """
    try:
        option.argument_validator.validate_python(value)
    except ValidationError as error:
        return all(not fault["loc"] for fault in option.refusal(error).errors)
    return False


def _input_schema(field: Field) -> TypedDictField:
    """Returns the schema of one graph input, inside the validator's schema.

    The input is required unless it has a default, and its value is checked by its type.
    """
    schema = field.whole.argument_schema
    required = field.default is REQUIRED
    if not required:
        default = field.default
        schema = core_schema.with_default_schema(
            schema, default_factory=lambda: default
        )  # the default itself, never a copy
    return core_schema.typed_dict_field(schema, required=required)


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
        InvalidCallError: One ``invalid_limit`` for each limit refused, in the order above. It quotes the value
            given. For ``timeout`` and ``checkpoint_every``, it's ``invalid_limit_seconds``.
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
                    )(name, value=given[name])
                    for name in refused
                ]
            ) from None


class ConstantValidator(Validator[JsonValue, Any]):
    """The validator for a value the graph record gives one input of a node.

    The value is a constant, or a graph input's default. The record stores it as JSON, and this validator converts
    it by the input's type.

    Raises (when called):
        InvalidCallError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """

    def __init__(self, node_id: str, field: Field) -> None:
        """Builds the validator for the input ``field`` of the node ``node_id``."""
        self._node_id = node_id
        self._field = field

    def __call__(self, value: JsonValue) -> Any:
        """Checks ``value`` and returns it converted by the input's type.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            return self._field.whole.argument_validator.validate_python(value)
        except ValidationError as error:
            refused = self._field.whole.refusal(error)
            raise InvalidCallError(
                [
                    compile_problems.invalid_value(
                        self._node_id,
                        self._field.name,
                        reason=refused.reason,
                        errors=refused.errors,
                        column=refused.column,
                    )
                ]
            ) from None
