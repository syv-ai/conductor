"""The validators of what a graph is given: the graph inputs a caller gives to start a run, the run's limits, and a
constant or a graph input's default written in the graph record.

The inputs. Example: the graph of one node, ``make_graph(shout)``, has the graph inputs ``line`` (required) and
``mark`` (default ``"!"``). ``graph.run(line="b")`` gives the validator ``{"line": "b"}``, which returns
``{"line": "b", "mark": "!"}``: every graph input by name, its default where the caller gave none. Given
``{"line": 3, "colour": "red"}`` it raises ``InvalidCallError`` with every fault at once: ``not_offered`` for
``colour``, then ``invalid_input`` for ``line``.

The limits (``max_items``, ``timeout``, ``checkpoint_every``) are the same for every graph, so they are read by
one validator, ``LimitsValidator``, built with nothing. A yes or no is not a number: ``max_items=True`` is refused, as is ``max_items=2.5``.

The constants, read by their input's type when the graph is compiled. Example: ``shout(line: str, mark: str = "!")``
with ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}`` gets ``"?"`` on every call.
``{"mark": {"constant": 3}}`` is refused with ``invalid_value`` at ``("shout", "mark")``, which says what the type
refused and quotes nothing of the value. A record holds JSON, so a value is read as its JSON reads: ``"2026-01-02"``
is a ``date`` for an input that takes one.
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
from conductor._internal.validation import Validator, schema_of, schema_validator_of


class InputsValidator(Validator[Mapping[Any, Any], dict[str, Any]]):
    """Reads the graph inputs a caller gives, by name, into the value of every graph input.

    Built once per graph. One pydantic validator reads them all: an input with no default must be given, nothing
    else may be, and each value is read by its input's type, as pydantic reads a Python value (``"3"`` for an
    ``int`` is 3).

    Raises (when called):
        InvalidCallError: With every fault, in this order: each name the caller gave that the graph does not offer
            (``not_offered``), in the order given; then each graph input in turn that is missing (``missing_input``)
            or whose value its type refuses (``invalid_input``, and its variant ``make_a_series`` for a list given to
            an input whose type takes no list).
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the validator for a graph's inputs.

        Args:
            inputs: Each graph input by name, as its field, whose default is used when the caller gives none
                (``REQUIRED`` for none).
        """
        self._inputs = dict(inputs)
        # each option of each input that is not a Series, without limits or choices, held so that its validator is
        # built once, the first time a list given there is refused
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
        """Returns the value of every graph input by name, in the graph's order: the caller's, read by the input's
        type, else the input's default.

        Args:
            given: What the caller gave, each value under the input's name.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            values: dict[str, Any] = self._validator.validate_python(given)
        except ValidationError as error:
            raise InvalidCallError(self._problems(given, error)) from None
        return values

    def _problems(self, given: Mapping[Any, Any], error: ValidationError) -> list[Problem]:
        """The problems of pydantic's ``error`` for ``given``: the names the graph does not take first, in the
        order given, then the faults of each input, in the graph's order."""
        names: list[Problem] = []
        first_faults: dict[str, ErrorDetails] = {}
        for fault in error.errors():
            key = fault["loc"][0]
            if fault["type"] in ("extra_forbidden", "invalid_key"):
                names.append(
                    call_problems.not_offered(
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
        """The problem of one graph input whose value is missing or refused by its type; ``first`` is the first
        fault of pydantic's ``error`` that is in this input."""
        field = self._inputs[name]
        key, *inside = first["loc"]
        # the input itself, not a field inside its value
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
            )  # the one mistake whose fix is in the graph
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
    """Whether ``option``, one option of an input's type without its limits and choices, refuses the list ``value``
    as a whole rather than for an item in it.

    Example: ``str`` refuses ``["a", "b"]`` so. ``list[str]`` takes ``[]``, which
    ``Annotated[list[str], Param(min_length=1)]`` refuses only by its limit, and ``list[int]`` refuses ``["x"]`` for
    its item, so ``int | list[int]`` given ``["x"]`` is no question of looping.
    """
    try:
        schema_validator_of(option).validate_python(value)
    except ValidationError as error:
        return all(not fault["loc"] for fault in option.refusal(error).errors)
    return False


def _input_schema(field: Field) -> TypedDictField:
    """The schema of one graph input inside the validator's: required unless it has a default, and read by its type."""
    schema = schema_of(field.whole)
    required = field.default is REQUIRED
    if not required:
        default = field.default
        schema = core_schema.with_default_schema(
            schema, default_factory=lambda: default
        )  # the default itself, never a copy
    return core_schema.typed_dict_field(schema, required=required)


class Limits(TypedDict, total=False):
    """A run's limits by name, as a caller gives them to ``LimitsValidator`` and as it returns them: a limit given as
    ``None``, or left out, is one the caller did not give. What the validator returns has all three, ``None`` for a
    limit not given."""

    max_items: int | None
    timeout: float | None
    checkpoint_every: float | None


#: Reads the limits a caller gives a run. A limit left out is ``None``, which is not checked.
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
    """Reads the limits a caller gives a run, and returns them as the run keeps them: ``timeout`` and
    ``checkpoint_every`` as a ``float``, so ``timeout=Decimal("1.5")`` is ``1.5``.

    ``max_items`` is a whole number of at least 1. ``timeout`` and ``checkpoint_every`` are seconds, finite and
    above 0. A yes or no is none of these. ``LimitsValidator()({"max_items": 0})`` raises.

    Raises (when called):
        InvalidCallError: With an ``invalid_limit`` for each limit refused, in the order above, which quotes the value
            given; its variant ``seconds`` for ``timeout`` and ``checkpoint_every``.
    """

    def __call__(self, limits: Limits) -> Limits:
        """Returns the limits the caller gave, by name; a limit left out, or ``None``, is one it did not give.

        Raises:
            InvalidCallError: See the class.
            TypeError: ``limits`` names a limit there is none of, such as ``max_item``: a fault in conductor's own
                call, which mypy refuses too.
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


class RecordValueValidator(Validator[JsonValue, Any]):
    """Reads a value a graph record holds for one input of a node, a constant or a graph input's default, out of its
    JSON by the input's type.

    Raises (when called):
        InvalidCallError: The type refuses the value: one ``invalid_value`` problem at ``(node_id, input)``.
    """

    def __init__(self, node_id: str, field: Field) -> None:
        """Builds the validator for the input ``field`` of the node ``node_id``."""
        self._node_id = node_id
        self._field = field

    def __call__(self, value: JsonValue) -> Any:
        """Returns ``value`` read by the input's type.

        Raises:
            InvalidCallError: See the class.
        """
        field = self._field
        try:
            return schema_validator_of(field.whole).validate_python(value)
        except ValidationError as error:
            refused = field.whole.refusal(error)
            raise InvalidCallError(
                [
                    compile_problems.invalid_value(
                        self._node_id,
                        field.name,
                        reason=refused.reason,
                        errors=refused.errors,
                        column=refused.column,
                    )
                ]
            ) from None
