"""The inputs door: the graph inputs a caller gives to start a run, and the run's limits.

Example: the graph of one node, ``make_graph(shout)``, has the graph inputs ``line`` (required) and ``mark``
(default ``"!"``). ``graph.run(line="b")`` gives the door ``{"line": "b"}``, which returns ``{"line": "b", "mark":
"!"}``: every graph input by name, its default where the caller gave none. Given ``{"line": 3, "colour": "red"}``
it raises ``RefusedError`` with every fault at once: ``not_offered`` for ``colour``, then ``invalid_input`` for
``line``.

The limits (``max_items``, ``timeout``, ``checkpoint_every``) are the same for every graph, so they are checked by
one function, ``check_limits``. A yes or no is not a number: ``max_items=True`` is refused, as is ``max_items=2.5``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails, SchemaValidator, core_schema
from pydantic_core.core_schema import TypedDictField

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field
from conductor._internal.problem import Problem
from conductor._internal.problem import catalogue as problem

from .refused import RefusedError
from .rules import schema_of


class InputsDoor:
    """Reads the graph inputs a caller gives, by name, into the value of every graph input.

    Built once per graph. One pydantic validator reads them all: an input with no default must be given, nothing
    else may be, and each value is read by its input's type, as pydantic reads a Python value (``"3"`` for an
    ``int`` is 3).

    Raises (when called):
        RefusedError: With every fault, in this order: each name the caller gave that the graph does not offer
            (``not_offered``), in the order given; then each graph input in turn that is missing (``missing_input``)
            or whose value its type refuses (``invalid_input``, and its variant ``make_a_series`` for a list given to
            an input of one value).
    """

    def __init__(self, inputs: Mapping[str, Field]) -> None:
        """Builds the door for a graph's inputs.

        Args:
            inputs: Each graph input by name, as its field, whose default is used when the caller gives none
                (``REQUIRED`` for none).
        """
        self._inputs = dict(inputs)
        self._validator = SchemaValidator(core_schema.typed_dict_schema(
            {name: _input_schema(field) for name, field in inputs.items()}, extra_behavior="forbid",
        ))

    def __call__(self, given: Mapping[Any, Any]) -> dict[str, Any]:
        """Returns the value of every graph input by name, in the graph's order: the caller's, read by the input's
        type, else the input's default.

        Args:
            given: What the caller gave, each value under the input's name.

        Raises:
            RefusedError: See the class.
        """
        try:
            values: dict[str, Any] = self._validator.validate_python(given)
        except ValidationError as error:
            raise RefusedError(self._problems(given, error)) from None
        return values

    def _problems(self, given: Mapping[Any, Any], error: ValidationError) -> list[Problem]:
        """The problems of pydantic's ``error`` for ``given``: the names the graph does not take first, in the
        order given, then the faults of each input, in the graph's order."""
        names: list[Problem] = []
        first_faults: dict[str, ErrorDetails] = {}
        for fault in error.errors():
            key = fault["loc"][0]
            if fault["type"] in ("extra_forbidden", "invalid_key"):
                names.append(problem.not_offered("inputs", key, name=str(key), inputs=list(self._inputs)))
            else:
                first_faults.setdefault(str(key), fault)
        return names + [self._input_problem(name, given, error, first_faults[name]) for name in self._inputs if name in first_faults]

    def _input_problem(self, name: str, given: Mapping[Any, Any], error: ValidationError, first: ErrorDetails) -> Problem:
        """The problem of one graph input whose value is missing or refused by its type; ``first`` is the first
        fault of pydantic's ``error`` that is in this input."""
        field = self._inputs[name]
        key, *inside = first["loc"]
        if first["type"] == "missing" and not inside:  # the input itself, not a field inside its value
            return problem.missing_input("inputs", name, name=name)
        refused = field.whole.refusal(error, under=(key,))
        value = given[name]
        if not field.is_series and type(value) is list and all(not error["loc"] for error in refused.errors):
            return problem.invalid_input_make_a_series("inputs", name, name=name, expected=field.whole.id, length=len(value))  # the one mistake whose fix is in the graph
        return problem.invalid_input("inputs", name, name=name, reason=refused.reason, errors=refused.errors, column=refused.column)


def _input_schema(field: Field) -> TypedDictField:
    """The schema of one graph input inside the door's: required unless it has a default, and read by its type."""
    schema = schema_of(field.whole)
    required = field.default is REQUIRED
    if not required:
        default = field.default
        schema = core_schema.with_default_schema(schema, default_factory=lambda: default)  # the default itself, never a copy
    return core_schema.typed_dict_field(schema, required=required)


#: Reads the limits a caller gives a run. A limit left out is ``None``, which is not checked.
_LIMITS = SchemaValidator(core_schema.typed_dict_schema({
    "max_items": core_schema.typed_dict_field(core_schema.nullable_schema(core_schema.int_schema(strict=True, ge=1))),
    "timeout": core_schema.typed_dict_field(core_schema.nullable_schema(core_schema.float_schema(strict=True, gt=0, allow_inf_nan=False))),
    "checkpoint_every": core_schema.typed_dict_field(core_schema.nullable_schema(core_schema.float_schema(strict=True, gt=0, allow_inf_nan=False))),
}))

#: The problem of each limit, by its name.
_LIMIT_PROBLEMS = {
    "max_items": lambda value: problem.invalid_max_items("max_items", max_items=value),
    "timeout": lambda value: problem.invalid_timeout("timeout", timeout=value),
    "checkpoint_every": lambda value: problem.invalid_checkpoint_every("checkpoint_every", checkpoint_every=value),
}


def check_limits(*, max_items: Any = None, timeout: Any = None, checkpoint_every: Any = None) -> None:
    """Checks the limits a caller gives a run, and returns nothing: a run keeps each limit as it was given.

    ``max_items`` is a whole number of at least 1. ``timeout`` and ``checkpoint_every`` are seconds, finite and
    above 0. A yes or no is none of these. ``None`` is a limit the caller did not give. ``check_limits(max_items=0)``
    raises.

    Raises:
        RefusedError: With a problem for each limit refused, in the order above (``invalid_max_items``,
            ``invalid_timeout``, ``invalid_checkpoint_every``), which quotes the value given.
    """
    given = {"max_items": max_items, "timeout": timeout, "checkpoint_every": checkpoint_every}
    try:
        _LIMITS.validate_python(given)
    except ValidationError as error:
        raise RefusedError([_LIMIT_PROBLEMS[name](given[name]) for name in dict.fromkeys(str(e["loc"][0]) for e in error.errors())]) from None
