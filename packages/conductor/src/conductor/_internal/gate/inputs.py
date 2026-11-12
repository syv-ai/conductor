"""The inputs door: the graph inputs a caller gives to start a run, and the run's limits.

Example: the graph of one node, ``make_graph(shout)``, has the graph inputs ``line`` (``shout.line``, required) and
``mark`` (``shout.mark``, default ``"!"``). ``graph.run(line="b")`` gives the door ``{"line": "b"}``, which returns
``{"shout.line": "b", "shout.mark": "!"}``: every graph input by its address, its default where the caller gave
none. Given ``{"line": 3, "shout.line": "b", "colour": "red"}`` it raises ``RefusedError`` with every fault at
once: ``input_given_twice`` for ``line``, ``not_offered`` for ``colour``, then ``invalid_input`` for ``line``.

The limits (``max_items``, ``timeout``, ``checkpoint_every``) are the same for every graph, so they are checked by
one function, ``check_limits``. A yes or no is not a number: ``max_items=True`` is refused, as is ``max_items=2.5``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails, SchemaValidator, core_schema
from pydantic_core.core_schema import TypedDictField

from conductor._internal.field import Field
from conductor._internal.gate.refused import RefusedError
from conductor._internal.gate.rules import schema_of
from conductor._internal.markers import REQUIRED
from conductor._internal.problem import Problem
from conductor._internal.problem import catalogue as problem


class InputsDoor:
    """Reads the graph inputs a caller gives, by name or by address, into the value of every graph input by address.

    Built once per graph. One pydantic validator reads them all: an input is found under its name or its address,
    an input with no default must be given, nothing else may be, and each value is read by its input's type, as
    pydantic reads a Python value (``"3"`` for an ``int`` is 3).

    Raises (when called):
        RefusedError: With every fault, in this order: each name the caller gave that the graph does not offer
            (``not_offered``) or that names an input given already (``input_given_twice``), in the order given;
            then each graph input in turn that is missing (``missing_input``) or whose value its type refuses
            (``invalid_input``, and its variant ``make_a_series`` for a list given to an input of one value).
    """

    def __init__(self, inputs: Mapping[str, tuple[str, Field]]) -> None:
        """Builds the door for a graph's inputs.

        Args:
            inputs: Each graph input by its address (``lines.text``), with its name in ``graph.inputs`` (``text``,
                or the address when two graph inputs share a name) and its field, whose default is used when the
                caller gives none (``REQUIRED`` for none).
        """
        self._inputs = dict(inputs)
        self._addresses = {**{address: address for address in inputs}, **{name: address for address, (name, _) in inputs.items()}}
        self._validator = SchemaValidator(core_schema.typed_dict_schema(
            {address: _input_schema(name, address, field) for address, (name, field) in inputs.items()}, extra_behavior="forbid",
        ))

    def __call__(self, given: Mapping[Any, Any]) -> dict[str, Any]:
        """Returns the value of every graph input by address, in the graph's order: the caller's, read by the
        input's type, else the input's default.

        Args:
            given: What the caller gave, each value under the input's name or its address.

        Raises:
            RefusedError: See the class.
        """
        try:
            values: dict[str, Any] = self._validator.validate_python(given)
        except ValidationError as error:
            raise RefusedError(self._problems(given, error.errors(include_url=False, include_input=False))) from None
        return values

    def addresses(self, given: Mapping[Any, Any]) -> set[str]:
        """The addresses of the graph inputs that ``given`` names, by name or by address; a name the graph does not
        offer names none. For ``make_graph(shout)``, ``{"line": ..., "shout.mark": ...}`` names ``{"shout.line", "shout.mark"}``."""
        return {self._addresses[key] for key in given if key in self._addresses}

    def _problems(self, given: Mapping[Any, Any], errors: list[ErrorDetails]) -> list[Problem]:
        """The problems of pydantic's ``errors`` for ``given``: the names the graph does not take first, in the
        order given, then the faults of each input, in the graph's order."""
        names: list[Problem] = []
        by_input: dict[str, list[ErrorDetails]] = {}
        for error in errors:
            key, *inside = error["loc"]
            if error["type"] in ("extra_forbidden", "invalid_key"):
                names.append(self._name_problem(key))
            else:  # a fault of an input, under the text it was found by: its name or its address
                by_input.setdefault(self._addresses[str(key)], []).append({**error, "loc": tuple(inside)})
        return names + [self._input_problem(address, given, by_input[address]) for address in self._inputs if address in by_input]

    def _name_problem(self, key: Any) -> Problem:
        """The problem of a key the caller gave that the door does not take: a name the graph does not offer, or
        the name or the address of an input found under its other key."""
        address = self._addresses.get(key)
        if address is None:
            return problem.not_offered("inputs", key, name=key, inputs=[name for name, _ in self._inputs.values()])
        name, _ = self._inputs[address]
        return problem.input_given_twice("inputs", name, name=name, address=address)

    def _input_problem(self, address: str, given: Mapping[Any, Any], errors: list[ErrorDetails]) -> Problem:
        """The problem of one graph input whose value is missing or refused by its type."""
        name, field = self._inputs[address]
        if errors[0]["type"] == "missing" and not errors[0]["loc"]:  # the input itself, not a field inside its value
            return problem.missing_input("inputs", name, name=name)
        refused = field.whole.refusal(errors)
        value = given[name] if name in given else given[address]
        if not field.is_series and type(value) is list and all(not error["loc"] for error in refused.errors):
            return problem.invalid_input_make_a_series("inputs", name, name=name, expected=field.whole.id, length=len(value))  # the one mistake whose fix is in the graph
        return problem.invalid_input("inputs", name, name=name, reason=refused.reason, errors=refused.errors, column=refused.column)


def _input_schema(name: str, address: str, field: Field) -> TypedDictField:
    """The schema of one graph input inside the door's: found under its name or its address, required unless it
    has a default, and read by its type."""
    schema = schema_of(field.whole)
    required = field.default is REQUIRED
    if not required:
        default = field.default
        schema = core_schema.with_default_schema(schema, default_factory=lambda: default)  # the default itself, never a copy
    found_under: list[list[str | int]] = [[name], [address]]
    return core_schema.typed_dict_field(schema, required=required, validation_alias=found_under)


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
