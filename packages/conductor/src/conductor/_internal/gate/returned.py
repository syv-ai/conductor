"""The return door: what a node's function returns, read into the node's outputs by one pydantic validator.

Example: ``shout(line: str) -> str`` returns ``"B!"``, and its door gives ``{"result": "B!"}``. Returned ``3``, the
door raises ``RefusedError`` with the problem ``invalid_output`` for the output ``result``.

A value is read strictly: a value already of its class is kept as it is, a dict is read into a model, and ``"12"``
is never an ``int``. A node that returns ``SKIPPED`` gets ``SKIPPED`` for every output.

A node with several outputs returns a frozen dataclass, or a mapping when its ``compute_outputs`` hook gave the
outputs. Its door reads every output in one call of one validator. First the shape: the dataclass or a mapping,
no output missing or unknown, and of an exclusive group at most one output with a value. Then each output's value,
by its type, in the order the outputs are declared. Only the first fault is reported, as one problem.

Every rule inside the validator raises ``PydanticCustomError``: any other exception would escape pydantic with the
value in its message.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    SchemaValidator,
    core_schema,
)

from conductor._internal.base import SKIPPED, comma_separated
from conductor._internal.field import Field
from conductor._internal.problem import run_problems

from .refused import RefusedError
from .rules import exact_classes, output_schema_of, output_validator_of


@dataclass(frozen=True, slots=True)
class OneResult:
    """The function returns one value, its output ``result``."""


@dataclass(frozen=True, slots=True)
class OutputsClass:
    """The function returns a frozen dataclass, one output per field.

    Attributes:
        cls: The dataclass.
    """

    cls: type


@dataclass(frozen=True, slots=True)
class HookOutputs:
    """The function returns a mapping of the outputs its ``compute_outputs`` hook gave when the graph was compiled."""


@dataclass(frozen=True, slots=True)
class NoOutputs:
    """The function returns ``None``: the node has no outputs."""


#: How a node's function returns its outputs, read from its signature by ``@node`` and read back by the door.
ReturnShape = OneResult | OutputsClass | HookOutputs | NoOutputs


class ReturnDoor:
    """Reads what one node's function returned into the node's outputs by name, as a run keeps them.

    Built once per compiled node, on the node's first call. Calling it with the returned value gives the outputs.

    Raises (when called):
        RefusedError: With one ``invalid_output`` problem, which names the output when there is one to name and quotes
            nothing of the value.
    """

    def __init__(self, returns: ReturnShape, outputs: Mapping[str, Field]) -> None:
        """Builds the door for a node whose function returns its outputs as ``returns`` says.

        Args:
            returns: How the function returns its outputs: one value, a dataclass, a mapping, or nothing.
            outputs: The node's outputs, by name.
        """
        self._names = tuple(outputs)
        self._read: Callable[[Any], dict[str, Any]]
        match returns:
            case NoOutputs():
                self._read = _no_outputs
            case OneResult():
                self._read = _one_output(outputs["result"])
            case OutputsClass(cls=cls):
                self._read = _several_outputs(outputs, _dataclass_fields(cls, self._names))
            case HookOutputs():
                self._read = _several_outputs(outputs, _mapping_items(self._names))

    def __call__(self, returned: Any) -> dict[str, Any]:
        """Returns the outputs of the node that returned ``returned``.

        Raises:
            RefusedError: The value is not what the node declares (``invalid_output``).
        """
        if returned is SKIPPED:
            return dict.fromkeys(self._names, SKIPPED)
        return self._read(returned)


def _no_outputs(returned: Any) -> dict[str, Any]:
    """The outputs of a node declared ``-> None``: none, when it returned ``None``."""
    if returned is not None:
        raise RefusedError([run_problems.invalid_output(reason=f"the node declares no outputs, but returned a {type(returned).__name__}")])
    return {}


def _one_output(field: Field) -> Callable[[Any], dict[str, Any]]:
    """The door of a node with the one output ``result``. A value exactly of one of the type's plain classes (a
    ``str`` for ``-> str``) is taken as it is, with no call to pydantic."""
    exact = exact_classes(field.whole)
    validate = output_validator_of(field.whole).validate_python

    def read(returned: Any) -> dict[str, Any]:
        if type(returned) in exact:
            return {"result": returned}
        try:
            return {"result": validate(returned, strict=True)}
        except ValidationError as error:
            raise _refused_output(field, error) from None

    return read


def _several_outputs(outputs: Mapping[str, Field], by_name: Callable[[Any], dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    """The door of a node with several outputs: one validator for all of them.

    First the shape of what was returned is checked, then each output's value: the before-function takes the
    value apart by output name and holds each exclusive group to one output with a value; the fields read the
    values; the after-function gives each output that was ``SKIPPED`` its ``SKIPPED`` back.

    Args:
        outputs: The node's outputs, by name.
        by_name: Takes the returned value apart into its outputs by name, leaving out each output that is
            ``SKIPPED``; it raises ``PydanticCustomError`` for a value of the wrong shape.
    """
    names = list(outputs)
    groups: dict[str, list[str]] = {}
    for name, field in outputs.items():
        if field.param.exclusive_group is not None:
            groups.setdefault(field.param.exclusive_group, []).append(name)

    def taken_apart(returned: Any) -> dict[str, Any]:
        given = by_name(returned)
        for group, members in groups.items():
            chosen = [name for name in members if name in given]
            if len(chosen) > 1:
                raise PydanticCustomError(
                    "alternatives_both_given", "'{first}' and '{second}' are alternatives of '{group}', and both hold a value",
                    {"first": chosen[0], "second": chosen[1], "group": group},
                )
        return given

    def with_skipped(read: dict[str, Any]) -> dict[str, Any]:
        return read if len(read) == len(names) else {name: read.get(name, SKIPPED) for name in names}

    fields: dict[str, Any] = {name: core_schema.typed_dict_field(output_schema_of(field.whole), required=False) for name, field in outputs.items()}
    values: CoreSchema = core_schema.no_info_after_validator_function(with_skipped, core_schema.typed_dict_schema(fields))
    validate = SchemaValidator(core_schema.no_info_before_validator_function(taken_apart, values)).validate_python

    def read(returned: Any) -> dict[str, Any]:
        try:
            outputs_read: dict[str, Any] = validate(returned, strict=True)
        except ValidationError as error:
            raise _first_fault(outputs, error) from None
        return outputs_read

    return read


def _first_fault(outputs: Mapping[str, Field], error: ValidationError) -> RefusedError:
    """The ``invalid_output`` of the first fault pydantic's ``error`` names, raised by the validator of a node with
    several outputs.

    A fault of the whole value (not the dataclass, not a mapping, an output missing or unknown) and two outputs of
    one group both given name the output their message is about; a value an output's type refuses is said as that
    type says it, with every error pydantic found in that output.
    """
    first = error.errors()[0]
    ctx = first.get("ctx", {})
    if first["type"] == "alternatives_both_given":
        return RefusedError([run_problems.invalid_output(reason=first["msg"], name=ctx["second"])])
    if first["type"] in ("output_missing", "output_unknown"):
        return RefusedError([run_problems.invalid_output(reason=first["msg"], name=ctx["name"])])
    if not first["loc"]:
        return RefusedError([run_problems.invalid_output(reason=first["msg"])])
    name = str(first["loc"][0])
    return _refused_output(outputs[name], error, under=(name,))


def _refused_output(field: Field, error: ValidationError, under: tuple[str, ...] = ()) -> RefusedError:
    """The ``invalid_output`` of a value the type of the output ``field`` refused; ``error`` and ``under`` are as
    ``Type.refusal`` takes them."""
    refused = field.whole.refusal(error, under)
    return RefusedError([run_problems.invalid_output(reason=refused.reason, name=field.name, errors=refused.errors, column=refused.column)])


def _dataclass_fields(cls: type, names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Takes a returned frozen dataclass of outputs apart into its fields, by name."""

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, cls):
            raise PydanticCustomError("not_the_class", "the node returns a {cls}, not a {got}", {"cls": cls.__name__, "got": type(returned).__name__})
        return {name: value for name in names if (value := getattr(returned, name)) is not SKIPPED}

    return by_name


def _mapping_items(names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Takes a returned mapping of outputs as a dict, as a node whose ``compute_outputs`` hook gave its outputs
    returns them: the first output it lacks is a fault, then the first key it has that is no output, by name."""
    declared, listed = frozenset(names), comma_separated(names, empty="nothing")

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, Mapping):
            raise PydanticCustomError("not_a_mapping", "the node returns its outputs as a mapping of {names}, not a {got}", {"names": listed, "got": type(returned).__name__})
        for name in names:
            if name not in returned:
                raise PydanticCustomError("output_missing", "'{name}' is missing from the outputs it returns", {"name": name})
        if unknown := sorted(set(returned) - declared):
            raise PydanticCustomError("output_unknown", "'{name}' is not one of the outputs it returns", {"name": str(unknown[0])})
        return {name: value for name, value in returned.items() if value is not SKIPPED}

    return by_name
