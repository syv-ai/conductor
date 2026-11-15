"""The validators of what a node's code returns: what its function returns, read into its outputs, and what its
``compute_added_inputs`` or ``compute_outputs`` hook returns, read into fields when the graph is compiled.

The return validator. Example: ``shout(line: str) -> str`` returns ``"B!"``, and its validator gives
``{"result": "B!"}``. Returned ``3``, the validator raises ``InvalidCallError`` with the problem ``invalid_output``
for the output ``result``.

A value is read strictly: a value already of its class is kept as it is, a dict is read into a model, and ``"12"``
is never an ``int``. A node that returns ``SKIPPED`` gets ``SKIPPED`` for every output.

A node with several outputs returns a frozen dataclass, or a mapping when its ``compute_outputs`` hook gave the
outputs. Its validator reads every output in one call of one validator. First the shape: the dataclass or a mapping,
no output missing or unknown, and of each exclusive group exactly one output with a value (``field/exclusive_group.py``).
Then each output's value, by its type, in the order the outputs are declared. Only the first fault is reported, as
one problem.

Every rule inside the return validator raises ``PydanticCustomError``: any other exception would escape pydantic
with the value in its message.

The hook validator. Example: a node ``fields(names: str)`` whose ``compute_outputs`` hook, given ``names="a,b"``,
returns ``{"a": str, "b": (int, 0)}`` gets two outputs, ``a: str`` and ``b: int`` with the default 0. A hook returns
each field's type, or ``(type, default)`` as pydantic's ``create_model`` takes one. Given ``names="a.b"`` it returns
``{"a.b": str}``, which is the hook's failure, ``compute_hook_failed`` with ``error`` ``not_a_name``: a name holds
none of the characters places in a graph are written with (``Name``, the rule every name in a graph record is held
to).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    SchemaValidator,
    core_schema,
)

from conductor._internal.base import REQUIRED, SKIPPED, Name
from conductor._internal.field import Field, exclusive_groups, groups_without_one_value
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    compile_problems,
    run_problems,
)
from conductor._internal.utils import comma_separated
from conductor._internal.validation import (
    Validator,
    exact_classes,
    output_schema_of,
    schema_output_validator_of,
)


class ReturnValidator(Validator[Any, dict[str, Any]]):
    """Reads what one node's function returned into the node's outputs by name, as a run keeps them.

    Built once per compiled node, on the node's first call. Calling it with the returned value gives the outputs.

    Raises (when called):
        InvalidCallError: With one ``invalid_output`` problem, which names the output when there is one to name and quotes
            nothing of the value.
    """

    def __init__(
        self, outputs_dataclass: type | None, computes_outputs: bool, outputs: Mapping[str, Field]
    ) -> None:
        """Builds the validator for a node whose function returns ``outputs``.

        The function returns them as a frozen dataclass when it declares one, as a mapping when its
        ``compute_outputs`` hook gave them, as ``None`` when there are none, and otherwise as one value, ``result``.

        Args:
            outputs_dataclass: The frozen dataclass the function returns its outputs as, or ``None``.
            computes_outputs: Whether the node's ``compute_outputs`` hook gave its outputs.
            outputs: The node's outputs, by name.
        """
        self._names = tuple(outputs)
        self._read: Callable[[Any], dict[str, Any]]
        if outputs_dataclass is not None:
            self._read = _several_outputs(
                outputs, _dataclass_fields(outputs_dataclass, self._names)
            )
        elif computes_outputs:
            self._read = _several_outputs(outputs, _mapping_items(self._names))
        elif not outputs:
            self._read = _no_outputs
        else:
            self._read = _one_output(outputs["result"])

    def __call__(self, returned: Any) -> dict[str, Any]:
        """Returns the outputs of the node that returned ``returned``.

        Raises:
            InvalidCallError: The value is not what the node declares (``invalid_output``).
        """
        if returned is SKIPPED:
            return dict.fromkeys(self._names, SKIPPED)
        return self._read(returned)


def _no_outputs(returned: Any) -> dict[str, Any]:
    """The outputs of a node declared ``-> None``: none, when it returned ``None``."""
    if returned is not None:
        raise InvalidCallError(
            [
                run_problems.invalid_output(
                    reason=f"the node declares no outputs, but returned a {type(returned).__name__}"
                )
            ]
        )
    return {}


def _one_output(field: Field) -> Callable[[Any], dict[str, Any]]:
    """The validator of a node with the one output ``result``. A value exactly of one of the type's plain classes (a
    ``str`` for ``-> str``) is taken as it is, with no call to pydantic."""
    exact = exact_classes(field.whole)
    validate = schema_output_validator_of(field.whole).validate_python

    def read(returned: Any) -> dict[str, Any]:
        if type(returned) in exact:
            return {"result": returned}
        try:
            return {"result": validate(returned, strict=True)}
        except ValidationError as error:
            raise _refused_output(field, error) from None

    return read


def _several_outputs(
    outputs: Mapping[str, Field], by_name: Callable[[Any], dict[str, Any]]
) -> Callable[[Any], dict[str, Any]]:
    """The validator of a node with several outputs: one pydantic validator for all of them.

    First the shape of what was returned is checked, then each output's value: the before-function takes the
    value apart by output name and holds each exclusive group to one output with a value; the fields read the
    values; the after-function gives each output that was ``SKIPPED`` its ``SKIPPED`` back.

    Args:
        outputs: The node's outputs, by name.
        by_name: Takes the returned value apart into its outputs by name, leaving out each output that is
            ``SKIPPED``; it raises ``PydanticCustomError`` for a value of the wrong shape.
    """
    names = list(outputs)
    groups = exclusive_groups(outputs)

    def taken_apart(returned: Any) -> dict[str, Any]:
        given = by_name(returned)
        fault = next(groups_without_one_value(groups, given), None)
        if fault is None:
            return given
        if len(fault.given) > 1:
            quoted = [f"'{name}'" for name in fault.given]
            holders = f"{comma_separated(quoted[:-1])} and {quoted[-1]}"
            holding = "both" if len(quoted) == 2 else "all"
            raise PydanticCustomError(
                "alternatives_both_given",
                f"{holders} are alternatives of '{fault.group}', and {holding} hold a value",
                {"given": fault.given, "group": fault.group},
            )
        raise PydanticCustomError(
            "no_alternative_given",
            "none of the exclusive group '{group}' ({members}) holds a value; give one of them a value and the "
            "others SKIPPED, or return SKIPPED to choose no branch",
            {"group": fault.group, "members": comma_separated(fault.members)},
        )

    def with_skipped(read: dict[str, Any]) -> dict[str, Any]:
        return (
            read if len(read) == len(names) else {name: read.get(name, SKIPPED) for name in names}
        )

    fields: dict[str, Any] = {
        name: core_schema.typed_dict_field(output_schema_of(field.whole), required=False)
        for name, field in outputs.items()
    }
    values: CoreSchema = core_schema.no_info_after_validator_function(
        with_skipped, core_schema.typed_dict_schema(fields)
    )
    validate = SchemaValidator(
        core_schema.no_info_before_validator_function(taken_apart, values)
    ).validate_python

    def read(returned: Any) -> dict[str, Any]:
        try:
            outputs_read: dict[str, Any] = validate(returned, strict=True)
        except ValidationError as error:
            raise _first_fault(outputs, error) from None
        return outputs_read

    return read


def _first_fault(outputs: Mapping[str, Field], error: ValidationError) -> InvalidCallError:
    """The ``invalid_output`` of the first fault pydantic's ``error`` names, raised by the validator of a node with
    several outputs.

    A fault of the whole value (not the dataclass, not a mapping, an output missing or unknown) and two outputs of
    one group both given name the output their message is about; a value an output's type refuses is said as that
    type says it, with every error pydantic found in that output.
    """
    first = error.errors()[0]
    ctx = first.get("ctx", {})
    if first["type"] == "alternatives_both_given":
        return InvalidCallError(
            [run_problems.invalid_output(reason=first["msg"], name=ctx["given"][1])]
        )
    if first["type"] in ("output_missing", "output_unknown"):
        return InvalidCallError(
            [run_problems.invalid_output(reason=first["msg"], name=ctx["name"])]
        )
    if not first["loc"]:
        return InvalidCallError([run_problems.invalid_output(reason=first["msg"])])
    name = str(first["loc"][0])
    return _refused_output(outputs[name], error, under=(name,))


def _refused_output(
    field: Field, error: ValidationError, under: tuple[str, ...] = ()
) -> InvalidCallError:
    """The ``invalid_output`` of a value the type of the output ``field`` refused; ``error`` and ``under`` are as
    ``Type.refusal`` takes them."""
    refused = field.whole.refusal(error, under)
    return InvalidCallError(
        [
            run_problems.invalid_output(
                reason=refused.reason, name=field.name, errors=refused.errors, column=refused.column
            )
        ]
    )


def _dataclass_fields(cls: type, names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Takes a returned frozen dataclass of outputs apart into its fields, by name."""

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, cls):
            raise PydanticCustomError(
                "not_the_class",
                "the node returns a {cls}, not a {got}",
                {"cls": cls.__name__, "got": type(returned).__name__},
            )
        return {name: value for name in names if (value := getattr(returned, name)) is not SKIPPED}

    return by_name


def _mapping_items(names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Takes a returned mapping of outputs as a dict, as a node whose ``compute_outputs`` hook gave its outputs
    returns them: the first output it lacks is a fault, then the first key it has that is no output, by name."""
    declared, listed = frozenset(names), comma_separated(names, empty="nothing")

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, Mapping):
            raise PydanticCustomError(
                "not_a_mapping",
                "the node returns its outputs as a mapping of {names}, not a {got}",
                {"names": listed, "got": type(returned).__name__},
            )
        for name in names:
            if name not in returned:
                raise PydanticCustomError(
                    "output_missing",
                    "'{name}' is missing from the outputs it returns",
                    {"name": name},
                )
        if unknown := sorted(set(returned) - declared):
            raise PydanticCustomError(
                "output_unknown",
                "'{name}' is not one of the outputs it returns",
                {"name": str(unknown[0])},
            )
        return {name: value for name, value in returned.items() if value is not SKIPPED}

    return by_name


#: What a hook returns: a type, or ``(type, default)``, for each name.
_HOOK_FIELDS = TypeAdapter(dict[Name, Any])


def read_hook_fields(
    node_id: str, kind: str, returned: Any, read_field: Callable[[str, Any, Any], Field]
) -> dict[str, Field]:
    """Returns what the hook ``kind`` of the node ``node_id`` returned, as fields by name, each read by
    ``read_field`` as ``@node`` reads one: ``declared_input`` for ``compute_added_inputs``, ``declared_output`` for
    ``compute_outputs``.

    Raises:
        InvalidCallError: ``compute_hook_failed``, one at each name that cannot be one (``error`` ``not_a_name``), or
            one at the node when the hook returned no mapping (``error`` ``not_a_mapping``), or a field ``@node``
            would refuse (``error`` the declaration code: ``type_not_carried`` for a type conductor does not support,
            ``exclusive_group_on_input`` or ``default_invalid`` for an added input that is a branch, ``default_invalid``
            for an output of an exclusive group with a default other than ``SKIPPED``, ``constant_only_on_output``).
    """
    try:
        hints = _HOOK_FIELDS.validate_python(returned)
    except ValidationError as invalid:
        if not invalid.errors()[0]["loc"]:  # the value returned is not a mapping
            reason = f"its {kind} hook returned a {type(returned).__name__}; a hook returns its fields by name, as a mapping of each name to its type"
            raise InvalidCallError(
                [
                    compile_problems.compute_hook_failed(
                        node_id, reason=reason, error="not_a_mapping"
                    )
                ]
            ) from None
        raise InvalidCallError(
            compile_problems.compute_hook_failed(
                node_id,
                str(fault["loc"][0]),
                reason=f"what its {kind} hook returned has a field name it cannot have ({fault['msg']})",
                error="not_a_name",
            )
            for fault in invalid.errors()
        ) from None
    try:
        return {name: read_field(name, *_hint_and_default(hint)) for name, hint in hints.items()}
    except NodeDeclarationError as error:  # a field @node would refuse is the hook's failure
        raise InvalidCallError(
            [
                compile_problems.compute_hook_failed(
                    node_id, reason=str(error), error=error.problems[0].type
                )
            ]
        ) from None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """Reads what a hook returns for one field: ``(type, default)``, or a type alone, which then has no default."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED
