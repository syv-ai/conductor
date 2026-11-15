"""Checking what a node's own code returns: its function on each call, and its hooks when a graph is compiled.

``ReturnValidator`` checks what a node's function returns, and turns it into the node's outputs by name. Say
``shout(line: str) -> str`` returns ``"B!"``. The validator gives ``{"result": "B!"}``. If it returns ``3``
instead, the call fails with an ``invalid_output`` problem on the output ``result``.

Values are checked strictly. A value that already has the right class is kept as it is, a dict becomes a model,
but ``"12"`` is never an ``int``. A node with several outputs returns a frozen dataclass, or a mapping when a
``compute_outputs`` hook gave its outputs. Its whole return is checked in one go: first the shape, then each value.
Only the first fault is reported.

``HookValidator`` checks what a ``compute_added_inputs`` or ``compute_outputs`` hook returns, and turns it into
fields. Say a node's ``compute_outputs`` hook, given ``names="a,b"``, returns ``{"a": str, "b": (int, 0)}``. The
node gets two outputs, ``a: str``, and ``b: int`` with the default 0. Each field is held to the same rules as one
``@node`` declares. A hook that returns ``{"a.b": str}`` fails, since ``a.b`` isn't a valid name.
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
from conductor._internal.field import (
    DeclaredInputValidator,
    DeclaredOutputValidator,
    Field,
    exclusive_groups,
    groups_without_one_value,
)
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

from .type import HookName


class ReturnValidator(Validator[Any, dict[str, Any]]):
    """A validator for what a node's function returns.

    It checks the returned value against the node's outputs, and returns the outputs by name. A node that returns
    ``SKIPPED`` gets ``SKIPPED`` on every output.

    One is built for each node of a compiled graph, on the node's first call.

    Raises (when called):
        InvalidCallError: One ``invalid_output`` problem. It names the output when there is one to name, and never
            quotes the value.
    """

    def __init__(
        self, outputs_dataclass: type | None, computes_outputs: bool, outputs: Mapping[str, Field]
    ) -> None:
        """Builds the validator for one node.

        A function returns its outputs in one of four shapes. It returns a frozen dataclass if it declares one, a
        mapping if its ``compute_outputs`` hook gave the outputs, and ``None`` if it has none. Otherwise it returns
        one value, the output ``result``.

        Args:
            outputs_dataclass: The frozen dataclass the function returns its outputs in, or ``None``.
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
        """Checks what the function returned, and returns the node's outputs by name.

        Raises:
            InvalidCallError: See the class.
        """
        if returned is SKIPPED:
            return dict.fromkeys(self._names, SKIPPED)
        return self._read(returned)


def _no_outputs(returned: Any) -> dict[str, Any]:
    """Checks the return of a node declared ``-> None``, which must be ``None``, and returns no outputs."""
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
    """Builds the check for a node with one output, ``result``.

    A value whose class is exactly one of the type's plain classes, like a ``str`` for ``-> str``, is kept as it is
    without calling pydantic.
    """
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
    """Builds the check for a node with several outputs, as one pydantic validator for all of them.

    It works in three steps. First, ``taken_apart`` splits the returned value by output name, and checks that each
    exclusive group has exactly one output with a value. Then pydantic checks each output's value against its type.
    Last, ``with_skipped`` puts ``SKIPPED`` back on each output that was skipped.

    Every check raises ``PydanticCustomError``. Any other exception would escape pydantic with the value in its
    message.

    Args:
        outputs: The node's outputs, by name.
        by_name: Splits the returned value into its outputs by name, leaving out each output that is ``SKIPPED``.
            It raises ``PydanticCustomError`` for a value of the wrong shape.
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
                f"{holders} are in the exclusive group '{fault.group}', and {holding} have a value. Give only one of them a value",
                {"given": fault.given, "group": fault.group},
            )
        raise PydanticCustomError(
            "no_alternative_given",
            "no output of the exclusive group '{group}' ({members}) has a value. Give one of them a value and the "
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
    """Turns the first fault in a node's several outputs into an ``invalid_output`` error.

    Some faults are about the shape of the whole value: not the dataclass, not a mapping, or an output missing or
    unknown. Others are about two outputs of one group both having a value. These name the output their message is
    about. A value that an output's type refuses is described as the type describes it, with every error pydantic
    found in that output.
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
    """Turns a value that an output's type refused into an ``invalid_output`` error.

    Args:
        field: The output.
        error: The error pydantic raised, as ``Type.refusal`` takes it.
        under: Where the output's value sits in ``error``, as ``Type.refusal`` takes it.
    """
    refused = field.whole.refusal(error, under)
    return InvalidCallError(
        [
            run_problems.invalid_output(
                reason=refused.reason, name=field.name, errors=refused.errors, column=refused.column
            )
        ]
    )


def _dataclass_fields(cls: type, names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Builds the function that splits a returned frozen dataclass into its outputs, by name."""

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, cls):
            raise PydanticCustomError(
                "not_the_class",
                "the node must return a {cls}, not a {got}",
                {"cls": cls.__name__, "got": type(returned).__name__},
            )
        return {name: value for name in names if (value := getattr(returned, name)) is not SKIPPED}

    return by_name


def _mapping_items(names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Builds the function that checks a returned mapping of outputs, from a node whose ``compute_outputs`` hook gave
    its outputs.

    The first output missing from the mapping is the fault. If none is missing, the first key that isn't an output
    is.
    """
    declared, listed = frozenset(names), comma_separated(names, empty="nothing")

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, Mapping):
            raise PydanticCustomError(
                "not_a_mapping",
                "the node must return its outputs as a mapping of {names}, not a {got}",
                {"names": listed, "got": type(returned).__name__},
            )
        for name in names:
            if name not in returned:
                raise PydanticCustomError(
                    "output_missing",
                    "'{name}' is missing from the outputs the node returned",
                    {"name": name},
                )
        if unknown := sorted(set(returned) - declared):
            raise PydanticCustomError(
                "output_unknown",
                "'{name}' isn't one of the node's outputs",
                {"name": str(unknown[0])},
            )
        return {name: value for name, value in returned.items() if value is not SKIPPED}

    return by_name


#: Checks what a hook returns: a mapping of valid names, each to a type or to ``(type, default)``.
_HOOK_FIELDS = TypeAdapter(dict[Name, Any])


class HookValidator(Validator[Any, dict[str, Field]]):
    """A validator for what one hook of a node returns.

    A hook returns its fields as a mapping of name to type, or to ``(type, default)``. The validator turns each into
    a ``Field``, held to the same rules ``@node`` holds a declared field to. Fields from ``compute_added_inputs`` are
    checked like inputs, with ``DeclaredInputValidator``. Fields from ``compute_outputs`` are checked like outputs,
    with ``DeclaredOutputValidator``.

    Raises (when called):
        InvalidCallError: ``compute_hook_failed`` problems. There is one at each name that isn't a valid name
            (``error`` ``not_a_name``). If the hook returned no mapping, there is one at the node (``not_a_mapping``).
            If ``@node`` would refuse a field, its ``error`` is the declaration's code, like ``default_invalid``.
    """

    def __init__(self, node_id: str, hook: HookName) -> None:
        """Builds the validator for one hook of one node.

        Args:
            node_id: The node's id. The problems name it.
            hook: Which hook it is.
        """
        self._node_id = node_id
        self._hook = hook
        self._read_field: type[DeclaredInputValidator | DeclaredOutputValidator] = (
            DeclaredInputValidator if hook == "compute_added_inputs" else DeclaredOutputValidator
        )

    def __call__(self, returned: Any) -> dict[str, Field]:
        """Checks what the hook returned, and returns its fields by name.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            hints = _HOOK_FIELDS.validate_python(returned)
        except ValidationError as invalid:
            if not invalid.errors()[0]["loc"]:  # no location: not a mapping
                reason = (
                    f"its {self._hook} hook returned a {type(returned).__name__}, not a mapping of each field's name"
                    " to its type"
                )
                raise InvalidCallError(
                    [
                        compile_problems.compute_hook_failed(
                            self._node_id, reason=reason, error="not_a_mapping"
                        )
                    ]
                ) from None
            raise InvalidCallError(
                compile_problems.compute_hook_failed(
                    self._node_id,
                    str(fault["loc"][0]),
                    reason=f"its {self._hook} hook returned a name a field can't have ({fault['msg']})",
                    error="not_a_name",
                )
                for fault in invalid.errors()
            ) from None
        try:
            fields: dict[str, Field] = {}
            for name, given in hints.items():
                hint, default = _hint_and_default(given)
                fields[name] = self._read_field(name, hint)(default)
            return fields
        except NodeDeclarationError as error:  # @node would refuse it: the hook failed
            raise InvalidCallError(
                [
                    compile_problems.compute_hook_failed(
                        self._node_id, reason=str(error), error=error.problems[0].type
                    )
                ]
            ) from None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """Splits what a hook returns for one field into its type and its default.

    A hook gives ``(type, default)``, or a type alone. A type alone has no default, so it gets ``REQUIRED``.
    """
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED
