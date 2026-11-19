"""Checking what a node's own code returns: its function on each call, its hooks when a graph is compiled, and its upgrades.

``ReturnValidator`` checks what a node's function returns, and turns it into the node's outputs by name. Say
``shout(line: str) -> str`` returns ``"B!"``. The validator gives ``{"result": "B!"}``. If it returns ``3``
instead, the call fails with an ``invalid_output`` problem on the output ``result``.

Values are checked strictly. A value that already has the right class is kept as it is, a dict becomes a model,
but ``"12"`` is never an ``int``. A node with several outputs returns a frozen dataclass with no ``@host_type``, or
a mapping when a ``compute_outputs`` hook gave its outputs. Its whole return is checked in one go: first the shape,
then each value. Only the first fault is reported. ``SKIPPED`` may be an output's whole value, or an item of a
``Series``, but nowhere inside a value, where it has no JSON form.

``HookValidator`` checks what a ``compute_added_inputs`` or ``compute_outputs`` hook returns, and turns it into
fields. Say a node's ``compute_outputs`` hook, given ``names="a,b"``, returns ``{"a": str, "b": (int, 0)}``. The
node gets two outputs, ``a: str``, and ``b: int`` with the default 0. Each field is held to the same rules as one
``@node`` declares. A hook that returns ``{"a.b": str}`` fails, since ``a.b`` isn't a valid name.

``RewriteValidator`` checks what the ``rewrite`` of a node's ``Upgrade`` returns, when ``registry.upgrade`` moves a
graph record to a newer version of the node. It must be a dict of input name to value, and each value is written as
JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import (
    CoreSchema,
    SchemaValidator,
    core_schema,
)

from conductor._internal.base import REQUIRED, SKIPPED, Name
from conductor._internal.field import (
    DeclaredInputValidator,
    DeclaredOutputValidator,
    Field,
    check_branches,
)
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    call_problems,
    compile_problems,
    conductor_error,
    run_problems,
)
from conductor._internal.record import WrittenValuesValidator
from conductor._internal.utils import comma_separated
from conductor._internal.validation import Validator, holds_skipped

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

        A function returns its outputs in one of four shapes. It returns a frozen dataclass if it declares one with
        no ``@host_type``, a mapping if its ``compute_outputs`` hook gave the outputs, and ``None`` if it has none.
        Otherwise it returns one value, the output ``result``.

        Args:
            outputs_dataclass: The frozen dataclass the function returns its outputs in, or ``None``.
            computes_outputs: Whether the node's ``compute_outputs`` hook gave its outputs.
            outputs: The node's outputs, by name.
        """
        self._names = tuple(outputs)
        self._outputs = outputs
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
        read = self._read(returned)
        for name, value in read.items():
            if _skipped_inside(self._outputs[name], value):
                raise InvalidCallError(
                    [
                        run_problems.invalid_output(
                            reason="SKIPPED is inside the value, where it has no JSON form: only an output's whole "
                            "value, or an item of a Series, may be SKIPPED",
                            output=name,
                        )
                    ]
                )
        return read


def _skipped_inside(field: Field, value: Any) -> bool:
    """Whether an output's value has ``SKIPPED`` inside it, below the output's whole value and a ``Series``'s items."""
    items = value if field.is_series and isinstance(value, list) else [value]
    return any(holds_skipped(item) for item in items if item is not SKIPPED)


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
    exact = field.whole.exact_classes
    validate = field.whole.output_validator.validate_python

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

    It works in three steps. First, ``taken_apart`` splits the returned value by output name. It checks that only
    branches are ``SKIPPED``, and that at most one branch has a value. Then pydantic checks each output's value
    against its type. Last, ``with_skipped`` puts ``SKIPPED`` back on each output that was skipped.

    Every check raises ``PydanticCustomError``. Any other exception would escape pydantic with the value in its
    message.

    Args:
        outputs: The node's outputs, by name.
        by_name: Splits the returned value into its outputs by name, leaving out each output that is ``SKIPPED``.
            It raises ``PydanticCustomError`` for a value of the wrong shape.
    """
    names = list(outputs)

    def taken_apart(returned: Any) -> dict[str, Any]:
        given = by_name(returned)
        not_branches = [name for name in names if name not in given and not outputs[name].is_branch]
        if not_branches:
            raise conductor_error(
                "skipped_not_a_branch",
                "'{name}' is SKIPPED, but it isn't a branch: its default isn't SKIPPED",
                {"name": not_branches[0]},
            )
        check_branches(outputs, given)
        return given

    def with_skipped(read: dict[str, Any]) -> dict[str, Any]:
        return (
            read if len(read) == len(names) else {name: read.get(name, SKIPPED) for name in names}
        )

    fields: dict[str, Any] = {
        name: core_schema.typed_dict_field(field.whole.output_schema, required=False)
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
    unknown. Others are about branches: an output that isn't a branch is ``SKIPPED``, or two branches have a value.
    These name the output their message is about. A value that an output's type refuses is described as the type describes it, with every error pydantic
    found in that output.
    """
    first = error.errors()[0]
    ctx = first.get("ctx", {})
    if first["type"] == "several_branches_given":
        return InvalidCallError(
            [run_problems.invalid_output(reason=first["msg"], output=ctx["given"][1])]
        )
    if first["type"] in ("output_missing", "output_unknown", "skipped_not_a_branch"):
        return InvalidCallError(
            [run_problems.invalid_output(reason=first["msg"], output=ctx["name"])]
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
    return InvalidCallError(
        [run_problems.invalid_output(output=field.name, **field.whole.refusal(error, under))]
    )


def _dataclass_fields(cls: type, names: tuple[str, ...]) -> Callable[[Any], dict[str, Any]]:
    """Builds the function that splits a returned frozen dataclass into its outputs, by name."""

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, cls):
            raise conductor_error(
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
            raise conductor_error(
                "not_a_mapping",
                "the node must return a mapping with one key per output ({names}), not a {got}",
                {"names": listed, "got": type(returned).__name__},
            )
        for name in names:
            if name not in returned:
                raise conductor_error(
                    "output_missing",
                    "'{name}' is missing from the outputs the node returned",
                    {"name": name},
                )
        if unknown := sorted(set(returned) - declared):
            raise conductor_error(
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
        InvalidCallError: ``compute_hook_returned_invalid`` problems. There is one at each name that isn't a valid name
            (``error`` ``not_a_name``). If the hook returned no mapping, there is one at the node (``not_a_mapping``).
            If ``@node`` would refuse a field, its ``error`` is the declaration's code, like ``default_invalid``, and
            its message names the field but quotes none of the declaration's text, which can quote the default.
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
                        compile_problems.compute_hook_returned_invalid(
                            self._node_id, reason=reason, error="not_a_mapping"
                        )
                    ]
                ) from None
            raise InvalidCallError(
                compile_problems.compute_hook_returned_invalid(
                    self._node_id,
                    str(fault["loc"][0]),
                    reason=f"its {self._hook} hook returned a name a field can't have ({fault['msg']})",
                    error="not_a_name",
                )
                for fault in invalid.errors()
            ) from None
        fields: dict[str, Field] = {}
        for name, given in hints.items():
            hint, default = _hint_and_default(given)
            try:
                fields[name] = self._read_field(name, hint)(default)
            # @node would refuse the field, so the hook failed. The declaration's message can quote the default the
            # hook gave, so the problem keeps only its code.
            except NodeDeclarationError as error:
                reason = (
                    f"its {self._hook} hook gave '{name}' a type or a default that @node refuses"
                )
                raise InvalidCallError(
                    [
                        compile_problems.compute_hook_returned_invalid(
                            self._node_id, reason=reason, error=error.problems[0].type
                        )
                    ]
                ) from None
        return fields


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """Splits what a hook returns for one field into its type and its default.

    A hook gives ``(type, default)``, or a type alone. A type alone has no default, so it gets ``REQUIRED``.
    """
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


class RewriteValidator(Validator[Any, dict[str, JsonValue]]):
    """A validator for what the ``rewrite`` of an ``Upgrade`` returns, for one step of ``registry.upgrade``.

    A rewrite returns the node's constants and graph-input defaults as a dict of input name to value. The validator
    checks that it is such a dict, and writes each value as the JSON a graph record stores. Say a rewrite returns
    ``{"day": date(2026, 1, 2)}``. The validator gives ``{"day": "2026-01-02"}``.

    Raises (when called):
        InvalidCallError: ``cannot_move_to_version_rewrite_failed``, at the node with ``error``
            ``not_a_mapping``, if the rewrite returned no dict, or one with a key that isn't text. ``invalid_value``
            at the node and input for a value that can't be written as JSON.
    """

    def __init__(self, node_id: str, node: str, version: int, target: int) -> None:
        """Builds the validator for one step of one node.

        Args:
            node_id: The node's id in the graph. The problems name it.
            node: The id of its node definition.
            version: The version the step moves from.
            target: The version the step moves to.
        """
        self._node_id = node_id
        self._node = node
        self._version = version
        self._target = target

    def __call__(self, returned: Any) -> dict[str, JsonValue]:
        """Checks what the rewrite returned, and returns it as JSON, by input name.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            values = _REWRITTEN_VALUES.validate_python(returned)
        except ValidationError as invalid:
            reason = (
                f"it returned a {type(returned).__name__}, not a dict of input names to values"
                if not invalid.errors()[0]["loc"]
                else "it returned a dict with a key that isn't text"
            )
            raise InvalidCallError(
                [
                    call_problems.cannot_move_to_version_rewrite_failed(
                        self._node_id,
                        node=self._node,
                        version=self._version,
                        target=self._target,
                        reason=reason,
                        error="not_a_mapping",
                    )
                ]
            ) from None
        return WrittenValuesValidator(self._node_id)(values)


#: Checks what an ``Upgrade``'s rewrite returns: a dict whose keys are text.
_REWRITTEN_VALUES = TypeAdapter(dict[str, Any])
