"""Checking what a node's own code returns: its function on each call, its hooks when a graph is compiled, and its upgrades.

``ReturnValidator`` checks what a node's function returns, and turns it into the node's outputs by name. Say
``shout(line: str) -> str`` returns ``"B!"``. The validator gives ``{"result": "B!"}``. If it returns ``3``
instead, the call fails with an ``invalid_output`` problem on the output ``result``.

Values are checked strictly. A value that already has the right class is kept as it is, a dict becomes a model,
but ``"12"`` is never an ``int``. A node with several outputs returns a frozen dataclass with no ``@host_type``, or
a mapping when a ``compute_outputs`` hook gave its outputs. Its whole return is checked in one go, by the rules a
person's answer to the node is checked by too (``outputs_schema``), and only the first fault is reported: a fault in
the shape before a fault in a value. ``SKIPPED`` may be an output's whole value, or an item of a
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
from pydantic_core import ErrorDetails, PydanticCustomError, SchemaValidator, core_schema

from conductor._internal.base import REQUIRED, SKIPPED, Name
from conductor._internal.field import (
    DeclaredInputValidator,
    DeclaredOutputValidator,
    Field,
    outputs_schema,
)
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    call_problems,
    compile_problems,
    conductor_error,
    run_problems,
    shape_error,
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
        self._outputs = outputs
        by_name: Callable[[Any], dict[str, Any]]
        if outputs_dataclass is not None:
            by_name = _dataclass_fields(outputs_dataclass, outputs)
        elif computes_outputs:
            by_name = _mapping_items(outputs)
        elif not outputs:
            by_name = _no_outputs
        else:
            by_name = _one_output
        # an output a node can't return SKIPPED on must be returned; a branch may be left out of a mapping
        fields = {
            name: core_schema.typed_dict_field(
                field.whole.output_schema, required=not field.is_branch
            )
            for name, field in outputs.items()
        }
        self._validate = SchemaValidator(outputs_schema(outputs, fields, by_name)).validate_python

    def __call__(self, returned: Any) -> dict[str, Any]:
        """Checks what the function returned, and returns the node's outputs by name.

        Raises:
            InvalidCallError: See the class.
        """
        if returned is SKIPPED:
            return dict.fromkeys(self._outputs, SKIPPED)
        try:
            read: dict[str, Any] = self._validate(returned, strict=True)
        except ValidationError as error:
            raise _first_fault(self._outputs, error, returned) from None
        for name, value in read.items():
            if _skipped_inside(self._outputs[name], value):
                raise _refused_shape(
                    conductor_error(
                        "skipped_inside",
                        "SKIPPED is inside the value, where it has no JSON form: only an output's whole value, or an "
                        "item of a Series, may be SKIPPED",
                    ),
                    output=name,
                )
        return read


def _skipped_inside(field: Field, value: Any) -> bool:
    """Whether an output's value has ``SKIPPED`` inside it, below the output's whole value and a ``Series``'s items."""
    items = value if field.is_series and isinstance(value, list) else [value]
    return any(holds_skipped(item) for item in items if item is not SKIPPED)


def _one_output(returned: Any) -> dict[str, Any]:
    """Splits the return of a node with one output, ``result``: the value it returns is that output's."""
    return {"result": returned}


def _no_outputs(returned: Any) -> dict[str, Any]:
    """Splits the return of a node declared ``-> None``, which must be ``None``, into no outputs."""
    if returned is not None:
        raise conductor_error(
            "none_required",
            "the node declares no outputs, but returned a {got}",
            {"got": type(returned).__name__},
        )
    return {}


def _dataclass_fields(cls: type, outputs: Mapping[str, Field]) -> Callable[[Any], dict[str, Any]]:
    """Builds the function that splits a returned frozen dataclass into its outputs, by name."""

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, cls):
            raise conductor_error(
                "not_the_class",
                "the node must return a {cls}, not a {got}",
                {"cls": cls.__name__, "got": type(returned).__name__},
            )
        return _given(outputs, {name: getattr(returned, name) for name in outputs})

    return by_name


def _mapping_items(outputs: Mapping[str, Field]) -> Callable[[Any], dict[str, Any]]:
    """Builds the function that splits a returned mapping of outputs, from a node whose ``compute_outputs`` hook gave
    its outputs."""
    listed = comma_separated(outputs, empty="nothing")

    def by_name(returned: Any) -> dict[str, Any]:
        if not isinstance(returned, Mapping):
            raise conductor_error(
                "not_a_mapping",
                "the node must return a mapping with one key per output ({names}), not a {got}",
                {"names": listed, "got": type(returned).__name__},
            )
        return _given(outputs, returned)

    return by_name


def _given(outputs: Mapping[str, Field], returned: Mapping[Any, Any]) -> dict[str, Any]:
    """Returns the outputs a return gives a value, by name, leaving out each branch that is ``SKIPPED``.

    Only a branch may be ``SKIPPED``. A name that isn't an output is kept, for the schema to refuse.

    Raises:
        PydanticCustomError: ``skipped_not_a_branch`` for the first output that is ``SKIPPED`` but isn't a branch.
    """
    given: dict[str, Any] = {}
    for name, value in returned.items():
        field = outputs.get(name)
        if value is not SKIPPED or field is None:
            given[name] = value
        elif not field.is_branch:
            raise conductor_error(
                "skipped_not_a_branch",
                "'{name}' is SKIPPED, but it isn't a branch: its default isn't SKIPPED",
                {"name": name},
            )
    return given


def _refused_shape(
    fault: ErrorDetails | PydanticCustomError, output: str | None = None
) -> InvalidCallError:
    """Turns a fault in the shape of what a node returned into an ``invalid_output`` error.

    Args:
        fault: The fault, as pydantic gave it, or as the error conductor raises for it.
        output: The output the fault is about, when there is one to name.
    """
    error = shape_error(fault)
    return InvalidCallError(
        [run_problems.invalid_output(reason=error["msg"], output=output, errors=[error])]
    )


def _first_fault(
    outputs: Mapping[str, Field], error: ValidationError, returned: Any
) -> InvalidCallError:
    """Turns the first fault in what a node returned into an ``invalid_output`` error.

    A fault in the return's shape comes first: not the dataclass, not a mapping, an output that isn't a branch is
    ``SKIPPED``, then an output missing from a returned mapping, a key that isn't an output, and two branches with a
    value. A fault in an output's value comes after them. An output missing, or a key that isn't an output, is
    pydantic's own ``missing``, ``extra_forbidden`` or ``invalid_key``, at the output's name, as an answer to the
    node gives it, while the reason says it of a return. A value that an output's type refuses is described as the
    type describes it, with every error pydantic found in that output.
    """

    def about_a_name(fault: ErrorDetails) -> str | None:
        """The reason for a fault about an output's name rather than its value, or ``None``."""
        if len(fault["loc"]) != 1:
            return None
        name = fault["loc"][0]
        if name not in outputs:
            return f"'{name}' isn't one of the node's outputs"
        if fault["type"] == "missing" and isinstance(returned, Mapping) and name not in returned:
            return f"'{name}' is missing from the outputs the node returned"
        return None

    faults = error.errors()
    first = next((fault for fault in faults if not fault["loc"] or about_a_name(fault)), faults[0])
    if not first["loc"]:
        ctx = first.get("ctx", {})
        return _refused_shape(
            first, ctx["given"][1] if first["type"] == "several_branches_given" else ctx.get("name")
        )
    name = str(first["loc"][0])
    if reason := about_a_name(first):
        return InvalidCallError(
            [
                run_problems.invalid_output(
                    reason=reason, output=name, errors=[shape_error(first, loc=[name])]
                )
            ]
        )
    return InvalidCallError(
        [
            run_problems.invalid_output(
                output=name, **outputs[name].whole.refusal(error, under=(name,))
            )
        ]
    )


#: Checks what a hook returns: a mapping of valid names, each to a type or to ``(type, default)``.
_HOOK_FIELDS = TypeAdapter(dict[Name, Any])


class HookValidator(Validator[Any, dict[str, Field]]):
    """A validator for what one hook of a node returns.

    A hook returns its fields as a mapping of name to type, or to ``(type, default)``. The validator turns each into
    a ``Field``, held to the same rules ``@node`` holds a declared field to. Fields from ``compute_added_inputs`` are
    checked like inputs, with ``DeclaredInputValidator``. Fields from ``compute_outputs`` are checked like outputs,
    with ``DeclaredOutputValidator``.

    Raises (when called):
        InvalidCallError: ``compute_hook_returned_invalid`` problems, each with one error in ``ctx.errors``. There is
            one at each name that isn't a valid name (``not_a_name``). If the hook returned no mapping, there is one
            at the node (``dict_type``). If ``@node`` would refuse a field, its error's ``type`` is the declaration's
            code, like ``default_invalid``, and its message names the field but quotes none of the declaration's
            text, which can quote the default.
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
                            self._node_id, reason=reason, errors=[shape_error(invalid.errors()[0])]
                        )
                    ]
                ) from None
            raise InvalidCallError(
                compile_problems.compute_hook_returned_invalid(
                    self._node_id,
                    str(fault["loc"][0]),
                    reason=f"its {self._hook} hook returned a name a field can't have ({fault['msg']})",
                    errors=[shape_error(fault)],
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
                refused = shape_error(
                    conductor_error(
                        error.problems[0].type,
                        "its {hook} hook gave '{name}' a type or a default that @node refuses",
                        {"hook": self._hook, "name": name},
                    )
                )
                raise InvalidCallError(
                    [
                        compile_problems.compute_hook_returned_invalid(
                            self._node_id, reason=refused["msg"], errors=[refused]
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
        InvalidCallError: ``cannot_move_to_version_rewrite_returned_invalid`` at the node, with pydantic's error in
            ``ctx.errors``, if the rewrite returned no dict, or one with a key that isn't text. ``invalid_value`` at the
            node and input for a value that can't be written as JSON.
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
            first = invalid.errors()[0]
            raise InvalidCallError(
                [
                    call_problems.cannot_move_to_version_rewrite_returned_invalid(
                        self._node_id,
                        node=self._node,
                        version=self._version,
                        target=self._target,
                        reason=f"a {type(returned).__name__}"
                        if not first["loc"]
                        else "a dict with a key that isn't text",
                        errors=[shape_error(first)],
                    )
                ]
            ) from None
        return WrittenValuesValidator(self._node_id)(values)


#: Checks what an ``Upgrade``'s rewrite returns: a dict whose keys are text.
_REWRITTEN_VALUES = TypeAdapter(dict[str, Any])
