"""One call of a node: what it receives checked where its edges don't settle it, and what it returns checked
against its declared outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pydantic_core

from conductor._internal.compiled import FlatFunctionNode
from conductor._internal.field import Field
from conductor._internal.markers import SKIPPED
from conductor._internal.node_definition import ReturnShape
from conductor._internal.problem import CallRejectedError
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import FieldValueError, untyped_as_json


def call(node: FlatFunctionNode, arguments: dict[str, Any], context: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
    """Call ``node`` on what a unit received: its outputs as a run keeps them, or what went wrong.

    A value an edge brings is checked against the input first where its source's type doesn't settle it, and what
    the node returns against its declared outputs; either refusal is a ``CallRejectedError``, whose problem quotes no value.
    """
    compiled_node = node.compiled_node
    definition = compiled_node.definition
    try:
        for name in node.validated_inputs:
            arguments = {**arguments, name: _checked(compiled_node.inputs[name].field, name, arguments[name], returned=False)}
        if definition.context:
            arguments = {**arguments, **{name: context[needed] for name, needed in definition.context.items()}}
        returned = definition.function(**arguments)
    except BaseException as error:  # noqa: BLE001  a node's own failure, or its Ask, is the run's to record
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return None, error
    try:
        return checked_outputs(definition.returns, compiled_node.outputs, returned), None
    except CallRejectedError as wrong:
        return None, wrong


def checked_outputs(returns: ReturnShape, outputs: Mapping[str, Field], returned: Any) -> dict[str, Any]:
    """What a node returned, as its outputs by name as a run keeps them; ``SKIPPED`` returned for the whole node
    is ``SKIPPED`` on every output.

    Each is read strictly by its declared type (a value already of its class as it is, a model from a dict, never
    ``"12"`` for an ``int``), and of an ``exclusive_group`` at most one output holds a value.

    Args:
        returns: How the node's function returns its outputs.
        outputs: The node's outputs, by name.
        returned: What the function returned.

    Raises:
        CallRejectedError: An output is not of its type, or two of one group hold a value (``invalid_output``).
    """
    if returned is SKIPPED:
        return dict.fromkeys(outputs, SKIPPED)
    spread, settled = returns.spread(returned, outputs)
    if settled:
        return spread
    groups: dict[str, str] = {}
    for name, field in outputs.items():
        spread[name] = _checked(field, name, spread[name], returned=True)
        group = field.param.exclusive_group
        if group is not None and spread[name] is not SKIPPED:
            if group in groups:
                raise CallRejectedError(problem.invalid_output(reason=f"'{groups[group]}' and '{name}' are alternatives of '{group}', and both hold a value", name=name))
            groups[group] = name
    return spread


def _checked(field: Field, name: str, value: Any, *, returned: bool) -> Any:
    """``value`` as the field ``name`` reads it: an output a node returned, or what an edge brought to an input.

    Raises:
        CallRejectedError: The field's type refuses the value: ``invalid_output`` for a return, else ``invalid_edge_value``.
    """
    if value is SKIPPED:
        return value
    try:
        read = field.whole.read_returned(value) if returned else field.whole.read(value)
    except FieldValueError as refused:
        if returned:
            raise CallRejectedError(problem.invalid_output(reason=refused.reason, name=name, errors=refused.errors, column=refused.column)) from None
        raise CallRejectedError(problem.invalid_edge_value(name=name, reason=refused.reason, errors=refused.errors, column=refused.column)) from None
    exact = field.whole.exact_classes
    if returned and exact and not isinstance(read, exact):
        raise CallRejectedError(problem.invalid_output(reason=f"its type's own check gave a {type(read).__name__}, not a {field.whole.id}", name=name))
    return read




def jsonable(arguments: dict[str, Any]) -> Any:
    """What a unit received, as JSON; a float that isn't finite as text. ``CallRejectedError`` as ``question_unwritable`` names
    the first input with no JSON form."""
    try:
        return pydantic_core.from_json(pydantic_core.to_json(arguments, fallback=untyped_as_json, inf_nan_mode="strings"))
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError) as error:
        for name, value in arguments.items():
            try:
                pydantic_core.to_json(value, fallback=untyped_as_json, inf_nan_mode="strings")
            except (TypeError, ValueError, pydantic_core.PydanticSerializationError) as unwritable:
                raise CallRejectedError(problem.question_unwritable(reason=str(unwritable), input=name)) from None
        raise error
