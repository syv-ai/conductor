"""The schema that reads a value for each output of a node, by output name.

Three things fill a node's outputs: what its function returns, a person's answer when it asks, and the default an
``Ask`` proposes. Each is given as one value per output. Say ``route`` has the outputs ``long`` and ``short``, both
branches, and ``note``. A return of ``Route(long="hi", note="n")`` and an answer of ``{"long": "hi", "note": "n"}``
are both read into ``{"long": "hi", "short": SKIPPED, "note": "n"}``.

``outputs_schema`` holds what is the same for all three. The whole value is split into a value per output name, as
the node returns its outputs or as a person answers them. Every name must be an output. Every output that must be
given is there. ``SKIPPED`` is taken only on a branch, an output whose default is ``SKIPPED`` (``read_skipped``).
Each output not given is ``SKIPPED``. And at most one branch has a value. The reference has the rule for branches,
under "Branches" (``python -m conductor.about the-node-contract``). How each output's value is read belongs to the
caller: a return is checked by ``ReturnValidator``, and an answer and an ``Ask``'s default by the validators of
``ask/``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Container, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import (
    CoreSchema,
    ErrorDetails,
    InitErrorDetails,
    PydanticCustomError,
    core_schema,
)
from pydantic_core.core_schema import TypedDictField, ValidatorFunctionWrapHandler

from conductor._internal.base import SKIPPED
from conductor._internal.problem import conductor_error
from conductor._internal.utils import comma_separated

from .type import Field

#: How a node's function returns its outputs. ``"value"`` is one value, the output ``result``. ``"mapping"`` is a
#: mapping by output name, from a node whose ``compute_outputs`` hook gives its outputs. ``"nothing"`` is ``None``,
#: from a node declared ``-> None``. A class is the dataclass of outputs the node declares, one output per field.
type ReturnShape = Literal["value", "mapping", "nothing"] | type


def outputs_schema(
    outputs: Mapping[str, Field],
    fields: Mapping[str, TypedDictField],
    given_as: ReturnShape | Literal["answer"],
) -> CoreSchema:
    """Builds the schema that reads a value for each of a node's outputs into the outputs, by name.

    Every fault is found in one pass: a name that isn't an output (``extra_forbidden``), an output that must be given
    and isn't (``missing``), ``SKIPPED`` on an output that isn't a branch (``skipped_not_a_branch``), each fault in an
    output's value, and more than one branch given a value (``several_branches_given``, ``_check_branches``). The
    caller that wants only the first takes the first.

    Args:
        outputs: The node's outputs, by name.
        fields: How each output's value is read, by output name: its schema, whether it must be given, and its
            default.
        given_as: How the whole value is given: returned in the node's ``ReturnShape``, or as a person answers
            (``answer_by_output``).

    Returns:
        A schema whose value is every output, by name, ``SKIPPED`` where it isn't given. A branch left out is
        ``SKIPPED`` when another branch is given, even when it has a default. Each error's ``loc`` starts with the
        output's name; a fault of the whole value has the ``loc`` ``()``.
    """

    def split(given: Any) -> Any:
        if given_as == "answer":
            return answer_by_output(outputs, given)
        return _returned_by_output(given_as, outputs, given)

    def all_at_once(given: Any, read_by_name: ValidatorFunctionWrapHandler) -> dict[str, Any]:
        errors: list[InitErrorDetails] = []
        read: dict[str, Any] = {}
        branches: list[str] = []
        try:
            read = read_by_name(given)
        except ValidationError as error:
            errors = [_raised_again(fault) for fault in error.errors()]
        if isinstance(given, Mapping):
            try:
                branches = _check_branches(
                    outputs, [name for name, value in given.items() if value is not SKIPPED]
                )
            except PydanticCustomError as fault:
                errors.append({"type": fault, "loc": (), "input": given})
        if errors:
            raise ValidationError.from_exception_data("outputs", errors)
        filled: dict[str, Any] = {}
        for name, field in outputs.items():
            if field.is_branch and branches and name not in branches:
                filled[name] = SKIPPED
            else:
                filled[name] = read.get(name, SKIPPED)
        return filled

    return core_schema.no_info_before_validator_function(
        split,
        core_schema.no_info_wrap_validator_function(
            all_at_once,
            core_schema.typed_dict_schema(
                {
                    name: {**field, "schema": _taking_skipped(outputs[name], field["schema"])}
                    for name, field in fields.items()
                },
                extra_behavior="forbid",
            ),
        ),
    )


def read_skipped(output: Field) -> Any:
    """Reads ``SKIPPED`` given as the value of one output: a branch not taken.

    It is the one rule for ``SKIPPED`` in a return, an answer and an ``Ask``'s default, and in an answer a host
    queues for later. Only a branch, an output whose default is ``SKIPPED``, may be ``SKIPPED``.

    Returns:
        ``SKIPPED``.

    Raises:
        PydanticCustomError: ``skipped_not_a_branch``, with ``ctx.name`` the output, when it isn't a branch.
    """
    if not output.is_branch:
        raise conductor_error(
            "skipped_not_a_branch",
            "'{name}' is SKIPPED, but it isn't a branch: its default isn't SKIPPED",
            {"name": output.name},
        )
    return SKIPPED


def answer_by_output(outputs: Mapping[str, Field], answer: Any) -> Any:
    """Splits a person's answer, or an ``Ask``'s default, into a value per output name.

    A node of one output is answered with its value, so the value is put under the output's name. A node of several
    is answered with a mapping by output name, or a dataclass or a pydantic model of them, read as the mapping of its
    fields. A node with no outputs is answered with ``None``: the person only says to go on. Any other answer is
    returned as it is, for the schema to refuse as no mapping.
    """
    if len(outputs) == 1:
        (name,) = outputs
        return {name: answer}
    if answer is None and not outputs:
        return {}
    if dataclasses.is_dataclass(answer) and not isinstance(answer, type):
        return {field.name: getattr(answer, field.name) for field in dataclasses.fields(answer)}
    if isinstance(answer, BaseModel):
        return {name: getattr(answer, name) for name in type(answer).model_fields}
    return answer


def _returned_by_output(shape: ReturnShape, outputs: Mapping[str, Field], returned: Any) -> Any:
    """Splits what a node's function returned into a value per output name, by the node's ``ReturnShape``.

    Raises:
        PydanticCustomError: ``none_required``, ``not_a_mapping`` or ``not_the_class`` for a return that isn't
            of the shape.
    """
    match shape:
        case "value":
            return {"result": returned}
        case "nothing":
            if returned is not None:
                raise conductor_error(
                    "none_required",
                    "the node declares no outputs, but returned a {got}",
                    {"got": type(returned).__name__},
                )
            return {}
        case "mapping":
            if not isinstance(returned, Mapping):
                raise conductor_error(
                    "not_a_mapping",
                    "the node must return a mapping with one key per output ({names}), not a {got}",
                    {
                        "names": comma_separated(outputs, empty="nothing"),
                        "got": type(returned).__name__,
                    },
                )
            return returned
        case _:
            if not isinstance(returned, shape):
                raise conductor_error(
                    "not_the_class",
                    "the node must return a {cls}, not a {got}",
                    {"cls": shape.__name__, "got": type(returned).__name__},
                )
            return {name: getattr(returned, name) for name in outputs}


def _taking_skipped(output: Field, schema: CoreSchema) -> CoreSchema:
    """Wraps the schema of one output's value, so that ``SKIPPED`` is read by ``read_skipped`` and not by the type.

    A default stays outside, where pydantic looks for it.
    """
    if schema["type"] == "default":
        return {**schema, "schema": _taking_skipped(output, schema["schema"])}
    return core_schema.no_info_wrap_validator_function(
        lambda value, read: read_skipped(output) if value is SKIPPED else read(value), schema
    )


def _check_branches(outputs: Mapping[str, Field], given: Container[str]) -> list[str]:
    """Refuses a call that gives more than one branch among ``outputs`` a value, and returns the one it gives, if any.

    For ``route`` returning a value on both ``long`` and ``short``, the message is "'long' and 'short' both have a
    value, but they default to SKIPPED, so they are branches of one node". One branch or none keeps the rule.

    Args:
        outputs: The node's outputs, by name.
        given: The names of the outputs that got a value on the call: each one named whose value isn't ``SKIPPED``.

    Returns:
        The branch given a value, in a list, or an empty list when none is.

    Raises:
        PydanticCustomError: ``several_branches_given``, with ``ctx.given`` the branches given a value, in the order
            the node declares them.
    """
    taken = [name for name, field in outputs.items() if field.is_branch and name in given]
    if len(taken) <= 1:
        return taken
    quoted = [f"'{name}'" for name in taken]
    holding = "both" if len(quoted) == 2 else "all"
    raise conductor_error(
        "several_branches_given",
        f"{comma_separated(quoted[:-1])} and {quoted[-1]} {holding} have a value, but they default to SKIPPED, so"
        " they are branches of one node",
        {"given": taken},
    )


def _raised_again(fault: ErrorDetails) -> InitErrorDetails:
    """Returns one of pydantic's errors as it can be raised again, beside an error of conductor's own.

    Pydantic raises again only an error whose type it knows, so each is given as the custom error of its own type,
    message and ``ctx``.
    """
    error = PydanticCustomError(fault["type"], fault["msg"], fault.get("ctx"))
    return {"type": error, "loc": fault["loc"], "input": fault["input"]}
