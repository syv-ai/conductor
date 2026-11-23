"""The schema that reads a value for each output of a node, by output name.

Three things fill a node's outputs: what its function returns, a person's answer when it asks, and the default an
``Ask`` proposes. Each is given as one value per output. Say ``route`` has the outputs ``long`` and ``short``, both
branches, and ``note``. A return of ``Route(long="hi", note="n")`` and an answer of ``{"long": "hi", "note": "n"}``
are both read into ``{"long": "hi", "short": SKIPPED, "note": "n"}``.

``outputs_schema`` holds what is the same for all three. The whole value is split into a value per output name, as
the node returns its outputs or as a person answers them. Every name must be an output. Every output that must be
given is there. ``SKIPPED`` on a branch, an output whose default is ``SKIPPED``, is a branch not chosen, as if it
were left out, and on any other output it is refused (``without_skipped``). Each output not given is ``SKIPPED``.
And at most one branch has a value. The reference has the rule for branches, under "Branches" (``python -m
conductor.about the-node-contract``). How each output's value is read belongs to the caller: a return is checked by ``ReturnValidator``, and an answer and an ``Ask``'s default by the validators of
``ask/``.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Container, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import (
    CoreSchema,
    InitErrorDetails,
    PydanticCustomError,
    core_schema,
)
from pydantic_core.core_schema import TypedDictField, ValidatorFunctionWrapHandler

from conductor._internal.base import SKIPPED
from conductor._internal.problem import conductor_error, raised_again
from conductor._internal.utils import comma_separated

from .type import Port

#: How a node's function returns its outputs. ``"value"`` is one value, the output ``result``. ``"mapping"`` is a
#: mapping by output name, from a node whose ``compute_outputs`` hook gives its outputs. ``"nothing"`` is ``None``,
#: from a node declared ``-> None``. A class is the dataclass of outputs the node declares, one output per field.
type ReturnShape = Literal["value", "mapping", "nothing"] | type


def outputs_schema(
    outputs: Mapping[str, Port],
    schemas: Mapping[str, TypedDictField],
    given_as: ReturnShape | Literal["answer"],
) -> CoreSchema:
    """Builds the schema that reads a value for each of a node's outputs into the outputs, by name.

    Every fault is found in one pass: a name that isn't an output (``extra_forbidden``), an output that must be given
    and isn't (``missing``), ``SKIPPED`` on an output that isn't a branch (``skipped_not_a_branch``), each fault in an
    output's value, and more than one branch given a value (``several_branches_given``, ``_check_branches``). The
    caller that wants only the first takes the first.

    A node that returns one value (``"value"``) or nothing (``"nothing"``) can't give a fault of the whole after the
    split, so its schema only checks the value. A value node's one output, ``result``, is required and isn't a branch.
    A node that returns nothing has no outputs. And a whole ``SKIPPED`` is taken before the split (``ReturnValidator``).

    Args:
        outputs: The node's outputs, by name.
        schemas: How each output's value is read, by output name: its schema, whether it must be given, and its
            default.
        given_as: How the whole value is given: returned in the node's ``ReturnShape``, or as a person answers
            (``answer_by_output``).

    Returns:
        A schema whose value is every output, by name, ``SKIPPED`` where it isn't given. A branch left out is
        ``SKIPPED`` when another branch is given, even when it has a default. Each error's ``loc`` starts with the
        output's name; a fault of the whole value has the ``loc`` ``()``.
    """

    split = (
        functools.partial(answer_by_output, outputs)
        if given_as == "answer"
        else functools.partial(_returned_by_output, given_as, outputs)
    )

    def all_at_once(given: Any, read_by_name: ValidatorFunctionWrapHandler) -> dict[str, Any]:
        errors: list[InitErrorDetails] = []
        refused: list[InitErrorDetails] = []
        read: dict[str, Any] = {}
        branches: list[str] = []
        by_name = isinstance(given, Mapping)
        if by_name:
            given, refused = without_skipped(outputs, given)
        try:
            read = read_by_name(given)
        except ValidationError as error:
            # an output refused for SKIPPED isn't also missing
            skipped = {fault["loc"] for fault in refused}
            errors = [
                raised_again(fault)
                for fault in error.errors()
                if not (fault["type"] == "missing" and fault["loc"] in skipped)
            ]
        if by_name:
            try:
                branches = _check_branches(outputs, given)
            except PydanticCustomError as fault:
                errors.append({"type": fault, "loc": (), "input": given})
        if refused or errors:
            raise ValidationError.from_exception_data(
                "outputs", _in_order(outputs, [*refused, *errors])
            )
        filled: dict[str, Any] = {}
        for name, port in outputs.items():
            if port.is_branch and branches and name not in branches:
                filled[name] = SKIPPED
            else:
                filled[name] = read.get(name, SKIPPED)
        return filled

    by_name = core_schema.typed_dict_schema(dict(schemas), extra_behavior="forbid")
    if given_as in ("value", "nothing"):
        return core_schema.no_info_before_validator_function(split, by_name)
    return core_schema.no_info_before_validator_function(
        split, core_schema.no_info_wrap_validator_function(all_at_once, by_name)
    )


def without_skipped(
    outputs: Mapping[str, Port], given: Mapping[Any, Any]
) -> tuple[Mapping[Any, Any], list[InitErrorDetails]]:
    """Reads ``SKIPPED`` in a value given for each output: the one rule for a return, an answer, an ``Ask``'s default,
    and an answer a host queues for later.

    ``SKIPPED`` on a branch, an output whose default is ``SKIPPED``, is a branch not chosen, so it is left out, as if
    it weren't given. On any other output it is refused. A name that isn't an output is kept, for the caller to
    refuse.

    Args:
        outputs: The node's outputs, by name.
        given: The value given for each output, by name.

    Returns:
        The values given without ``SKIPPED``, and a ``skipped_not_a_branch`` fault at ``(name,)`` for each output
        that isn't a branch and is ``SKIPPED``, with ``ctx.name`` the output.
    """
    # every value a node returns is given here, so a value with no SKIPPED in it is kept as it is
    if not [name for name, value in given.items() if value is SKIPPED]:
        return given, []
    kept: dict[Any, Any] = {}
    refused: list[InitErrorDetails] = []
    for name, value in given.items():
        output = outputs.get(name) if isinstance(name, str) else None
        if value is not SKIPPED or output is None:
            kept[name] = value
        elif not output.is_branch:
            fault = conductor_error(
                "skipped_not_a_branch",
                "'{name}' is SKIPPED, but it isn't a branch: its default isn't SKIPPED",
                {"name": name},
            )
            refused.append({"type": fault, "loc": (name,), "input": value})
    return kept, refused


def answer_by_output(outputs: Mapping[str, Port], answer: Any) -> Any:
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


def _returned_by_output(shape: ReturnShape, outputs: Mapping[str, Port], returned: Any) -> Any:
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


def _in_order(
    outputs: Mapping[str, Port], errors: list[InitErrorDetails]
) -> list[InitErrorDetails]:
    """Puts the faults in the order of the outputs they are at, then those at a name that isn't an output, then those
    of the whole value, keeping the order of the faults at one place."""
    order = {name: index for index, name in enumerate(outputs)}

    def place(error: InitErrorDetails) -> int:
        loc = error["loc"]
        if not loc:
            return len(order) + 1
        return order.get(str(loc[0]), len(order))

    return sorted(errors, key=place)


def _check_branches(outputs: Mapping[str, Port], given: Container[str]) -> list[str]:
    """Refuses a call that gives more than one branch among ``outputs`` a value, and returns the one it gives, if any.

    For ``route`` returning a value on both ``long`` and ``short``, the message is "'long' and 'short' both have a
    value, but they default to SKIPPED, so they are branches of one node". One branch or none keeps the rule.

    Args:
        outputs: The node's outputs, by name.
        given: The names of the outputs that got a value on the call.

    Returns:
        The branch given a value, in a list, or an empty list when none is.

    Raises:
        PydanticCustomError: ``several_branches_given``, with ``ctx.given`` the branches given a value, in the order
            the node declares them.
    """
    taken = [name for name, port in outputs.items() if port.is_branch and name in given]
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
