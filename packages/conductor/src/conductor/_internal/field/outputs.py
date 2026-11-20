"""The schema that reads a value for each output of a node, by output name.

Three things fill a node's outputs: what its function returns, a person's answer when it asks, and the default an
``Ask`` proposes. Each is given as one value per output. Say ``route`` has the outputs ``long`` and ``short``, both
branches, and ``note``. A return of ``Route(long="hi", note="n")`` and an answer of ``{"long": "hi", "note": "n"}``
are both read into ``{"long": "hi", "short": SKIPPED, "note": "n"}``.

``outputs_schema`` holds what is the same for all three. Every name must be an output. Every output that must be
given is there. Each output not given is ``SKIPPED``. And at most one branch has a value: a branch is an output whose
default is ``SKIPPED``, and the reference has its rule, under "Branches" (``python -m conductor.about
the-node-contract``). How the whole value is split into names, and how each output's value is read, belong to the
caller: a return is checked by ``ReturnValidator``, and an answer and an ``Ask``'s default by the validators of
``ask/``.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import (
    CoreSchema,
    ErrorDetails,
    InitErrorDetails,
    PydanticCustomError,
    core_schema,
)
from pydantic_core.core_schema import TypedDictField

from conductor._internal.base import SKIPPED
from conductor._internal.problem import conductor_error
from conductor._internal.utils import comma_separated

from .type import Field


def outputs_schema(
    outputs: Mapping[str, Field],
    fields: Mapping[str, TypedDictField],
    by_name: Callable[[Any], Any],
) -> CoreSchema:
    """Builds the schema that reads a value for each of a node's outputs into the outputs, by name.

    Every fault is found in one pass: a name that isn't an output (``extra_forbidden``), an output that must be given
    and isn't (``missing``), each fault in an output's value, and more than one branch given a value
    (``several_branches_given``, ``_check_branches``). The caller that wants only the first takes the first.

    Args:
        outputs: The node's outputs, by name.
        fields: How each output's value is read, by output name: its schema, whether it must be given, and its
            default.
        by_name: Splits the whole value into a mapping by output name, leaving out each output that isn't given. It
            raises ``PydanticCustomError`` for a value it can't split, and returns any other value it doesn't know
            as it is, for the schema to refuse as no mapping.

    Returns:
        A schema whose value is every output, by name, ``SKIPPED`` where it isn't given. A branch left out is
        ``SKIPPED`` when another branch is given, even when it has a default. Each error's ``loc`` starts with the
        output's name; a fault of the whole value has the ``loc`` ``()``.
    """

    def all_at_once(given: Any, read_by_name: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        errors: list[InitErrorDetails] = []
        read: dict[str, Any] = {}
        branches: list[str] = []
        try:
            read = read_by_name(given)
        except ValidationError as error:
            errors = [_raised_again(fault) for fault in error.errors()]
        if isinstance(given, Mapping):
            try:
                branches = _check_branches(outputs, given)
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
        by_name,
        core_schema.no_info_wrap_validator_function(
            all_at_once, core_schema.typed_dict_schema(dict(fields), extra_behavior="forbid")
        ),
    )


def _check_branches(outputs: Mapping[str, Field], given: Container[str]) -> list[str]:
    """Refuses a call that gives more than one branch among ``outputs`` a value, and returns the one it gives, if any.

    For ``route`` returning a value on both ``long`` and ``short``, the message is "'long' and 'short' both have a
    value, but they default to SKIPPED, so they are branches of one node". One branch or none keeps the rule.

    Args:
        outputs: The node's outputs, by name.
        given: The names of the outputs that got a value on the call. A return gives an output a value when the
            value isn't ``SKIPPED``. An answer and an ``Ask``'s default give a value to each output they name.

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
