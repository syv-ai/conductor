"""The declaration door: a default a node author declares, read by its input's or output's type when the node is
declared, and read again by the type an edge binds its type variable to when a graph is compiled.

Example: ``shout(line: str, mark: str = "!")`` declares the default ``"!"``, and ``declared_field("mark", str, "!")``
gives the input ``mark`` with that default. ``times: Annotated[int, Param(ge=0)] = -1`` is refused when the module
that declares the node is imported: ``NodeDeclarationError`` with the code ``default_invalid``, naming the input.
``first(one: T, other: T = 0)`` wired from an output of ``str`` binds ``T`` to ``str``, and ``bound_field`` refuses
``0`` for it: ``invalid_value`` (variant ``bound_default``), a problem of the graph.

A default is read as a Python value is read (``1`` for a ``float`` is ``1.0``). The defaults a graph record gives
are read by the constants door instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.field import Field
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    compile_problems,
    declaration_problems,
)
from conductor._internal.types import Type

from .rules import validator_of


def declared_field(name: str, hint: Any, default: Any) -> Field:
    """Returns the input or output ``name`` that a node declares with the type hint ``hint`` and ``default``, the
    default read by the field's type.

    Args:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as written: ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch not taken.

    Raises:
        NodeDeclarationError: Naming the field: what ``Field`` refuses of the hint, or a default the type refuses or
            that has no JSON form (``default_invalid``).
    """
    whole = Field(name, hint).whole
    try:
        return Field(name, hint, _read_default(whole, default))
    except ValidationError as error:
        refused = declaration_problems.default_none_not_taken() if default is None else declaration_problems.default_invalid(default=repr(default), type=whole.id, reason=whole.refusal(error).reason)
        raise NodeDeclarationError(refused, name) from None


def bound_field(node_id: str, field: Field, bound: Mapping[TypeVar, Any]) -> Field:
    """Returns ``field`` with its type variables replaced by what the edges into the node ``node_id`` bind them to
    (``bound``), and its default read by the type it then has; ``field`` itself when its type has no variable.

    For ``same(value: T = "-")`` wired from an output of ``str``, ``T`` is bound to ``str`` and the default stays
    ``"-"``. A variable no edge binds is read as its bound, else ``Any``.

    Raises:
        InvalidCallError: The bound type refuses the default the node declares, ``None`` included: one ``invalid_value``
            problem (variant ``bound_default``) at ``(node_id, field)``.
    """
    if not field.whole.variables:
        return field
    hint = Type(field.hint).bind(bound).hint
    whole = Field(field.name, hint).whole
    try:
        return Field(field.name, hint, _read_default(whole, field.default))
    except ValidationError as error:
        refused = whole.refusal(error)
        raise InvalidCallError([compile_problems.invalid_value_bound_default(node_id, field.name, type=whole.id, reason=refused.reason, errors=refused.errors)]) from None


def _read_default(whole: Type, default: Any) -> Any:
    """Returns ``default`` read by ``whole``, the type of a field's whole value. ``REQUIRED`` and ``SKIPPED`` are
    kept as they are; any other default, ``None`` included, must be a value of the type with a JSON form, since an
    editor shows it and a saved graph stores it.

    Raises:
        ValidationError: The type refuses the default (``None`` for a type that does not take it), or the value it
            reads has no JSON form (``no_json_form``).
    """
    if default is REQUIRED or default is SKIPPED:
        return default
    read = validator_of(whole).validate_python(default)
    try:
        whole.to_json(read)
    except (PydanticSerializationError, UnicodeDecodeError):
        no_json_form = PydanticCustomError("no_json_form", "it has no JSON form")
        raise ValidationError.from_exception_data(whole.id, [{"type": no_json_form, "loc": (), "input": default}]) from None
    return read
