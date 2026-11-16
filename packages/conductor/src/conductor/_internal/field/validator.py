"""Checking the default a node author writes for an input or an output.

A default is checked against the field's type, so a node with a bad default fails early. Say a node declares
``times: Annotated[int, Param(ge=0)] = -1``. The default ``-1`` is below the limit, so importing the module that
declares the node raises ``NodeDeclarationError`` with the code ``default_invalid``.

A type variable makes this happen twice. In ``first(one: T, other: T = 0)``, the default ``0`` is fine for ``T`` on
its own. Once a graph wires ``one`` from an output of type ``str``, ``T`` is ``str``, and ``0`` no longer fits. So
the default is checked again when the graph is compiled, and the graph gets an ``invalid_value`` problem.

A default is checked as a Python value, so ``1`` for a ``float`` becomes ``1.0``. The defaults written in a graph
record are JSON, and ``ConstantValidator`` in ``graph/`` checks those.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    compile_problems,
    declaration_problems,
)
from conductor._internal.types import Type
from conductor._internal.validation import Validator, schema_validator_of

from .type import Field


def _declared_field(name: str, hint: Any, default: Any, *, is_output: bool) -> Field:
    """Builds the field a node declares, with its default checked against its type.

    Args:
        name: The parameter's name, or the output's.
        hint: The type hint, as the author wrote it.
        default: The default, as the author wrote it. ``REQUIRED`` means there is none. ``SKIPPED`` means an
            output is left empty on a branch not taken.
        is_output: Whether the field is an output.

    Raises:
        NodeDeclarationError: The hint is one ``Field`` refuses, or the type refuses the default, or the default
            can't be written as JSON (``default_invalid``). It names the field.
    """
    whole = Field(name, hint, is_output=is_output).whole
    try:
        return Field(name, hint, _read_default(whole, default), is_output=is_output)
    except ValidationError as error:
        refused = (
            declaration_problems.default_none_not_taken()
            if default is None
            else declaration_problems.default_invalid(
                default=repr(default), type=whole.id, reason=whole.refusal(error).reason
            )
        )
        raise NodeDeclarationError(refused, name) from None


class DeclaredInputValidator(Validator[Any, Field]):
    """A validator for one input a node declares, called with the input's default.

    It checks the default against the input's type and returns the input as a ``Field``. It also refuses what only
    an output may have. An input is never a branch, so it can't be in an exclusive group, and its default can't be
    ``SKIPPED``.

    ``@node`` uses it for each parameter of the function. ``HookValidator`` uses it for each input a
    ``compute_added_inputs`` hook adds. So ``text: Annotated[str, Param(exclusive_group="x")]`` is refused in both.

    Raises (when called):
        NodeDeclarationError: Naming the input. The default is wrong for the type (``default_invalid``), the input
            is in an exclusive group (``exclusive_group_on_input``), or its default is ``SKIPPED``
            (``default_invalid``, variant ``skipped_on_input``).
    """

    def __init__(self, name: str, hint: Any) -> None:
        """Builds the validator for one input.

        Args:
            name: The input's name.
            hint: The input's type hint.
        """
        self._name = name
        self._hint = hint

    def __call__(self, default: Any) -> Field:
        """Checks ``default`` and returns the input. ``REQUIRED`` means the input has no default.

        Raises:
            NodeDeclarationError: See the class.
        """
        if default is SKIPPED:
            raise NodeDeclarationError(declaration_problems.default_skipped_on_input(), self._name)
        field = _declared_field(self._name, self._hint, default, is_output=False)
        if field.param.exclusive_group is not None:
            raise NodeDeclarationError(declaration_problems.exclusive_group_on_input(), self._name)
        return field


class DeclaredOutputValidator(Validator[Any, Field]):
    """A validator for one output a node declares, called with the output's default.

    It checks the default against the output's type and returns the output as a ``Field``. It also refuses what an
    output may not have. ``constant_only`` is for inputs only. And an output in an exclusive group can't have a real
    default: when it isn't returned or answered, it is a branch not chosen and stays empty. So its default is
    ``SKIPPED`` or none.

    ``@node`` uses it for each field of a returned dataclass. ``HookValidator`` uses it for each output a
    ``compute_outputs`` hook gives. So ``keep: Annotated[str, Param(exclusive_group="way")] = "x"`` is refused in
    both.

    Raises (when called):
        NodeDeclarationError: Naming the output. The default is wrong for the type (``default_invalid``), the
            output is ``constant_only`` (``constant_only_on_output``), or an output in an exclusive group has a
            default other than ``SKIPPED`` (``default_invalid``, variant ``on_exclusive_group``).
    """

    def __init__(self, name: str, hint: Any) -> None:
        """Builds the validator for one output.

        Args:
            name: The output's name.
            hint: The output's type hint.
        """
        self._name = name
        self._hint = hint

    def __call__(self, default: Any) -> Field:
        """Checks ``default`` and returns the output.

        ``REQUIRED`` means the output has no default. ``SKIPPED`` means it stays empty on a branch not taken.

        Raises:
            NodeDeclarationError: See the class.
        """
        field = _declared_field(self._name, self._hint, default, is_output=True)
        if field.param.constant_only:
            raise NodeDeclarationError(declaration_problems.constant_only_on_output(), self._name)
        if (
            field.param.exclusive_group is not None
            and default is not REQUIRED
            and default is not SKIPPED
        ):
            raise NodeDeclarationError(
                declaration_problems.default_on_exclusive_group(), self._name
            )
        return field


class BoundDefaultValidator(Validator[Field, Field]):
    """A validator that checks a field's default again, once a graph has fixed its type variables.

    A node can use a type variable, as in ``same(value: T = "-")``. When a graph is compiled, the edges into the
    node decide what ``T`` is. Wired from an output of type ``str``, ``T`` is ``str``, and the default ``"-"`` still
    fits. Wired from an ``int``, it doesn't. A variable no edge decides becomes its bound, or ``Any`` if it has none.

    The compiler builds one for each node and calls it with each of the node's fields.

    Raises (when called):
        InvalidCallError: The type refuses the default, ``None`` included. It carries one ``invalid_value`` problem
            (variant ``bound_default``) on the node and the field.
    """

    def __init__(self, node_id: str, bound: Mapping[TypeVar, Any]) -> None:
        """Builds the validator for one node.

        Args:
            node_id: The node's id. The problem names it.
            bound: What the edges into the node fix each type variable to.
        """
        self._node_id = node_id
        self._bound = bound

    def __call__(self, field: Field) -> Field:
        """Checks the default of ``field`` by its fixed type, and returns the field with that type.

        A field whose type has no type variable is returned as it is.

        Raises:
            InvalidCallError: See the class.
        """
        if not field.whole.variables:
            return field
        hint = Type(field.hint).bind(self._bound).hint
        whole = Field(field.name, hint, is_output=field.is_output).whole
        try:
            return Field(
                field.name, hint, _read_default(whole, field.default), is_output=field.is_output
            )
        except ValidationError as error:
            refused = whole.refusal(error)
            raise InvalidCallError(
                [
                    compile_problems.invalid_value_bound_default(
                        self._node_id,
                        field.name,
                        type=whole.id,
                        reason=refused.reason,
                        errors=refused.errors,
                    )
                ]
            ) from None


def _read_default(whole: Type, default: Any) -> Any:
    """Checks a default against a field's type, and returns it converted to that type.

    ``REQUIRED`` and ``SKIPPED`` aren't values, so they are returned as they are. Any other default, ``None``
    included, must be a value of the type. It must also be one that can be written as JSON, since an editor shows
    it and a saved graph stores it.

    Args:
        whole: The type of the field's whole value. For a ``Series[int]``, that is the ``Series``, not ``int``.
        default: The default to check.

    Raises:
        ValidationError: The type refuses the default, or the converted value can't be written as JSON
            (``no_json_form``).
    """
    if default is REQUIRED or default is SKIPPED:
        return default
    read = schema_validator_of(whole).validate_python(default)
    try:
        whole.to_json(read)
    except (PydanticSerializationError, UnicodeDecodeError):
        no_json_form = PydanticCustomError("no_json_form", "it can't be written as JSON")
        raise ValidationError.from_exception_data(
            whole.id, [{"type": no_json_form, "loc": (), "input": default}]
        ) from None
    return read
