"""The declaration validator: a default a node author declares, read by its input's or output's type when the node is
declared, and read again by the type an edge binds its type variable to when a graph is compiled.

Example: ``shout(line: str, mark: str = "!")`` declares the default ``"!"``, and
``DeclaredInputValidator("mark", str)("!")`` gives the input ``mark`` with that default.
``times: Annotated[int, Param(ge=0)] = -1`` is refused when the module that declares the node is imported:
``NodeDeclarationError`` with the code ``default_invalid``, naming the input.
``first(one: T, other: T = 0)`` wired from an output of ``str`` binds ``T`` to ``str``, and ``BoundDefaultValidator``
refuses ``0`` for it: ``invalid_value`` (variant ``bound_default``), a problem of the graph.

A default is read as a Python value is read (``1`` for a ``float`` is ``1.0``). The defaults a graph record gives
are read by ``RecordValueValidator`` instead.
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
    """Returns the input or output ``name`` that a node declares with the type hint ``hint`` and ``default``, the
    default read by the field's type.

    Args:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as written: ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch not taken.
        is_output: Whether it is an output.

    Raises:
        NodeDeclarationError: Naming the field: what ``Field`` refuses of the hint, or a default the type refuses or
            that has no JSON form (``default_invalid``).
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
    """Reads the default of an input a node declares into the input, as ``_declared_field`` does, refusing what only
    an output can have: an input is never a branch.

    ``@node`` reads each parameter of the function with one, and the hook validator each input a
    ``compute_added_inputs`` hook returns, so ``text: Annotated[str, Param(exclusive_group="x")]`` is refused in both
    places.

    Raises (when called):
        NodeDeclarationError: Naming the input: what ``_declared_field`` refuses, an ``exclusive_group``
            (``exclusive_group_on_input``), or the default ``SKIPPED`` (``default_invalid``, variant
            ``skipped_on_input``).
    """

    def __init__(self, name: str, hint: Any) -> None:
        """Builds the validator for the input ``name``, declared with the type hint ``hint``."""
        self._name = name
        self._hint = hint

    def __call__(self, default: Any) -> Field:
        """Returns the input with ``default`` read by its type: ``REQUIRED`` for none.

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
    """Reads the default of an output a node declares into the output, as ``_declared_field`` does, refusing a default
    an output of an exclusive group cannot have: left out of an answer, or not returned, it is a branch not chosen, so
    its default is ``SKIPPED`` or none.

    ``@node`` reads each field of a returned dataclass with one, and the hook validator each output a
    ``compute_outputs`` hook returns, so ``keep: Annotated[str, Param(exclusive_group="way")] = "x"`` is refused in
    both places.

    Raises (when called):
        NodeDeclarationError: Naming the output: what ``_declared_field`` refuses, ``constant_only``, which only an
            input takes (``constant_only_on_output``), or a default other than ``SKIPPED`` on an output of an
            exclusive group (``default_invalid``, variant ``on_exclusive_group``).
    """

    def __init__(self, name: str, hint: Any) -> None:
        """Builds the validator for the output ``name``, declared with the type hint ``hint``."""
        self._name = name
        self._hint = hint

    def __call__(self, default: Any) -> Field:
        """Returns the output with ``default`` read by its type: ``REQUIRED`` for none, ``SKIPPED`` for a branch not
        taken.

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
    """Reads the declared default of each input or output of one node again, by the type it has once its type
    variables are replaced by what the edges into the node bind them to.

    For ``same(value: T = "-")`` wired from an output of ``str``, ``T`` is bound to ``str`` and the default stays
    ``"-"``. A variable no edge binds is read as its bound, else ``Any``.

    Raises (when called):
        InvalidCallError: The bound type refuses the default the node declares, ``None`` included: one
            ``invalid_value`` problem (variant ``bound_default``) at ``(node_id, field)``.
    """

    def __init__(self, node_id: str, bound: Mapping[TypeVar, Any]) -> None:
        """Builds the validator for the node ``node_id``, whose edges bind its type variables as ``bound`` says."""
        self._node_id = node_id
        self._bound = bound

    def __call__(self, field: Field) -> Field:
        """Returns ``field`` with its type variables bound and its default read by the type it then has; ``field``
        itself when its type has no variable.

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
    """Returns ``default`` read by ``whole``, the type of a field's whole value. ``REQUIRED`` and ``SKIPPED`` are
    kept as they are; any other default, ``None`` included, must be a value of the type with a JSON form, since an
    editor shows it and a saved graph stores it.

    Raises:
        ValidationError: The type refuses the default (``None`` for a type that does not take it), or the value it
            reads has no JSON form (``no_json_form``).
    """
    if default is REQUIRED or default is SKIPPED:
        return default
    read = schema_validator_of(whole).validate_python(default)
    try:
        whole.to_json(read)
    except (PydanticSerializationError, UnicodeDecodeError):
        no_json_form = PydanticCustomError("no_json_form", "it has no JSON form")
        raise ValidationError.from_exception_data(
            whole.id, [{"type": no_json_form, "loc": (), "input": default}]
        ) from None
    return read
