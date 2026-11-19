"""Checking the default a node author writes for an input or an output.

A default is checked against the field's type, so a node with a bad default fails early. Say a node declares
``times: Annotated[int, Param(ge=0)] = -1``. The default ``-1`` is below the limit, so importing the module that
declares the node raises ``NodeDeclarationError`` with the code ``default_invalid``.

A type variable makes this happen twice. In ``first(one: T, other: T = 0)``, the default ``0`` is fine for ``T`` on
its own. Once a graph wires ``one`` from an output of type ``str``, ``T`` is ``str``, and ``0`` no longer fits. So
the default is checked again when the graph is compiled, and the graph gets an ``invalid_value`` problem.

A default is read strictly, as its JSON text would be (``read_as_json``), the way a caller's graph input and a
graph record's constant are read. A default is written into a graph record and read back from it, so it must read
the same from either. So ``"3"`` and ``5.0`` are refused for an ``int``, ``1`` for a ``bool`` and a ``datetime``
for a ``date``, and ``1`` for a ``float`` becomes ``1.0``.

A field's type may hold no set, at any depth: a set has no order, so it has no JSON form a default could be written
in and read back by. ``list`` or ``tuple`` is declared instead. Nor may it hold three kinds of pydantic's types: a
secret, whose JSON form is a mask, an ``ImportString``, which runs the code its value names, and a type that decodes
its value, such as ``Json[int]`` or ``Base64Bytes``, which a decoded value fails the next time it is checked.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Mapping, Set
from typing import Annotated, Any, TypeVar, get_args, get_origin

from annotated_types import Not, Predicate
from pydantic import BaseModel, ImportString, Json, Secret, SecretBytes, SecretStr, ValidationError
from pydantic.dataclasses import is_pydantic_dataclass
from pydantic.types import EncodedBytes, EncodedStr

from conductor._internal.base import REQUIRED, SKIPPED
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    compile_problems,
    declaration_problems,
)
from conductor._internal.table import Table, check_table_row
from conductor._internal.types import Type, as_written, hint_parts
from conductor._internal.validation import Validator, read_as_json

from .type import Field

#: annotated-types' named predicates that ask for a number that isn't finite: ``IsNan``, ``IsInfinite`` and
#: ``IsNotFinite``, which may also be written ``Not(math.isfinite)``. A number given to a graph from outside is finite, so no input passes them; a run's own values
#: may be ``inf`` or ``nan``, so an output may declare them.
_NOT_FINITE = (
    Predicate(math.isnan),
    Predicate(math.isinf),
    Predicate(Not(math.isfinite)),
    Not(math.isfinite),
)


def _declared_field(name: str, hint: Any, default: Any, *, is_output: bool) -> Field:
    """Builds the field a node declares, with its default checked against its type.

    Args:
        name: The parameter's name, or the output's.
        hint: The type hint, as the author wrote it.
        default: The default, as the author wrote it. ``REQUIRED`` means there is none. ``SKIPPED`` makes an
            output a branch.
        is_output: Whether the field is an output.

    Raises:
        NodeDeclarationError: The hint is one ``Field`` refuses, it holds a set (``type_not_supported_set``), a
            secret (``type_not_supported_secret``), an ``ImportString`` (``type_not_supported_import_string``) or a
            type that decodes its value (``type_not_supported_decoding``), a host's pydantic model holds a ``bytes``
            field it reads as text (``type_not_supported_bytes_as_text``), a
            ``Table`` in it has a row type that names no columns (``table_row_not_a_typed_dict``), an input asks for
            a number that isn't finite (``predicate_refuses_every_number``), or the type
            refuses the default read as its JSON (``default_invalid``). It names the field.
    """
    try:
        for part in hint_parts(hint):
            if get_origin(part) is Table:
                check_table_row(*get_args(part))
            elif _is_a_set(part):
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported_set(type=as_written(part))
                )
            elif _marked(part, (Secret, SecretStr, SecretBytes)):
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported_secret(type=as_written(part))
                )
            elif _marked(part, (ImportString,)):
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported_import_string(type=as_written(part))
                )
            elif _marked(part, (Json, EncodedBytes, EncodedStr)):
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported_decoding(type=as_written(part))
                )
            elif not is_output and any(part == never for never in _NOT_FINITE):
                raise NodeDeclarationError(
                    declaration_problems.predicate_refuses_every_number(predicate=repr(part))
                )
        if (as_text := _bytes_read_as_text(hint)) is not None:
            cls, field = as_text
            raise NodeDeclarationError(
                declaration_problems.type_not_supported_bytes_as_text(cls=cls, field=field)
            )
    except NodeDeclarationError as error:
        raise error.within(name) from None
    whole = Field(name, hint, is_output=is_output).whole
    try:
        return Field(name, hint, _read_default(whole, default), is_output=is_output)
    except ValidationError as error:
        refused = (
            declaration_problems.default_none_not_taken()
            if default is None
            else declaration_problems.default_invalid(
                default=repr(default), type=whole.id, **whole.refusal(error)
            )
        )
        raise NodeDeclarationError(refused, name) from None


class DeclaredInputValidator(Validator[Any, Field]):
    """A validator for one input a node declares, called with the input's default.

    It checks the default against the input's type and returns the input as a ``Field``. It also refuses what only
    an output may have. An input is never a branch, so its default can't be ``SKIPPED``.

    ``@node`` uses it for each parameter of the function. ``HookValidator`` uses it for each input a
    ``compute_added_inputs`` hook adds. So ``text: str = SKIPPED`` is refused in both.

    Raises (when called):
        NodeDeclarationError: Naming the input. The default is wrong for the type (``default_invalid``), or it is
            ``SKIPPED`` (``default_skipped_on_input``).
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
        return _declared_field(self._name, self._hint, default, is_output=False)


class DeclaredOutputValidator(Validator[Any, Field]):
    """A validator for one output a node declares, called with the output's default.

    It checks the default against the output's type and returns the output as a ``Field``. It also refuses what an
    output may not have: ``constant_only`` is for inputs only.

    ``@node`` uses it for each field of a returned dataclass. ``HookValidator`` uses it for each output a
    ``compute_outputs`` hook gives. So ``keep: Annotated[str, Param(constant_only=True)]`` is refused in both.

    Raises (when called):
        NodeDeclarationError: Naming the output. The default is wrong for the type (``default_invalid``), or the
            output is ``constant_only`` (``constant_only_on_output``).
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

        ``REQUIRED`` means the output has no default. ``SKIPPED`` makes it a branch.

        Raises:
            NodeDeclarationError: See the class.
        """
        field = _declared_field(self._name, self._hint, default, is_output=True)
        if field.param.constant_only:
            raise NodeDeclarationError(declaration_problems.constant_only_on_output(), self._name)
        return field


class BoundDefaultValidator(Validator[Field, Field]):
    """A validator that checks a field's default again, once a graph has fixed its type variables.

    A node can use a type variable, as in ``same(value: T = "-")``. When a graph is compiled, the edges into the
    node decide what ``T`` is. Wired from an output of type ``str``, ``T`` is ``str``, and the default ``"-"`` still
    fits. Wired from an ``int``, it doesn't. A variable no edge decides becomes its bound, or ``Any`` if it has none.

    The compiler builds one for each node and calls it with each of the node's fields.

    Raises (when called):
        InvalidCallError: The type refuses the default, ``None`` included. It carries one ``invalid_value_bound_default``
            problem on the node and an input, or ``output_default_bound_invalid`` on an output.
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
            kind = (
                compile_problems.output_default_bound_invalid
                if field.is_output
                else compile_problems.invalid_value_bound_default
            )
            raise InvalidCallError(
                [kind(self._node_id, field.name, type=whole.id, **whole.refusal(error))]
            ) from None


def _is_a_set(hint: Any) -> bool:
    """Whether ``hint`` is a set type: ``set``, ``frozenset``, ``collections.abc.Set`` or a subclass of one, bare or
    with its item type."""
    origin = get_origin(hint) or hint
    return isinstance(origin, type) and issubclass(origin, Set)


def _marked(hint: Any, classes: tuple[type, ...]) -> bool:
    """Whether ``hint`` is one of ``classes``, bare or with its arguments, or an ``Annotated`` with one of them in its
    metadata.

    Pydantic writes some of its types either way: ``SecretStr`` and ``Secret[int]`` are classes, while
    ``Json[int]`` is ``Annotated[int, Json()]`` and ``Base64Bytes`` is ``Annotated[bytes, EncodedBytes(...)]``.
    """
    origin = get_origin(hint) or hint
    if isinstance(origin, type) and issubclass(origin, classes):
        return True
    return get_origin(hint) is Annotated and any(
        isinstance(metadata, classes) for metadata in get_args(hint)[1:]
    )


def _bytes_read_as_text(
    hint: Any, reading_as_text: tuple[str, str] | None = None, walked: frozenset[int] = frozenset()
) -> tuple[str, str] | None:
    """Finds a ``bytes`` field that a class in ``hint`` with a pydantic config of its own reads as text.

    Conductor writes and reads bytes as base64 text (``JSON_CONFIG``). A pydantic model, a pydantic dataclass, and
    a ``TypedDict`` or a plain dataclass given a config with pydantic's ``with_config``, read and write their fields
    by their own config instead, and a config that doesn't set the bytes takes them as text. A ``TypedDict`` or a
    plain dataclass with no config of its own takes the config it sits in. A class inside another with a config of
    its own takes its own.

    Say ``class Attachment(BaseModel)`` has the field ``data: bytes`` and the default config. Then
    ``list[Attachment]`` gives ``("Attachment", "data")``. With ``ser_json_bytes="base64"`` and
    ``val_json_bytes="base64"`` in its config, it gives ``None``.

    Args:
        hint: The hint to look in.
        reading_as_text: The class and field that ``hint`` is inside, when that class reads bytes as text.
        walked: The classes already looked in (by ``id``), so a class that holds itself ends.

    Returns:
        The class's name and its field, or ``None`` when no bytes in ``hint`` are read as text.
    """
    for part in hint_parts(hint):
        if isinstance(part, type) and issubclass(part, bytes) and reading_as_text is not None:
            return reading_as_text
        cls = get_origin(part) or part
        if not isinstance(cls, type) or id(part) in walked:
            continue
        # a model, and any class pydantic's with_config gave a config, reads its fields by its own config
        config = (
            cls.model_config
            if issubclass(cls, BaseModel)
            else getattr(cls, "__pydantic_config__", None)
        )
        if issubclass(cls, BaseModel) or is_pydantic_dataclass(cls):
            # a generic model's fields have its arguments filled in
            fields = cls.model_fields if issubclass(cls, BaseModel) else cls.__pydantic_fields__
            hints = {name: info.annotation for name, info in fields.items()}
        elif dataclasses.is_dataclass(cls) or typing.is_typeddict(cls) and config is not None:
            hints = typing.get_type_hints(cls, include_extras=True)
        else:
            # hint_parts walks a TypedDict with no config of its own, under the config it sits in
            continue
        walked |= {id(part)}
        as_text = config is not None and (
            config.get("ser_json_bytes") != "base64" or config.get("val_json_bytes") != "base64"
        )
        for name, field_hint in hints.items():
            inside = None
            if config is None:
                # a plain dataclass reads its fields by the config it sits in
                inside = reading_as_text
            elif as_text:
                inside = (cls.__name__, name)
            if (found := _bytes_read_as_text(field_hint, inside, walked)) is not None:
                return found
    return None


def _read_default(whole: Type, default: Any) -> Any:
    """Reads a default by a field's type strictly, as its JSON text would be read, and returns what it reads.

    ``REQUIRED`` and ``SKIPPED`` aren't values, so they are returned as they are. Any other default, ``None``
    included, is read as a caller's graph input is (``read_as_json``): an editor shows it and a saved graph stores
    it as JSON, and the graph reads it back from there.

    Args:
        whole: The type of the field's whole value. For a ``Series[int]``, that is the ``Series``, not ``int``.
        default: The default to read.

    Raises:
        ValidationError: The type refuses the default, the default can't be written as JSON (``no_json_form``), or
            a number in it isn't finite (``finite_number``).
    """
    if default is REQUIRED or default is SKIPPED:
        return default
    return read_as_json(whole, whole.argument_validator, default)
