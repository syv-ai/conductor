"""``Type``: one type hint as conductor reads it. ``Type(int)`` returns the same object every time.

It gives the hint's id and title, validates a value, writes it as JSON, and says which other types fit it.
Validation is pydantic's. ``Series`` and ``Table`` each have a subclass of ``Type``, declared in their own
packages.
"""

from __future__ import annotations

import datetime
import functools
import types
import typing
import weakref
from collections.abc import Callable, Mapping
from typing import Annotated, Any, ClassVar, Literal, TypeVar, Union, get_args, get_origin

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    SchemaError,
    core_schema,
    from_json,
    to_json,
)

from conductor._internal.problem import NodeDeclarationError, value_free_errors
from conductor._internal.value_type.choice import Choice
from conductor._internal.value_type.fit import fit
from conductor._internal.value_type.hint import (
    READ_AS,
    aliased,
    is_alias,
    is_record,
    substituted,
    union_members,
    union_of,
)
from conductor._internal.value_type.host_type import NAMES, OWN_IDS, declaration_of
from conductor._internal.value_type.not_a_bool import NUMBERS_REFUSE_BOOLS
from conductor._internal.value_type.param import Param, split_hint
from conductor._internal.value_type.refused import FieldValueError

#: The classes whose values are plain in JSON: no keys to order, and read back by their type as they were.
_SCALARS = frozenset({str, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta})

#: The generic classes whose arguments ``Type`` reads as types of their own, so its rules reach their items.
_CONTAINERS = (list, dict, tuple, set, frozenset)

#: Maps ``Series`` and ``Table`` to their ``Type`` subclass (``SeriesType``, ``TableType``).
_KINDS: dict[Any, type[Type]] = {}


def _holds_a_model(schema: Any) -> bool:
    """Says whether a core schema holds a pydantic model or a dataclass anywhere in it."""
    if isinstance(schema, dict):
        return schema.get("type") in ("model", "dataclass") or any(map(_holds_a_model, schema.values()))
    return isinstance(schema, list) and any(map(_holds_a_model, schema))


def _written(hint: Any) -> str:
    """One bare hint as ``Type.reads_as`` writes it; an ``Annotated`` inside it is written as the type it is."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Annotated:
        return Type(hint).reads_as
    if origin is Literal:
        return f"Literal{[repr(value) for value in args]}"
    if is_alias(hint):
        return f"{hint.__module__}.{hint.__name__}"
    if args:
        name = "Union" if origin in (Union, types.UnionType) else _written(origin)
        return f"{name}[{', '.join(map(_written, args))}]"
    return f"{hint.__module__}.{hint.__qualname__}" if isinstance(hint, type) else repr(hint)


class Type:
    """One annotation as conductor reads it. ``Type(hint)`` gives the one object of a hint, so what a hint costs
    to read is paid once, and nothing is built until it is asked for.

    The one object is kept for as long as something holds it (a field, a loaded graph), so a hint made while a
    graph loads is let go with the graph.

    ``Series`` and ``Table`` each have a subclass of ``Type``, registered with the class it is for::

        class SeriesType(Type, of=Series): ...

    ``Type(Series[int])`` and ``Type(Series[int] | None)`` then return a ``SeriesType``. Code that has a ``Type``
    calls the same attributes and methods whichever class it is; a subclass overrides the ones whose answer
    differs for its values, so no caller tests for a ``Series`` or a ``Table``.

    Attributes:
        hint: The annotation, with its ``Annotated`` metadata and its type variables as written.
        keys_in_order: Whether the JSON of a value always has its object keys in the same order, so its JSON can
            be hashed without sorting the keys. True for a table.
        may_write_once: Whether ``written_once`` can return an object for a stored value of the type.
        typed_in_advice: Text added to the error for a value of the type that was typed into a graph and has no
            JSON form, saying what to give instead; empty for most types.
    """

    hint: Any
    keys_in_order: ClassVar[bool] = False
    may_write_once: bool = False
    typed_in_advice: ClassVar[str] = ""
    _made: ClassVar[weakref.WeakValueDictionary[Any, Type]] = weakref.WeakValueDictionary()

    def __init_subclass__(cls, *, of: Any, value_class: type | None = None) -> None:
        """Registers the subclass as the ``Type`` of hints written with the generic class ``of``.

        Args:
            of: The generic class a node author writes in a hint (``Series``, ``Table``).
            value_class: The class of the values at run time, when an author may write that class as the hint
                too (``polars.DataFrame`` for ``Table``). A value of this class found where no type is declared
                (inside an ``Any``) is written, hashed and compared by the subclass's ``untyped_`` methods.
        """
        _KINDS[of] = cls
        OWN_IDS.add(of.__qualname__)  # the generic class is named as a host imports it, which is its id
        if value_class is not None:
            READ_AS[value_class] = of

    def __new__(cls, hint: Any) -> Type:
        """Returns the ``Type`` of ``hint``: the existing object if an equal hint has one, else a new object of
        the subclass registered for the hint."""
        try:
            return Type._made[hint]
        except KeyError:
            made = Type._made[hint] = super().__new__(_kind_of(hint))
        except TypeError:  # a hint that can't be hashed (a Param whose Choice holds a dict) is read each time
            made = super().__new__(_kind_of(hint))
        made.hint = hint
        return made

    def __repr__(self) -> str:
        return f"Type({self.hint!r})"

    @functools.cached_property
    def closed(self) -> Any:
        """The hint with each type variable read as its bound, else ``Any``."""
        return substituted(self.hint, {})

    @functools.cached_property
    def _parts(self) -> tuple[Any, Param, tuple[Any, ...]]:
        """The closed hint as ``split_hint`` takes it apart."""
        return split_hint(self.closed)

    @property
    def bare(self) -> Any:
        """The closed hint with ``Annotated`` taken off its members, which is what a hook is given as the type arriving."""
        return self._parts[0]

    @functools.cached_property
    def options(self) -> tuple[Any, ...]:
        """The members of the bare hint, ``| None`` taken off."""
        return tuple(option for option in union_members(self.bare) if option is not type(None))

    @functools.cached_property
    def optional(self) -> bool:
        """Whether a blank (``None``) may arrive: ``X | None``, ``Any``, or a type alias that stands for either."""
        options = union_members(self.bare)
        return Any in options or type(None) in options or any(is_alias(option) and Type(aliased(option)).optional for option in options)

    @functools.cached_property
    def limited(self) -> bool:
        """Whether a value of the right class may still be refused: the hint holds a limit, choices or other metadata."""
        _, param, others = self._parts
        return bool(others or param.limits) or param.choices is not None

    @functools.cached_property
    def reads_as(self) -> str:
        """A text that is the same in every process and differs between any two types that read a value differently.

        A class is written by its module and name, a generic by its arguments, a ``Literal`` by its values, a type
        alias by its name, and a limit or a choice by what it holds. A title, a description and a widget are left
        out, as they change nothing that is read. What a unit's result rests on about a field is this text.
        """
        bare, param, others = self._parts
        held = [f"{limit}={value!r}" for limit, value in sorted(param.limits.items())]
        if param.choices is not None:
            held.append(f"choices={[choice.id for choice in param.choices]!r}")
        held += [repr(other) for other in others]
        return f"{_written(bare)}{held}" if held else _written(bare)

    @functools.cached_property
    def variables(self) -> tuple[TypeVar, ...]:
        """The type variables the hint leaves open, which an edge binds."""
        if isinstance(self.hint, TypeVar):
            return (self.hint,)
        return () if isinstance(self.hint, type) else getattr(self.hint, "__parameters__", ())  # typing's own, on every generic hint

    @functools.cached_property
    def id(self) -> str:
        """The name an editor and a graph record know the type by, ``| None`` off.

        A host's class is known by the id its ``@host_type`` declares, a union by its members' (``money | int``), a generic by its
        class and arguments (``list[int]``, a bare ``list`` for ``list[Any]``), a ``Literal`` by the types of its values, a ``TypedDict`` as
        ``record``, ``Series`` and ``Table`` by the id their own ``Type`` subclass gives (``Series[int]``), and a type alias
        by its own name. A type variable is read as its bound,
        else ``Any``.
        """
        return self._named[0]

    @property
    def text(self) -> str:
        """The id with its blank said (``str | None``): how a type inside another, and a type an edge brings, is named."""
        return f"{self.id} | None" if type(None) in union_members(self.bare) and self.options else self.id

    @functools.cached_property
    def title(self) -> str:
        """What a person reads for the type, from the titles its classes declare."""
        return self._named[1]

    @functools.cached_property
    def classes(self) -> tuple[type, ...]:
        """The classes ``id`` names, each once: what a registry holding the type offers by id."""
        return self._named[2]

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The type's id, its title and the classes they name."""
        members = list(self.options) or [type(None)]
        if len(members) > 1:
            named = [Type(member)._named for member in members]
            return " | ".join(n[0] for n in named), " | ".join(n[1] for n in named), tuple(dict.fromkeys(c for n in named for c in n[2]))
        (one,) = members
        origin, args = get_origin(one) or one, get_args(one)
        if is_alias(one):
            return one.__name__, one.__name__, ()
        if origin is Literal:
            return Type(union_of(dict.fromkeys(type(value) for value in args)))._named
        if typing.is_typeddict(one):  # a record declares no name of its own
            return "record", "Record", ()
        own_id, title = NAMES[origin] if origin in NAMES else _host_name(origin)
        own = () if origin in (Any, type(None)) else (origin,)
        if all(arg is Any for arg in args):  # a generic that says nothing of its items is said bare
            return own_id, title, own
        inner = [Type(arg) for arg in args if arg is not Ellipsis]
        arguments_text = ", ".join("..." if arg is Ellipsis else Type(arg).text for arg in args)
        return f"{own_id}[{arguments_text}]", title, tuple(dict.fromkeys((*own, *(c for of in inner for c in of.classes))))

    @functools.cached_property
    def _built(self) -> tuple[Any, TypeAdapter[Any]]:
        """The hint pydantic reads for the type, and pydantic's adapter of it.

        A container's arguments are each read as a type of their own. Every ``int`` and ``float`` in the schema
        refuses a ``bool``, which pydantic alone reads as 1 (``numbers_refuse_bools``).

        Raises:
            NodeDeclarationError: A ``Param`` that doesn't fit, a misplaced ``Context``, or a type pydantic has no
                schema for (``type_not_carried``).
        """
        bare, param, others = self._parts
        options = []
        for option in union_members(bare):
            if get_origin(option) in _CONTAINERS or get_origin(option) in _KINDS:
                option = get_origin(option)[tuple(arg if arg is Ellipsis else Type(arg).carried for arg in get_args(option))]
            options.append(option)
        read: Any = union_of(options)
        if metadata := (*others, *([param] if param.limits or param.choices is not None else [])):
            read = Annotated[(read, *metadata)]
        marked = Annotated[read, NUMBERS_REFUSE_BOOLS]
        try:  # a record is written by its own config; any other type writes a float that isn't finite as its text
            adapter = TypeAdapter(marked) if len(options) == 1 and is_record(options[0]) else TypeAdapter(marked, config=ConfigDict(ser_json_inf_nan="strings"))
        except SchemaError as error:  # a limit of the wrong kind for the type, such as ge="a" on an int
            raise NodeDeclarationError(f"the limits {param.limits!r} don't fit {self.id}, so change one: {error}", code="bound_invalid") from None
        except PydanticSchemaGenerationError:
            raise NodeDeclarationError(
                f"conductor can't check or write {self.id} as JSON; use a type pydantic knows, such as a pydantic model or a dataclass, "
                "or give the class its own __get_pydantic_core_schema__",
                code="type_not_carried",
            ) from None
        return read, adapter

    @property
    def carried(self) -> Any:
        """The hint pydantic reads for the type, for a hint that holds it (a list's item, a table's column)."""
        return self._built[0]

    @property
    def adapter(self) -> TypeAdapter[Any]:
        """Pydantic's reader and writer of the type."""
        return self._built[1]

    @functools.cached_property
    def _validate(self) -> Callable[..., Any]:
        """Pydantic-core's own validator, which costs a quarter of the adapter's method on each call."""
        return self.adapter.validator.validate_python

    @functools.cached_property
    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of one value; ``{}`` for a type that writes none, which an editor offers no control for."""
        try:
            return self.adapter.json_schema()
        except PydanticInvalidForJsonSchema:
            return {}

    @functools.cached_property
    def choices(self) -> tuple[Choice, ...] | None:
        """The values the type takes when it names them: ``Param(choices=)``, else a ``Literal``'s. Each id is the
        JSON an answer picking it reads as."""
        given = self._parts[1].choices
        if given is None and len(self.options) == 1 and get_origin(self.options[0]) is Literal:
            given = tuple(Choice(value, str(value)) for value in get_args(self.options[0]))
        if given is None:
            return None
        return tuple(choice.model_copy(update={"id": self.to_json(self.read_json(choice.id))}) for choice in given)

    @functools.cached_property
    def exact_classes(self) -> tuple[type, ...]:
        """The classes whose instances, of exactly that class, are taken as they are with no reading.

        Empty when a value must always be read: an option is no plain class (``list[int]``, a ``Literal``, a
        ``TypedDict``, ``Any``, a ``Series`` or a table), or the type is ``limited``.
        """
        options = union_members(self.bare)
        plain = all(isinstance(option, type) and not typing.is_typeddict(option) and option is not Any and option not in _KINDS for option in options)
        return options if plain and not self.limited else ()

    @functools.cached_property
    def hashes_plainly(self) -> bool:
        """Whether its values are plain in JSON: no keys to order, and no float beside a text it could be read as. So
        what a unit received hashes as it is written, and a float that isn't finite reads back from its text."""
        classes = {option for option in union_members(self.bare) if not _KINDS.get(get_origin(option) or option, Type).keys_in_order}
        return classes <= _SCALARS and not {float, str} <= classes

    @property
    def item(self) -> Type | None:
        """The type of one item when the type is a ``Series``; ``None`` for every other type."""
        return None

    @property
    def record(self) -> Type | None:
        """The type of one row when the type is a table (its ``TypedDict``, or ``dict``); ``None`` for every other
        type. An input of a table type wired to a ``Series`` of records is called with all the records as one
        table, and each record must be valid for this type."""
        return None

    @property
    def stored_hint(self) -> Any:
        """The type hint a saved run reads and writes a stored value of the type by. For ``Type(int)`` it is
        ``int``; a ``Series`` type gives the hint of the classes a run stores a Series as."""
        return self.hint

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The hint of each field a record declares, by name: a model's, a dataclass's, a ``TypedDict``'s. None
        for any other type."""
        one = self.options[0] if len(self.options) == 1 else None
        return {name: hint for name, hint in typing.get_type_hints(one).items() if get_origin(hint) is not ClassVar} if is_record(one) else {}

    def read(self, value: Any, *, strict: bool = False) -> Any:
        """Reads ``value`` as the type holds it: as it is when exactly of a plain class, else by pydantic.

        Args:
            value: What a node returned (``strict``), or what a caller gave, a person typed in or an edge brought.
            strict: Pydantic's strict mode: a value already of its type, never ``"12"`` for an ``int``.

        Raises:
            FieldValueError: The type refuses the value.
        """
        if type(value) in self.exact_classes:
            return value
        try:
            return self._validate(value, strict=strict)
        except ValidationError as error:
            raise self._refused(error) from None

    def read_returned(self, value: Any) -> Any:
        """Reads what a node returned for the type, strictly, as a unit holds it.

        Raises:
            FieldValueError: The type refuses the value.
        """
        return self.read(value, strict=True)

    def written_once(self, stored: Any) -> Any | None:
        """Returns the object inside ``stored``, a stored value of the type, that a save should write only once
        when several outputs have that same object; ``None`` when the type has no such object. For a table
        type it is the frame; for ``Type(int)`` and ``5`` it is ``None``."""
        return None

    def read_answer(self, value: Any) -> Any:
        """Validates a person's answer from its JSON form and returns it as the run stores it.
        ``Type(int).read_answer(3)`` is ``3``; for a ``Series`` the list is returned as a ``SeriesItems``.

        Raises:
            FieldValueError: ``value`` has no JSON form, or is not valid for the type.
        """
        return self.read_json(value)

    def read_json(self, value: Any) -> Any:
        """Reads ``value`` strictly from its JSON form, as a person's answer and a declared choice are.

        A yes or no is ``true`` or ``false`` and a number is a number, never a text that spells one; a date is
        still its ISO text, as JSON writes it.

        Raises:
            FieldValueError: The value has no JSON form, or the type refuses it.
        """
        try:
            return self.adapter.validator.validate_json(to_json(value, inf_nan_mode="constants", fallback=untyped_as_json), strict=True)
        except PydanticSerializationError:
            raise FieldValueError("the value has no JSON form") from None
        except ValidationError as error:
            raise self._refused(error) from None

    def to_json(self, value: Any) -> JsonValue:
        """Writes ``value``, one the type holds, in its JSON form; a float that isn't finite as its text.

        Raises:
            PydanticSerializationError: The value has no JSON form.
        """
        written: JsonValue = from_json(to_json(self.adapter.serializer.to_python(value, mode="json"), inf_nan_mode="strings"))
        return written

    @functools.cached_property
    def saved_schema(self) -> CoreSchema:
        """The schema a saved run reads and writes a value of the type by: pydantic's own of the type.

        A float that isn't finite is written as its text, which a ``float`` reads back as the float and a type
        that takes a text keeps as the text. A pydantic model and a dataclass write by their own config whatever
        stands around them, and by default write such a float as ``null``, which a ``float`` field doesn't read
        back. So a type that holds one is written as ``to_json`` writes it: first as Python's JSON values, which
        keep the float.
        """
        schema = self.adapter.core_schema
        if not _holds_a_model(schema):
            return schema
        as_json = core_schema.plain_serializer_function_ser_schema(functools.partial(self.adapter.serializer.to_python, mode="json"))
        if schema["type"] == "definitions":  # what pydantic defined beside the type stays beside it
            return {**schema, "schema": {**schema["schema"], "serialization": as_json}}  # type: ignore[typeddict-unknown-key]
        return {**schema, "serialization": as_json}  # type: ignore[typeddict-unknown-key]

    def fits(self, source: Type) -> dict[TypeVar, Any] | None:
        """Says whether a value of ``source`` may land on this type, and what that binds its type variables to.

        A blank that may arrive lands only where a hint says ``| None``; anywhere else the node isn't called for
        it. So a variable that is the whole hint binds to ``source`` without its blank: ``value: T`` fed
        ``str | None`` is a ``str``, and ``value: T | None`` takes the blank.

        Returns:
            What each variable binds to (``{}`` for a type that has none); ``None`` when the value may not land.
            Two places sharing a variable join, and a source outside a variable's bound does not fit.
        """
        bound: dict[TypeVar, Any] = {}
        lands_without_blank = isinstance(split_hint(self.hint)[0], TypeVar) and source.options
        arriving = union_of(source.options) if lands_without_blank else source.closed
        return bound if fit(self.hint, arriving, bound, False) else None

    def holds(self, source: Type) -> bool:
        """Says whether every value of ``source`` already is one of this type as it is, so nothing is read per call."""
        return not self.limited and fit(self.closed, source.closed, {}, True)

    def bind(self, bound: Mapping[TypeVar, Any]) -> Type:
        """Gives this type with its variables replaced by what ``bound`` binds them to."""
        return Type(substituted(self.hint, bound))

    def _refused(self, error: ValidationError) -> FieldValueError:
        """Makes pydantic's ``error`` one that quotes nothing of the value.

        A ``loc`` keeps list indexes, record fields and table columns, and is cut at the first other step, such
        as a dict key, which is part of the value. An error in a table's column names the column.
        """
        errors = value_free_errors(error)
        columns: list[str] = []
        for error_record, raw in zip(errors, error.errors(include_url=False, include_input=False), strict=True):
            at, kept, column, owner = self, list[int | str](), None, self
            for step in raw["loc"]:
                walked = at._walked(step)
                if walked is None:
                    break
                inside, is_a_place, part = walked
                if part is not None:
                    column, owner = part, at
                at = inside
                if is_a_place:
                    kept.append(step)
            error_record["loc"] = kept
            if column is not None:
                columns.append(column)
                error_record["msg"] = owner._fault_in(column, raw["type"], error_record["msg"])
            elif whole := raw.get("ctx", {}).get("column"):  # a fault of a whole column of a frame
                columns.append(whole)
                error_record["loc"] = [whole]
        return FieldValueError(errors[0]["msg"], errors=errors, column=columns[0] if columns else None)

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Takes one step of a pydantic error's ``loc`` (a list index or a field name) and returns where it leads.
        For ``Type(list[Money])`` and the step ``0`` it leads to ``Type(Money)``.

        Returns:
            A tuple of the type of the part the step leads to, whether the step is kept in the ``loc`` a host is
            shown, and a column name when the step names a table's column. ``None`` when the step is not an index
            or field of this type, such as a dict's key.
        """
        origin = (get_origin(self.options[0]) or self.options[0]) if len(self.options) == 1 else None
        if isinstance(step, int) and isinstance(origin, type) and (issubclass(origin, (list, tuple, set, frozenset)) or origin in _KINDS):
            args = [arg for arg in get_args(self.options[0]) if arg is not Ellipsis]
            return Type(args[min(step, len(args) - 1)] if args else Any), True, None
        if isinstance(step, str) and step in self.fields:
            return Type(self.fields[step]), True, None
        return None

    def _fault_in(self, part: str, kind: str, msg: str) -> str:
        """Returns the error message ``msg`` reworded to name ``part``, a column that ``_walked`` returned. ``kind``
        is pydantic's error type. Only a table rewords; any other type returns ``msg``."""
        return msg

    @staticmethod
    def untyped_as_json(value: Any) -> Any:
        """Returns the JSON form of ``value``, a value of this type's ``value_class`` found where no type is
        declared. It is called as pydantic's ``to_json`` ``fallback``.

        Raises:
            TypeError: Always, unless a subclass overrides this. Pydantic's ``to_json`` expects ``fallback`` to
                raise ``TypeError`` for a value it can't write.
        """
        raise TypeError(type(value).__name__)

    @classmethod
    def untyped_to_hash(cls, value: Any) -> Any:
        """Returns what is hashed for a value of this kind found in what a call received: its JSON, unless a
        subclass has a shorter stand-in for it.

        Raises:
            TypeError: The value has no JSON form.
        """
        return cls.untyped_as_json(value)

    @staticmethod
    def untyped_equal(one: Any, other: Any) -> bool:
        """Returns whether two values found where no type is declared are equal. If ``==`` raises for them, they
        are equal only when they are the same object."""
        try:
            return bool(one == other)
        except (ValueError, TypeError):
            return one is other


def _host_name(cls: type) -> tuple[str, str]:
    """Returns the id and title of a host's class: what its ``@host_type`` declared. ``Box[int]``, the class
    pydantic makes of a generic model ``Box``, is ``box[int]`` with ``Box``'s title.

    Raises:
        NodeDeclarationError: The class is not declared (``type_not_named``).
    """
    generic = vars(cls).get("__pydantic_generic_metadata__")
    if generic is not None and generic["origin"] is not None:
        declared = declaration_of(generic["origin"])
        return f"{declared.id}[{', '.join(Type(arg).text for arg in generic['args'])}]", declared.title
    declared = declaration_of(cls)
    return declared.id, declared.title


def _kind_of(hint: Any) -> type[Type]:
    """Returns the class to make the ``Type`` of ``hint`` from: the subclass registered for ``Series`` or ``Table``
    when the hint is that generic class, with or without ``| None``, else ``Type``."""
    if type(hint) is type and hint not in READ_AS and hint not in _KINDS:
        return Type
    options = [option for option in union_members(split_hint(substituted(hint, {}))[0]) if option is not type(None)]
    if len(options) != 1:
        return Type
    origin = get_origin(options[0]) or options[0]
    return _KINDS.get(origin, Type) if isinstance(origin, type) else Type


def kind_of_value(value: Any) -> type[Type]:
    """Returns the ``Type`` subclass registered for the class of ``value``, else ``Type``: ``TableType`` for a
    frame, ``Type`` for ``5``. Used for a value found where no type is declared, such as inside an ``Any``."""
    return _KINDS.get(READ_AS.get(type(value)), Type)


def untyped_as_json(value: Any) -> Any:
    """Returns the JSON form of a value pydantic can't write itself, such as a frame inside an ``Any``. It is
    passed to pydantic's ``to_json`` as ``fallback``.

    Raises:
        TypeError: No ``Type`` subclass is registered for the value's class.
    """
    return kind_of_value(value).untyped_as_json(value)


def untyped_to_hash(value: Any) -> Any:
    """Returns what to hash for a value pydantic has no JSON form for, found in what a call received. It is the
    ``fallback`` given to ``to_json`` when a call's arguments are fingerprinted: a frame becomes
    ``{"table": "<digest>"}``.

    Raises:
        TypeError: No ``Type`` subclass is registered for the value's class.
    """
    return kind_of_value(value).untyped_to_hash(value)
