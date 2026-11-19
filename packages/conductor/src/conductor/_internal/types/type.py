"""``Type``: one type hint, wrapped with everything conductor needs to know about it.

A node's hint, like ``int`` or ``Annotated[list[str], Param(max_length=3)]``, is just a Python object. ``Type``
wraps it and answers conductor's questions about it. What is its id, and what title does a person see? What is its
pydantic schema? How is a value written as JSON? Which other types fit it, so an edge may join them?

``Type`` owns the check an argument of it gets, ``argument_validator``: its schema with conductor's rules added
(``rules.py``). Compiling asks it of a ``Literal``'s values, the run checks an edge's value with it, and a saved run is
read by it. When it refuses a value, ``Type.refusal`` explains why without quoting the value.

``Type(int)`` returns the same object every time, so the work of reading a hint is done once. ``Series`` and
``Table`` each have a subclass of ``Type``, in their own packages.
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Mapping, Set
from enum import Enum
from types import FunctionType, MethodType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    TypeAliasType,
    TypedDict,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from weakref import WeakValueDictionary

import typing_extensions
from annotated_types import Predicate, Timezone, Unit
from pydantic import (
    BaseModel,
    ImportString,
    Json,
    JsonValue,
    Secret,
    SecretBytes,
    SecretStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.types import EncodedBytes, EncodedStr
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    SchemaError,
    SchemaSerializer,
    SchemaValidator,
)

from conductor._internal.base import (
    JSON_CONFIG,
    JSON_CORE_CONFIG,
    canonical_json,
    plain_json,
)
from conductor._internal.problem import (
    NodeDeclarationError,
    declaration_problems,
    is_conductor_error,
    value_free_errors,
)
from conductor._internal.utils import qualified_name

from .choice import Choice, declared_choices
from .fit import fit
from .hint import (
    READ_AS,
    declares_fields,
    hint_parts,
    is_alias,
    literal_classes,
    substituted,
    union_members,
    union_of,
)
from .host_type import NAMES, OWN_IDS, PYTHON_NAMES, declaration_of
from .param import LIMITS, Param, as_written, not_taken, split_hint
from .rules import with_rules

#: Generic classes whose arguments are each read as a ``Type`` too, so conductor's rules reach their items.
_CONTAINERS = (list, dict, tuple, set, frozenset)

#: The ``Type`` subclass for each generic class that has one: ``Series`` to ``SeriesType``, ``Table`` to
#: ``TableType``.
_KINDS: dict[Any, type[Type]] = {}


def _made_key(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns the key ``Type`` keeps the ``Type`` of ``hint`` by: the hint, with its arguments in the order written.

    Python counts some hints equal that are written in another order: ``int | str == str | int``, and
    ``Literal["a", "b"] == Literal["b", "a"]``, at any depth. Keyed by the hint alone, ``Type(int | str)`` would
    return the ``Type`` made for ``str | int`` while that one lives, and its ``hint`` would read ``str | int``.
    """
    return hint, tuple(_made_key(arg) for arg in get_args(hint))


def _written(hint: Any) -> str:
    """Writes one bare hint as text, for ``Type.reads_as``.

    An ``Annotated`` nested inside it is written in full, with its metadata.
    """
    origin, args = get_origin(hint), get_args(hint)
    if origin is Annotated:
        return Type(hint).reads_as
    if origin is Literal:
        return f"Literal{[repr(value) for value in args]}"
    if is_alias(hint):
        return f"{hint.__module__}.{hint.__name__}"
    if args:
        name = "Union" if origin in (Union, UnionType) else _written(origin)
        return f"{name}[{', '.join(map(_written, args))}]"
    return qualified_name(hint) if isinstance(hint, type) else repr(hint)


def _written_metadata(metadata: Any) -> str:
    """Writes one piece of ``Annotated`` metadata as text, for ``Type.reads_as``::

        _written_metadata(AfterValidator(strip))  # "AfterValidator(func=my_module.strip)"

    The text must be the same in every process. Python's ``repr`` of a function, or of an object with no ``repr``
    of its own, has a memory address in it, which changes every time. So:

    - a function is written as its module and qualified name.
    - a dataclass, such as pydantic's ``AfterValidator`` or ``annotated_types``' ``Ge``, is written as its class
      name with each field written the same way.
    - an object with no ``repr`` of its own is written as its class.
    - anything else is its ``repr``.

    Two lambdas in the same function have the same name, so they are written the same.
    """
    if isinstance(metadata, (FunctionType, MethodType)):
        return qualified_name(metadata)
    # written as the pieces pydantic takes it apart into, so constr(to_lower=True) reads as it always has
    if isinstance(metadata, StringConstraints):
        return ", ".join(repr(piece) for piece in metadata)
    if dataclasses.is_dataclass(metadata):
        written = ", ".join(
            f"{f.name}={_written_metadata(getattr(metadata, f.name))}"
            for f in dataclasses.fields(metadata)
            if f.repr
        )
        return f"{type(metadata).__qualname__}({written})"
    if type(metadata).__repr__ is object.__repr__:
        return qualified_name(type(metadata))
    return repr(metadata)


class TypeRefusal(TypedDict):
    """Why a type refused a value, without quoting the value.

    ``Type.refusal`` returns one. Every kind of problem about a value a type refused takes these keys as its own, so a
    validator passes the refusal whole: ``invalid_input("inputs", name, name=name, **refusal)``.

    Attributes:
        reason: The first error's message.
        errors: Pydantic's errors with the value taken out (``value_free_errors``), each with its ``type``,
            ``loc``, ``msg`` and maybe ``ctx``. Each ``loc`` keeps only steps the type declares, like list indexes
            and field names.
        column: The table column the first error is in, when it is in one.
    """

    reason: str
    errors: list[dict[str, Any]]
    column: NotRequired[str]


class Type:
    """One type hint, with everything conductor needs to know about it.

    ``Type(hint)`` returns the same object for equal hints written in the same order, so the work of reading a hint
    is done once. Nothing is worked out until it is asked for. The object lives as long as something uses it, such as a field or a loaded
    graph. A hint made while a graph loads goes away with the graph.

    ``Series`` and ``Table`` each have a subclass, registered with the class it is for::

        class SeriesType(Type, of=Series): ...

    After that, ``Type(Series[int])`` and ``Type(Series[int] | None)`` return a ``SeriesType``. Every subclass has
    the same attributes and methods. A subclass overrides only the ones whose answer differs for its values. So
    code that has a ``Type`` never checks whether it is a ``Series`` or a ``Table``.

    Attributes:
        hint: The hint, with its ``Annotated`` metadata and its type variables as written.
    """

    hint: Any
    _made: ClassVar[WeakValueDictionary[Any, Type]] = WeakValueDictionary()

    def __init_subclass__(cls, *, of: Any, value_class: type | None = None) -> None:
        """Registers the subclass as the ``Type`` of every hint written with the generic class ``of``.

        Args:
            of: The generic class a node author writes in a hint, such as ``Series`` or ``Table``.
            value_class: The class of the values at run time, if an author may write that class as the hint too,
                such as ``polars.DataFrame`` for ``Table``. A hint written with it is read as one written with
                ``of``.
        """
        _KINDS[of] = cls
        # the generic class's id is its name as a host imports it, so a host can't declare that id
        OWN_IDS.add(of.__qualname__)
        if value_class is not None:
            READ_AS[value_class] = of

    def __new__(cls, hint: Any) -> Type:
        """Returns the ``Type`` of ``hint``.

        If a hint written the same already has one, that object is returned. Otherwise a new one is made, of the
        subclass registered for the hint.
        """
        key = _made_key(hint)
        try:
            return Type._made[key]
        except KeyError:
            made = Type._made[key] = super().__new__(_kind_of(hint))
        # a hint that can't be hashed, like a Param whose Choice has a dict, gets a new object each time
        except TypeError:
            made = super().__new__(_kind_of(hint))
        made.hint = hint
        return made

    def __repr__(self) -> str:
        return f"Type({self.hint!r})"

    @functools.cached_property
    def closed(self) -> Any:
        """The hint with each type variable replaced by its bound, or by ``Any`` if it has none."""
        return substituted(self.hint, {})

    @functools.cached_property
    def _parts(self) -> tuple[Any, Param, tuple[Any, ...]]:
        """The closed hint, taken apart by ``split_hint`` into its bare type, ``Param`` and other metadata."""
        return split_hint(self.closed)

    @property
    def bare(self) -> Any:
        """The closed hint with ``Annotated`` taken off its members.

        This is the type a hook gets, in a parameter annotated ``type``, for the type arriving on an edge.
        """
        return self._parts[0]

    @functools.cached_property
    def options(self) -> tuple[Any, ...]:
        """The members of the bare hint, without ``None``."""
        return tuple(option for option in union_members(self.bare) if option is not type(None))

    @functools.cached_property
    def optional(self) -> bool:
        """Whether the type takes ``None``, as its own check says.

        It asks pydantic, which checks every value of the type, so no second rule says it. ``str | None``, ``Any``,
        ``object`` and ``Literal["a", None]`` take ``None``. ``str``, and ``Annotated[str | None, AfterValidator(f)]``
        where ``f`` refuses ``None``, don't.

        A host's own check may raise something else for ``None``, such as ``float(None)``'s ``TypeError``. That is a
        refusal too: a node returning ``None`` there fails the same way.
        """
        return self.takes_value(None)

    def takes_value(self, value: Any, *, as_it_is: bool = False) -> bool:
        """Says whether the type's argument check (``argument_validator``) takes ``value``, as the run checks it.

        This is the very check a value an edge gives gets on its call, laxly, so a question asked at compile of a
        type's values is answered by the run's own rule. ``fit`` asks it of a ``Literal`` source's values: ``1`` is
        taken by ``Annotated[float, Param(choices=[1.0])]``, ``None`` by any type that takes ``None`` beside its
        choices, and ``True`` is not taken by ``Literal[1]``.

        Args:
            value: The value to check.
            as_it_is: Also ask that the check gives ``value`` back as it is, of the same class, so it needs no
                converting: ``float`` takes ``1``, but gives back ``1.0``.
        """
        validate = self.argument_validator.validate_python  # a fault in the type itself raises here
        try:
            read = validate(value, strict=False)
        # a host's own check may raise something else where pydantic would refuse, like float(None)'s TypeError;
        # that is a refusal too, as ``optional`` says
        except Exception:  # noqa: BLE001
            return False
        return not as_it_is or (type(read) is type(value) and read == value)

    def refuses_as_a_list(self, value: list[Any]) -> bool:
        """Says whether every option of the type refuses the list ``value`` as a whole, not because of an item in it.

        It is the one rule for "this input takes no list", so a list wired or given to it should have been a loop
        over its items: compiling asks it with ``[]`` of an edge's input, and a call with the list it was given.
        ``str`` and a ``TypedDict`` refuse ``["a"]`` as a whole. ``list[str]``, ``tuple[int, ...]``, ``set[int]`` and a
        table take ``[]``, and ``Annotated[list[str], Param(min_length=1)]`` refuses it only by its limit. For
        ``int | list[int]`` given ``["x"]``, ``list[int]`` refuses it because of its item, so a loop isn't the fix.
        Each option is checked without its limits and choices.
        """
        for option in self.options:
            one = Type(option)
            try:
                one.argument_validator.validate_python(value)
            except ValidationError as error:
                if all(not fault["loc"] for fault in one.refusal(error)["errors"]):
                    continue
            # a host's own check may raise something else where pydantic would refuse, as ``takes_value`` says
            except Exception:  # noqa: BLE001
                continue
            return False
        return True

    def check_declared(self) -> None:
        """Refuses the type if a node may not declare it on an input or an output.

        It is asked of each input and output a node declares, so ``@node`` and a hook's fields are held to one set
        of rules. A ``Type`` conductor makes for itself, like ``Type(Any)``, a saved run's, or one whose type variable
        a graph filled in, is never asked.

        A rule about the hint as its author wrote it, such as pydantic's ``Field(...)`` written where ``Param`` goes,
        walks every hint inside it (``hint_parts``): a list's item, a ``TypedDict``'s field and a table's row, but not
        a field of a host's own model. The rule for bytes reads the schema pydantic built, which knows the config each
        class really reads its fields by, so it reaches inside a host's model too. The rules a type's own schema
        holds, its ``Param`` among them, reach as far as conductor builds that schema: a list's item is built as a
        ``Type`` of its own, but a ``TypedDict``'s field is read by pydantic itself. So
        ``Annotated[Series[int], Param(ge=0)]`` limits each item on a field and in a list, and is refused as a
        ``TypedDict``'s field (``param_limit_not_supported``).

        Raises:
            NodeDeclarationError: Pydantic's ``Field(...)`` in the hint (``pydantic_field_in_hint``, or
                ``pydantic_discriminator_in_hint`` with a discriminator), a ``Timezone`` (``timezone_in_hint``), a
                ``Unit`` (``unit_in_hint``), a named predicate written as metadata (``predicate_alias_in_hint``),
                a set (``type_not_supported_set``), a secret (``type_not_supported_secret``), an ``ImportString``
                (``type_not_supported_import_string``), a type that decodes its value
                (``type_not_supported_decoding``), a ``Table`` whose row names no columns
                (``table_row_not_a_typed_dict``), a type pydantic has no schema for (``type_not_supported``), a
                class with no ``@host_type`` (``type_not_named``), named values that don't read back
                (``named_value_invalid``), choices that can't be read (``choices_invalid``), a ``Param`` that
                doesn't fit, or a class with a config of its own that reads a ``bytes`` field as text
                (``type_not_supported_bytes_as_text``).
        """
        for part in hint_parts(self.hint):
            _check_written(part)
            if get_origin(part) in _KINDS:
                Type(part)._check_own_arguments()
        # each of these raises for a fault in what it builds: the schema, a Series of rows stored as a table, the
        # id of a class with no @host_type, named values that don't read back, and choices that can't be read
        self.returned_schema  # noqa: B018
        self.id  # noqa: B018
        self.argument_schema  # noqa: B018
        self.choices  # noqa: B018
        if (as_text := _bytes_read_as_text(self.schema)) is not None:
            cls, field = as_text
            raise NodeDeclarationError(
                declaration_problems.type_not_supported_bytes_as_text(cls=cls, field=field)
            )

    def _check_own_arguments(self) -> None:
        """Refuses the arguments of a ``Series`` or ``Table`` hint that a node may not declare.

        ``check_declared`` calls it for each such hint inside a field's hint, the field's own included. ``Type``'s own
        takes every argument. A subclass with a rule overrides it: ``TableType`` refuses a row that names no columns,
        and ``SeriesType`` items whose choices can't be read.
        """

    @functools.cached_property
    def argument_schema(self) -> CoreSchema:
        """The schema an argument of the type is checked by: the type's ``schema`` with conductor's rules added
        (``with_rules``), so a number refuses a ``bool``.

        A saved run's values are read by it too, so the rules reach every door.
        """
        return with_rules(self.schema)

    @functools.cached_property
    def argument_validator(self) -> SchemaValidator:
        """Pydantic's validator built from ``argument_schema``, once per type.

        It checks a constant, a default, a graph input and a value from an edge, laxly or strictly, and reads the
        type's own choices. When it refuses a value, ``Type.refusal`` explains why without quoting the value.
        """
        return SchemaValidator(self.argument_schema, config=JSON_CORE_CONFIG)

    @functools.cached_property
    def output_schema(self) -> CoreSchema:
        """The schema an output of the type is checked by, strictly: what a node returns, or a person's answer.

        It is ``returned_schema`` with conductor's rules added (``with_rules``). A value is what the type's own check
        gives, whatever its class: pydantic's ``EmailStr`` gives a ``str``, and ``AwareDatetime`` a ``datetime``.
        """
        return with_rules(self.returned_schema)

    @functools.cached_property
    def output_validator(self) -> SchemaValidator:
        """Pydantic's validator built from ``output_schema``, once per type.

        Validators call it with ``strict=True``, so ``"12"`` is never an ``int`` and ``"no"`` is never a ``bool``. A
        ``Series`` comes out as the type a run stores it as (``SeriesItems`` or ``SeriesRows``).
        """
        return SchemaValidator(self.output_schema, config=JSON_CORE_CONFIG)

    @functools.cached_property
    def exact_classes(self) -> tuple[type, ...]:
        """The classes whose values a validator can keep as they are, without calling pydantic.

        A value whose class is exactly one of these needs no checking. ``Type(str)`` gives ``(str,)``, and
        ``Type(int | None)`` gives ``(int, type(None))``.

        Only Python's own classes (``PYTHON_NAMES``) are trusted this way, since every value of one is a value of its
        type. Any other class may make a value its own check refuses, like pydantic's ``ByteSize(-5)`` or a host's
        ``str`` subclass whose constructor checks nothing, so pydantic checks every value of it. It is ``()`` when any
        option of the type is not one of Python's own classes, or when the type has a limit, choices or other
        metadata.
        """
        options = union_members(self.bare)
        own = all(isinstance(option, type) and option in PYTHON_NAMES for option in options)
        return options if own and not self.limited else ()

    @functools.cached_property
    def limited(self) -> bool:
        """Whether a value of the right class may still be refused, because the hint has limits, choices or other
        metadata."""
        _, param, others = self._parts
        return bool(others or param.limits) or param.choices is not None

    @functools.cached_property
    def reads_as(self) -> str:
        """A text for the type that is the same in every process, and differs for any two types that check a value
        differently.

        A class is written as its module and name, a generic with its arguments, a ``Literal`` with its values, and
        a type alias by its name. A limit or a choice is written with its value. Anything else in ``Annotated`` is
        written by ``_written_metadata``. A title, a description and a widget are left out, since they change
        nothing about how a value is checked.

        This is the text that goes into the hash that decides whether a call can be reused.
        """
        bare, param, others = self._parts
        held = [
            f"{limit}={value!r}"
            for limit in sorted(LIMITS)
            if (value := getattr(param, limit)) is not None
        ]
        if param.choices is not None and self.choices is not None:
            held.append(f"choices={[choice.id for choice in self.choices]!r}")
        held += [_written_metadata(other) for other in others]
        return f"{_written(bare)}{held}" if held else _written(bare)

    @functools.cached_property
    def variables(self) -> tuple[TypeVar, ...]:
        """The type variables in the hint, which an edge fills in."""
        if isinstance(self.hint, TypeVar):
            return (self.hint,)
        return (
            () if isinstance(self.hint, type) else getattr(self.hint, "__parameters__", ())
        )  # typing sets __parameters__ on every generic hint

    @functools.cached_property
    def id(self) -> str:
        """The name an editor and a graph record know the type by, without ``| None``.

        For example:

        - a host's class has the id its ``@host_type`` declares, like ``money``.
        - a union joins its members' ids: ``money | int``.
        - a generic has its class and arguments: ``list[int]``, or a bare ``list`` for ``list[Any]``.
        - a ``Literal`` has the types of its values, and a ``TypedDict`` is ``row``.
        - ``Series`` and ``Table`` get their id from their own subclass, like ``Series[int]``.
        - a type alias has its own name, and a type variable the id of its bound, or ``Any``.
        """
        return self._named[0]

    @property
    def text(self) -> str:
        """The id with ``| None`` added if the type allows ``None``, like ``str | None``.

        This is the name used for a type inside another type, and for the type an edge brings.
        """
        return (
            f"{self.id} | None"
            if type(None) in union_members(self.bare) and self.options
            else self.id
        )

    @functools.cached_property
    def title(self) -> str:
        """What a person sees for the type, made from the titles its classes declare."""
        return self._named[1]

    @functools.cached_property
    def classes(self) -> tuple[type, ...]:
        """The classes the id names, each once. A registry lists them by id in ``registry.types``."""
        return self._named[2]

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The type's id, its title and the classes they name, worked out together."""
        members = list(self.options) or [type(None)]
        if len(members) > 1:
            named = [Type(member)._named for member in members]
            return (
                " | ".join(n[0] for n in named),
                " | ".join(n[1] for n in named),
                tuple(dict.fromkeys(c for n in named for c in n[2])),
            )
        (one,) = members
        origin, args = get_origin(one) or one, get_args(one)
        if is_alias(one):
            return one.__name__, one.__name__, ()
        if origin is Literal:
            return Type(literal_classes(one))._named
        if (
            origin is Annotated
        ):  # a union's member keeps what checks it, which doesn't change its name
            return Type(one)._named
        # a TypedDict has no declared name, so every one is "row"
        if typing_extensions.is_typeddict(one):
            return "row", "Row", ()
        # checked before NAMES is looked up, since a value that is no type, like [int], may not be hashable
        _check_is_type(one)
        own_id, title = NAMES[origin] if origin in NAMES else _host_name(origin)
        own = () if origin in (Any, type(None)) else (origin,)
        if all(
            arg is Any for arg in args
        ):  # a generic that says nothing about its items is named bare
            return own_id, title, own
        inner = [Type(arg) for arg in args if arg is not Ellipsis]
        arguments_text = ", ".join("..." if arg is Ellipsis else Type(arg).text for arg in args)
        return (
            f"{own_id}[{arguments_text}]",
            title,
            tuple(dict.fromkeys((*own, *(c for of in inner for c in of.classes)))),
        )

    @functools.cached_property
    def _built(self) -> tuple[Any, TypeAdapter[Any]]:
        """The hint pydantic is given for the type, and pydantic's adapter for it.

        The type's schema and serializer come from the adapter. The arguments of a container, like the ``int`` of
        ``list[int]``, are each built as a ``Type`` of their own first, so conductor's rules reach them too.

        Raises:
            NodeDeclarationError: A ``Param`` that doesn't fit, a misplaced ``Context``, or a type pydantic has no
                schema for (``type_not_supported``).
        """
        bare, param, others = self._parts
        if isinstance(bare, Mapping):
            # pydantic reads a dict as a core schema, so a dict given as a type, like {"a": int}, is refused before
            # its schema is built
            _check_is_type(bare)
        options = []
        for option in union_members(bare):
            if get_origin(option) in _CONTAINERS or get_origin(option) in _KINDS:
                option = get_origin(option)[
                    tuple(arg if arg is Ellipsis else Type(arg).carried for arg in get_args(option))
                ]
            elif get_origin(option) is Annotated:  # a union's member, built as a type of its own
                option = Type(option).carried
            options.append(option)
        read: Any = union_of(options)
        if metadata := (*others, *([param] if param.limits or param.choices is not None else [])):
            read = Annotated[(read, *metadata)]
        # Every type writes and reads by JSON_CONFIG: a float that isn't finite as text, and bytes as base64. Pydantic
        # refuses a config beside a class that declares its fields, so such a class is handed to it under a type alias.
        # A model and a pydantic dataclass still write their fields by their own config, wherever they sit. A TypedDict
        # or a plain dataclass takes the config it sits in, as inside a list.
        try:
            given: Any = (
                TypeAliasType("Fields", read)
                if len(options) == 1 and declares_fields(options[0])
                else read
            )
            adapter = TypeAdapter(given, config=JSON_CONFIG)
        # a limit of the wrong kind for the type, like ge="a" on an int
        except SchemaError as error:
            raise NodeDeclarationError(
                declaration_problems.param_limit_value_invalid(
                    limits=repr(param.limits), type=self.id, reason=str(error)
                )
            ) from None
        except PydanticSchemaGenerationError:
            raise NodeDeclarationError(
                declaration_problems.type_not_supported(type=self.id)
            ) from None
        # a StringConstraints that sets more than a pattern is pydantic's, and held to its type as a Param's limits are
        for constraints in (other for other in others if isinstance(other, StringConstraints)):
            given = [
                f.name
                for f in dataclasses.fields(constraints)
                if getattr(constraints, f.name) is not None
            ]
            if unknown := not_taken(adapter.core_schema, given):
                raise NodeDeclarationError(
                    declaration_problems.param_limit_not_supported(
                        unknown_limits=unknown, type=as_written(bare)
                    )
                )
        return read, adapter

    @property
    def carried(self) -> Any:
        """The hint pydantic is given for the type, for use inside another hint, like a list's item or a table's
        column."""
        return self._built[0]

    @property
    def schema(self) -> CoreSchema:
        """Pydantic's core schema of the type, which pydantic checks and writes a value of it by.

        Raises ``NodeDeclarationError`` (``type_not_supported``) for a type pydantic can't handle.
        """
        return self._built[1].core_schema

    @property
    def serializer(self) -> SchemaSerializer:
        """Pydantic's serializer for a value of the type."""
        return self._built[1].serializer

    @functools.cached_property
    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of one value of the type.

        It is ``{}`` for a type pydantic can't write a JSON schema for. An editor then offers no control for it.
        """
        try:
            return self._built[1].json_schema()
        except PydanticInvalidForJsonSchema:
            return {}

    @functools.cached_property
    def choices(self) -> tuple[Choice, ...] | None:
        """The choices the type takes, if it names any, or ``None``.

        They come from ``Param(choices=...)``, or else from a ``Literal``'s values or an enum's members. Each is
        checked and converted by the type's argument check (``declared_choices``), so ``Param(choices=[1])`` on a
        ``float`` has the id ``1.0``, and the enum ``Color``'s are its values, like ``"red"``.

        Raises:
            NodeDeclarationError: ``choices_invalid`` when ``Param``'s or a ``Literal``'s can't be read, as with a
                limit no value meets. ``Field`` reads them, so ``@node`` refuses such a type. An enum's members and a
                ``Literal``'s values read back by the time they are read here (``with_rules``).
        """
        given = self._parts[1].choices
        if given is None and len(self.options) == 1:
            (option,) = self.options
            if get_origin(option) is Literal:
                given = get_args(option)
            elif isinstance(option, type) and issubclass(option, Enum):
                given = tuple(option)
        if given is None:
            return None
        return declared_choices(given, self.argument_validator, self.to_json)

    def narrowed(self, choices: tuple[Choice, ...]) -> Type:
        """Returns this type with its values limited to ``choices``, as an ``Ask`` limits a question's.

        The choices are a ``Param(choices=)`` on the type's classes, with its limits kept. A ``Literal`` is read as its
        values' classes, since a ``Literal``'s own choices are always all its values. So narrowing
        ``Literal["ida", "bo"] | None`` to ``"ida"`` gives ``Annotated[str | None, Param(choices=["ida"])]``, and its
        JSON schema's ``enum`` is ``["ida", None]``, written as for any other type with choices.

        Any other metadata, like a validator, is left out. The choices were read through it once already, and a check
        that converts a value, like one that adds ``"!"``, would convert them again, so its own choice would be
        refused. It changes nothing in the schema, which is what a narrowed type is for.

        Args:
            choices: Some of the type's values, each id in its JSON form, as ``read_choices`` gives them.
        """
        bare, param, _ = self._parts
        classes = Type(literal_classes(bare))
        # each id back as the value its classes read it as, so a float's "Infinity" is inf again
        values = [
            choice.model_copy(
                update={"id": classes.argument_validator.validate_json(canonical_json(choice.id))}
            )
            for choice in choices
        ]
        return Type(Annotated[(classes.hint, *param.limits, Param(choices=values))])

    @property
    def item(self) -> Type | None:
        """The type of one item, for a ``Series``. ``None`` for every other type."""
        return None

    @property
    def row(self) -> Type | None:
        """The type of one row, for a table: its ``TypedDict``, or ``dict``. ``None`` for every other type.

        A table input wired to a ``Series`` of rows gets all the rows as one table. Each row must be a valid
        value of this type.
        """
        return None

    @property
    def stored_hint(self) -> Any:
        """The type hint a saved run uses to write and read back a stored value of the type.

        For ``Type(int)`` it is ``int``. A ``Series`` gives the hint of the classes a run stores a Series as.
        """
        return self.hint

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The type hint of each field, by name, for a type that declares its fields: a model, a dataclass or a
        ``TypedDict``. Empty for any other type."""
        one = self.options[0] if len(self.options) == 1 else None
        return (
            {
                name: hint
                for name, hint in typing.get_type_hints(one).items()
                if get_origin(hint) is not ClassVar
            }
            if declares_fields(one)
            else {}
        )

    @property
    def returned_schema(self) -> CoreSchema:
        """The schema that checks what a node returned for an output of the type, or what a person answered.

        For most types it is ``schema``. A ``Series`` differs, since a run stores it as a ``SeriesItems`` or
        ``SeriesRows``.
        """
        return self.schema

    def to_json(self, value: Any) -> JsonValue:
        """Converts ``value``, a value of the type, to JSON data.

        A float that isn't finite, like ``inf``, becomes its text.

        Raises:
            PydanticSerializationError: The value can't be written as JSON.
        """
        return plain_json(self.serializer.to_python(value, mode="json"))

    def fits(self, source: Type) -> dict[TypeVar, Any] | None:
        """Says whether an edge may pass a value of ``source`` to this type, and what that binds its type variables
        to.

        A ``None`` reaches an input only if its hint says ``| None``. Anywhere else, the node isn't called for it.
        So when a type variable is the whole hint, it binds to ``source`` without its ``None``. Fed
        ``str | None``, ``value: T`` binds ``T`` to ``str``, and ``value: T | None`` takes the ``None`` too.

        Returns:
            What each type variable binds to, or ``{}`` for a type without any. ``None`` if the value may not be
            passed. When a variable appears in two places, it binds to the union of both. A source outside a
            variable's bound doesn't fit.
        """
        bound: dict[TypeVar, Any] = {}
        lands_without_blank = isinstance(split_hint(self.hint)[0], TypeVar) and source.options
        arriving = union_of(source.options) if lands_without_blank else source.closed
        return bound if fit(self.hint, arriving, bound, False) else None

    def holds(self, source: Type) -> bool:
        """Says whether every value of ``source`` is already a value of this type, so it needs no check per call."""
        return fit(self.closed, source.closed, {}, True)

    def bind(self, bound: Mapping[TypeVar, Any]) -> Type:
        """Returns this type with its type variables replaced by what ``bound`` binds them to."""
        return Type(substituted(self.hint, bound))

    def refusal(
        self, error: ValidationError, under: tuple[str | int, ...] = (), *, exactly: bool = False
    ) -> TypeRefusal:
        """Explains why the type refused a value, without quoting the value.

        Say ``Type(list[int])`` refuses ``[1, "x"]``. The ``TypeRefusal`` has the ``reason``
        ``"Input should be a valid integer, unable to parse string as an integer"``, and one error with the
        ``loc`` ``[1]``.

        A class's own check that refuses a value of exactly that class says so in the ``reason``, since that is a fault
        in the class's schema, not in the value: a schema must take its own instance.

        A ``loc`` keeps list indexes, declared fields and table columns. It stops at the first other step, such as a
        dict key, since a key is part of the value. An error in a table's column names the column.

        Args:
            error: What pydantic raised when it checked a value of this type, or a value with one of this type
                inside it (see ``under``).
            under: The ``loc`` steps from what pydantic checked down to the value of this type. Say pydantic
                checked a mapping of outputs, and this is the type of the output ``count``: then ``under`` is
                ``("count",)``. Only the errors inside that value are used, and their ``loc`` starts there.
            exactly: Use only the errors at ``under`` itself, none inside it. A fault of a whole that holds values
                of other types, such as an answer of several questions that isn't a mapping, is explained this way,
                by ``Type(Any)``.
        """
        start = len(under)
        pairs = [
            (value_free_error, raw)
            for value_free_error, raw in zip(value_free_errors(error), error.errors(), strict=True)
            if (raw["loc"] if exactly else raw["loc"][:start]) == under
        ]
        value_free = [value_free_error for value_free_error, _ in pairs]
        columns: list[str] = []
        for value_free_error, raw in pairs:
            at, kept, column, owner = self, list[int | str](), None, self
            in_cell: tuple[int | str, ...] = ()
            steps = raw["loc"][start:]
            for number, step in enumerate(steps):
                walked = at._walked(step)
                if walked is None:
                    break
                inside, is_a_place, part = walked
                if part is not None:
                    column, owner, in_cell = part, at, tuple(steps[number + 1 :])
                at = inside
                if is_a_place:
                    kept.append(step)
            value_free_error["loc"] = kept
            if column is not None:
                columns.append(column)
                value_free_error["type"], value_free_error["msg"] = owner._fault_in(
                    column, raw["type"], value_free_error["msg"], in_cell
                )
            # an error conductor raised about a whole column of a frame; a host's error may put anything in its ctx
            elif is_conductor_error(raw) and (whole := raw.get("ctx", {}).get("column")):
                columns.append(whole)
                value_free_error["loc"] = [whole]
        refusal = TypeRefusal(
            reason=self._own_class_refused(pairs[0][1], value_free[0]["msg"], start)
            or value_free[0]["msg"],
            errors=value_free,
        )
        if columns:
            refusal["column"] = columns[0]
        return refusal

    def _own_class_refused(self, first: Any, message: str, start: int) -> str | None:
        """Returns the reason for a refusal of a value of exactly the type's own class, or ``None`` for any other.

        Say a host's ``Colour`` reads only text, and a node returns ``Colour("red")``. Pydantic's message,
        "Input should be a valid string", says nothing of the cause: the class's schema doesn't take a ``Colour``. A
        type with a limit, choices or other metadata may refuse a value of its class, so it is left out, and so is one
        of Python's own classes, whose check refuses its own value only by conductor's rules, like an ``inf``. So is a
        refusal conductor raised itself, such as ``no_json_form``: the class's own check never ran.

        Args:
            first: The first of pydantic's raw errors, with the value it refused in ``input``.
            message: The same error's message without the value (``value_free_errors``), which the reason quotes,
                since a host's own check may write the value into its message.
            start: How many steps of its ``loc`` lead down to the value of this type, as ``refusal``'s ``under``.
        """
        refused = type(first.get("input"))
        if (
            first["loc"][start:]
            or is_conductor_error(first)
            or self.limited
            or refused in PYTHON_NAMES
            or refused not in self.options
        ):
            return None
        return f"the type's own check refuses a {refused.__name__}, a value of its own class ({message})"

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of a pydantic error's ``loc``, a list index or a field name, and returns where it leads.

        For ``Type(list[Money])`` and the step ``0``, it leads to ``Type(Money)``.

        Returns:
            Three things: the type the step leads to, whether the step is kept in the ``loc`` a host is shown, and
            a column name if the step names a table's column. ``None`` if the step isn't an index or a field of
            this type, such as a dict's key.
        """
        origin = (
            (get_origin(self.options[0]) or self.options[0]) if len(self.options) == 1 else None
        )
        if (
            isinstance(step, int)
            and isinstance(origin, type)
            and (issubclass(origin, (list, tuple, set, frozenset)) or origin in _KINDS)
        ):
            args = [arg for arg in get_args(self.options[0]) if arg is not Ellipsis]
            return Type(args[min(step, len(args) - 1)] if args else Any), True, None
        if isinstance(step, str) and step in self.fields:
            return Type(self.fields[step]), True, None
        return None

    def _fault_in(
        self, part: str, kind: str, msg: str, in_cell: tuple[int | str, ...]
    ) -> tuple[str, str]:
        """Returns the error's type and message, reworded to name the column ``part`` that ``_walked`` returned.

        ``kind`` is pydantic's error type and ``msg`` its message. ``in_cell`` is the rest of the error's ``loc``
        after the column, such as ``("money", "cents")`` for a field missing from a value inside a cell. It is
        ``()`` when the error is about the column itself. Only a table rewords the error. Any other type returns
        ``kind`` and ``msg`` as they are.
        """
        return kind, msg


def json_by_its_class(value: Any) -> JsonValue:
    """Writes an object as JSON by its own class's schema, where pydantic can't write it by itself.

    Pydantic writes a value it is handed without knowing its type. An object whose class gives its own schema, such
    as pydantic-extra-types' ``Color``, can only be written by that schema. Every place that writes or hashes a value
    no type is known for passes this, or a function that calls it, as pydantic's ``fallback``, so such an object is
    written one way everywhere: a value read from its JSON, a call's arguments hashed to reuse it, and the inputs a
    waiting call shows.

    Raises:
        TypeError: The object's class has no schema that writes it. Pydantic expects a ``fallback`` to raise it.
    """
    try:
        return Type(type(value)).to_json(value)
    # a class pydantic has no schema for, or whose schema can't write it
    except (NodeDeclarationError, PydanticSerializationError):
        raise TypeError(type(value).__name__) from None


def _check_written(part: Any) -> None:
    """Refuses one hint inside a field's hint, as its author wrote it, if a node may not declare it.

    ``check_declared`` calls it for every hint ``hint_parts`` finds, and an ``Annotated``'s metadata is found as a
    part of its own. Pydantic would build ``Timezone`` and ``Unit``, and a named predicate such as ``LowerCase``
    written as metadata, and check nothing. Its ``Field(...)`` belongs on a model's field, and a ``Param`` goes in
    a node's hint. A set has no order, so it has no JSON form a value could be written in and read back by. A secret
    writes a mask as its JSON, an ``ImportString`` runs the code its value names, and a type that decodes its value,
    such as ``Json[int]`` or ``Base64Bytes``, refuses the decoded value the next time it is checked.

    Raises:
        NodeDeclarationError: The kinds ``check_declared`` lists for each of these.
    """
    if isinstance(part, FieldInfo):
        if part.discriminator is not None:
            raise NodeDeclarationError(declaration_problems.pydantic_discriminator_in_hint())
        raise NodeDeclarationError(declaration_problems.pydantic_field_in_hint())
    if isinstance(part, Timezone):
        raise NodeDeclarationError(declaration_problems.timezone_in_hint())
    if isinstance(part, Unit):
        raise NodeDeclarationError(declaration_problems.unit_in_hint())
    if get_origin(part) is Annotated:
        for metadata in get_args(part)[1:]:
            predicates = (
                [m for m in get_args(metadata)[1:] if isinstance(m, Predicate)]
                if get_origin(metadata) is Annotated
                else []
            )
            if predicates:
                raise NodeDeclarationError(
                    declaration_problems.predicate_alias_in_hint(predicate=repr(predicates[0]))
                )
    origin = get_origin(part) or part
    if isinstance(origin, type) and issubclass(origin, Set):
        raise NodeDeclarationError(
            declaration_problems.type_not_supported_set(type=as_written(part))
        )
    if _marked(part, (Secret, SecretStr, SecretBytes)):
        raise NodeDeclarationError(
            declaration_problems.type_not_supported_secret(type=as_written(part))
        )
    if _marked(part, (ImportString,)):
        raise NodeDeclarationError(
            declaration_problems.type_not_supported_import_string(type=as_written(part))
        )
    if _marked(part, (Json, EncodedBytes, EncodedStr)):
        raise NodeDeclarationError(
            declaration_problems.type_not_supported_decoding(type=as_written(part))
        )


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


#: The core schema types of a class that reads its fields by a pydantic config: a model, a pydantic or plain
#: dataclass, and a ``TypedDict``. Pydantic writes on each the config it really reads that class's fields by.
_CONFIGURED = frozenset({"model", "dataclass", "typed-dict"})


def _bytes_read_as_text(
    schema: Any,
    definitions: dict[str, Any] | None = None,
    owner: tuple[str, str] | None = None,
) -> tuple[str, str] | None:
    """Finds a ``bytes`` field, in a type's core schema, that its class reads and writes as text.

    Conductor writes and reads bytes as base64 text (``JSON_CONFIG``). A class with a config of its own, such as a
    pydantic model, or a ``TypedDict`` or plain dataclass given one with pydantic's ``with_config``, reads its fields
    by that config, and a config that doesn't set the bytes to base64 takes them as text, which a saved run can't read
    back. Pydantic writes on the schema of each model, dataclass and ``TypedDict`` the config it reads that class's
    fields by. A ``TypedDict`` or plain dataclass with no config of its own takes the config it sits in, so the class
    at fault is the one around it that has the config, and the field named is that class's.

    Say ``class Attachment(BaseModel)`` has the field ``data: bytes`` and the default config. Then the schema of
    ``list[Attachment]`` gives ``("Attachment", "data")``. With ``ser_json_bytes="base64"`` and
    ``val_json_bytes="base64"`` in its config, it gives ``None``.

    Args:
        schema: A core schema, or a part of one.
        definitions: The schemas pydantic declared beside it, by their ``ref``. Each is taken out as it is walked,
            so it is walked once.
        owner: The class with a config of its own that ``schema`` is inside, when that class reads bytes as text,
            and its field, or ``""`` until the walk reaches one of its fields.

    Returns:
        The class's name and its field, or ``None`` when no bytes in ``schema`` are read as text.
    """
    definitions = {} if definitions is None else definitions
    parts: list[tuple[Any, tuple[str, str] | None]] = []
    if isinstance(schema, (list, tuple)):
        parts = [(part, owner) for part in schema]
    elif isinstance(schema, dict):
        kind = schema.get("type")
        if kind == "bytes":
            return owner
        if not isinstance(kind, str):
            # not a schema but fields by name; a field of the class with the config names the field
            parts = [(part, _in_field(owner, name)) for name, part in schema.items()]
        elif kind == "definition-ref":
            parts = [(definitions.pop(schema["schema_ref"], None), owner)]
        else:
            if kind == "definitions":
                definitions.update({one["ref"]: one for one in schema["definitions"]})
            # a class with no config of its own, and a schema conductor builds itself, like a table's, which has no
            # class, read by the config they sit in
            if kind in _CONFIGURED and _has_own_config(schema.get("cls")):
                config = schema.get("config", {})
                as_text = (
                    config.get("ser_json_bytes") != "base64"
                    or config.get("val_json_bytes") != "base64"
                )
                owner = (schema["cls"].__name__, "") if as_text else None
            if kind == "dataclass-field":
                owner = _in_field(owner, schema["name"])
            parts = [
                (part, owner)
                for key, part in schema.items()
                if key not in ("serialization", "metadata", "config", "definitions")
            ]
    for part, inside in parts:
        if (found := _bytes_read_as_text(part, definitions, inside)) is not None:
            return found
    return None


def _has_own_config(cls: Any) -> bool:
    """Whether ``cls`` reads its fields by a pydantic config of its own: a model, a pydantic dataclass, or a class
    pydantic's ``with_config`` gave one. ``False`` for ``None``, a schema with no class."""
    return isinstance(cls, type) and (
        issubclass(cls, BaseModel) or getattr(cls, "__pydantic_config__", None) is not None
    )


def _in_field(owner: tuple[str, str] | None, name: str) -> tuple[str, str] | None:
    """Returns ``owner`` inside its field ``name``, when the walk has not yet reached one of its fields.

    A field further down, of a class with no config of its own, keeps the owner's field, which is the one to fix.
    """
    return (owner[0], name) if owner is not None and not owner[1] else owner


def _check_is_type(hint: Any) -> None:
    """Checks that a hint is a type, or a generic of one, and not some other value like ``5``, ``[int]`` or ``{}``.

    Raises:
        NodeDeclarationError: ``type_not_supported`` if the hint is no type.
    """
    if not isinstance(get_origin(hint) or hint, type):
        raise NodeDeclarationError(declaration_problems.type_not_supported(type=repr(hint)))


def _host_name(cls: type) -> tuple[str, str]:
    """Returns the id and title a host's class declared with ``@host_type``.

    For a generic pydantic model ``Box``, pydantic makes a class ``Box[int]``. Its id is ``box[int]``, and its
    title is ``Box``'s.

    Raises:
        NodeDeclarationError: ``type_not_named`` if the class has no declaration.
    """
    generic = vars(cls).get("__pydantic_generic_metadata__")
    if generic is not None and generic["origin"] is not None:
        declared = declaration_of(generic["origin"])
        return (
            f"{declared.id}[{', '.join(Type(arg).text for arg in generic['args'])}]",
            declared.title,
        )
    declared = declaration_of(cls)
    return declared.id, declared.title


def _kind_of(hint: Any) -> type[Type]:
    """Returns the class to make the ``Type`` of ``hint`` from.

    That is the subclass registered for ``Series`` or ``Table`` when the hint is one of them, with or without
    ``| None``. For any other hint it is ``Type``.
    """
    if type(hint) is type and hint not in READ_AS and hint not in _KINDS:
        return Type
    options = [
        option
        for option in union_members(split_hint(substituted(hint, {}))[0])
        if option is not type(None)
    ]
    if len(options) != 1:
        return Type
    origin = get_origin(options[0]) or options[0]
    return _KINDS.get(origin, Type) if isinstance(origin, type) else Type
