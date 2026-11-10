"""``NodeRegistry`` — the nodes a host offers, by id, and the types they use.

Defining a node (``@node``) and offering it are two acts; a registry is the
second. Nodes go on one as FastAPI's routes go on a router::

    text = NodeRegistry(category="text", tags=["builtin"])

    @text.node("upper")
    def upper(value: str) -> str:
        return value.upper()

    registry = NodeRegistry()
    registry.add_subregistry(text)
    registry.add(shout)              # a plain @node from elsewhere

Every node that enters a registry, by ``add``, ``replace``, the decorator
or a subregistry, takes the registry's defaults by one rule: its tags
follow the registry's, its ``category`` is its own when it sets one, and
each ``extra`` key is its own when it sets that key. A subregistry is
copied in when it is added, so changing it afterwards changes nothing here.

The registry also holds which value types exist: the ones its nodes
declare on their inputs and outputs, plus the ``types`` a host names, each
under its ``id``. Nothing about a type is process-wide, so two registries
in one process may each hold a ``text`` of their own; one id claimed by
two classes in one registry raises where the second arrives, naming both.
``accepted_as`` — every type here that admits a given one — is answered
here, and ``describe()`` is what an editor reads: every node's record and
every type's, once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Concatenate

from pydantic import JsonValue

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.dtype import DType, _PlainPythonType, _UntypedValue, dtype_of
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.node_definition import NodeDefinition, NodeDescription, checked_extra, node


class TypeDescription(ConductorModel):
    """One type of a registry as an editor reads it.

    Built by ``NodeRegistry.describe()`` for every type in the registry,
    never stored: ``id`` and ``title`` are the type's own, ``accepted_as``
    is the registry's answer to "where may a value of this type land?" —
    the ids of every type here whose ``accepts`` admits it,
    its own included. A field's record (``DType.describe()``) carries the
    id alone and an editor looks the rest up here, so the fact is served
    once per type rather than on every field that names it.
    """

    id: str
    title: str
    accepted_as: tuple[str, ...]


class RegistryDescription(ConductorModel):
    """A registry as an editor reads it.

    ``nodes`` is every registered node's ``NodeDescription``; ``types`` is
    the types, one ``TypeDescription`` each. Built on demand by
    ``NodeRegistry.describe()`` and dumped through pydantic.
    """

    nodes: tuple[NodeDescription, ...]
    types: tuple[TypeDescription, ...]


def _takes_the_arguments_of[**P, R](
    declare: Callable[P, R],
) -> Callable[[Callable[..., Any]], Callable[Concatenate[Any, P], R]]:
    """Types a method as taking ``declare``'s arguments, so ``@registry.node`` is written as ``@node`` is."""

    def typed(method: Callable[..., Any]) -> Callable[Concatenate[Any, P], R]:
        return method

    return typed


class NodeRegistry:
    """The nodes a host offers, by id, and the types they use.

    ``"upper" in registry`` and ``registry["upper"]`` look a node up by id;
    looping yields the definitions in the order they arrived. It is not a
    ``Mapping``: there is no ``get`` to answer a misspelled id with ``None``.
    """

    def __init__(
        self,
        *,
        category: str = "",
        tags: Sequence[str] = (),
        extra: Mapping[str, JsonValue] | None = None,
        types: Iterable[type[DType]] = (),
    ) -> None:
        """An empty registry whose nodes default to ``category``, ``tags`` and ``extra``, holding the types in ``types``."""
        self._category = category
        self._tags = tuple(tags)
        self._extra = checked_extra(extra)
        #: The types a host names, whether or not a node declares them.
        self._named_types = tuple(
            dtype for dtype in (_registered_type_of(dtype_of(named) or named, named_by_host=True) for named in types) if dtype is not None
        )
        #: The definitions, by id, in the order they arrived.
        self._nodes: dict[str, NodeDefinition[..., Any]] = {}
        #: Every type, by id: the ones the nodes declare, then the named ones.
        self._types: dict[str, type[DType]] = _types_by_id(self._named_types, ())

    def __repr__(self) -> str:
        """How many nodes it holds, then each as its own repr, one per line."""
        if not self._nodes:
            return "<NodeRegistry of 0 nodes>"
        lines = "".join(f"    {definition!r}\n" for definition in self._nodes.values())
        return f"<NodeRegistry of {len(self._nodes)} node{'s' if len(self._nodes) > 1 else ''}:\n{lines}>"

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> NodeDefinition[..., Any]:
        """The definition registered under ``node_id``; an unknown id is a ``KeyError`` naming the ids there are."""
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeDefinition[..., Any]]:
        """The definitions, in the order they arrived."""
        return iter(tuple(self._nodes.values()))

    # -- what it holds -----------------------------------------------------------

    @_takes_the_arguments_of(node)
    def node(self, *args: Any, **kwargs: Any) -> Any:
        """Declare a node as ``@node`` does, with the same arguments, and add it here.

        The function's name is bound to the definition this registry holds,
        with the registry's defaults applied.
        """
        declare = node(*args, **kwargs)

        def decorate(function: Callable[..., Any]) -> NodeDefinition[..., Any]:
            definition = declare(function)
            self.add(definition)
            return self._nodes[definition.id]

        return decorate

    def add(self, *definitions: NodeDefinition[..., Any]) -> None:
        """Add these nodes, each under its id; an id already here raises, naming where both were declared."""
        self._hold(self._nodes_with(definitions), self._named_types)

    def replace(self, definition: NodeDefinition[..., Any]) -> None:
        """Put ``definition`` in place of the node registered under its id; an id not here is a ``KeyError``."""
        _raise_unless_definition(definition)
        if definition.id not in self._nodes:
            raise self._not_here(definition.id)
        self._hold({**self._nodes, definition.id: self._with_defaults(definition)}, self._named_types)

    def remove(self, node_id: str) -> None:
        """Take out the node registered under ``node_id``; an id not here is a ``KeyError``."""
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        self._hold({held_id: definition for held_id, definition in self._nodes.items() if held_id != node_id}, self._named_types)

    def add_subregistry(self, subregistry: NodeRegistry) -> None:
        """Add every node of ``subregistry`` as ``add`` does, and the types it names.

        Its nodes arrive with its defaults applied, and this registry's go
        over them. They are copied in, so a later change to ``subregistry``
        does not reach this one.
        """
        named = (*self._named_types, *(dtype for dtype in subregistry._named_types if dtype not in self._named_types))
        self._hold(self._nodes_with(tuple(subregistry)), named)

    def filter(self, predicate: Callable[[NodeDefinition[..., Any]], bool]) -> NodeRegistry:
        """A new registry with this one's defaults and named types, holding the nodes ``predicate`` keeps."""
        kept = NodeRegistry(category=self._category, tags=self._tags, extra=self._extra, types=self._named_types)
        kept._hold({definition.id: definition for definition in self._nodes.values() if predicate(definition)}, self._named_types)
        return kept

    def _nodes_with(self, definitions: Iterable[NodeDefinition[..., Any]]) -> dict[str, NodeDefinition[..., Any]]:
        """What this registry holds plus ``definitions``, each under this registry's defaults; a repeated id raises."""
        nodes = dict(self._nodes)
        for definition in definitions:
            _raise_unless_definition(definition)
            held = nodes.get(definition.id)
            if held is not None:
                raise NodeDeclarationError(
                    f"two nodes have the id {definition.id!r}: {_where_declared(held)} and {_where_declared(definition)}; "
                    "ids are unique, so swap one for the other with registry.replace(...)",
                    code="duplicate_node_id",
                )
            nodes[definition.id] = self._with_defaults(definition)
        return nodes

    def _with_defaults(self, definition: NodeDefinition[..., Any]) -> NodeDefinition[..., Any]:
        """``definition`` under this registry's defaults: tags after the registry's, the node's own category and extra keys winning."""
        tags = tuple(dict.fromkeys((*self._tags, *definition.tags)))
        category = definition.category or self._category
        extra = {**self._extra, **definition.extra}
        if (tags, category, extra) == (definition.tags, definition.category, dict(definition.extra)):
            return definition
        return dataclasses.replace(definition, tags=tags, category=category, extra=MappingProxyType(extra))

    def _hold(self, nodes: dict[str, NodeDefinition[..., Any]], named_types: tuple[type[DType], ...]) -> None:
        """Hold exactly ``nodes`` and ``named_types``, once their types give one class per id."""
        self._types = _types_by_id(named_types, nodes.values())
        self._nodes, self._named_types = nodes, named_types

    def _not_here(self, node_id: object) -> KeyError:
        return KeyError(f"{node_id!r} is not registered here; the ids are {', '.join(self._nodes) or 'none'}")

    # -- the types ---------------------------------------------------------------

    @property
    def types(self) -> Mapping[str, type[DType]]:
        """Every type by its id: the ones the nodes declare, then the ones a host named. Read-only."""
        return MappingProxyType(self._types)

    def accepted_as(self, dtype: type[DType]) -> tuple[str, ...]:
        """The ids of every type here a value of ``dtype`` may land on, its own included.

        Derived from each type's ``accepts``, so a type that widens its
        welcome is served without a table kept by hand. A bug in a type's ``accepts`` surfaces here, not as a
        silent absence.
        """
        return tuple(target.id for target in self._types.values() if target.accepts(dtype))

    def describe(self) -> RegistryDescription:
        """This registry as an editor reads it: every node's record and every type's, once."""
        return RegistryDescription(
            nodes=tuple(definition.describe() for definition in self._nodes.values()),
            types=tuple(
                TypeDescription(id=dtype.id, title=dtype.display_title, accepted_as=self.accepted_as(dtype))
                for dtype in self._types.values()
            ),
        )


def _types_by_id(named: Iterable[type[DType]], definitions: Iterable[NodeDefinition[..., Any]]) -> dict[str, type[DType]]:
    """The types ``definitions`` declare, then the named ones, by id; raises when one id names two classes."""
    by_id: dict[str, type[DType]] = {}
    for dtype in (*(dtype for definition in definitions for dtype in _types_declared_by(definition)), *named):
        existing = by_id.setdefault(dtype.id, dtype)
        if existing is not dtype:
            raise NodeDeclarationError(
                f"type id {dtype.id!r} is declared by both {_qualified_name(existing)} and {_qualified_name(dtype)}; "
                "one registry holds one type per id",
                code="duplicate_type_id",
            )
    return by_id


def _raise_unless_definition(definition: Any) -> None:
    if not isinstance(definition, NodeDefinition):
        raise NodeDeclarationError(f"{definition!r} is not a node; declare one with @node", code="not_a_node_definition")


def _where_declared(definition: NodeDefinition[..., Any]) -> str:
    """Where a node's newest version was written: its function's module and name, or that it is a graph."""
    body = definition.versions[definition.current_version].body
    if not callable(body):
        return "a graph"
    return f"{getattr(body, '__module__', '?')}.{getattr(body, '__qualname__', '?')}"


def _registered_type_of(declared: Any, *, named_by_host: bool = False) -> type[DType] | None:
    """The registry type a declared type names, or ``None`` when it names none.

    A scalar type is itself; a ``Table[...]`` is ``Table``. An untyped field and
    a type no edge can carry name none. With ``named_by_host`` — the host
    naming a type in ``types`` — the base ``DType`` and a non-type raise
    rather than silently naming nothing.
    """
    if not (isinstance(declared, type) and issubclass(declared, DType)):
        if named_by_host:
            raise NodeDeclarationError(f"{declared!r} is not a DType subclass", code="not_a_dtype")
        return None
    if issubclass(declared, _UntypedValue) or (issubclass(declared, _PlainPythonType) and not declared.travels_on_edges):
        return None
    if declared is DType:
        if named_by_host:
            raise NodeDeclarationError("the base DType is not a type to add; add a concrete type", code="base_dtype_added")
        return None
    if declared._generic_base is not None:
        return _registered_type_of(declared._generic_base, named_by_host=named_by_host)
    return declared


def _types_declared_by(definition: NodeDefinition[..., Any]) -> list[type[DType]]:
    """Every type a node's versions declare on their inputs and outputs, each once, in declaration order."""
    dtypes: list[type[DType]] = []
    for version in definition.versions.values():
        for field in (*version.interface.inputs, *version.interface.outputs):
            dtype = _registered_type_of(field.dtype)
            if dtype is not None and dtype not in dtypes:
                dtypes.append(dtype)
    return dtypes


def _qualified_name(dtype: type[DType]) -> str:
    return f"{dtype.__module__}.{dtype.__qualname__}"
