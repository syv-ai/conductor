"""``NodeRegistry`` — the nodes a host offers, by id, and the types they use.

Defining a node (``@node``) and offering it are two acts. A registry is the
second: it is built whole, ``NodeRegistry([upper, *text.NODES])``, never
changes, and checks what only makes sense for a catalogue — one definition
per id, an alternative that exists. Compile resolves each node in a graph
against it once; the engine then runs what compile resolved.

The registry also holds which value types exist: the ones its nodes
declare on their inputs and outputs, plus the ``types`` a host names, each
under its ``id``. Nothing about a type is process-wide, so two registries
in one process may each hold a ``text`` of their own; one id claimed by
two classes on one registry raises where the second arrives, naming both.
``accepted_as`` — every type here that admits a given one — is answered
here, and ``describe()`` is what an editor reads: every node's record and
every type's, once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.dtype import DType, _PlainPythonType, _UntypedValue, dtype_of
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.node_definition import NodeDefinition, NodeDescription


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


class NodeRegistry:
    """Maps a node id to its definition, and a type id to the type. Immutable.

    It is a container of definitions by id — ``"upper" in registry``,
    ``registry["upper"]``, ``len(registry)``, and iteration over the ids —
    and not a ``Mapping``: there is no ``get`` to answer a misspelled id with
    ``None``. One entry per type id, read through ``types``.
    """

    def __init__(self, nodes: Iterable[NodeDefinition[..., Any]] = (), *, types: Iterable[type[DType]] = ()) -> None:
        """A registry holding ``nodes`` and the types in ``types``, checked whole."""
        definitions = list(nodes)
        for definition in definitions:
            _raise_unless_definition(definition)
        #: The definitions, by id, in the order given.
        self._nodes: dict[str, NodeDefinition[..., Any]] = {}
        #: The types, by id, in the order they arrived.
        self._types: dict[str, type[DType]] = {}
        self._add_definitions(definitions)
        self._types.update(self._types_checked_for_id_clash([_registered_type_of(dtype_of(dtype) or dtype, named_by_host=True) for dtype in types]))

    def __repr__(self) -> str:
        """Its nodes, each as its own repr, one per line: ``NodeRegistry(nodes=(NodeDefinition(...),))``."""
        if not self._nodes:
            return "NodeRegistry(nodes=())"
        lines = "".join(f"    {definition!r},\n" for definition in self._nodes.values())
        return f"NodeRegistry(nodes=(\n{lines}))"

    def _add_definitions(self, definitions: list[NodeDefinition[..., Any]]) -> None:
        """Add these, after the checks every node passes: one definition per id,
        each deprecation pointing at something that is here."""
        for definition in definitions:
            existing = self._nodes.get(definition.id)
            if existing is not None and existing is not definition:
                raise NodeDeclarationError(f"{definition.id!r} is declared twice; ids are unique.", code="duplicate_node_id")
            self._nodes[definition.id] = definition
            self._types.update(self._types_checked_for_id_clash(_types_declared_by(definition)))
        for definition in definitions:
            for notice in (version.deprecation for version in definition.versions.values()):
                if notice is not None and notice.alternative is not None and notice.alternative not in self._nodes:
                    raise NodeDeclarationError(
                        f"{definition.id!r} names {notice.alternative!r} as its alternative, which is not here",
                        code="alternative_missing",
                    )

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> NodeDefinition[..., Any]:
        """The definition registered under ``node_id``; an unknown id is a ``KeyError`` naming the ids there are."""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"{node_id!r} is not registered here; the ids are {', '.join(self._nodes) or 'none'}") from None

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[str]:
        """The registered ids, in registration order."""
        return iter(self._nodes)

    @property
    def nodes(self) -> tuple[NodeDefinition[..., Any], ...]:
        """Every definition, in the order given."""
        return tuple(self._nodes.values())

    def extended_with(self, definitions: Iterable[NodeDefinition[..., Any]]) -> NodeRegistry:
        """A new registry holding these definitions beside everything here.

        A definition whose id this registry already holds adds nothing, its
        types included, so a host cannot redefine a built-in node by loading
        something under its name. Two loaded definitions with one id raise.
        """
        extended = NodeRegistry()
        extended._nodes, extended._types = dict(self._nodes), dict(self._types)
        new_definitions = list(definitions)
        for definition in new_definitions:
            _raise_unless_definition(definition)
        extended._add_definitions([definition for definition in new_definitions if definition.id not in self._nodes])
        return extended

    # -- the types ---------------------------------------------------------------

    @property
    def types(self) -> Mapping[str, type[DType]]:
        """Every type by its id, in the order they arrived. Read-only."""
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

    def _types_checked_for_id_clash(self, dtypes: Iterable[type[DType]]) -> dict[str, type[DType]]:
        """``dtypes`` by id; raises when one id names two classes, among these or against what the registry holds."""
        admitted: dict[str, type[DType]] = {}
        for dtype in dtypes:
            existing = self._types.get(dtype.id, admitted.get(dtype.id))
            if existing is not None and existing is not dtype:
                raise NodeDeclarationError(
                    f"type id {dtype.id!r} is declared by both {_qualified_name(existing)} and {_qualified_name(dtype)}; "
                    "one registry holds one type per id",
                code="duplicate_type_id",
                )
            admitted[dtype.id] = dtype
        return admitted


def _raise_unless_definition(definition: Any) -> None:
    if not isinstance(definition, NodeDefinition):
        raise NodeDeclarationError(f"{definition!r} is not a node; declare one with @node", code="not_a_node_definition")


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
