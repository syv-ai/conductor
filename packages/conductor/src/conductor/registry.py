"""``NodeRegistry`` — the nodes a host offers, by id, and the type vocabulary they speak.

Defining a node (``@node``) and offering it are two acts. A registry is the
second: it is built whole, ``NodeRegistry([upper, *text.NODES])``, never
changes, and checks what only makes sense for a catalogue — one definition
per id, an alternative that exists. Compile resolves each node in a graph
against it once; the engine then runs what compile resolved.

The registry also owns the **vocabulary**: which value types exist. A
registry's types are the ones its nodes declare on their inputs and
outputs plus the ``types`` a host names, each filed under its ``id``. Nothing about a type is process-wide, so two registries in one
process may each hold a ``text`` of their own; one id claimed by two
classes on one registry is refused where the second arrives, naming
both. ``accepted_as`` — every type in the vocabulary that admits a given
one — is answered here, over these words, and ``describe()`` is the
palette an editor reads: every node's record and every type's, once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from conductor.conductor_model import ConductorModel
from conductor.dtype import DType, _Setting, _UntypedValue
from conductor.errors import NodeDeclarationError
from conductor.node_decorator import NodeDefinition, NodeDescription, NodeVersion
from conductor.series import Series


class TypeDescription(ConductorModel):
    """One word of a registry's vocabulary as an editor reads it.

    Built by ``NodeRegistry.describe()`` for every type in the registry,
    never stored: ``id`` and ``title`` are the type's own, ``accepted_as``
    is the registry's answer to "where may a value of this type land?" —
    the ids of every type in this vocabulary whose ``accepts`` admits it,
    its own included. A field's record (``DType.describe()``) carries the
    id alone and an editor looks the rest up here, so the fact is served
    once per type rather than on every field that names it.
    """

    id: str
    title: str
    accepted_as: tuple[str, ...]


class RegistryDescription(ConductorModel):
    """A registry as an editor reads it: the palette.

    ``nodes`` is every registered node's ``NodeDescription``; ``types`` is
    the vocabulary, one ``TypeDescription`` per word. Built on demand by
    ``NodeRegistry.describe()`` and dumped through pydantic.
    """

    nodes: tuple[NodeDescription, ...]
    types: tuple[TypeDescription, ...]


class NodeRegistry:
    """Maps a node id to its definition, and a type id to the type. Immutable.

    It is a container of definitions by id — ``"upper" in registry``,
    ``registry["upper"]``, ``len(registry)``, and iteration over the ids —
    and not a ``Mapping``: there is no ``get`` to answer a misspelled id with
    ``None``. One entry per type id: the vocabulary, read through ``types``.
    """

    def __init__(self, nodes: Iterable[NodeDefinition[..., Any]] = (), *, types: Iterable[type[DType]] = ()) -> None:
        """A registry holding ``nodes`` and the words in ``types``, checked whole."""
        given = list(nodes)
        for definition in given:
            _refuse_unless_definition(definition)
        #: The definitions, by id, in the order given.
        self._nodes: dict[str, NodeDefinition[..., Any]] = {}
        #: The vocabulary, by id, in the order the words arrived.
        self._types: dict[str, type[DType]] = {}
        self._file(given)
        self._types.update(self._admitted([_word_of(dtype, adding=True) for dtype in types]))

    def __repr__(self) -> str:
        """Its nodes, each as its own repr, one per line: ``NodeRegistry(nodes=(NodeDefinition(...),))``."""
        if not self._nodes:
            return "NodeRegistry(nodes=())"
        lines = "".join(f"    {definition!r},\n" for definition in self._nodes.values())
        return f"NodeRegistry(nodes=(\n{lines}))"

    def _file(self, definitions: list[NodeDefinition[..., Any]]) -> None:
        """File these, after the checks every node passes: one definition per id,
        each deprecation pointing at something that is here."""
        for definition in definitions:
            existing = self._nodes.get(definition.id)
            if existing is not None and existing is not definition:
                raise NodeDeclarationError(f"{definition.id!r} is declared twice; ids are unique.", code="duplicate_node_id", node_id=definition.id)
            self._nodes[definition.id] = definition
            self._types.update(self._admitted(_declared_words(definition)))
        for definition in definitions:
            notices = [definition.deprecation] + [v.deprecation for v in definition.versions.values() if isinstance(v, NodeVersion)]
            for notice in notices:
                if notice is not None and notice.alternative is not None and notice.alternative not in self._nodes:
                    raise NodeDeclarationError(
                        f"{definition.id!r} names {notice.alternative!r} as its alternative, which is not here",
                        code="alternative_missing", node_id=definition.id,
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
        something under its name. Two loaded definitions with one id are refused.
        """
        extended = NodeRegistry()
        extended._nodes, extended._types = dict(self._nodes), dict(self._types)
        fresh = list(definitions)
        for definition in fresh:
            _refuse_unless_definition(definition)
        extended._file([definition for definition in fresh if definition.id not in self._nodes])
        return extended

    # -- the vocabulary ----------------------------------------------------------

    @property
    def types(self) -> Mapping[str, type[DType]]:
        """The vocabulary: every type by its id, in the order the words arrived. Read-only."""
        return MappingProxyType(self._types)

    def accepted_as(self, dtype: type[DType]) -> tuple[str, ...]:
        """The ids of every type in this vocabulary a value of ``dtype`` may land on, its own included.

        Derived from each type's ``accepts``, so a type that widens its
        welcome is served without a table kept by hand. A series is judged
        as its element is, since a node declared with the element runs once
        per row. A bug in a type's ``accepts`` surfaces here, not as a
        silent absence.
        """
        one_value = dtype.element or dtype
        return tuple(word.id for word in self._types.values() if word.accepts(one_value))

    def describe(self) -> RegistryDescription:
        """This registry as the palette: every node's record and the vocabulary, each type once."""
        return RegistryDescription(
            nodes=tuple(definition.describe() for definition in self._nodes.values()),
            types=tuple(
                TypeDescription(id=word.id, title=word.title, accepted_as=self.accepted_as(word))
                for word in self._types.values()
            ),
        )

    def _admitted(self, words: Iterable[type[DType]]) -> dict[str, type[DType]]:
        """``words`` by id, refused as a whole when one id names two classes — among the words or against what is filed."""
        admitted: dict[str, type[DType]] = {}
        for word in words:
            existing = self._types.get(word.id, admitted.get(word.id))
            if existing is not None and existing is not word:
                raise NodeDeclarationError(
                    f"type id {word.id!r} is declared by both {_where(existing)} and {_where(word)}; "
                    "one registry holds one type per id",
                code="duplicate_type_id",
                )
            admitted[word.id] = word
        return admitted


def _refuse_unless_definition(definition: Any) -> None:
    if not isinstance(definition, NodeDefinition):
        raise NodeDeclarationError(f"{definition!r} is not a node; declare one with @node", code="not_a_node_definition")


def _word_of(declared: Any, *, adding: bool = False) -> type[DType] | None:
    """The vocabulary word a declared type names, or ``None`` when it names none.

    A scalar type is its own word; a ``Table[...]`` is ``Table``'s; a
    ``Series[X]`` is ``X``'s. An untyped field and a closed input's plain
    Python type name no word. With ``adding`` — the host
    naming a type in ``types`` — a series, the base ``DType`` and a
    non-type are refused rather than silently naming nothing.
    """
    if not (isinstance(declared, type) and issubclass(declared, DType)):
        if adding:
            raise NodeDeclarationError(f"{declared!r} is not a DType subclass", code="not_a_dtype")
        return None
    if issubclass(declared, (_UntypedValue, _Setting)):
        return None
    if declared is DType:
        if adding:
            raise NodeDeclarationError("the base DType is not a word; add a concrete type", code="base_dtype_added")
        return None
    if issubclass(declared, Series):
        if adding:
            raise NodeDeclarationError(f"{declared!r} is a series; its element is the word to add", code="series_added")
        return _word_of(declared.element)
    if declared._parameterised_from is not None:
        return _word_of(declared._parameterised_from, adding=adding)
    return declared


def _declared_words(definition: NodeDefinition[..., Any]) -> list[type[DType]]:
    """Every word a node's versions declare on their inputs and outputs, each once, in declaration order."""
    words: list[type[DType]] = []
    for version in definition.versions.values():
        for field in (*version.interface.inputs, *version.interface.outputs):
            word = _word_of(field.dtype)
            if word is not None and word not in words:
                words.append(word)
    return words


def _where(dtype: type[DType]) -> str:
    return f"{dtype.__module__}.{dtype.__qualname__}"
