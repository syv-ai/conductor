"""``NodeRegistry`` — the node classes a host offers, by id, and the type vocabulary they speak.

Defining a node (subclassing ``NodeDefinition``) and offering it are two
acts. A registry is the second: ``register(cls)`` files the class under
its ``id`` and checks the rules that only make sense for a catalogue —
versions numbered from 1 with no holes, a deprecated current version
pointing somewhere, an alternative that exists. ``runner_for`` gives the
engine a plain callable for one registered version.

The registry also owns the **vocabulary**: which value types exist. A
registry's types are the ones its nodes declare on their inputs and
outputs plus what the host adds with ``add_types``, each filed under its
``id``. Nothing about a type is process-wide, so two registries in one
process may each hold a ``text`` of their own; one id claimed by two
classes on one registry is refused where the second arrives, naming
both. ``accepted_as`` — every type in the vocabulary that admits a given
one — is answered here, over these words, and ``describe()`` is the
palette an editor reads: every node's record and every type's, once.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Callable

from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph, GraphNode
from conductor.model import ConductorModel
from conductor.node import NodeDefinition, NodeDescription, NodeVersion
from conductor.ref import Ref
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
    """Maps a node id to the node class itself, and a type id to the type.

    One entry per node id, not per (id, version): the class knows which
    versions it declares, and a caller picks one with
    ``registry.get(node.type).versions[node.version]``. One entry per
    type id: the vocabulary, read through ``types``.
    """

    def __init__(self, nodes: Iterable[type[NodeDefinition]] = ()) -> None:
        """A registry holding ``nodes``, each registered in order — the form its repr prints."""
        #: The classes, by id, in registration order.
        self._nodes: dict[str, type[NodeDefinition]] = {}
        #: The vocabulary, by id, in the order the words arrived.
        self._types: dict[str, type[DType]] = {}
        for node_cls in nodes:
            self.register(node_cls)

    def __repr__(self) -> str:
        """Its nodes, each as its own repr, one per line: ``NodeRegistry(nodes=(Greet(...),))``."""
        if not self._nodes:
            return "NodeRegistry(nodes=())"
        lines = "".join(f"    {cls!r},\n" for cls in self._nodes.values())
        return f"NodeRegistry(nodes=(\n{lines}))"

    # -- nodes -------------------------------------------------------------------

    def register(self, node_cls: type[NodeDefinition]) -> None:
        """File ``node_cls`` under its id, and its declared types under theirs.

        Nothing is filed if anything is refused: a version numbering with a
        hole, a deprecation pointing nowhere, or a declared type whose id
        another class already holds here.
        """
        if not (isinstance(node_cls, type) and issubclass(node_cls, NodeDefinition)):
            raise TypeError(f"{node_cls!r} must be a NodeDefinition subclass")
        existing = self._nodes.get(node_cls.id)
        if existing is not None and existing is not node_cls:
            raise ValueError(
                f"{node_cls.id!r} is already registered by {existing.__name__}; ids are unique."
            )
        declared = set(node_cls.versions)
        if declared != set(range(1, max(declared) + 1)):
            raise ValueError(
                f"{node_cls.id!r} declares versions {sorted(declared)}; "
                "a registered node numbers from 1 with no hole — a graph can pin "
                "any version up to the current one. A loaded definition "
                "(``extended_with``) carries exactly the versions its host admitted."
            )
        current = node_cls.versions[node_cls.current]
        if (
            isinstance(current, NodeVersion)
            and current.deprecation is not None
            and node_cls.deprecation is None
        ):
            raise ValueError(
                f"{node_cls.id!r} deprecates its current version {node_cls.current} with no "
                "newer one to move to; retire the node (@deprecated on the class) or add a version"
            )
        notices = [node_cls.deprecation] + [
            v.deprecation for v in node_cls.versions.values() if isinstance(v, NodeVersion)
        ]
        for notice in notices:
            if notice is not None and notice.alternative is not None and not self.contains(notice.alternative):
                raise ValueError(
                    f"{node_cls.id!r} names {notice.alternative!r} as its alternative, which is not "
                    "registered here; register the replacement before the node it replaces"
                )
        words = self._admitted(_declared_words(node_cls))
        self._nodes[node_cls.id] = node_cls
        self._types.update(words)

    def get(self, node_id: str) -> type[NodeDefinition] | None:
        """The class registered under ``node_id``, or ``None``."""
        return self._nodes.get(node_id)

    def contains(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> tuple[type[NodeDefinition], ...]:
        """Every registered node class, in registration order."""
        return tuple(self._nodes.values())

    def upgraded(self, graph: Graph, node_id: str, *, to: int | None = None) -> Graph:
        """``graph`` with one node moved from the version it was saved at to ``to``, the current one by default.

        Runs the node's ``@upgrade`` steps in order over its bindings and
        sets its version; every edge elsewhere in the graph that read an
        output a step renamed now reads the new name. A node already at
        ``to`` returns ``graph`` itself. A version with no chain of steps to
        ``to`` — a downgrade, or a definition a host handed over by value
        with no steps — is a ``ValueError``; a node the graph does not have
        is a ``KeyError``. The graph is not compiled here: a host compiles
        the result to show the author what the new version makes of it.
        """
        node = next((n for n in graph.nodes if n.id == node_id), None)
        if node is None:
            raise KeyError(f"{node_id!r} is not a node of this graph; it has {[n.id for n in graph.nodes]}")
        node_cls = self._nodes[node.type]
        target = node_cls.current if to is None else to
        if target == node.version:
            return graph
        steps = [node_cls.upgrades.get((n, n + 1)) for n in range(node.version, target)]
        if target < node.version or None in steps:
            raise ValueError(f"{node.type!r} has no upgrade from version {node.version} to {target}")
        values = {name: b.value if isinstance(b, Static) else b for name, b in node.bindings.items()}
        #: Each output the chain renames, from its name at the saved version to its name at ``target``.
        renamed: dict[str, str] = {}
        for step in steps:
            values = step.rewrite(dict(values))
            renamed = {old: step.outputs.get(now, now) for old, now in renamed.items()} | {
                old: new for old, new in step.outputs.items() if old not in renamed.values()
            }
        moved = node.model_copy(update={
            "version": target,
            "bindings": {name: v if isinstance(v, Edges) else Static(value=v) for name, v in values.items()},
        })
        return graph.model_copy(update={
            "nodes": [moved if n is node else _reading_renamed(n, node_id, renamed) for n in graph.nodes],
        })

    def extended_with(
        self, definitions: Mapping[str, type[NodeDefinition]]
    ) -> "NodeRegistry":
        """A new registry holding these definitions plus everything a host loaded.

        A new object rather than a mutation: the registry is process-wide
        and the loaded definitions are per run. A registered type wins over
        a loaded one of the same id, so a host cannot redefine a built-in
        node by loading something under its name. Loaded definitions are
        not held to the numbering rule — a version loaded because a graph
        pinned it may be the only one. The vocabulary carries over, plus
        whatever the loaded definitions declare.
        """
        extended = NodeRegistry()
        extended._nodes = {**definitions, **self._nodes}
        extended._types = dict(self._types)
        for definition in definitions.values():
            extended.add_types(*_declared_words(definition))
        return extended

    # -- the vocabulary ----------------------------------------------------------

    @property
    def types(self) -> Mapping[str, type[DType]]:
        """The vocabulary: every type by its id, in the order the words arrived. Read-only."""
        return MappingProxyType(self._types)

    def add_types(self, *types: type[DType]) -> None:
        """Add the host's own words: types no node here declares, or all of them, stated once.

        Each is a concrete scalar ``DType``; a ``Series[X]`` is refused,
        since the word is ``X``. The same class twice is nothing; the same
        id from another class is a collision, refused naming both.
        """
        self._types.update(self._admitted([_word_of(dtype, adding=True) for dtype in types]))

    def accepted_as(self, dtype: type[DType]) -> tuple[str, ...]:
        """The ids of every type in this vocabulary a value of ``dtype`` may land on, its own included.

        Derived from each type's ``accepts``, so a type that widens its
        welcome is served without a table kept by hand. A series is judged
        as its element is, since a node declared with the element runs once
        per row. A bug in a type's ``accepts`` surfaces here, not as a
        silent absence.
        """
        return tuple(word.id for word in self._types.values() if word.accepts(dtype))

    def describe(self) -> RegistryDescription:
        """This registry as the palette: every node's record and the vocabulary, each type once."""
        return RegistryDescription(
            nodes=tuple(cls.describe() for cls in self._nodes.values()),
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
                raise ValueError(
                    f"type id {word.id!r} is declared by both {_where(existing)} and {_where(word)}; "
                    "one registry holds one type per id"
                )
            admitted[word.id] = word
        return admitted

    # -- running -------------------------------------------------------------------

    def runner_for(self, node_id: str, version: int) -> Callable[..., Any]:
        """The callable for one registered version, for the engine to dispatch.

        ``node_id`` and ``version`` are the two facts a node stores. An
        unknown id or version is a ``KeyError``: the compiler has resolved
        every pin before the engine asks, so a miss is a bug. A
        ``GraphVersion`` is a ``TypeError``: the compiler expands it, so the
        engine never runs it as one unit. Nothing is cached, so a reloaded
        module runs its new definition.
        """
        node_cls = self.get(node_id)
        if node_cls is None:
            raise KeyError(f"no definition registered under {node_id!r}")
        declared = node_cls.versions[version]
        if not isinstance(declared, NodeVersion):
            raise TypeError(
                f"{node_id!r} version {version} declares a graph, not a run; compile "
                "expands it under the placement's name"
            )
        return self._class_runner(node_cls, declared.run)

    @staticmethod
    def _class_runner(
        node_cls: type[NodeDefinition], method: Callable[..., Any]
    ) -> Callable[..., Any]:
        """A plain callable for one version's method: a fresh instance per call.

        ``__signature__`` is the method's minus ``self``, so the engine's
        keyword filtering sees the node's parameters.
        """

        def runner(**kwargs: Any) -> Any:
            return method(node_cls(), **kwargs)

        signature = inspect.signature(method)
        runner.__signature__ = signature.replace(
            parameters=[p for name, p in signature.parameters.items() if name != "self"]
        )
        runner.__name__ = f"{node_cls.__name__}.{method.__name__}"
        return runner


def _word_of(declared: Any, *, adding: bool = False) -> type[DType] | None:
    """The vocabulary word a declared type names, or ``None`` when it names none.

    A scalar type is its own word; a parameterisation (``Table[columns]``)
    is its base's; a ``Series[X]`` is ``X``'s. ``Any`` and a closed
    input's plain Python type name no word. With ``adding`` — the host
    handing a type to ``add_types`` — a series, the base ``DType`` and a
    non-type are refused rather than silently naming nothing.
    """
    if not (isinstance(declared, type) and issubclass(declared, DType)):
        if adding:
            raise TypeError(f"{declared!r} is not a DType subclass")
        return None
    if declared is DType:
        if adding:
            raise TypeError("the base DType is not a word; add a concrete type")
        return None
    if issubclass(declared, Series):
        if adding:
            raise TypeError(f"{declared!r} is a series; its element is the word to add")
        return _word_of(declared.element)
    if declared.parameterises is not None:
        return _word_of(declared.parameterises, adding=adding)
    return declared


def _declared_words(node_cls: type[NodeDefinition]) -> list[type[DType]]:
    """Every word a node's versions declare on their inputs and outputs, each once, in declaration order."""
    words: list[type[DType]] = []
    fields: Iterable[Any] = (
        field
        for version in node_cls.versions.values()
        for field in (*version.interface.inputs, *version.interface.outputs)
    )
    for field in fields:
        word = _word_of(field.dtype)
        if word is not None and word not in words:
            words.append(word)
    return words


def _reading_renamed(node: GraphNode, source: str, renamed: Mapping[str, str]) -> GraphNode:
    """``node`` with every edge it has from ``source``'s renamed outputs pointing at the new names."""
    bindings = {
        name: Edges(refs=tuple(
            Ref(source, renamed[ref.field]) if ref.node_id == source and ref.field in renamed else ref
            for ref in binding.refs
        )) if isinstance(binding, Edges) else binding
        for name, binding in node.bindings.items()
    }
    return node if bindings == dict(node.bindings) else node.model_copy(update={"bindings": bindings})


def _where(dtype: type[DType]) -> str:
    return f"{dtype.__module__}.{dtype.__qualname__}"
