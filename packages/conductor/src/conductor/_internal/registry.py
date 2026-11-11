"""``NodeRegistry`` — the nodes a host offers, by id, and the types they use.

Defining a node (``@node``) and offering it are two acts; a registry is the
second. Nodes go on one as FastAPI's routes go on a router::

    text = NodeRegistry(category="text", tags=["builtin"])

    @text.node("upper", title="Upper", description="Upper-cases a text.")
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
under its id: conductor's own types from a fixed table, a host's class by
the ``type_id`` it declares. Nothing about a type is process-wide, so two
registries in one process may each hold a ``text`` of their own; one id
claimed by two classes in one registry raises where the second arrives,
naming both. ``describe()`` is what an editor reads: every node and every
type, once, as a ``RegistryDescription``, each type with the ids of every
type here that admits it (``accepted_as``).

``registry.load(record)`` reads a stored graph into a ``Graph`` that keeps
its own copy of the definitions it resolved, so a later change to the
registry doesn't reach it. ``registry.upgrade(record, node_id)`` moves one
node of a record to a newer version.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Concatenate

from pydantic import JsonValue

from conductor._internal import problem
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.fields import NamedType, Type
from conductor._internal.graph import Graph
from conductor._internal.graph_record import GraphRecord, NodeRecord, as_record, typed_in
from conductor._internal.loaded_graph import load_graph
from conductor._internal.node_definition import (
    FunctionDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeDescription,
    StoredGraphDescription,
    declared_inputs,
    node,
)
from conductor._internal.problem import InvalidCallError, NodeDeclarationError


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
        tags: list[str] | tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] = MappingProxyType({}),
        types: Iterable[type[NamedType]] = (),
    ) -> None:
        """An empty registry whose nodes default to ``category``, ``tags`` and ``extra``, holding the host's types in
        ``types`` whether or not a node uses them."""
        self._category = category
        self._tags = tuple(tags)
        self._extra = dict(extra)
        #: The types a host names, whether or not a node declares them.
        self._named_types = tuple(types)
        #: The definitions, by id, in the order they arrived.
        self._nodes: dict[str, NodeDefinition] = {}
        #: Every type, by id: the ones the nodes declare, then the named ones.
        self._types: dict[str, type] = _by_id(self._named_types)
        #: Each stored graph's description, by id, as these nodes make it; emptied whenever what is held changes.
        self._graph_descriptions: dict[str, StoredGraphDescription] = {}

    def __repr__(self) -> str:
        """How many nodes it holds, then each as its own repr, one per line."""
        if not self._nodes:
            return "<NodeRegistry of 0 nodes>"
        lines = "".join(f"    {definition!r}\n" for definition in self._nodes.values())
        return f"<NodeRegistry of {len(self._nodes)} node{'s' if len(self._nodes) > 1 else ''}:\n{lines}>"

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> NodeDefinition:
        """The definition registered under ``node_id``; an unknown id is a ``KeyError`` naming the ids there are."""
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeDefinition]:
        """The definitions, in the order they arrived."""
        return iter(self._nodes.values())

    # -- what it holds -----------------------------------------------------------

    @_takes_the_arguments_of(node)
    def node(self, *args: Any, **kwargs: Any) -> Any:
        """Declare a node as ``@node`` does, with the same arguments, and add it here.

        An id already held raises, whatever ``replaces=`` names; ``replace``
        swaps a held node. The older versions of a chain are plain ``@node``
        definitions never registered, and only the newest is declared here.
        The function's name is bound to the definition this registry holds,
        with the registry's defaults applied.
        """
        declare = node(*args, **kwargs)

        def decorate(target: Callable[..., Any] | GraphRecord | str | bytes | Mapping[str, Any]) -> NodeDefinition:
            definition = declare(target)
            self.add(definition)
            return self._nodes[definition.id]

        return decorate

    def add(self, *definitions: NodeDefinition) -> None:
        """Add these nodes, each under its id; an id already here raises, naming where both were declared."""
        self._set_contents(self._nodes_with(definitions), self._named_types)

    def replace(self, definition: NodeDefinition) -> None:
        """Put ``definition`` in place of the node registered under its id; an id not here is a ``KeyError``."""
        if definition.id not in self._nodes:
            raise self._not_here(definition.id)
        nodes = {**self._nodes, definition.id: self._with_defaults(definition)}
        self._set_contents(nodes, self._named_types)

    def remove(self, node_id: str) -> None:
        """Take out the node registered under ``node_id``; an id not here is a ``KeyError``."""
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        nodes = {held_id: definition for held_id, definition in self._nodes.items() if held_id != node_id}
        self._set_contents(nodes, self._named_types)

    def add_subregistry(self, subregistry: NodeRegistry) -> None:
        """Add every node of ``subregistry`` as ``add`` does, and the types it names.

        Its nodes arrive with its defaults applied, and this registry's go
        over them. They are copied in, so a later change to ``subregistry``
        does not reach this one.
        """
        named = (*self._named_types, *(dtype for dtype in subregistry._named_types if dtype not in self._named_types))
        added = tuple(subregistry)
        self._set_contents(self._nodes_with(added), named)

    def filter(self, predicate: Callable[[NodeDefinition], bool]) -> NodeRegistry:
        """A new registry with this one's defaults and named types, holding the nodes ``predicate`` keeps."""
        kept = copy.copy(self)
        nodes = {definition.id: definition for definition in self._nodes.values() if predicate(definition)}
        kept._set_contents(nodes, self._named_types)
        return kept

    def _nodes_with(self, definitions: Iterable[NodeDefinition]) -> dict[str, NodeDefinition]:
        """What this registry holds plus ``definitions``, each under this registry's defaults; a repeated id raises."""
        nodes = dict(self._nodes)
        for definition in definitions:
            held = nodes.get(definition.id)
            if held is not None:
                raise NodeDeclarationError(
                    f"two nodes have the id {definition.id!r}: {_where_declared(held)} and {_where_declared(definition)}; "
                    "ids are unique, so swap one for the other with registry.replace(...)",
                    code="duplicate_node_id",
                )
            nodes[definition.id] = self._with_defaults(definition)
        return nodes

    def _with_defaults(self, definition: NodeDefinition) -> NodeDefinition:
        """``definition`` and each version it replaces under this registry's defaults: tags after the registry's, the node's own category and extra keys winning."""
        tags = tuple(dict.fromkeys((*self._tags, *definition.tags)))
        category = definition.category or self._category
        extra = {**self._extra, **definition.extra}
        replaces = None if definition.replaces is None else self._with_defaults(definition.replaces)
        if (tags, category, extra, replaces) == (definition.tags, definition.category, dict(definition.extra), definition.replaces):
            return definition
        return dataclasses.replace(definition, tags=tags, category=category, extra=MappingProxyType(extra), replaces=replaces)

    def _set_contents(self, nodes: dict[str, NodeDefinition], named_types: tuple[type, ...]) -> None:
        """Hold exactly ``nodes`` and ``named_types``, and the types they give: each node's declared ones in order, then the named ones."""
        self._types = _by_id((*_types_declared_by(nodes.values()), *named_types))
        self._nodes, self._named_types = nodes, named_types
        self._graph_descriptions = {}

    def _not_here(self, node_id: str) -> KeyError:
        return KeyError(f"{node_id!r} is not registered here; the ids are {', '.join(self._nodes) or 'none'}")

    # -- the types ---------------------------------------------------------------

    @property
    def types(self) -> Mapping[str, type]:
        """Every type by its id: the ones the nodes declare, then the ones a host named. Read-only."""
        return MappingProxyType(self._types)

    def describe(self) -> RegistryDescription:
        """This registry as an editor reads it: every node, and every type with where a value of it may land.

        Where a value lands, its own type included, is worked out from subclassing and each type's ``accepts``, so no
        table of it is kept by hand.
        """
        types = self._types.items()
        return RegistryDescription(
            nodes=[self._described(definition) for definition in self._nodes.values()],
            types=[
                TypeDescription(id=type_id_, title=Type(held).title, accepted_as=[landing for landing, target in types if Type(target).fits(Type(held)) is not None])
                for type_id_, held in types
            ],
        )

    def _described(self, definition: NodeDefinition) -> NodeDescription:
        """A node as the palette shows it; a stored graph with the fields it offers, worked out against this registry once."""
        match definition:
            case FunctionDefinition():
                return definition.describe()
            case GraphDefinition():
                pass
        if definition.id not in self._graph_descriptions:
            placed = self.load(GraphRecord(nodes=[NodeRecord(id="node", node=definition.id, version=definition.version)])).describe().nodes["node"]
            self._graph_descriptions[definition.id] = definition.described_as(
                StoredGraphDescription, inputs=[placed_input.field for placed_input in placed.inputs], outputs=placed.outputs,
            )
        return self._graph_descriptions[definition.id]

    # -- graphs ------------------------------------------------------------------

    def load(self, record: GraphRecord | str | bytes | Mapping[str, Any]) -> Graph:
        """Load a stored graph, given as a record, JSON text or a mapping read from JSON.

        It never raises for a half-finished drawing: what is wrong is in
        ``graph.problems``. The graph keeps its own copy of the definitions
        as they are when it loads.
        """
        return load_graph(as_record(record), self._nodes, Graph)

    def upgrade(self, record: GraphRecord | str | bytes | Mapping[str, Any], node_id: str, *, to: int | None = None) -> GraphRecord:
        """``record`` with the node ``node_id`` moved from the version it was saved at to ``to``, the newest by default.

        Runs each newer version's ``Upgrade`` in order and sets the version.
        Each step first moves what the record holds on the inputs it renames
        (values, defaults and the edges into them), then rewrites
        what is typed into the node, values and defaults as one map: a key
        that was a default stays one, and every other key the rewrite returns
        is a value, fixed as the author's. An edge into an input the step
        neither keeps nor renames goes, as what was typed into it goes unless
        the rewrite returns it. Every edge out of a renamed output,
        or out of a column of it, reads the new name. A node already at ``to``
        returns the record as it is. The record returned is not loaded here:
        load it to show the author what the newer version makes of it.

        What a rewrite returns is written as JSON, as ``make_graph`` writes a
        value typed in. A node the record does not have, a node this registry
        does not, a version it can't move to and a rewrite returning a value
        with no JSON form (``invalid_value``) are each an ``InvalidCallError``.
        """
        graph = as_record(record)
        placed = next((n for n in graph.nodes if n.id == node_id), None)
        if placed is None:
            raise InvalidCallError([problem.node_not_in_graph(node_id, nodes=[n.id for n in graph.nodes])])
        if placed.node not in self._nodes:
            raise InvalidCallError([problem.unknown_node(node_id, node=placed.node, choices=sorted(self._nodes))])
        newest = self._nodes[placed.node]
        target = newest.version if to is None else to
        if target == placed.version:
            return graph
        if target < placed.version or target not in newest.versions or placed.version not in newest.versions:
            raise InvalidCallError([problem.cannot_move_to_version(
                node_id,
                node=placed.node, version=placed.version, target=target, versions=sorted(newest.versions),
            )])
        lineage = newest.version_of(target).lineage
        steps = [(newer, older) for newer, older in zip(lineage, lineage[1:], strict=False) if newer.version > placed.version]
        values, defaults, edges = dict(placed.values), dict(placed.defaults), list(graph.edges)
        for step, older in reversed(steps):
            upgrade = step.upgrade
            values, defaults = _with_keys_renamed(values, upgrade.renamed_inputs), _with_keys_renamed(defaults, upgrade.renamed_inputs)
            if upgrade.rewrite is not None:
                rewritten, faults = typed_in(node_id, upgrade.rewrite({**defaults, **values}))
                if faults:
                    raise InvalidCallError(faults)
                defaults = {name: value for name, value in rewritten.items() if name in defaults and name not in values}
                values = {name: value for name, value in rewritten.items() if name not in defaults}
            gone = {f"{node_id}.{name}" for name in declared_inputs(older).keys() - declared_inputs(step).keys() - upgrade.renamed_inputs.keys()}
            edges = [
                (_renamed_on(source, node_id, upgrade.renamed_outputs), _renamed_on(target_, node_id, upgrade.renamed_inputs))
                for source, target_ in edges
                if target_ not in gone
            ]
        moved = placed.model_copy(update={"version": target, "values": values, "defaults": defaults})
        return GraphRecord(nodes=[moved if n is placed else n for n in graph.nodes], edges=edges)


class TypeDescription(ConductorModel):
    """A type as an editor reads it: its ``id``, its ``title``, and the ids of every type a value of it may land on."""

    id: str
    title: str
    accepted_as: list[str]


class RegistryDescription(ConductorModel):
    """A registry as an editor's palette reads it: every node, and every type."""

    nodes: list[NodeDescription]
    types: list[TypeDescription]


def _with_keys_renamed[V](by_name: Mapping[str, V], renames: Mapping[str, str]) -> dict[str, V]:
    """``by_name`` with each key a step renames under its new name."""
    return {renames.get(name, name): value for name, value in by_name.items()}


def _renamed_on(address: str, node_id: str, renames: Mapping[str, str]) -> str:
    """``address`` with its field renamed when it is on ``node_id`` and the step renames it.

    A field of a record named after it (``result.email``) moves with it. An embedded graph's output is named by an
    inner address (``tail.result``), so the longest renamed name the address goes on with is the one it names.
    """
    owner, _, rest = address.partition(".")
    if owner != node_id:
        return address
    for name in sorted(renames, key=len, reverse=True):
        if rest == name or rest.startswith(name + "."):
            return f"{owner}.{renames[name]}{rest[len(name):]}"
    return address


def _by_id(types: Iterable[type]) -> dict[str, type]:
    """``types`` by id; raises when one id names two classes."""
    by_id: dict[str, type] = {}
    for held in types:
        existing = by_id.setdefault(Type(held).id, held)
        if existing is not held:
            raise NodeDeclarationError(
                f"type id {Type(held).id!r} is declared by both {_qualified_name(existing)} and {_qualified_name(held)}; "
                "one registry holds one type per id",
                code="duplicate_type_id",
            )
    return by_id


def _where_declared(definition: NodeDefinition) -> str:
    """Where a node was written: its function's module and name, or the graph it is."""
    match definition:
        case FunctionDefinition(function=function):
            return f"{function.__module__}.{function.__qualname__}"
        case GraphDefinition():
            return f"a node that is a graph, titled {definition.title!r} (versions {', '.join(map(str, definition.versions))})"


def _types_declared_by(definitions: Iterable[NodeDefinition]) -> list[type]:
    """Every class the versions of ``definitions`` declare on their inputs and outputs, each once, node by node in declaration order."""
    found: list[type] = []
    for definition in definitions:
        for step in definition.lineage:
            match step:
                case FunctionDefinition():
                    specs = (*step.inputs.values(), *step.outputs.values())
                case GraphDefinition():  # a graph declares no classes of its own
                    continue
            found.extend(cls for spec in specs for cls in spec.whole.classes)
    return list(dict.fromkeys(found))


def _qualified_name(held: type) -> str:
    return f"{held.__module__}.{held.__qualname__}"
