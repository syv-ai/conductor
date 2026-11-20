"""``NodeRegistry``, the nodes a host offers, by id, and the types they use.

Making a node and offering it are two steps. ``@node`` makes it, and a registry offers it. Nodes go on a registry
much like routes go on a FastAPI router::

    text = NodeRegistry(category="text", tags=["builtin"])

    @text.node("upper", title="Upper", description="Upper-cases a text.")
    def upper(value: str) -> str:
        return value.upper()

    registry = NodeRegistry()
    registry.add_subregistry(text)
    registry.add(shout)              # a plain @node from elsewhere

A registry can give its nodes a default ``category``, ``tags`` and ``extra``. A node keeps any of its own. Here
``upper`` gets the category ``"text"`` and the tag ``"builtin"``.

A registry also knows every value type its nodes use, each by its id. Types belong to one registry, not the whole
process, so two registries may each have their own ``text`` type. ``registry.describe()`` shows an editor every node
and every type.

``Graph.compile(record, registry)`` compiles a graph record against a registry. The graph keeps its own copy of the
nodes it used, so a later change to the registry doesn't reach it.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Concatenate, overload

from pydantic import JsonValue

from conductor._internal.graph import Graph
from conductor._internal.node import (
    DeclarationValidator,
    FunctionDefinition,
    NodeDefinition,
    NodeDescription,
    SubgraphDefinition,
    node,
    subgraph,
    with_defaults,
)
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.record import GraphRecord
from conductor._internal.types import Type
from conductor._internal.utils import comma_separated, plural, qualified_name

from .describe import RegistryDescription, TypeDescription
from .upgrade import upgraded

_logger = logging.getLogger("conductor")


def _takes_the_arguments_of[**P, R](
    declare: Callable[P, R],
) -> Callable[[Callable[..., Any]], Callable[Concatenate[Any, P], R]]:
    """Gives a method ``declare``'s arguments for type checkers, so ``@registry.node`` reads like ``@node``."""

    def typed(method: Callable[..., Any]) -> Callable[Concatenate[Any, P], R]:
        return method

    return typed


class NodeRegistry[C]:
    """The nodes a host offers, by id, and the types they use.

    Look a node up with ``"upper" in registry`` or ``registry["upper"]``. Looping over a registry gives the nodes in
    the order they were added. It isn't a ``Mapping``, so there is no ``get`` that would quietly answer a misspelled
    id with ``None``.

    Every node that comes in, by ``add``, ``replace``, ``@registry.node`` or a subregistry, gets the registry's
    defaults. The registry's tags come first, then the node's own. A node's own ``category`` and ``extra`` keys win.

    A registry can name the class of the run's context, the one object a host passes to each run:
    ``NodeRegistry(context_type=HostContext)``. That registry is a ``NodeRegistry[HostContext]``. Every graph
    compiled against it is a ``Graph[HostContext]``, and its runs take ``context=HostContext(...)``. A node whose
    ``Context[X]`` the class doesn't satisfy is a problem in that graph. ``NodeRegistry()`` names no class, and is a
    ``NodeRegistry[None]``.
    """

    @overload
    def __init__(
        self: NodeRegistry[None],
        *,
        category: str = "",
        tags: list[str] | tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] = MappingProxyType({}),
        types: Iterable[type] = (),
        context_type: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: NodeRegistry[C],
        *,
        category: str = "",
        tags: list[str] | tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] = MappingProxyType({}),
        types: Iterable[type] = (),
        context_type: type[C],
    ) -> None: ...

    def __init__(
        self,
        *,
        category: str = "",
        tags: list[str] | tuple[str, ...] = (),
        extra: Mapping[str, JsonValue] = MappingProxyType({}),
        types: Iterable[type] = (),
        context_type: type[C] | None = None,
    ) -> None:
        """Makes an empty registry.

        Args:
            category: The category a node gets if it has none of its own.
            tags: Tags every node gets, before its own.
            extra: ``extra`` keys every node gets, unless it sets the same key itself.
            types: Host types to offer, whether or not a node uses them.
            context_type: The class of the object the host passes as ``context=`` to every run of a graph compiled
                here. Each node's ``Context[X]`` is checked against it when a graph is compiled.

        Raises:
            NodeDeclarationError: ``invalid_declaration_argument`` if ``category``, ``tags`` or ``extra`` isn't of the
                type the signature says, as ``@node`` holds its own.
        """
        read = DeclarationValidator("NodeRegistry")(
            {"category": category, "tags": tags, "extra": extra}
        )
        self._context_type = context_type
        self._category = category
        self._tags = tuple(read["tags"])
        self._extra = dict(read["extra"])
        #: The types the host named, whether or not a node uses them.
        self._named_types = tuple(types)
        #: The nodes, by id, in the order they were added.
        self._nodes: dict[str, NodeDefinition] = {}
        #: Every type, by id: first the ones the nodes use, then the ones the host named.
        self._types: dict[str, type] = _by_id(self._named_types)
        #: Each subgraph's own graph, by id, compiled against this registry. Cleared whenever the nodes change.
        self._subgraph_graphs: dict[str, Graph[C]] = {}

    def __repr__(self) -> str:
        """Shows how many nodes there are, then each node's own repr, one per line."""
        if not self._nodes:
            return "<NodeRegistry of 0 nodes>"
        lines = "".join(f"    {definition!r}\n" for definition in self._nodes.values())
        return f"<NodeRegistry of {plural(len(self._nodes), 'node')}:\n{lines}>"

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __getitem__(self, node_id: str) -> NodeDefinition:
        """Returns the node with the id ``node_id``.

        Raises:
            KeyError: There is no such node. The message lists the ids there are.
        """
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        return self._nodes[node_id]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[NodeDefinition]:
        """Loops over the nodes, in the order they were added."""
        return iter(self._nodes.values())

    # -- the nodes ---------------------------------------------------------------

    @_takes_the_arguments_of(node)
    def node(self, *args: Any, **kwargs: Any) -> Any:
        """Makes a node, as ``@node`` does with the same arguments, and adds it here.

        An id that is already here raises, whatever ``replaces=`` names. Use ``replace`` to swap a node. Only the
        newest version goes here. Its older versions are plain ``@node`` definitions that are never added.

        The decorated name is bound to the node as this registry has it, with the registry's defaults applied.
        """
        declare = node(*args, **kwargs)

        def decorate(function: Callable[..., Any]) -> NodeDefinition:
            definition = declare(function)
            self.add(definition)
            return self._nodes[definition.id]

        return decorate

    @_takes_the_arguments_of(subgraph)
    def subgraph(self, *args: Any, **kwargs: Any) -> Any:
        """Turns a graph record into a node, as ``subgraph`` does with the same arguments, and adds it here.

        ::

            enrich = registry.subgraph("enrich", title="Enrich", description="Adds the company's details.")(record)

        An id that is already here raises, as with ``registry.node``. What it returns is the node as this registry
        has it, with the registry's defaults applied.
        """
        declare = subgraph(*args, **kwargs)

        def add(record: GraphRecord | str | bytes) -> NodeDefinition:
            definition = declare(record)
            self.add(definition)
            return self._nodes[definition.id]

        return add

    def add(self, *definitions: NodeDefinition) -> None:
        """Adds nodes, each under its id.

        Raises:
            NodeDeclarationError: ``duplicate_node_id`` if an id is already here. It names where both were declared.
        """
        self._set_contents(self._nodes_with(definitions), self._named_types)

    def replace(self, definition: NodeDefinition) -> None:
        """Puts ``definition`` in place of the node with the same id.

        Raises:
            KeyError: There is no node with that id.
        """
        if definition.id not in self._nodes:
            raise self._not_here(definition.id)
        nodes = {
            **self._nodes,
            definition.id: with_defaults(
                definition, tags=self._tags, category=self._category, extra=self._extra
            ),
        }
        self._set_contents(nodes, self._named_types)

    def remove(self, node_id: str) -> None:
        """Takes out the node with the id ``node_id``.

        Raises:
            KeyError: There is no such node.
        """
        if node_id not in self._nodes:
            raise self._not_here(node_id)
        nodes = {
            held_id: definition for held_id, definition in self._nodes.items() if held_id != node_id
        }
        self._set_contents(nodes, self._named_types)

    def add_subregistry(self, subregistry: NodeRegistry[C] | NodeRegistry[None]) -> None:
        """Adds every node of ``subregistry``, as ``add`` does, and the types it names.

        Its nodes come with its own defaults applied, and then this registry's are applied on top. They are copied
        in, so a later change to ``subregistry`` doesn't reach this one.

        ``subregistry`` must name the same ``context_type`` as this registry, or none, as a registry of pre-built
        nodes does. Its nodes are checked against this registry's class when a graph is compiled.

        Raises:
            NodeDeclarationError: ``subregistry`` names another context class, or names one while this registry
                names none. Name it here too.
        """
        if subregistry._context_type not in (None, self._context_type):
            if self._context_type is None:
                raise NodeDeclarationError(
                    declaration_problems.context_type_on_subregistry_only(
                        subregistry_type=subregistry._context_type.__name__
                    )
                )
            raise NodeDeclarationError(
                declaration_problems.context_type_differs(
                    subregistry_type=subregistry._context_type.__name__,
                    registry_type=self._context_type.__name__,
                )
            )
        named = (
            *self._named_types,
            *(dtype for dtype in subregistry._named_types if dtype not in self._named_types),
        )
        added = tuple(subregistry)
        self._set_contents(self._nodes_with(added), named)

    def filter(self, predicate: Callable[[NodeDefinition], bool]) -> NodeRegistry[C]:
        """Returns a new registry with only the nodes ``predicate`` keeps.

        It has the same defaults, named types and ``context_type`` as this one.
        """
        kept = copy.copy(self)
        nodes = {
            definition.id: definition
            for definition in self._nodes.values()
            if predicate(definition)
        }
        kept._set_contents(nodes, self._named_types)
        return kept

    def _nodes_with(self, definitions: Iterable[NodeDefinition]) -> dict[str, NodeDefinition]:
        """Returns this registry's nodes plus ``definitions``, with the registry's defaults applied.

        Raises:
            NodeDeclarationError: ``duplicate_node_id`` if an id is already here.
        """
        nodes = dict(self._nodes)
        for definition in definitions:
            held = nodes.get(definition.id)
            if held is not None:
                raise NodeDeclarationError(
                    declaration_problems.duplicate_node_id(
                        id=definition.id,
                        first=held.where_declared,
                        second=definition.where_declared,
                    ),
                    definition.id,
                    version=definition.version,
                )
            nodes[definition.id] = with_defaults(
                definition, tags=self._tags, category=self._category, extra=self._extra
            )
        return nodes

    def _set_contents(
        self, nodes: dict[str, NodeDefinition], named_types: tuple[type, ...]
    ) -> None:
        """Replaces what the registry has with ``nodes`` and ``named_types``, and works out its types again.

        The types are each node's own, in order, then the named ones.
        """
        self._types = _by_id((*_types_declared_by(nodes.values()), *named_types))
        self._nodes, self._named_types = nodes, named_types
        self._subgraph_graphs = {}

    def _not_here(self, node_id: str) -> KeyError:
        return KeyError(
            f"{node_id!r} is not registered here; the ids are {comma_separated(self._nodes)}"
        )

    # -- the types ---------------------------------------------------------------

    @property
    def context_type(self) -> type[C] | None:
        """The class of the run's context every graph compiled here takes, or ``None`` if it names none."""
        return self._context_type

    @property
    def types(self) -> Mapping[str, type]:
        """Every type by its id, first the ones the nodes use, then the ones the host named. Read-only."""
        return MappingProxyType(self._types)

    def describe(self) -> RegistryDescription:
        """Describes the registry for an editor's palette: every node, and every type.

        A subgraph whose graph doesn't compile on this registry's nodes has no fields to offer, and is left out.
        ``check_registry`` reports its problems (``subgraph_broken``), and so does compiling a graph that uses it.

        Each type lists the types that take a value of it, its own type included. That list is worked out from
        subclassing and each type's ``accepts``, so nobody keeps a table of it by hand.
        """
        types = self._types.items()
        return RegistryDescription(
            nodes=[
                described
                for definition in self._nodes.values()
                if (described := self._described(definition)) is not None
            ],
            types=[
                TypeDescription(
                    id=type_id_,
                    title=Type(held).title,
                    accepted_as=[
                        landing
                        for landing, target in types
                        if Type(target).fits(Type(held)) is not None
                    ],
                )
                for type_id_, held in types
            ],
        )

    def _described(self, definition: NodeDefinition) -> NodeDescription | None:
        """Describes one node for the palette, or returns ``None`` for a subgraph whose graph has problems.

        A subgraph offers its own graph's inputs and outputs, and asks when a node inside it asks. Those depend on
        this registry's nodes, so its graph is compiled here once, and kept until the nodes change. When that graph
        has problems, one warning on the ``conductor`` logger names the subgraph and the codes of its problems, so a
        host that never runs ``check_registry`` still hears why the subgraph is missing.
        """
        match definition:
            case FunctionDefinition():
                return definition.describe()
            case SubgraphDefinition():
                pass
        if definition.id not in self._subgraph_graphs:
            compiled = Graph.compile(definition.graph, self)
            if compiled.problems:
                _logger.warning(
                    "the subgraph %r is left out of describe(): its graph has the problems %s",
                    definition.id,
                    comma_separated(sorted({problem.type for problem in compiled.problems})),
                )
            self._subgraph_graphs[definition.id] = compiled
        graph = self._subgraph_graphs[definition.id]
        if graph.problems:
            return None
        return definition.described_as(
            inputs=list(graph.inputs.values()),
            outputs=list(graph.outputs.values()),
            asks=graph.asks,
            takes_added_inputs=False,
            computes_outputs=False,
        )

    # -- graphs ------------------------------------------------------------------

    def upgrade(
        self, record: GraphRecord | str | bytes, node_id: str, *, to: int | None = None
    ) -> GraphRecord:
        """Returns ``record`` with the node ``node_id`` moved to a newer version.

        The node moves from the version it was saved at to ``to``, or to the newest if ``to`` is ``None``. Each newer
        version's ``Upgrade`` runs in turn. A step works like this:

        - Each renamed input takes along what it got: its constant, its graph input or the edges into it.
        - The ``rewrite``, if there is one, gets the node's constants as one dict. Every key it returns becomes a
          constant. A graph input's default belongs to the whole graph, so no step changes it, and an input that
          takes a graph input or edges isn't in the dict: for a node whose inputs all take graph inputs, the rewrite
          gets ``{}``. A key it returns for such an input gives the input two kinds of entry, and the record is
          refused (``invalid_record``, whose ``ctx.written_by`` names the step).
        - An input the newer version neither has nor renames is dropped, with whatever it got. The rewrite still
          sees its constant first. If the rewrite returns its key, it stays in the record as a constant. When a
          dropped input took a graph input that no other input takes, that graph input is dropped from the
          record's ``inputs`` too, so it leaves what a caller gives the graph.
        - Every edge and graph output from a renamed output, or from a field of it, moves to the new name.

        A node already at ``to`` gives back the record as it is. The record returned isn't compiled. Compile it to
        show the author what the newer version makes of it. Compiling never upgrades on its own.

        Args:
            record: The graph record to upgrade.
            node_id: The id of the node in the record to move.
            to: The version to move it to, or ``None`` for the newest.

        Raises:
            InvalidCallError: The record has no node ``node_id``, this registry doesn't have its node, or it can't
                move to ``to``. Also ``cannot_move_to_version_rewrite_failed`` or ``cannot_move_to_version_raised`` if a rewrite raises,
                ``cannot_move_to_version_rewrite_returned_invalid`` if it returns no dict of input names,
                ``invalid_value`` if it returns a value that can't be written as JSON, and ``invalid_record`` if it
                gives a value to an input that edges lead into.
        """
        return upgraded(self._nodes, record, node_id, to=to)


def _by_id(types: Iterable[type]) -> dict[str, type]:
    """Returns ``types`` by id.

    Raises:
        NodeDeclarationError: ``duplicate_type_id`` if two classes have the same id.
    """
    by_id: dict[str, type] = {}
    for held in types:
        existing = by_id.setdefault(Type(held).id, held)
        if existing is not held:
            raise NodeDeclarationError(
                declaration_problems.duplicate_type_id(
                    id=Type(held).id, first=qualified_name(existing), second=qualified_name(held)
                )
            )
    return by_id


def _types_declared_by(definitions: Iterable[NodeDefinition]) -> list[type]:
    """Returns every class the nodes use on their inputs and outputs, each once.

    Every version of each node counts. The classes come node by node, in the order they are declared.
    """
    found: list[type] = []
    for definition in definitions:
        for step in definition.versions.values():
            match step:
                case FunctionDefinition():
                    fields = (*step.inputs.values(), *step.outputs.values())
                case SubgraphDefinition():  # a subgraph declares no classes of its own
                    continue
            found.extend(cls for field in fields for cls in field.whole.classes)
    return list(dict.fromkeys(found))
