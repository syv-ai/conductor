"""A read-only registry overlay that serves host-supplied node definitions.

The static :class:`~conductor.registry.NodeRegistry` knows only the node
types registered into it. A host often needs the compiler to type-check
edges into node types that are *not* statically registered — types resolved
per run from host state (a database row, a tenant's saved subgraph, a
plugin loaded for one request). :class:`RegistryView` layers such dynamic
definitions over a base registry without mutating it.

**It is a per-use overlay, not a global hook.** Construct one view per
compile / run, passing exactly the :class:`DefinitionSource`\\ s that context
has admitted. Two callers with different visibility never see each other's
dynamic types, because each holds its own view. This is deliberate: a
mutable global registry hook would leak one caller's types to another, which
is unsafe for hosts whose visibility rules are per-caller.

Only the point lookups — :meth:`RegistryView.get` and
:meth:`RegistryView.contains` — consult the sources. Enumeration
(``all()``, ``all_current()``) and every other attribute delegate
unchanged to the base registry: dynamic definitions are resolved on demand,
not catalog entries, so they never appear in the palette.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from conductor.node import NodeDefinition
    from conductor.registry import NodeRegistry


@runtime_checkable
class DefinitionSource(Protocol):
    """Supplies a :class:`NodeDefinition` for a node type, or ``None``.

    A source owns whatever lookup its context needs (parse the type,
    hit a cache, query a store). Returning ``None`` means "not mine" —
    the view moves on to the next source.
    """

    def get_definition(self, node_type: str) -> NodeDefinition | None: ...


class RegistryView:
    """Read-only overlay of dynamic definitions over a base registry.

    Lookups resolve base-registry types first (a statically registered
    type is never shadowed by a source), then each source in order.
    Everything else delegates to the base registry, so the view is a
    drop-in wherever a :class:`NodeRegistry` is held.
    """

    def __init__(
        self, base: NodeRegistry, sources: Sequence[DefinitionSource]
    ) -> None:
        self._base = base
        self._sources = tuple(sources)

    def _from_sources(self, node_type: str) -> NodeDefinition | None:
        for source in self._sources:
            definition = source.get_definition(node_type)
            if definition is not None:
                return definition
        return None

    def get(self, full_id: str) -> NodeDefinition | None:
        nd = self._base.get(full_id)
        if nd is not None:
            return nd
        return self._from_sources(full_id)

    def contains(self, node_id: str) -> bool:
        if self._base.contains(node_id):
            return True
        return self._from_sources(node_id) is not None

    def __getattr__(self, name: str) -> Any:
        # Everything not overlaid above defers to the base registry, so the
        # view is transparent to serialize_registry, all(), get_latest(),
        # is_deprecated(), etc.
        #
        # Guard the recursion trap: reading ``self._base`` re-enters
        # __getattr__ on an instance that never ran __init__ (constructed via
        # __new__ / copy / pickle). Raising AttributeError for dunder and
        # leading-underscore names terminates the lookup the way Python
        # expects instead of recursing forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)


