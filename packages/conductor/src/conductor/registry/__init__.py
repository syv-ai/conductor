"""``NodeRegistry`` — the node classes a host offers, by id.

Defining a node (subclassing ``NodeDefinition``) and offering it are two
acts. A registry is the second: ``register(cls)`` files the class under
its ``id`` and checks the rules that only make sense for a catalogue —
versions numbered from 1 with no holes, a deprecated current version
pointing somewhere, an alternative that exists. ``runner_for`` gives the
engine a plain callable for one registered version.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from conductor.node import NodeDefinition, NodeVersion, upgrade_methods


class NodeRegistry:
    """Maps a node id to the node class itself.

    One entry per id, not per (id, version): the class knows which versions
    it declares, and a caller picks one with
    ``registry.get(node.type).versions[node.version]``.
    """

    def __init__(self) -> None:
        #: The classes, by id, in registration order.
        self._nodes: dict[str, type[NodeDefinition]] = {}
        self._upgrades: dict[str, dict[tuple[int, int], Callable[..., Any]]] = {}

    def register(self, node_cls: type[NodeDefinition]) -> None:
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
        self._nodes[node_cls.id] = node_cls
        self._upgrades[node_cls.id] = upgrade_methods(node_cls)

    def get(self, node_id: str) -> type[NodeDefinition] | None:
        """The class registered under ``node_id``, or ``None``."""
        return self._nodes.get(node_id)

    def contains(self, node_id: str) -> bool:
        return node_id in self._nodes

    def definitions(self) -> tuple[type[NodeDefinition], ...]:
        """Every registered class, in registration order.

        A palette is ``[d.describe() for d in registry.definitions()]``.
        """
        return tuple(self._nodes.values())

    def upgrade_path(
        self, node_id: str, from_version: int, to_version: int
    ) -> Callable[..., Any] | None:
        """The ``@upgrade`` function for one version step, or ``None``."""
        return self._upgrades.get(node_id, {}).get((from_version, to_version))


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


def runner_for(
    registry: "NodeRegistry", node_id: str, version: int
) -> Callable[..., Any]:
    """The callable for one registered version, for the engine to dispatch.

    ``node_id`` and ``version`` are the two facts a placement stores. An
    unknown id or version is a ``KeyError``: the compiler has resolved
    every pin before the engine asks, so a miss is a bug. A
    ``GraphVersion`` is a ``TypeError``: the compiler expands it, so the
    engine never runs it as one unit. Nothing is cached, so a reloaded
    module runs its new definition.
    """
    node_cls = registry.get(node_id)
    if node_cls is None:
        raise KeyError(f"no definition registered under {node_id!r}")
    declared = node_cls.versions[version]
    if not isinstance(declared, NodeVersion):
        raise TypeError(
            f"{node_id!r} version {version} declares a graph, not a run; compile "
            "expands it under the placement's name"
        )
    return _class_runner(node_cls, declared.run)
