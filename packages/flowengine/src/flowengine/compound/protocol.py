"""Compound node protocols and Region dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from flowengine.graph.model import GraphEdge, GraphNode


@dataclass(frozen=True)
class Region:
    """A group of related nodes managed by a compound node."""

    start_id: str
    end_id: str
    body_ids: frozenset[str]

    @property
    def all_ids(self) -> frozenset[str]:
        return self.body_ids | {self.start_id, self.end_id}


class NodeExecutor(Protocol):
    """The contract every executable unit implements."""

    def execute(self, req: Any) -> Any: ...


@dataclass(frozen=True)
class CompoundNodeType:
    """Registration for a compound node type.

    The compiler calls discover() to find regions, then factory()
    to create an executor for each region.
    """

    start_type_prefix: str
    end_type_prefix: str
    discover: Callable[
        [list[GraphNode], list[GraphEdge]],
        list[Region],
    ]
    factory: Callable[
        [Region, tuple[str, ...]],
        NodeExecutor,
    ]
