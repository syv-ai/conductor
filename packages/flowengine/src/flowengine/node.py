"""BaseNode ABC for class-based nodes."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from flowengine.types import NodeCategory


class BaseNode(ABC):
    """Abstract base class for class-based nodes.

    Use for complex nodes that need setup/teardown, internal state,
    or custom dispatch. For simple data transformations, prefer
    the @registry.node() decorator.
    """

    node_id: ClassVar[str]
    node_name: ClassVar[str]
    node_description: ClassVar[str]
    node_version: ClassVar[int] = 1
    node_tags: ClassVar[tuple[str, ...]] = ()
    node_category: ClassVar[NodeCategory] = NodeCategory.IO

    @abstractmethod
    def execute(self, req: Any) -> Any:
        """Execute the node. Receives a NodeExecRequest."""
        ...
