"""NodeDefinition frozen dataclass."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from conductor.metadata import InputMetadata, OutputMetadata
from conductor.types import NodeCategory, ResultFormat


@dataclass(frozen=True)
class NodeDefinition:
    """Immutable definition of a registered node."""

    id: str
    base_id: str
    version: int
    name: str
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    category: NodeCategory = NodeCategory.IO
    inputs: tuple[InputMetadata, ...] = field(default_factory=tuple)
    outputs: tuple[OutputMetadata, ...] = field(default_factory=tuple)
    result_format: ResultFormat = ResultFormat.SINGLE
    validation_model: type | None = None
    func: Callable[..., Any] | None = None
    _node_class: type | None = None
    max_retries: int = 0
    retry_delay: float = 1.0
    width: int | None = None
    docs: str | None = None
