"""Types for the ``compute_outputs`` hook.

A node may register a ``compute_outputs`` callable to declare its output
shape based on its instance ``data`` and the resolved outputs of upstream
producers. The callable is invoked at compile time, in topological order,
so each consumer sees its predecessors' resolved outputs.

The runtime contract is unchanged: a node still returns a ``dict`` keyed by
handle name and ``extract_output`` reads keys verbatim. ``compute_outputs``
only affects compile-time shape (palette serialization, type-checking,
shared-reference handle validation, compound runtime handle ordering).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from conductor.metadata import OutputMetadata


@dataclass(frozen=True)
class IncomingBinding:
    """A single resolved upstream connection feeding a node input.

    Attributes:
        target_handle: The input handle on this node receiving the value.
        source_node_id: The producer node's id.
        source_handle: The producer's output handle name.
        source_output: The producer's resolved (post-hook) output metadata.
            If the producer also has a ``compute_outputs`` hook, this is
            the post-hook value, not the static schema.
    """

    target_handle: str
    source_node_id: str
    source_handle: str
    source_output: OutputMetadata


@dataclass(frozen=True)
class ComputeOutputsContext:
    """Context passed to a ``compute_outputs`` hook.

    Attributes:
        data: The node instance's ``data`` payload (widget values, host
            metadata) — same dict the engine passes to ``execute``.
        incoming: All resolved incoming bindings on this node, in stable
            order.
        node_id: The node instance id (useful for error messages).
        defaults: The static outputs declared on the registered
            ``NodeDefinition``. Hooks may return these unchanged when no
            dynamic shape applies.
    """

    data: dict[str, Any]
    incoming: tuple[IncomingBinding, ...]
    node_id: str
    defaults: tuple[OutputMetadata, ...]


ComputeOutputsFn = Callable[[ComputeOutputsContext], list[OutputMetadata]]
"""Signature of a compute_outputs hook.

Must return a ``list[OutputMetadata]``. The resolver validates uniqueness of
output names and (when ``dynamic_handles=False`` on the node definition)
that all statically declared output names are still present.
"""
