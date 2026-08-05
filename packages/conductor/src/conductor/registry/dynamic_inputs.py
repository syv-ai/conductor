"""Types for the ``compute_inputs`` hook.

A node may register a ``compute_inputs`` callable to declare its input
handles from its instance ``data``. The callable runs at compile time,
before shared-reference and edge validation, so those checks see the
handles the node actually exposes rather than only its static signature.

Unlike ``compute_outputs`` this hook receives no upstream bindings and runs
in no particular order. An incoming binding is keyed by its
``target_handle``, and the target handles are precisely what this hook
decides — consulting them would be circular. Input resolution therefore
depends on the node's own ``data`` alone.

The runtime contract is unchanged. A node whose hook invents handles
declares ``**kwargs`` and receives them verbatim (``_filter_to_signature``
does not filter a ``VAR_KEYWORD`` callable), and ``dynamic_handles=True``
already gives it a validation model that tolerates the extra keys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from conductor.metadata import InputMetadata

__all__ = [
    "ComputeInputsContext",
    "ComputeInputsFn",
]


@dataclass(frozen=True)
class ComputeInputsContext:
    """Context passed to a ``compute_inputs`` hook.

    Attributes:
        data: The node instance's ``data`` payload (widget values, host
            metadata) — the same dict the engine passes to ``execute``.
        node_id: The node instance id (useful for error messages).
        defaults: The static inputs declared on the registered
            ``NodeDefinition``. Hooks may return these unchanged when no
            dynamic shape applies.
        validated_data: The node's ``data`` run through the registered
            ``validation_model`` and re-serialized via ``model_dump()``.
            ``None`` when the node has no model or when validation fails —
            the latter is expected during in-progress editing, where
            ``data`` may be incomplete. Hooks must handle ``None`` and fall
            back to ``data`` if they need a value.
    """

    data: dict[str, Any]
    node_id: str
    defaults: tuple[InputMetadata, ...]
    validated_data: dict[str, Any] | None = None


ComputeInputsFn = Callable[[ComputeInputsContext], list[InputMetadata]]
"""Signature of a compute_inputs hook.

Must return a ``list[InputMetadata]``. The resolver validates uniqueness of
input names and (when ``dynamic_handles=False`` on the node definition)
that all statically declared input names are still present.
"""
