"""Result normalization and output extraction."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from conductor._sentinel import SKIPPED, is_skipped
from conductor.types import OUTPUT_PREFIX, RESULT_KEY, NodeResult

__all__ = [
    "normalize_result",
    "extract_output",
    "filter_skipped",
    "filter_all_skipped",
    "OutputRef",
    "project_outputs",
]


def normalize_result(raw: Any) -> NodeResult:
    """Convert raw node output to container format.

    - tuple  -> {output_1: v1, output_2: v2, ...}
    - dict   -> {result: dict, **dict}
    - other  -> {result: value}
    """
    if is_skipped(raw):
        return {RESULT_KEY: SKIPPED}

    raw = _serialize_pydantic(raw)

    if isinstance(raw, tuple):
        return {f"{OUTPUT_PREFIX}{i + 1}": v for i, v in enumerate(raw)}
    elif isinstance(raw, dict):
        return {RESULT_KEY: raw, **raw}
    else:
        return {RESULT_KEY: raw}


def extract_output(result: NodeResult, output_handle: str) -> Any | None:
    """Extract a specific output value from a node result by handle."""
    if is_skipped(result):
        return SKIPPED

    if isinstance(result, dict) and output_handle in result:
        return result[output_handle]

    # Backward compat: result -> output_1 fallback
    if (
        isinstance(result, dict)
        and output_handle == RESULT_KEY
        and f"{OUTPUT_PREFIX}1" in result
    ):
        return result[f"{OUTPUT_PREFIX}1"]

    return None


@dataclass(frozen=True)
class OutputRef:
    """A named projection of one graph output.

    The address :func:`project_outputs` reads a value from — a producing node
    and one of its output handles — plus the public ``name`` to expose that
    value under.

    Attributes:
        name: Public name to expose the projected value under.
        node_id: Id of the node that produces the value.
        handle: The node's output handle to read. ``None`` selects the node's
            sole result (``RESULT_KEY``, with the ``output_1`` back-compat
            fallback :func:`extract_output` applies).
    """

    name: str
    node_id: str
    handle: str | None = None


def project_outputs(
    results: dict[str, NodeResult],
    outputs: Iterable[OutputRef],
) -> dict[str, Any]:
    """Project a run's ``{node_id: NodeResult}`` into a flat ``{name: value}`` map.

    The graph-level counterpart to :func:`extract_output`: for each
    :class:`OutputRef`, read its ``(node_id, handle)`` value out of ``results``
    and expose it under the ref's ``name``. The single way to turn a whole run's
    per-node results into a host's declared public outputs.

    Handle-membership decides presence, so a produced ``None`` is a real value
    and survives. Omitted (never added to the output map) are: an absent node,
    an absent handle, and a ``SKIPPED`` value — whether the whole node was
    skipped (its result is the bare sentinel) or just this handle — so the
    non-serializable sentinel never reaches a projected output.
    """
    projected: dict[str, Any] = {}
    for ref in outputs:
        node_result = results.get(ref.node_id)
        if not isinstance(node_result, dict):
            # Absent node, or the bare SKIPPED sentinel for a skipped node.
            continue
        handle = ref.handle if ref.handle is not None else RESULT_KEY
        if handle in node_result:
            value = node_result[handle]
        elif handle == RESULT_KEY and f"{OUTPUT_PREFIX}1" in node_result:
            value = node_result[f"{OUTPUT_PREFIX}1"]
        else:
            continue
        if is_skipped(value):
            continue
        projected[ref.name] = value
    return projected


def filter_skipped(result: NodeResult) -> NodeResult:
    """Remove SKIPPED values from a result dict."""
    return {k: v for k, v in result.items() if not is_skipped(v)}


def filter_all_skipped(results: dict[str, NodeResult]) -> dict[str, Any]:
    """Filter SKIPPED from all node results."""
    return {
        nid: filter_skipped(res)
        for nid, res in results.items()
        if not is_skipped(res)
    }


def _serialize_pydantic(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, tuple):
        return tuple(_serialize_pydantic(v) for v in value)
    if isinstance(value, list):
        return [_serialize_pydantic(v) for v in value]
    return value
