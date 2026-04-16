"""Result normalization and output extraction."""

from typing import Any

from pydantic import BaseModel

from flowengine._sentinel import SKIPPED, is_skipped
from flowengine.types import OUTPUT_PREFIX, RESULT_KEY, NodeResult


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
