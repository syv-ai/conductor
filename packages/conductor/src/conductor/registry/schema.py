"""JSON serialization of the registry for frontend consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry
    from conductor.registry.definition import NodeDefinition


def serialize_registry(registry: NodeRegistry) -> list[dict[str, Any]]:
    """Serialize all nodes to frontend JSON."""
    return [_serialize_node(nd, registry) for nd in registry.all()]


def _serialize_node(nd: NodeDefinition, registry: NodeRegistry) -> dict[str, Any]:
    latest = registry.get_latest(nd.base_id)
    out: dict[str, Any] = {
        "id": nd.id,
        "base_id": nd.base_id,
        "version": nd.version,
        "name": nd.name,
        "description": nd.description,
        "tags": list(nd.tags),
        "category": nd.category.value if hasattr(nd.category, "value") else nd.category,
        "inputs": [_serialize_input(inp) for inp in nd.inputs],
        "outputs": [_serialize_output(out) for out in nd.outputs],
        "width": nd.width,
        "deprecated": latest is not None and latest.version > nd.version,
        "latest_version": latest.version if latest else nd.version,
        "docs": nd.docs,
    }
    # Process-standard metadata (only emit when set to keep the payload small).
    if nd.actor is not None:
        out["actor"] = nd.actor.to_dict()
    if nd.timeout_seconds is not None:
        out["timeout_seconds"] = nd.timeout_seconds
    if nd.idempotency_key:
        out["idempotency_key"] = nd.idempotency_key
    if nd.uses:
        out["uses"] = list(nd.uses)
    if nd.is_decision:
        out["is_decision"] = True
    if nd.is_signal:
        out["is_signal"] = True
    # Surface the presence of a ``compute_outputs`` hook so frontends
    # know to call back to the host for the resolved schema rather than
    # treating the static ``outputs`` array as authoritative.
    if getattr(nd, "compute_outputs", None) is not None:
        out["has_dynamic_outputs"] = True
    return out


def _serialize_input(inp: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": inp.name,
        "type": inp.type_str,
        "label": inp.label,
        "description": inp.description,
        "widget": inp.widget.value if hasattr(inp.widget, "value") else inp.widget,
        "default": inp.default,
        "optional": inp.optional,
        "disable_handle": inp.disable_handle,
    }
    if inp.widget_config:
        data.update(inp.widget_config)
    return data


def _serialize_output(out: Any) -> dict[str, Any]:
    return {
        "name": out.name,
        "type": out.type_str,
        "label": out.label,
        "description": out.description,
        "optional": out.optional,
        "download": out.download,
        "filename": out.filename,
    }
