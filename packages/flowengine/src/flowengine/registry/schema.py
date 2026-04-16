"""JSON serialization of the registry for frontend consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flowengine.registry import NodeRegistry
    from flowengine.registry.definition import NodeDefinition


def serialize_registry(registry: NodeRegistry) -> list[dict[str, Any]]:
    """Serialize all nodes to frontend JSON."""
    return [_serialize_node(nd, registry) for nd in registry.all()]


def _serialize_node(nd: NodeDefinition, registry: NodeRegistry) -> dict[str, Any]:
    latest = registry.get_latest(nd.base_id)
    return {
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
