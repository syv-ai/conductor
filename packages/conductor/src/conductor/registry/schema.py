"""JSON serialization of the registry for frontend consumption.

``serialize_input`` / ``serialize_output`` are the wire primitives (plain
dicts). ``serialize_input_model`` / ``serialize_output_model`` are their typed
counterparts: they validate the dict into the :mod:`conductor.registry.serialized`
models, so hosts that want types get them without re-deriving the shape. The
dict route stays the source of truth for the wire; the model route is a typed
view guaranteed to conform (pinned by ``tests/test_core/test_serialized_schema.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conductor.registry.serialized import (
    SerializedInput,
    SerializedNode,
    SerializedOutput,
)

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry
    from conductor.registry.definition import NodeDefinition


def serialize_registry(registry: NodeRegistry) -> list[dict[str, Any]]:
    """Serialize all nodes to frontend JSON."""
    return [serialize_node(nd, registry) for nd in registry.all()]


def serialize_node(nd: NodeDefinition, registry: NodeRegistry | None = None) -> dict[str, Any]:
    """Serialize one node to the frontend catalog dict.

    ``registry`` resolves ``deprecated`` / ``latest_version`` against the other
    versions of the same ``base_id``. Omit it when the node isn't registered
    (e.g. a host serializing an ad-hoc definition): the node is then reported as
    the latest, non-deprecated.
    """
    latest = registry.get_latest(nd.base_id) if registry is not None else None
    out: dict[str, Any] = {
        "id": nd.id,
        "base_id": nd.base_id,
        "version": nd.version,
        "name": nd.name,
        "description": nd.description,
        "tags": list(nd.tags),
        "category": nd.category.value if hasattr(nd.category, "value") else nd.category,
        "inputs": [serialize_input(inp) for inp in nd.inputs],
        "outputs": [serialize_output(out) for out in nd.outputs],
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


def serialize_input(inp: Any) -> dict[str, Any]:
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


def serialize_output(out: Any) -> dict[str, Any]:
    return {
        "name": out.name,
        "type": out.type_str,
        "label": out.label,
        "description": out.description,
        "optional": out.optional,
        "download": out.download,
        "filename": out.filename,
    }


def serialize_input_model(inp: Any) -> SerializedInput:
    """Serialize one input and validate it into the typed model.

    The typed accessor for hosts that want :class:`SerializedInput` rather
    than the loose dict from :func:`serialize_input`.
    """
    return SerializedInput.model_validate(serialize_input(inp))


def serialize_output_model(out: Any) -> SerializedOutput:
    """Serialize one output and validate it into the typed model."""
    return SerializedOutput.model_validate(serialize_output(out))


def serialize_node_model(
    nd: NodeDefinition, registry: NodeRegistry | None = None
) -> SerializedNode:
    """Serialize one node and validate it into the typed model.

    The typed accessor for hosts that want :class:`SerializedNode` rather than
    the loose dict from :func:`serialize_node`. ``registry`` is optional for the
    same reason it is there.
    """
    return SerializedNode.model_validate(serialize_node(nd, registry))
