"""Node registry — registration, lookup, versioning."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from conductor.metadata import InputMetadata, OutputMetadata
from conductor.registry.definition import NodeDefinition
from conductor.types import (
    RESULT_KEY,
    OUTPUT_PREFIX,
    NodeCategory,
    ResultFormat,
    WidgetType,
)
from conductor.validation import _extract_type_string, _is_injectable, create_validation_model
from conductor.widgets import Output, Widget


class NodeRegistry:
    """Versioned node registry. Nodes identified as base_id@version."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeDefinition] = {}
        self._by_base_id: dict[str, list[NodeDefinition]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def node(
        self,
        base_id: str,
        *,
        version: int = 1,
        name: str,
        description: str,
        tags: list[str] | None = None,
        category: NodeCategory = NodeCategory.IO,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        width: int | None = None,
        docs: str | None = None,
    ) -> Callable:
        """Decorator to register a function as a node."""

        def decorator(func: Callable) -> Callable:
            full_id = f"{base_id}@{version}"
            if full_id in self._nodes:
                raise ValueError(f"Node '{full_id}' is already registered")

            inputs, outputs, result_format = _introspect_function(func)
            validation_model = create_validation_model(func)

            node_def = NodeDefinition(
                id=full_id,
                base_id=base_id,
                version=version,
                name=name,
                description=description,
                tags=tuple(tags or []),
                category=category,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                result_format=result_format,
                validation_model=validation_model,
                func=func,
                max_retries=max_retries,
                retry_delay=retry_delay,
                width=width,
                docs=docs,
            )

            self._nodes[full_id] = node_def
            self._by_base_id.setdefault(base_id, []).append(node_def)

            return func

        return decorator

    def register_class(self, node_cls: type, *, version: int | None = None) -> None:
        """Register a class-based node (BaseNode subclass)."""
        from conductor.node import BaseNode

        if not (isinstance(node_cls, type) and issubclass(node_cls, BaseNode)):
            raise TypeError(f"{node_cls} must be a BaseNode subclass")

        base_id = node_cls.node_id
        ver = version or getattr(node_cls, "node_version", 1)
        full_id = f"{base_id}@{ver}"
        if full_id in self._nodes:
            raise ValueError(f"Node '{full_id}' is already registered")

        category = getattr(node_cls, "node_category", NodeCategory.IO)
        tags = getattr(node_cls, "node_tags", ())

        node_def = NodeDefinition(
            id=full_id,
            base_id=base_id,
            version=ver,
            name=node_cls.node_name,
            description=node_cls.node_description,
            tags=tuple(tags),
            category=category,
            inputs=(),
            outputs=(),
            result_format=ResultFormat.SINGLE,
            validation_model=None,
            func=None,
            _node_class=node_cls,
        )

        self._nodes[full_id] = node_def
        self._by_base_id.setdefault(base_id, []).append(node_def)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, full_id: str) -> NodeDefinition | None:
        return self._nodes.get(full_id)

    def get_latest(self, base_id: str) -> NodeDefinition | None:
        versions = self._by_base_id.get(base_id)
        if not versions:
            return None
        return max(versions, key=lambda nd: nd.version)

    def is_deprecated(self, full_id: str) -> bool:
        nd = self._nodes.get(full_id)
        if nd is None:
            return False
        latest = self.get_latest(nd.base_id)
        return latest is not None and latest.version > nd.version

    def all(self) -> list[NodeDefinition]:
        return list(self._nodes.values())

    def all_current(self) -> list[NodeDefinition]:
        result = []
        for base_id in self._by_base_id:
            latest = self.get_latest(base_id)
            if latest:
                result.append(latest)
        return result

    def contains(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, package_name: str) -> int:
        """Auto-discover and register nodes from a Python package."""
        from conductor.registry.discovery import discover_nodes

        return discover_nodes(package_name, self)


# ------------------------------------------------------------------
# Introspection helpers
# ------------------------------------------------------------------


def _introspect_function(
    func: Callable,
) -> tuple[list[InputMetadata], list[OutputMetadata], ResultFormat]:
    """Extract input/output metadata from a function signature."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    inputs = _extract_inputs(sig, type_hints)
    outputs, result_format = _extract_outputs(type_hints)
    return inputs, outputs, result_format


def _extract_inputs(
    sig: inspect.Signature,
    type_hints: dict[str, Any],
) -> list[InputMetadata]:
    inputs: list[InputMetadata] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = type_hints.get(param_name, param.annotation)

        # Skip injectable types (FlowStore, etc.)
        if _is_injectable(annotation):
            continue

        has_default = param.default != inspect.Parameter.empty
        default = param.default if has_default else None

        widget_instance: Widget | None = None
        base_type = annotation

        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            base_type = args[0]
            for arg in args[1:]:
                if isinstance(arg, Widget):
                    widget_instance = arg
                    break

        type_str = _extract_type_string(base_type)
        expects_list = type_str.startswith("list[")

        if widget_instance:
            wt = widget_instance.widget_type
            inputs.append(InputMetadata(
                name=param_name,
                type_str=type_str,
                label=widget_instance.label,
                description=widget_instance.description,
                widget=wt,
                default=default,
                optional=has_default,
                expects_list=expects_list,
                uses_connection_list=(wt == WidgetType.CONNECTION_LIST),
                disable_handle=widget_instance.disable_handle,
                widget_config={
                    k: v
                    for k, v in widget_instance.to_schema().items()
                    if k not in ("widget", "label", "description", "disable_handle")
                },
            ))
        else:
            inputs.append(InputMetadata(
                name=param_name,
                type_str=type_str,
                label=param_name,
                default=default,
                optional=has_default,
                expects_list=expects_list,
            ))

    return inputs


def _extract_outputs(
    type_hints: dict[str, Any],
) -> tuple[list[OutputMetadata], ResultFormat]:
    return_hint = type_hints.get("return", inspect.Parameter.empty)

    if return_hint is inspect.Parameter.empty or return_hint is type(None):
        return [OutputMetadata(name=RESULT_KEY, type_str="none", label="Output")], ResultFormat.SINGLE

    origin = get_origin(return_hint)

    # Multi-output: tuple[Annotated[T, Output(...)], ...]
    if origin is tuple:
        args = get_args(return_hint)
        outputs: list[OutputMetadata] = []
        for i, arg in enumerate(args):
            name = f"{OUTPUT_PREFIX}{i + 1}"
            if get_origin(arg) is Annotated:
                inner_args = get_args(arg)
                base_type = inner_args[0]
                out_widget = None
                for a in inner_args[1:]:
                    if isinstance(a, Output):
                        out_widget = a
                        break
                outputs.append(OutputMetadata(
                    name=name,
                    type_str=_extract_type_string(base_type),
                    label=out_widget.label if out_widget else name,
                    description=out_widget.description if out_widget else None,
                    download=out_widget.download if out_widget else False,
                    filename=out_widget.filename if out_widget else None,
                ))
            else:
                outputs.append(OutputMetadata(
                    name=name,
                    type_str=_extract_type_string(arg),
                    label=name,
                ))
        return outputs, ResultFormat.MULTI

    # Single output: Annotated[T, Output(...)]
    if get_origin(return_hint) is Annotated:
        args = get_args(return_hint)
        base_type = args[0]
        out_widget = None
        for a in args[1:]:
            if isinstance(a, Output):
                out_widget = a
                break
        return [OutputMetadata(
            name=RESULT_KEY,
            type_str=_extract_type_string(base_type),
            label=out_widget.label if out_widget else "Output",
            description=out_widget.description if out_widget else None,
            download=out_widget.download if out_widget else False,
            filename=out_widget.filename if out_widget else None,
        )], ResultFormat.SINGLE

    # Plain type
    return [OutputMetadata(
        name=RESULT_KEY,
        type_str=_extract_type_string(return_hint),
        label="Output",
    )], ResultFormat.SINGLE
