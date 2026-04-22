"""Node registry — registration, lookup, versioning."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from conductor.metadata import InputMetadata, OutputMetadata
from conductor.registry.definition import NodeDefinition
from conductor.types import (
    OUTPUT_PREFIX,
    RESULT_KEY,
    NodeCategory,
    ResultFormat,
    WidgetType,
)
from conductor.validation import _extract_type_string, _is_injectable, create_validation_model
from conductor.widgets import (
    Checkbox,
    DatePicker,
    FileUpload,
    List,
    Number,
    Output,
    SchemaBuilder,
    Text,
    Widget,
)


def _default_widget(base_type: Any, param_name: str) -> Widget | None:
    """Return a default widget for a given Python type, or ``None`` if the
    type is something we don't have a sensible default for.

    Rules:
        str           -> Text
        int / float   -> Number (with integer_only for int)
        bool          -> Checkbox
        Date (alias)  -> DatePicker
        Base64Str, NamedFile, MultiNamedFile (aliases) -> FileUpload
        list[T]       -> List (item widget inferred from T)
        dict, dict[str, Any] -> SchemaBuilder
        Any other     -> None (caller falls back to no widget)

    Explicit ``Annotated[T, Widget(...)]`` on a parameter always wins — this
    function is only consulted when no ``Widget`` instance was found.
    """
    origin = get_origin(base_type)

    # list[T]
    if origin is list or base_type is list:
        inner_args = get_args(base_type)
        inner_widget = _default_widget(inner_args[0], param_name) if inner_args else None
        return List(label=param_name, item_widget=inner_widget)

    # dict[...]
    if origin is dict or base_type is dict:
        return SchemaBuilder(label=param_name)

    if base_type is str:
        return Text(label=param_name)
    if base_type is int:
        return Number(label=param_name, integer_only=True)
    if base_type is float:
        return Number(label=param_name)
    if base_type is bool:
        return Checkbox(label=param_name)

    # Custom type aliases surface via the normalized type_str
    type_str = _extract_type_string(base_type).lower()
    if type_str == "date":
        return DatePicker(label=param_name)
    if type_str in ("base64str", "namedfile", "multinamedfile"):
        return FileUpload(label=param_name)

    return None


def _duplicate_registration_message(base_id: str, version: int) -> str:
    """Error text for a duplicate `@registry.node` / `register_class` call.

    Optimized for the two cases that actually happen in practice: the caller
    is trying to ship a new version and forgot to bump the number, or they
    re-ran a notebook cell that already registered once.
    """
    next_version = version + 1
    return (
        f"Node '{base_id}@{version}' is already registered on this registry.\n"
        f"  - To register a new version, bump the `version` argument, e.g. "
        f"`@registry.node(\"{base_id}\", version={next_version}, ...)`.\n"
        f"  - If you're re-running a notebook cell, create a fresh "
        f"`NodeRegistry()` (or restart the kernel) so registrations start "
        f"from an empty state.\n"
        f"  - If you meant to replace the existing version, pick a different "
        f"base_id — conductor never silently overwrites a registered node."
    )


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
                raise ValueError(_duplicate_registration_message(base_id, version))

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
            raise ValueError(_duplicate_registration_message(base_id, ver))

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
    # Composition
    # ------------------------------------------------------------------

    def include(self, category: NodeCategory) -> None:
        """Apply every ``@category.node(...)`` decorated function to this registry.

        The idiomatic way for node-author packages to expose their nodes
        without forcing a ``register(registry)`` wrapper: declare a module-
        level ``NodeCategory`` instance, decorate each function with
        ``@category.node(...)``, then let the host do
        ``registry.include(category)``.

        The category is auto-assigned to each registered node, so nodes from
        the same file always group together in the palette.

        Args:
            category: A ``NodeCategory`` whose ``.node(...)`` decorator was
                used at import time to collect pending registrations.
        """
        for base_id, kwargs, func in category.nodes:
            self.node(base_id, **kwargs)(func)

    def merge(
        self,
        other: "NodeRegistry",
        *,
        on_conflict: str = "raise",
    ) -> "NodeRegistry":
        """Copy every node from ``other`` into this registry.

        Versions coexist naturally: if self has ``foo@1`` and other has
        ``foo@2``, the merged registry carries both. A *conflict* is the
        same ``base_id@version`` appearing on both sides.

        Args:
            other: Source registry. Not modified.
            on_conflict: What to do when a full id (``base_id@version``) is
                present on both registries:

                - ``"raise"`` (default) — raise ``ValueError`` on the first
                  conflict with actionable guidance.
                - ``"skip"`` — keep the existing node, ignore the incoming one.
                - ``"error-summary"`` — collect every conflict, then raise
                  one ``ValueError`` listing them all. Useful for surfacing
                  the full collision set in one pass.

        Returns:
            ``self``, so calls can chain: ``reg.merge(a).merge(b)``.
        """
        if on_conflict not in ("raise", "skip", "error-summary"):
            raise ValueError(
                f"Unknown on_conflict mode: {on_conflict!r}. "
                f"Choose one of 'raise', 'skip', 'error-summary'."
            )

        conflicts: list[str] = []
        for full_id, node_def in other._nodes.items():
            if full_id in self._nodes:
                if on_conflict == "raise":
                    raise ValueError(
                        f"Registry merge conflict: '{full_id}' is registered "
                        f"on both registries.\n"
                        f"  - Pass `on_conflict='skip'` to keep the existing "
                        f"node and ignore the incoming one.\n"
                        f"  - Pass `on_conflict='error-summary'` to collect "
                        f"every conflict and raise once at the end.\n"
                        f"  - If you meant to add a new version of the same "
                        f"node, bump `version` on one side before merging."
                    )
                if on_conflict == "skip":
                    continue
                # error-summary — record and keep going
                conflicts.append(full_id)
                continue

            self._nodes[full_id] = node_def
            self._by_base_id.setdefault(node_def.base_id, []).append(node_def)

        if conflicts:
            joined = "\n".join(f"  - {cid}" for cid in conflicts)
            raise ValueError(
                f"Registry merge had {len(conflicts)} conflict(s):\n"
                f"{joined}\n"
                f"Pass `on_conflict='skip'` to accept existing versions, or "
                f"bump `version` on one side to avoid the collision."
            )

        return self

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

        # If the user didn't annotate a widget, fall back to a sensible
        # default based on the parameter's type. Explicit widgets still win.
        if widget_instance is None:
            widget_instance = _default_widget(base_type, param_name)

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
