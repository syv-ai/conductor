"""Node registry — registration, lookup, versioning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from conductor.interface import Interface, model_of
from conductor.registry.definition import Actor, NodeDefinition
from conductor.types import NodeCategory

__all__ = [
    "NodeRegistry",
    "NodeDefinition",
    "Actor",
]


def _parse_timeout(value: Any) -> float | None:
    """Accept a number of seconds or an ISO 8601 duration; return seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"timeout must be positive, got {value}")
        return float(value)
    if isinstance(value, str):
        return _parse_iso8601_duration(value)
    raise TypeError(
        f"timeout must be number of seconds or ISO 8601 string, got {type(value).__name__}"
    )


_ISO8601_RE = None


def _parse_iso8601_duration(source: str) -> float:
    """Parse a (limited) ISO 8601 duration into seconds.

    Accepts ``PT<n>S``, ``PT<n>M``, ``PT<n>H``, and plain ``<n>s`` /
    ``<n>ms`` / ``<n>m`` / ``<n>h`` shorthand. Anything we don't recognize
    raises ``ValueError``.
    """
    import re

    s = source.strip()
    if s.startswith(("P", "p")):
        # Full ISO 8601
        m = re.fullmatch(
            r"[Pp](?:(\d+)[Dd])?(?:[Tt](?:(\d+)[Hh])?(?:(\d+)[Mm])?(?:(\d+(?:\.\d+)?)[Ss])?)?",
            s,
        )
        if not m:
            raise ValueError(f"Unrecognized ISO 8601 duration: {source!r}")
        days, hours, minutes, seconds = m.groups()
        total = 0.0
        if days:
            total += int(days) * 86400
        if hours:
            total += int(hours) * 3600
        if minutes:
            total += int(minutes) * 60
        if seconds:
            total += float(seconds)
        if total == 0:
            raise ValueError(f"ISO 8601 duration evaluated to zero: {source!r}")
        return total

    # Shorthand: 30s, 250ms, 2h, 5m
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?", s)
    if not m:
        raise ValueError(f"Unrecognized duration: {source!r}")
    value_str, unit = m.groups()
    value = float(value_str)
    factor = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, None: 1}[unit]
    return value * factor


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
        actor: Any = None,
        timeout: Any = None,
        idempotency_key: str | None = None,
        uses: list[str] | None = None,
        is_decision: bool = False,
        is_signal: bool = False,
        dynamic_handles: bool = False,
        compute_outputs: Callable | None = None,
        compute_inputs: Callable | None = None,
    ) -> Callable:
        """Decorator to register a function as a node.

        New process-standard kwargs (all optional):

        * ``actor`` — who performs this step. Accepts an :class:`Actor`,
          a dict (``{"kind": "human", "role": "finance_manager"}``), or a
          bare string (``"system"``).
        * ``timeout`` — seconds (``float``) or ISO 8601 duration
          (``"PT30S"``). Engine wraps ``execute`` with
          ``asyncio.wait_for`` and raises :class:`NodeTimeoutError` on
          expiry.
        * ``idempotency_key`` — CEL expression evaluated against the
          node's resolved inputs. Surfaced on ``node_start`` events and
          injected into the function as the ``idempotency_key`` parameter
          when declared.
        * ``uses`` — list of flow-level dependency ids this node touches.
          Validated at compile time against the flow's ``dependencies``
          manifest.
        * ``is_decision`` / ``is_signal`` — low-level markers used by the
          engine to detect decision and signal nodes.
        """

        def decorator(func: Callable) -> Callable:
            full_id = f"{base_id}@{version}"
            if full_id in self._nodes:
                raise ValueError(_duplicate_registration_message(base_id, version))

            iface = Interface.of(func)

            node_def = NodeDefinition(
                id=full_id,
                base_id=base_id,
                version=version,
                name=name,
                description=description,
                tags=tuple(tags or []),
                category=category,
                inputs=iface.inputs,
                outputs=iface.outputs,
                validation_model=model_of(iface.inputs),
                func=func,
                max_retries=max_retries,
                retry_delay=retry_delay,
                width=width,
                docs=docs,
                actor=Actor.coerce(actor),
                timeout_seconds=_parse_timeout(timeout),
                idempotency_key=idempotency_key,
                uses=tuple(uses or []),
                is_decision=is_decision,
                is_signal=is_signal,
                dynamic_handles=dynamic_handles,
                compute_outputs=compute_outputs,
                compute_inputs=compute_inputs,
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
        actor = Actor.coerce(getattr(node_cls, "node_actor", None))
        timeout = _parse_timeout(getattr(node_cls, "node_timeout", None))
        idem_key = getattr(node_cls, "node_idempotency_key", None)
        uses = tuple(getattr(node_cls, "node_uses", ()) or ())
        is_decision = bool(getattr(node_cls, "node_is_decision", False))
        is_signal = bool(getattr(node_cls, "node_is_signal", False))
        compute_outputs = getattr(node_cls, "compute_outputs", None)
        # Reject bound-method shapes — class authors should wrap with
        # ``staticmethod`` so the hook gets called as a plain function.
        if compute_outputs is not None and not callable(compute_outputs):
            compute_outputs = None

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
            validation_model=None,
            func=None,
            _node_class=node_cls,
            actor=actor,
            timeout_seconds=timeout,
            idempotency_key=idem_key,
            uses=uses,
            is_decision=is_decision,
            is_signal=is_signal,
            compute_outputs=compute_outputs,
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

