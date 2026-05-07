"""NodeDefinition frozen dataclass."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from conductor.metadata import InputMetadata, OutputMetadata
from conductor.types import NodeCategory, ResultFormat


@dataclass(frozen=True)
class Actor:
    """Declarative actor metadata — who performs this step?

    A frontend can render human steps differently from system steps; an audit
    trail can attribute actions; SLA trackers can weight human steps vs
    system steps. The runtime is completely indifferent to actor metadata.

    Attributes:
        kind: One of ``"system"``, ``"human"``, ``"agent"``, or
            ``"external_service"``. Fixed set for tooling interop.
        role: Optional free-form role inside the kind (e.g. ``"finance_manager"``,
            ``"claude-3-sonnet"``, ``"stripe"``).
    """

    kind: str
    role: str | None = None

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        ("system", "human", "agent", "external_service")
    )

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(
                f"Invalid actor kind {self.kind!r}. Valid kinds: "
                f"{sorted(self._KINDS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.role:
            out["role"] = self.role
        return out

    @classmethod
    def coerce(cls, value: Any) -> "Actor | None":
        """Accept a bare string, dict, or Actor and return an Actor (or None)."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, dict):
            return cls(kind=value["kind"], role=value.get("role"))
        raise TypeError(
            f"Cannot coerce {type(value).__name__} to Actor. Pass a string, dict, or Actor."
        )


@dataclass(frozen=True)
class NodeDefinition:
    """Immutable definition of a registered node."""

    id: str
    base_id: str
    version: int
    name: str
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    category: NodeCategory = NodeCategory.IO
    inputs: tuple[InputMetadata, ...] = field(default_factory=tuple)
    outputs: tuple[OutputMetadata, ...] = field(default_factory=tuple)
    result_format: ResultFormat = ResultFormat.SINGLE
    validation_model: type | None = None
    func: Callable[..., Any] | None = None
    _node_class: type | None = None
    max_retries: int = 0
    retry_delay: float = 1.0
    width: int | None = None
    docs: str | None = None
    # --- Process-standard additions ---
    actor: Actor | None = None
    timeout_seconds: float | None = None
    idempotency_key: str | None = None  # CEL expression
    uses: tuple[str, ...] = field(default_factory=tuple)
    is_decision: bool = False
    is_signal: bool = False
    # Compound markers (for-each-start/end, etc.) accept arbitrary
    # input/output handle names beyond what the function signature
    # declares. The declared inputs/outputs are templates the host can
    # use as a starting point; the compiler skips strict handle
    # validation for these nodes and their validation model accepts
    # extra fields. Dynamic-shape behavior (parallel-zip, fan-out) is
    # supplied by the compound runtime that owns the marker.
    dynamic_handles: bool = False
