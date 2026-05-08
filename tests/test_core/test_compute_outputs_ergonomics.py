"""Tests for ergonomics improvements to the ``compute_outputs`` hook.

Covers three independent improvements landing together:

* :class:`conductor.NodeCategory.node` forwards ``compute_outputs`` and
  ``dynamic_handles`` to the underlying registration so hosts don't need
  a side-channel ``register(registry)`` workaround.
* :func:`conductor.registry.dynamic_outputs.strip_sub_output_prefix` is
  promoted as a canonical helper for hooks that read sub-output handle
  names.
* :class:`ComputeOutputsContext` now carries a ``validated_data`` field
  populated by the resolver — hooks for nodes with non-trivial widget
  config no longer re-implement validation-model coercion.
"""

from __future__ import annotations

from typing import Annotated, Any

from conductor import GraphNode, NodeCategory, NodeRegistry, compile
from conductor.metadata import OutputMetadata
from conductor.registry.dynamic_outputs import (
    ComputeOutputsContext,
    strip_sub_output_prefix,
)
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Issue 1 — NodeCategory.node() forwards compute_outputs / dynamic_handles
# ---------------------------------------------------------------------------


class TestCategoryForwardsHookKwargs:
    def test_compute_outputs_forwarded_via_category(self) -> None:
        """``@category.node(..., compute_outputs=fn)`` populates the
        resulting NodeDefinition's ``compute_outputs``."""
        cat = NodeCategory("primitives", label="Primitives")

        def hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="x", type_str="str", label="X")]

        @cat.node(
            "via-category",
            version=1,
            name="ViaCat",
            description="Forwards hook",
            dynamic_handles=True,
            compute_outputs=hook,
        )
        def fn(text: Annotated[str, Text(label="In")]) -> Any:
            return {"x": text}

        registry = NodeRegistry()
        registry.include(cat)

        node_def = registry.get("via-category@1")
        assert node_def.compute_outputs is hook
        assert node_def.dynamic_handles is True

    def test_defaults_unchanged_when_kwargs_omitted(self) -> None:
        """Omitting both kwargs preserves the prior default behaviour."""
        cat = NodeCategory("p2")

        @cat.node("plain", version=1, name="Plain", description="X")
        def fn() -> Annotated[str, Output(label="Out")]:
            return "x"

        registry = NodeRegistry()
        registry.include(cat)

        node_def = registry.get("plain@1")
        assert node_def.compute_outputs is None
        assert node_def.dynamic_handles is False

    def test_hook_runs_at_compile_when_registered_via_category(self) -> None:
        """End-to-end: a category-registered hook fires during compile."""
        cat = NodeCategory("p3")

        def hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            count = ctx.data.get("count", 1)
            return [
                OutputMetadata(name=f"slot_{i}", type_str="str", label=f"Slot {i}")
                for i in range(count)
            ]

        @cat.node(
            "splitter",
            version=1,
            name="Splitter",
            description="Splits",
            dynamic_handles=True,
            compute_outputs=hook,
        )
        def fn(text: Annotated[str, Text(label="In")]) -> Any:
            return {"slot_0": text}

        registry = NodeRegistry()
        registry.include(cat)

        compiled = compile(
            nodes=[GraphNode("n1", "splitter@1", {"count": 3})],
            edges=[],
            registry=registry,
        )
        assert tuple(o.name for o in compiled.node_outputs["n1"]) == (
            "slot_0",
            "slot_1",
            "slot_2",
        )


# ---------------------------------------------------------------------------
# Issue 2 — strip_sub_output_prefix helper
# ---------------------------------------------------------------------------


class TestStripSubOutputPrefix:
    def test_single_segment_returned_verbatim(self) -> None:
        assert strip_sub_output_prefix("plain") == "plain"

    def test_two_segments_drops_parent(self) -> None:
        assert strip_sub_output_prefix("result.foo") == "foo"

    def test_multi_segment_drops_only_first_segment(self) -> None:
        assert strip_sub_output_prefix("result.foo.bar") == "foo.bar"

    def test_output_n_prefix_dropped(self) -> None:
        assert strip_sub_output_prefix("output_3.col") == "col"

    def test_empty_string_unchanged(self) -> None:
        assert strip_sub_output_prefix("") == ""

    def test_trailing_dot_yields_empty_remainder(self) -> None:
        # Edge case: a name ending in '.' should produce ''.
        assert strip_sub_output_prefix("result.") == ""


# ---------------------------------------------------------------------------
# Issue 3 — ComputeOutputsContext.validated_data
# ---------------------------------------------------------------------------


class TestValidatedData:
    def test_validated_data_populated_from_validation_model(self) -> None:
        """Hooks receive the model-validated dict, not just the raw data."""
        registry = NodeRegistry()
        observed: dict[str, Any] = {}

        def hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            observed["validated"] = ctx.validated_data
            observed["raw"] = ctx.data
            return list(ctx.defaults)

        @registry.node(
            "with-validation",
            version=1,
            name="WV",
            description="X",
            compute_outputs=hook,
        )
        def fn(
            value: Annotated[str, Text(label="V")] = "default",
        ) -> Annotated[str, Output(label="Out")]:
            return value

        compile(
            nodes=[GraphNode("n1", "with-validation@1", {"value": "hello"})],
            edges=[],
            registry=registry,
        )

        assert observed["validated"] == {"value": "hello"}
        assert observed["raw"] == {"value": "hello"}

    def test_validated_data_applies_defaults(self) -> None:
        """Missing optional fields are filled from the validation model."""
        registry = NodeRegistry()
        observed: dict[str, Any] = {}

        def hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            observed["validated"] = ctx.validated_data
            return list(ctx.defaults)

        @registry.node(
            "with-default",
            version=1,
            name="WD",
            description="X",
            compute_outputs=hook,
        )
        def fn(
            value: Annotated[str, Text(label="V")] = "the-default",
        ) -> Annotated[str, Output(label="Out")]:
            return value

        compile(
            nodes=[GraphNode("n1", "with-default@1", {})],
            edges=[],
            registry=registry,
        )

        # validation_model fills the default, even though raw ``data`` was empty.
        assert observed["validated"] == {"value": "the-default"}

    def test_validated_data_is_none_when_validation_fails(self) -> None:
        """In-progress edits with missing required fields don't crash the
        resolver; ``validated_data`` is ``None`` and ``data`` still
        carries whatever the host passed."""
        registry = NodeRegistry()
        observed: dict[str, Any] = {}

        def hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            observed["validated"] = ctx.validated_data
            observed["raw"] = ctx.data
            return list(ctx.defaults)

        @registry.node(
            "needs-required",
            version=1,
            name="NR",
            description="X",
            compute_outputs=hook,
        )
        def fn(
            value: Annotated[str, Text(label="V")],  # required, no default
        ) -> Annotated[str, Output(label="Out")]:
            return value

        # Compile with empty data — validation fails (missing required field)
        # but the resolver still calls the hook with validated_data=None.
        compile(
            nodes=[GraphNode("n1", "needs-required@1", {})],
            edges=[],
            registry=registry,
        )

        assert observed["validated"] is None
        assert observed["raw"] == {}
