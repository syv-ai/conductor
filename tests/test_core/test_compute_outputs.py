"""Tests for the compile-time ``compute_outputs`` hook."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import CompilationError
from conductor.metadata import OutputMetadata
from conductor.registry.dynamic_outputs import (
    ComputeOutputsContext,
    IncomingBinding,
)
from conductor.registry.schema import serialize_registry
from conductor.widgets import Output, Text


# ---------------------------------------------------------------------------
# Basic single-node hook
# ---------------------------------------------------------------------------


class TestBasicSingleNodeHook:
    def test_hook_replaces_static_outputs_in_compiled_graph(self) -> None:
        """A node with a hook reports its hook-derived outputs on
        ``CompiledGraph.node_outputs``."""
        reg = NodeRegistry()

        def my_outputs(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            count = ctx.data.get("count", 1)
            return [
                OutputMetadata(name=f"slot_{i}", type_str="str", label=f"Slot {i}")
                for i in range(count)
            ]

        @reg.node(
            "splitter",
            version=1,
            name="Splitter",
            description="Splits into N outputs",
            dynamic_handles=True,
            compute_outputs=my_outputs,
        )
        def splitter(text: Annotated[str, Text(label="In")]) -> Any:
            return {"slot_0": text}

        compiled = compile(
            nodes=[GraphNode("n1", "splitter@1", {"count": 3})],
            edges=[],
            registry=reg,
        )

        outputs = compiled.node_outputs["n1"]
        assert tuple(o.name for o in outputs) == ("slot_0", "slot_1", "slot_2")
        assert all(o.type_str == "str" for o in outputs)

    def test_no_hook_falls_back_to_static_outputs(self) -> None:
        """Nodes without a hook get their static outputs verbatim."""
        reg = NodeRegistry()

        @reg.node("noop", version=1, name="Noop", description="Pass-through")
        def noop(
            text: Annotated[str, Text(label="In")],
        ) -> Annotated[str, Output(label="Out")]:
            return text

        compiled = compile(
            nodes=[GraphNode("n1", "noop@1", None)],
            edges=[],
            registry=reg,
        )

        node_def = reg.get("noop@1")
        assert compiled.node_outputs["n1"] == tuple(node_def.outputs)

    def test_class_based_hook_via_staticmethod(self) -> None:
        """``register_class`` reads ``compute_outputs`` off the class."""
        from conductor.node import BaseNode

        reg = NodeRegistry()

        def class_outputs(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="custom", type_str="int", label="Custom")]

        class MyNode(BaseNode):
            node_id = "klass"
            node_name = "Klass"
            node_description = "Class with hook"
            compute_outputs = staticmethod(class_outputs)

            def execute(self, req: Any) -> Any:
                return {"custom": 42}

        reg.register_class(MyNode)
        node_def = reg.get("klass@1")
        assert node_def.compute_outputs is class_outputs


# ---------------------------------------------------------------------------
# Topological resolution — A → B with hooks on both
# ---------------------------------------------------------------------------


class TestTopologicalResolution:
    def test_downstream_hook_sees_upstream_resolved_outputs(self) -> None:
        """Node B's hook receives A's *resolved* outputs via incoming bindings."""
        reg = NodeRegistry()

        def producer_outputs(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="result", type_str="custom_int", label="Out")]

        def consumer_outputs(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            # Ingest the upstream type and propagate it.
            assert len(ctx.incoming) == 1
            incoming: IncomingBinding = ctx.incoming[0]
            return [
                OutputMetadata(
                    name="result",
                    type_str=f"echoed[{incoming.source_output.type_str}]",
                    label="Echoed",
                )
            ]

        @reg.node(
            "producer", version=1, name="Producer", description="A",
            compute_outputs=producer_outputs,
        )
        def producer(
            v: Annotated[str, Text(label="V")] = "hi",
        ) -> Annotated[str, Output(label="Out")]:
            return v

        @reg.node(
            "consumer", version=1, name="Consumer", description="B",
            compute_outputs=consumer_outputs,
        )
        def consumer(
            x: Annotated[str, Text(label="X")],
        ) -> Annotated[str, Output(label="Out")]:
            return x

        compiled = compile(
            nodes=[
                GraphNode("a", "producer@1", None),
                GraphNode("b", "consumer@1", None),
            ],
            edges=[GraphEdge("e1", "a", "b", "result", "x")],
            registry=reg,
        )

        a_out = compiled.node_outputs["a"]
        b_out = compiled.node_outputs["b"]
        assert a_out[0].type_str == "custom_int"
        assert b_out[0].type_str == "echoed[custom_int]"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_non_list_return_raises(self) -> None:
        reg = NodeRegistry()

        def bad(ctx: ComputeOutputsContext) -> Any:
            return "not a list"

        @reg.node(
            "bad", version=1, name="Bad", description="Bad",
            dynamic_handles=True, compute_outputs=bad,
        )
        def bad_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        with pytest.raises(CompilationError, match="must return list"):
            compile(
                nodes=[GraphNode("n1", "bad@1", None)],
                edges=[],
                registry=reg,
            )

    def test_duplicate_names_raises(self) -> None:
        reg = NodeRegistry()

        def dup(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [
                OutputMetadata(name="x", type_str="str", label="X"),
                OutputMetadata(name="x", type_str="str", label="X2"),
            ]

        @reg.node(
            "dup", version=1, name="Dup", description="Dup",
            dynamic_handles=True, compute_outputs=dup,
        )
        def dup_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        with pytest.raises(CompilationError, match="duplicate output name"):
            compile(
                nodes=[GraphNode("n1", "dup@1", None)],
                edges=[],
                registry=reg,
            )

    def test_dropping_static_handle_without_dynamic_handles_raises(self) -> None:
        reg = NodeRegistry()

        def drop(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            # Returns nothing — drops the static "result" handle.
            return []

        @reg.node(
            "drop", version=1, name="Drop", description="Drop",
            compute_outputs=drop,
        )
        def drop_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        with pytest.raises(CompilationError, match="dropped statically declared"):
            compile(
                nodes=[GraphNode("n1", "drop@1", None)],
                edges=[],
                registry=reg,
            )

    def test_dropping_static_handle_with_dynamic_handles_allowed(self) -> None:
        reg = NodeRegistry()

        def replace(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="other", type_str="int", label="Other")]

        @reg.node(
            "replace", version=1, name="Replace", description="Replace",
            dynamic_handles=True, compute_outputs=replace,
        )
        def replace_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        compiled = compile(
            nodes=[GraphNode("n1", "replace@1", None)],
            edges=[],
            registry=reg,
        )
        assert tuple(o.name for o in compiled.node_outputs["n1"]) == ("other",)

    def test_hook_exception_wrapped_as_compilation_error(self) -> None:
        reg = NodeRegistry()

        def boom(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            raise RuntimeError("kaboom")

        @reg.node(
            "boom", version=1, name="Boom", description="Boom",
            dynamic_handles=True, compute_outputs=boom,
        )
        def boom_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        with pytest.raises(CompilationError, match="compute_outputs failed"):
            compile(
                nodes=[GraphNode("n1", "boom@1", None)],
                edges=[],
                registry=reg,
            )


# ---------------------------------------------------------------------------
# Extension nodes coexist with hook-based nodes
# ---------------------------------------------------------------------------


class TestExtensionCoexistence:
    def test_extension_node_alongside_hook_node(self) -> None:
        """An extension node with no NodeDefinition coexists with a hook-driven
        registered node — the resolver tolerates the missing definition."""
        reg = NodeRegistry()

        def hooked(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="result", type_str="str", label="Result")]

        @reg.node(
            "hooked", version=1, name="Hooked", description="Hooked",
            compute_outputs=hooked,
        )
        def hooked_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        class MockExtensionResolver:
            def is_known_type(self, node_type: str) -> bool:
                return node_type.startswith("ext:")

            def create_executor(self, node_type: str) -> Any:
                return None

        compiled = compile(
            nodes=[
                GraphNode("h", "hooked@1", None),
                GraphNode("e", "ext:thing@1", None),
            ],
            edges=[],
            registry=reg,
            extension_resolver=MockExtensionResolver(),
        )

        # Hook-based node has resolved outputs; extension node has empty tuple.
        assert tuple(o.name for o in compiled.node_outputs["h"]) == ("result",)
        assert compiled.node_outputs["e"] == ()


# ---------------------------------------------------------------------------
# Type-check uses the resolved type string
# ---------------------------------------------------------------------------


class TestTypeCheckUsesResolved:
    def test_type_warning_uses_hook_derived_type(self) -> None:
        """When a hook re-types its output, the type-check sees the new type
        and warns accordingly on incompatible downstream connections."""
        reg = NodeRegistry()

        def retype(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return [OutputMetadata(name="result", type_str="custom_blob", label="Out")]

        @reg.node(
            "retype", version=1, name="Retype", description="Retype",
            compute_outputs=retype,
        )
        def retype_node() -> Annotated[str, Output(label="Out")]:
            return "x"

        @reg.node("int-in", version=1, name="Int In", description="Int In")
        def int_in(
            num: Annotated[int, Text(label="N")],
        ) -> Annotated[int, Output(label="Out")]:
            return num

        compiled = compile(
            nodes=[
                GraphNode("a", "retype@1", None),
                GraphNode("b", "int-in@1", None),
            ],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=reg,
        )

        # The warning should reference the hook-derived type, not "str".
        mismatches = [w for w in compiled.type_warnings if w.code == "type-mismatch"]
        assert any(w.source_type == "custom_blob" for w in mismatches), (
            f"expected source_type=custom_blob in {[w.source_type for w in mismatches]}"
        )


# ---------------------------------------------------------------------------
# Schema serialization flag
# ---------------------------------------------------------------------------


class TestSerializationFlag:
    def test_has_dynamic_outputs_emitted_when_hook_present(self) -> None:
        reg = NodeRegistry()

        def fn(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
            return list(ctx.defaults)

        @reg.node(
            "with-hook", version=1, name="WithHook", description="X",
            compute_outputs=fn,
        )
        def with_hook() -> Annotated[str, Output(label="Out")]:
            return "x"

        @reg.node("no-hook", version=1, name="NoHook", description="X")
        def no_hook() -> Annotated[str, Output(label="Out")]:
            return "x"

        payload = {n["id"]: n for n in serialize_registry(reg)}
        assert payload["with-hook@1"].get("has_dynamic_outputs") is True
        assert "has_dynamic_outputs" not in payload["no-hook@1"]
