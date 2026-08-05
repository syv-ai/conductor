"""Tests for the compile-time ``compute_inputs`` hook."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.category import NodeCategory
from conductor.errors import CompilationError
from conductor.graph.dynamic_inputs import resolve_node_inputs
from conductor.metadata import InputMetadata
from conductor.registry.dynamic_inputs import ComputeInputsContext
from conductor.types import WidgetType
from conductor.widgets import Output, Text


def _hook(ctx: ComputeInputsContext) -> list[InputMetadata]:
    return [
        InputMetadata(
            name="customers",
            type_str="table",
            label="customers",
            widget=WidgetType.TEXT,
        )
    ]


class TestContextShape:
    def test_context_carries_data_defaults_and_node_id(self) -> None:
        ctx = ComputeInputsContext(
            data={"code": "def f(): pass"},
            node_id="n1",
            defaults=(InputMetadata(name="code", type_str="str", label="Kode"),),
        )
        assert ctx.data["code"] == "def f(): pass"
        assert ctx.node_id == "n1"
        assert ctx.defaults[0].name == "code"
        assert ctx.validated_data is None

    def test_context_is_frozen(self) -> None:
        ctx = ComputeInputsContext(data={}, node_id="n1", defaults=())
        with pytest.raises(Exception):
            ctx.node_id = "n2"  # type: ignore[misc]


class TestRegistration:
    def test_registry_node_stores_the_hook(self) -> None:
        reg = NodeRegistry()

        @reg.node(
            "dyn",
            version=1,
            name="Dyn",
            description="Dyn",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def dyn(**kwargs: Any) -> Annotated[str, Output(label="Ud")]:
            return "x"

        assert reg.get("dyn@1").compute_inputs is _hook

    def test_category_forwards_the_hook(self) -> None:
        cat = NodeCategory("prims", label="Prims")

        @cat.node(
            "via-cat",
            version=1,
            name="ViaCat",
            description="Forwards",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def fn(**kwargs: Any) -> Annotated[str, Output(label="Ud")]:
            return "x"

        reg = NodeRegistry()
        reg.include(cat)
        assert reg.get("via-cat@1").compute_inputs is _hook

    def test_absent_hook_defaults_to_none(self) -> None:
        reg = NodeRegistry()

        @reg.node("plain", version=1, name="Plain", description="Plain")
        def plain(
            text: Annotated[str, Text(label="T")],
        ) -> Annotated[str, Output(label="Ud")]:
            return text

        assert reg.get("plain@1").compute_inputs is None


class TestResolver:
    def test_no_hook_returns_static_inputs(self) -> None:
        reg = NodeRegistry()

        @reg.node("plain2", version=1, name="P", description="P")
        def plain2(
            text: Annotated[str, Text(label="T")],
        ) -> Annotated[str, Output(label="U")]:
            return text

        got = resolve_node_inputs(
            node=GraphNode("n1", "plain2@1", None), node_def=reg.get("plain2@1")
        )
        assert [i.name for i in got] == ["text"]

    def test_hook_result_replaces_the_roster(self) -> None:
        reg = NodeRegistry()

        @reg.node(
            "dyn2",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def dyn2(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        got = resolve_node_inputs(
            node=GraphNode("n1", "dyn2@1", {"code": "x"}), node_def=reg.get("dyn2@1")
        )
        assert [i.name for i in got] == ["customers"]

    def test_an_extension_node_resolves_to_nothing(self) -> None:
        got = resolve_node_inputs(node=GraphNode("n1", "unknown@1", None), node_def=None)
        assert got == ()

    def test_duplicate_names_raise(self) -> None:
        def dup(ctx: ComputeInputsContext) -> list[InputMetadata]:
            return [
                InputMetadata(name="a", type_str="str", label="A"),
                InputMetadata(name="a", type_str="str", label="A2"),
            ]

        reg = NodeRegistry()

        @reg.node(
            "dup-in",
            version=1,
            name="Dup",
            description="Dup",
            dynamic_handles=True,
            compute_inputs=dup,
        )
        def dup_node(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        with pytest.raises(CompilationError, match="duplicate input name"):
            resolve_node_inputs(
                node=GraphNode("n1", "dup-in@1", None), node_def=reg.get("dup-in@1")
            )

    def test_wrong_return_type_raises(self) -> None:
        reg = NodeRegistry()

        @reg.node(
            "bad-in",
            version=1,
            name="Bad",
            description="Bad",
            dynamic_handles=True,
            compute_inputs=lambda ctx: "nope",
        )
        def bad(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        with pytest.raises(CompilationError, match="must return list"):
            resolve_node_inputs(
                node=GraphNode("n1", "bad-in@1", None), node_def=reg.get("bad-in@1")
            )

    def test_a_raising_hook_becomes_a_compilation_error(self) -> None:
        def boom(ctx: ComputeInputsContext) -> list[InputMetadata]:
            raise RuntimeError("kaboom")

        reg = NodeRegistry()

        @reg.node(
            "boom-in",
            version=1,
            name="Boom",
            description="Boom",
            dynamic_handles=True,
            compute_inputs=boom,
        )
        def boom_node(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        with pytest.raises(CompilationError, match="kaboom"):
            resolve_node_inputs(
                node=GraphNode("n1", "boom-in@1", None), node_def=reg.get("boom-in@1")
            )

    def test_dropping_a_static_input_without_dynamic_handles_raises(self) -> None:
        reg = NodeRegistry()

        @reg.node(
            "strict",
            version=1,
            name="S",
            description="S",
            compute_inputs=lambda ctx: [
                InputMetadata(name="other", type_str="str", label="Other")
            ],
        )
        def strict(
            text: Annotated[str, Text(label="T")],
        ) -> Annotated[str, Output(label="U")]:
            return text

        with pytest.raises(CompilationError, match="dropped statically declared"):
            resolve_node_inputs(
                node=GraphNode("n1", "strict@1", None), node_def=reg.get("strict@1")
            )


class TestCompileIntegration:
    def test_compiled_graph_carries_resolved_inputs(self) -> None:
        reg = NodeRegistry()

        @reg.node(
            "dyn3",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def dyn3(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        compiled = compile(
            nodes=[GraphNode("n1", "dyn3@1", {"code": "x"})], edges=[], registry=reg
        )
        assert [i.name for i in compiled.node_inputs["n1"]] == ["customers"]

    def test_a_node_without_a_hook_gets_its_static_inputs(self) -> None:
        reg = NodeRegistry()

        @reg.node("plain3", version=1, name="P", description="P")
        def plain3(
            text: Annotated[str, Text(label="T")],
        ) -> Annotated[str, Output(label="U")]:
            return text

        compiled = compile(
            nodes=[GraphNode("n1", "plain3@1", None)], edges=[], registry=reg
        )
        assert [i.name for i in compiled.node_inputs["n1"]] == ["text"]


class TestEdgesIntoDynamicInputs:
    def _registry(self) -> NodeRegistry:
        reg = NodeRegistry()

        @reg.node("src2", version=1, name="S", description="S")
        def src2() -> Annotated[str, Output(label="U")]:
            return "x"

        @reg.node(
            "dyn4",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def dyn4(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        return reg

    def test_an_edge_into_a_hook_declared_handle_compiles(self) -> None:
        compiled = compile(
            nodes=[
                GraphNode("a", "src2@1", None),
                GraphNode("b", "dyn4@1", {"code": "x"}),
            ],
            edges=[GraphEdge("e1", "a", "b", "result", "customers")],
            registry=self._registry(),
        )
        assert "b" in compiled.execution_order

    def test_a_hook_declared_handle_is_actually_type_checked(self) -> None:
        # ``src2`` produces str; the hook declares ``customers`` as table.
        # Before the overlay this edge was skipped silently (see the
        # characterization suite); afterwards it must warn like any other
        # mistyped edge.
        compiled = compile(
            nodes=[
                GraphNode("a", "src2@1", None),
                GraphNode("b", "dyn4@1", {"code": "x"}),
            ],
            edges=[GraphEdge("e1", "a", "b", "result", "customers")],
            registry=self._registry(),
        )
        assert compiled.type_warnings != ()

    def test_a_handle_the_hook_did_not_declare_is_still_skipped(self) -> None:
        # Unchanged: unknown target handles are ignored, not rejected. The
        # overlay must not turn this into an error.
        compiled = compile(
            nodes=[
                GraphNode("a", "src2@1", None),
                GraphNode("b", "dyn4@1", {"code": "x"}),
            ],
            edges=[GraphEdge("e1", "a", "b", "result", "nope")],
            registry=self._registry(),
        )
        assert "b" in compiled.execution_order


class TestConsumeIntoDynamicInput:
    def test_consume_into_a_hook_declared_input_compiles(self) -> None:
        reg = NodeRegistry()

        @reg.node("src3", version=1, name="S", description="S")
        def src3() -> Annotated[str, Output(label="U")]:
            return "x"

        @reg.node(
            "dyn5",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def dyn5(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        compiled = compile(
            nodes=[
                GraphNode("a", "src3@1", None, produces={"result": "Delt"}),
                GraphNode(
                    "b",
                    "dyn5@1",
                    {"code": "x"},
                    consumes={"customers": ("a", "result")},
                ),
            ],
            edges=[],
            registry=reg,
        )
        assert "b" in compiled.execution_order


class TestPerInstanceParamInfo:
    def test_two_instances_of_one_dynamic_type_do_not_share_a_cache(self) -> None:
        from conductor.execution.resolver import InputResolver

        def per_node(ctx: ComputeInputsContext) -> list[InputMetadata]:
            if ctx.node_id == "n1":
                return [InputMetadata(name="a", type_str="list[str]", label="A")]
            return [InputMetadata(name="a", type_str="str", label="A")]

        reg = NodeRegistry()

        @reg.node(
            "dyn6",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=per_node,
        )
        def dyn6(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        compiled = compile(
            nodes=[GraphNode("n1", "dyn6@1", None), GraphNode("n2", "dyn6@1", None)],
            edges=[],
            registry=reg,
        )
        resolver = InputResolver(reg, node_inputs=compiled.node_inputs)
        # n1 declared list[str]; n2 declared str. expects_list must differ.
        assert resolver._param_info("dyn6@1", "a", node_id="n1")[1] is True
        assert resolver._param_info("dyn6@1", "a", node_id="n2")[1] is False


class TestValidationErrorLabels:
    def test_a_hook_declared_input_uses_its_label_in_errors(self) -> None:
        import pydantic
        from conductor.execution.engine import _format_validation_error

        class Model(pydantic.BaseModel):
            antal: int

        reg = NodeRegistry()

        @reg.node(
            "dyn7",
            version=1,
            name="D",
            description="D",
            dynamic_handles=True,
            compute_inputs=lambda ctx: [
                InputMetadata(name="antal", type_str="int", label="Antal rækker")
            ],
        )
        def dyn7(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        try:
            Model(antal="ikke et tal")
        except pydantic.ValidationError as exc:
            err = exc

        node_def = reg.get("dyn7@1")
        # Without the resolved roster the handle is absent from the static
        # schema, so its error reads as the bare name.
        assert "Antal rækker" not in _format_validation_error(err, node_def)
        # With it, the author sees the label they gave the parameter.
        resolved = compile(
            nodes=[GraphNode("n1", "dyn7@1", None)], edges=[], registry=reg
        ).node_inputs["n1"]
        assert "Antal rækker" in _format_validation_error(err, node_def, resolved)


class TestSerializationFlag:
    def test_has_dynamic_inputs_emitted_when_hook_present(self) -> None:
        from conductor.registry.schema import serialize_registry

        reg = NodeRegistry()

        @reg.node(
            "with-in-hook",
            version=1,
            name="W",
            description="W",
            dynamic_handles=True,
            compute_inputs=_hook,
        )
        def with_hook(**kwargs: Any) -> Annotated[str, Output(label="U")]:
            return "x"

        @reg.node("no-in-hook", version=1, name="N", description="N")
        def no_hook(
            text: Annotated[str, Text(label="T")],
        ) -> Annotated[str, Output(label="U")]:
            return text

        payload = {n["id"]: n for n in serialize_registry(reg)}
        assert payload["with-in-hook@1"].get("has_dynamic_inputs") is True
        assert "has_dynamic_inputs" not in payload["no-in-hook@1"]
