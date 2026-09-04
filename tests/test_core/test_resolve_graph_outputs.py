"""Graph-level output resolution ahead of compile (``resolve_graph_outputs``).

The same topological walk compile() runs, over host-supplied
``definitions`` instead of a registry. The equivalence test pins that the
two never diverge.
"""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile, resolve_graph_outputs
from conductor.dtype import DType
from conductor.errors import CompilationError, CycleDetectionError
from conductor.graph.binding import Static
from conductor.metadata import Output
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "resolve-outputs-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


class StaticSrc(NodeDefinition):
    id = "static-src"
    title = "Static"
    description = "Static source"
    category = "test"

    def run(self) -> Out:
        return Txt("x")


class Relay(NodeDefinition):
    id = "relay"
    title = "Relay"
    description = "Passes its input on"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return text


class Schema(NodeDefinition):
    """One extra output per comma-separated name the author typed."""

    id = "dyn-schema"
    title = "Schema"
    description = "Fields from a typed-in list"
    category = "test"

    def run(
        self, fields: Annotated[Txt, Textarea(title="Fields", show_handle=False)] = Txt("")
    ) -> Out:
        return Txt("")

    def compute_outputs(
        self,
        declared: tuple[Output, ...],
        values: Mapping[str, Any],
        arriving: Mapping[str, type[DType]],
    ) -> tuple[Output, ...]:
        names = [n for n in str(values.get("fields", "")).split(",") if n]
        return declared + tuple(Output(name=n, dtype=Txt, title=n) for n in names)


def _make_registry() -> NodeRegistry:
    reg = NodeRegistry()
    reg.register(StaticSrc)
    reg.register(Relay)
    reg.register(Schema)
    return reg


def _definitions(reg: NodeRegistry, nodes: list[GraphNode]) -> dict:
    return {n.type: reg.get(n.type) for n in nodes}


def test_static_only_graph_returns_static_declarations() -> None:
    reg = _make_registry()
    nodes = [GraphNode(id="a", type="static-src", version=1)]
    out = resolve_graph_outputs(nodes, [], _definitions(reg, nodes))
    assert [o.name for o in out["a"]] == ["result"]


def test_hook_adds_outputs_from_typed_in_values() -> None:
    reg = _make_registry()
    nodes = [GraphNode(id="a", type="dyn-schema", version=1, bindings={"fields": Static(value="amount,name")})]
    out = resolve_graph_outputs(nodes, [], _definitions(reg, nodes))
    assert [o.name for o in out["a"]] == ["result", "amount", "name"]
    assert out["a"][1].dtype is Txt


def test_none_definition_is_extension_semantics() -> None:
    # A ``None`` value means "known to the host, no definition" (an
    # extension node): it resolves to ().
    reg = _make_registry()
    nodes = [
        GraphNode(id="ext", type="flow-version:abc", version=1),
        GraphNode(id="b", type="relay", version=1),
    ]
    edges = [
        GraphEdge(
            id="e1", source="ext", target="b",
            source_handle="result", target_handle="text",
        )
    ]
    definitions = {**_definitions(reg, nodes), "flow-version:abc": None}
    out = resolve_graph_outputs(nodes, edges, definitions)
    assert out["ext"] == ()
    assert [o.name for o in out["b"]] == ["result"]


def test_missing_definitions_key_raises() -> None:
    nodes = [GraphNode(id="a", type="mystery", version=1)]
    with pytest.raises(CompilationError, match="mystery"):
        resolve_graph_outputs(nodes, [], {})


def test_dangling_edge_endpoint_raises() -> None:
    reg = _make_registry()
    nodes = [GraphNode(id="a", type="static-src", version=1)]
    edges = [
        GraphEdge(id="e1", source="a", target="ghost",
                  source_handle="result", target_handle="x")
    ]
    with pytest.raises(CompilationError, match="ghost"):
        resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))


def test_cycle_raises() -> None:
    reg = _make_registry()
    nodes = [
        GraphNode(id="a", type="relay", version=1),
        GraphNode(id="b", type="relay", version=1),
    ]
    edges = [
        GraphEdge(id="e1", source="a", target="b",
                  source_handle="result", target_handle="text"),
        GraphEdge(id="e2", source="b", target="a",
                  source_handle="result", target_handle="text"),
    ]
    with pytest.raises(CycleDetectionError):
        resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))


def test_hook_exception_propagates() -> None:
    """Nothing wraps a hook that raises: its own exception surfaces where
    the placement is resolved."""

    class Boom(NodeDefinition):
        id = "boom"
        title = "Boom"
        description = "Raising hook"
        category = "test"

        def run(self) -> Out:
            return Txt("")

        def compute_outputs(
            self,
            declared: tuple[Output, ...],
            values: Mapping[str, Any],
            arriving: Mapping[str, type[DType]],
        ) -> tuple[Output, ...]:
            raise ValueError("kaputt")

    nodes = [GraphNode(id="a", type="boom", version=1)]
    with pytest.raises(ValueError, match="kaputt"):
        resolve_graph_outputs(nodes, [], {"boom": Boom})


def test_equivalence_with_compile_node_outputs() -> None:
    # The public API and compile() must never diverge: same graph, same map.
    reg = _make_registry()
    nodes = [
        GraphNode(id="a", type="dyn-schema", version=1, bindings={"fields": Static(value="field")}),
        GraphNode(id="b", type="relay", version=1),
    ]
    edges = [
        GraphEdge(id="e1", source="a", target="b",
                  source_handle="field", target_handle="text"),
    ]
    compiled = compile(nodes=nodes, edges=edges, registry=reg)
    standalone = resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))
    assert standalone == compiled.node_outputs
