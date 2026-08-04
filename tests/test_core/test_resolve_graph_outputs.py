"""Public graph-level output resolution (``resolve_graph_outputs``).

The ahead-of-compile twin of compile() step 8b: same topological walk,
same per-node hook engine, host-supplied ``definitions`` instead of a
registry. The equivalence test pins that the two can never diverge.
"""

from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile, resolve_graph_outputs
from conductor.errors import CompilationError, CycleDetectionError
from conductor.metadata import OutputMetadata
from conductor.registry.dynamic_outputs import ComputeOutputsContext
from conductor.widgets import Output


def _make_registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("static-src", version=1, name="Static", description="Static source")
    def static_src() -> Annotated[str, Output(label="Tekst")]:
        return "x"

    def passthrough_hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
        extras = [
            OutputMetadata(
                name=b.source_output.name,
                type_str=b.source_output.type_str,
                label=b.source_output.label,
            )
            for b in ctx.incoming
        ]
        return list(ctx.defaults) + extras

    @reg.node(
        "dyn-passthrough", version=1, name="Passthrough", description="Passes incoming",
        compute_outputs=passthrough_hook,
    )
    def dyn_passthrough(inputs: str = "") -> Annotated[str, Output(label="Resultat")]:
        return inputs

    def schema_hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
        rows = ctx.data.get("fields") or []
        return list(ctx.defaults) + [
            OutputMetadata(name=r["name"], type_str=r["type"], label=r["name"])
            for r in rows
        ]

    @reg.node(
        "dyn-schema", version=1, name="Schema", description="Schema-driven fields",
        compute_outputs=schema_hook,
    )
    def dyn_schema(fields: list | None = None) -> Annotated[str, Output(label="Resultat")]:
        return ""

    return reg


def _definitions(reg: NodeRegistry, nodes: list[GraphNode]) -> dict:
    return {n.type: reg.get(n.type) for n in nodes}


def test_static_only_graph_returns_static_declarations() -> None:
    reg = _make_registry()
    nodes = [GraphNode(id="a", type="static-src@1", data={})]
    out = resolve_graph_outputs(nodes, [], _definitions(reg, nodes))
    assert [o.name for o in out["a"]] == ["result"]


def test_downstream_hook_sees_upstream_resolved_outputs() -> None:
    # a (schema hook: dynamic field "Beløb": int) → b (passthrough hook).
    # b's binding must carry a's RESOLVED metadata, not a static decl or
    # an "any" synthesis.
    reg = _make_registry()
    nodes = [
        GraphNode(
            id="a",
            type="dyn-schema@1",
            data={"fields": [{"name": "Beløb", "type": "int"}]},
        ),
        GraphNode(id="b", type="dyn-passthrough@1", data={}),
    ]
    edges = [
        GraphEdge(
            id="e1", source="a", target="b",
            source_handle="Beløb", target_handle="inputs",
        )
    ]
    out = resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))
    passed = [o for o in out["b"] if o.name == "Beløb"]
    assert len(passed) == 1
    assert passed[0].type_str == "int"


def test_none_definition_is_extension_semantics() -> None:
    # A ``None`` value means "known to the host, no definition" (embedded
    # extension node): it resolves to () and a downstream binding onto its
    # handle synthesizes the permissive ``any`` placeholder.
    reg = _make_registry()
    nodes = [
        GraphNode(id="ext", type="flow-version@1:abc", data={}),
        GraphNode(id="b", type="dyn-passthrough@1", data={}),
    ]
    edges = [
        GraphEdge(
            id="e1", source="ext", target="b",
            source_handle="output_1", target_handle="inputs",
        )
    ]
    definitions = {**_definitions(reg, nodes), "flow-version@1:abc": None}
    out = resolve_graph_outputs(nodes, edges, definitions)
    assert out["ext"] == ()
    passed = [o for o in out["b"] if o.name == "output_1"]
    assert passed and passed[0].type_str == "any"


def test_missing_definitions_key_raises() -> None:
    nodes = [GraphNode(id="a", type="mystery@1", data={})]
    with pytest.raises(CompilationError, match="mystery@1"):
        resolve_graph_outputs(nodes, [], {})


def test_dangling_edge_endpoint_raises() -> None:
    reg = _make_registry()
    nodes = [GraphNode(id="a", type="static-src@1", data={})]
    edges = [
        GraphEdge(id="e1", source="a", target="ghost",
                  source_handle="result", target_handle="x")
    ]
    with pytest.raises(CompilationError, match="ghost"):
        resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))


def test_cycle_raises() -> None:
    reg = _make_registry()
    nodes = [
        GraphNode(id="a", type="dyn-passthrough@1", data={}),
        GraphNode(id="b", type="dyn-passthrough@1", data={}),
    ]
    edges = [
        GraphEdge(id="e1", source="a", target="b",
                  source_handle="result", target_handle="inputs"),
        GraphEdge(id="e2", source="b", target="a",
                  source_handle="result", target_handle="inputs"),
    ]
    with pytest.raises(CycleDetectionError):
        resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))


def test_hook_exception_wraps_in_compilation_error() -> None:
    reg = NodeRegistry()

    def bad_hook(ctx: ComputeOutputsContext) -> list[OutputMetadata]:
        raise ValueError("kaputt")

    @reg.node(
        "boom", version=1, name="Boom", description="Raising hook",
        compute_outputs=bad_hook,
    )
    def boom() -> Annotated[str, Output(label="X")]:
        return ""

    nodes = [GraphNode(id="a", type="boom@1", data={})]
    with pytest.raises(CompilationError, match="compute_outputs failed") as exc_info:
        resolve_graph_outputs(nodes, [], {"boom@1": reg.get("boom@1")})
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_equivalence_with_compile_node_outputs() -> None:
    # The public API and compile() step 8b must never diverge: same graph,
    # same map.
    reg = _make_registry()
    nodes = [
        GraphNode(
            id="a", type="dyn-schema@1",
            data={"fields": [{"name": "Felt", "type": "str"}]},
        ),
        GraphNode(id="b", type="dyn-passthrough@1", data={}),
    ]
    edges = [
        GraphEdge(id="e1", source="a", target="b",
                  source_handle="Felt", target_handle="inputs"),
    ]
    compiled = compile(nodes=nodes, edges=edges, registry=reg)
    standalone = resolve_graph_outputs(nodes, edges, _definitions(reg, nodes))
    assert standalone == compiled.node_outputs
