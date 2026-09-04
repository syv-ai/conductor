"""The ``compute_inputs`` hook as the compiler asks it.

A node overrides ``compute_inputs`` to shape the inputs one placement has
from the values the author typed; the inputs it adds arrive in ``run``
through ``**values``. ``resolve_node_inputs`` asks a fresh instance per
placement, ``compile`` stores the answer on ``CompiledGraph.node_inputs``,
and an edge or a consume binding may land on a handle only the hook
declared. A node with no override gets its declaration verbatim, an
extension node with no definition resolves to an empty tuple, and a hook
that raises does so where it is found. The engine validates a call
against the resolved roster, so a hook-declared input's title is what a
validation error shows.
"""

from __future__ import annotations

from typing import Annotated, Any

import pydantic
import pytest
from conductor import GraphNode, NodeRegistry, compile
from conductor.dtype import DType
from conductor.execution.engine import _format_validation_error
from conductor.graph.binding import Sources, Static
from conductor.graph.dynamic_inputs import resolve_node_inputs
from conductor.graph.model import Flow
from conductor.metadata import Input
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.widgets import Number as NumberWidget
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "compute-inputs-test-text"
    title = "Text"


class Num(DType, int):
    id = "compute-inputs-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Out")]

CUSTOMERS = Input(
    name="customers", dtype=Txt, title="Customers", widget=Textarea(title="Customers")
)


class Dyn(NodeDefinition):
    """Replaces its declared roster with the one ``customers`` input."""

    id = "dyn"
    title = "Dyn"
    description = "Hook-declared inputs"
    category = "test"

    def run(self, code: Annotated[Txt, Textarea(title="Code")] = Txt(""), **values: Any) -> Out:
        return Txt("x")

    def compute_inputs(self, declared, values):
        return (CUSTOMERS,)


class Plain(NodeDefinition):
    id = "plain"
    title = "Plain"
    description = "No shaping"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="T")] = Txt("")) -> Out:
        return text


class Src(NodeDefinition):
    id = "src"
    title = "Src"
    description = "Produces one text"
    category = "test"

    def run(self) -> Out:
        return Txt("x")


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in (Dyn, Plain, Src):
        reg.register(node_cls)
    return reg


class TestResolver:
    def test_no_hook_returns_static_inputs(self):
        got = resolve_node_inputs(node=GraphNode("n1", "plain", 1), node_def=Plain)
        assert [i.name for i in got] == ["text"]

    def test_hook_result_replaces_the_roster(self):
        got = resolve_node_inputs(node=GraphNode("n1", "dyn", 1, bindings={"code": Static(value="x")}), node_def=Dyn)
        assert [i.name for i in got] == ["customers"]

    def test_an_extension_node_resolves_to_nothing(self):
        got = resolve_node_inputs(node=GraphNode("n1", "unknown", 1), node_def=None)
        assert got == ()

    def test_a_raising_hook_raises_where_it_is_found(self):
        class Boom(NodeDefinition):
            id = "boom-in"
            title = "Boom"
            description = "Boom"
            category = "test"

            def run(self, **values: Any) -> Out:
                return Txt("x")

            def compute_inputs(self, declared, values):
                raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            resolve_node_inputs(node=GraphNode("n1", "boom-in", 1), node_def=Boom)


class TestCompileIntegration:
    def test_compiled_graph_carries_resolved_inputs(self):
        compiled = compile(Flow(nodes=[GraphNode("n1", "dyn", 1, bindings={"code": Static(value="x")})]), _registry())
        assert [i.name for i in compiled.node_inputs["n1"]] == ["customers"]

    def test_a_node_without_a_hook_gets_its_static_inputs(self):
        compiled = compile(Flow(nodes=[GraphNode("n1", "plain", 1)]), _registry())
        assert [i.name for i in compiled.node_inputs["n1"]] == ["text"]


class TestBindingsIntoHookDeclaredInputs:
    def test_an_edge_into_a_hook_declared_handle_compiles(self):
        compiled = compile(Flow(nodes=[GraphNode("a", "src", 1), GraphNode("b", "dyn", 1, bindings={"code": Static(value="x"), "customers": Sources(refs=(Ref('a', 'result'),))})]), _registry())
        assert "b" in compiled.execution_order

def test_a_hook_declared_input_uses_its_title_in_errors():
    class Rows(NodeDefinition):
        id = "rows"
        title = "Rows"
        description = "A hook-declared count"
        category = "test"

        def run(self, **values: Any) -> Out:
            return Txt("x")

        def compute_inputs(self, declared, values):
            return (
                Input(name="count", dtype=Num, title="Row count", widget=NumberWidget(title="Row count")),
            )

    class Model(pydantic.BaseModel):
        count: int

    with pytest.raises(pydantic.ValidationError) as caught:
        Model(count="not a number")
    err = caught.value

    reg = NodeRegistry()
    reg.register(Rows)
    declared = Rows.versions[1].interface.inputs
    resolved = compile(Flow(nodes=[GraphNode("n1", "rows", 1)]), reg).node_inputs["n1"]

    # The declaration has no such input, so the error reads as the bare name.
    assert "Row count" not in _format_validation_error(err, declared)
    # The resolved roster carries the title the hook gave it.
    assert "Row count" in _format_validation_error(err, resolved)
