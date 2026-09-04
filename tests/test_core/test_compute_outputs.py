"""The ``compute_outputs`` hook as the compiler asks it.

A node overrides ``compute_outputs`` to shape the outputs one placement
has from the values the author typed. ``compile`` asks a fresh instance
per placement and stores the answer on ``CompiledGraph.node_outputs``;
a node with no override gets its declaration verbatim, and an extension
node with no definition resolves to an empty tuple. Nothing is checked:
a hook that raises does so where it is found.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import GraphNode, NodeRegistry, compile
from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.graph.binding import Static
from conductor.metadata import Output
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor.widgets import Number as NumberWidget
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "compute-outputs-test-text"
    title = "Text"


class Num(DType, int):
    id = "compute-outputs-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Out")]


class Splitter(NodeDefinition):
    """One ``slot_i`` output per ``count``."""

    id = "splitter"
    title = "Splitter"
    description = "Splits into N outputs"
    category = "test"

    def run(
        self,
        text: Annotated[Txt, Textarea(title="In")] = Txt(""),
        count: Annotated[Num, NumberWidget(title="Count")] = Num(1),
    ) -> Mapping[str, Any]:
        return {f"slot_{i}": text for i in range(count)}

    def compute_outputs(self, declared, values, arriving):
        return tuple(
            Output(name=f"slot_{i}", dtype=Txt, title=f"Slot {i}")
            for i in range(values.get("count", 1))
        )


class Noop(NodeDefinition):
    id = "noop"
    title = "Noop"
    description = "Pass-through"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Out:
        return text


def test_hook_replaces_static_outputs_in_compiled_graph():
    reg = NodeRegistry()
    reg.register(Splitter)

    compiled = compile(
        nodes=[GraphNode("n1", "splitter", 1, bindings={"count": Static(value=3)})], edges=[], registry=reg
    )

    outputs = compiled.node_outputs["n1"]
    assert tuple(o.name for o in outputs) == ("slot_0", "slot_1", "slot_2")
    assert all(o.dtype is Txt for o in outputs)


def test_no_hook_falls_back_to_static_outputs():
    reg = NodeRegistry()
    reg.register(Noop)

    compiled = compile(nodes=[GraphNode("n1", "noop", 1)], edges=[], registry=reg)

    assert compiled.node_outputs["n1"] == Noop.versions[1].interface.outputs


def test_a_mapping_return_lands_on_the_computed_roster():
    """A node whose outputs come from the hook returns them by name."""

    class Klass(NodeDefinition):
        id = "klass"
        title = "Klass"
        description = "Outputs named by the hook"
        category = "test"

        def run(self) -> Mapping[str, Any]:
            return {"custom": Num(42)}

        def compute_outputs(self, declared, values, arriving):
            return (Output(name="custom", dtype=Num, title="Custom"),)

    reg = NodeRegistry()
    reg.register(Klass)

    compiled = compile(nodes=[GraphNode("n1", "klass", 1)], edges=[], registry=reg)

    assert tuple(o.name for o in compiled.node_outputs["n1"]) == ("custom",)
    assert execute_sync(compiled)["n1"]["custom"] == 42


def test_a_raising_hook_raises_where_it_is_found():
    class Boom(NodeDefinition):
        id = "boom"
        title = "Boom"
        description = "Boom"
        category = "test"

        def run(self) -> Out:
            return Txt("x")

        def compute_outputs(self, declared, values, arriving):
            raise RuntimeError("kaboom")

    reg = NodeRegistry()
    reg.register(Boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        compile(nodes=[GraphNode("n1", "boom", 1)], edges=[], registry=reg)


def test_extension_node_alongside_hook_node():
    """An extension node with no definition coexists with a hook-driven
    registered node: the resolver tolerates the missing definition."""

    class Hooked(NodeDefinition):
        id = "hooked"
        title = "Hooked"
        description = "Hooked"
        category = "test"

        def run(self) -> Out:
            return Txt("x")

        def compute_outputs(self, declared, values, arriving):
            return (Output(name="result", dtype=Txt, title="Result"),)

    class MockExtensionResolver:
        def is_known_type(self, node_type: str) -> bool:
            return node_type.startswith("ext:")

        def create_executor(self, node_type: str) -> Any:
            return None

    reg = NodeRegistry()
    reg.register(Hooked)

    compiled = compile(
        nodes=[GraphNode("h", "hooked", 1), GraphNode("e", "ext:thing", 1)],
        edges=[],
        registry=reg,
        extension_resolver=MockExtensionResolver(),
    )

    assert tuple(o.name for o in compiled.node_outputs["h"]) == ("result",)
    assert compiled.node_outputs["e"] == ()
