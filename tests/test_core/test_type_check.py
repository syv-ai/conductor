"""Compile-time type checking for edge connections."""

from typing import Annotated

import pytest

from flowengine import GraphEdge, GraphNode, NodeRegistry, compile
from flowengine.errors import CompilationError
from flowengine.types import NodeCategory
from flowengine.widgets import Checkbox, ConnectionList, Dropdown, Output, Range, Text, Textarea


@pytest.fixture
def typed_registry():
    reg = NodeRegistry()

    @reg.node("str-out", version=1, name="Text Out", description="Outputs str")
    def str_out(v: Annotated[str, Text(label="V")] = "hi") -> Annotated[str, Output(label="Out")]:
        return v

    @reg.node("int-out", version=1, name="Int Out", description="Outputs int")
    def int_out(v: Annotated[int, Text(label="V")] = 0) -> Annotated[int, Output(label="Out")]:
        return v

    @reg.node("float-out", version=1, name="Float Out", description="Outputs float")
    def float_out(v: Annotated[float, Range(label="V")] = 0.0) -> Annotated[float, Output(label="Out")]:
        return v

    @reg.node("bool-out", version=1, name="Bool Out", description="Outputs bool")
    def bool_out(v: Annotated[bool, Checkbox(label="V")] = False) -> Annotated[bool, Output(label="Out")]:
        return v

    @reg.node("list-str-out", version=1, name="List Out", description="Outputs list[str]")
    def list_str_out() -> Annotated[list[str], Output(label="Out")]:
        return ["a", "b"]

    @reg.node("str-in", version=1, name="Text In", description="Accepts str")
    def str_in(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
        return text

    @reg.node("int-in", version=1, name="Int In", description="Accepts int")
    def int_in(num: Annotated[int, Text(label="In")]) -> Annotated[int, Output(label="Out")]:
        return num

    @reg.node("float-in", version=1, name="Float In", description="Accepts float")
    def float_in(num: Annotated[float, Text(label="In")]) -> Annotated[float, Output(label="Out")]:
        return num

    @reg.node("list-in", version=1, name="List In", description="Accepts list[str]")
    def list_in(items: Annotated[list[str], Text(label="In")]) -> Annotated[list[str], Output(label="Out")]:
        return items

    @reg.node("bool-in", version=1, name="Bool In", description="Accepts bool")
    def bool_in(flag: Annotated[bool, Checkbox(label="In")]) -> Annotated[bool, Output(label="Out")]:
        return flag

    @reg.node("conn-list-in", version=1, name="CL In", description="ConnectionList input")
    def conn_list_in(
        items: Annotated[dict[str, float], ConnectionList(label="Numbers")],
    ) -> Annotated[float, Output(label="Out")]:
        return sum(items.values())

    return reg


class TestCompatibleTypes:
    """Connections that should pass without warnings."""

    def test_str_to_str(self, typed_registry):
        c = compile(
            nodes=[GraphNode("a", "str-out@1", {}), GraphNode("b", "str-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "text")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_int_to_float(self, typed_registry):
        """Numeric types are interchangeable."""
        c = compile(
            nodes=[GraphNode("a", "int-out@1", {}), GraphNode("b", "float-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_float_to_int(self, typed_registry):
        c = compile(
            nodes=[GraphNode("a", "float-out@1", {}), GraphNode("b", "int-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_int_to_str(self, typed_registry):
        """Everything can become a string."""
        c = compile(
            nodes=[GraphNode("a", "int-out@1", {}), GraphNode("b", "str-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "text")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_str_to_list_str(self, typed_registry):
        """Single value auto-wraps into list."""
        c = compile(
            nodes=[GraphNode("a", "str-out@1", {}), GraphNode("b", "list-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "items")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_list_str_to_str(self, typed_registry):
        """list[str] into str — lossy but allowed."""
        c = compile(
            nodes=[GraphNode("a", "list-str-out@1", {}), GraphNode("b", "str-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "text")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0

    def test_anything_to_connection_list(self, typed_registry):
        """ConnectionList accepts any type."""
        c = compile(
            nodes=[
                GraphNode("a", "str-out@1", {}),
                GraphNode("b", "int-out@1", {}),
                GraphNode("c", "conn-list-in@1", None),
            ],
            edges=[
                GraphEdge("e1", "a", "c", "result", "items"),
                GraphEdge("e2", "b", "c", "result", "items"),
            ],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 0


class TestIncompatibleTypes:
    """Connections that should produce warnings."""

    def test_str_to_int_warns(self, typed_registry):
        """str -> int is not guaranteed to work."""
        c = compile(
            nodes=[GraphNode("a", "str-out@1", {}), GraphNode("b", "int-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 1
        w = c.type_warnings[0]
        assert w.source_type == "str"
        assert w.target_type == "int"
        assert "Text Out" in w.message
        assert "Int In" in w.message

    def test_str_to_bool_warns(self, typed_registry):
        c = compile(
            nodes=[GraphNode("a", "str-out@1", {}), GraphNode("b", "bool-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "flag")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 1

    def test_bool_to_int_warns(self, typed_registry):
        c = compile(
            nodes=[GraphNode("a", "bool-out@1", {}), GraphNode("b", "int-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 1

    def test_multiple_warnings(self, typed_registry):
        """Multiple bad edges produce multiple warnings."""
        c = compile(
            nodes=[
                GraphNode("a", "str-out@1", {}),
                GraphNode("b", "bool-out@1", {}),
                GraphNode("c", "int-in@1", None),
                GraphNode("d", "float-in@1", None),
            ],
            edges=[
                GraphEdge("e1", "a", "c", "result", "num"),  # str -> int
                GraphEdge("e2", "b", "d", "result", "num"),  # bool -> float
            ],
            registry=typed_registry,
        )
        assert len(c.type_warnings) == 2


class TestStrictMode:
    def test_strict_raises_on_mismatch(self, typed_registry):
        with pytest.raises(CompilationError, match="Type errors"):
            compile(
                nodes=[GraphNode("a", "str-out@1", {}), GraphNode("b", "int-in@1", None)],
                edges=[GraphEdge("e1", "a", "b", "result", "num")],
                registry=typed_registry,
                strict_types=True,
            )

    def test_strict_passes_on_compatible(self, typed_registry):
        """No error when types are compatible."""
        c = compile(
            nodes=[GraphNode("a", "int-out@1", {}), GraphNode("b", "float-in@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "num")],
            registry=typed_registry,
            strict_types=True,
        )
        assert len(c.type_warnings) == 0


class TestWarningDetails:
    def test_warning_has_all_fields(self, typed_registry):
        c = compile(
            nodes=[GraphNode("n1", "str-out@1", {}), GraphNode("n2", "int-in@1", None)],
            edges=[GraphEdge("e1", "n1", "n2", "result", "num")],
            registry=typed_registry,
        )
        w = c.type_warnings[0]
        assert w.edge_id == "e1"
        assert w.source_node == "n1"
        assert w.source_output == "result"
        assert w.source_type == "str"
        assert w.target_node == "n2"
        assert w.target_input == "num"
        assert w.target_type == "int"
        assert isinstance(w.message, str)
