"""What the run provides, and a static that iterates — two things a host's catalog needs of the engine."""

from typing import Annotated

import pytest
from conductor import FromRun, NodeRegistry
from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.interface import Interface
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import ConnectionList, List, Text


class Txt(DType, str):
    id = "txt"
    title = "Txt"


class Who:
    def __init__(self, name):
        self.name = name


class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Text(title="T")], who: Annotated[Who, FromRun()]) -> Annotated[Txt, Result(title="R")]:
        return Txt(f"{text} {who.name}")


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, List(title="T")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return Txt(text.upper())


class JoinAll(NodeDefinition):
    id = "joinall"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="T")] = ()) -> Annotated[Txt, Result(title="R")]:
        return Txt("-".join(texts))


def _registry():
    r = NodeRegistry()
    for c in (Greet, Upper, JoinAll):
        r.register(c)
    return r


def test_a_from_run_parameter_is_a_need_not_an_input():
    iface = Interface.of(Greet.run)
    assert [i.name for i in iface.inputs] == ["text"]
    assert iface.needs == {"who": Who}


def test_the_run_supplies_a_need_by_type():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="g", type="greet", version=1, bindings={"text": Static(value=Txt("hello"))})]), _registry())
    assert compiled.is_runnable, compiled.problems
    assert execute_sync(compiled, from_run={Who: Who("Ida")})["g"]["result"] == "hello Ida"


def test_a_need_the_host_did_not_provide_is_refused_before_anything_runs():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="g", type="greet", version=1, bindings={"text": Static(value=Txt("hello"))})]), _registry())
    with pytest.raises(TypeError, match="needs a Who"):
        execute_sync(compiled)


def test_a_static_sequence_on_a_scalar_input_makes_the_node_iterate():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static(value=[Txt("a"), Txt("b"), Txt("c")])}),
        GraphNode(id="j", type="joinall", version=1, bindings={"texts": Edges(refs=(Ref("u", "result"),))}),
    ]), _registry())
    assert compiled.is_runnable, compiled.problems
    assert compiled.node("u").iterates_on == Index("u.text")
    assert compiled.field(Ref("u", "text")).type is Series[Txt]
    assert compiled.field(Ref("u", "result")).index == Index("u.text")
    results = execute_sync(compiled)
    assert list(results["u"]["result"]) == ["A", "B", "C"]
    assert results["u"]["result"].rows == ((0,), (1,), (2,))
    assert results["j"]["result"] == "A-B-C"


def test_an_empty_static_sequence_is_an_empty_series():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static(value=[])}),
        GraphNode(id="j", type="joinall", version=1, bindings={"texts": Edges(refs=(Ref("u", "result"),))}),
    ]), _registry())
    results = execute_sync(compiled)
    assert list(results["u"]["result"]) == []
    assert results["j"]["result"] == ""


def test_a_static_string_is_one_value_not_a_sequence():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="u", type="upper", version=1, bindings={"text": Static(value=Txt("abc"))})]), _registry())
    assert compiled.node("u").iterates_on is None
    assert execute_sync(compiled)["u"]["result"] == "ABC"
