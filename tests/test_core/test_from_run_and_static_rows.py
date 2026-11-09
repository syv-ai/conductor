"""What the run provides, and a static that iterates — two things a host's catalog needs of the engine."""

from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    FromRun,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeRegistry,
    Param,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor._internal.interface import read_signature
from conductor.widgets import ListWidget, TextWidget


class Txt(DType, str):
    id = "txt"
    display_title = "Txt"


class Who:
    def __init__(self, name):
        self.name = name


@node("greet", title="Greet", description="d", category="test")
def greet(text: Annotated[Txt, Param(title="T", widget=TextWidget())], who: FromRun[Who]) -> Annotated[Txt, Result(title="R")]:
    return Txt(f"{text} {who.name}")


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="T", widget=ListWidget())] = Txt("")) -> Annotated[Txt, Result(title="R")]:
    return Txt(text.upper())


@node("joinall", title="Join", description="d", category="test")
def join_all(texts: Annotated[Series[Txt], Param(title="T")] = ()) -> Annotated[Txt, Result(title="R")]:
    return Txt("-".join(texts))


def _registry():
    r = NodeRegistry()
    for c in (greet, upper_node, join_all):
        r = r.extended_with([c])
    return r


def test_a_from_run_parameter_is_a_need_not_an_input():
    iface = read_signature(greet.versions[1].run).interface
    assert [i.name for i in iface.inputs] == ["text"]
    assert iface.needs == {Who}


def test_the_run_supplies_a_need_by_type():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="g", type="greet", version=1, bindings={"text": Static(Txt("hello"))})]), _registry())
    assert compiled.is_runnable, compiled.problems
    assert compiled.results(compiled.run_sync(from_run={Who: Who("Ida")}))["g"]["result"] == "hello Ida"


def test_a_need_the_host_did_not_provide_is_refused_before_anything_runs():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="g", type="greet", version=1, bindings={"text": Static(Txt("hello"))})]), _registry())
    with pytest.raises(InvalidCallError, match="needs a Who"):
        compiled.run_sync()


def test_a_static_sequence_on_a_scalar_input_makes_the_node_iterate():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static([Txt("a"), Txt("b"), Txt("c")])}),
        GraphNode(id="j", type="joinall", version=1, bindings={"texts": From("u.result")}),
    ]), _registry())
    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["u"].runs_per_row_of == "u.text"
    assert compiled.field(Ref("u", "text")).type is Series[Txt]
    assert compiled.run_plan.nodes["u"].inputs["text"].rows_of == "u.text"
    results = compiled.results(compiled.run_sync())
    assert list(results["u"]["result"]) == ["A", "B", "C"]
    assert results["u"]["result"].rows == ((0,), (1,), (2,))
    assert results["j"]["result"] == "A-B-C"


def test_an_empty_static_sequence_is_an_empty_series():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static([])}),
        GraphNode(id="j", type="joinall", version=1, bindings={"texts": From("u.result")}),
    ]), _registry())
    results = compiled.results(compiled.run_sync())
    assert list(results["u"]["result"]) == []
    assert results["j"]["result"] == ""


def test_a_static_string_is_one_value_not_a_sequence():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="u", type="upper", version=1, bindings={"text": Static(Txt("abc"))})]), _registry())
    assert compiled.run_plan.nodes["u"].runs_per_row_of is None
    assert compiled.results(compiled.run_sync())["u"]["result"] == "ABC"
