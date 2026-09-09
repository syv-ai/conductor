"""The condition under which an output appears, derived from choice and edges."""

from dataclasses import dataclass, replace
from typing import Annotated, Any

from conductor import NodeRegistry
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.conditions import ALWAYS, Atom
from conductor.graph.model import Graph, GraphNode
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import ConnectionList, Switch, Textarea


class Txt(DType, str):
    id = "conditions-test-txt"
    title = "Text"


class Flag(DType, int):
    id = "conditions-test-flag"
    title = "Yes/No"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Branches:
    if_true: Annotated[Any, Result(title="If true", choice="branches")]
    if_false: Annotated[Any, Result(title="If false", choice="branches")]


class Gate(NodeDefinition):
    id = "gate"
    title = "If/else"
    description = "d"
    category = "test"

    def run(
        self,
        value: Annotated[Any, ConnectionList(title="Value")],
        when: Annotated[Flag, Switch(title="When")] = Flag(1),
    ) -> Branches:
        return Branches(if_true=value, if_false=SKIPPED) if when else Branches(if_true=SKIPPED, if_false=value)

    def compute_outputs(self, declared, values, arriving):
        dtype = arriving.get("value", Any)
        return tuple(replace(out, dtype=dtype) for out in declared)


class Holder(NodeDefinition):
    id = "holder"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return value


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return Txt(text.upper())


class Single(NodeDefinition):
    """The merge for scalars: a `Series[X]` input drops the skipped edges, so
    whichever branch fired is the one value it receives."""

    id = "single"
    title = "Only one"
    description = "d"
    category = "test"

    def run(self, values: Annotated[Series[Any], ConnectionList(title="Values")]) -> Annotated[Any, Result(title="The value")]:
        return values[0]

    def compute_outputs(self, declared, values, arriving):
        series = arriving.get("values")
        dtype = series.element if series is not None else Any
        return tuple(replace(out, dtype=dtype) for out in declared)


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Textarea(title="Folder")] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a")]


def _compiled(nodes):
    registry = NodeRegistry()
    for node_cls in (Gate, Holder, Upper, Docs, Single):
        registry.register(node_cls)
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def _edge(*refs):
    return Edges(refs=tuple(Ref(n, f) for n, f in refs))


def test_an_output_nothing_gates_appears_always():
    compiled = _compiled([GraphNode(id="h", type="holder", version=1, bindings={"value": Static(value="x")})])

    assert compiled.field(Ref("h", "result")).condition == ALWAYS


def test_a_branch_appears_when_its_decision_went_that_way():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static(value="x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g", "if_true")).condition == frozenset({frozenset({Atom("g", "branches", "if_true")})})
    assert compiled.field(Ref("yes", "result")).condition == frozenset({frozenset({Atom("g", "branches", "if_true")})})
    assert compiled.decisions() == {"g": {"branches": ("if_true", "if_false")}}


def test_a_merge_adds_an_alternative():
    """A `Series[X]` input drops the skipped edges, so past the
    merge the output appears when either branch fired."""
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static(value="x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("g", "if_false"))}),
        GraphNode(id="m", type="single", version=1, bindings={"values": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("m", "result")).condition == frozenset({
        frozenset({Atom("g", "branches", "if_true")}), frozenset({Atom("g", "branches", "if_false")}),
    })


def test_two_decisions_in_series_conjoin_and_a_contradiction_is_dropped():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static(value="x")}),
        GraphNode(id="g1", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="g2", type="gate", version=1, bindings={"value": _edge(("g1", "if_true"))}),
        GraphNode(id="both", type="single", version=1, bindings={"values": _edge(("g2", "if_true"), ("g1", "if_false"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g2", "if_true")).condition == frozenset({
        frozenset({Atom("g1", "branches", "if_true"), Atom("g2", "branches", "if_true")}),
    })
    assert compiled.field(Ref("both", "result")).condition == frozenset({
        frozenset({Atom("g1", "branches", "if_true"), Atom("g2", "branches", "if_true")}),
        frozenset({Atom("g1", "branches", "if_false")}),
    })


def test_a_lifted_decision_masks_rows_and_is_not_a_condition():
    """Both branches produce, each sparse, so nothing below a
    iterating decision is conditional on it — and it is not a dimension."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("docs", "result"))}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("g").iterates_on is not None
    assert compiled.field(Ref("yes", "result")).condition == ALWAYS
    assert compiled.decisions() == {}
