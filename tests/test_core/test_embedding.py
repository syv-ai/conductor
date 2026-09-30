"""A node whose version is a compiled graph: placed as one node, then lifted under the row it runs on.

A host compiles each graph version once and hands the ``CompiledGraph``
to conductor as the version. The outer compile treats the placed graph as
a plain node with the compiled graph's interface, which decides the row it
runs on; the lift (``graph_as_placed``) then puts the graph's own records
under that row. What the engine reads must come out exactly as inlining
the raw graph gave it: the oracle below checks every scenario, field by
field, against the values that path gave, pinned in
``fixtures/embedded_expected.json``, and a run of some of them against
``fixtures/embedded_expected_results.json``.

Which scenario pins which lift rule:

1. an index renamed and hung under the row: ``born_inside``, ``nested_iterating``;
2. a value that ran once becomes a series on the row: ``entering_scalar``, ``gather_two_inside``;
3. ``Whole`` → ``Group``, a ``Gather`` of single values → ``Group``: ``born_inside``, ``gather_two_inside``;
4. a crossing takes the outer walk's record: ``two_crossings``, ``series_field``, ``gathered_into_the_placed_node``;
5. an output read outside takes the inner field's index: ``born_inside_exposed``, ``read_from_outside``, ``born_under_a_row_inside``;
6. conditions put in place: every ``cond_`` scenario.

A value typed on the placed node's input is not lifted: the graph is
compiled again with it, as if typed inside (``value_on_the_placed_node``,
``many_values_on_the_placed_node``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar

import pytest
from conductor import NodeRegistry, run_sync
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import Refuses
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Param, Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Series
from conductor.widgets import Switch, Textarea

FIXTURE = Path(__file__).parent / "fixtures" / "embedded_expected.json"
RESULTS = Path(__file__).parent / "fixtures" / "embedded_expected_results.json"


class Txt(DType, str):
    id = "embedding-test-txt"
    title = "Text"


class Flag(DType, int):
    id = "embedding-test-flag"
    title = "Yes/No"


Out = Annotated[Txt, Result(title="Result")]


class Holder(NodeDefinition):
    id = "holder"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return value


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper case"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Join(NodeDefinition):
    """A reduction."""

    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a"), Txt("b")]


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


class Pair(NodeDefinition):
    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"

    def run(
        self,
        a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
        b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
    ) -> Out:
        return Txt(f"{a}:{b}")


class Need(NodeDefinition):
    """A required input with a type and no default."""

    id = "need"
    title = "Need"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())]) -> Out:
        return text


class Both(NodeDefinition):
    """Two required inputs with a type and no default."""

    id = "both"
    title = "Both"
    description = "d"
    category = "test"

    def run(
        self,
        a: Annotated[Txt, Param(title="A", widget=Textarea())],
        b: Annotated[Txt, Param(title="B", widget=Textarea())],
    ) -> Out:
        return Txt(f"{a}:{b}")


class Picky(NodeDefinition):
    """Refuses the mode ``bad`` in its hook."""

    id = "picky"
    title = "Picky"
    description = "d"
    category = "test"

    def run(self, mode: Annotated[Txt, Param(title="Mode", widget=Textarea())] = Txt("ok")) -> Out:
        return mode

    def compute_inputs(self, declared, values):
        if values.get("mode") == "bad":
            raise Refuses("picky_refuses", "Not that mode.")
        return declared


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
        value: Annotated[Any, Param(title="Value")],
        when: Annotated[Flag, Param(title="When", widget=Switch())] = Flag(1),
    ) -> Branches:
        return Branches(if_true=value, if_false=SKIPPED) if when else Branches(if_true=SKIPPED, if_false=value)

    def compute_outputs(self, declared, values, arriving):
        dtype = arriving.get("value", Any)
        return tuple(out.model_copy(update={"dtype": dtype}) for out in declared)


class Single(NodeDefinition):
    """The merge for scalars: whichever branch fired is the one value it receives."""

    id = "single"
    title = "Only one"
    description = "d"
    category = "test"

    def run(self, values: Annotated[Series[Any], Param(title="Values")]) -> Annotated[Any, Result(title="The value")]:
        return values[0]

    def compute_outputs(self, declared, values, arriving):
        series = arriving.get("values")
        dtype = series.element if series is not None else Any
        return tuple(out.model_copy(update={"dtype": dtype}) for out in declared)


PLAIN = (Holder, Upper, Join, Docs, Lines, Pair, Need, Both, Picky, Gate, Single)
N = GraphNode


# -- the scenarios ----------------------------------------------------------------------
#
# Each is its inner graphs, innermost first, and the outer graph. Each inner
# graph becomes a definition whose version is the graph compiled on its own,
# against the registry of the ones before it.


@dataclass(frozen=True)
class Scenario:
    graphs: tuple[tuple[str, tuple[GraphNode, ...]], ...]
    nodes: tuple[GraphNode, ...]


def _inner_graph():
    return (
        N(id="holder", type="holder", version=1, title="Text", bindings={"value": Static("inner")}),
        N(id="up", type="upper", version=1, title="Upper", bindings={"text": From("holder.result")}),
        N(id="join", type="join", version=1, title="Join", bindings={"texts": From("up.result")}),
    )


def _graph_like_a_host_stores(width=8):
    """A graph like the ones a host stores: a holder for its input, a chain, a split into lines, a reduction."""
    nodes = [N(id="holder", type="holder", version=1, title="Text", bindings={"value": Static("x")})]
    before = "holder"
    for i in range(width):
        nodes.append(N(id=f"s{i}", type="upper", version=1, bindings={"text": From(f"{before}.result")}))
        before = f"s{i}"
    nodes.append(N(id="lines", type="lines", version=1, bindings={"text": From(f"{before}.result")}))
    nodes.append(N(id="up", type="upper", version=1, bindings={"text": From("lines.result")}))
    nodes.append(N(id="join", type="join", version=1, bindings={"texts": From("up.result")}))
    return tuple(nodes)


def _branching():
    """holder → gate → an upper on the true branch, and whichever branch fired."""
    return (
        N(id="holder", type="holder", version=1, bindings={"value": Static("x")}),
        N(id="gate", type="gate", version=1, bindings={"value": From("holder.result")}),
        N(id="up", type="upper", version=1, bindings={"text": From("gate.if_true")}),
        N(id="single", type="single", version=1, bindings={"values": From("gate.if_true", "gate.if_false")}),
    )


def _middle_branching():
    return (
        N(id="h", type="holder", version=1, bindings={"value": Static("m")}),
        N(id="g", type="gate", version=1, bindings={"value": From("h.result")}),
        N(id="b", type="branchy", version=1, bindings={"holder.value": From("g.if_true")}),
        N(id="s", type="single", version=1, bindings={"values": From("b.up.result", "g.if_false")}),
    )


def _docs():
    return N(id="docs", type="docs", version=1)


def _placed_many(k):
    return Scenario(
        graphs=(("stored-graph", _graph_like_a_host_stores()),),
        nodes=(_docs(), *(N(id=f"e{i}", type="stored-graph", version=1, bindings={"holder.value": From("docs.result")}) for i in range(k))),
    )


def _nested(depth):
    graphs = [("level0", _graph_like_a_host_stores())]
    for level in range(1, depth):
        graphs.append((f"level{level}", (
            N(id="holder", type="holder", version=1, bindings={"value": Static("x")}),
            N(id="inner", type=f"level{level - 1}", version=1, bindings={"holder.value": From("holder.result")}),
            N(id="lines", type="lines", version=1, bindings={"text": From("inner.join.result")}),
            N(id="up", type="upper", version=1, bindings={"text": From("lines.result")}),
            N(id="join", type="join", version=1, bindings={"texts": From("up.result")}),
        )))
    return Scenario(
        graphs=tuple(graphs),
        nodes=(_docs(), N(id="top", type=f"level{depth - 1}", version=1, bindings={"holder.value": From("docs.result")})),
    )


SCENARIOS: dict[str, Scenario] = {
    "runs_once": Scenario((("inner-graph", _inner_graph()),), (N(id="emb", type="inner-graph", version=1),)),
    "runs_once_fed": Scenario((("inner-graph", _inner_graph()),), (
        N(id="h", type="holder", version=1, bindings={"value": Static("z")}),
        N(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("h.result")}),
    )),
    "entering_scalar": Scenario((("inner-graph", _inner_graph()),), (
        _docs(), N(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("docs.result")}),
    )),
    "read_from_outside": Scenario((("inner-graph", _inner_graph()),), (
        _docs(),
        N(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("docs.result")}),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.join.result")}),
    )),
    "born_inside": Scenario((("splitter", (
        N(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
        N(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
        N(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
    )),), (_docs(), N(id="emb", type="splitter", version=1, bindings={"holder.value": From("docs.result")}))),
    "born_inside_unfed": Scenario((("splitter", (
        N(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
        N(id="entered", type="holder", version=1),
        N(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
        N(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
    )),), (_docs(), N(id="emb", type="splitter", version=1, bindings={"entered.value": From("docs.result")}))),
    "born_inside_exposed": Scenario((("splitter", (
        N(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
        N(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
    )),), (
        _docs(),
        N(id="emb", type="splitter", version=1, bindings={"holder.value": From("docs.result")}),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.lines.result")}),
    )),
    "two_crossings": Scenario((("pair-graph", (
        N(id="a", type="holder", version=1), N(id="b", type="holder", version=1),
        N(id="ua", type="upper", version=1, bindings={"text": From("a.result")}),
    )),), (
        _docs(),
        N(id="lines", type="lines", version=1, bindings={"text": From("docs.result")}),
        N(id="emb", type="pair-graph", version=1, bindings={"a.value": From("docs.result"), "b.value": From("lines.result")}),
    )),
    "series_field": Scenario((("joiner", (N(id="join", type="join", version=1),)),), (
        _docs(), N(id="emb", type="joiner", version=1, bindings={"join.texts": From("docs.result")}),
    )),
    "gathered_into_the_placed_node": Scenario((("joiner", (N(id="join", type="join", version=1),)),), (
        N(id="h1", type="holder", version=1, bindings={"value": Static("a")}),
        N(id="h2", type="holder", version=1, bindings={"value": Static("b")}),
        N(id="emb", type="joiner", version=1, bindings={"join.texts": From("h1.result", "h2.result")}),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.join.result")}),
    )),
    "born_under_a_row_inside": Scenario((("rows-inside", (
        _docs(),
        N(id="lines", type="lines", version=1, bindings={"text": From("docs.result")}),
    )),), (
        N(id="emb", type="rows-inside", version=1),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.lines.result")}),
    )),
    "per_row_output_read_outside": Scenario((("splitter", (
        N(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
        N(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
        N(id="up", type="upper", version=1, bindings={"text": From("lines.result")}),
    )),), (
        _docs(),
        N(id="emb", type="splitter", version=1, bindings={"holder.value": From("docs.result")}),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.up.result")}),
    )),
    "value_and_edge_per_row": Scenario((("pair-graph", (
        N(id="a", type="holder", version=1), N(id="b", type="holder", version=1),
        N(id="ua", type="upper", version=1, bindings={"text": From("a.result")}),
        N(id="lines", type="lines", version=1, bindings={"text": From("b.result")}),
        N(id="p", type="pair", version=1, bindings={"a": From("ua.result"), "b": From("lines.result")}),
    )),), (
        _docs(),
        N(id="emb", type="pair-graph", version=1, bindings={"a.value": From("docs.result"), "b.value": Static("x\ny")}),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.p.result")}),
    )),
    "required_input_filled_inside": Scenario((("filled", (
        N(id="n", type="need", version=1, bindings={"text": Static("inside")}),
        N(id="u", type="upper", version=1, bindings={"text": From("n.result")}),
    )),), (
        N(id="emb", type="filled", version=1),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.u.result")}),
    )),
    "edge_into_a_value_filled_inside": Scenario((("filled", (
        N(id="n", type="need", version=1, bindings={"text": Static("inside")}),
        N(id="u", type="upper", version=1, bindings={"text": From("n.result")}),
    )),), (
        N(id="h", type="holder", version=1, bindings={"value": Static("outside")}),
        N(id="emb", type="filled", version=1, bindings={"n.text": From("h.result")}),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.u.result")}),
    )),
    "edge_into_a_list_filled_inside": Scenario((("listy", (
        N(id="n", type="upper", version=1, bindings={"text": Static(["p", "q"])}),
        N(id="j", type="join", version=1, bindings={"texts": From("n.result")}),
    )),), (
        N(id="h", type="holder", version=1, bindings={"value": Static("outside")}),
        N(id="emb", type="listy", version=1, bindings={"n.text": From("h.result")}),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.j.result")}),
    )),
    "typed_list_inside": Scenario((("typed-inside", (
        N(id="h", type="upper", version=1),
        N(id="t", type="pair", version=1, bindings={"a": From("h.result"), "b": Static(["p", "q"])}),
        N(id="j", type="join", version=1, bindings={"texts": From("t.result")}),
    )),), (_docs(), N(id="emb", type="typed-inside", version=1, bindings={"h.text": From("docs.result")}))),
    "gather_two_inside": Scenario((("two", (
        N(id="holder", type="holder", version=1, bindings={"value": Static("x")}),
        N(id="up", type="upper", version=1, bindings={"text": From("holder.result")}),
        N(id="join", type="join", version=1, bindings={"texts": From("holder.result", "up.result")}),
    )),), (_docs(), N(id="emb", type="two", version=1, bindings={"holder.value": From("docs.result")}))),
    "placed_3": _placed_many(3),
    "nested_2": _nested(2),
    "nested_4": _nested(4),
    "nested_iterating": Scenario((
        ("leaf", _graph_like_a_host_stores(3)),
        ("middle", (
            N(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
            N(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
            N(id="leafy", type="leaf", version=1, bindings={"holder.value": From("lines.result")}),
            N(id="join", type="join", version=1, bindings={"texts": From("leafy.join.result")}),
        )),
    ), (
        _docs(),
        N(id="top", type="middle", version=1, bindings={"holder.value": From("docs.result")}),
        N(id="after", type="join", version=1, bindings={"texts": From("top.join.result")}),
    )),
    "required_input_left_empty": Scenario((("needy", (
        N(id="n", type="need", version=1),
        N(id="u", type="upper", version=1, bindings={"text": From("n.result")}),
        N(id="join", type="join", version=1, bindings={"texts": From("u.result")}),
    )),), (_docs(), N(id="emb", type="needy", version=1, bindings={"n.text": From("docs.result")}))),
    "value_on_the_placed_node": Scenario((("inner-graph", _inner_graph()),), (
        N(id="emb", type="inner-graph", version=1, bindings={"holder.value": Static("set outside")}),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.join.result")}),
    )),
    "many_values_on_the_placed_node": Scenario((("inner-graph", _inner_graph()),), (
        N(id="emb", type="inner-graph", version=1, bindings={"holder.value": Static(["a", "b"])}),
        N(id="after", type="join", version=1, bindings={"texts": From("emb.join.result")}),
    )),
    "cond_inner_choice_once": Scenario((("branchy", _branching()),), (
        N(id="emb", type="branchy", version=1),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.up.result")}),
    )),
    "cond_inner_choice_per_row": Scenario((("branchy", _branching()),), (
        _docs(),
        N(id="emb", type="branchy", version=1, bindings={"holder.value": From("docs.result")}),
        N(id="after", type="single", version=1, bindings={"values": From("emb.up.result")}),
    )),
    "cond_outer_gate_into_graph": Scenario((("branchy", _branching()),), (
        N(id="h", type="holder", version=1, bindings={"value": Static("y")}),
        N(id="og", type="gate", version=1, bindings={"value": From("h.result")}),
        N(id="emb", type="branchy", version=1, bindings={"holder.value": From("og.if_false")}),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.single.result")}),
    )),
    "cond_merge_inside": Scenario((("merge", (
        N(id="a", type="holder", version=1, bindings={"value": Static("x")}),
        N(id="b", type="holder", version=1, bindings={"value": Static("z")}),
        N(id="g", type="gate", version=1, bindings={"value": From("b.result")}),
        N(id="single", type="single", version=1, bindings={"values": From("a.result", "g.if_true")}),
    )),), (
        N(id="h", type="holder", version=1, bindings={"value": Static("y")}),
        N(id="og", type="gate", version=1, bindings={"value": From("h.result")}),
        N(id="emb", type="merge", version=1, bindings={"a.value": From("og.if_true")}),
    )),
    "cond_nested_once": Scenario((("branchy", _branching()), ("mid", _middle_branching())), (
        N(id="top", type="mid", version=1),
        N(id="after", type="upper", version=1, bindings={"text": From("top.s.result")}),
    )),
    "cond_nested_per_row": Scenario((("branchy", _branching()), ("mid", _middle_branching())), (
        _docs(),
        N(id="top", type="mid", version=1, bindings={"h.value": From("docs.result")}),
        N(id="after", type="single", version=1, bindings={"values": From("top.s.result")}),
    )),
}


# -- the builder -----------------------------------------------------------------------


def _plain_registry() -> NodeRegistry:
    registry = NodeRegistry()
    for node_cls in PLAIN:
        registry.register(node_cls)
    return registry


def _definition(type_id: str, version: Any) -> type[NodeDefinition]:
    class Embedded(NodeDefinition):
        id = type_id
        title = f"Graph {type_id}"
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, Any]] = {1: version}

    return Embedded


def _registry(scenario: Scenario) -> NodeRegistry:
    """The plain nodes, then each inner graph compiled once and offered as a definition, innermost first."""
    registry = _plain_registry()
    for type_id, nodes in scenario.graphs:
        registry = registry.extended_with({type_id: _definition(type_id, CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry))})
    return registry


def _compiled(name: str) -> CompiledGraph:
    scenario = SCENARIOS[name]
    return CompiledGraph.from_graph(Graph(nodes=list(scenario.nodes)), _registry(scenario))


# -- the oracle --------------------------------------------------------------------------


def _condition(condition) -> list[list[str]]:
    """A condition in one order, so two equal ones write the same."""
    return sorted(sorted(repr(atom) for atom in alternative) for alternative in condition)


def _what_the_engine_reads(compiled: CompiledGraph) -> dict[str, Any]:
    """Every run node, in order: the row it runs on, the values typed on it,
    and each field's type, rows, receipt, binding and condition, as reprs.
    What an editor draws is here too, so a value that no longer holds can't
    hide behind a run that reads right."""
    assert compiled.is_runnable, compiled.problems
    nodes: dict[str, Any] = {}
    for node_id in compiled.execution_order:
        node = compiled.node(node_id)
        inputs = {inp.name for inp in node.interface.inputs}
        fields: dict[str, Any] = {}
        for declared in (*node.interface.inputs, *node.interface.outputs):
            found = compiled.field(Ref(node_id, declared.name))
            fields[declared.name] = {
                "type": repr(found.type),
                "index": repr(found.index),
                **({
                    "receives": repr(found.receives), "binding": repr(found.binding), "listed": found.listed,
                } if declared.name in inputs else {"condition": _condition(found.condition)}),
            }
        nodes[node_id] = {
            "iterates_on": repr(node.iterates_on),
            "statics": repr(dict(sorted(node.statics.items()))),
            "fields": fields,
        }
    return {"order": list(compiled.execution_order), "nodes": nodes}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_compiled_graph_placed_reads_exactly_as_the_raw_graph_inlined(name):
    assert _what_the_engine_reads(_compiled(name)) == json.loads(FIXTURE.read_text())[name]


def test_the_pinned_values_are_the_scenarios():
    assert set(json.loads(FIXTURE.read_text())) == set(SCENARIOS)



RUN = (
    "entering_scalar", "born_inside_exposed", "gathered_into_the_placed_node", "many_values_on_the_placed_node",
    "nested_iterating", "cond_outer_gate_into_graph", "cond_nested_per_row",
    "per_row_output_read_outside", "value_and_edge_per_row", "required_input_filled_inside",
    "edge_into_a_value_filled_inside", "edge_into_a_list_filled_inside",
)


def _results(compiled: CompiledGraph) -> dict[str, dict[str, str]]:
    """What a run produced, each value as its repr."""
    return {
        node_id: {name: repr(value) for name, value in outputs.items()}
        for node_id, outputs in run_sync(compiled).state.results(compiled).items()
    }


@pytest.mark.parametrize("name", RUN)
def test_a_compiled_graph_placed_runs_to_the_same_results(name):
    assert _results(_compiled(name)) == json.loads(RESULTS.read_text())[name]


def test_the_pinned_results_are_the_run_scenarios():
    assert set(json.loads(RESULTS.read_text())) == set(RUN)

# -- the placed node ---------------------------------------------------------------------


def test_the_placed_node_is_a_graph_whose_version_is_the_compiled_graph():
    compiled = _compiled("read_from_outside")
    node = compiled.node("emb")

    assert (node.state, node.kind) == ("ready", "graph")
    assert isinstance(node.version, CompiledGraph)
    assert compiled.node("emb/join").embedded_in == "emb"
    assert compiled.field(Ref("emb", "join.result")) is compiled.field(Ref("emb/join", "result"))


def test_a_value_set_on_the_placed_node_is_read_by_the_inner_node():
    compiled = _compiled("value_on_the_placed_node")

    assert compiled.node("emb/holder").statics == {"value": Txt("set outside")}
    assert compiled.node("emb").statics == {"holder.value": Txt("set outside")}
    assert compiled.field(Ref("emb/holder", "value")).binding == Static("set outside")


def _broken_graph_registry():
    broken = CompiledGraph.from_graph(Graph(nodes=[
        N(id="a", type="holder", version=1),
        N(id="b", type="upper", version=1, bindings={"text": From("a.nope")}),
    ]), _plain_registry())
    return broken, _plain_registry().extended_with({"broken": _definition("broken", broken)})


def test_a_broken_graph_is_one_problem_on_the_placed_node():
    """The graph's own problems stay on the graph, in its own words; where
    it is placed, the author sees one problem saying the graph needs fixing."""
    broken, registry = _broken_graph_registry()
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        N(id="emb", type="broken", version=1),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.a.result")}),
    ]), registry)

    (found,) = compiled.problems
    assert (found.code, found.fatal, found.node_id, found.field) == ("embedded_graph_broken", True, "emb", None)
    assert found.details == {"graph": "Graph broken", "problems": 1}
    assert compiled.node("emb").state == "wiring_failed"
    assert compiled.node("after").state == "wiring_failed"
    assert compiled.node("emb").version is broken
    assert [p.code for p in compiled.node("emb").version.problems] == ["unknown_ref_output"]


def test_an_outer_edge_may_only_read_what_the_graph_offers():
    """``up`` feeds ``join`` inside, so the graph does not offer its output."""
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        N(id="emb", type="inner-graph", version=1),
        N(id="after", type="upper", version=1, bindings={"text": From("emb.up.result")}),
    ]), _registry(SCENARIOS["runs_once"]))

    assert [(p.code, p.node_id) for p in compiled.problems] == [("unknown_ref_output", "after")]



def test_two_series_born_by_different_nodes_inside_do_not_line_up_outside():
    """Each series output of a placed graph sits on the index of the node
    inside that births it, so two unrelated ones are ``misaligned`` where they
    meet, exactly as when the raw graph was inlined."""
    scenario = Scenario((("two-series", (
        N(id="a", type="lines", version=1, bindings={"text": Static("x\ny")}),
        N(id="b", type="lines", version=1, bindings={"text": Static("p\nq")}),
    )),), (
        N(id="emb", type="two-series", version=1),
        N(id="after", type="pair", version=1, bindings={"a": From("emb.a.result"), "b": From("emb.b.result")}),
    ))
    problems = CompiledGraph.from_graph(Graph(nodes=list(scenario.nodes)), _registry(scenario)).problems

    assert [(p.code, p.node_id) for p in problems] == [("misaligned", "after")]



def test_a_graph_that_cannot_run_is_refused_even_when_every_node_is_ready():
    """A value no type can read is fatal without breaking its node."""
    graph = CompiledGraph.from_graph(Graph(nodes=[
        N(id="h", type="holder", version=1),
        N(id="g", type="gate", version=1, bindings={"value": From("h.result"), "when": Static("not a number")}),
    ]), _plain_registry())
    assert not graph.is_runnable
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[N(id="emb", type="bad", version=1)]), _plain_registry().extended_with({"bad": _definition("bad", graph)}),
    )

    assert [(p.code, p.details) for p in compiled.problems] == [("embedded_graph_broken", {"graph": "Graph bad", "problems": 1})]


def test_a_graph_whose_required_input_is_left_empty_can_be_placed():
    """The value arrives from outside, so the one fatal problem inside is not the graph's fault."""
    compiled = _compiled("required_input_left_empty")

    assert compiled.is_runnable, compiled.problems


def _placed_alone(inner: tuple[GraphNode, ...], **bindings: Any) -> CompiledGraph:
    graph = CompiledGraph.from_graph(Graph(nodes=list(inner)), _plain_registry())
    registry = _plain_registry().extended_with({"g": _definition("g", graph)})
    return CompiledGraph.from_graph(Graph(nodes=[N(id="emb", type="g", version=1, bindings=bindings)]), registry)


def test_a_graph_with_an_empty_input_it_does_not_offer_cannot_be_placed():
    """Only an input the graph offers can be filled where it is placed. A
    locked one, or one on a node that also has an edge in, stays empty."""
    locked = _placed_alone((N(id="n", type="need", version=1, locked=("text",)),))
    fed_beside = _placed_alone((
        N(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        N(id="p", type="both", version=1, bindings={"a": From("h.result")}),
    ))

    for compiled in (locked, fed_beside):
        assert [(p.code, p.details) for p in compiled.problems] == [("embedded_graph_broken", {"graph": "Graph g", "problems": 1})]


def test_a_value_typed_on_a_placed_graph_is_checked_like_the_graph():
    """Compiled again with the value, the graph can refuse to be placed."""
    compiled = _placed_alone((N(id="p", type="picky", version=1),), **{"p.mode": Static("bad")})

    assert [p.code for p in compiled.problems] == ["embedded_graph_broken"]


def test_a_broken_graph_with_a_value_typed_on_it_is_one_problem():
    broken, registry = _broken_graph_registry()
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        N(id="emb", type="broken", version=1, bindings={"a.value": Static("v")}),
    ]), registry)

    assert [p.code for p in compiled.problems] == ["embedded_graph_broken"]


def test_a_graph_the_walk_leaves_out_is_one_node_with_nothing_inside():
    """Its edge in is broken, or it can't be placed: either way it stays one
    ``graph`` node, saying why it is not ready, and nothing inside it is a
    node of this graph. What is inside is read through its version."""
    upstream_broken = CompiledGraph.from_graph(Graph(nodes=[
        N(id="bad", type="nope", version=1),
        N(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("bad.result")}),
    ]), _registry(SCENARIOS["runs_once"]))
    broken, registry = _broken_graph_registry()
    unplaceable = CompiledGraph.from_graph(Graph(nodes=[N(id="emb", type="broken", version=1)]), registry)

    for compiled, cause in ((upstream_broken, "unknown_node_type"), (unplaceable, "embedded_graph_broken")):
        node = compiled.node("emb")
        assert (node.state, node.kind, node._cause.code) == ("wiring_failed", "graph", cause)
        with pytest.raises(KeyError):
            compiled.node("emb/holder" if compiled is upstream_broken else "emb/a")
    assert unplaceable.node("emb").version.node("a").state == "ready"


def test_a_value_on_a_field_the_graph_does_not_offer_is_a_stale_binding():
    """``up.text`` is fed inside, so the graph does not offer it; its value
    inside is not what the placed node reports holding."""
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        N(id="emb", type="inner-graph", version=1, bindings={"up.text": Static("x")}),
    ]), _registry(SCENARIOS["runs_once"]))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("stale_binding", "emb", "up.text")]
    assert compiled.node("emb").statics == {}


def test_an_edge_into_a_field_the_graph_does_not_offer_is_a_stale_binding():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        N(id="h", type="holder", version=1),
        N(id="emb", type="inner-graph", version=1, bindings={"up.text": From("h.result")}),
    ]), _registry(SCENARIOS["runs_once"]))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("stale_binding", "emb", "up.text")]

