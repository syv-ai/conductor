"""A graph compiled on its own and placed as a node: its nodes run under the placement's name, under the row the placement runs on."""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import NodeRegistry, run_sync
from conductor.dtype import DType
from conductor.errors import NodeKindError, NodeWiringError
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.metadata import Output, Param, Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "expansion-test-txt"
    title = "Text"


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


def _inner_graph():
    """The embedded graph: a holder, uppercased, then joined. Standalone it
    takes one text at `holder.value` and returns `join.result`."""
    return (
        GraphNode(id="holder", type="holder", version=1, title="Text", bindings={"value": Static("inner")}),
        GraphNode(id="up", type="upper", version=1, title="Upper", bindings={"text": From("holder.result")}),
        GraphNode(id="join", type="join", version=1, title="Join", bindings={"texts": From("up.result")}),
    )


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (Holder, Upper, Join, Docs, *extra):
        registry.register(node_cls)
    return registry


def _inner_definition():
    return embedded_graph_node("inner-graph", _inner_graph(), _registry())


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(Graph(nodes=nodes), _registry(_inner_definition(), *extra))


# --- the nodes of a placement ------------------------------------------------


def test_the_inner_nodes_are_nodes_of_the_one_run_under_the_placements_name():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("src.result")}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.join.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order == ("src", "emb/holder", "emb/up", "emb/join", "after")
    assert compiled.node("emb/up").embedded_in == "emb"
    assert compiled.node("src").embedded_in is None
    # The node that embeds the graph is not a unit of the run; its inner nodes are.
    emb = compiled.node("emb")
    assert (emb.kind, emb.graph_node.type, compiled.node("src").kind) == ("graph", "inner-graph", "node")
    for asked in ("runner", "fingerprint"):
        with pytest.raises(NodeKindError, match="a node of kind 'graph' does not run as one unit"):
            getattr(emb, asked)
    with pytest.raises(NodeKindError):
        emb.validate({})


def test_a_graph_holds_the_values_typed_on_it_by_inner_address():
    """Typed on the embedding node, read where it landed: the inner field's type reads it."""
    compiled = _compiled([
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Static("outer"), "gone.x": Static("y")}),
    ])

    assert compiled.node("emb").statics == {"holder.value": "outer"}
    assert compiled.node("emb").statics["holder.value"] == compiled.node("emb/holder").statics["value"]


def test_the_placements_bindings_move_onto_the_inner_fields_they_name():
    """An edge into `emb.holder.value` replaces the inner author's static;
    an outer edge from `emb.join.result` reads the inner node's output."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("src.result")}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.join.result")}),
    ])

    assert compiled.field(Ref("emb/holder", "value")).binding == From("src.result")
    assert compiled.field(Ref("after", "text")).binding == From("emb/join.result")


def test_an_unconnected_placement_keeps_the_inner_statics_and_expands_flat():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1)])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("emb/holder", "value")).binding == Static("inner")
    assert compiled.node("emb").iterates_on is None
    assert all(compiled.node(node_id).iterates_on is None for node_id in compiled.execution_order)


def test_a_placement_is_a_node_in_the_interface_named_by_inner_address():
    """At the interface the embedded graph is one node
    whose fields are its graph's, under the placement's name."""
    compiled = _compiled([
        GraphNode(
            id="emb", type="inner-graph", version=1, title="Approve",
            fields={"holder.value": FieldContent(title="Application"), "join.result": FieldContent(title="Answer")},
        ),
    ])

    assert [(i.name, i.title) for i in compiled.interface.inputs] == [("emb/holder.value", "Application")]
    assert [(o.name, o.title) for o in compiled.interface.outputs] == [("emb/join.result", "Answer")]
    assert [i.name for i in compiled.node("emb").interface.inputs] == ["holder.value"]


def test_a_field_inside_a_placement_has_one_address():
    """``emb/join.result`` is the field; the author's spelling of it,
    ``emb.join.result``, names a field of the graph node, which has none."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("src.result")}),
    ])

    assert compiled.field(Ref("emb/join", "result")).type is Txt
    assert compiled.field(Ref("emb/holder", "value")).binding == From("src.result")
    with pytest.raises(KeyError, match="'emb' is a graph"):
        compiled.field(Ref("emb", "join.result"))


class Sheet(NodeDefinition):
    """One output named after a column, as a table's fold names them: free text, dots and all."""

    id = "sheet"
    title = "Sheet"
    description = "d"
    category = "t"

    def run(self) -> Mapping[str, Any]:
        return {"excl. VAT": Txt("net")}

    def compute_outputs(self, declared, values, arriving):
        return (Output(name="excl. VAT", dtype=Txt, title="excl. VAT"),)


def test_a_field_name_with_a_dot_beside_an_embedded_graph_is_a_plain_field():
    """The first dot splits an address, and the field is free text: ``sheet.excl. VAT``
    is the field ``excl. VAT`` on ``sheet``, wherever an embedded graph sits."""
    compiled = _compiled([
        GraphNode(id="sheet", type="sheet", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From(Ref("sheet", "excl. VAT"))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.join.result")}),
    ], Sheet)

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("sheet", "excl. VAT")).type is Txt
    assert compiled.field(Ref("emb/holder", "value")).binding == From(Ref("sheet", "excl. VAT"))
    assert run_sync(compiled).state.results(compiled)["after"]["result"] == "NET"


def test_a_broken_edge_into_a_placement_is_the_placements_own_problem():
    """The edge sits on the node the author placed, on the field it names,
    in the plain words any node gets; the placed node is not worked out, and
    says why."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="upper", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("ghost.result")}),
    ])

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id, problem.field) == ("unknown_ref_node", "emb", "holder.value")
    assert problem.details == {"source_node": "ghost"}
    assert compiled.node("emb").state == "wiring_failed"
    assert compiled.node("emb")._cause is problem


def test_a_stale_key_on_the_placement_is_reported_on_the_placement():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1, bindings={"nope.value": Static(1)})])

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("stale_binding", True, "emb", "nope.value")


# --- the row a placement runs on ------------------------------------------------


def test_a_series_entering_through_a_scalar_field_makes_the_placement_iterate():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("docs.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").iterates_on == Index("docs")
    assert compiled.node("emb/holder").iterates_on == Index("docs")
    assert compiled.node("emb/up").iterates_on == Index("docs")


def test_an_inner_reduction_over_the_entering_series_is_a_fold_of_one():
    """The join test: standalone the graph joins one text; embedded and fed
    three, it joins one text three times — never all three once."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": From("docs.result")}),
    ])

    assert compiled.node("emb/join").iterates_on == Index("docs")
    assert (compiled.field(Ref("emb/join", "result")).type, compiled.field(Ref("emb/join", "result")).index) == (Series[Txt], Index("docs"))


def test_a_series_born_inside_reduces_to_the_outer_row_by_lineage_alone():
    """An iterating inner unfold births a child of the outer index; an inner
    reduction over it collapses to the outer row without any scope rule.
    ``Index.__eq__`` reads the id alone, so the parent is asserted by name."""
    inner = embedded_graph_node("splitter", (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
        ), _registry(Lines))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter", version=1, bindings={"holder.value": From("docs.result")}),
        ]),
        _registry(inner, Lines),
    )

    assert compiled.is_runnable, compiled.problems
    born = compiled.field(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    assert compiled.node("emb/join").iterates_on == Index("docs")

    # The same, when the series is born off an inner *static* the entering
    # series never touches: the block iterates, so the birth is a child of
    # the entering index all the same — never a root beside the outer rows.
    unfed = embedded_graph_node("splitter-unfed", (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
            GraphNode(id="entered", type="holder", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
        ), _registry(Lines))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter-unfed", version=1, bindings={"entered.value": From("docs.result")}),
        ]),
        _registry(unfed, Lines),
    )
    assert compiled.is_runnable, compiled.problems
    born = compiled.field(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    # The nodes the entering series never reaches still run once per outer row.
    assert compiled.node("emb/holder").iterates_on == Index("docs")
    assert compiled.node("emb/lines").iterates_on == Index("docs")
    assert compiled.node("emb/join").iterates_on == Index("docs")


def test_two_crossings_on_one_lineage_make_the_whole_block_iterate_on_the_deeper():
    """`docs` → `lines` (a child of `docs`); the block takes one field from
    each. The block folds over the deeper index, and so does every node
    inside it — the one fed from the shallower index broadcasts down;
    it does not run per `docs` row while its neighbours run per line."""
    inner = embedded_graph_node("pair", (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
            GraphNode(id="ua", type="upper", version=1, bindings={"text": From("a.result")}),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("docs.result")}),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": From("docs.result"),
                "b.value": From("lines.result"),
            }),
        ]),
        _registry(inner, Lines),
    )

    assert compiled.is_runnable, compiled.problems
    per_line = Index("lines", parent=Index("docs"))
    assert compiled.node("emb").iterates_on == per_line
    assert compiled.node("emb/a").iterates_on == per_line
    assert compiled.node("emb/ua").iterates_on == per_line
    assert compiled.node("emb/b").iterates_on == per_line
    assert compiled.field(Ref("emb/ua", "result")).index == per_line


def test_a_series_entering_a_series_field_is_read_whole_and_the_block_expands_flat():
    """The graph declares it takes a series, so a series is one value to it:
    no scalar crossing, no scope, one reduction over the whole pile."""
    inner = embedded_graph_node("joiner", (GraphNode(id="join", type="join", version=1),), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="joiner", version=1, bindings={"join.texts": From("docs.result")}),
        ]),
        _registry(inner),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").iterates_on is None
    assert compiled.node("emb/join").iterates_on is None
    assert compiled.field(Ref("emb/join", "result")).type is Txt


def test_two_unrelated_series_entering_one_placement_are_its_misaligned():
    inner = embedded_graph_node("pair", (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": From("d1.result"), "b.value": From("d2.result"),
            }),
        ]),
        _registry(inner),
    )

    assert [(p.code, p.node_id) for p in compiled.problems] == [("misaligned", "emb")]
    (misaligned,) = compiled.problems
    placement = compiled.node("emb")
    assert placement.state == "wiring_failed"
    assert placement.problems == compiled.problems
    with pytest.raises(NodeWiringError) as raised:
        placement.iterates_on
    assert raised.value.problems[0] == misaligned
    assert "emb" not in compiled.decisions


def test_a_nested_embedded_graph_is_addressed_by_its_path():
    outer = embedded_graph_node("outer-graph", (
            GraphNode(id="pre", type="holder", version=1, bindings={"value": Static("x")}),
            GraphNode(id="inner", type="inner-graph", version=1, bindings={"holder.value": From("pre.result")}),
        ), _registry(_inner_definition()))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="top", type="outer-graph", version=1),
            GraphNode(id="after", type="upper", version=1, bindings={"text": From("top.inner/join.result")}),
        ]),
        _registry(_inner_definition(), outer),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order == ("top/pre", "top/inner/holder", "top/inner/up", "top/inner/join", "after")
    assert compiled.node("top/inner/up").embedded_in == "top/inner"
    assert compiled.node("top/pre").embedded_in == "top"
    assert compiled.field(Ref("after", "text")).binding == From("top/inner/join.result")
    assert compiled.field(Ref("top/inner/join", "result")).type is Txt



def test_a_broken_graph_inside_a_graph_breaks_the_graph_around_it():
    """An unknown inner node leaves its graph unplaceable, and so the graph
    that places it too. Each level shows one ``embedded_graph_broken`` where
    it is placed; the unknown node is the innermost graph's own problem.
    Each is one node where it is placed, read inside through its version."""
    broken = embedded_graph_node("broken-graph", (GraphNode(id="x", type="nothing", version=1), GraphNode(id="up", type="upper", version=1)), _registry())
    wrapper = embedded_graph_node("wrapper-graph", (GraphNode(id="mid", type="broken-graph", version=1),), _registry(broken))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="top", type="wrapper-graph", version=1)]), _registry(broken, wrapper),
    )

    assert [(p.code, p.node_id) for p in compiled.problems] == [("embedded_graph_broken", "top")]
    top = compiled.node("top")
    assert [(p.code, p.node_id) for p in top.version.problems] == [("embedded_graph_broken", "mid")]
    assert [(p.code, p.node_id) for p in top.version.node("mid").version.problems] == [("unknown_node_type", "x")]
    for node in (top, top.version.node("mid")):
        assert (node.state, node.kind) == ("wiring_failed", "graph")
        assert node.interface is not None  # the box and its handles still draw
        with pytest.raises(NodeWiringError) as raised:
            node.iterates_on
        assert raised.value.problems[0].code == "embedded_graph_broken"


# --- what compile refuses about an embedded graph ---------------------------------------


def test_an_authored_id_with_a_slash_is_refused_and_never_collides_with_an_inner_node():
    """``/`` names the nodes of an embedded graph, so an authored ``e/holder``
    beside a placement ``e`` is refused on its own: it neither appears twice
    in the order nor is read by the inner nodes."""
    compiled = _compiled([
        GraphNode(id="e/holder", type="holder", version=1, bindings={"value": Static("impostor")}),
        GraphNode(id="e", type="inner-graph", version=1),
    ])

    assert [(p.code, p.fatal, p.node_id) for p in compiled.problems] == [("invalid_node_id", True, "e/holder")]
    assert compiled.execution_order.count("e/holder") == 1
    assert compiled.field(Ref("e/holder", "value")).binding == Static("inner")


def test_a_refused_authored_id_is_not_the_problem_of_the_inner_node_sharing_it():
    """`e` can be placed, so its `holder` runs as `e/holder`; the author also
    wrote `e/holder`, which is refused. The refusal is the author's: it stays
    in the graph's problems and never shows on the inner node."""
    compiled = _compiled([
        GraphNode(id="e", type="inner-graph", version=1),
        GraphNode(id="e/holder", type="upper", version=1),
    ])

    assert [(p.code, p.node_id) for p in compiled.problems] == [("invalid_node_id", "e/holder")]
    inner = compiled.node("e/holder")
    assert (inner.state, inner.embedded_in, inner.problems) == ("ready", "e", ())


def test_an_authored_id_holding_a_slash_is_refused_and_stays_the_authors():
    """The author wrote `e/holder` beside an embedded graph `e` whose own
    `holder` has a type the registry lacks. The authored id is refused; the
    graph cannot be placed, which is one problem on `e`, and
    `node("e/holder")` is the author's node, refused."""
    lost = embedded_graph_node("lost", (GraphNode(id="holder", type="nope", version=1),), _registry())
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="e", type="lost", version=1),
        GraphNode(id="e/holder", type="upper", version=1),
    ]), _registry(lost))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [
        ("invalid_node_id", "e/holder", None), ("embedded_graph_broken", "e", None),
    ]
    assert [(p.code, p.node_id) for p in compiled.node("e").version.problems] == [("unknown_node_type", "holder")]
    assert compiled.node("e").state == "wiring_failed"
    refused = compiled.node("e/holder")
    assert (refused.state, refused.embedded_in, refused._cause.code) == ("resolution_failed", None, "invalid_node_id")


def test_misaligned_inside_an_embedded_graph_is_reported_once_with_the_authors_addresses():
    """One inner node fed two unrelated outer series is reported once, on the
    placement, with the addresses the author sees, in the shape every
    ``misaligned`` has."""

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
            return Txt(a + b)

    inner = embedded_graph_node("pairs", (GraphNode(id="p", type="pair", version=1),), _registry(Pair))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pairs", version=1, bindings={"p.a": From("d1.result"), "p.b": From("d2.result")}),
        ]),
        _registry(inner, Pair),
    )

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id) == ("misaligned", "emb")
    assert problem.details == {"a": "emb/p.a", "b": "emb/p.b"}
    assert "emb.p" not in problem.message and "emb.p" not in str(problem.details)


def test_an_edge_into_a_field_fed_inside_the_graph_is_a_stale_binding():
    """``up.text`` is fed by ``holder`` inside, so the graph does not offer
    it: an edge from outside reaches nothing, as a field any node lacks."""
    compiled = _compiled([
        GraphNode(id="d1", type="docs", version=1),
        GraphNode(id="d2", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"up.text": From("d1.result", "d2.result")}),
    ])

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id, problem.field) == ("stale_binding", "emb", "up.text")


