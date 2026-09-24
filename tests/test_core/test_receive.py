"""Compile stores how each input receives its value, and the ledger reads that record.

Every shape the walk over the edges can leave an input in is compiled here
and its record asserted without a run: one value per row, the whole
series, a group under a parent row, unrelated sources gathered. The last
section runs the two shapes the review found the ledger deciding
differently from compile.
"""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry, run_sync
from conductor.dtype import DType, Single
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.graph.receive import Broadcast, Gather, Group, Iterate, Whole
from conductor.interface import Interface
from conductor.metadata import Input, Output, Param, Result
from conductor.node import GraphVersion, NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "receive-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt(t) for t in folder.split(",") if t]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper case"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Script(NodeDefinition):
    """An open interface: every connected name is received whole."""

    id = "script"
    title = "Script"
    description = "d"
    category = "test"

    def run(self, **inputs: Single) -> Out:
        return Txt(",".join(f"{name}={list(value) if isinstance(value, Series) else value}" for name, value in sorted(inputs.items())))


class Tags(NodeDefinition):
    """A closed input typed as a list: one value that happens to be a list."""

    id = "tags"
    title = "Tags"
    description = "d"
    category = "test"

    def run(self, tags: Annotated[list[str], Param(title="Tags", show_handle=False)] = ["x"]) -> Out:
        return Txt("|".join(tags))


def _embedded(node_id, graph, inputs, outputs):
    class Embedded(NodeDefinition):
        id = node_id
        title = "Embedded"
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, GraphVersion]] = {
            1: GraphVersion(graph=graph, interface=Interface(inputs=inputs, outputs=outputs, returns=Mapping))
        }

    return Embedded


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Lines, Join, Script, Tags, *extra):
        registry.register(node_cls)
    return registry


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(Graph(nodes=nodes), _registry(*extra))


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


def _receives(compiled, node_id, field):
    return compiled.field(Ref(node_id, field)).receives


# --- fed by an edge -------------------------------------------------------------


def test_a_scalar_input_fed_a_scalar_receives_one_value_once():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("a", "result"))}),
    ])

    assert _receives(compiled, "b", "text") == Broadcast()


def test_a_scalar_input_fed_a_series_receives_one_value_per_row_of_its_index():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "result"))}),
    ])

    assert _receives(compiled, "up", "text") == Iterate(Index("docs"))
    assert compiled.node("up").iterates_on == Index("docs")


def test_a_series_input_fed_one_series_on_a_root_receives_it_whole():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("docs", "result"))}),
    ])

    assert _receives(compiled, "j", "texts") == Whole()
    assert compiled.node("j").iterates_on is None


def test_a_series_input_fed_a_series_on_a_child_reduces_under_the_parent_row():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "result"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("lines", "result"))}),
    ])

    assert _receives(compiled, "j", "texts") == Group(Index("lines"), depth=1)
    assert compiled.node("j").iterates_on == Index("docs")


def test_a_series_input_fed_unrelated_sources_gathers_them_onto_the_fields_own_index():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])

    received = _receives(compiled, "j", "texts")
    assert received == Gather(Index(Ref("j", "texts")))
    assert received.index == compiled.field(Ref("j", "texts")).index


def test_an_open_parameter_receives_its_edge_whole_even_when_a_series_arrives():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="s", type="script", version=1, bindings={"v": _edge(("docs", "result"))}),
    ])

    assert _receives(compiled, "s", "v") == Whole()
    assert compiled.node("s").iterates_on is None


# --- nothing feeds it -------------------------------------------------------------


def test_a_typed_in_scalar_is_received_once_and_a_typed_in_list_once_per_value():
    compiled = _compiled([
        GraphNode(id="one", type="upper", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="many", type="upper", version=1, bindings={"text": Static(["a", "b"])}),
        GraphNode(id="default", type="upper", version=1),
    ])

    assert _receives(compiled, "one", "text") == Broadcast()
    assert _receives(compiled, "many", "text") == Iterate(Index(Ref("many", "text")))
    assert compiled.node("many").iterates_on == Index(Ref("many", "text"))
    assert _receives(compiled, "default", "text") == Broadcast()


def test_a_series_input_with_a_typed_in_list_or_a_default_receives_it_whole_on_its_own_index():
    compiled = _compiled([
        GraphNode(id="typed", type="join", version=1, bindings={"texts": Static(["a", "b"])}),
        GraphNode(id="default", type="join", version=1),
    ])

    assert _receives(compiled, "typed", "texts") == Whole()
    assert compiled.field(Ref("typed", "texts")).index == Index(Ref("typed", "texts"))
    assert _receives(compiled, "default", "texts") == Whole()


def test_a_closed_input_typed_as_a_list_holds_one_value_and_the_node_runs_once():
    """C3: ``['a', 'b', 'c']`` on ``tags: list[str]`` is the one value the
    type read, not three values to run per; whether the author typed many is
    decided by which validation succeeded, never by the shape of the result."""
    compiled = _compiled([GraphNode(id="t", type="tags", version=1, bindings={"tags": Static(["a", "b", "c"])})])

    assert compiled.is_runnable, compiled.problems
    assert _receives(compiled, "t", "tags") == Broadcast()
    assert compiled.node("t").iterates_on is None
    assert compiled.node("t").statics == {"tags": ["a", "b", "c"]}


def test_a_default_is_not_a_static():
    """C3: the declared default applies when nothing is bound; it is the
    node's, not a value the author typed, so ``statics`` does not hold it."""
    compiled = _compiled([GraphNode(id="t", type="tags", version=1)])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("t").statics == {}
    assert _receives(compiled, "t", "tags") == Broadcast()


def test_an_output_has_no_receive_state():
    compiled = _compiled([GraphNode(id="a", type="upper", version=1, bindings={"text": Static("hi")})])

    with pytest.raises(KeyError):
        compiled.field(Ref("a", "result")).receives


# --- inside an embedded graph ----------------------------------------------------------


def test_an_inner_reduction_over_the_entering_series_reduces_to_its_own_row():
    inner = _embedded(
        "inner-graph",
        (
            GraphNode(id="holder", type="upper", version=1),
            GraphNode(id="join", type="join", version=1, bindings={"texts": _edge(("holder", "result"))}),
        ),
        inputs=(Input(name="holder.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.text": _edge(("docs", "result"))}),
    ], inner)

    assert compiled.is_runnable, compiled.problems
    assert _receives(compiled, "emb/holder", "text") == Iterate(Index("docs"))
    assert _receives(compiled, "emb", "join.texts") == Group(Index("docs"), depth=1)
    assert compiled.node("emb/join").iterates_on == Index("docs")


def test_unrelated_rows_inside_an_embedded_graph_are_misaligned_not_paired():
    """C1: one inner node fed rows of ``a``, another fed a reduction over
    lines of ``b``. On one flat node this is ``misaligned``; inside a
    placement it used to run, pairing A's row i with B's row i."""
    inner = _embedded(
        "pair-graph",
        (
            GraphNode(id="up", type="upper", version=1),
            GraphNode(id="j", type="join", version=1),
        ),
        inputs=(
            Input(name="up.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),
            Input(name="j.texts", dtype=Series[Txt], title="Texts", default=(), optional=True),
        ),
        outputs=(Output(name="up.result", dtype=Txt, title="Upper"), Output(name="j.result", dtype=Txt, title="Joined")),
    )
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="lb", type="lines", version=1, bindings={"text": _edge(("b", "result"))}),
        GraphNode(id="e", type="pair-graph", version=1, bindings={"up.text": _edge(("a", "result")), "j.texts": _edge(("lb", "result"))}),
    ], inner)

    assert [(p.code, p.node_id) for p in compiled.problems] == [("misaligned", "e")]
    assert not compiled.is_runnable


def test_an_edge_from_another_nodes_input_is_not_a_source():
    """C2: only outputs are sources. An edge from ``h.value``, an input,
    compiled and then the run died with nothing left to run."""
    compiled = _compiled([
        GraphNode(id="h", type="upper", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="u", type="upper", version=1, bindings={"text": _edge(("h", "text"))}),
    ])

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("unknown_ref_output", "u", "text")]
    assert not compiled.is_runnable


def test_an_open_parameter_inside_an_embedded_graph_receives_whole_as_it_does_standalone():
    """C4: the inner graph behaves as it would standalone. Standalone,
    ``**inputs: Single`` fed a series receives it whole; embedded it used to
    make the placement run per row."""
    inner = _embedded(
        "script-graph",
        (GraphNode(id="s", type="script", version=1),),
        inputs=(Input(name="s.v", dtype=Series[Txt], title="V", default=(), optional=True),),
        outputs=(Output(name="s.result", dtype=Txt, title="Result"),),
    )
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="script-graph", version=1, bindings={"s.v": _edge(("docs", "result"))}),
    ], inner)

    assert compiled.is_runnable, compiled.problems
    assert _receives(compiled, "emb/s", "v") == Whole()
    assert compiled.node("emb/s").iterates_on is None
    assert compiled.node("emb").iterates_on is None


# --- the ledger reads the record -----------------------------------------------------


def test_a_closed_list_input_runs_once_with_its_list_and_with_its_default():
    typed = _compiled([GraphNode(id="t", type="tags", version=1, bindings={"tags": Static(["a", "b", "c"])})])
    defaulted = _compiled([GraphNode(id="t", type="tags", version=1)])

    assert run_sync(typed).results["t"]["result"] == "a|b|c"
    assert run_sync(defaulted).results["t"]["result"] == "x"


def test_an_open_parameter_embedded_receives_the_series_whole_at_run_time():
    inner = _embedded(
        "script-graph",
        (GraphNode(id="s", type="script", version=1),),
        inputs=(Input(name="s.v", dtype=Series[Txt], title="V", default=(), optional=True),),
        outputs=(Output(name="s.result", dtype=Txt, title="Result"),),
    )
    standalone = _compiled([
        GraphNode(id="docs", type="docs", version=1, bindings={"folder": Static("fa,fb")}),
        GraphNode(id="s", type="script", version=1, bindings={"v": _edge(("docs", "result"))}),
    ])
    embedded = _compiled([
        GraphNode(id="docs", type="docs", version=1, bindings={"folder": Static("fa,fb")}),
        GraphNode(id="emb", type="script-graph", version=1, bindings={"s.v": _edge(("docs", "result"))}),
    ], inner)

    assert run_sync(standalone).results["s"]["result"] == "v=['fa', 'fb']"
    assert run_sync(embedded).results["emb/s"]["result"] == "v=['fa', 'fb']"
