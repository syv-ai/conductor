"""Compile stores how each input receives its value, and the ledger reads that record.

Every shape the walk over the edges can leave an input in is compiled here
and its record asserted without a run: one value per row, the whole
series, a group under a parent row, unrelated sources gathered. The last
section runs two of those shapes, where the ledger must decide as
compile did.
"""

from typing import Annotated

from conductor import AsItArrives, CompiledGraph, DType, Param, Result, Series, node
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Textarea
from helpers import placed, record, registry_of

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "receive-test-txt"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Result")]


@node("docs", title="Documents", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
    return [Txt(t) for t in folder.split(",") if t]


@node("upper", title="Upper case", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("lines", title="Lines", description="d", category="test")
def lines(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return text.splitlines()


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


@node("script", title="Script", description="d", category="test")
def script(**inputs: AsItArrives) -> Out:
    """An open interface: every connected name is received whole."""
    return Txt(",".join(f"{name}={value}" for name, value in sorted(inputs.items())))


@node("tags", title="Tags", description="d", category="test")
def tags_node(tags: Annotated[list[str], Param(title="Tags", connectable=False)] = ["x"]) -> Out:
    """A closed input typed as a list: one value that happens to be a list."""
    return Txt("|".join(tags))


def _registry(*extra):
    return registry_of(docs, upper_node, lines, join_node, script, tags_node, *extra)


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(record(*nodes), _registry(*extra))


def _input_plan(compiled, node_id, field):
    return compiled.run_plan.nodes[node_id].inputs[field]


# --- fed by an edge -------------------------------------------------------------


def test_a_scalar_input_fed_a_scalar_receives_one_value_once():
    compiled = _compiled([
        placed("a", "upper", text=Static("hi")),
        placed("b", "upper", text=From("a.result")),
    ])

    assert _input_plan(compiled, "b", "text").receive_mode == "same_value_every_row"


def test_a_scalar_input_fed_a_series_receives_one_value_per_row_of_its_index():
    compiled = _compiled([
        placed("docs", "docs"),
        placed("up", "upper", text=From("docs.result")),
    ])

    assert (_input_plan(compiled, "up", "text").receive_mode, _input_plan(compiled, "up", "text").rows_of) == ("value_at_row", "docs.result")
    assert compiled.run_plan.nodes["up"].runs_per_row_of == "docs.result"


def test_a_series_input_fed_one_series_on_a_root_receives_it_whole():
    compiled = _compiled([
        placed("docs", "docs"),
        placed("j", "join", texts=From("docs.result")),
    ])

    assert _input_plan(compiled, "j", "texts").receive_mode == "entire_value"
    assert compiled.run_plan.nodes["j"].runs_per_row_of is None


def test_a_series_input_fed_a_series_on_a_child_reduces_under_the_parent_row():
    compiled = _compiled([
        placed("docs", "docs"),
        placed("lines", "lines", text=From("docs.result")),
        placed("j", "join", texts=From("lines.result")),
    ])

    assert (_input_plan(compiled, "j", "texts").receive_mode, _input_plan(compiled, "j", "texts").rows_of) == ("all_rows_under_row", "lines.result")
    assert compiled.run_plan.nodes["j"].runs_per_row_of == "docs.result"


def test_a_series_input_fed_unrelated_sources_gathers_them_onto_the_fields_own_index():
    compiled = _compiled([
        placed("a", "upper", text=Static("a")),
        placed("b", "upper", text=Static("b")),
        placed("j", "join", texts=From("a.result", "b.result")),
    ])

    assert _input_plan(compiled, "j", "texts").receive_mode == "collected"


def test_an_open_parameter_receives_its_edge_whole_even_when_a_series_arrives():
    compiled = _compiled([
        placed("docs", "docs"),
        placed("s", "script", v=From("docs.result")),
    ])

    assert _input_plan(compiled, "s", "v").receive_mode == "entire_value"
    assert compiled.run_plan.nodes["s"].runs_per_row_of is None


# --- nothing feeds it -------------------------------------------------------------


def test_a_typed_in_scalar_is_received_once_and_a_typed_in_list_once_per_value():
    compiled = _compiled([
        placed("one", "upper", text=Static("hi")),
        placed("many", "upper", text=Static(["a", "b"])),
        placed("default", "upper"),
    ])

    assert _input_plan(compiled, "one", "text").receive_mode == "same_value_every_row"
    assert (_input_plan(compiled, "many", "text").receive_mode, _input_plan(compiled, "many", "text").rows_of) == ("value_at_row", "many.text")
    assert compiled.run_plan.nodes["many"].runs_per_row_of == "many.text"
    assert _input_plan(compiled, "default", "text").receive_mode == "same_value_every_row"


def test_a_series_input_with_a_typed_in_list_or_a_default_receives_it_whole_on_its_own_index():
    compiled = _compiled([
        placed("typed", "join", texts=Static(["a", "b"])),
        placed("default", "join"),
    ])

    assert _input_plan(compiled, "typed", "texts").receive_mode == "entire_value"
    assert _input_plan(compiled, "default", "texts").receive_mode == "entire_value"


def test_a_default_is_not_a_static():
    """The declared default applies when nothing is bound; it is the
    node's, not a value the author typed, so ``statics`` does not hold it."""
    compiled = _compiled([placed("t", "tags")])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("t").statics == {}
    assert _input_plan(compiled, "t", "tags").receive_mode == "same_value_every_row"


# --- inside an embedded graph ----------------------------------------------------------


def test_an_inner_reduction_over_the_entering_series_reduces_to_its_own_row():
    inner = embedded_graph_node("inner-graph", (
            placed("holder", "upper"),
            placed("join", "join", texts=From("holder.result")),
        ), _registry())
    compiled = _compiled([
        placed("docs", "docs"),
        placed("emb", "inner-graph", text=From("docs.result")),
    ], inner)

    assert compiled.is_runnable, compiled.problems
    assert (_input_plan(compiled, "emb/holder", "text").receive_mode, _input_plan(compiled, "emb/holder", "text").rows_of) == ("value_at_row", "docs.result")
    gather = _input_plan(compiled, "emb/join", "texts")
    assert (gather.receive_mode, [read.depth for read in gather.source_reads]) == ("collected", [1])
    assert compiled.run_plan.nodes["emb/join"].runs_per_row_of == "docs.result"


def test_unrelated_rows_inside_an_embedded_graph_are_misaligned_not_paired():
    """One inner node fed rows of ``a``, another fed a reduction over
    lines of ``b``. On one flat node this is ``misaligned``, and inside an
    embedded graph too: A's row i is never paired with B's row i."""
    inner = embedded_graph_node("pair-graph", (
            placed("up", "upper"),
            placed("j", "join"),
        ), _registry())
    compiled = _compiled([
        placed("a", "docs"),
        placed("b", "docs"),
        placed("lb", "lines", text=From("b.result")),
        placed("e", "pair-graph", text=From("a.result"), texts=From("lb.result")),
    ], inner)

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("misaligned", "e")]
    assert not compiled.is_runnable


def test_an_edge_from_another_nodes_input_is_not_a_source():
    """Only outputs are sources. An edge from ``h.text``, an input, is an
    ``unknown_ref_output`` at compile, not a run with nothing left to run."""
    compiled = _compiled([
        placed("h", "upper", text=Static("hi")),
        placed("u", "upper", text=From("h.text")),
    ])

    assert [(p.type, p.loc[0], p.loc[1]) for p in compiled.problems] == [("unknown_ref_output", "u", "text")]
    assert not compiled.is_runnable


def test_an_open_parameter_inside_an_embedded_graph_is_not_an_input_it_offers():
    """``**inputs`` takes a field per edge connected to it. Inside a graph
    compiled on its own nothing is connected to it, so the graph offers no
    field there, and an edge from outside into ``s.v`` reaches nothing."""
    inner = embedded_graph_node("script-graph", (placed("s", "script"),), _registry())
    compiled = _compiled([
        placed("docs", "docs"),
        placed("emb", "script-graph", v=From("docs.result")),
    ], inner)

    assert [(p.type, p.loc[0], p.loc[1]) for p in compiled.problems] == [("stale_binding", "emb", "v")]


# --- the ledger reads the record -----------------------------------------------------


def test_a_closed_list_input_runs_once_with_its_list_and_with_its_default():
    typed = _compiled([placed("t", "tags", tags=Static(["a", "b", "c"]))])
    defaulted = _compiled([placed("t", "tags")])

    assert typed.results(typed.run_sync())["t"]["result"] == "a|b|c"
    assert defaulted.results(defaulted.run_sync())["t"]["result"] == "x"
