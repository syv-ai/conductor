"""A subgraph offers an output made inside loops as one ``Series`` per loop, and a node outside the stored
graph that reads it is called as it would be with the same nodes drawn in its own graph.

The subgraph of most tests is ``documents -> lines -> shout``: ``documents`` returns a list of documents
(``None`` for an empty one), ``lines`` is called once per document and returns its lines, ``shout`` is called once
per line. So ``shout`` is called inside two loops, and its return values are a list per document.
"""

from __future__ import annotations

from typing import Any

from conductor import Graph, NodeRegistry, Series, make_graph

REGISTRY = NodeRegistry()


def _node(node_id: str) -> Any:
    return REGISTRY.node(node_id, title=node_id, description=node_id)


@_node("texts")
def texts(text: str) -> Series[str]:
    return text.split("|")


@_node("documents")
def documents(text: str) -> Series[str | None]:
    return [document or None for document in text.split(";")]


@_node("lines")
def lines(document: str) -> Series[str]:
    return document.split(",")


@_node("shout")
def shout(line: str) -> str:
    return line.upper()


@_node("whisper")
def whisper(line: str) -> str:
    return line.lower()


@_node("show")
def show(values: Series[str]) -> str:
    return "+".join(values)


@_node("mark")
def mark(value: str) -> str:
    return value + "!"


@_node("side-by-side")
def side_by_side(loud: Series[str], quiet: Series[str]) -> str:
    return f"{'+'.join(loud)}={'+'.join(quiet)}"


#: The edges of ``documents -> lines -> shout``, drawn in a graph or stored as one.
SHOUT_EDGES = [("documents.result", "lines.document"), ("lines.result", "shout.line")]
WHISPER_EDGES = [*SHOUT_EDGES, ("lines.result", "whisper.line")]

REGISTRY.subgraph("shout-lines", title="Shout lines", description="Every line of every document, in capitals.")(
    make_graph(documents, lines, shout, edges=SHOUT_EDGES)
)
REGISTRY.subgraph("split-lines", title="Split lines", description="The lines of every document.")(
    make_graph(documents, lines, edges=SHOUT_EDGES[:1])
)
REGISTRY.subgraph("shout-and-whisper", title="Shout and whisper", description="Every line in capitals, and in small letters.")(
    make_graph(documents, lines, shout, whisper, edges=WHISPER_EDGES)
)
REGISTRY.subgraph("wrapped", title="Wrapped", description="A subgraph whose one node is the subgraph shout-lines.")(
    make_graph(REGISTRY["shout-lines"])
)


def results(nodes: list[Any], edges: list[tuple[str, str]], node_id: str, text: str) -> Any:
    """The ``result`` of the node ``node_id`` after a run of the graph of ``nodes`` and ``edges`` on ``text``."""
    graph = Graph.compile(make_graph(*nodes, edges=edges), REGISTRY)
    assert not graph.problems, graph.problems
    run = graph.run(text=text)
    assert run.status == "complete"
    return run.results[node_id]["result"]


def test_a_series_input_outside_gets_one_documents_lines_per_call_as_it_does_with_the_nodes_drawn_in_its_graph() -> None:
    drawn = results([documents, lines, shout, show], [*SHOUT_EDGES, ("shout.result", "show.values")], "show", "a,b;;c")
    stored = results([REGISTRY["shout-lines"], show], [("shout-lines.shout.result", "show.values")], "show", "a,b;;c")
    assert drawn == stored == ["A+B", None, "C"]  # the second document is None, so show is not called for it


def test_a_node_outside_and_a_host_see_the_same_list_per_document() -> None:
    @_node("as-given")
    def as_given(values: Series[str]) -> list[str]:
        return list(values)

    graph = Graph.compile(make_graph(REGISTRY["shout-lines"], as_given, edges=[("shout-lines.shout.result", "as-given.values")]), REGISTRY)
    run = graph.run(text="a,b;;c")
    assert run.results["shout-lines"]["shout.result"] == run.results["as-given"]["result"] == [["A", "B"], None, ["C"]]


def test_the_output_is_described_as_a_series_of_series() -> None:
    graph = Graph.compile(make_graph(REGISTRY["shout-lines"], show, edges=[("shout-lines.shout.result", "show.values")]), REGISTRY)
    described = graph.describe().nodes["shout-lines"].outputs[0]
    assert (described.is_series, described.type, described.json_schema) == (True, "Series[str]", {"type": "array", "items": {"type": "string"}})
    in_palette = next(entry for entry in REGISTRY.describe().nodes if entry.id == "shout-lines").outputs[0]
    assert in_palette == described
    assert graph.runs_on["show"].series == "shout-lines.documents.result"  # show is called once per document


def test_an_input_of_one_value_outside_is_called_once_per_line_of_each_document() -> None:
    drawn = results([documents, lines, shout, mark], [*SHOUT_EDGES, ("shout.result", "mark.value")], "mark", "a,b;;c")
    stored = results([REGISTRY["shout-lines"], mark], [("shout-lines.shout.result", "mark.value")], "mark", "a,b;;c")
    assert drawn == stored == [["A!", "B!"], None, ["C!"]]


def test_a_stored_graph_used_once_per_item_of_an_outer_list_gives_a_node_outside_one_documents_lines_per_call() -> None:
    per_text = [("texts.result", "documents.text")]
    drawn = results([texts, documents, lines, shout, show], [*per_text, *SHOUT_EDGES, ("shout.result", "show.values")], "show", "a,b;;c|d;e,f")
    stored_edges = [("texts.result", "shout-lines.text"), ("shout-lines.shout.result", "show.values")]
    stored = results([texts, REGISTRY["shout-lines"], show], stored_edges, "show", "a,b;;c|d;e,f")
    assert drawn == stored == [["A+B", None, "C"], ["D", "E+F"]]


def test_a_series_returned_once_per_document_is_offered_as_a_series_of_series_too() -> None:
    drawn_edges = SHOUT_EDGES[:1]
    for reader, port, expected in ((show, "values", ["a+b", None, "c"]), (mark, "value", [["a!", "b!"], None, ["c!"]])):
        reader_id = reader.id
        drawn = results([documents, lines, reader], [*drawn_edges, ("lines.result", f"{reader_id}.{port}")], reader_id, "a,b;;c")
        stored = results([REGISTRY["split-lines"], reader], [("split-lines.lines.result", f"{reader_id}.{port}")], reader_id, "a,b;;c")
        assert drawn == stored == expected
    described = Graph.compile(make_graph(REGISTRY["split-lines"]), REGISTRY).describe().nodes["split-lines"].outputs[0]
    assert (described.is_series, described.type) == (True, "Series[str]")


def test_two_outputs_made_per_line_are_collected_per_document_by_one_node() -> None:
    reads = [("shout.result", "side-by-side.loud"), ("whisper.result", "side-by-side.quiet")]
    drawn = results([documents, lines, shout, whisper, side_by_side], [*WHISPER_EDGES, *reads], "side-by-side", "A,b;;c")
    stored_reads = [(f"shout-and-whisper.{source}", target) for source, target in reads]
    stored = results([REGISTRY["shout-and-whisper"], side_by_side], stored_reads, "side-by-side", "A,b;;c")
    assert drawn == stored == ["A+B=a+b", None, "C=c"]


def test_a_stored_graph_inside_a_stored_graph_offers_the_same_series_of_series() -> None:
    stored = results([REGISTRY["wrapped"], show], [("wrapped.shout-lines.shout.result", "show.values")], "show", "a,b;;c")
    assert stored == ["A+B", None, "C"]


@_node("title")
def title(document: str) -> str:
    return "T:" + document


@_node("pair")
def pair(a: str, b: str) -> str:
    return a + b


@_node("label")
def label(heading: str, values: Series[str]) -> str:
    return f"{heading}={'+'.join(values)}"


TITLE_EDGES = [*WHISPER_EDGES, ("documents.result", "title.document")]
REGISTRY.subgraph("titled", title="Titled", description="Each document's title, and every line in capitals and in small letters.")(
    make_graph(documents, lines, shout, whisper, title, edges=TITLE_EDGES)
)


def test_one_item_of_each_of_two_outputs_returned_in_the_same_loop_inside_reach_one_call_together() -> None:
    reads = [("shout.result", "pair.a"), ("whisper.result", "pair.b")]
    drawn = results([documents, lines, shout, whisper, title, pair], [*TITLE_EDGES, *reads], "pair", "A,b;;c")
    stored = results([REGISTRY["titled"], pair], [(f"titled.{source}", target) for source, target in reads], "pair", "A,b;;c")
    assert drawn == stored == [["Aa", "Bb"], None, ["Cc"]]


def test_one_value_per_document_beside_the_list_of_that_documents_lines_reach_one_call_together() -> None:
    reads = [("title.result", "label.heading"), ("shout.result", "label.values")]
    drawn = results([documents, lines, shout, whisper, title, label], [*TITLE_EDGES, *reads], "label", "a,b;;c")
    stored = results([REGISTRY["titled"], label], [(f"titled.{source}", target) for source, target in reads], "label", "a,b;;c")
    assert drawn == stored == ["T:a,b=A+B", None, "T:c=C"]


def test_one_item_of_a_list_inside_a_stored_graph_and_of_a_list_outside_it_is_refused_with_what_the_author_can_do() -> None:
    """``whisper`` runs once per line inside the subgraph; outside, each title is split into a list of its
    own. A node that takes one item of each has two lists side by side, and no edge outside can nest them."""
    edges = [("titled.title.result", "lines.document"), ("lines.result", "pair.b"), ("titled.whisper.result", "pair.a")]
    refused = Graph.compile(make_graph(REGISTRY["titled"], lines, pair, edges=edges), REGISTRY).problems[0]
    assert (refused.type, refused.ctx["variant"], refused.ctx["subgraphs"]) == ("series_not_nested", "inside_subgraph", ["titled"])
    assert sorted(refused.ctx["series"]) == ["lines.result", "titled.lines.result"]
    assert "wire '" not in refused.msg and "Give one of them whole to a Series input" in refused.msg
