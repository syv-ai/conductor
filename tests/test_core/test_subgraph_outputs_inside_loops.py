"""A subgraph offers an output returned inside loops as one ``Series`` per loop, and a node of the outer graph that
reads it is called as it would be with the same nodes in one graph.

The subgraph of most tests is ``documents -> lines -> shout``: ``documents`` returns a list of documents (``None``
for an empty one), ``lines`` is called once per document and returns its lines, ``shout`` is called once per line.
So ``shout`` is inside two loops, and its outputs are a list per document. Each test runs the same nodes in one graph
(``in_one_graph``) and through the subgraph (``through_subgraph``) and asserts that both give the same result.
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


#: The edges of ``documents -> lines -> shout``, in one graph or as the subgraph.
SHOUT_EDGES = [("documents.result", "lines.document"), ("lines.result", "shout.line")]
WHISPER_EDGES = [*SHOUT_EDGES, ("lines.result", "whisper.line")]

REGISTRY.subgraph("shout-lines", title="Shout lines", description="Every line of every document, in capitals.")(
    make_graph(documents, lines, shout, edges=SHOUT_EDGES)
)
REGISTRY.subgraph("split-lines", title="Split lines", description="The lines of every document.")(
    make_graph(documents, lines, edges=SHOUT_EDGES[:1])
)
REGISTRY.subgraph("shout-and-whisper", title="Shout and whisper", description="Every line in capitals, and in small letters.")(
    make_graph(documents, lines, shout, whisper, edges=WHISPER_EDGES, outputs={"loud": "shout.result", "quiet": "whisper.result"})
)
REGISTRY.subgraph("wrapped", title="Wrapped", description="A subgraph whose one node is the subgraph shout-lines.")(
    make_graph(REGISTRY["shout-lines"])
)


def results(nodes: list[Any], edges: list[tuple[str, str]], node_id: str, text: str) -> Any:
    """The ``result`` of the node ``node_id`` after a run of the graph of ``nodes`` and ``edges`` on ``text``."""
    graph = Graph.compile(make_graph(*nodes, edges=edges, outputs={"shown": f"{node_id}.result"}), REGISTRY)
    assert not graph.problems, graph.problems
    run = graph.run(text=text)
    assert run.status == "complete"
    return run.results[node_id]["result"]


def test_a_series_input_outside_gets_one_documents_lines_per_call_as_in_one_graph() -> None:
    in_one_graph = results([documents, lines, shout, show], [*SHOUT_EDGES, ("shout.result", "show.values")], "show", "a,b;;c")
    through_subgraph = results([REGISTRY["shout-lines"], show], [("shout-lines.result", "show.values")], "show", "a,b;;c")
    assert in_one_graph == through_subgraph == ["A+B", None, "C"]  # the second document is None, so show is not called for it


def test_a_node_outside_and_a_host_see_the_same_list_per_document() -> None:
    @_node("as-given")
    def as_given(values: Series[str]) -> list[str]:
        return list(values)

    graph = Graph.compile(make_graph(REGISTRY["shout-lines"], as_given, edges=[("shout-lines.result", "as-given.values")]), REGISTRY)
    run = graph.run(text="a,b;;c")
    assert run.results["shout-lines"]["result"] == run.results["as-given"]["result"] == [["A", "B"], None, ["C"]]


def test_the_output_is_described_as_a_series_of_series() -> None:
    graph = Graph.compile(make_graph(REGISTRY["shout-lines"], show, edges=[("shout-lines.result", "show.values")]), REGISTRY)
    described = graph.describe().nodes["shout-lines"].outputs[0]
    assert (described.is_series, described.type, described.json_schema) == (True, "Series[str]", {"type": "array", "items": {"type": "string"}})
    in_palette = next(entry for entry in REGISTRY.describe().nodes if entry.id == "shout-lines").outputs[0]
    assert in_palette == described
    assert graph.runs_on["show"].series == "shout-lines.documents.result"  # show is called once per document


def test_an_input_of_one_value_outside_is_called_once_per_line_of_each_document() -> None:
    in_one_graph = results([documents, lines, shout, mark], [*SHOUT_EDGES, ("shout.result", "mark.value")], "mark", "a,b;;c")
    through_subgraph = results([REGISTRY["shout-lines"], mark], [("shout-lines.result", "mark.value")], "mark", "a,b;;c")
    assert in_one_graph == through_subgraph == [["A!", "B!"], None, ["C!"]]


def test_a_subgraph_inside_an_outer_loop_gives_a_node_outside_one_documents_lines_per_call() -> None:
    per_text = [("texts.result", "documents.text")]
    in_one_graph = results([texts, documents, lines, shout, show], [*per_text, *SHOUT_EDGES, ("shout.result", "show.values")], "show", "a,b;;c|d;e,f")
    subgraph_edges = [("texts.result", "shout-lines.text"), ("shout-lines.result", "show.values")]
    through_subgraph = results([texts, REGISTRY["shout-lines"], show], subgraph_edges, "show", "a,b;;c|d;e,f")
    assert in_one_graph == through_subgraph == [["A+B", None, "C"], ["D", "E+F"]]


def test_a_series_returned_once_per_document_is_offered_as_a_series_of_series_too() -> None:
    one_graph_edges = SHOUT_EDGES[:1]
    for reader, input_name, expected in ((show, "values", ["a+b", None, "c"]), (mark, "value", [["a!", "b!"], None, ["c!"]])):
        reader_id = reader.id
        in_one_graph = results([documents, lines, reader], [*one_graph_edges, ("lines.result", f"{reader_id}.{input_name}")], reader_id, "a,b;;c")
        through_subgraph = results([REGISTRY["split-lines"], reader], [("split-lines.result", f"{reader_id}.{input_name}")], reader_id, "a,b;;c")
        assert in_one_graph == through_subgraph == expected
    described = Graph.compile(make_graph(REGISTRY["split-lines"]), REGISTRY).describe().nodes["split-lines"].outputs[0]
    assert (described.is_series, described.type) == (True, "Series[str]")


def test_two_outputs_returned_per_line_are_collected_per_document_by_one_node() -> None:
    reads = [("shout.result", "side-by-side.loud"), ("whisper.result", "side-by-side.quiet")]
    in_one_graph = results([documents, lines, shout, whisper, side_by_side], [*WHISPER_EDGES, *reads], "side-by-side", "A,b;;c")
    subgraph_reads = [("shout-and-whisper.loud", "side-by-side.loud"), ("shout-and-whisper.quiet", "side-by-side.quiet")]
    through_subgraph = results([REGISTRY["shout-and-whisper"], side_by_side], subgraph_reads, "side-by-side", "A,b;;c")
    assert in_one_graph == through_subgraph == ["A+B=a+b", None, "C=c"]


def test_a_subgraph_inside_a_subgraph_offers_the_same_series_of_series() -> None:
    through_subgraph = results([REGISTRY["wrapped"], show], [("wrapped.result", "show.values")], "show", "a,b;;c")
    assert through_subgraph == ["A+B", None, "C"]


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
    make_graph(documents, lines, shout, whisper, title, edges=TITLE_EDGES, outputs={"loud": "shout.result", "quiet": "whisper.result", "title": "title.result"})
)


def test_one_item_of_each_of_two_outputs_returned_in_the_same_loop_inside_reach_one_call_together() -> None:
    reads = [("shout.result", "pair.a"), ("whisper.result", "pair.b")]
    in_one_graph = results([documents, lines, shout, whisper, title, pair], [*TITLE_EDGES, *reads], "pair", "A,b;;c")
    through_subgraph = results([REGISTRY["titled"], pair], [("titled.loud", "pair.a"), ("titled.quiet", "pair.b")], "pair", "A,b;;c")
    assert in_one_graph == through_subgraph == [["Aa", "Bb"], None, ["Cc"]]


def test_one_value_per_document_beside_the_list_of_that_documents_lines_reach_one_call_together() -> None:
    reads = [("title.result", "label.heading"), ("shout.result", "label.values")]
    in_one_graph = results([documents, lines, shout, whisper, title, label], [*TITLE_EDGES, *reads], "label", "a,b;;c")
    through_subgraph = results([REGISTRY["titled"], label], [("titled.title", "label.heading"), ("titled.loud", "label.values")], "label", "a,b;;c")
    assert in_one_graph == through_subgraph == ["T:a,b=A+B", None, "T:c=C"]


def test_a_node_taking_items_of_a_list_inside_a_subgraph_and_of_one_outside_is_refused_saying_what_to_do() -> None:
    """``whisper`` is called once per line inside the subgraph; outside, each title is split into a list of its
    own. A node that takes one item of each is inside two loops side by side, and no edge outside can nest them."""
    edges = [("titled.title", "lines.document"), ("lines.result", "pair.b"), ("titled.quiet", "pair.a")]
    refused = Graph.compile(make_graph(REGISTRY["titled"], lines, pair, edges=edges), REGISTRY).problems[0]
    assert (refused.type, refused.ctx["variant"], refused.ctx["subgraphs"]) == ("series_not_nested", "inside_subgraph", ["titled"])
    assert sorted(refused.ctx["series"]) == ["lines.result", "titled.lines.result"]
    assert "wire '" not in refused.msg and "Give one of them whole to a Series input" in refused.msg
