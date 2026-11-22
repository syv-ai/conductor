"""Tests for a subgraph output that is made inside loops.

A node inside a subgraph may be called once per item of a list. The subgraph then offers that node's output as a
``Series``, one level per loop. A node in the outer graph that reads it should be called just as it would be if all
the nodes were in one graph.

Most tests use the subgraph ``documents -> lines -> shout``. ``documents`` returns a list of documents, with
``None`` for an empty one. ``lines`` is called once per document and returns its lines. ``shout`` is called once per
line. So ``shout`` is inside two loops, and its output is a list of lines per document.

Each test runs the same nodes twice: in one graph (``in_one_graph``) and through the subgraph
(``through_subgraph``). Both must give the same result.
"""

from __future__ import annotations

from typing import Any

import pytest
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

REGISTRY.subgraph(
    "shout-lines", title="Shout lines", description="Every line of every document, in capitals."
)(make_graph(documents, lines, shout, edges=SHOUT_EDGES))
REGISTRY.subgraph("split-lines", title="Split lines", description="The lines of every document.")(
    make_graph(documents, lines, edges=SHOUT_EDGES[:1])
)
REGISTRY.subgraph(
    "shout-and-whisper",
    title="Shout and whisper",
    description="Every line in capitals, and in small letters.",
)(
    make_graph(
        documents,
        lines,
        shout,
        whisper,
        edges=WHISPER_EDGES,
        outputs={"loud": "shout.result", "quiet": "whisper.result"},
    )
)
REGISTRY.subgraph(
    "wrapped", title="Wrapped", description="A subgraph whose one node is the subgraph shout-lines."
)(make_graph(REGISTRY["shout-lines"]))


def results(nodes: list[Any], edges: list[tuple[str, str]], node_id: str, text: str) -> Any:
    """Runs the graph of ``nodes`` and ``edges`` on ``text``, and returns the ``result`` of the node ``node_id``."""
    graph = Graph.compile(
        make_graph(*nodes, edges=edges, outputs={"shown": f"{node_id}.result"}), REGISTRY
    )
    assert not graph.problems, graph.problems
    run = graph.run({"text": text})
    assert run.status == "complete"
    return run.results[node_id]["result"]


def test_a_node_outside_and_a_host_see_the_same_list_per_document() -> None:
    @_node("as-given")
    def as_given(values: Series[str]) -> list[str]:
        return list(values)

    graph = Graph.compile(
        make_graph(
            REGISTRY["shout-lines"], as_given, edges=[("shout-lines.result", "as-given.values")]
        ),
        REGISTRY,
    )
    run = graph.run({"text": "a,b;;c"})
    assert (
        run.results["shout-lines"]["result"]
        == run.results["as-given"]["result"]
        == [["A", "B"], None, ["C"]]
    )


def test_the_output_is_described_as_a_list_per_loop_of_the_value_returned_inside() -> None:
    graph = Graph.compile(
        make_graph(REGISTRY["shout-lines"], show, edges=[("shout-lines.result", "show.values")]),
        REGISTRY,
    )
    described = graph.describe().nodes["shout-lines"].outputs[0]
    assert (described.is_series, described.type, described.json_schema) == (
        True,
        "str",
        {"type": "string"},
    )
    assert described.loops == [
        "documents.result",
        "lines.result",
    ]  # a list per document of a list per line
    in_palette = next(
        entry for entry in REGISTRY.describe().nodes if entry.id == "shout-lines"
    ).outputs[0]
    assert in_palette == described
    assert (
        graph.describe().nodes["show"].per_item_of == "shout-lines/documents.result"
    )  # show is called once per document


def test_a_series_returned_once_per_document_is_described_as_a_series_of_series_too() -> None:
    described = (
        Graph.compile(make_graph(REGISTRY["split-lines"]), REGISTRY)
        .describe()
        .nodes["split-lines"]
        .outputs[0]
    )
    assert (described.is_series, described.type, described.loops) == (
        True,
        "str",
        ["documents.result", "lines.result"],
    )


def test_a_subgraph_inside_a_subgraph_offers_the_same_series_of_series() -> None:
    through_subgraph = results(
        [REGISTRY["wrapped"], show], [("wrapped.result", "show.values")], "show", "a,b;;c"
    )
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
REGISTRY.subgraph(
    "titled",
    title="Titled",
    description="Each document's title, and every line in capitals and in small letters.",
)(
    make_graph(
        documents,
        lines,
        shout,
        whisper,
        title,
        edges=TITLE_EDGES,
        outputs={"loud": "shout.result", "quiet": "whisper.result", "title": "title.result"},
    )
)


def test_a_node_taking_items_of_a_list_inside_a_subgraph_and_of_one_outside_is_refused_saying_what_to_do() -> (
    None
):
    """A node can't take one item of a loop inside a subgraph and one item of a loop outside it.

    Inside the subgraph, ``whisper`` is called once per line. Outside, each title is split into a list of its own. A
    node that takes one item of each sits in two loops side by side, and no edge outside can nest one in the other.
    The problem says what to do instead.
    """
    edges = [
        ("titled.title", "lines.document"),
        ("lines.result", "pair.b"),
        ("titled.quiet", "pair.a"),
    ]
    refused = Graph.compile(
        make_graph(REGISTRY["titled"], lines, pair, edges=edges), REGISTRY
    ).problems[0]
    assert (refused.type, refused.ctx["subgraphs"]) == (
        "series_not_nested_inside_subgraph",
        ["titled"],
    )
    assert sorted(refused.ctx["series"]) == ["lines.result", "titled/lines.result"]
    assert "wire '" not in refused.msg and "Give one of them whole to a Series input" in refused.msg


_PER_TEXT = [("texts.result", "documents.text")]
_TITLE_READS = [("shout.result", "pair.a"), ("whisper.result", "pair.b")]


@pytest.mark.parametrize(
    ("in_one_graph", "through_subgraph", "reader", "text", "expected"),
    [
        (
            ([documents, lines, shout, show], [*SHOUT_EDGES, ("shout.result", "show.values")]),
            ([REGISTRY["shout-lines"], show], [("shout-lines.result", "show.values")]),
            "show",
            "a,b;;c",
            ["A+B", None, "C"],  # the second document is None, so show is not called for it
        ),
        (
            ([documents, lines, shout, mark], [*SHOUT_EDGES, ("shout.result", "mark.value")]),
            ([REGISTRY["shout-lines"], mark], [("shout-lines.result", "mark.value")]),
            "mark",
            "a,b;;c",
            [["A!", "B!"], None, ["C!"]],
        ),
        (
            (
                [texts, documents, lines, shout, show],
                [*_PER_TEXT, *SHOUT_EDGES, ("shout.result", "show.values")],
            ),
            (
                [texts, REGISTRY["shout-lines"], show],
                [("texts.result", "shout-lines.text"), ("shout-lines.result", "show.values")],
            ),
            "show",
            "a,b;;c|d;e,f",
            [["A+B", None, "C"], ["D", "E+F"]],
        ),
        (
            ([documents, lines, show], [*SHOUT_EDGES[:1], ("lines.result", "show.values")]),
            ([REGISTRY["split-lines"], show], [("split-lines.result", "show.values")]),
            "show",
            "a,b;;c",
            ["a+b", None, "c"],
        ),
        (
            ([documents, lines, mark], [*SHOUT_EDGES[:1], ("lines.result", "mark.value")]),
            ([REGISTRY["split-lines"], mark], [("split-lines.result", "mark.value")]),
            "mark",
            "a,b;;c",
            [["a!", "b!"], None, ["c!"]],
        ),
        (
            (
                [documents, lines, shout, whisper, side_by_side],
                [
                    *WHISPER_EDGES,
                    ("shout.result", "side-by-side.loud"),
                    ("whisper.result", "side-by-side.quiet"),
                ],
            ),
            (
                [REGISTRY["shout-and-whisper"], side_by_side],
                [
                    ("shout-and-whisper.loud", "side-by-side.loud"),
                    ("shout-and-whisper.quiet", "side-by-side.quiet"),
                ],
            ),
            "side-by-side",
            "A,b;;c",
            ["A+B=a+b", None, "C=c"],
        ),
        (
            (
                [documents, lines, shout, whisper, title, pair],
                [*TITLE_EDGES, *_TITLE_READS],
            ),
            ([REGISTRY["titled"], pair], [("titled.loud", "pair.a"), ("titled.quiet", "pair.b")]),
            "pair",
            "A,b;;c",
            [["Aa", "Bb"], None, ["Cc"]],
        ),
        (
            (
                [documents, lines, shout, whisper, title, label],
                [*TITLE_EDGES, ("title.result", "label.heading"), ("shout.result", "label.values")],
            ),
            (
                [REGISTRY["titled"], label],
                [("titled.title", "label.heading"), ("titled.loud", "label.values")],
            ),
            "label",
            "a,b;;c",
            ["T:a,b=A+B", None, "T:c=C"],
        ),
    ],
    ids=[
        "a-series-input-outside-gets-one-documents-lines-per-call",
        "an-input-of-one-value-outside-is-called-once-per-line-of-each-document",
        "inside-an-outer-loop-a-node-outside-gets-one-documents-lines-per-call",
        "a-series-returned-once-per-document-into-a-series-input",
        "a-series-returned-once-per-document-into-an-input-of-one-value",
        "two-outputs-returned-per-line-are-collected-per-document-by-one-node",
        "one-item-of-each-of-two-outputs-of-one-loop-inside-reach-one-call-together",
        "one-value-per-document-beside-the-list-of-its-lines-reach-one-call-together",
    ],
)
def test_a_node_outside_a_subgraph_gets_what_it_would_get_in_one_graph(
    in_one_graph: tuple[list[Any], list[tuple[str, str]]],
    through_subgraph: tuple[list[Any], list[tuple[str, str]]],
    reader: str,
    text: str,
    expected: Any,
) -> None:
    assert (
        results(*in_one_graph, reader, text) == results(*through_subgraph, reader, text) == expected
    )
