"""``Series[X]``: the word an author writes for many values of one type, and what a node gets for it."""

from typing import Annotated, Any

import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeRegistry,
    Output,
    Param,
    Series,
    Static,
    node,
)
from conductor._internal.dtype import read_field_annotation
from conductor._internal.index import Index


class Text(DType, str):
    id = "series-test-text"
    display_title = "Text"


class Num(DType, float):
    id = "series-test-num"
    display_title = "Number"


# --- the word ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "read"),
    [
        (Text, (Text, False, False)),
        (Text | None, (Text, False, True)),
        (Series[Text], (Text, True, False)),
        (Series[Text | None], (Text, True, True)),
        (Series[Text] | None, (Text, True, True)),
        (Series[Any], (Any, True, False)),
        (Series, (Any, True, False)),
    ],
)
def test_what_a_field_annotation_says(written, read):
    assert read_field_annotation(written) == read


def test_a_series_is_a_word_and_never_a_value():
    with pytest.raises(TypeError):
        Series[Text](["a"])
    with pytest.raises(TypeError):
        isinstance([], Series)


def test_a_field_holds_the_type_of_one_value_and_says_it_is_many():
    assert Input("x", Series[Text]).model_dump(include={"dtype", "many", "takes_blank"}, mode="json") == {"dtype": {"id": "series-test-text"}, "many": True, "takes_blank": False}
    assert Output("x", Series[Text | None]).model_dump(include={"dtype", "many", "takes_blank"}, mode="json") == {"dtype": {"id": "series-test-text"}, "many": True, "takes_blank": True}
    assert Input("x", Text).many is False


def test_a_series_of_series_does_not_exist():
    with pytest.raises(NodeDeclarationError) as raised:
        Input("x", Series[Series[Text]])
    assert raised.value.code == "nested_series"


def test_a_series_holds_what_an_edge_can_carry():
    with pytest.raises(NodeDeclarationError) as raised:
        Output("x", Series[list[int]])
    assert raised.value.code == "series_element_invalid"


def test_a_plain_list_stays_one_value():
    assert Input("x", list[float]).many is False
    assert Input("x", list[float]).connectable is False


# --- what a node receives ----------------------------------------------------


@node("split-series")
def split_series(text: Text) -> Series[Text]:
    return tuple(Text(part) for part in text.split(","))


@node("collect")
def collect(parts: Annotated[Series[Text], Param(title="Parts")]) -> Text:
    assert type(parts) is list and all(type(part) is Text for part in parts)
    return Text("+".join(parts))


@node("collect-with-default")
def collect_with_default(parts: Series[Text] | None = None) -> Text:
    return Text(repr(parts))


REGISTRY = NodeRegistry([split_series, collect, collect_with_default])


def _run(*nodes: GraphNode) -> dict[str, Any]:
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), REGISTRY)
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())


def test_a_scalar_is_not_silently_wrapped_into_many():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="c", type="collect", version=1, bindings={"parts": Static("x")})]), REGISTRY)
    assert [problem.type for problem in compiled.problems] == ["invalid_static"]


def test_a_field_on_rows_is_many_and_its_type_is_one_value():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="s", type="split-series", version=1, bindings={"text": Static("a")}), GraphNode(id="c", type="collect", version=1, bindings={"parts": From("s.result")})]),
        REGISTRY,
    )
    field = compiled.node("s")._wiring.fields["result"]
    assert (field.type, field.many) == (Text, True)


# --- an index ----------------------------------------------------------------


def test_a_root_index_has_no_parent():
    assert Index("docs").parent is None and Index("docs").depth == 1


def test_an_index_is_its_identity_and_lineage():
    docs = Index("docs")
    lines = Index("lines", parent=docs)
    assert lines.depth == 2 and lines.lineage == (docs, lines)
    assert Index("lines") == lines and hash(Index("lines")) == hash(lines)
    assert Index("docs") != lines
