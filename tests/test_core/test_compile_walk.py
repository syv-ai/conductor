"""Compile as one walk, and ``compiled.run_plan``: what it says, and what a graph with a fault looks like."""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import (
    Arriving,
    AsItArrives,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    Input,
    InvalidCallError,
    NodeDefinition,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Series,
    Static,
    Table,
    node,
)


class Txt(DType, str):
    id = "txt"


class Num(DType, float):
    id = "num"


READS: list[str] = []


class CountedTxt(DType, str):
    id = "counted"

    def __new__(cls, value: str):
        READS.append(value)
        return super().__new__(cls, value)


@node("upload")
def upload() -> Series[Txt]:
    return [Txt("a.pdf"), Txt("b.pdf")]


@node("split")
def split(text: Txt) -> Series[Txt]:
    return [] if text == "b.pdf" else [Txt("l1"), Txt("l2")]


@node("count")
def count(lines: Series[Txt]) -> Num:
    return Num(len(lines))


@node("note")
def note(text: Txt, doc: Annotated[Txt, Param()] = Txt("")) -> Txt:
    return Txt(text + doc)


@node("excel")
def excel(table: Table) -> Txt:
    return Txt(str(table.frame.height))


@node("table")
def make_table() -> Table:
    import polars as pl
    return Table(pl.DataFrame({"x": [1, 2]}))


@node("rebuild")
def rebuild(**inputs: AsItArrives) -> Table:
    import polars as pl
    return Table(pl.DataFrame({"y": [1, 2]}))


@node("rebuild-keeping-ids")
def rebuild_keeping_ids(**inputs: AsItArrives) -> Table:
    return Table(inputs["t"].frame)


def _table_out(t: Arriving) -> list[Output]:
    return [Output("result", Table)]


@node("from-any", compute_outputs=_table_out)
def from_any(t: Any) -> Mapping[str, Any]:
    import polars as pl
    return {"result": Table(pl.DataFrame({"y": [1, 2]}))}


@node("hooked", compute_inputs=lambda: [Input("extra", Txt)])
def hooked(**more: Any) -> Txt:
    return Txt("x")


NODES = [upload, split, count, note, excel, make_table, rebuild, rebuild_keeping_ids, from_any, hooked]
REGISTRY = NodeRegistry(NODES)


def _compiled(*nodes: GraphNode, registry: NodeRegistry = REGISTRY) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def _node(id: str, type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=id, type=type, version=1, bindings=bindings)


FLOW = (
    _node("upload", "upload"),
    _node("split", "split", text=From("upload.result")),
    _node("count", "count", lines=From("split.result")),
    _node("excel", "excel", table=From("upload.result", "count.result")),
)


def test_the_run_plan_says_where_each_node_runs_and_what_each_input_receives():
    plan = _compiled(*FLOW).run_plan

    assert plan.nodes["upload"].runs_per_row_of is None and plan.nodes["upload"].makes_rows == (Ref("upload.result"),)
    assert plan.nodes["split"].runs_per_row_of == "upload.result"
    assert plan.nodes["split"].inputs["text"].arrives == "value_at_row"
    assert plan.nodes["split"].inputs["text"].rows_of == "upload.result"
    assert plan.nodes["count"].runs_per_row_of == "upload.result"
    counted = plan.nodes["count"].inputs["lines"]
    assert (counted.arrives, counted.rows_of, counted.grouped_under) == ("all_rows_under_row", "split.result", "upload.result")
    merged = plan.nodes["excel"].inputs["table"]
    assert merged.arrives == "merged"
    assert [(part.sources, part.arrives) for part in merged.parts] == [((Ref("upload.result"),), "whole"), ((Ref("count.result"),), "whole")]


def test_a_parent_merged_with_rows_under_it_is_copied_down():
    plan = _compiled(
        _node("upload", "upload"),
        _node("split", "split", text=From("upload.result")),
        _node("excel", "excel", table=From("upload.result", "split.result")),
    ).run_plan

    assert [part.copied_down for part in plan.nodes["excel"].inputs["table"].parts] == [True, False]
    assert "table: merged: upload.result [copied down] + split.result" in str(plan)


def test_the_run_plan_reads_as_text():
    text = str(_compiled(*FLOW).run_plan)

    assert "upload" in text and "once" in text
    assert "text: the value at my row (upload.result)" in text
    assert "lines: all rows of split.result under my row" in text and "called with an empty list" in text
    assert "table: merged: upload.result + count.result" in text


def test_the_run_plan_says_a_blank_and_a_typed_in_value():
    @node("blank")
    def blank(a: Txt | None, b: Txt = Txt("b")) -> Txt:
        return b

    plan = _compiled(_node("b", "blank", a=From("upload.result")), _node("upload", "upload"), registry=NodeRegistry([upload, blank])).run_plan

    assert plan.nodes["b"].inputs["a"].takes_blank
    assert (plan.nodes["b"].inputs["b"].arrives, plan.nodes["b"].inputs["b"].takes_blank) == ("same_value_every_row", False)


def test_a_table_part_at_a_nodes_own_row_and_a_deep_column_are_said():
    @node("deep")
    def deep(doc: Txt, lines: Series[Txt]) -> Txt:
        return doc

    registry = NodeRegistry([upload, split, deep])
    plan = _compiled(
        _node("upload", "upload"),
        _node("para", "split", text=From("upload.result")),
        _node("line", "split", text=From("para.result")),
        _node("d", "deep", doc=From("upload.result"), lines=From("line.result")),
        registry=registry,
    ).run_plan

    assert plan.nodes["d"].runs_per_row_of == "upload.result"
    lines = plan.nodes["d"].inputs["lines"]
    assert (lines.arrives, lines.rows_of, lines.grouped_under) == ("all_rows_under_row", "line.result", "upload.result")


def test_a_node_that_is_not_ready_is_absent_from_the_run_plan():
    compiled = _compiled(_node("a", "split", text=From("ghost.x")))

    assert "a" not in compiled.run_plan.nodes
    assert compiled.node("a").wiring is None and compiled.node("a").cause.type == "unknown_ref_node"


def test_a_graph_with_faults_says_so_by_what_each_node_lacks():
    compiled = _compiled(
        _node("gone", "nope"),
        _node("after", "split", text=From("gone.result")),
    )

    assert compiled.node("gone").interface is None and compiled.node("gone").state == "resolution_failed"
    assert compiled.node("after").interface is not None and compiled.node("after").wiring is None
    assert compiled.node("after").state == "wiring_failed"
    assert compiled.node("after").cause.type == "unknown_node_type"
    with pytest.raises(KeyError, match="no fields compile could work out"):
        compiled.field(Ref("after", "text"))


def test_a_series_of_any_into_a_typed_or_table_input_is_a_problem_not_a_crash():
    @node("any-series", compute_outputs=lambda: [Output("result", Series[Any])])
    def any_series() -> Mapping[str, Any]:
        return {"result": []}

    registry = NodeRegistry([any_series, split, excel])
    for target, field in (("split", "text"), ("excel", "table")):
        compiled = _compiled(_node("a", "any-series"), _node("b", target, **{field: From("a.result")}), registry=registry)
        assert [p.type for p in compiled.problems] == ["type_mismatch"]


def test_a_static_is_read_once():
    READS.clear()

    @node("reads-one")
    def reads_one(value: CountedTxt) -> Txt:
        return Txt(value)

    _compiled(_node("a", "reads-one", value=Static("x")), registry=NodeRegistry([reads_one]))

    assert READS == ["x"]


def test_a_series_of_a_table_subclass_is_refused_as_any_series_of_tables():
    class Sheet(Table):
        id = "sheet"

    @node("sheets", compute_outputs=lambda: [Output("result", Series[Sheet])])
    def sheets() -> Mapping[str, Any]:
        return {"result": []}

    compiled = _compiled(_node("a", "sheets"), registry=NodeRegistry([sheets], types=[Sheet]))

    assert [p.type for p in compiled.problems] == ["series_of_tables"]


def test_a_graph_takes_any_sequence_of_nodes():
    assert Graph(nodes=[_node("a", "upload")]).nodes == Graph(nodes=(_node("a", "upload"),)).nodes
    with pytest.raises(ValueError):
        Graph(nodes="ab")


def test_from_compiled_graphs_takes_one_graph_as_version_one():
    compiled = _compiled(_node("a", "upload"))

    definition = NodeDefinition.from_compiled_graphs("flow", title="F", description="", versions=compiled)

    assert definition.versions == {1: compiled} and definition.current == 1


def test_with_inputs_takes_an_address_as_a_mapping_beside_keywords():
    compiled = _compiled(_node("a", "note"), _node("b", "note", text=From("a.result")))

    filled = compiled.with_inputs({"a.text": "x"}, doc="y")

    assert filled.node("a").statics == {"text": "x", "doc": "y"}
    with pytest.raises(InvalidCallError):
        compiled.with_inputs({"zzz.text": "x"})


def test_a_table_rebuilt_in_an_open_input_node_and_merged_with_its_source_loses_its_row_ids():
    compiled = _compiled(
        _node("t", "table"),
        _node("py", "rebuild", t=From("t.result")),
        _node("excel", "excel", table=From("t.result", "py.result")),
    )

    assert Ref("py.result") in compiled._row_ids_must_survive
    state = compiled.run_sync()
    assert state.status == "failed" and state.failure.type == "row_ids_lost" and state.failure.loc[0] == "py"


def test_a_table_rebuilt_with_its_ids_kept_passes():
    compiled = _compiled(
        _node("t", "table"),
        _node("py", "rebuild-keeping-ids", t=From("t.result")),
        _node("excel", "excel", table=From("t.result", "py.result")),
    )

    state = compiled.run_sync()
    assert state.status == "complete", state.failure


def test_a_table_reaching_a_typed_any_input_counts_too():
    compiled = _compiled(
        _node("t", "table"),
        _node("a", "from-any", t=From("t.result")),
        _node("excel", "excel", table=From("t.result", "a.result")),
    )

    assert Ref("a.result") in compiled._row_ids_must_survive


def test_an_untyped_input_left_empty_inside_a_graph_cant_be_embedded():
    @node("any-in", compute_outputs=lambda: [Output("r", Txt)])
    def any_in(value: Any) -> Mapping[str, Any]:
        return {"r": Txt("x")}

    registry = NodeRegistry([any_in, split, upload])
    inner = _compiled(_node("x", "any-in"), registry=registry)
    emb = NodeDefinition.from_compiled_graphs("emb", title="e", description="", versions=inner)
    extended = registry.extended_with([emb])

    compiled = _compiled(_node("s", "split", text=Static("t")), _node("e", "emb", **{"x.value": From("s.result")}), registry=extended)

    assert [p.type for p in compiled.problems] == ["embedded_graph_broken"]
    assert compiled.node("e").wiring is None


def test_a_hook_refusing_leaves_a_node_that_is_only_known_by_its_problem():
    from conductor import CompileProblem

    def refuse():
        raise CompileProblem("nope", "Nej.", field=None)

    @node("refused", compute_inputs=refuse)
    def refused(**more: Any) -> Txt:
        return Txt("")

    compiled = _compiled(_node("r", "refused"), registry=NodeRegistry([refused]))

    assert compiled.node("r").interface is None and compiled.node("r").version is not None
    assert compiled.node("r").cause.type == "nope"
