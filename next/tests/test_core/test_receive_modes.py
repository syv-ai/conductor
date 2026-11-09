"""The record compile hands the run for each input: read against a small ledger, written out and read back, and a table on its own passed through."""


from typing import TypedDict

import polars as pl
from conductor import (
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    RunPlan,
    Series,
    Static,
    Table,
    node,
)
from conductor._internal.graph.receive_modes import (
    FixedValue,
    ReceiveMode,
    reads_of_input,
)
from pydantic import TypeAdapter


@node("words")
def words(text: str = "a b") -> Series[str]:
    return text.split()


@node("upper")
def upper(text: str) -> str:
    return text.upper()


@node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("records")
def records(n: int = 2) -> Table:
    return pl.DataFrame({"a": list(range(n))})


seen: list[pl.DataFrame] = []


class _Record(TypedDict):
    a: str


@node("typed_records")
def typed_records(n: int = 2) -> Table[_Record]:
    return pl.DataFrame({"a": [str(i) for i in range(n)]})


@node("hold")
def hold(table: Table) -> Table:
    seen.append(table)
    return table


@node("join_two_tables")
def join_two_tables(left: Series[str], right: str = "x") -> str:
    return right + "+".join(left)


REGISTRY = NodeRegistry([words, upper, join, records, typed_records, hold, join_two_tables])


def _gn(node_id, node_type, **bindings):
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def _compiled(*nodes):
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), REGISTRY)


def test_a_default_is_held_by_the_record():
    compiled = _compiled(_gn("w", "words"))
    mode = compiled.field("w.text")._receive_mode

    assert isinstance(mode, FixedValue) and mode.value == "a b"


def test_every_mode_lists_what_it_reads_at_which_depth():
    compiled = _compiled(_gn("w", "words"), _gn("up", "upper", text=From("w.result")), _gn("j", "join", texts=From("w.result")))

    assert reads_of_input(compiled.field("up.text")._receive_mode) == [("w.result", 1)]
    assert reads_of_input(compiled.field("j.texts")._receive_mode) == [("w.result", 0)]
    assert reads_of_input(compiled.field("w.text")._receive_mode) == []


def test_a_run_plan_written_out_reads_back_equal():
    compiled = _compiled(
        _gn("w", "words", text=Static("a b")), _gn("up", "upper", text=From("w.result")), _gn("j", "join", texts=From("up.result")),
        _gn("list", "upper", text=Static(["p", "q"])), _gn("r", "records"), _gn("h", "hold", table=From("r.result", "w.result")),
    )
    plan = compiled.run_plan

    assert RunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert TypeAdapter(ReceiveMode).validate_python({"kind": "fixed", "receive_mode": "entire_value"}).value is None


def test_the_value_a_fixed_value_holds_is_not_written_in_the_run_plan():
    frame = pl.DataFrame({"a": [1]})
    compiled = _compiled(_gn("h", "hold", table=Static(frame)))

    written = compiled.run_plan.model_dump(mode="json")["nodes"]["h"]["inputs"]["table"]
    assert written == {"kind": "fixed", "receive_mode": "same_value_every_row"}
    assert compiled.node("h").fingerprint is not None


def test_an_input_makes_no_index_of_its_own_and_still_says_it_holds_many():
    @node("one_text", category="test")
    def one_text() -> str:
        return "a"

    @node("takes_many", category="test")
    def takes_many(values: Series[str] = ("x",)) -> int:  # type: ignore[assignment]
        return len(values)

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            one_text.graph_node("a"), one_text.graph_node("b"),
            takes_many.graph_node("gathers", values=From("a.result", "b.result")), takes_many.graph_node("left_to_its_default"),
        ]),
        NodeRegistry([one_text, takes_many]),
    )
    assert "collected_onto" not in compiled.run_plan.model_dump(mode="json")["nodes"]["gathers"]["inputs"]["values"]
    for node_id in ("gathers", "left_to_its_default"):
        assert compiled.field(f"{node_id}.values").many and compiled.field(f"{node_id}.values")._index is None
