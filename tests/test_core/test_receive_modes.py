"""The record compile hands the run for each input: read against a small ledger, written out and read back, and a table on its own passed through."""


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
from conductor._internal.execution.run_ledger import RunLedger, SkippedAtRow
from conductor._internal.graph.receive_modes import (
    Cell,
    FixedValue,
    Gather,
    Merge,
    ReceiveMode,
    RowsUnder,
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


@node("hold")
def hold(table: Table) -> Table:
    seen.append(table)
    return table


@node("join_two_tables")
def join_two_tables(left: Series[str], right: str = "x") -> str:
    return right + "+".join(left)


REGISTRY = NodeRegistry([words, upper, join, records, hold, join_two_tables])


def _gn(node_id, node_type, **bindings):
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def _compiled(*nodes):
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), REGISTRY)


def _ledger_after(compiled, *units):
    """A ledger with these units recorded, each the way a run would."""
    ledger = RunLedger(compiled)
    ledger.units_ready_to_start()
    for unit, outputs in units:
        ledger.record(unit, outputs)
    return ledger


def test_cell_is_the_value_at_the_row_cut_to_the_fields_depth():
    compiled = _compiled(_gn("w", "words"), _gn("up", "upper", text=From("w.result")))
    mode = compiled.field("up.text")._receive_mode
    ledger = _ledger_after(compiled, (("w", ()), {"result": ["a", "b"]}))

    assert isinstance(mode, Cell) and (mode.receive_mode, mode.rows_of, mode.depth) == ("value_at_row", "w.result", 1)
    assert ledger.read_input(mode, ("up", (1,))) == "b"


def test_a_value_on_a_node_is_held_by_the_record():
    compiled = _compiled(_gn("up", "upper", text=Static("hello")))
    mode = compiled.field("up.text")._receive_mode

    assert isinstance(mode, FixedValue) and mode.value == "hello" and mode.receive_mode == "same_value_every_row"
    assert RunLedger(compiled).read_input(mode, ("up", ())) == "hello"


def test_a_default_is_held_by_the_record():
    compiled = _compiled(_gn("w", "words"))
    mode = compiled.field("w.text")._receive_mode

    assert isinstance(mode, FixedValue) and mode.value == "a b"


def test_a_list_on_a_node_reads_as_an_edge_reads_it():
    compiled = _compiled(_gn("up", "upper", text=Static(["p", "q"])), _gn("after", "upper", text=From("up.text")))
    mode = compiled.field("up.text")._receive_mode
    ledger = RunLedger(compiled)

    assert isinstance(mode, Cell) and mode.sources == ("up.text",)
    assert ledger.read_input(mode, ("up", (1,))) == "q"
    assert ledger.read_input(compiled.field("after.text")._receive_mode, ("after", (1,))) == "q"


def test_rows_under_gives_the_rows_under_the_unit_or_all_of_them():
    compiled = _compiled(_gn("w", "words"), _gn("j", "join", texts=From("w.result")))
    mode = compiled.field("j.texts")._receive_mode
    ledger = _ledger_after(compiled, (("w", ()), {"result": ["a", "b"]}))

    assert isinstance(mode, RowsUnder) and (mode.receive_mode, mode.depth) == ("entire_value", 0)
    assert list(ledger.read_input(mode, ("j", ()))) == ["a", "b"]


def test_gather_collects_under_the_row_it_was_given():
    compiled = _compiled(_gn("a", "upper", text=Static("x")), _gn("b", "upper", text=Static("y")), _gn("j", "join", texts=From("a.result", "b.result")))
    mode = compiled.field("j.texts")._receive_mode
    ledger = _ledger_after(compiled, (("a", ()), {"result": "X"}), (("b", ()), {"result": "Y"}))

    assert isinstance(mode, Gather)
    assert list(ledger.read_input(mode, ("j", ()))) == ["X", "Y"]


def test_a_merge_reads_each_source_as_its_own_record():
    frame = pl.DataFrame({"__row": ["r.result:0"], "a": [1]})
    compiled = _compiled(_gn("w", "words"), _gn("r", "records"), _gn("h", "hold", table=From("r.result", "w.result")))
    mode = compiled.field("h.table")._receive_mode
    ledger = _ledger_after(compiled, (("r", ()), {"result": frame}), (("w", ()), {"result": ["x"]}))

    assert isinstance(mode, Merge) and [part.kind for part in mode.source_plans] == ["cell", "rows_under"]
    assert ledger.read_input(mode, ("h", ())).columns == ["__row", "a", "result"]


def test_a_skip_comes_back_with_the_row_it_was_found_at():
    compiled = _compiled(_gn("w", "words"), _gn("up", "upper", text=From("w.result")))
    mode = compiled.field("up.text")._receive_mode
    ledger = _ledger_after(compiled, (("w", ()), SkippedAtRow(row=())))

    assert ledger.read_input(mode, ("up", (0,))) == SkippedAtRow(row=())


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


def test_a_table_wired_in_alone_passes_through_as_the_same_frame():
    seen.clear()
    compiled = _compiled(_gn("r", "records"), _gn("first", "hold", table=From("r.result")), _gn("second", "hold", table=From("first.result")))
    mode = compiled.field("first.table")._receive_mode

    assert isinstance(mode, Cell)
    state = compiled.run_sync()
    assert state.status == "complete"
    assert seen[0] is seen[1]


def test_a_value_set_on_a_series_input_is_a_series_on_its_own_index():
    compiled = _compiled(_gn("j", "join", texts=Static(["a", "b"])))
    mode = compiled.field("j.texts")._receive_mode

    assert isinstance(mode, FixedValue) and mode.receive_mode == "entire_value"
    assert list(RunLedger(compiled).read_input(mode, ("j", ()))) == ["a", "b"]


def test_the_value_a_fixed_value_holds_is_not_written_in_the_run_plan():
    frame = pl.DataFrame({"a": [1]})
    compiled = _compiled(_gn("h", "hold", table=Static(frame)))

    written = compiled.run_plan.model_dump(mode="json")["nodes"]["h"]["inputs"]["table"]
    assert written == {"kind": "fixed", "receive_mode": "same_value_every_row", "takes_blank": False}
    assert compiled.node("h").fingerprint is not None
