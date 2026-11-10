"""Silent wrong answers found by running real graphs, each pinned through the doors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    AskPerson,
    CompiledGraph,
    FromRun,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Policy,
    Result,
    RunState,
    Series,
    Table,
    node,
    table_with_columns,
)
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.graph.binding import From, Static
from helpers import Placed, placed, record, registry_of
from pydantic import BaseModel


@dataclass(frozen=True)
class _RunBy:
    name: str


def _compiled(*placed: Placed, nodes: list[Any]) -> CompiledGraph:
    return CompiledGraph.from_graph(record(*list(placed)), registry_of(*nodes))


class _Amount(TypedDict):
    amount: int


@node("whole-amount")
def _whole_amount() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.75]})


def test_a_fraction_in_an_int_column_fails_the_node_rather_than_being_cut_off() -> None:
    compiled = _compiled(placed("a", _whole_amount.id), nodes=[_whole_amount])
    state = compiled.run_sync()
    assert state.status == "failed"
    assert state.cause is not None and state.cause.type == "invalid_output"


@node("whole-number")
def _whole_number() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.0, None]})


def test_a_whole_number_held_as_a_float_reads_as_an_int() -> None:
    compiled = _compiled(placed("a", _whole_number.id), nodes=[_whole_number])
    assert compiled.results(compiled.run_sync())["a"]["result"]["amount"].to_list() == [12, None]


@node("late-records")
def _late_records() -> Table:
    return [{"n": 1}] * 150 + [{"n": 2.5, "late": "x"}]  # type: ignore[return-value]


def test_records_are_read_by_all_their_rows_not_the_first_hundred() -> None:
    compiled = _compiled(placed("a", _late_records.id), nodes=[_late_records])
    frame = compiled.results(compiled.run_sync())["a"]["result"]
    assert frame["n"].to_list()[-1] == 2.5
    assert frame["late"].to_list()[-1] == "x"


class _Inner(BaseModel):
    value: float


class _Reading(BaseModel):
    value: float
    maybe: float | None
    inner: _Inner


@node("reading")
def _reading() -> _Reading:
    return _Reading(value=math.nan, maybe=math.inf, inner=_Inner(value=-math.inf))


@node("read-reading")
def _read_reading(reading: _Reading) -> float:
    return reading.inner.value


def test_a_float_that_isnt_finite_in_a_model_survives_a_saved_state() -> None:
    compiled = _compiled(placed("a", _reading.id), placed("b", _read_reading.id, reading=From("a.result")), nodes=[_reading, _read_reading])
    state = RunState.model_validate_json(compiled.run_sync().model_dump_json())
    restored = compiled.results(state)["a"]["result"]
    assert math.isnan(restored.value)
    assert restored.maybe == math.inf
    assert restored.inner.value == -math.inf
    assert compiled.run_sync(state).status == "complete"


class _Rule:
    """A host's own value, set on a closed input: no pydantic, no dataclass."""


@node("with-rule")
def _with_rule(rule: Annotated[_Rule, Param(connectable=False)] = _Rule()) -> str:  # noqa: B008
    return "x"


@dataclass
class _Branch:
    title: str


@node("with-branches")
def _with_branches(branches: Annotated[list[_Branch], Param(connectable=False)] = [_Branch("a")]) -> str:  # noqa: B006, B008
    return "x"


def test_describe_writes_as_json_whatever_a_host_set_on_a_closed_input() -> None:
    compiled = _compiled(placed("a", _with_rule.id), placed("b", _with_branches.id), nodes=[_with_rule, _with_branches])
    written = compiled.describe().model_dump_json()
    assert '"title":"a"' in written
    registry_of(_with_rule, _with_branches).describe().model_dump_json()


@node("people")
def _people() -> Series[str]:
    return ["Ida", "Bo", "Nina"]


@node("skip-bo")
def _skip_bo(name: str) -> str:
    return SKIPPED if name == "Bo" else name.upper()  # type: ignore[return-value]


@node("skip-nina")
def _skip_nina(name: str) -> str:
    return SKIPPED if name == "Nina" else name.lower()  # type: ignore[return-value]


@node("zip-pairs")
def _zip_pairs(upper: Series[str | None], lower: Series[str | None]) -> str:
    return ",".join(f"{a}/{b}" for a, b in zip(upper, lower, strict=True))


@node("zip-only-full")
def _zip_only_full(upper: Series[str], lower: Series[str]) -> str:
    return ",".join(f"{a}/{b}" for a, b in zip(upper, lower, strict=True))


@node("zip-open")
def _zip_open(**columns: Series) -> str:
    return ",".join("/".join(str(cell) for cell in row) for row in zip(*columns.values(), strict=True))


def _two_columns_with_skips(reader: Any, **wired: Any) -> CompiledGraph:
    return _compiled(
        placed("p", _people.id),
        placed("up", _skip_bo.id, name=From("p.result")),
        placed("low", _skip_nina.id, name=From("p.result")),
        reader,
        nodes=[_people, _skip_bo, _skip_nina, _zip_pairs, _zip_only_full, _zip_open],
    )


def test_two_series_with_skips_on_different_rows_keep_their_rows_as_blanks() -> None:
    compiled = _two_columns_with_skips(placed("z", _zip_pairs.id, upper=From("up.result"), lower=From("low.result")))
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida,None/bo,NINA/None"


def test_a_row_skipped_on_one_series_leaves_every_series_that_takes_no_blank() -> None:
    compiled = _two_columns_with_skips(placed("z", _zip_only_full.id, upper=From("up.result"), lower=From("low.result")))
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida"


def test_an_open_series_input_keeps_every_edge_in_step() -> None:
    compiled = _two_columns_with_skips(placed("z", "zip-open", a=From("up.result"), b=From("low.result")))
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida,None/bo,NINA/None"


# -- a node that reads the world ----------------------------------------------


_FILE: dict[str, str] = {}
_CALLS: dict[str, int] = {}
_SEEN_ROWS: list[str] = []


@node("read-file", policy=Policy(reuse_when_inputs_match=False))
def _read_file(path: str) -> Series[str]:
    _CALLS["read"] = _CALLS.get("read", 0) + 1
    return _FILE[path].splitlines()


@node("mail-line")
def _mail_line(line: str) -> str:
    _SEEN_ROWS.append(line)
    return f"sent {line}"


@node("ask-line")
def _ask_line(line: str) -> str | AskPerson:
    return AskPerson()


def _file_flow(*extra: Placed, nodes: tuple[Any, ...] = ()) -> CompiledGraph:
    _CALLS.clear()
    _SEEN_ROWS.clear()
    return _compiled(placed("read", _read_file.id, path=Static("a.csv")), placed("mail", _mail_line.id, line=From("read.result")), *extra, nodes=[_read_file, _mail_line, *nodes])


def test_a_node_that_reads_the_world_is_called_again_on_a_new_leg_from_a_complete_state() -> None:
    _FILE["a.csv"] = "ida\nbo\n"
    compiled = _file_flow()
    state = compiled.run_sync()
    _FILE["a.csv"] = "ida\nbo\nnina\n"
    _SEEN_ROWS.clear()

    again = compiled.run_sync(RunState.model_validate_json(state.model_dump_json()))

    assert again.status == "complete" and _CALLS["read"] == 2
    assert compiled.results(again)["mail"]["result"]["values"] == ["sent ida", "sent bo", "sent nina"]
    assert _SEEN_ROWS == ["nina"]  # below it rows still reuse


def test_a_node_that_reads_the_world_is_not_read_again_when_a_run_goes_on_from_waiting() -> None:
    _FILE["a.csv"] = "ida\n"
    compiled = _file_flow(placed("ask", _ask_line.id, line=From("read.result")), nodes=(_ask_line,))
    paused = compiled.run_sync()
    assert paused.status == "waiting"
    _FILE["a.csv"] = "changed\n"

    done = compiled.run_sync(compiled.answer(paused, {paused.waiting[0].id: {"result": "ok"}}))

    assert done.status == "complete" and _CALLS["read"] == 1


def test_outputs_of_a_complete_state_are_still_read_for_a_node_that_reads_the_world() -> None:
    _FILE["a.csv"] = "ida\n"
    compiled = _file_flow()
    assert compiled.outputs(compiled.run_sync()) == {"mail.result": {"rows": [(0,)], "values": ["sent ida"]}}


# -- outputs that are alternatives ------------------------------------------------


class _Named(TypedDict):
    name: str


class _Route(TypedDict):
    route: str


@dataclass(frozen=True)
class _Routes:
    yes: Annotated[Table[_Named], Result(exclusive_group="r")]
    no: Annotated[Table[_Named], Result(exclusive_group="r")]


@node("route-once")
def _route_once(go: bool) -> _Routes:
    frame = pl.DataFrame({"name": ["taken"]})
    return _Routes(frame, SKIPPED) if go else _Routes(SKIPPED, frame)  # type: ignore[arg-type]


@node("flags")
def _flags() -> Series[bool]:
    return [True, False]


@node("show-table")
def _show_table(table: Table) -> str:
    return ",".join(table["name"].to_list()) + "|" + ",".join(name for name in table.columns if not name.startswith("__"))


@node("copy-table")
def _copy_table(table: Table[_Named]) -> Table[_Named]:
    return table


def test_alternative_tables_into_one_table_input_are_the_branch_that_ran() -> None:
    for go in (True, False):
        compiled = _compiled(placed("r", _route_once.id, go=Static(go)), placed("s", _show_table.id, table=From("r.yes", "r.no")), nodes=[_route_once, _show_table])
        assert compiled.run_plan.nodes["s"].inputs["table"].type == "cell"
        assert compiled.results(compiled.run_sync())["s"]["result"] == "taken|name"


def test_alternatives_per_row_take_the_branch_each_row_took() -> None:
    compiled = _compiled(placed("f", _flags.id), placed("r", _route_once.id, go=From("f.result")), placed("s", _show_table.id, table=From("r.yes", "r.no")), nodes=[_flags, _route_once, _show_table])
    assert compiled.results(compiled.run_sync())["s"]["result"]["values"] == ["taken|name", "taken|name"]


def test_what_hangs_off_two_branches_is_also_the_branch_that_ran() -> None:
    compiled = _compiled(
        placed("r", _route_once.id, go=Static(False)), placed("a", _copy_table.id, table=From("r.yes")), placed("b", _copy_table.id, table=From("r.no")),
        placed("s", _show_table.id, table=From("a.result", "b.result")), nodes=[_route_once, _copy_table, _show_table],
    )
    assert compiled.is_runnable, compiled.problems
    assert compiled.results(compiled.run_sync())["s"]["result"] == "taken|name"


# -- what a merged column is called -----------------------------------------------


class _Contact(TypedDict):
    name: str
    email: str


@node("contacts")
def _contacts() -> Table[_Contact]:
    return pl.DataFrame({"name": ["Ida", "Bo", "Nina"], "email": ["i", "b", "n"]})


@node("first-two")
def _first_two(contacts: Table[_Contact]) -> Table[_Contact]:
    """Keeps two rows, and writes one blank and one new value into a column it carries on."""
    return contacts.head(2).with_columns(pl.Series("email", [None, "B"]))


@node("sent-to")
def _sent_to(email: str) -> Annotated[str, Result(title="Sendt")]:
    return f"to {email}"


@node("shout")
def _shout(name: str) -> str:
    return name.upper()


@node("sheet-of")
def _sheet_of(table: Table) -> Table:
    return table


def _sheet_from(*nodes: Placed, wired: tuple[str, ...]) -> tuple[CompiledGraph, pl.DataFrame]:
    compiled = _compiled(*nodes, placed("sheet", _sheet_of.id, table=From(*wired)), nodes=[_contacts, _first_two, _sent_to, _shout, _sheet_of])
    return compiled, compiled.results(compiled.run_sync())["sheet"]["result"].drop("__row")


def test_a_column_a_derived_table_carries_is_one_column_beside_its_source() -> None:
    compiled, sheet = _sheet_from(placed("c", _contacts.id), placed("p", _first_two.id, contacts=From("c.result")), wired=("c.result", "p.result"))
    assert sheet.columns == ["name", "email"] == [name for name, _ in compiled.field("sheet.table").dtype.columns]
    assert sheet.to_dicts() == [{"name": "Ida", "email": None}, {"name": "Bo", "email": "B"}, {"name": "Nina", "email": "n"}]


def test_a_column_named_after_its_output_never_its_title() -> None:
    _, sheet = _sheet_from(placed("c", _contacts.id), placed("m", _sent_to.id, email=From("c.result.email")), wired=("c.result", "m.result"))
    assert sheet.columns == ["name", "email", "result"]


def test_a_real_clash_takes_the_outputs_name_in_wiring_order() -> None:
    _, sheet = _sheet_from(
        placed("c", _contacts.id), placed("m", _sent_to.id, email=From("c.result.email")), placed("loud", _shout.id, name=From("c.result.name")),
        wired=("c.result", "m.result", "loud.result"),
    )
    assert sheet.columns == ["name", "email", "result", "result_2"]


# -- placing a node, in the checker's eye ---------------------------------------------


@node("open-text")
def _open_text(text: str, suffix: str) -> str:
    return text + suffix


@node("who-runs")
def _who_runs(greeting: str, who: FromRun[_RunBy]) -> str:
    return f"{greeting} {who.name}"


def test_an_input_the_record_leaves_empty_is_the_callers_to_fill() -> None:
    compiled = _compiled(placed("t", _open_text.id, suffix=Static("!")), nodes=[_open_text])
    ready = compiled.with_inputs(text="hi")
    assert ready.outputs(ready.run_sync()) == {"t.result": "hi!"}


def test_a_from_run_input_is_left_empty_in_the_record() -> None:
    compiled = _compiled(placed("h", _who_runs.id, greeting=Static("hej")), nodes=[_who_runs])
    assert compiled.outputs(compiled.run_sync(from_run={_RunBy: _RunBy("Ida")})) == {"h.result": "hej Ida"}


def test_a_compiled_graph_does_not_offer_a_value_with_inputs_doesnt_take() -> None:
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node=_open_text.id, values={"text": "x", "suffix": "!"}),
        NodeRecord(id="b", node=_open_text.id),
    ], edges=[("a.result", "b.text")])
    compiled = CompiledGraph.from_graph(graph, registry_of(_open_text))
    with pytest.raises(InvalidCallError, match="not_offered"):
        compiled.with_inputs(**{"a.suffix": "!!", "a.nope": 1})


def test_a_table_typed_dict_may_say_a_column_can_be_blank() -> None:
    class Person(TypedDict):
        age: int | None

    assert Table[Person] is table_with_columns({"age": int})


# -- what an error says -------------------------------------------------------------


@node("three-letters")
def _three_letters() -> Series[str]:
    return ["a", "b", "c"]


@node("boom-at-one")
def _boom_at_one(n: str) -> str:
    if n == "b":
        raise ValueError("no")
    return n


def test_a_failed_runs_outputs_error_names_the_node_and_the_row() -> None:
    compiled = _compiled(placed("p", _three_letters.id), placed("r", _boom_at_one.id, n=From("p.result")), nodes=[_three_letters, _boom_at_one])
    state = compiled.run_sync()
    with pytest.raises(InvalidCallError, match=r"node 'r', row 1 \(ValueError\)"):
        compiled.outputs(state)


def test_an_unknown_param_argument_is_called_unknown() -> None:
    with pytest.raises(NodeDeclarationError, match="no argument 'titel'"):
        Param(titel="Text")  # type: ignore[call-arg]


def test_a_plan_reads_like_what_a_node_was_wired_to() -> None:
    compiled = _compiled(placed("c", _contacts.id), placed("loud", _shout.id, name=From("c.result.name")), nodes=[_contacts, _shout])
    assert "c.result.name" in str(compiled.run_plan)
    once = _compiled(placed("a", _open_text.id, text=Static("x"), suffix=Static("y")), placed("b", _open_text.id, text=From("a.result"), suffix=Static("!")), nodes=[_open_text])
    assert "text: a.result" in str(once.run_plan)


# -- series inputs on opposite branches -----------------------------------------------


@dataclass(frozen=True)
class _Parity:
    even: Annotated[int, Result(exclusive_group="parity")]
    odd: Annotated[int, Result(exclusive_group="parity")]


@node("numbers")
def _numbers() -> Series[int]:
    return [1, 2, 3, 4]


@node("parity")
def _parity(n: int) -> _Parity:
    return _Parity(n, SKIPPED) if n % 2 == 0 else _Parity(SKIPPED, n)  # type: ignore[arg-type]


@node("add-both")
def _add_both(a: Series[int], b: Series[int]) -> int:
    return sum(a) + sum(b)


@node("pair-both")
def _pair_both(a: Series[int | None], b: Series[int | None]) -> str:
    return ",".join(f"{x}/{y}" for x, y in zip(a, b, strict=True))


@node("add-all")
def _add_all(values: Series[int]) -> int:
    return sum(values)


_PARITY_NODES = [_numbers, _parity, _add_both, _pair_both, _add_all]


def test_series_inputs_on_opposite_branches_are_refused_where_every_row_would_leave() -> None:
    compiled = _compiled(placed("n", _numbers.id), placed("r", _parity.id, n=From("n.result")), placed("c", _add_both.id, a=From("r.even"), b=From("r.odd")), nodes=_PARITY_NODES)
    (told,) = compiled.problems
    assert (told.type, told.fatal, told.loc) == ("series_inputs_never_share_a_row", True, ("c", "a"))
    assert (told.ctx["a"], told.ctx["b"]) == ("a", "b")
    assert "Series[X | None]" in told.msg and not compiled.is_runnable


def test_series_inputs_on_opposite_branches_that_take_a_blank_keep_every_row() -> None:
    compiled = _compiled(placed("n", _numbers.id), placed("r", _parity.id, n=From("n.result")), placed("c", _pair_both.id, a=From("r.even"), b=From("r.odd")), nodes=_PARITY_NODES)
    assert compiled.results(compiled.run_sync())["c"]["result"] == "None/1,2/None,None/3,4/None"


def test_opposite_branches_through_one_input_are_one_column() -> None:
    compiled = _compiled(placed("n", _numbers.id), placed("r", _parity.id, n=From("n.result")), placed("c", _add_all.id, values=From("r.even", "r.odd")), nodes=_PARITY_NODES)
    assert compiled.results(compiled.run_sync())["c"]["result"] == 10


@node("plus-one")
def _plus_one(x: int) -> int:
    return x + 1


def test_series_inputs_reading_what_hangs_off_opposite_per_row_branches_are_refused() -> None:
    compiled = _compiled(
        placed("n", _numbers.id), placed("r", _parity.id, n=From("n.result")),
        placed("x", _plus_one.id, x=From("r.even")), placed("y", _plus_one.id, x=From("r.odd")), placed("c", _add_both.id, a=From("x.result"), b=From("y.result")),
        nodes=[*_PARITY_NODES, _plus_one],
    )
    assert [problem.type for problem in compiled.problems] == ["series_inputs_never_share_a_row"]


def test_a_node_reading_both_branches_through_one_input_is_on_neither() -> None:
    compiled = _compiled(
        placed("n", _numbers.id), placed("r", _parity.id, n=From("n.result")),
        placed("x", _plus_one.id, x=From("r.even", "r.odd")), placed("y", _plus_one.id, x=From("r.odd")), placed("c", _add_both.id, a=From("x.result"), b=From("y.result")),
        nodes=[*_PARITY_NODES, _plus_one],
    )
    assert compiled.is_runnable, compiled.problems
    assert compiled.results(compiled.run_sync())["c"]["result"] == 2 + 4 + 2 + 4  # rows 0 and 2: only those have both


def test_tables_hanging_off_per_row_branches_are_the_branch_that_ran() -> None:
    compiled = _compiled(
        placed("f", _flags.id), placed("r", _route_once.id, go=From("f.result")), placed("a", _copy_table.id, table=From("r.yes")), placed("b", _copy_table.id, table=From("r.no")),
        placed("s", _show_table.id, table=From("a.result", "b.result")), nodes=[_flags, _route_once, _copy_table, _show_table],
    )
    assert compiled.results(compiled.run_sync())["s"]["result"]["values"] == ["taken|name", "taken|name"]
