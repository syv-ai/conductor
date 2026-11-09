"""A leg going on from a state, over a changed graph or table, reuses what it can.

A node nothing above has changed is restored whole. A changed node's units run
again, except a unit whose inputs match one an earlier unit of the node
received: its result is written at its new row, and the node isn't called. So
one row added to a CSV of 5,000 sends one mail.
"""

import asyncio
from collections import Counter
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    From,
    FromRun,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeRegistry,
    Param,
    Policy,
    Result,
    RunState,
    Series,
    Static,
    Table,
    Upgrade,
    node,
)
from conductor._internal.execution.input_hash import hash_of_inputs
from conductor._internal.execution.run_state import OutputColumn
from conductor._internal.ref import Ref
from conductor.events import NodeComplete, NodeProgress, NodeStart, UnitWaiting
from helpers import answer

from test_core.embedded import embedded_graph_node


class Contact(TypedDict):
    name: str
    email: str


CALLS: Counter = Counter()
FAILING = {"address": ""}


@node("csv", category="test")
def csv(rows: Annotated[str, Param(title="Rows", connectable=False)]) -> Annotated[Table[Contact], Result(title="Contacts")]:
    parsed = [line.split("|") for line in rows.splitlines() if line]
    return pl.DataFrame({"name": [p[0] for p in parsed], "email": [p[1] for p in parsed]}, schema={"name": pl.String, "email": pl.String})


@node("mail", category="test")
def mail(email: str, body: str) -> Annotated[str, Result(title="sent")]:
    CALLS["mail"] += 1
    CALLS["mail to " + email] += 1
    return f"sent:{email}:{body}"


@node("ask", category="test")
def ask(email: str) -> Annotated[str, Result(title="ok")]:
    CALLS["ask"] += 1
    return AskPerson(prompt=email)


@node("export", category="test")
def export(table: Table) -> Annotated[int, Result(title="Rows")]:
    CALLS["export"] += 1
    return table.height


@node("split", category="test")
def split(text: str) -> Annotated[Series[str], Result(title="lines")]:
    CALLS["split"] += 1
    return text.split(",")


@node("per", category="test")
def per(line: str) -> Annotated[str, Result(title="x")]:
    CALLS["per"] += 1
    return line.upper()


@node("count", category="test")
def count(lines: Series[str]) -> Annotated[int, Result(title="n")]:
    CALLS["count"] += 1
    return len(lines)


@node("doc_table", category="test")
def doc_table(text: str) -> Annotated[Table[Contact], Result(title="Rows")]:
    CALLS["doc_table"] += 1
    return pl.DataFrame({"name": [text], "email": [text + "@x"]}, schema={"name": pl.String, "email": pl.String})


@node("rate", category="test", policy=Policy(reuse_when_inputs_match=False))
def rate(email: str) -> Annotated[str, Result(title="r")]:
    CALLS["rate"] += 1
    return email


@node("fail_on", category="test", policy=Policy(concurrency=1))
def fail_on(email: str) -> Annotated[str, Result(title="sent")]:
    CALLS["fail_on"] += 1
    CALLS["fail_on " + email] += 1
    if email == FAILING["address"]:
        raise RuntimeError("the mail server is down")
    return "ok:" + email


@node("note_dict", category="test")
def note_dict(email: str, stuff: dict[str, Any]) -> Annotated[str, Result(title="n")]:
    CALLS["note_dict"] += 1
    return email




@node("parse", category="test")
def parse(contacts: str = "") -> Annotated[Table[Contact], Result(title="Contacts")]:
    return csv(rows=contacts)


registry = NodeRegistry([csv, parse, mail, ask, export, split, per, count, doc_table, rate, fail_on, note_dict])


def n(node_id: str, node_type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def mail_graph(rows: str, body: str = "Hej") -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("m", "mail", email=From("c.result.email"), body=Static(body)), n("x", "export", table=From("c.result", "m.result"))]), registry)


def through_json(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


BASE = "a|a@x\nb|b@x\nc|c@x\n"


@pytest.fixture(autouse=True)
def _no_calls_left_over():
    CALLS.clear()
    FAILING["address"] = ""


def first_leg(rows: str = BASE, body: str = "Hej") -> RunState:
    state = mail_graph(rows, body).run_sync()
    CALLS.clear()
    return state


# -- the seven changes to a CSV ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edited", "body", "mails", "reused"),
    [
        pytest.param(BASE + "d|d@x\n", "Hej", 1, 3, id="append a row"),
        pytest.param("z|z@x\n" + BASE, "Hej", 1, 3, id="insert a row at the top"),
        pytest.param("c|c@x\nb|b@x\na|a@x\n", "Hej", 0, 3, id="sort"),
        pytest.param("a|a@x\nc|c@x\n", "Hej", 0, 2, id="delete a row"),
        pytest.param("a|a@x\nb|B@x\nc|c@x\n", "Hej", 1, 2, id="edit one email"),
        pytest.param("A|a@x\nb|b@x\nc|c@x\n", "Hej", 0, 3, id="edit one name, not wired into the mail"),
        pytest.param(BASE, "Hej2", 3, 0, id="change the body static"),
    ],
)
def test_a_changed_csv_sends_only_the_mails_it_has_to(edited: str, body: str, mails: int, reused: int) -> None:
    state = first_leg()
    compiled = mail_graph(edited, body)

    ending = compiled.run_sync(through_json(state))

    assert CALLS["mail"] == mails and ending.reused_units.get("m", 0) == reused
    assert ending.status == "complete" and compiled.results(ending)["x"]["result"] == edited.count("\n")
    # the result of a reused unit is at its new row, and all of them are
    assert [v.split(":")[1] for v in compiled.results(ending)["m"]["result"]["values"]] == [line.split("|")[1] for line in edited.splitlines()]


def test_the_common_case_with_inputs_on_a_compiled_graph() -> None:
    compiled = CompiledGraph.from_graph(Graph(nodes=[n("p", "parse", contacts=Static(BASE)), n("m", "mail", email=From("p.result.email"), body=Static("Hej"))]), registry).with_inputs(contacts=BASE)
    state = compiled.run_sync()
    CALLS.clear()

    ending = compiled.with_inputs(contacts=BASE + "d|d@x\n").run_sync(state)

    assert CALLS["mail"] == 1 and ending.reused_units == {"m": 3}


def test_identical_rows_never_share_a_result() -> None:
    state = first_leg("a|d@x\nb|d@x\n")  # the same input twice, called twice

    ending = mail_graph("a|d@x\nb|d@x\nc|d@x\n").run_sync(through_json(state))

    assert CALLS["mail"] == 1 and ending.reused_units["m"] == 2  # the third identical row gets a call of its own


def test_a_duplicate_taken_out_keeps_the_other_rows_result() -> None:
    state = first_leg("a|d@x\nb|d@x\nc|d@x\n")

    ending = mail_graph("b|d@x\nc|d@x\n").run_sync(through_json(state))

    assert CALLS["mail"] == 0 and ending.reused_units["m"] == 2


def test_a_static_edit_is_seen_because_a_static_is_an_input() -> None:
    state = first_leg()

    ending = mail_graph(BASE, body="Ny").run_sync(state)

    assert CALLS["mail"] == 3 and "m" not in ending.reused_units


def test_a_changed_node_function_is_new_work() -> None:
    state = first_leg()

    def mail_v2(email: str, body: str) -> Annotated[str, Result(title="sent")]:
        CALLS["mail"] += 1
        return "v2"

    newer = node("mail", category="test", replaces=node("mail")(mail.versions[1].body), upgrade=Upgrade(renamed_outputs={"result": "result"}))(mail_v2)
    graph = Graph(nodes=[n("c", "csv", rows=Static(BASE)), GraphNode(id="m", type="mail", version=2, bindings={"email": From("c.result.email"), "body": Static("Hej")})])
    ending = CompiledGraph.from_graph(graph, NodeRegistry([csv, newer])).run_sync(through_json(state))

    assert CALLS["mail"] == 3 and "m" not in ending.reused_units


# -- what the key sees -------------------------------------------------------------------------------------------------


def test_a_node_nothing_changed_is_restored_with_no_hashing_and_no_reuse_count() -> None:
    state = first_leg()

    ending = mail_graph(BASE).run_sync(through_json(state))

    assert CALLS["mail"] == 0 and ending.reused_units == {}


def test_a_series_reordered_is_new_work_and_one_unchanged_is_not() -> None:
    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("cnt", "count", lines=From("c.result.email"))]), registry)

    state = graph(BASE).run_sync()
    CALLS.clear()
    assert graph(BASE).run_sync(state).reused_units == {} and CALLS["count"] == 0  # unchanged: restored

    resorted = "c|c@x\nb|b@x\na|a@x\n"
    assert graph(resorted).run_sync(through_json(state)).reused_units == {} and CALLS["count"] == 1
    CALLS.clear()
    assert graph("A|a@x\nB|b@x\nC|c@x\n").run_sync(through_json(state)).reused_units == {"cnt": 1} and CALLS["count"] == 0  # another column changed: the emails are the same series


def test_a_whole_table_input_is_read_with_its_row_ids() -> None:
    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("x", "export", table=From("c.result"))]), registry)

    state = graph(BASE).run_sync()
    CALLS.clear()

    ending = graph("z|z@x\n" + BASE).run_sync(through_json(state))

    assert CALLS["export"] == 1 and "x" not in ending.reused_units  # the row ids shifted, and it was given them
    CALLS.clear()
    assert graph(BASE).run_sync(through_json(state)).reused_units == {} and CALLS["export"] == 0


def test_a_table_read_back_from_json_has_the_digest_it_was_written_under() -> None:
    """The same content identity a state files tables under is a key's: ids included, however the frame got built."""
    from conductor._internal.table import ROW_ID_COLUMN, digest_of_table

    ids = pl.Series(ROW_ID_COLUMN, ["c:0", "c:1"], dtype=pl.String)
    built = pl.DataFrame([ids, pl.Series("a", ["x", "y"])])
    sorted_frame = built.sort("a")
    concatenated = pl.concat([built[:1], built[1:]], rechunk=False)
    sliced = pl.concat([pl.DataFrame({ROW_ID_COLUMN: ["c:9"], "a": ["q"]}), built]).slice(1, 2)
    assert {digest_of_table(frame) for frame in (built, sorted_frame, concatenated, sliced)} == {digest_of_table(built)}
    assert digest_of_table(built.with_columns(pl.Series(ROW_ID_COLUMN, ["c:0", "c:7"]))) != digest_of_table(built)


def test_long_values_read_off_a_table_never_share_a_hash_through_their_address() -> None:
    """Two different long strings from table cells, each garbage after its unit: an address held after the value is gone would hand the second the first's digest."""
    long = "x" * 600
    rows = "".join(f"{long}{index}|{long}{index}@x\n" for index in range(5))
    state = first_leg(rows)

    ending = mail_graph(rows.replace(f"{long}0@x", f"{long}7@x")).run_sync(through_json(state))

    assert CALLS["mail"] == 1 and ending.reused_units["m"] == 4


def test_a_value_with_no_json_form_is_called_and_never_fails_the_leg() -> None:
    class NoJson:
        pass

    seed = b"seed"
    assert hash_of_inputs(seed, {"a": NoJson()}, {}) != hash_of_inputs(seed, {"a": NoJson()}, {})  # nothing to compare it to, so it is new work


def test_what_a_value_is_stays_apart_in_the_hash() -> None:
    def hash_of(**inputs: Any) -> str:
        return hash_of_inputs(b"seed", inputs, {})

    values = [hash_of(a="ab", b="c"), hash_of(a="a", b="bc"), hash_of(a=1, b=None), hash_of(a="1", b=None), hash_of(a=1.0, b=None), hash_of(a=True, b=None), hash_of(a=None, b=None)]
    assert len(set(values)) == len(values)
    assert hash_of(a=1) == hash_of(a=1) and hash_of_inputs(b"other", {"a": 1}, {}) != hash_of(a=1)


def test_many_values_are_their_values_in_order() -> None:
    assert hash_of_inputs(b"s", {"x": ["a", "b"]}, {}) != hash_of_inputs(b"s", {"x": ["b", "a"]}, {})
    assert hash_of_inputs(b"s", {"x": ["a"]}, {}) != hash_of_inputs(b"s", {"x": ["ab"]}, {})


def test_a_value_that_reaches_every_row_is_read_once_and_its_memo_is_tied_to_the_value() -> None:
    memo: dict = {}
    value = {"a": 1}
    hash_of_inputs(b"s", {"x": value}, memo)
    hash_of_inputs(b"s", {"x": value}, memo)
    assert len(memo) == 1 and next(iter(memo.values()))[0] is value


def test_a_list_or_mapping_is_identified_by_its_content() -> None:
    def graph(stuff: Any) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(BASE)), n("no", "note_dict", email=From("c.result.email"), stuff=Static(stuff))]), registry)

    state = graph({"a": [1, 2]}).run_sync()
    CALLS.clear()
    assert graph({"a": [1, 2]}).run_sync(through_json(state)).reused_units == {} and CALLS["note_dict"] == 0  # unchanged
    assert graph({"a": [1, 3]}).run_sync(through_json(state)).status == "complete" and CALLS["note_dict"] == 3


def test_a_from_run_value_is_not_part_of_what_a_unit_received() -> None:
    """A new client or person must not bust every row, so a state goes back only to the runner it came from."""

    class Client:
        pass

    @node("via_client", category="test")
    def via_client(email: str, client: FromRun[Client]) -> Annotated[str, Result(title="sent")]:
        CALLS["via_client"] += 1
        return email

    reg = NodeRegistry([csv, via_client])

    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("v", "via_client", email=From("c.result.email"))]), reg)

    state = graph(BASE).run_sync(from_run={Client: Client()})
    CALLS.clear()

    ending = graph(BASE + "d|d@x\n").run_sync(through_json(state), from_run={Client: Client()})

    assert CALLS["via_client"] == 1 and ending.reused_units == {"v": 3}


def test_a_policy_says_a_result_depends_on_the_world() -> None:
    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("r", "rate", email=From("c.result.email"))]), registry)

    state = graph(BASE).run_sync()
    CALLS.clear()

    ending = graph(BASE + "d|d@x\n").run_sync(state)

    assert CALLS["rate"] == 4 and ending.reused_units == {}


# -- a result that moved -------------------------------------------------------------------------------------------------


def test_a_table_output_reused_at_another_row_says_where_it_now_is() -> None:
    """The ids a unit minted name its row, so moved with it; a downstream merge on them would match wrong rows otherwise."""

    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("d", "doc_table", text=From("c.result.name")), n("m", "mail", email=From("d.result.email"), body=Static("b"))]), registry)

    state = graph(BASE).run_sync()
    CALLS.clear()
    edited = "zz|z@x\n" + BASE

    compiled = graph(edited)
    ending = compiled.run_sync(through_json(state))

    assert CALLS["doc_table"] == 1 and ending.reused_units == {"d": 3, "m": 3}
    written = compiled.results(ending)["d"]["result"]["values"]
    assert [frame["__row"].to_list() for frame in written] == [["d.result:0.0"], ["d.result:1.0"], ["d.result:2.0"], ["d.result:3.0"]]
    assert [frame["name"].to_list() for frame in written] == [["zz"], ["a"], ["b"], ["c"]]
    # and the merge on those ids still matches
    merged = compiled.results(ending)["m"]["result"]
    assert merged["values"] == ["sent:zz@x:b", "sent:a@x:b", "sent:b@x:b", "sent:c@x:b"]
    # the state a later leg reads moved with it: nothing downstream sees an old id
    assert {frame["__row"].to_list()[0] for frame in written} == {"d.result:0.0", "d.result:1.0", "d.result:2.0", "d.result:3.0"}


def test_a_row_inside_a_list_moves_with_the_row_it_is_under() -> None:
    def graph(text: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(text)), n("sp", "split", text=From("c.result.name")), n("p", "per", line=From("sp.result")), n("cn", "count", lines=From("p.result"))]), registry)

    first = "x,y|e\nu,v,w|f\n"
    state = graph(first).run_sync()
    CALLS.clear()

    compiled = graph("new|e\n" + first)
    ending = compiled.run_sync(through_json(state))

    assert CALLS["split"] == 1 and CALLS["per"] == 1 and CALLS["count"] == 1
    assert ending.reused_units == {"sp": 2, "p": 5, "cn": 2}
    assert compiled.results(ending)["p"]["result"]["values"] == ["NEW", "X", "Y", "U", "V", "W"]
    assert compiled.results(ending)["cn"]["result"]["values"] == [1, 2, 3]


def test_a_changed_row_is_new_work_all_the_way_down() -> None:
    def graph(text: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(text)), n("sp", "split", text=From("c.result.name")), n("p", "per", line=From("sp.result"))]), registry)

    state = graph("x,y|e\n").run_sync()
    CALLS.clear()

    graph("x,z|e\n").run_sync(state)

    assert CALLS["split"] == 1 and CALLS["per"] == 1  # z is the only new line; x is the same work


# -- waiting units and answers -------------------------------------------------------------------------------------------


def ask_graph(rows: str) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("a", "ask", email=From("c.result.email")), n("x", "export", table=From("c.result", "a.result"))]), registry)


def test_a_waiting_unit_goes_on_waiting_at_its_new_row_with_no_new_question() -> None:
    compiled = ask_graph(BASE)
    state = compiled.run_sync()
    CALLS.clear()

    moved = ask_graph("z|z@x\n" + BASE)
    ending = moved.run_sync(through_json(state))

    assert CALLS["ask"] == 1  # only the new row is asked
    assert [unit.prompt for unit in ending.waiting] == ["z@x", "a@x", "b@x", "c@x"]
    assert [unit.row for unit in ending.waiting] == [(0,), (1,), (2,), (3,)]
    assert ending.reused_units == {"a": 3}


def test_a_waiting_unit_put_back_is_announced_on_the_stream() -> None:
    compiled = ask_graph(BASE)
    state = compiled.run_sync()

    async def waits() -> list[str]:
        async with ask_graph("z|z@x\n" + BASE).stream(state) as stream:
            return [event.unit.prompt async for event in stream if isinstance(event, UnitWaiting)]

    assert sorted(asyncio.run(waits())) == ["a@x", "b@x", "c@x", "z@x"]


def test_answers_stay_with_their_row_when_a_row_is_inserted_above() -> None:
    compiled = ask_graph(BASE)
    state = compiled.run_sync()
    state = compiled.run_sync(answer(compiled, state, {"a@0": {"result": "A-yes"}, "a@1": {"result": "B-no"}}))
    CALLS.clear()
    waiting_c_id = state.waiting[0].id

    moved = ask_graph("new|n@x\n" + BASE)
    ending = moved.run_sync(through_json(state))

    assert CALLS["ask"] == 1 and sorted(unit.prompt for unit in ending.waiting) == ["c@x", "n@x"]
    assert [unit.row for unit in ending.waiting] == [(0,), (3,)]
    ending = moved.run_sync(moved.answer(ending, {unit.id: {"result": "ok:" + unit.prompt} for unit in ending.waiting}))
    assert moved.results(ending)["a"]["result"]["values"] == ["ok:n@x", "A-yes", "B-no", "ok:c@x"]
    assert ending.answered_rows == {"a": [(0,), (1,), (2,), (3,)]} and ending.reused_units == {}

    with pytest.raises(InvalidCallError) as stale:
        moved.answer(ending, {waiting_c_id: {"result": "stale"}})
    assert stale.value.errors()[0].type == "not_waiting"


def test_an_edited_row_is_a_different_question_and_asks_again() -> None:
    compiled = ask_graph(BASE)
    state = compiled.run_sync()
    state = answer(compiled, state, {"a@0": {"result": "yes"}})
    state = compiled.run_sync(state)
    CALLS.clear()

    edited = ask_graph("a|a2@x\nb|b@x\nc|c@x\n")
    ending = edited.run_sync(state)

    assert CALLS["ask"] == 1 and sorted(unit.prompt for unit in ending.waiting) == ["a2@x", "b@x", "c@x"]


def test_an_answer_keyed_for_a_row_that_moved_is_refused() -> None:
    state = ask_graph(BASE).run_sync()
    old_id_of_row_one = state.waiting[1].id  # a@1 ... b@x

    moved = ask_graph("n|n@x\n" + BASE)
    again = moved.run_sync(through_json(state))

    with pytest.raises(InvalidCallError) as stale:
        moved.answer(again, {old_id_of_row_one: {"result": "meant for b, row 1 is now a"}})
    assert stale.value.errors()[0].type == "not_waiting"


def test_identical_answered_rows_are_asked_again_when_none_is_at_this_row() -> None:
    """Two rows with the same inputs, answered differently: moved rows can't say whose answer was whose."""

    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("a", "ask", email=From("c.result.email"))]), registry)

    compiled = graph("a|d@x\nb|d@x\n")
    state = compiled.run_sync()
    state = compiled.run_sync(answer(compiled, state, {"a@0": {"result": "yes"}, "a@1": {"result": "no"}}))
    assert state.status == "complete"
    CALLS.clear()

    ending = graph("z|z@x\ny|y@x\na|d@x\nb|d@x\n").run_sync(through_json(state))  # both moved down, neither at its own row

    assert CALLS["ask"] == 4 and len(ending.waiting) == 4
    # at its own row, an answer keeps its place
    CALLS.clear()
    kept = graph("a|d@x\nb|d@x\n").run_sync(through_json(state))
    assert kept.status == "complete" and CALLS["ask"] == 0
    assert graph("a|d@x\nb|d@x\n").results(kept)["a"]["result"]["values"] == ["yes", "no"]


def test_an_answered_unit_keeps_its_answer_through_a_row_edit_elsewhere() -> None:
    compiled = ask_graph(BASE)
    state = compiled.run_sync()
    state = compiled.run_sync(answer(compiled, state, {"a@0": {"result": "yes"}, "a@1": {"result": "no"}, "a@2": {"result": "?"}}))
    CALLS.clear()

    edited = ask_graph("a|a@x\nb|b@x\nc|c@x\nd|d@x\n")
    ending = edited.run_sync(through_json(state))

    assert CALLS["ask"] == 1 and ending.waiting[0].prompt == "d@x" and ending.reused_units["a"] == 3


# -- a leg that stops short keeps what it hasn't reached -------------------------------------------------------------------


def fail_graph(rows: str) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("f", "fail_on", email=From("c.result.email"))]), registry)


def test_a_failed_leg_does_not_lose_what_it_has_not_reached_yet() -> None:
    """Fix the mail server, run again from the failed state: every earlier mail stays sent."""
    state = fail_graph(BASE + "d|d@x\n").run_sync()
    CALLS.clear()

    FAILING["address"] = "n@x"
    edited = "n|n@x\n" + BASE + "d|d@x\n"
    failed = fail_graph(edited).run_sync(through_json(state))  # the new row fails first: its siblings aren't reached
    assert failed.status == "failed"
    CALLS.clear()

    FAILING["address"] = ""
    ending = fail_graph(edited).run_sync(through_json(failed))

    assert ending.status == "complete"
    assert CALLS["fail_on"] == 1 and CALLS["fail_on n@x"] == 1
    assert ending.reused_units["f"] == 4 - failed.reused_units.get("f", 0)


def test_a_state_a_leg_left_short_carries_only_what_is_unclaimed() -> None:
    state = fail_graph(BASE).run_sync()

    FAILING["address"] = "n@x"
    edited = "n|n@x\n" + BASE
    failed = fail_graph(edited).run_sync(through_json(state))
    FAILING["address"] = ""

    assert failed.status == "failed" and failed.earlier_work is not None
    assert sum(len(rows) for rows in failed.earlier_work.finished_rows.values()) == 3 - sum(failed.reused_units.values())

    # the leg that finishes the node has none to carry
    ending = fail_graph(edited).run_sync(through_json(failed))
    assert ending.status == "complete" and ending.earlier_work is None


def test_a_second_edit_after_a_failed_leg_still_reuses_what_the_failed_leg_had_not_reached() -> None:
    state = fail_graph(BASE + "d|d@x\n").run_sync()
    FAILING["address"] = "n@x"
    failed = fail_graph("n|n@x\n" + BASE + "d|d@x\n").run_sync(through_json(state))
    FAILING["address"] = ""
    CALLS.clear()

    ending = fail_graph("n|n@x\n" + BASE + "d|d@x\ne|e@x\n").run_sync(through_json(failed))

    assert ending.status == "complete" and CALLS["fail_on"] == 2 and CALLS["fail_on n@x"] == 1 and CALLS["fail_on e@x"] == 1


def test_a_cancelled_leg_keeps_its_earlier_work() -> None:
    state = mail_graph(BASE + "d|d@x\n").run_sync()
    edited = mail_graph("n|n@x\n" + BASE + "d|d@x\n")
    CALLS.clear()

    async def go() -> RunState:
        cancel = asyncio.Event()
        async with edited.stream(state, cancel=cancel) as stream:
            async for event in stream:
                if isinstance(event, NodeStart) and event.node_id == "c":
                    cancel.set()
                if event.type == "run_end":
                    return event.state
        raise AssertionError

    stopped = asyncio.run(go())
    assert stopped.status == "cancelled" and stopped.earlier_work is not None
    CALLS.clear()

    ending = mail_graph("n|n@x\n" + BASE + "d|d@x\n").run_sync(through_json(stopped))
    assert ending.status == "complete" and CALLS["mail"] == 1


def test_an_answer_on_a_changed_graph_keeps_the_changed_nodes_earlier_work() -> None:
    """Answering runs no node: the mails the edited graph hasn't reached stay reusable."""

    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("q", "ask", email=Static("?")), n("m", "mail", email=From("c.result.email"), body=From("q.result"))]), registry)

    compiled = graph(BASE)
    state = compiled.run_sync(answer(compiled, compiled.run_sync(), {"q": {"result": "Hej"}}))
    CALLS.clear()

    edited = graph(BASE + "d|d@x\n")
    # the graph changed, and an answer changes nothing the graph runs; a leg later still sends only the new mail
    again = edited.run_sync(through_json(state))
    assert again.status == "complete" and CALLS["mail"] == 1 and again.reused_units == {"m": 3}


def test_a_later_leg_keeps_reused_work_as_its_own() -> None:
    state = first_leg()
    second = mail_graph(BASE + "d|d@x\n").run_sync(through_json(state))
    CALLS.clear()

    third = mail_graph("z|z@x\n" + BASE + "d|d@x\n").run_sync(through_json(second))

    assert CALLS["mail"] == 1 and third.reused_units == {"m": 4}


def test_without_nodes_forgets_earlier_work_too() -> None:
    FAILING["address"] = "n@x"
    bad = fail_graph("n|n@x\n" + BASE).run_sync(fail_graph(BASE).run_sync())
    FAILING["address"] = ""
    assert bad.earlier_work is not None
    forgotten = bad.without_nodes("f")
    assert forgotten.earlier_work.finished_rows == {} and "f" not in forgotten.input_hashes
    CALLS.clear()
    assert fail_graph("n|n@x\n" + BASE).run_sync(forgotten).status == "complete" and CALLS["fail_on"] == 4


# -- what is reported -------------------------------------------------------------------------------------------------------


def test_a_node_entirely_reused_starts_and_completes_and_says_how_much() -> None:
    state = first_leg()

    async def events() -> list[Any]:
        async with mail_graph("c|c@x\nb|b@x\na|a@x\n").stream(through_json(state)) as stream:
            return [event async for event in stream]

    seen = asyncio.run(events())
    assert any(isinstance(event, NodeStart) and event.node_id == "m" for event in seen)
    done = next(event for event in seen if isinstance(event, NodeComplete) and event.node_id == "m")
    assert done.reused_units == 3
    progress = [event for event in seen if isinstance(event, NodeProgress) and event.node_id == "m"]
    assert (progress[0].done, progress[-1].done) == (1, 3)


def test_a_nothing_reused_leg_reports_no_reuse() -> None:
    async def events() -> list[Any]:
        async with mail_graph(BASE).stream() as stream:
            return [event async for event in stream]

    done = next(event for event in asyncio.run(events()) if isinstance(event, NodeComplete) and event.node_id == "m")
    assert done.reused_units == 0


# -- embedded, restored state, the state's own bookkeeping -----------------------------------------------------------


def test_inner_units_of_an_embedded_graph_are_reused_like_any_others() -> None:
    inner = (GraphNode(id="t", type="mail", version=1, bindings={"body": Static("inner")}),)
    sub = embedded_graph_node("sub", inner, NodeRegistry([csv, mail]))
    reg = NodeRegistry([csv, mail, sub])

    def graph(rows: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("s", "sub", email=From("c.result.email"))]), reg)

    state = graph(BASE).run_sync()
    CALLS.clear()

    edited = graph("z|z@x\n" + BASE)
    ending = edited.run_sync(through_json(state))

    assert CALLS["mail"] == 1 and ending.reused_units == {"s/t": 3}
    assert edited.results(ending)["s/t"]["result"]["values"] == ["sent:z@x:inner", "sent:a@x:inner", "sent:b@x:inner", "sent:c@x:inner"]


def test_results_leave_out_a_changed_node_until_its_leg_has_run() -> None:
    state = first_leg()
    edited = mail_graph(BASE + "d|d@x\n")

    assert "m" not in edited.results(state)  # stale until its leg has run; reuse never makes it look current
    ending = edited.run_sync(state)
    assert edited.results(ending)["m"]["result"]["values"][-1] == "sent:d@x:Hej"


def test_the_state_keeps_each_units_hash_through_a_restored_leg() -> None:
    state = first_leg()
    assert len(state.input_hashes["m"]) == 3 and all(state.input_hashes["m"])

    restored = mail_graph(BASE).run_sync(through_json(state))

    assert restored.input_hashes == state.input_hashes and restored.finished_rows == state.finished_rows


def test_an_input_hash_list_that_doesnt_line_up_is_unreadable() -> None:
    state = first_leg()
    broken = state.model_copy(update={"input_hashes": {**state.input_hashes, "m": ["only one"]}})

    with pytest.raises(InvalidCallError) as refused:
        mail_graph(BASE).run_sync(broken)
    assert refused.value.errors()[0].type == "state_unreadable"


def test_a_column_of_a_changed_node_that_no_longer_reads_means_no_reuse() -> None:
    state = first_leg()
    tampered = state.model_copy(update={"output_columns": {**state.output_columns, Ref("m", "result"): OutputColumn(rows=[(0,), (1,), (2,)], values=[{"not": "a string"}] * 3)}})

    ending = mail_graph(BASE + "d|d@x\n").run_sync(tampered)

    assert ending.status == "complete" and CALLS["mail"] == 4 and "m" not in ending.reused_units


def test_a_later_run_of_the_same_graph_takes_no_earlier_hashes_when_there_are_none() -> None:
    state = first_leg()
    bare = state.model_copy(update={"input_hashes": {}})
    ending = mail_graph(BASE + "d|d@x\n").run_sync(bare)

    assert ending.status == "complete" and CALLS["mail"] == 4 and "m" not in ending.reused_units


def test_the_ids_of_waiting_units_change_with_their_inputs_and_row() -> None:
    state = ask_graph(BASE).run_sync()
    ids = [unit.id for unit in state.waiting]
    assert all(id_.startswith("a@") and "#" in id_ for id_ in ids) and len(set(ids)) == 3
    again = ask_graph(BASE).run_sync(through_json(state))  # unchanged: the same units, not re-asked
    assert [unit.id for unit in again.waiting] == ids
