"""A leg going on from a state, over a changed graph or table, reuses what it can.

A node nothing above has changed is restored whole. A changed node's units run
again, except a unit whose inputs match one an earlier unit of the node
received: its result is written at its new row, and the node isn't called. So
one row added to a CSV of 5,000 sends one mail.
"""

from collections import Counter
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Param,
    Policy,
    Result,
    RunState,
    Series,
    Static,
    Table,
    node,
)


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


@node("doc_table_or_fail", category="test", policy=Policy(concurrency=1))
def doc_table_or_fail(text: str) -> Annotated[Table[Contact], Result(title="Rows")]:
    CALLS["doc_table_or_fail"] += 1
    if text == FAILING["address"]:
        raise RuntimeError("the mail server is down")
    return pl.DataFrame({"name": [text], "email": [text + "@x"]}, schema={"name": pl.String, "email": pl.String})


@node("ask_or_fail", category="test", policy=Policy(concurrency=1))
def ask_or_fail(email: str) -> Annotated[str, Result(title="ok")]:
    CALLS["ask_or_fail"] += 1
    if email == FAILING["address"]:
        raise RuntimeError("the mail server is down")
    return AskPerson(prompt=email)


@node("note_dict", category="test")
def note_dict(email: str, stuff: dict[str, Any]) -> Annotated[str, Result(title="n")]:
    CALLS["note_dict"] += 1
    return email


@node("parse", category="test")
def parse(contacts: str = "") -> Annotated[Table[Contact], Result(title="Contacts")]:
    return csv(rows=contacts)


registry = NodeRegistry([csv, parse, mail, ask, export, split, per, count, doc_table, doc_table_or_fail, ask_or_fail, rate, fail_on, note_dict])


def n(node_id: str, node_type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def mail_graph(rows: str, body: str = "Hej") -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("m", "mail", email=From("c.result.email"), body=Static(body)), n("x", "export", table=From("c.result", "m.result"))]), registry)


def through_json(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


def reused(state: RunState) -> dict[str, int]:
    """Per node, how many units this leg took from earlier work, zeros left out."""
    return {node_id: work.reused_units for node_id, work in state.nodes.items() if work.reused_units}


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


# -- what the key sees -------------------------------------------------------------------------------------------------


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


# -- a result that moved -------------------------------------------------------------------------------------------------


# -- waiting units and answers -------------------------------------------------------------------------------------------


def ask_graph(rows: str) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("a", "ask", email=From("c.result.email")), n("x", "export", table=From("c.result", "a.result"))]), registry)


# -- a leg that stops short keeps what it hasn't reached -------------------------------------------------------------------


def fail_graph(rows: str) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(rows)), n("f", "fail_on", email=From("c.result.email"))]), registry)


# -- a chain of stopped legs is one leg --------------------------------------------------------------------------------------


def rows_ids_of_tables(compiled: CompiledGraph, state: RunState, node_id: str) -> list[list[str]]:
    return [frame["__row"].to_list() for frame in compiled.results(state)[node_id]["result"]["values"]]


def contacts_for_rows(addresses: list[str]) -> str:
    return "".join(f"{address}|{address}@x\n" for address in addresses)


def ask_or_fail_graph(addresses: list[str]) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(contacts_for_rows(addresses))), n("a", "ask_or_fail", email=From("c.result.email"))]), registry)


def answered_state_of(addresses: list[str]) -> RunState:
    compiled = ask_or_fail_graph(addresses)
    state = compiled.run_sync()
    state = compiled.answer(state, {unit.id: {"result": f"answered {unit.row[0]}"} for unit in state.waiting})
    assert not state.waiting
    CALLS.clear()
    return state


def doc_table_or_fail_graph(addresses: list[str]) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=[n("c", "csv", rows=Static(contacts_for_rows(addresses))), n("t", "doc_table_or_fail", text=From("c.result.email"))]), registry)


# -- what is reported -------------------------------------------------------------------------------------------------------


# -- embedded, restored state, the state's own bookkeeping -----------------------------------------------------------


