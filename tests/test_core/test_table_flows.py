"""The acceptance flows of "every level is a table" that read a table's columns, built with small nodes and run end to end.

A node returns a table; each of its columns is a field of its own
(``csv.result.email``). Wiring a column into a single-value input runs the
node once per row of the table; wiring it into a ``Series[X]`` input gives
every row under the node's row. Results land on the table's rows.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

import polars as pl
from conductor import (
    SKIPPED,
    Asks,
    CompiledGraph,
    GraphNode,
    NodeRegistry,
    Param,
    Static,
    Table,
    run_sync,
)
from conductor.execution.engine import execute
from conductor.graph.binding import From
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.table import ROW_ID
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node
from test_core.test_table import Num, Txt

Contacts = Table[("name", Txt), ("email", Txt)]
Invoices = Table[("vendor", Txt), ("amount", Num)]
Text = Annotated[Txt, Param(title="Text", widget=Textarea())]


class ReadCsv(NodeDefinition):
    """``name,email`` per line, lines split by ``;``; an empty email is a blank cell."""

    id = "csv"
    title = "CSV"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Contacts, Result(title="Rows")]:
        rows = [line.split(",") for line in text.split(";")]
        return pl.DataFrame({"name": [name for name, _ in rows], "email": [email or None for _, email in rows]})


class SendMail(NodeDefinition):
    id = "send"
    title = "Send mail"
    description = "d"
    category = "test"

    def run(self, email: Annotated[Txt, Param(title="Email", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Sent")]:
        return Txt("no address" if email is None else f"sent to {email}")


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Series[Txt], Result(title="Documents")]:
        return [Txt(part) for part in text.split("|")]


class Structured(NodeDefinition):
    """``vendor:amount`` pairs, comma-separated: the invoices in one document."""

    id = "structured"
    title = "Structured"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Invoices, Result(title="Invoices")]:
        pairs = [part.split(":") for part in text.split(",") if part]
        return pl.DataFrame({"vendor": [v for v, _ in pairs], "amount": [float(a) for _, a in pairs]}, schema={"vendor": pl.String, "amount": pl.Float64})


class Check(NodeDefinition):
    """Approves a vendor, or skips one it has no word on (``unknown``)."""

    id = "check"
    title = "Check vendor"
    description = "d"
    category = "test"

    def run(self, vendor: Annotated[Txt, Param(title="Vendor", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Approved")]:
        return SKIPPED if vendor == "unknown" else Txt(f"{vendor} ok")


class Total(NodeDefinition):
    id = "total"
    title = "Total"
    description = "d"
    category = "test"

    def run(self, amounts: Annotated[Series[Num], Param(title="Amounts")] = ()) -> Annotated[Num, Result(title="Total")]:
        return Num(sum(amounts))


class JoinTexts(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Annotated[Txt, Result(title="Joined")]:
        return Txt("+".join(texts))


@dataclass(frozen=True)
class Known:
    known: Annotated[Txt, Result(title="Known")]
    unknown: Annotated[Txt, Result(title="Unknown")]


class Route(NodeDefinition):
    """A branch per row: a contact with an email goes one way, one without the other."""

    id = "route"
    title = "Route"
    description = "d"
    category = "test"

    def run(self, name: Annotated[Txt, Param(title="Name", widget=Textarea())] = Txt(""), email: Annotated[Txt, Param(title="Email", widget=Textarea())] = Txt("")) -> Known:
        return Known(known=Txt(f"Dear {name}"), unknown=SKIPPED) if email is not None else Known(known=SKIPPED, unknown=Txt(f"Call {name}"))


class Approve(NodeDefinition):
    """A person approves each invoice."""

    id = "approve"
    title = "Approve"
    description = "d"
    category = "test"

    def run(self, vendor: Annotated[Txt, Param(title="Vendor", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Approved by")] | Asks:
        return Asks()


class Gate(NodeDefinition):
    """Passes a document on, or skips it when it says ``skip``."""

    id = "gate"
    title = "Gate"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Txt, Result(title="Passed")]:
        return SKIPPED if text == "skip" else text


def _registry() -> NodeRegistry:
    return NodeRegistry(nodes=(ReadCsv, SendMail, Docs, Structured, Check, Total, JoinTexts, Route, Approve, Gate))


def _compiled(*nodes: GraphNode, registry: NodeRegistry | None = None) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _results(compiled: CompiledGraph) -> dict[str, dict[str, Any]]:
    return run_sync(compiled).state.results(compiled)


def _leg(compiled: CompiledGraph, **kw: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [event async for event in execute(compiled, **kw)]

    return asyncio.run(run())


CSV = GraphNode(id="csv", type="csv", version=1, bindings={"text": Static("Ida,ida@x;Bo,;Nina,nina@x")})


def test_send_mail_runs_once_per_contact_and_bos_blank_email_runs_blank():
    compiled = _compiled(CSV, GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}))
    results = _results(compiled)

    sent = results["send"]["result"]
    assert sent.index == Index("csv.result") and sent.rows == ((0,), (1,), (2,))
    assert list(sent) == ["sent to ida@x", "no address", "sent to nina@x"]


def test_the_stored_table_carries_its_row_ids():
    compiled = _compiled(CSV, GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}))
    contacts = _results(compiled)["csv"]["result"]

    assert type(contacts) is Contacts
    assert contacts.frame[ROW_ID].to_list() == ["csv.result:0", "csv.result:1", "csv.result:2"]


def test_a_table_nobody_reads_a_column_of_makes_no_rows_and_is_stored_once():
    compiled = _compiled(CSV)
    ending = run_sync(compiled)

    assert [entry.ref for entry in ending.state.values] == [Ref("csv", "result")]


def test_invoices_per_document_get_their_check_and_a_skipped_check_is_a_gap_on_its_own_row():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,unknown:5|beta:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="check", type="check", version=1, bindings={"vendor": From("read.result.vendor")}),
        GraphNode(id="total", type="total", version=1, bindings={"amounts": From("read.result.amount")}),
    )
    results = _results(compiled)

    approved = results["check"]["result"]
    assert approved.index == Index("read.result", parent=Index("docs.result"))
    assert approved.rows == ((0, 0), (1, 0))
    assert list(approved) == ["acme ok", "beta ok"]
    assert list(results["total"]["result"]) == [15.0, 7.0]


def test_a_document_with_no_invoices_totals_nothing():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10|")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="total", type="total", version=1, bindings={"amounts": From("read.result.amount")}),
    )

    assert list(_results(compiled)["total"]["result"]) == [10.0, 0.0]


def test_a_branch_per_row_fills_only_its_rows_and_the_next_node_reads_whichever_is_filled():
    compiled = _compiled(
        CSV,
        GraphNode(id="route", type="route", version=1, bindings={"name": From("csv.result.name"), "email": From("csv.result.email")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("route.known", "route.unknown")}),
    )
    results = _results(compiled)

    assert results["route"]["known"].rows == ((0,), (2,))
    assert results["route"]["unknown"].rows == ((1,),)
    assert list(results["send"]["result"]) == ["sent to Dear Ida", "sent to Call Bo", "sent to Dear Nina"]


def test_a_column_into_a_series_input_on_a_node_that_runs_once_is_the_whole_column():
    compiled = _compiled(CSV, GraphNode(id="all", type="join", version=1, bindings={"texts": From("csv.result.name")}))

    assert _results(compiled)["all"]["result"] == "Ida+Bo+Nina"


def test_a_person_approves_each_invoice_and_the_document_totals_once_its_invoices_are_done():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="approve", type="approve", version=1, bindings={"vendor": From("read.result.vendor")}),
        GraphNode(id="all", type="join", version=1, bindings={"texts": From("approve.result")}),
    )
    first = _leg(compiled)[-1]
    assert first.type == "graph_pending"
    assert [unit.row for unit in first.pending] == [(0, 0), (0, 1), (1, 0)]

    # A paused run resumes on the same rows: the stored tables make them again.
    second = _leg(compiled, state=first.state, cache={"approve": {"result": ["Ida", "Bo", "Nina"]}})[-1]

    assert second.type == "graph_complete"
    results = second.state.results(compiled)
    assert results["approve"]["result"].rows == ((0, 0), (0, 1), (1, 0))
    assert list(results["all"]["result"]) == ["Ida+Bo", "Nina"]


def test_a_sub_flow_run_per_row_lands_its_result_on_the_contacts():
    registry = _registry()
    sub = embedded_graph_node("mailer", (GraphNode(id="send", type="send", version=1),), registry, "Mailer")
    compiled = _compiled(
        CSV,
        GraphNode(id="emb", type="mailer", version=1, bindings={"send.email": From("csv.result.email")}),
        registry=registry.extended_with({"mailer": sub}),
    )
    results = _results(compiled)

    sent = results["emb/send"]["result"]
    assert sent.index == Index("csv.result")
    assert list(sent) == ["sent to ida@x", "no address", "sent to nina@x"]


def test_a_table_made_inside_a_sub_flow_offers_its_columns_at_the_graph_node():
    registry = _registry()
    sub = embedded_graph_node("reader", (GraphNode(id="csv", type="csv", version=1),), registry, "Reader")
    compiled = _compiled(
        GraphNode(id="emb", type="reader", version=1, bindings={"csv.text": Static("Ida,ida@x;Bo,")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("emb.csv.result.email")}),
        registry=registry.extended_with({"reader": sub}),
    )

    assert list(_results(compiled)["send"]["result"]) == ["sent to ida@x", "no address"]


def test_a_document_skipped_above_its_table_totals_nothing_and_the_others_still_total():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10|skip|beta:7")}),
        GraphNode(id="gate", type="gate", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("gate.result")}),
        GraphNode(id="total", type="total", version=1, bindings={"amounts": From("read.result.amount")}),
    )
    total = _results(compiled)["total"]["result"]

    assert total.rows == ((0,), (2,))
    assert list(total) == [10.0, 7.0]
