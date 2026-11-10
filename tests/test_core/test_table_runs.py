"""The acceptance scenarios of "every level is a table" that read a table's columns, built with small nodes and run end to end.

A node returns a table; each of its columns is a field of its own
(``csv.result.email``). Wiring a column into a single-value input runs the
node once per row of the table; wiring it into a ``Series[X]`` input gives
every row under the node's row. Results land on the table's rows.
"""

from dataclasses import dataclass
from typing import Annotated, Any

import polars as pl
from conductor import (
    ROW_ID_COLUMN,
    SKIPPED,
    AskPerson,
    CompiledGraph,
    NodeRegistry,
    Param,
    Result,
    Series,
    node,
    table_with_columns,
)
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Textarea
from helpers import Placed, events_of, placed, record, registry_of, with_added

from test_core.embedded import embedded_graph_node
from test_core.test_table import Num, Txt

Contacts = table_with_columns({"name": Txt, "email": Txt})
Invoices = table_with_columns({"vendor": Txt, "amount": Num})
Text = Annotated[Txt, Param(title="Text", widget=Textarea())]


@node("csv", title="CSV", description="d", category="test")
def read_csv(text: Text = Txt("")) -> Annotated[Contacts, Result(title="Rows")]:
    """``name,email`` per line, lines split by ``;``; an empty email is a blank cell."""
    rows = [line.split(",") for line in text.split(";")]
    return pl.DataFrame({"name": [name for name, _ in rows], "email": [email or None for _, email in rows]})


@node("send", title="Send mail", description="d", category="test")
def send_mail(email: Annotated[Txt | None, Param(title="Email", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Sent")]:
    return Txt("no address" if email is None else f"sent to {email}")


@node("docs", title="Docs", description="d", category="test")
def docs(text: Text = Txt("")) -> Annotated[Series[Txt], Result(title="Documents")]:
    return [Txt(part) for part in text.split("|")]


@node("structured", title="Structured", description="d", category="test")
def structured(text: Text = Txt("")) -> Annotated[Invoices, Result(title="Invoices")]:
    """``vendor:amount`` pairs, comma-separated: the invoices in one document."""
    pairs = [part.split(":") for part in text.split(",") if part]
    return pl.DataFrame({"vendor": [v for v, _ in pairs], "amount": [float(a) for _, a in pairs]}, schema={"vendor": pl.String, "amount": pl.Float64})


@node("check", title="Check vendor", description="d", category="test")
def check(vendor: Annotated[Txt, Param(title="Vendor", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Approved")]:
    """Approves a vendor, or skips one it has no word on (``unknown``)."""
    return SKIPPED if vendor == "unknown" else Txt(f"{vendor} ok")


@node("total", title="Total", description="d", category="test")
def total_node(amounts: Annotated[Series[Num], Param(title="Amounts")] = ()) -> Annotated[Num, Result(title="Total")]:
    return Num(sum(amounts))


@node("join", title="Join", description="d", category="test")
def join_texts(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Annotated[Txt, Result(title="Joined")]:
    return Txt("+".join(texts))


@dataclass(frozen=True)
class Known:
    known: Annotated[Txt, Result(title="Known")]
    unknown: Annotated[Txt, Result(title="Unknown")]


@node("route", title="Route", description="d", category="test")
def route(name: Annotated[Txt, Param(title="Name", widget=Textarea())] = Txt(""), email: Annotated[Txt | None, Param(title="Email", widget=Textarea())] = Txt("")) -> Known:
    """A branch per row: a contact with an email goes one way, one without the other."""
    return Known(known=Txt(f"Dear {name}"), unknown=SKIPPED) if email is not None else Known(known=SKIPPED, unknown=Txt(f"Call {name}"))


@node("approve", title="Approve", description="d", category="test")
def approve(vendor: Annotated[Txt, Param(title="Vendor", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Approved by")] | AskPerson:
    """A person approves each invoice."""
    return AskPerson()


@node("gate", title="Gate", description="d", category="test")
def gate(text: Text = Txt("")) -> Annotated[Txt, Result(title="Passed")]:
    """Passes a document on, or skips it when it says ``skip``."""
    return SKIPPED if text == "skip" else text


def _registry() -> NodeRegistry:
    return registry_of(read_csv, send_mail, docs, structured, check, total_node, join_texts, route, approve, gate)


def _compiled(*nodes: Placed, registry: NodeRegistry | None = None) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(record(*list(nodes)), registry or _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _results(compiled: CompiledGraph) -> dict[str, dict[str, Any]]:
    return compiled.results(compiled.run_sync())


CSV = placed("csv", "csv", text=Static("Ida,ida@x;Bo,;Nina,nina@x"))


def test_send_mail_runs_once_per_contact_and_bos_blank_email_runs_blank():
    compiled = _compiled(CSV, placed("send", "send", email=From("csv.result.email")))
    results = _results(compiled)

    sent = results["send"]["result"]
    assert sent == {"rows": [(0,), (1,), (2,)], "values": ["sent to ida@x", "no address", "sent to nina@x"]}


def test_the_stored_table_carries_its_row_ids():
    compiled = _compiled(CSV, placed("send", "send", email=From("csv.result.email")))
    contacts = _results(compiled)["csv"]["result"]

    assert isinstance(contacts, pl.DataFrame)
    assert contacts[ROW_ID_COLUMN].to_list() == ["csv.result:0", "csv.result:1", "csv.result:2"]


def test_a_table_nobody_reads_a_column_of_makes_no_rows_and_is_stored_once():
    compiled = _compiled(CSV)
    ending = compiled.run_sync()

    assert [(node_id, list(work.outputs)) for node_id, work in ending.nodes.items()] == [("csv", ["result"])]


def test_invoices_per_document_get_their_check_and_a_skipped_check_is_a_gap_on_its_own_row():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10,unknown:5|beta:7")),
        placed("read", "structured", text=From("docs.result")),
        placed("check", "check", vendor=From("read.result.vendor")),
        placed("total", "total", amounts=From("read.result.amount")),
    )
    results = _results(compiled)

    approved = results["check"]["result"]
    assert approved == {"rows": [(0, 0), (1, 0)], "values": ["acme ok", "beta ok"]}
    assert results["total"]["result"]["values"] == [15.0, 7.0]


def test_a_document_with_no_invoices_totals_nothing():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10|")),
        placed("read", "structured", text=From("docs.result")),
        placed("total", "total", amounts=From("read.result.amount")),
    )

    assert _results(compiled)["total"]["result"]["values"] == [10.0, 0.0]


def test_a_branch_per_row_fills_only_its_rows_and_the_next_node_reads_whichever_is_filled():
    compiled = _compiled(
        CSV,
        placed("route", "route", name=From("csv.result.name"), email=From("csv.result.email")),
        placed("send", "send", email=From("route.known", "route.unknown")),
    )
    results = _results(compiled)

    assert results["route"]["known"]["rows"] == [(0,), (2,)]
    assert results["route"]["unknown"]["rows"] == [(1,)]
    assert results["send"]["result"]["values"] == ["sent to Dear Ida", "sent to Call Bo", "sent to Dear Nina"]


def test_a_column_into_a_series_input_on_a_node_that_runs_once_is_the_whole_column():
    compiled = _compiled(CSV, placed("all", "join", texts=From("csv.result.name")))

    assert _results(compiled)["all"]["result"] == "Ida+Bo+Nina"


def test_a_person_approves_each_invoice_and_the_document_totals_once_its_invoices_are_done():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10,beta:5|gamma:7")),
        placed("read", "structured", text=From("docs.result")),
        placed("approve", "approve", vendor=From("read.result.vendor")),
        placed("all", "join", texts=From("approve.result")),
    )
    first = events_of(compiled)[-1].state
    assert first.status == "waiting"
    assert [unit.row for unit in first.waiting] == [(0, 0), (0, 1), (1, 0)]

    # A paused run resumes on the same rows: the stored tables make them again.
    second = events_of(compiled, state=first, answers={"approve@0,0": {"result": "Ida"}, "approve@0,1": {"result": "Bo"}, "approve@1,0": {"result": "Nina"}})[-1].state

    assert second.status == "complete"
    results = compiled.results(second)
    assert results["approve"]["result"]["rows"] == [(0, 0), (0, 1), (1, 0)]
    assert results["all"]["result"]["values"] == ["Ida+Bo", "Nina"]


def test_an_embedded_graph_run_per_row_lands_its_result_on_the_contacts():
    registry = _registry()
    sub = embedded_graph_node("mailer", (placed("send", "send"),), registry, "Mailer")
    compiled = _compiled(
        CSV,
        placed("emb", "mailer", email=From("csv.result.email")),
        registry=with_added(registry, sub),
    )
    results = _results(compiled)

    sent = results["emb/send"]["result"]
    assert sent["values"] == ["sent to ida@x", "no address", "sent to nina@x"]


def test_a_table_made_inside_an_embedded_graph_offers_its_columns_at_the_graph_node():
    registry = _registry()
    sub = embedded_graph_node("reader", (placed("csv", "csv"),), registry, "Reader")
    compiled = _compiled(
        placed("emb", "reader", text=Static("Ida,ida@x;Bo,")),
        placed("send", "send", email=From("emb.result.email")),
        registry=with_added(registry, sub),
    )

    assert _results(compiled)["send"]["result"]["values"] == ["sent to ida@x", "no address"]


def test_a_document_skipped_above_its_table_totals_nothing_and_the_others_still_total():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10|skip|beta:7")),
        placed("gate", "gate", text=From("docs.result")),
        placed("read", "structured", text=From("gate.result")),
        placed("total", "total", amounts=From("read.result.amount")),
    )
    total = _results(compiled)["total"]["result"]

    assert total == {"rows": [(0,), (2,)], "values": [10.0, 7.0]}
