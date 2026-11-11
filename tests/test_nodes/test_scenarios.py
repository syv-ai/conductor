"""Fourteen things an author builds with tables, lists and documents, each a graph of this package's nodes and a
host's own: how many boxes it takes, what runs how often, and what comes out."""

import io
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, TypedDict

import conductor_nodes
import polars as pl
import pytest
from conductor import SKIPPED, Ask, NodeRegistry, Param, Series, Table, make_graph, node
from conductor_nodes.control import decision
from conductor_nodes.tables import collect, split
from conductor_nodes.text import split as split_text
from conductor_nodes.text import text_list
from pydantic import BaseModel


class File(BaseModel):
    type_id: ClassVar[str] = "file"
    type_title: ClassVar[str] = "File"
    name: str
    text: str


class Invoice(TypedDict):
    number: str
    vendor: str
    amount: float


class Contact(TypedDict):
    name: str
    email: str | None


#: Every call of a model stand-in, as the text it was asked about.
ASKED: list[str] = []


# -- the host's palette ---------------------------------------------------------------------------------


@node("upload", title="Upload", description="The files a person uploads.")
def upload(files: list[File]) -> Series[File]:
    return files


@node("read-csv", title="Read CSV", description="Reads a CSV file into a table.")
def read_csv(file: File) -> Table:
    return pl.read_csv(io.StringIO(file.text))


@node("read-invoices", title="Read invoices", description="Reads a CSV of invoices.")
def read_invoices(file: File) -> Table[Invoice]:
    return pl.read_csv(io.StringIO(file.text), schema={"number": pl.String, "vendor": pl.String, "amount": pl.Float64})


@node("read-contacts", title="Read contacts", description="Reads a CSV of contacts.")
def read_contacts(file: File) -> Table[Contact]:
    return pl.read_csv(io.StringIO(file.text), schema={"name": pl.String, "email": pl.String})


@node("read-pdf", title="Read PDF", description="Reads one PDF as text.")
def read_pdf(file: File) -> str:
    return file.text


@node("llm", title="Ask the model", description="Asks a language model about a text.")
def llm(prompt: str, text: str) -> str:
    ASKED.append(text)
    return f"{prompt} {text}"


@node("llm-all", title="Ask the model about several", description="Asks a language model about several texts at once.")
def llm_all(prompt: str, texts: Series[str]) -> str:
    ASKED.append(" + ".join(texts))
    return f"{prompt} {len(texts)} texts"


@node("find-invoices", title="Find invoices", description="Finds the invoices in a document.")
def find_invoices(text: str) -> Series[Invoice]:
    found = [line.split(",") for line in text.splitlines()]
    return [Invoice(number=number, vendor=vendor, amount=float(amount)) for number, vendor, amount in found]


@node("send-mail", title="Send mail", description="Sends a mail and says how it went.")
def send_mail(to: str, body: str) -> str:
    return f"sent to {to}"


@node("excel", title="Write Excel", description="Writes a table to an Excel file.")
def excel(table: Table, name: str = "result") -> File:
    return File(name=f"{name}.xlsx", text=table.write_csv())


@node("approve", title="Approve", description="Asks a person to approve an invoice over 500.")
def approve(number: str, amount: float) -> bool:
    if amount <= 500:
        return True
    raise Ask(f"Approve {number} for {amount}?")


@node("count", title="Count", description="Counts what it is given.")
def count(values: Series[Any]) -> int:
    return len(values)


@node("approved-total", title="Approved total", description="Adds up the approved invoices.")
def approved_total(amounts: Series[float], approved: Series[bool]) -> float:
    return sum(amount for amount, yes in zip(amounts, approved, strict=True) if yes)


@dataclass(frozen=True)
class BySize:
    large: Annotated[Invoice, Param(exclusive_group="size")] = SKIPPED
    small: Annotated[Invoice, Param(exclusive_group="size")] = SKIPPED


@node("by-size", title="Large or small", description="Sends an invoice one way when it is over the limit, the other when not.")
def by_size(invoice: Invoice, limit: float = 500) -> BySize:
    return BySize(large=invoice) if invoice["amount"] > limit else BySize(small=invoice)


@node("review", title="Review", description="Writes a review note for a large invoice.")
def review(invoice: Invoice) -> str:
    return f"review {invoice['number']}"


@node("book", title="Book", description="Books a small invoice.")
def book(invoice: Invoice) -> str:
    return f"booked {invoice['number']}"


@node("report", title="Report", description="Lists what happened to every invoice.")
def report(notes: Series[str]) -> str:
    return "; ".join(notes)


@node("describe-table", title="Describe a table", description="Writes what a table holds as text.")
def describe_table(table: Table) -> str:
    return f"{table.height} lines of {', '.join(table.columns)}"


@node("join-tables", title="Join tables", description="Joins two tables where a column of each holds the same value.")
def join_tables(left: Table, right: Table, on: str) -> Table:
    return left.join(right, on=on)


PALETTE = (
    upload, read_csv, read_invoices, read_contacts, read_pdf, llm, llm_all, find_invoices, send_mail, excel, approve, count,
    approved_total, by_size, review, book, report, describe_table, join_tables,
)


def registry_with(*more: Any) -> NodeRegistry:
    registry = NodeRegistry()
    registry.add_subregistry(conductor_nodes.registry)
    for definition in (*PALETTE, *more):
        registry.add(definition)
    return registry


REGISTRY = registry_with()

INVOICES = File(name="invoices.csv", text="number,vendor,amount\nA1,Acme,100\nA2,Bolt,900\nA3,Core,300\n")
CONTACTS = File(name="contacts.csv", text="name,email\nAnn,ann@x.dk\nBo,\nCy,cy@x.dk\n")
A_PDF = File(name="a.pdf", text="A1,Acme,100\nA2,Bolt,900")
B_PDF = File(name="b.pdf", text="")
C_PDF = File(name="c.pdf", text="C1,Core,300\nC2,Dune,700\nC3,Echo,50")


def answered(run: Any, yes: set[str]) -> Any:
    """``run`` resumed with every waiting approval answered: yes for the invoice numbers in ``yes``."""
    return run.answer({waiting.id: waiting.inputs["number"] in yes for waiting in run.waiting}).resume()


# -- 1. per-invoice approval: 4 boxes -----------------------------------------------------------------------


def test_scenario_1_per_invoice_approval():
    graph = REGISTRY.load(make_graph(read_invoices, split, approve, approved_total, edges=[
        ("read-invoices.result", "split.value"),
        ("split.result.number", "approve.number"),
        ("split.result.amount", "approve.amount"),
        ("split.result.amount", "approved-total.amounts"),
        ("approve.result", "approved-total.approved"),
    ]))
    assert graph.plan["approve"].series == "split.result" and graph.plan["approved-total"].mode == "once"

    run = graph.run(file=INVOICES)
    assert run.status == "waiting" and [(w.loc[0], w.loc[1:], w.prompt) for w in run.waiting] == [("approve", (1,), "Approve A2 for 900.0?")]
    assert run.answer(run.waiting[0], True).resume().outputs == {"approved-total.result": 1300.0}
    assert run.answer(run.waiting[0], False).resume().outputs == {"approved-total.result": 400.0}


# -- 2 and 5. CSV, a model call per record, a summary; edit one record and run again: 4 boxes ---------------

COMPLAINTS = "id,text,amount\n1,late delivery,10\n2,broken lid,20\n3,wrong colour,30\n"


def _csv_llm_summary():
    return REGISTRY.load(make_graph(read_csv, split, llm, llm_all, edges=[
        ("read-csv.result", "split.value"),
        ("split.result.text", "llm.text"),
        ("llm.result", "llm-all.texts"),
    ], values={"llm.prompt": "Classify:", "llm-all.prompt": "Summarise"}))


def test_scenario_2_csv_then_a_model_call_per_record_then_a_summary():
    ASKED.clear()
    run = _csv_llm_summary().run(file=File(name="complaints.csv", text=COMPLAINTS))
    assert run.results["llm"]["result"] == ["Classify: late delivery", "Classify: broken lid", "Classify: wrong colour"]
    assert run.outputs == {"llm-all.result": "Summarise 3 texts"}
    assert ASKED == ["late delivery", "broken lid", "wrong colour", "Classify: late delivery + Classify: broken lid + Classify: wrong colour"]


def test_scenario_5_editing_one_record_calls_the_model_for_that_record_only():
    first = _csv_llm_summary().run(file=File(name="complaints.csv", text=COMPLAINTS))
    ASKED.clear()
    second = first.again(file=File(name="complaints.csv", text=COMPLAINTS.replace("broken lid", "cracked lid")))
    assert ASKED == ["cracked lid", "Classify: late delivery + Classify: cracked lid + Classify: wrong colour"]
    assert second.results["llm"]["result"][1] == "Classify: cracked lid"

    ASKED.clear()
    third = second.again(file=File(name="complaints.csv", text=COMPLAINTS.replace("broken lid", "cracked lid").replace(",20", ",25")))
    assert ASKED == [] and third.outputs == second.outputs  # only `text` reaches the model, so another amount calls nothing


# -- 3. documents, their invoices, an Excel per document with the document's name on each line: 4 boxes -----


def test_scenario_3_an_excel_per_document_with_the_documents_name_as_a_column():
    graph = REGISTRY.load(make_graph(upload, find_invoices, collect, excel, edges=[
        ("upload.result.text", "find-invoices.text"),
        ("find-invoices.result", "table-collect.records"),
        ("upload.result.name", "table-collect.document"),
        ("table-collect.result", "excel.table"),
    ], values={"table-collect.constant_columns": ["document"]}))
    assert {n: graph.plan[n].series for n in ("find-invoices", "table-collect", "excel")} == dict.fromkeys(("find-invoices", "table-collect", "excel"), "upload.result")
    assert list(graph.describe().nodes["table-collect"].outputs[0].columns) == ["number", "vendor", "amount", "document"]

    sheets = graph.run(files=[A_PDF, C_PDF]).results["excel"]["result"]
    assert [sheet.text for sheet in sheets] == [
        "number,vendor,amount,document\nA1,Acme,100.0,a.pdf\nA2,Bolt,900.0,a.pdf\n",
        "number,vendor,amount,document\nC1,Core,300.0,c.pdf\nC2,Dune,700.0,c.pdf\nC3,Echo,50.0,c.pdf\n",
    ]


def test_scenario_3_a_document_with_no_invoices_gets_an_excel_with_the_columns_and_no_lines():
    graph = REGISTRY.load(make_graph(upload, find_invoices, collect, excel, edges=[
        ("upload.result.text", "find-invoices.text"),
        ("find-invoices.result", "table-collect.records"),
        ("upload.result.name", "table-collect.document"),
        ("table-collect.result", "excel.table"),
    ], values={"table-collect.constant_columns": ["document"]}))
    run = graph.run(files=[A_PDF, B_PDF])
    assert run.status == "complete" and run.results["excel"]["result"][1].text == "number,vendor,amount,document\n"


def test_scenario_3_without_the_name_a_table_input_takes_the_invoices_of_each_document():
    graph = REGISTRY.load(make_graph(upload, find_invoices, excel, edges=[
        ("upload.result.text", "find-invoices.text"), ("find-invoices.result", "excel.table"), ("upload.result.name", "excel.name"),
    ]))
    sheets = graph.run(files=[A_PDF, C_PDF]).results["excel"]["result"]
    assert [(sheet.name, sheet.text.count("\n") - 1) for sheet in sheets] == [("a.pdf.xlsx", 2), ("c.pdf.xlsx", 3)]


# -- 4. eleven contracts, each, all together, or both: 2, 2 and 3 boxes -------------------------------------

CONTRACTS = [File(name=f"{n}.pdf", text=f"contract {n}") for n in range(11)]


def test_scenario_4_each_contract_alone():
    ASKED.clear()
    graph = REGISTRY.load(make_graph(upload, llm, edges=[("upload.result.text", "llm.text")], values={"llm.prompt": "Summarise:"}))
    assert len(graph.run(files=CONTRACTS).outputs["llm.result"]) == 11 and len(ASKED) == 11


def test_scenario_4_all_contracts_together():
    ASKED.clear()
    graph = REGISTRY.load(make_graph(upload, llm_all, edges=[("upload.result.text", "llm-all.texts")], values={"llm-all.prompt": "Compare"}))
    assert graph.run(files=CONTRACTS).outputs == {"llm-all.result": "Compare 11 texts"} and len(ASKED) == 1


def test_scenario_4_each_contract_and_then_all_of_them():
    ASKED.clear()
    graph = REGISTRY.load(make_graph(upload, llm, llm_all, edges=[
        ("upload.result.text", "llm.text"), ("llm.result", "llm-all.texts"),
    ], values={"llm.prompt": "Summarise:", "llm-all.prompt": "Give an overview of"}))
    assert graph.run(files=CONTRACTS).outputs == {"llm-all.result": "Give an overview of 11 texts"} and len(ASKED) == 12


# -- 6. documents, their lines, a count per document; b.pdf gives 0: 3 boxes --------------------------------


def test_scenario_6_a_document_with_no_lines_counts_zero():
    graph = REGISTRY.load(make_graph(upload, split_text, count, edges=[
        ("upload.result.text", "text-split.text"), ("text-split.result", "count.values"),
    ]))
    run = graph.run(files=[A_PDF, B_PDF, C_PDF])
    assert run.results["count"]["result"] == [2, 0, 3] and run.not_called == []


# -- 7. CSV, a mail per contact, an Excel of everyone; Bo's email is blank: 5 boxes -------------------------


@pytest.mark.parametrize("reader", [read_contacts, read_csv], ids=["a table whose email column may be blank", "a table of unknown columns"])
def test_scenario_7_a_contact_without_an_email_gets_no_mail_and_stays_in_the_excel(reader: Any):
    graph = REGISTRY.load(make_graph(reader, split, send_mail, collect, excel, edges=[
        (f"{reader.id}.result", "split.value"),
        ("split.result.email", "send-mail.to"),
        ("split.result", "table-collect.records"),
        ("send-mail.result", "table-collect.sent"),
        ("table-collect.result", "excel.table"),
    ], values={"send-mail.body": "Hello", "table-collect.columns": ["sent"]}))
    run = graph.run(file=CONTACTS)
    assert run.status == "complete" and [(n.loc[0], n.loc[1:], n.input) for n in run.not_called] == [("send-mail", (1,), "to")]
    assert run.results["send-mail"]["result"] == ["sent to ann@x.dk", None, "sent to cy@x.dk"]
    assert run.outputs["excel.result"].text == "name,email,sent\nAnn,ann@x.dk,sent to ann@x.dk\nBo,,\nCy,cy@x.dk,sent to cy@x.dk\n"


# -- 8. invoices, an approval each, an Excel per document of the approved ones: 5 boxes ---------------------


def test_scenario_8_an_excel_per_document_of_the_approved_invoices():
    graph = REGISTRY.load(make_graph(upload, find_invoices, approve, decision, excel, edges=[
        ("upload.result.text", "find-invoices.text"),
        ("find-invoices.result.number", "approve.number"),
        ("find-invoices.result.amount", "approve.amount"),
        ("find-invoices.result", "decision.value"),
        ("approve.result", "decision.when"),
        ("decision.if_true", "excel.table"),
        ("upload.result.name", "excel.name"),
    ]))
    assert graph.plan["decision"].series == "find-invoices.result" and graph.plan["excel"].series == "upload.result"

    only_large = File(name="d.pdf", text="D1,Dune,800")
    run = graph.run(files=[A_PDF, C_PDF, only_large])
    assert [(w.loc[1:], w.inputs["number"]) for w in run.waiting] == [((0, 1), "A2"), ((1, 1), "C2"), ((2, 0), "D1")]
    sheets = answered(run, yes={"A2"}).results["excel"]["result"]
    assert [sheet.name for sheet in sheets] == ["a.pdf.xlsx", "c.pdf.xlsx", "d.pdf.xlsx"]
    assert [sheet.text for sheet in sheets] == [
        "number,vendor,amount\nA1,Acme,100.0\nA2,Bolt,900.0\n",
        "number,vendor,amount\nC1,Core,300.0\nC3,Echo,50.0\n",
        "\n",  # every invoice rejected: a sheet with no lines
    ]


# -- 9. a branch per invoice and back together: 6 boxes -----------------------------------------------------


def test_scenario_9_a_branch_per_item_and_back_together_in_the_items_order():
    graph = REGISTRY.load(make_graph(read_invoices, split, by_size, review, book, report, edges=[
        ("read-invoices.result", "split.value"),
        ("split.result", "by-size.invoice"),
        ("by-size.large", "review.invoice"),
        ("by-size.small", "book.invoice"),
        ("review.result", "report.notes"),
        ("book.result", "report.notes"),
    ]))
    run = graph.run(file=INVOICES)
    assert run.outputs == {"report.result": "booked A1; review A2; booked A3"}
    assert run.results["review"]["result"] == [None, "review A2", None] and run.results["book"]["result"] == ["booked A1", None, "booked A3"]


# -- 10. a person approves each invoice of each document; saved and resumed: 3 boxes ------------------------


def test_scenario_10_a_nested_approval_saved_and_resumed_in_another_process():
    graph = REGISTRY.load(make_graph(upload, find_invoices, approve, edges=[
        ("upload.result.text", "find-invoices.text"),
        ("find-invoices.result.number", "approve.number"),
        ("find-invoices.result.amount", "approve.amount"),
    ]))
    run = graph.run(files=[A_PDF, B_PDF, C_PDF])
    first, second = run.waiting
    assert (first.loc[0], first.loc[1:], second.loc[1:]) == ("approve", (0, 1), (2, 1))  # invoice 1 of document 0, invoice 1 of document 2

    saved = run.to_json()
    resumed = graph.load_run(saved).answer(second.id, False).resume()
    assert resumed.status == "waiting" and [w.loc[1:] for w in resumed.waiting] == [(0, 1)]
    done = graph.load_run(resumed.to_json()).answer(first.id, True).resume()
    assert done.outputs == {"approve.result": [True, True, True, False, True]}  # every item, in order: two of a.pdf, none of b.pdf, three of c.pdf


# -- 11. a sub-graph per item: 3 boxes ------------------------------------------------------------------------


@pytest.mark.skip(reason="embedding: nested run not built")
def test_scenario_11_a_sub_graph_per_item():
    enrich = node("enrich", title="Enrich a contact", description="Looks a contact up and writes a greeting.")(
        make_graph(llm, values={"llm.prompt": "Write a greeting for"})
    )
    graph = registry_with(enrich).load(make_graph(read_contacts, split, enrich, edges=[
        ("read-contacts.result", "split.value"), ("split.result.name", "enrich.text"),
    ]))
    assert graph.plan["enrich"].series == "split.result"
    assert graph.run(file=CONTACTS).outputs == {"enrich.result": ["Write a greeting for Ann", "Write a greeting for Bo", "Write a greeting for Cy"]}


# -- 12. a sub-graph given the whole table: 2 boxes -----------------------------------------------------------


@pytest.mark.skip(reason="embedding: nested run not built")
def test_scenario_12_a_sub_graph_given_the_whole_table():
    table_report = node("table-report", title="Report on a table", description="Writes a report on a table.")(
        make_graph(describe_table, llm, edges=[("describe-table.result", "llm.text")], values={"llm.prompt": "Write a report on"})
    )
    graph = registry_with(table_report).load(make_graph(read_csv, table_report, edges=[("read-csv.result", "table-report.table")]))
    assert graph.plan["table-report"].mode == "once"
    assert graph.run(file=File(name="complaints.csv", text=COMPLAINTS)).outputs == {"table-report.result": "Write a report on 3 lines of id, text, amount"}


# -- 13. contacts and orders joined on a key: 3 boxes ---------------------------------------------------------


def test_scenario_13_two_tables_joined_on_a_key_are_two_values():
    graph = REGISTRY.load(make_graph(read_csv, read_csv, join_tables, edges=[
        ("read-csv-1.result", "join-tables.left"), ("read-csv-2.result", "join-tables.right"),
    ], values={"join-tables.on": "customer_id"}))
    assert all(runs.mode == "once" for runs in graph.plan.values())
    customers = File(name="customers.csv", text="customer_id,name\n1,Ann\n2,Bo\n")
    orders = File(name="orders.csv", text="customer_id,item\n2,lamp\n1,desk\n1,chair\n")
    joined = graph.run({"read-csv-1.file": customers, "read-csv-2.file": orders}).outputs["join-tables.result"]
    assert joined.sort("item").to_dicts() == [
        {"customer_id": 1, "name": "Ann", "item": "chair"}, {"customer_id": 1, "name": "Ann", "item": "desk"},
        {"customer_id": 2, "name": "Bo", "item": "lamp"},
    ]


# -- 14. each document by three typed questions: 3 boxes ------------------------------------------------------

QUESTIONS = ["Who are the parties?", "When does it end?", "What does it cost?"]


def test_scenario_14_each_document_by_three_typed_questions():
    graph = REGISTRY.load(make_graph(upload, text_list, llm, edges=[
        ("upload.result", "text-list.for_each"),
        ("text-list.result", "llm.prompt"),
        ("upload.result.text", "llm.text"),
    ], values={"text-list.items": QUESTIONS}))
    assert graph.plan["text-list"].series == "upload.result" and graph.plan["llm"].series == "text-list.result"
    feeds = {i.name: (i.feed.mode, i.feed.series) for i in graph.describe().nodes["llm"].inputs}
    assert feeds == {"prompt": ("per_item", "text-list.result"), "text": ("outer_value", "upload.result")}

    ASKED.clear()
    first = graph.run(files=CONTRACTS)
    assert len(ASKED) == 33 and first.outputs["llm.result"][12:15] == [f"{question} contract 4" for question in QUESTIONS]

    ASKED.clear()
    changed = [*CONTRACTS[:4], File(name="4.pdf", text="contract 4, amended"), *CONTRACTS[5:]]
    assert first.again(files=changed).outputs["llm.result"][12] == "Who are the parties? contract 4, amended" and len(ASKED) == 3


def test_scenario_14_without_the_for_each_wire_the_two_series_do_not_nest_and_the_problem_names_the_wire():
    graph = REGISTRY.load(make_graph(upload, text_list, llm, edges=[
        ("text-list.result", "llm.prompt"), ("upload.result.text", "llm.text"),
    ], values={"text-list.items": QUESTIONS}))
    (problem,) = graph.problems
    assert (problem.type, problem.loc) == ("series_not_nested", ("llm",))
    assert (problem.ctx["wire_from"], problem.ctx["wire_into"], problem.ctx["inputs"]) == ("upload.result", "text-list", ["prompt", "text"])


def test_scenario_14b_the_summary_of_several_takes_one_document_as_a_series_of_one():
    ASKED.clear()
    graph = REGISTRY.load(make_graph(read_pdf, llm_all, edges=[("read-pdf.result", "llm-all.texts")], values={"llm-all.prompt": "Summarise"}))
    assert graph.run(file=A_PDF).outputs == {"llm-all.result": "Summarise 1 texts"} and ASKED == [A_PDF.text]
