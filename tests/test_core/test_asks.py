"""A node that may ask a person says ``asks=True``, and the answer is its outputs.

A node that needs the answer reads it from an edge, as it reads any value. "Approve, then pay" is two nodes:
``approval(invoice) -> bool`` returns ``True`` for a small invoice and asks for a large one, and
``pay(invoice, ok: bool)`` reads its ``result``. A host that offers them as one palette entry makes them a ``subgraph``.
"""

import json
from typing import Any, TypedDict

import pytest
from conductor import (
    Ask,
    ExternalError,
    Graph,
    InvalidCallError,
    NodeError,
    NodeRegistry,
    Policy,
    Run,
    Series,
    make_graph,
    node,
)

from tests.test_core.nodes import registry_of


class Invoice(TypedDict):
    number: str
    amount: int


calls: list[tuple[Any, ...]] = []


@node("invoices", title="Invoices", description="For a test.")
def invoices(text: str) -> Series[Invoice]:
    found = [line.split(":") for line in text.split(",") if line]
    return [Invoice(number=number, amount=int(amount)) for number, amount in found]


@node("approval", title="Approval", description="For a test.", asks=True)
def approval(invoice: Invoice) -> bool:
    calls.append(("approval", invoice["number"]))
    if invoice["amount"] <= 500:
        return True
    raise Ask(f"Pay {invoice['number']}?")


@node("pay", title="Pay", description="For a test.")
def pay(invoice: Invoice, ok: bool) -> str:
    calls.append(("pay", invoice["number"], ok))
    return f"{'paid' if ok else 'held'} {invoice['number']}"


@node("summary", title="Summary", description="For a test.")
def summary(lines: Series[str]) -> str:
    return "; ".join(lines)


_EDGES = [
    ("invoices.result", "approval.invoice"), ("invoices.result", "pay.invoice"), ("approval.result", "pay.ok"), ("pay.result", "summary.lines"),
]


def _graph(paying: Any = pay) -> Graph:
    return Graph.compile(make_graph(invoices, approval, paying, summary, edges=_EDGES), registry_of(invoices, approval, paying, summary))


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    calls.clear()


def _refused(call: Any) -> list[Any]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    return raised.value.errors()


# -- a node that asks says so ------------------------------------------------------------------------


def test_a_raise_of_ask_from_a_node_that_does_not_declare_asks_fails_the_run_naming_the_node():
    @node("reason", title="Reason", description="For a test.")
    def reason(ticket: str) -> str:
        raise Ask(f"Why was {ticket} closed?")

    run = Graph.compile(make_graph(reason), registry_of(reason)).run(ticket="T-1")
    assert run.status == "failed"
    assert (run.state.cause.type, run.state.cause.loc) == ("ask_not_declared", ("reason",))
    assert "asks=True" in run.state.cause.msg


def test_an_undeclared_ask_on_a_thread_fails_the_same_way():
    @node("reason", title="Reason", description="For a test.", policy=Policy(concurrency=2))
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    run = Graph.compile(make_graph(reason), registry_of(reason)).run(ticket="T-1")
    assert run.state.cause.type == "ask_not_declared"


def test_describe_says_whether_a_node_asks_and_a_registry_node_takes_asks_as_node_does():
    registry = NodeRegistry()

    @registry.node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    assert (registry["reason"].describe().asks, approval.describe().asks, pay.describe().asks) == (True, True, False)


# -- approve, then pay -------------------------------------------------------------------------------


def test_approval_then_pay_waits_only_when_approval_asks():
    run = _graph().run(text="A:100,B:200")
    assert run.outputs == {"result": "paid A; paid B"}
    assert calls == [("approval", "A"), ("approval", "B"), ("pay", "A", True), ("pay", "B", True)]

    calls.clear()
    run = _graph().run(text="A:100,B:900")
    (waiting,) = run.waiting
    assert waiting.prompt == "Pay B?" and waiting.id.startswith("approval@1~")
    assert [(question.name, question.type) for question in waiting.questions] == [("result", "bool")]
    assert waiting.inputs == {"invoice": {"number": "B", "amount": 900}}
    assert ("pay", "A", True) in calls and not any(call[:2] == ("pay", "B") for call in calls)


def test_an_answer_resumes_pay_with_ok_and_asks_nothing_again():
    run = _graph().run(text="A:100,B:900")
    calls.clear()
    done = run.answer(run.waiting[0], False).resume()
    assert done.outputs == {"result": "paid A; held B"}
    assert calls == [("pay", "B", False)]


def test_an_answer_saved_before_resume_and_read_back_in_another_process_resumes_pay_with_it():
    graph = _graph()
    run = Run.from_json(graph.run(text="B:900").to_json(), graph)
    answered = Run.from_json(run.answer(run.waiting[0], True).to_json(), graph)
    assert answered.waiting == []
    calls.clear()
    assert answered.resume().outputs == {"result": "paid B"}
    assert calls == [("pay", "B", True)]


def test_pay_failing_after_the_answer_is_retried_and_resumed_and_nothing_is_asked_again():
    attempts: list[bool] = []

    @node("pay", title="Pay", description="For a test.", policy=Policy(retries=1, retry_delay=0))
    def flaky_pay(invoice: Invoice, ok: bool) -> str:
        attempts.append(ok)
        if len(attempts) == 1:
            raise ExternalError("the bank is down")
        if len(attempts) == 2:
            raise NodeError("still down", code="bank_down")
        return f"paid {invoice['number']}"

    graph = _graph(flaky_pay)
    run = graph.run(text="B:900")
    failed = run.answer(run.waiting[0], True).resume()
    assert failed.status == "failed" and attempts == [True, True]
    calls.clear()
    assert Run.from_json(failed.to_json(), graph).resume().outputs == {"result": "paid B"}
    assert attempts == [True, True, True] and calls == []


# -- with_inputs reuses the answer ---------------------------------------------------------------------


def test_with_inputs_reuses_the_answer_and_asks_only_for_the_invoice_that_changed():
    graph = _graph()
    run = graph.run(text="A:900,B:900")
    old_a, old_b = (waiting.id for waiting in run.waiting)
    run = run.answer({old_a: True, old_b: False}).resume()
    calls.clear()
    again = graph.with_inputs(text="A:900,B:950", previous=run).resume()
    assert [waiting.prompt for waiting in again.waiting] == ["Pay B?"]
    assert calls == [("approval", "B")]  # pay for A is reused too: its arguments are the same
    (problem,) = _refused(lambda: again.answer(old_b, True))
    assert problem.type == "answer_stale"


def test_two_identical_invoices_answered_differently_keep_their_answers_when_one_is_put_in_front():
    graph = _graph()
    run = graph.run(text="A:900,A:900")
    first, second = run.waiting
    run = run.answer({first.id: True, second.id: False}).resume()
    assert run.outputs == {"result": "paid A; held A"}
    calls.clear()
    again = graph.with_inputs(text="C:100,A:900,A:900", previous=run).resume()
    assert again.outputs == {"result": "paid C; paid A; held A"}
    assert calls == [("approval", "C"), ("pay", "C", True)]


def test_an_answer_given_and_not_yet_resumed_is_reused_by_a_run_of_changed_inputs():
    graph = _graph()
    run = graph.run(text="A:900")
    answered = run.answer(run.waiting[0], True)
    moved = graph.with_inputs(text="A:900,B:100", previous=answered).resume()
    assert moved.outputs == {"result": "paid A; paid B"}


# -- one palette entry ---------------------------------------------------------------------------------


def test_approval_and_pay_as_a_subgraph_is_one_node_that_waits_where_approval_asks():
    record = make_graph(approval, pay, edges=[("approval.result", "pay.ok")], inputs={"invoice": ["approval.invoice", "pay.invoice"]})
    registry = registry_of(invoices, approval, pay, summary)
    approve_and_pay = registry.subgraph("approve_and_pay", title="Approve and pay", description="For a test.")(record)
    edges = [("invoices.result", "approve_and_pay.invoice"), ("approve_and_pay.result", "summary.lines")]
    graph = Graph.compile(make_graph(invoices, approve_and_pay, summary, edges=edges), registry)
    assert sorted(graph.describe().nodes) == ["approve_and_pay", "invoices", "summary"]
    assert "approve_and_pay" in {entry.id for entry in registry.describe().nodes}

    run = graph.run(text="A:100,B:900")
    (waiting,) = run.waiting
    assert waiting.loc == ("approve_and_pay", 1, "approval") and waiting.prompt == "Pay B?"
    assert run.answer(waiting, True).resume().outputs == {"result": "paid A; paid B"}


# -- what a saved run holds ----------------------------------------------------------------------------


def test_a_saved_run_with_answers_kept_beside_its_outcomes_is_refused():
    graph = _graph()
    run = graph.run(text="A:900")
    saved = json.loads(run.answer(run.waiting[0], True).to_json())
    assert "answers" not in saved
    (problem,) = _refused(lambda: Run.from_json(json.dumps({**saved, "answers": {"pay": {"0": True}}}), graph))
    assert problem.type == "saved_run_unreadable"
