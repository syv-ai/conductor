"""A node that asks says so: ``asks=True`` on its ``@node``, or a parameter filled by a function that may ask.

Both follow one rule: a function that raises ``Ask`` is answered with a value of its return type. A node whose own
body asks is answered with its outputs. A parameter ``ok: Annotated[bool, Asked(approval)]`` is answered with a
``bool``: ``approval`` is called with the node's inputs it names, and its return value, or else the person's answer,
is ``ok`` when the node is called. The node is called once, after the answer.
"""

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest
from conductor import (
    Ask,
    Asked,
    Choice,
    Context,
    ExternalError,
    Graph,
    InvalidCallError,
    NodeDeclarationError,
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


def approval(invoice: Invoice) -> bool:
    calls.append(("approval", invoice["number"]))
    if invoice["amount"] <= 500:
        return True
    raise Ask(f"Pay {invoice['number']}?")


@node("approve", title="Approve", description="For a test.")
def approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str:
    calls.append(("approve", invoice["number"], ok))
    return f"{'paid' if ok else 'held'} {invoice['number']}"


@node("summary", title="Summary", description="For a test.")
def summary(lines: Series[str]) -> str:
    return "; ".join(lines)


def _graph(*more: Any) -> Graph:
    return Graph.compile(make_graph(invoices, approve, summary), registry_of(invoices, approve, summary, *more))


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    calls.clear()


def _refused(call: Any) -> list[Any]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    return raised.value.errors()


def _declaration_code(declare: Any) -> str:
    with pytest.raises(NodeDeclarationError) as raised:
        declare()
    return raised.value.code


# -- a node whose own body asks says asks=True -------------------------------------------------------


def test_a_node_whose_body_asks_declares_asks_and_waits():
    @node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask(f"Why was {ticket} closed?")

    run = Graph.compile(make_graph(reason), registry_of(reason)).run(ticket="T-1")
    assert run.status == "waiting"
    assert run.answer(run.waiting[0], "fixed").resume().outputs == {"reason.result": "fixed"}


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


def test_describe_says_whether_a_node_asks_and_which_parameter_a_person_fills():
    @node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    assert (reason.describe().asks, reason.describe().asked) == (True, None)
    assert (invoices.describe().asks, invoices.describe().asked) == (False, None)
    described = approve.describe()
    assert described.asks
    assert described.asked is not None and (described.asked.name, described.asked.type) == ("ok", "bool")
    assert [field.name for field in described.inputs] == ["invoice"]


def test_a_registry_node_takes_asks_as_node_does():
    registry = NodeRegistry()

    @registry.node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    assert registry["reason"].describe().asks


# -- a parameter filled by a function that may ask ---------------------------------------------------


def test_the_asking_function_returns_and_the_node_is_called_once_and_nothing_waits():
    run = _graph().run(text="A:100,B:200")
    assert run.outputs == {"summary.result": "paid A; paid B"}
    assert calls == [("approval", "A"), ("approve", "A", True), ("approval", "B"), ("approve", "B", True)]


def test_the_asking_function_asks_and_the_answer_of_its_return_type_fills_the_parameter():
    run = _graph().run(text="A:100,B:900")
    assert run.status == "waiting"
    (waiting,) = run.waiting
    assert waiting.prompt == "Pay B?" and waiting.id.startswith("approve@1~")
    assert [(question.name, question.type) for question in waiting.questions] == [("ok", "bool")]
    assert waiting.inputs == {"invoice": {"number": "B", "amount": 900}}
    assert ("approve", "B", False) not in calls and ("approve", "B", True) not in calls
    calls.clear()
    done = run.answer(waiting, False).resume()
    assert done.outputs == {"summary.result": "paid A; held B"}
    assert calls == [("approve", "B", False)]  # neither function is called again for A, nor the question again for B


def test_an_answer_of_another_type_is_refused_by_the_answer_door():
    run = _graph().run(text="B:900")
    (problem,) = _refused(lambda: run.answer(run.waiting[0], "yes"))
    assert problem.type == "invalid_answer"


def test_the_asks_choices_and_default_are_checked_against_the_parameters_type():
    def owner_of(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), "bo"], default="ida")

    @node("assign", title="Assign", description="For a test.")
    def assign(ticket: str, owner: Annotated[str, Asked(owner_of)]) -> str:
        return f"{ticket} to {owner}"

    run = Graph.compile(make_graph(assign), registry_of(assign)).run(ticket="T-1")
    (question,) = run.waiting[0].questions
    assert [choice["id"] for choice in question.choices] == ["ida", "bo"] and question.default == "ida"
    (problem,) = _refused(lambda: run.answer(run.waiting[0], "al"))
    assert problem.type == "invalid_answer"
    assert run.answer(run.waiting[0], "bo").resume().outputs == {"assign.result": "T-1 to bo"}

    def wrong_choices(ticket: str) -> str:
        raise Ask("Who?", choices=[1, 2])

    @node("assign2", title="Assign", description="For a test.")
    def assign2(ticket: str, owner: Annotated[str, Asked(wrong_choices)]) -> str:
        return owner

    failed = Graph.compile(make_graph(assign2), registry_of(assign2)).run(ticket="T-1")
    assert failed.state.cause.type == "invalid_ask"


def test_a_value_the_asking_function_returns_is_read_by_the_parameters_type():
    def says_yes(text: str) -> bool:
        return "yes"  # type: ignore[return-value]

    @node("check", title="Check", description="For a test.")
    def check(text: str, ok: Annotated[bool, Asked(says_yes)]) -> bool:
        return ok

    run = Graph.compile(make_graph(check), registry_of(check)).run(text="a")
    assert (run.state.cause.type, run.state.cause.ctx["name"]) == ("invalid_asked_value", "ok")


def test_a_waiting_run_saved_and_read_back_in_another_process_calls_the_node_once_with_the_typed_answer():
    graph = _graph()
    run = Run.from_json(graph.run(text="B:900").to_json(), graph)
    answered = Run.from_json(run.answer(run.waiting[0], True).to_json(), graph)  # an answer given and not yet used is saved
    assert answered.waiting == []
    calls.clear()
    assert answered.resume().outputs == {"summary.result": "paid B"}
    assert calls == [("approve", "B", True)]
    done = answered.resume()
    assert Run.from_json(done.to_json(), graph).outputs == done.outputs


def test_a_node_that_fails_after_the_answer_is_retried_and_resumed_with_the_same_answer_and_asks_nothing_again():
    attempts: list[bool] = []

    @node("pay", title="Pay", description="For a test.", policy=Policy(retries=1, retry_delay=0))
    def pay(invoice: Invoice, ok: Annotated[bool, Asked(approval)]) -> str:
        attempts.append(ok)
        if len(attempts) == 1:
            raise ExternalError("the bank is down")
        if len(attempts) == 2:
            raise NodeError("still down", code="bank_down")
        return f"paid {invoice['number']}"

    graph = Graph.compile(make_graph(invoices, pay), registry_of(invoices, pay))
    run = graph.run(text="B:900")
    failed = run.answer(run.waiting[0], True).resume()
    assert failed.status == "failed" and attempts == [True, True]
    calls.clear()
    assert Run.from_json(failed.to_json(), graph).resume().results["pay"]["result"] == ["paid B"]
    assert attempts == [True, True, True] and calls == []


def test_again_with_one_invoice_changed_asks_for_that_invoice_only_and_its_old_waiting_id_is_stale():
    graph = _graph()
    run = graph.run(text="A:900,B:900")
    old_a, old_b = (waiting.id for waiting in run.waiting)
    again = run.again(text="A:900,B:950")
    assert [waiting.id for waiting in again.waiting][0] == old_a
    assert again.waiting[1].id != old_b
    (problem,) = _refused(lambda: again.answer(old_b, True))
    assert problem.type == "answer_stale"


def test_two_identical_invoices_answered_differently_keep_their_answers_when_one_is_put_in_front():
    graph = _graph()
    run = graph.run(text="A:900,A:900")
    first, second = run.waiting
    run = run.answer({first.id: True, second.id: False}).resume()
    assert run.outputs == {"summary.result": "paid A; held A"}
    calls.clear()
    again = run.again(text="C:100,A:900,A:900")
    assert again.outputs == {"summary.result": "paid C; paid A; held A"}
    assert calls == [("approval", "C"), ("approve", "C", True)]


def test_an_answer_given_and_not_yet_used_is_not_carried_to_a_run_of_changed_inputs():
    graph = _graph()
    run = graph.run(text="A:900")
    answered = run.answer(run.waiting[0], True)
    moved = answered.with_inputs(text="A:900,B:100").resume()
    assert [waiting.prompt for waiting in moved.waiting] == ["Pay A?"]


def test_the_asking_function_takes_the_nodes_context_by_its_name():
    @dataclass(frozen=True)
    class Team:
        people: tuple[str, ...]

    def owner_of(ctx: Context[Team], ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=list(ctx.people))

    registry = NodeRegistry(context_type=Team)

    @registry.node("assign", title="Assign", description="For a test.")
    def assign(ctx: Context[Team], ticket: str, owner: Annotated[str, Asked(owner_of)]) -> str:
        return f"{ticket} to {owner}"

    run = Graph.compile(make_graph(assign), registry).run(ticket="T-1", context=Team(("ida", "bo")))
    assert [choice["id"] for choice in run.waiting[0].questions[0].choices] == ["ida", "bo"]


def test_no_edge_or_constant_reaches_an_asked_parameter():
    record = make_graph(invoices, approve, summary, constants={"approve.ok": True})
    (problem,) = Graph.compile(record, registry_of(invoices, approve, summary)).problems
    assert problem.type == "unknown_input"


def test_a_changed_asking_function_asks_again():
    graph = _graph()
    run = graph.run(text="A:900")
    run = run.answer(run.waiting[0], True).resume()

    def stricter(invoice: Invoice) -> bool:
        raise Ask(f"Really pay {invoice['number']}?")

    @node("approve", title="Approve", description="For a test.")
    def approve_again(invoice: Invoice, ok: Annotated[bool, Asked(stricter)]) -> str:
        return f"{'paid' if ok else 'held'} {invoice['number']}"

    other = Graph.compile(make_graph(invoices, approve_again, summary), registry_of(invoices, approve_again, summary))
    moved = other.with_inputs(text="A:900", previous=run).resume()
    assert [waiting.prompt for waiting in moved.waiting] == ["Really pay A?"]


# -- what @node refuses -----------------------------------------------------------------------------


def _yes(text: str) -> bool:
    return True


def test_two_asked_parameters_are_refused():
    def two(text: str, a: Annotated[bool, Asked(_yes)], b: Annotated[bool, Asked(_yes)]) -> bool:
        return a and b

    assert _declaration_code(lambda: node("two", title="Two", description="For a test.")(two)) == "asked_twice"


def test_an_asked_parameter_on_a_node_that_also_declares_asks_is_refused():
    def both(text: str, ok: Annotated[bool, Asked(_yes)]) -> bool:
        raise Ask("And?")

    assert _declaration_code(lambda: node("both", title="Both", description="For a test.", asks=True)(both)) == "asked_and_asks"


def test_asked_on_a_return_type_is_refused():
    def returns(text: str) -> Annotated[bool, Asked(_yes)]:
        return True

    assert _declaration_code(lambda: node("returns", title="Returns", description="For a test.")(returns)) == "asked_not_on_a_parameter"


def test_an_asked_parameter_with_a_default_is_refused():
    def defaulted(text: str, ok: Annotated[bool, Asked(_yes)] = True) -> bool:
        return ok

    assert _declaration_code(lambda: node("defaulted", title="Defaulted", description="For a test.")(defaulted)) == "asked_with_default"


def test_an_asking_function_that_names_no_parameter_of_the_node_is_refused():
    def reads_other(other: str) -> bool:
        return True

    def asked(text: str, ok: Annotated[bool, Asked(reads_other)]) -> bool:
        return ok

    code = _declaration_code(lambda: node("asked", title="Asked", description="For a test.")(asked))
    assert code == "asking_function_parameter_unknown"


def test_an_asking_function_that_returns_another_type_than_the_parameter_is_refused():
    def says(text: str) -> str:
        return text

    def asked(text: str, ok: Annotated[bool, Asked(says)]) -> bool:
        return ok

    code = _declaration_code(lambda: node("asked", title="Asked", description="For a test.")(asked))
    assert code == "asking_function_returns_another_type"


def test_an_async_asking_function_is_refused():
    async def later(text: str) -> bool:
        return True

    def asked(text: str, ok: Annotated[bool, Asked(later)]) -> bool:
        return ok

    assert _declaration_code(lambda: node("asked", title="Asked", description="For a test.")(asked)) == "async_run"
