"""Asking a person: a node with ``asks=True`` raises ``Ask``, its return type says what the answer is, and the ``Ask``
adds what only the run knows.

A waiting call is a ``Waiting`` whose ``questions`` are the field records of the node's outputs with what the ``Ask``
set: ``approve(line: str) -> bool`` raising ``Ask("Approve b?")`` has one question, ``result``, a yes or no.
``run.answer`` checks an answer against the questions and the node's types, refuses every fault at once, and calls
nothing. An ``Asked`` parameter, the other way a node asks, is tested in ``test_asked.py``.
"""

import datetime
import json
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Choice,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    Param,
    Policy,
    Run,
    Series,
    Table,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Textarea

from tests.test_core.nodes import place, registry_of


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",") if text else []


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


def _waiting_outcomes(definition: Any, *more: Any, text: str = "a,b") -> Any:
    """A graph of ``split``, which splits ``text`` at commas, and ``definition`` as node ``q``, called once per piece;
    the outputs of ``q`` are the graph's outputs. Returns the graph and its first run."""
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": {
            "s": {"node": "split", "inputs": {"text": {"constant": text}}},
            "q": {"node": definition.id, "inputs": {next(iter(definition.inputs)): {"sources": [{"node": "s", "output": "result"}]}}},
        },
        "outputs": _outputs_of(definition),
    }), registry_of(split, definition, *more))
    return graph, graph.run()


def _once(definition: Any, **values: Any) -> Any:
    """A graph of ``definition`` alone as node ``q`` with ``values`` as its constants and its outputs as the graph's,
    run once; returns the graph and the run."""
    inputs = {name: {"constant": value} for name, value in values.items()}
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"q": {"node": definition.id, "inputs": inputs}}, "outputs": _outputs_of(definition)}), registry_of(definition))
    return graph, graph.run()


def _outputs_of(definition: Any) -> dict[str, Any]:
    """Each output of ``definition``, as node ``q``, as a graph output of its name."""
    return {name: {"node": "q", "output": name} for name in definition.outputs}


def _refused(call: Any) -> list[Any]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    return raised.value.errors()


# -- the return type says how to ask ---------------------------------------------------------------


def test_a_text_return_with_a_bare_ask_is_a_free_text_question():
    @node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask(f"Why was {ticket} closed?")

    _, run = _once(reason, ticket="T-1")
    (waiting,) = run.waiting
    (question,) = waiting.questions
    assert (waiting.prompt, question.type, question.choices, question.default, question.required) == ("Why was T-1 closed?", "str", None, None, True)
    assert run.answer(waiting, "a duplicate").resume().outputs == {"result": "a duplicate"}


def test_how_a_question_looks_is_its_types_so_a_hook_can_give_each_output_a_widget():
    def outputs(names: str) -> dict[str, Any]:
        return {name: Annotated[str, Param(widget=Textarea(rows=2))] for name in names.split(",")}

    @node("notes", title="Notes", description="For a test.", compute_outputs=outputs, asks=True)
    def notes(names: str) -> dict[str, Any]:
        raise Ask("Notes", default={"a": "none"})

    _, run = _once(notes, names="a,b")
    assert [(q.name, q.widget, q.default, q.required) for q in run.waiting[0].questions] == [
        ("a", {"type": "textarea", "rows": 2}, "none", False), ("b", {"type": "textarea", "rows": 2}, None, True),
    ]


def test_a_node_that_returns_nothing_is_answered_with_none_to_go_on():
    @node("ack", title="Ack", description="For a test.", asks=True)
    def ack(text: str) -> None:
        raise Ask(f"Seen {text}?")

    _, run = _once(ack, text="x")
    assert run.waiting[0].questions == []
    assert run.answer("q", None).resume().status == "complete"


def test_an_ask_that_sets_nothing_saves_only_its_prompt_and_inputs():
    @node("plain", title="Plain", description="For a test.", asks=True)
    def plain(text: str) -> str:
        raise Ask("?")

    _, run = _once(plain, text="x")
    assert '"questions":{"0":{"prompt":"?","inputs":{"text":"x"}}}' in run.to_json()


def test_a_bool_is_a_yes_or_no_and_a_word_for_it_is_refused():
    @node("confirm", title="Confirm", description="For a test.", asks=True)
    def confirm(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph, run = _once(confirm, name="F-1")
    (waiting,) = run.waiting
    (question,) = waiting.questions
    assert (place(waiting), waiting.prompt, question.json_schema, question.required) == ("q", "Approve F-1?", {"type": "boolean"}, True)
    (fault,) = _refused(lambda: run.answer(waiting, "maybe"))
    assert (fault.type, fault.loc, fault.ctx["waiting_id"]) == ("invalid_answer", ("q",), waiting.id)
    assert run.answer(waiting, True).resume().outputs == {"result": True}


def test_a_literal_offers_its_values_and_an_answer_outside_them_is_refused():
    @node("urgency", title="Urgency", description="For a test.", asks=True)
    def urgency(name: str) -> Literal["low", "high"]:
        raise Ask("How urgent?")

    _, run = _once(urgency, name="x")
    (question,) = run.waiting[0].questions
    assert question.choices == [{"id": "low", "title": "low"}, {"id": "high", "title": "high"}]
    assert _refused(lambda: run.answer("q", "urgent"))[0].type == "invalid_answer"
    assert run.answer("q", "high").resume().outputs == {"result": "high"}


def test_choices_the_run_knows_limit_the_answer_and_a_choice_gives_a_title():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), "bo"])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == [{"id": "ida", "title": "Ida B."}, {"id": "bo", "title": "bo"}]
    (fault,) = _refused(lambda: run.answer("q", "carl"))
    assert fault.msg == "That is not a valid answer: the answer is not one of the choices"
    assert run.answer("q", "bo").resume().outputs == {"result": "bo"}


class Person(Choice):
    team: str = ""


def test_a_hosts_own_choice_carries_its_fields_to_the_question():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Person("ida", "Ida B.", team="sales")])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == [{"id": "ida", "title": "Ida B.", "team": "sales"}]
    assert run.answer("q", "ida").resume().outputs == {"result": "ida"}


def test_a_number_and_a_date_are_read_from_their_json_form_into_the_outputs_type():
    @node("when", title="When", description="For a test.", asks=True)
    def when(name: str) -> datetime.date:
        raise Ask("When?")

    @node("weekday", title="Weekday", description="For a test.")
    def weekday(day: datetime.date) -> int:
        return day.isoweekday()

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"q": {"node": "when", "inputs": {"name": {"constant": "x"}}}, "w": {"node": "weekday", "inputs": {"day": {"sources": [{"node": "q", "output": "result"}]}}}}, "outputs": {"result": {"node": "w", "output": "result"}}}), registry_of(when, weekday))
    run = graph.run()
    assert run.waiting[0].questions[0].json_schema == {"format": "date", "type": "string"}
    done = Run.from_json(run.answer("q", "2026-11-10").to_json(), graph).resume()
    assert done.results["q"]["result"] == datetime.date(2026, 11, 10) and done.outputs == {"result": 2}
    assert _refused(lambda: run.answer("q", "next tuesday"))[0].type == "invalid_answer"


def test_a_lone_output_is_titled_by_its_own_label_never_by_the_node():
    @node("ask", description="For a test.", title="Approve the invoice", asks=True)
    def ask() -> str:
        raise Ask("?")

    @node("named", title="Named", description="For a test.", asks=True)
    def named() -> Annotated[str, Param(title="Decision", description="What you decided")]:
        raise Ask("?")

    assert _once(ask)[1].waiting[0].questions[0].title == "Result"
    question = _once(named)[1].waiting[0].questions[0]
    assert (question.title, question.description) == ("Decision", "What you decided")


def test_a_question_whose_type_takes_none_may_be_answered_with_none():
    @node("note", title="Note", description="For a test.", asks=True)
    def note(name: str) -> str | None:
        raise Ask("A note?")

    _, run = _once(note, name="x")
    assert run.waiting[0].questions[0].required is False
    assert run.answer("q", None).resume().outputs == {"result": None}


@dataclass(frozen=True)
class Review:
    approved: bool
    amount: float
    note: Annotated[str | None, Param(widget=Textarea(rows=6))]
    due: datetime.date = datetime.date(2026, 12, 1)


@node("review", title="Review", description="For a test.", asks=True)
def review(name: str) -> Review:
    raise Ask(f"Review {name}", default={"amount": 100.0})


def test_a_frozen_dataclass_is_a_form_of_one_question_per_field():
    _, run = _waiting_outcomes(review)
    approved, amount, note, due = run.waiting[0].questions
    assert [q.name for q in (approved, amount, note, due)] == ["approved", "amount", "note", "due"]
    assert (approved.required, amount.default, amount.required, note.required, note.widget) == (True, 100.0, False, False, {"type": "textarea", "rows": 6})
    assert (due.default, due.required) == ("2026-12-01", False)  # a field's own default is proposed


def test_a_form_answered_in_part_gives_each_field_left_out_its_default_or_none_and_refuses_one_with_neither():
    graph, run = _waiting_outcomes(review)
    answered = run.answer({"q@0": {"approved": True}, "q@1": {"approved": False, "amount": 5, "note": "late", "due": "2027-01-01"}}).resume()
    assert answered.results["q"] == {
        "approved": [True, False], "amount": [100.0, 5.0], "note": [None, "late"], "due": [datetime.date(2026, 12, 1), datetime.date(2027, 1, 1)],
    }
    faults = _refused(lambda: run.answer("q@0", {"amount": "lots", "colour": "red"}))
    assert [(f.ctx.get("question"), f.msg) for f in faults] == [
        ("colour", "That is not a valid answer: there is no question 'colour'"),
        ("approved", "That is not a valid answer: 'approved' was not answered"),
        ("amount", "That is not a valid answer: 'amount': Input should be a valid number"),
    ]
    assert _refused(lambda: run.answer("q@0", True))[0].msg == "That is not a valid answer: answer each question by name: a mapping of approved, amount, note, due"


@dataclass(frozen=True)
class Routed:
    note: str
    escalate: str = SKIPPED


@node("route", title="Route", description="For a test.", asks=True)
def route(text: str) -> Routed:
    raise Ask(f"Route {text}")


def test_a_form_field_whose_default_is_skipped_proposes_nothing_and_left_out_is_a_branch_not_taken():
    _, run = _waiting_outcomes(route)
    note, escalate = run.waiting[0].questions
    assert (note.required, escalate.required, escalate.default) == (True, False, None)
    run.to_json()

    answered = run.answer({"q@0": {"note": "fine"}, "q@1": {"note": "late", "escalate": "boss"}}).resume()

    assert answered.results["q"] == {"note": ["fine", "late"], "escalate": [None, "boss"]}
    assert answered.results["q"] == Run.from_json(answered.to_json(), run.graph).results["q"]


class Line(TypedDict):
    sku: str
    qty: int


@node("lines", title="Lines", description="For a test.", asks=True)
def lines(order: str) -> Table[Line]:
    raise Ask(f"Lines of {order}?", default=pl.DataFrame({"sku": ["A"], "qty": [1]}))


@node("each", title="Each", description="For a test.")
def each(sku: str, qty: int) -> str:
    return sku * qty


@node("line-items", title="Line items", description="For a test.")
def line_items(table: Table[Line]) -> Series[Line]:
    return table.to_dicts()


def _lines_graph() -> Any:
    """``lines`` asking for a table as node ``q``, split into one record per line, each read by ``each``."""
    return Graph.compile(GraphRecord.model_validate({"nodes": {"q": {"node": "lines", "inputs": {"order": {"constant": "O-1"}}}, "i": {"node": "line-items", "inputs": {"table": {"sources": [{"node": "q", "output": "result"}]}}}, "e": {"node": "each", "inputs": {"sku": {"sources": [{"node": "i", "output": "result", "field": "sku"}]}, "qty": {"sources": [{"node": "i", "output": "result", "field": "qty"}]}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), registry_of(lines, line_items, each))


def test_a_table_is_answered_as_records_which_are_its_rows_and_keep_through_json():
    graph = _lines_graph()
    run = graph.run()
    (question,) = run.waiting[0].questions
    assert not question.is_series and question.type.startswith("Table") and [c["properties"]["name"]["const"] for c in question.json_schema["properties"]["columns"]["prefixItems"]] == ["sku", "qty"]
    assert question.default == {"columns": [{"name": "sku", "type": "str", "values": ["A"]}, {"name": "qty", "type": "int", "values": [1]}]}  # a table as conductor writes one
    answered = Run.from_json(Run.from_json(run.to_json(), graph).answer("q", [{"sku": "B", "qty": 2}, {"sku": "C", "qty": 3}]).to_json(), graph)
    assert answered.resume().outputs == {"result": ["BB", "CCC"]}


def test_a_table_default_survives_saving_and_answers_like_any_value():
    graph = _lines_graph()
    run = Run.from_json(graph.run().to_json(), graph)
    assert run.answer("q", run.waiting[0].questions[0].default).resume().outputs == {"result": ["A"]}


def test_an_empty_list_answers_a_table_with_no_rows_and_a_wrong_record_is_refused():
    graph = _lines_graph()
    run = graph.run()
    done = run.answer("q", []).resume()
    assert done.status == "complete" and done.results["q"]["result"].columns == ["sku", "qty"] and done.results["e"]["result"] == []
    assert [f.msg for f in _refused(lambda: run.answer("q", [{"sku": "B"}]))] == [
        "That is not a valid answer: the column 'qty' its row type declares is missing",
    ]
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": "two"}]))[0].msg.startswith("That is not a valid answer: column 'qty':")
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": "2"}]))[0].msg.startswith("That is not a valid answer: column 'qty':")  # read as a one-value answer is: strictly
    assert _refused(lambda: run.answer("q", "B,2"))[0].msg == "That is not a valid answer: a table is a frame or a list of records, one object per row"


def test_a_table_answer_is_read_as_a_callers_records_are_with_its_cells_read_strictly():
    graph = _lines_graph()
    run = graph.run()
    extra = run.answer("q", [{"sku": "B", "qty": 2, "colour": "red"}]).resume()
    assert extra.results["q"]["result"].columns == ["sku", "qty"] and extra.outputs == {"result": ["BB"]}
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": True}, {"sku": "C", "qty": 1}]))[0].msg == (
        "That is not a valid answer: column 'qty': Input should be a valid integer"
    )


def test_a_hooks_outputs_are_asked_by_their_types():
    def outputs(names: str) -> dict[str, Any]:
        return {name: bool for name in names.split(",")}

    @node("checks", title="Checks", description="For a test.", compute_outputs=outputs, asks=True)
    def checks(names: str) -> dict[str, Any]:
        raise Ask("Check each")

    outputs_declared = {name: {"node": "q", "output": name} for name in ("signed", "paid")}  # a hook's outputs are given by name
    record = GraphRecord.model_validate({"nodes": {"q": {"node": "checks", "inputs": {"names": {"constant": "signed,paid"}}}}, "outputs": outputs_declared})
    run = Graph.compile(record, registry_of(checks)).run()
    assert [(q.name, q.json_schema) for q in run.waiting[0].questions] == [("signed", {"type": "boolean"}), ("paid", {"type": "boolean"})]
    assert run.answer("q", {"signed": True, "paid": False}).resume().outputs == {"signed": True, "paid": False}


@dataclass(frozen=True)
class Route:
    keep: Annotated[str, Param(exclusive_group="route")]
    drop: Annotated[str, Param(exclusive_group="route")]


def test_answering_one_branch_of_an_exclusive_group_leaves_the_other_not_taken():
    @node("route", title="Route", description="For a test.", asks=True)
    def route(text: str) -> Route:
        raise Ask(f"Keep {text}?")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"q": {"node": "route", "inputs": {"text": {"constant": "a"}}}, "k": {"node": "upper", "inputs": {"text": {"sources": [{"node": "q", "output": "keep"}]}}}, "d": {"node": "upper", "inputs": {"text": {"sources": [{"node": "q", "output": "drop"}]}}}}, "outputs": {"d_result": {"node": "d", "output": "result"}, "k_result": {"node": "k", "output": "result"}}}), registry_of(route, upper))
    run = graph.run()
    assert [(q.exclusive_group, q.required) for q in run.waiting[0].questions] == [("route", False), ("route", False)]
    done = run.answer("q", {"keep": "a"}).resume()
    assert done.status == "complete" and done.results["k"]["result"] == "A" and done.results["d"]["result"] is None
    assert _refused(lambda: run.answer("q", {"keep": "a", "drop": "a"}))[0].msg == "That is not a valid answer: answer exactly one of keep, drop"
    assert _refused(lambda: run.answer("q", {}))[0].msg == "That is not a valid answer: answer exactly one of keep, drop"


# -- picking several -----------------------------------------------------------------------------


@node("colours", title="Colours", description="For a test.", asks=True)
def colours(name: str) -> Series[Literal["red", "green", "blue"]]:
    raise Ask(f"Colours for {name}?", default=["red"])


def test_a_series_of_a_literal_is_a_pick_of_several_run_end_to_end_through_json():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"constant": "Ida,Bo"}}}, "q": {"node": "colours", "inputs": {"name": {"sources": [{"node": "s", "output": "result"}]}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), registry_of(split, colours, upper, join))
    run = Run.from_json(graph.run().to_json(), graph)
    (question,) = run.waiting[0].questions
    assert (question.is_series, question.default, [c["id"] for c in question.choices]) == (True, ["red"], ["red", "green", "blue"])
    faults = _refused(lambda: run.answer({"q@0": ["red", "red"], "q@1": ["pink"]}))
    assert [f.msg for f in faults] == ["That is not a valid answer: a choice is picked twice", "That is not a valid answer: Input should be 'red', 'green' or 'blue'"]
    answered = Run.from_json(run.answer({"q@0": ["red", "blue"], "q@1": []}).to_json(), graph)
    assert answered.results["u"]["result"] == [[None, None], []]  # the picks are items, a list for each name; nothing read them yet
    done = answered.resume()
    assert done.status == "complete" and done.results["q"]["result"] == [["red", "blue"], []] and done.results["j"]["result"] == ["RED+BLUE", ""]


def test_a_series_with_choices_the_run_knows_is_a_pick_of_several_of_those():
    @node("people", title="People", description="For a test.", asks=True)
    def people(team: str) -> Series[str]:
        raise Ask(f"Who in {team}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K."), Choice("carl", "Carl")])

    _, run = _once(people, team="sales")
    (question,) = run.waiting[0].questions
    assert (question.is_series, [c["title"] for c in question.choices]) == (True, ["Ida B.", "Bo K.", "Carl"])
    assert _refused(lambda: run.answer("q", ["ida", "eve"]))[0].msg == "That is not a valid answer: the answer is not one of the choices"
    assert run.answer("q", ["carl", "ida"]).resume().outputs == {"result": ["carl", "ida"]}


def test_a_node_per_item_of_a_list_answered_empty_shows_an_empty_list_before_and_after_resume():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"q": {"node": "colours", "inputs": {"name": {"constant": "x"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry_of(colours, upper))
    answered = graph.run().answer("q", [])
    assert answered.results["u"]["result"] == []
    assert answered.resume().outputs == {"result": []}


# -- asking wrongly fails the node --------------------------------------------------------------


@pytest.mark.parametrize(("ask", "reason"), [
    (lambda: Ask("?", default=3), "the default for 'result' is not an answer it takes: Input should be a valid string"),
    (lambda: Ask("?", default=SKIPPED), "the default for 'result' is not an answer it takes: the value has no JSON form"),
    (lambda: Ask("?", choices=["a", 2]), "a choice for 'result' is not what it returns: Input should be a valid string"),
    (lambda: Ask("?", choices=["a"], default="b"), "the default for 'result' is not an answer it takes: the answer is not one of the choices"),
    (lambda: Ask("?", choices=["a", object()]), "a choice for 'result' is not what it returns: the value has no JSON form"),
    (lambda: Ask("?", choices=[Choice(object(), "Thing")]), "a choice for 'result' is not what it returns: the value has no JSON form"),
    (lambda: Ask("?", choices=["a", "a"]), "a choice for 'result' is offered twice"),
    (lambda: Ask("?", choices=[]), "'result' is offered no choices"),
    (lambda: Ask("?", choices={"result": ["a"]}), "the node asks one question, so it is answered with a value, and choices= is a list"),
])
def test_a_node_asking_what_its_output_cant_be_answered_with_fails_naming_why(ask, reason):
    @node("wrong", title="Wrong", description="For a test.", asks=True)
    def wrong() -> str:
        raise ask()

    _, run = _once(wrong)
    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["reason"]) == ("failed", "invalid_ask", ("q",), reason)


@pytest.mark.parametrize("stream", [False, True])
def test_a_question_on_inputs_with_no_json_form_fails_its_node_naming_the_input(stream):
    @node("make", title="Make", description="For a test.")
    def make(text: str) -> Any:
        return object()

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(label: str, value: Any) -> bool:
        raise Ask("ok?")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "make", "inputs": {"text": {"graph_input": "text"}}}, "a": {"node": "approve", "inputs": {"label": {"constant": "x"}, "value": {"sources": [{"node": "m", "output": "result"}]}}}}, "outputs": {"result": {"node": "a", "output": "result"}}}), registry_of(make, approve))

    async def streamed() -> Any:
        return [e async for e in graph.with_inputs(text="x").stream()][-1].run

    import asyncio

    run = asyncio.run(streamed()) if stream else graph.run(text="x")
    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["input"]) == ("failed", "question_unwritable", ("a",), "value")


@pytest.mark.parametrize(("ask", "reason"), [
    (lambda: Ask("?", default="a"), "the node has 2 outputs, so it is answered by output name, and default= goes by output name too"),
    (lambda: Ask("?", default={"stay": "a"}), "the default names 'stay', which is not an output; the outputs are keep, drop"),
    (lambda: Ask("?", choices=["a"]), "the node has 2 outputs, so it is answered by output name, and choices= goes by output name too"),
])
def test_a_node_of_several_outputs_asks_by_output_name(ask, reason):
    @node("wrong", title="Wrong", description="For a test.", asks=True)
    def wrong() -> Route:
        raise ask()

    assert _once(wrong)[1].state.cause.ctx["reason"] == reason


def test_a_default_obeys_what_an_answer_must():
    @node("blank", title="Blank", description="For a test.", asks=True)
    def blank() -> str | None:
        raise Ask("?", choices=["a", "b"], default=None)

    @node("twice", title="Twice", description="For a test.", asks=True)
    def twice() -> Series[str]:
        raise Ask("?", choices=["t1", "t2"], default=["t1", "t1"])

    assert _once(blank)[1].state.cause.ctx["reason"] == "the default for 'result' is not an answer it takes: the answer is not one of the choices"
    assert _once(twice)[1].state.cause.ctx["reason"] == "the default for 'result' is not an answer it takes: a choice is picked twice"


def test_a_question_whose_choices_limit_it_must_be_answered_even_if_its_type_takes_none():
    @dataclass(frozen=True)
    class Pick:
        who: str | None
        note: str | None

    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> Pick:
        raise Ask("?", choices={"who": ["a"]})

    _, run = _once(pick)
    assert [q.required for q in run.waiting[0].questions] == [True, False]
    assert _refused(lambda: run.answer("q", {}))[0].msg == "That is not a valid answer: 'who' was not answered"
    assert run.answer("q", {"who": "a"}).resume().outputs == {"who": "a", "note": None}


@pytest.mark.parametrize(("returns", "wrong", "right"), [
    (bool, ["yes", 1, "true"], True),
    (int, ["3", True, 3.0 + 0.5], 3),
    (float, ["nan", True, "1.5"], 1.5),
    (datetime.date, [20261110, "10/11/2026"], "2026-11-10"),
])
def test_an_answer_is_read_strictly_from_its_json_form(returns, wrong, right):
    _, run = _once(node("strict", title="Asks", description="For a test.", asks=True)(_returning(returns)))
    for value in wrong:
        assert [f.type for f in _refused(lambda value=value: run.answer("q", value))] == ["invalid_answer"], value
    assert run.answer("q", right).resume().status == "complete"


def _returning(returns: Any) -> Any:
    def asks() -> None:
        raise Ask("?")

    asks.__annotations__ = {"return": returns}
    return asks


def test_a_choice_offered_for_a_table_is_a_table_like_any_choice_of_its_type():
    @node("wrong", title="Wrong", description="For a test.", asks=True)
    def wrong() -> Table[Line]:
        raise Ask("?", choices=[1])

    assert _once(wrong)[1].state.cause.ctx["reason"] == "a choice for 'result' is not what it returns: a table is a frame or a list of records, one object per row"


# -- answering ----------------------------------------------------------------------------------


def _counted() -> tuple[Any, list[str]]:
    called: list[str] = []

    @node("gate", title="Gate", description="For a test.", asks=True)
    def gate(text: str) -> str:
        called.append(text)
        if text.startswith("b"):
            raise Ask(f"{text}?")
        return text

    return gate, called


def test_answer_runs_nothing_and_leaves_the_run_it_was_given_as_it_was():
    gate, called = _counted()
    _, run = _waiting_outcomes(gate, text="a,b1,b2")
    before, calls = run.to_json(), list(called)
    answered = run.answer("q@1", "B1")
    assert called == calls and run.to_json() == before and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    assert [place(w) for w in answered.waiting] == ["q@2"]


def test_every_fault_of_one_answer_call_is_refused_at_once_and_nothing_is_answered():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="a,b1,b2")
    faults = _refused(lambda: run.answer({"q@1": 1, "q@0": "A", "nope@3": "x", "q@2": "B2"}))
    assert [(f.type, f.loc, f.ctx["waiting_id"]) for f in faults] == [
        ("invalid_answer", ("q", 1), "q@1"), ("not_waiting", ("q", 0), "q@0"), ("not_waiting", ("nope", 3), "nope@3"),
    ]
    assert [place(w) for w in run.waiting] == ["q@1", "q@2"]


def test_answering_with_ids_that_are_no_waiting_id_is_refused_as_problems():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    faults = _refused(lambda: run.answer({0: "B1", None: "B2", "q@x": "B", "q": "B", "q@0@1": "B"}))  # type: ignore[dict-item]
    assert [(f.type, f.loc) for f in faults] == [
        ("not_waiting", ("0",)), ("not_waiting", ("None",)), ("not_waiting", ("q",)), ("not_waiting", ("q",)), ("not_waiting", ("q", 0, 1)),
    ]


def test_answer_takes_one_waiting_call_and_its_value_or_a_mapping_of_several():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    assert run.answer(run.waiting[0], "1").answer("q@1", "2").to_json() == run.answer({"q@0": "1", "q@1": "2"}).to_json()
    with pytest.raises(TypeError):
        run.answer("q@0")  # type: ignore[call-overload]


def test_an_answer_is_given_once():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b")
    answered = run.answer("q@0", "B")
    assert [f.type for f in _refused(lambda: answered.answer("q@0", "B"))] == ["not_waiting"]


def test_a_value_no_question_can_read_is_refused_as_a_problem_never_an_exception_of_its_own():
    _, run = _waiting_outcomes(review)
    for value in (object(), {"approved": object()}, {"approved": True, "amount": float("nan"), "due": object()}):
        assert {f.type for f in _refused(lambda value=value: run.answer("q@0", value))} == {"invalid_answer"}
    graph = _lines_graph()
    for value in ([object()], [{"sku": object(), "qty": 1}], pl.DataFrame({"sku": [1]})):
        assert [f.type for f in _refused(lambda value=value: graph.run().answer("q", value))] == ["invalid_answer"]


def test_after_an_answer_resume_calls_only_what_waited_on_it_and_an_item_left_unanswered_waits_on():
    gate, called = _counted()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"constant": "a,b1,b2"}}}, "q": {"node": "gate", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry_of(split, gate, upper))
    run = graph.run()
    called.clear()
    resumed = Run.from_json(run.answer("q@1", "yes").to_json(), graph).resume()
    assert called == [] and resumed.status == "waiting" and [place(w) for w in resumed.waiting] == ["q@2"]
    assert resumed.results["u"]["result"] == ["A", "YES", None]
    assert resumed.waiting[0] == run.waiting[1]  # the question is as it was asked


def test_a_node_that_reads_the_world_is_not_called_again_when_a_run_goes_on_from_waiting():
    read: list[str] = []

    @node("now", title="Now", description="For a test.", policy=Policy(reuse_when_inputs_match=False))
    def now(text: str) -> str:
        read.append(text)
        return text

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(text: str) -> str:
        raise Ask("?")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "now", "inputs": {"text": {"constant": "x"}}}, "q": {"node": "ask", "inputs": {"text": {"sources": [{"node": "n", "output": "result"}]}}}}, "outputs": {"result": {"node": "q", "output": "result"}}}), registry_of(now, ask))
    graph.run().answer("q", "y").resume()
    assert read == ["x"]


def test_an_item_skipped_above_never_asks_and_the_others_complete_when_answered():
    @node("only-b", title="Only b", description="For a test.")
    def only_b(text: str) -> str:
        return text if text.startswith("b") else SKIPPED

    gate, called = _counted()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"constant": "a,b1,b2"}}}, "o": {"node": "only-b", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "q": {"node": "gate", "inputs": {"text": {"sources": [{"node": "o", "output": "result"}]}}}}, "outputs": {"result": {"node": "q", "output": "result"}}}), registry_of(split, only_b, gate))
    run = graph.run()
    assert called == ["b1", "b2"] and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    assert run.answer({"q@1": "1", "q@2": "2"}).resume().results["q"]["result"] == [None, "1", "2"]


def test_waiting_calls_are_listed_in_node_then_item_order_whatever_order_they_asked_in():
    @node("slow-ask", title="Slow ask", description="For a test.", policy=Policy(concurrency=4), asks=True)
    def slow_ask(text: str) -> str:
        time.sleep(0.02 * (3 - int(text)))
        raise Ask(text)

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(text: str) -> str:
        raise Ask(text)

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"constant": "0,1,2"}}}, "a": {"node": "slow-ask", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "b": {"node": "ask", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}}}), registry_of(split, slow_ask, ask))
    run = graph.run()
    assert [place(w) for w in run.waiting] == ["a@0", "a@1", "a@2", "b@0", "b@1", "b@2"]
    assert [place(w) for w in Run.from_json(run.to_json(), graph).waiting] == [place(w) for w in run.waiting]


def test_a_waiting_call_keeps_the_table_and_series_it_received_through_a_save():
    class Row(TypedDict):
        name: str

    @node("rows", title="Rows", description="For a test.")
    def rows() -> Table[Row]:
        return pl.DataFrame({"name": ["Ida", "Bo"]})

    @node("records", title="Records", description="For a test.")
    def records(table: Table[Row]) -> Series[Row]:
        return table.to_dicts()

    @node("look", title="Look", description="For a test.", asks=True)
    def look(names: Series[str], table: Table[Row]) -> bool:
        raise Ask("Right?")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "rows"}, "i": {"node": "records", "inputs": {"table": {"sources": [{"node": "r", "output": "result"}]}}}, "q": {"node": "look", "inputs": {"names": {"sources": [{"node": "i", "output": "result", "field": "name"}]}, "table": {"sources": [{"node": "r", "output": "result"}]}}}}, "outputs": {"result": {"node": "q", "output": "result"}}}), registry_of(rows, records, look))
    run = graph.run()
    assert run.waiting[0].inputs == {"names": ["Ida", "Bo"], "table": {"columns": [{"name": "name", "type": "str", "values": ["Ida", "Bo"]}]}}
    assert Run.from_json(run.to_json(), graph).waiting == run.waiting


def test_items_made_of_what_a_caller_gives_pause_save_answer_and_finish():
    gate, _ = _counted()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}}, "q": {"node": "gate", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "q", "output": "result"}}}), registry_of(split, gate))
    run = Run.from_json(graph.run(text="a,b").to_json(), graph)
    assert [place(w) for w in run.waiting] == ["q@1"]
    assert Run.from_json(run.answer("q@1", "B").to_json(), graph).resume().outputs == {"result": ["a", "B"]}


def test_an_answer_names_its_item_by_index_so_one_for_an_item_that_no_longer_waits_is_refused():
    gate, _ = _counted()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}}, "q": {"node": "gate", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "q", "output": "result"}}}), registry_of(split, gate))
    run = graph.run(text="b1,a")
    moved = run.again(text="a,b1")  # the same question, now on item 1
    assert [place(w) for w in run.waiting] == ["q@0"] and [place(w) for w in moved.waiting] == ["q@1"]
    assert [f.type for f in _refused(lambda: moved.answer("q@0", "B1"))] == ["not_waiting"]


def test_a_saved_waiting_call_the_graph_has_no_call_for_is_refused_on_load():
    gate, _ = _counted()
    graph, run = _waiting_outcomes(gate, text="a,b")
    saved = json.loads(run.to_json())
    asked = saved["outcomes"]["q"]
    asked["states"].append(None)
    asked["outputs"]["result"].append(None)
    asked["questions"]["2"] = asked["questions"]["1"]
    (fault,) = _refused(lambda: Run.from_json(json.dumps(saved), graph))
    assert (fault.type, fault.ctx["where"]) == ("saved_run_unreadable", "outcomes.q")


def test_a_saved_waiting_call_with_its_question_taken_out_reads_as_not_yet_made():
    """A saved call waits because its question is saved with it, so with the question taken out the call reads as not
    yet made."""
    gate, _ = _counted()
    graph, run = _waiting_outcomes(gate, text="a,b")
    saved = json.loads(run.to_json())
    assert saved["outcomes"]["q"]["states"] == ["done", None] and list(saved["outcomes"]["q"]["questions"]) == ["1"]
    del saved["outcomes"]["q"]["questions"]
    loaded = Run.from_json(json.dumps(saved), graph)
    assert loaded.waiting == []
    assert [place(w) for w in loaded.resume().waiting] == ["q@1"]  # the node is called for it, and asks again


def test_a_question_record_is_json_a_host_draws_from_with_no_graph_loaded():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> Annotated[str, Param(title="Assignee", widget=Textarea())]:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B.")], default="ida")

    _, run = _once(assign, ticket="T-1")
    written = run.waiting[0].model_dump(mode="json")
    assert re.fullmatch(r"q~[0-9a-f]{8}", written.pop("id"))  # the place, and a fingerprint of the question asked there
    assert written == {
        "loc": ["q"], "prompt": "Who takes T-1?", "inputs": {"ticket": "T-1"},
        "questions": [{
            "name": "result", "title": "Assignee", "is_series": False, "required": False, "type": "str", "type_variables": [], "schema": {"type": "string"}, "loops": [],
            "choices": [{"id": "ida", "title": "Ida B."}], "default": "ida", "widget": {"type": "textarea", "rows": 4},
            "description": None, "exclusive_group": None,
        }],
    }


def test_a_literal_says_the_type_of_its_values_on_an_input_and_on_a_question():
    @node("priority", title="Priority", description="For a test.", asks=True)
    def priority(level: Literal["a", "b"], ticket: str) -> Literal["low", "high"]:
        raise Ask(f"How urgent is {ticket}?")

    assert [field.type for field in priority.describe().inputs] == ["str", "str"]
    run = Graph.compile(make_graph(priority, constants={"priority.level": "a"}), registry_of(priority)).run(ticket="T-1")
    (question,) = run.waiting[0].questions
    assert question.type == "str" and question.json_schema == {"enum": ["low", "high"], "type": "string"}


def test_one_item_answered_twice_in_one_call_is_refused_and_nothing_is_answered():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    faults = _refused(lambda: run.answer({"q@0": "one", run.waiting[0].id: "two"}))
    assert [(f.type, f.loc) for f in faults] == [("invalid_answer", ("q", 0))]
    assert [place(w) for w in run.waiting] == ["q@0", "q@1"]


def test_an_id_whose_item_index_is_not_a_number_answers_nothing_not_even_a_node_called_once():
    @node("once", title="Once", description="For a test.", asks=True)
    def once(text: str) -> str:
        raise Ask("ok?")

    run = Graph.compile(make_graph(once), registry_of(once)).run(text="x")
    assert [(f.type, f.loc) for f in _refused(lambda: run.answer("once@banana", "y"))] == [("not_waiting", ("once",))]
    assert run.answer("once", "y").status == "waiting"  # answered, not yet resumed


async def test_a_cancelled_run_takes_an_answer_to_what_waits_in_it_as_every_run_that_ended_early_does():
    from asyncio import Event

    gate, called = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    cancel = Event()
    cancel.set()
    async for event in run.stream(cancel=cancel):
        if event.type == "run_end":
            cancelled = event.run
    assert cancelled.status == "cancelled" and [place(w) for w in cancelled.waiting] == ["q@0", "q@1"]

    answered = cancelled.answer("q@0", "B1")
    assert answered.status == "cancelled" and answered.state.cause.type == "run_cancelled"  # answered, not yet resumed
    resumed = answered.resume()
    assert resumed.status == "waiting" and [place(w) for w in resumed.waiting] == ["q@1"]


def test_a_question_inside_a_subgraph_is_the_inner_nodes_own_and_its_waiting_call_says_where_it_is():
    from conductor import GraphRecord, NodeRegistry, make_graph, node

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(text: str) -> str:
        raise Ask(f"{text}?")

    @node("ask2", title="Ask2", description="For a test.", asks=True)
    def ask2(text: str) -> str:
        raise Ask(f"again {text}?")

    registry = NodeRegistry()
    registry.add(ask, ask2, subgraph("both", title="Both", description="For a test.")(GraphRecord.model_validate({
        "nodes": {"q": {"node": "ask", "inputs": {"text": {"graph_input": "first"}}}, "q2": {"node": "ask2", "inputs": {"text": {"graph_input": "second"}}}},
        "outputs": {"first": {"node": "q", "output": "result"}, "second": {"node": "q2", "output": "result"}},
    })))
    run = Graph.compile(make_graph(registry["both"], constants={"both.first": "a", "both.second": "b"}), registry).run()
    assert [(w.id.partition("~")[0], w.loc, [q.name for q in w.questions]) for w in run.waiting] == [("both/q", ("both", "q"), ["result"]), ("both/q2", ("both", "q2"), ["result"])]
    done = run.answer({w.id: "yes" for w in run.waiting}).resume()
    assert done.outputs == {"first": "yes", "second": "yes"}


def test_a_form_inside_a_subgraph_is_answered_by_the_inner_nodes_own_field_names():
    from dataclasses import dataclass

    from conductor import GraphRecord, NodeRegistry, node

    @dataclass(frozen=True)
    class Two:
        a: str
        b: int = 0

    @node("form", title="Form", description="For a test.", asks=True)
    def form(t: str) -> Two:
        raise Ask("two?")

    registry = NodeRegistry()
    registry.add(form, subgraph("both", title="Both", description="For a test.")(GraphRecord.model_validate({"nodes": {"z1": {"node": "form", "inputs": {"t": {"graph_input": "z1_t"}}}, "z2": {"node": "form", "inputs": {"t": {"graph_input": "z2_t"}}}}, "outputs": {"z1_a": {"node": "z1", "output": "a"}, "z1_b": {"node": "z1", "output": "b"}, "z2_a": {"node": "z2", "output": "a"}, "z2_b": {"node": "z2", "output": "b"}}})))
    run = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "both", "inputs": {"z1_t": {"graph_input": "z1_t"}, "z2_t": {"graph_input": "z2_t"}}}}, "outputs": {"z1_a": {"node": "e", "output": "z1_a"}, "z1_b": {"node": "e", "output": "z1_b"}, "z2_a": {"node": "e", "output": "z2_a"}, "z2_b": {"node": "e", "output": "z2_b"}}}), registry).run(z1_t="1", z2_t="2")
    answers = {w.id: {q.name: "A" if q.name == "a" else 5 for q in w.questions} for w in run.waiting}
    assert {w.loc: sorted(answers[w.id]) for w in run.waiting} == {("e", "z1"): ["a", "b"], ("e", "z2"): ["a", "b"]}
    assert run.answer(answers).resume().results == {"e": {"z1_a": "A", "z1_b": 5, "z2_a": "A", "z2_b": 5}}
    (wrong,) = _refused(lambda: run.answer({run.waiting[0].id: {"a": 3}}))
    assert (wrong.type, wrong.loc, wrong.ctx["question"]) == ("invalid_answer", ("e", "z1"), "a")


def test_an_answer_to_a_run_with_nothing_waiting_says_how_the_run_stands():
    @node("plain", title="Plain", description="For a test.")
    def plain(text: str) -> str:
        return text

    run = Graph.compile(make_graph(plain), registry_of(plain)).run(text="a")
    (problem,) = _refused(lambda: run.answer("plain", "b"))
    assert (problem.type, problem.ctx["status"]) == ("not_waiting", "complete") and "(the run is complete)" in problem.msg


@pytest.mark.parametrize("choices", [[True, 2], ["1"]], ids=["a bool", "a number as text"])
def test_a_choice_is_read_as_its_answer_would_be_so_one_a_number_refuses_fails_the_node(choices: list[object]) -> None:
    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> int:
        raise Ask("?", choices=choices)

    _, run = _once(pick)
    assert (run.status, run.state.cause.type) == ("failed", "invalid_ask")
    assert run.state.cause.ctx["reason"] == "a choice for 'result' is not what it returns: Input should be a valid integer"


def test_a_model_that_isnt_a_choice_is_a_value_to_param_and_to_ask_alike() -> None:
    from pydantic import BaseModel

    class Option(BaseModel):  # a host's own option record, not a Choice
        id: str
        title: str

    ida = Option(id="ida", title="Ida B.")
    with pytest.raises(NodeDeclarationError) as declared:

        @node("assign_by_param", title="Assign by param", description="For a test.")
        def assign_by_param(who: Annotated[str, Param(choices=[ida])]) -> str:
            return who

    assert declared.value.code == "choices_invalid"

    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign() -> str:
        raise Ask("Who?", choices=[ida])

    _, run = _once(assign)
    assert (run.status, run.state.cause.type) == ("failed", "invalid_ask")
    assert run.state.cause.ctx["reason"] == "a choice for 'result' is not what it returns: Input should be a valid string"


def test_a_lone_output_is_asked_under_the_label_describe_shows_for_it():
    @node("reason", title="Closing reason", description="Asks why a ticket was closed.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why was it closed?")

    graph = Graph.compile(make_graph(reason), registry_of(reason))
    (output,) = graph.describe().nodes["reason"].outputs
    (waiting,) = graph.run(ticket="t").waiting

    assert waiting.questions[0].title == output.title == "Result"


def test_a_declared_choice_is_answered_by_the_id_its_question_shows() -> None:
    from decimal import Decimal

    @node("amount", title="Amount", description="Asks how much.", asks=True)
    def amount(ticket: str) -> Annotated[Decimal, Param(choices=[1, 2])]:
        raise Ask("How much?")

    @node("when", title="When", description="Asks when.", asks=True)
    def when(ticket: str) -> Annotated[datetime.datetime, Param(choices=["2024-01-02T09:00"])]:
        raise Ask("When?")

    for asking, read in ((amount, Decimal(1)), (when, datetime.datetime(2024, 1, 2, 9))):
        run = Graph.compile(make_graph(asking), registry_of(asking)).run(ticket="t")
        (waiting,) = run.waiting
        shown = waiting.questions[0].choices
        assert shown is not None
        assert run.answer(waiting, shown[0]["id"]).resume().outputs == {"result": read}


def test_an_ask_pickled_as_another_process_raises_it_still_proposes_no_default() -> None:
    import pickle

    @node("who", title="Who", description="Asks who takes a ticket.", asks=True)
    def who(ticket: str) -> str:
        raise pickle.loads(pickle.dumps(Ask("Who takes it?")))  # a node whose body ran in a process of its own

    _, run = _once(who, ticket="T-1")
    assert run.status == "waiting" and run.waiting[0].questions[0].required
