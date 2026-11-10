"""Asking a person: the return type says what the answer is, ``raise Ask`` adds what the run knows.

A waiting row is a ``Waiting`` whose ``questions`` are the outputs' field
records with what the ``Ask`` added; ``run.answer`` checks an answer against
them and the node's types, refuses every fault at once, and runs nothing.
"""

import datetime
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
    InvalidCallError,
    NodeDeclarationError,
    Param,
    Policy,
    Series,
    Table,
    make_graph,
    node,
)
from conductor.widgets import Textarea

from tests.test_core.nodes import place, registry_of


@node("split")
def split(text: str) -> Series[str]:
    return text.split(",") if text else []


@node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("upper")
def upper(text: str) -> str:
    return text.upper()


def _asking(definition: Any, *more: Any, text: str = "a,b") -> Any:
    """``split`` making a row per item of ``text``, each read by ``definition`` as node ``q``; the graph and its first run."""
    graph = registry_of(split, definition, *more).load({
        "nodes": [{"id": "s", "node": "split", "values": {"text": text}}, {"id": "q", "node": definition.id}],
        "edges": [["s.result", f"q.{next(iter(definition.inputs))}"]],
    })
    return graph, graph.run()


def _once(definition: Any, **values: Any) -> Any:
    """``definition`` alone as node ``q``, run once; the graph and its run."""
    graph = registry_of(definition).load({"nodes": [{"id": "q", "node": definition.id, "values": values}]})
    return graph, graph.run()


def _refused(call: Any) -> list[Any]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    return raised.value.errors()


# -- the return type says how to ask ---------------------------------------------------------------


def test_a_bool_is_a_yes_or_no_and_a_word_for_it_is_refused():
    @node("confirm")
    def confirm(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph, run = _once(confirm, name="F-1")
    (waiting,) = run.waiting
    (question,) = waiting.questions
    assert (place(waiting), waiting.prompt, question.json_schema, question.required) == ("q", "Approve F-1?", {"type": "boolean"}, True)
    (fault,) = _refused(lambda: run.answer(waiting, "maybe"))
    assert (fault.type, fault.loc, fault.ctx["waiting_id"]) == ("invalid_answer", ("q",), waiting.id)
    assert run.answer(waiting, True).resume().outputs == {"q.result": True}


def test_a_literal_offers_its_values_and_an_answer_outside_them_is_refused():
    @node("urgency")
    def urgency(name: str) -> Literal["low", "high"]:
        raise Ask("How urgent?")

    _, run = _once(urgency, name="x")
    (question,) = run.waiting[0].questions
    assert question.choices == [{"id": "low", "title": "low"}, {"id": "high", "title": "high"}]
    assert _refused(lambda: run.answer("q", "urgent"))[0].type == "invalid_answer"
    assert run.answer("q", "high").resume().outputs == {"q.result": "high"}


def test_choices_the_run_knows_limit_the_answer_and_a_choice_gives_a_title():
    @node("assign")
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), "bo"])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == [{"id": "ida", "title": "Ida B."}, {"id": "bo", "title": "bo"}]
    (fault,) = _refused(lambda: run.answer("q", "carl"))
    assert fault.msg == "That is not a valid answer: the answer is not one of the choices"
    assert run.answer("q", "bo").resume().outputs == {"q.result": "bo"}


class Person(Choice):
    team: str = ""


def test_a_hosts_own_choice_carries_its_fields_to_the_question():
    @node("assign")
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Person("ida", "Ida B.", team="sales")])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == [{"id": "ida", "title": "Ida B.", "team": "sales"}]
    assert run.answer("q", "ida").resume().outputs == {"q.result": "ida"}


def test_a_number_and_a_date_are_read_from_their_json_form_into_the_outputs_type():
    @node("when")
    def when(name: str) -> datetime.date:
        raise Ask("When?")

    @node("weekday")
    def weekday(day: datetime.date) -> int:
        return day.isoweekday()

    graph = registry_of(when, weekday).load({"nodes": [{"id": "q", "node": "when", "values": {"name": "x"}}, {"id": "w", "node": "weekday"}],
                                              "edges": [["q.result", "w.day"]]})
    run = graph.run()
    assert run.waiting[0].questions[0].json_schema == {"format": "date", "type": "string"}
    done = graph.load_run(run.answer("q", "2026-11-10").to_json()).resume()
    assert done.results["q"]["result"] == datetime.date(2026, 11, 10) and done.outputs == {"w.result": 2}
    assert _refused(lambda: run.answer("q", "next tuesday"))[0].type == "invalid_answer"


def test_a_lone_output_is_titled_by_the_node_unless_its_type_names_it():
    @node("ask", title="Approve the invoice")
    def ask() -> str:
        raise Ask("?")

    @node("named")
    def named() -> Annotated[str, Param(title="Decision", description="What you decided")]:
        raise Ask("?")

    assert _once(ask)[1].waiting[0].questions[0].title == "Approve the invoice"
    question = _once(named)[1].waiting[0].questions[0]
    assert (question.title, question.description) == ("Decision", "What you decided")


def test_a_question_whose_type_takes_a_blank_may_be_answered_with_none():
    @node("note")
    def note(name: str) -> str | None:
        raise Ask("A note?")

    _, run = _once(note, name="x")
    assert run.waiting[0].questions[0].required is False
    assert run.answer("q", None).resume().outputs == {"q.result": None}


@dataclass(frozen=True)
class Review:
    approved: bool
    amount: float
    note: str | None
    due: datetime.date = datetime.date(2026, 12, 1)


@node("review")
def review(name: str) -> Review:
    raise Ask(f"Review {name}", defaults={"amount": 100.0}, widgets={"note": Textarea(rows=6)})


def test_a_frozen_dataclass_is_a_form_of_one_question_per_field():
    _, run = _asking(review)
    approved, amount, note, due = run.waiting[0].questions
    assert [q.name for q in (approved, amount, note, due)] == ["approved", "amount", "note", "due"]
    assert (approved.required, amount.default, amount.required, note.required, note.widget) == (True, 100.0, False, False, {"type": "textarea", "rows": 6})
    assert (due.default, due.required) == ("2026-12-01", False)  # a field's own default is proposed


def test_a_form_left_short_takes_its_defaults_and_blanks_and_refuses_what_has_neither():
    graph, run = _asking(review)
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


@node("route")
def route(text: str) -> Routed:
    raise Ask(f"Route {text}")


def test_a_form_field_whose_default_is_skipped_proposes_nothing_and_left_out_is_a_branch_not_taken():
    _, run = _asking(route)
    note, escalate = run.waiting[0].questions
    assert (note.required, escalate.required, escalate.default) == (True, False, None)
    run.to_json()

    answered = run.answer({"q@0": {"note": "fine"}, "q@1": {"note": "late", "escalate": "boss"}}).resume()

    assert answered.results["q"] == {"note": ["fine", "late"], "escalate": [None, "boss"]}
    assert answered.results["q"] == run.graph.load_run(answered.to_json()).results["q"]


class Line(TypedDict):
    sku: str
    qty: int


@node("lines")
def lines(order: str) -> Table[Line]:
    raise Ask(f"Lines of {order}?", default=pl.DataFrame({"sku": ["A"], "qty": [1]}))


@node("each")
def each(sku: str, qty: int) -> str:
    return sku * qty


def _lines_graph() -> Any:
    return registry_of(lines, each).load({
        "nodes": [{"id": "q", "node": "lines", "values": {"order": "O-1"}}, {"id": "e", "node": "each"}],
        "edges": [["q.result.sku", "e.sku"], ["q.result.qty", "e.qty"]],
    })


def test_a_table_is_answered_as_records_which_are_new_rows_and_keep_through_json():
    graph = _lines_graph()
    run = graph.run()
    (question,) = run.waiting[0].questions
    assert question.shape == "table" and set(question.columns) == {"sku", "qty"}
    assert question.default == [{"sku": "A", "qty": 1}]  # what an answer looks like
    answered = graph.load_run(graph.load_run(run.to_json()).answer("q", [{"sku": "B", "qty": 2}, {"sku": "C", "qty": 3}]).to_json())
    assert answered.resume().outputs == {"e.result": ["BB", "CCC"]}


def test_a_table_default_survives_saving_and_answers_like_any_value():
    graph = _lines_graph()
    run = graph.load_run(graph.run().to_json())
    assert run.answer("q", run.waiting[0].questions[0].default).resume().outputs == {"e.result": ["A"]}


def test_an_empty_list_answers_a_table_with_no_rows_and_a_wrong_record_is_refused():
    graph = _lines_graph()
    run = graph.run()
    done = run.answer("q", []).resume()
    assert done.status == "complete" and done.results["q"]["result"].columns == ["sku", "qty"] and done.results["e"]["result"] == []
    assert [f.msg for f in _refused(lambda: run.answer("q", [{"sku": "B", "qty": 2, "colour": "red"}]))] == [
        "That is not a valid answer: the table has no column 'colour'; it has sku, qty",
    ]
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": "two"}]))[0].msg.startswith("That is not a valid answer: column 'qty':")
    assert _refused(lambda: run.answer("q", "B,2"))[0].msg == "That is not a valid answer: a table is a list of records, one object per row"


def test_a_hooks_outputs_are_asked_by_their_types():
    def outputs(names: str) -> dict[str, Any]:
        return {name: bool for name in names.split(",")}

    @node("checks", compute_outputs=outputs)
    def checks(names: str) -> dict[str, Any]:
        raise Ask("Check each")

    _, run = _once(checks, names="signed,paid")
    assert [(q.name, q.json_schema) for q in run.waiting[0].questions] == [("signed", {"type": "boolean"}), ("paid", {"type": "boolean"})]
    assert run.answer("q", {"signed": True, "paid": False}).resume().outputs == {"q.signed": True, "q.paid": False}


@dataclass(frozen=True)
class Route:
    keep: Annotated[str, Param(exclusive_group="route")]
    drop: Annotated[str, Param(exclusive_group="route")]


def test_answering_one_branch_of_an_exclusive_group_leaves_the_other_not_taken():
    @node("route")
    def route(text: str) -> Route:
        raise Ask(f"Keep {text}?")

    graph = registry_of(route, upper).load({
        "nodes": [{"id": "q", "node": "route", "values": {"text": "a"}}, {"id": "k", "node": "upper"}, {"id": "d", "node": "upper"}],
        "edges": [["q.keep", "k.text"], ["q.drop", "d.text"]],
    })
    run = graph.run()
    assert [(q.exclusive_group, q.required) for q in run.waiting[0].questions] == [("route", False), ("route", False)]
    done = run.answer("q", {"keep": "a"}).resume()
    assert done.status == "complete" and done.results["k"]["result"] == "A" and done.results["d"]["result"] is None
    assert _refused(lambda: run.answer("q", {"keep": "a", "drop": "a"}))[0].msg == "That is not a valid answer: answer exactly one of keep, drop"
    assert _refused(lambda: run.answer("q", {}))[0].msg == "That is not a valid answer: answer exactly one of keep, drop"


# -- picking several -----------------------------------------------------------------------------


@node("colours")
def colours(name: str) -> Series[Literal["red", "green", "blue"]]:
    raise Ask(f"Colours for {name}?", default=["red"])


def test_a_series_of_a_literal_is_a_pick_of_several_run_end_to_end_through_json():
    graph = registry_of(split, colours, upper, join).load({
        "nodes": [{"id": "s", "node": "split", "values": {"text": "Ida,Bo"}}, {"id": "q", "node": "colours"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"}],
        "edges": [["s.result", "q.name"], ["q.result", "u.text"], ["u.result", "j.texts"]],
    })
    run = graph.load_run(graph.run().to_json())
    (question,) = run.waiting[0].questions
    assert (question.shape, question.default, [c["id"] for c in question.choices]) == ("many", ["red"], ["red", "green", "blue"])
    faults = _refused(lambda: run.answer({"q@0": ["red", "red"], "q@1": ["pink"]}))
    assert [f.msg for f in faults] == ["That is not a valid answer: a choice is picked twice", "That is not a valid answer: Input should be 'red', 'green' or 'blue'"]
    answered = graph.load_run(run.answer({"q@0": ["red", "blue"], "q@1": []}).to_json())
    assert answered.results["u"]["result"] == [None, None]  # the picks are rows; nothing read them yet
    done = answered.resume()
    assert done.status == "complete" and done.results["q"]["result"] == [["red", "blue"], []] and done.results["j"]["result"] == ["RED+BLUE", ""]


def test_a_series_with_choices_the_run_knows_is_a_pick_of_several_of_those():
    @node("people")
    def people(team: str) -> Series[str]:
        raise Ask(f"Who in {team}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K."), Choice("carl", "Carl")])

    _, run = _once(people, team="sales")
    (question,) = run.waiting[0].questions
    assert (question.shape, [c["title"] for c in question.choices]) == ("many", ["Ida B.", "Bo K.", "Carl"])
    assert _refused(lambda: run.answer("q", ["ida", "eve"]))[0].msg == "That is not a valid answer: the answer is not one of the choices"
    assert run.answer("q", ["carl", "ida"]).resume().outputs == {"q.result": ["carl", "ida"]}


def test_a_reader_left_with_no_rows_by_an_answer_shows_so_before_the_next_leg():
    graph = registry_of(colours, upper).load({"nodes": [{"id": "q", "node": "colours", "values": {"name": "x"}}, {"id": "u", "node": "upper"}],
                                               "edges": [["q.result", "u.text"]]})
    answered = graph.run().answer("q", [])
    assert answered.results["u"]["result"] == []
    assert answered.resume().outputs == {"u.result": []}


# -- asking wrongly fails the node --------------------------------------------------------------


@pytest.mark.parametrize(("ask", "reason"), [
    (lambda: Ask("?", default=3), "the default for 'result' is not an answer it takes: Input should be a valid string"),
    (lambda: Ask("?", default=SKIPPED), "the default for 'result' is SKIPPED; a person answers a value"),
    (lambda: Ask("?", widget=Textarea), "the widget for 'result' has no JSON form; give an instance, such as Textarea()"),
    (lambda: Ask("?", choices=["a", 2]), "a choice for 'result' is not what it returns: Input should be a valid string"),
    (lambda: Ask("?", choices="ab"), "the choices for 'result' are a list"),
    (lambda: Ask("?", choices=["a"], default="b"), "the default for 'result' is not an answer it takes: the answer is not one of the choices"),
    (lambda: Ask("?", choices=["a", "a"]), "a choice for 'result' is offered twice"),
    (lambda: Ask("?", choices=[]), "'result' is offered no choices"),
    (lambda: Ask("?", choice=["a"]), "Ask has no argument 'choice'; it takes choices, default, widget, defaults, widgets"),  # type: ignore[call-arg]
    (lambda: Ask("?", defaults={"result": "a"}), "the node has one output, so it takes default=, widget= and a list of choices"),
])
def test_a_node_asking_what_its_output_cant_be_answered_with_fails_naming_why(ask, reason):
    @node("wrong")
    def wrong() -> str:
        raise ask()

    _, run = _once(wrong)
    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx["reason"]) == ("failed", "invalid_ask", ("q",), reason)


@pytest.mark.parametrize(("ask", "reason"), [
    (lambda: Ask("?", default={"keep": "a"}), "the node has 2 outputs, so it takes defaults=, widgets= and choices= by output name"),
    (lambda: Ask("?", defaults={"stay": "a"}), "the default names 'stay', which is not an output; the outputs are keep, drop"),
    (lambda: Ask("?", choices=["a"]), "the node has 2 outputs, so it takes defaults=, widgets= and choices= by output name"),
])
def test_a_node_of_several_outputs_asks_by_output_name(ask, reason):
    @node("wrong")
    def wrong() -> Route:
        raise ask()

    assert _once(wrong)[1].cause.ctx["reason"] == reason


def test_a_prompt_is_text():
    @node("wrong")
    def wrong() -> str:
        raise Ask(3)  # type: ignore[arg-type]

    assert _once(wrong)[1].cause.ctx["error"] == "TypeError"


def test_a_default_obeys_what_an_answer_must():
    @node("blank")
    def blank() -> str | None:
        raise Ask("?", choices=["a", "b"], default=None)

    @node("twice")
    def twice() -> Series[str]:
        raise Ask("?", choices=["t1", "t2"], default=["t1", "t1"])

    assert _once(blank)[1].cause.ctx["reason"] == "the default for 'result' is not an answer it takes: the answer is not one of the choices"
    assert _once(twice)[1].cause.ctx["reason"] == "the default for 'result' is not an answer it takes: a choice is picked twice"


def test_a_question_whose_choices_limit_it_must_be_answered_even_if_its_type_takes_a_blank():
    @dataclass(frozen=True)
    class Pick:
        who: str | None
        note: str | None

    @node("pick")
    def pick() -> Pick:
        raise Ask("?", choices={"who": ["a"]})

    _, run = _once(pick)
    assert [q.required for q in run.waiting[0].questions] == [True, False]
    assert _refused(lambda: run.answer("q", {}))[0].msg == "That is not a valid answer: 'who' was not answered"
    assert run.answer("q", {"who": "a"}).resume().outputs == {"q.who": "a", "q.note": None}


@pytest.mark.parametrize(("returns", "wrong", "right"), [
    (bool, ["yes", 1, "true"], True),
    (int, ["3", True, 3.0 + 0.5], 3),
    (float, ["nan", True, "1.5"], 1.5),
    (datetime.date, [20261110, "10/11/2026"], "2026-11-10"),
])
def test_an_answer_is_read_strictly_from_its_json_form(returns, wrong, right):
    _, run = _once(node("strict")(_returning(returns)))
    for value in wrong:
        assert [f.type for f in _refused(lambda value=value: run.answer("q", value))] == ["invalid_answer"], value
    assert run.answer("q", right).resume().status == "complete"


def _returning(returns: Any) -> Any:
    def asks() -> None:
        raise Ask("?")

    asks.__annotations__ = {"return": returns}
    return asks


def test_a_table_takes_no_choices():
    @node("wrong")
    def wrong() -> Table[Line]:
        raise Ask("?", choices=[1])

    assert _once(wrong)[1].cause.ctx["reason"] == "'result' is a table, which takes no choices"


def test_a_widget_with_no_json_form_is_refused_on_param_too():
    with pytest.raises(NodeDeclarationError) as raised:
        Param(widget=Textarea)
    assert raised.value.code == "widget_not_json"


# -- answering ----------------------------------------------------------------------------------


def _counted() -> tuple[Any, list[str]]:
    called: list[str] = []

    @node("gate")
    def gate(text: str) -> str:
        called.append(text)
        if text.startswith("b"):
            raise Ask(f"{text}?")
        return text

    return gate, called


def test_answer_runs_nothing_and_leaves_the_run_it_was_given_as_it_was():
    gate, called = _counted()
    _, run = _asking(gate, text="a,b1,b2")
    before, calls = run.to_json(), list(called)
    answered = run.answer("q@1", "B1")
    assert called == calls and run.to_json() == before and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    assert [place(w) for w in answered.waiting] == ["q@2"]


def test_every_fault_of_one_answer_call_is_refused_at_once_and_nothing_is_answered():
    gate, _ = _counted()
    _, run = _asking(gate, text="a,b1,b2")
    faults = _refused(lambda: run.answer({"q@1": 1, "q@0": "A", "nope@3": "x", "q@2": "B2"}))
    assert [(f.type, f.loc, f.ctx["waiting_id"]) for f in faults] == [
        ("invalid_answer", ("q", 1), "q@1"), ("not_waiting", ("q", 0), "q@0"), ("not_waiting", ("nope", 3), "nope@3"),
    ]
    assert [place(w) for w in run.waiting] == ["q@1", "q@2"]


def test_answering_by_row_with_ids_that_are_no_waiting_id_is_refused_as_problems():
    gate, _ = _counted()
    _, run = _asking(gate, text="b1,b2")
    faults = _refused(lambda: run.answer({0: "B1", None: "B2", "q@x": "B", "q": "B", "q@0@1": "B"}))  # type: ignore[dict-item]
    assert [(f.type, f.loc) for f in faults] == [
        ("not_waiting", ("0",)), ("not_waiting", ("None",)), ("not_waiting", ("q",)), ("not_waiting", ("q",)), ("not_waiting", ("q", 0, 1)),
    ]


def test_one_answer_is_a_unit_and_its_value_and_several_a_mapping():
    gate, _ = _counted()
    _, run = _asking(gate, text="b1,b2")
    assert run.answer(run.waiting[0], "1").answer("q@1", "2").to_json() == run.answer({"q@0": "1", "q@1": "2"}).to_json()
    with pytest.raises(TypeError):
        run.answer("q@0")  # type: ignore[call-overload]


def test_an_answer_is_given_once():
    gate, _ = _counted()
    _, run = _asking(gate, text="b")
    answered = run.answer("q@0", "B")
    assert [f.type for f in _refused(lambda: answered.answer("q@0", "B"))] == ["not_waiting"]


def test_a_value_no_question_can_read_is_refused_as_a_problem_never_an_exception_of_its_own():
    _, run = _asking(review)
    for value in (object(), {"approved": object()}, {"approved": True, "amount": float("nan"), "due": object()}):
        assert {f.type for f in _refused(lambda value=value: run.answer("q@0", value))} == {"invalid_answer"}
    graph = _lines_graph()
    for value in ([object()], [{"sku": object(), "qty": 1}], pl.DataFrame({"sku": [1]})):
        assert [f.type for f in _refused(lambda value=value: graph.run().answer("q", value))] == ["invalid_answer"]


def test_after_an_answer_the_next_leg_calls_only_what_waited_on_it_and_a_row_left_unanswered_waits_on():
    gate, called = _counted()
    graph = registry_of(split, gate, upper).load({
        "nodes": [{"id": "s", "node": "split", "values": {"text": "a,b1,b2"}}, {"id": "q", "node": "gate"}, {"id": "u", "node": "upper"}],
        "edges": [["s.result", "q.text"], ["q.result", "u.text"]],
    })
    run = graph.run()
    called.clear()
    leg = graph.load_run(run.answer("q@1", "yes").to_json()).resume()
    assert called == [] and leg.status == "waiting" and [place(w) for w in leg.waiting] == ["q@2"]
    assert leg.results["u"]["result"] == ["A", "YES", None]
    assert leg.waiting[0] == run.waiting[1]  # the question is as it was asked


def test_a_node_that_reads_the_world_is_not_called_again_when_a_run_goes_on_from_waiting():
    read: list[str] = []

    @node("now", policy=Policy(reuse_when_inputs_match=False))
    def now(text: str) -> str:
        read.append(text)
        return text

    @node("ask")
    def ask(text: str) -> str:
        raise Ask("?")

    graph = registry_of(now, ask).load({"nodes": [{"id": "n", "node": "now", "values": {"text": "x"}}, {"id": "q", "node": "ask"}], "edges": [["n.result", "q.text"]]})
    graph.run().answer("q", "y").resume()
    assert read == ["x"]


def test_a_row_skipped_above_never_asks_and_the_others_complete_when_answered():
    @node("only-b")
    def only_b(text: str) -> str:
        return text if text.startswith("b") else SKIPPED

    gate, called = _counted()
    graph = registry_of(split, only_b, gate).load({
        "nodes": [{"id": "s", "node": "split", "values": {"text": "a,b1,b2"}}, {"id": "o", "node": "only-b"}, {"id": "q", "node": "gate"}],
        "edges": [["s.result", "o.text"], ["o.result", "q.text"]],
    })
    run = graph.run()
    assert called == ["b1", "b2"] and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    assert run.answer({"q@1": "1", "q@2": "2"}).resume().results["q"]["result"] == [None, "1", "2"]


def test_waiting_rows_are_in_node_then_row_order_whatever_order_they_asked_in():
    @node("slow-ask", policy=Policy(concurrency=4))
    def slow_ask(text: str) -> str:
        time.sleep(0.02 * (3 - int(text)))
        raise Ask(text)

    @node("ask")
    def ask(text: str) -> str:
        raise Ask(text)

    graph = registry_of(split, slow_ask, ask).load({
        "nodes": [{"id": "s", "node": "split", "values": {"text": "0,1,2"}}, {"id": "a", "node": "slow-ask"}, {"id": "b", "node": "ask"}],
        "edges": [["s.result", "a.text"], ["s.result", "b.text"]],
    })
    run = graph.run()
    assert [place(w) for w in run.waiting] == ["a@0", "a@1", "a@2", "b@0", "b@1", "b@2"]
    assert [place(w) for w in graph.load_run(run.to_json()).waiting] == [place(w) for w in run.waiting]


def test_a_waiting_row_keeps_the_table_and_series_it_received_through_a_save():
    class Row(TypedDict):
        name: str

    @node("rows")
    def rows() -> Table[Row]:
        return pl.DataFrame({"name": ["Ida", "Bo"]})

    @node("look")
    def look(names: Series[str], table: Table[Row]) -> bool:
        raise Ask("Right?")

    graph = registry_of(rows, look).load({"nodes": [{"id": "r", "node": "rows"}, {"id": "q", "node": "look"}],
                                           "edges": [["r.result.name", "q.names"], ["r.result", "q.table"]]})
    run = graph.run()
    assert run.waiting[0].inputs == {"names": ["Ida", "Bo"], "table": {"name": ["Ida", "Bo"]}}
    assert graph.load_run(run.to_json()).waiting == run.waiting


def test_rows_a_caller_types_in_as_a_list_pause_save_answer_and_finish():
    gate, _ = _counted()
    graph = registry_of(gate).load({"nodes": [{"id": "q", "node": "gate"}]})
    run = graph.load_run(graph.run(text=["a", "b"]).to_json())
    assert [place(w) for w in run.waiting] == ["q@1"]
    assert graph.load_run(run.answer("q@1", "B").to_json()).resume().outputs == {"q.result": ["a", "B"]}


def test_an_answer_names_a_row_by_its_place_so_one_for_a_row_that_no_longer_waits_is_refused():
    gate, _ = _counted()
    graph = registry_of(split, gate).load({"nodes": [{"id": "s", "node": "split"}, {"id": "q", "node": "gate"}], "edges": [["s.result", "q.text"]]})
    run = graph.run(text="b1,a")
    moved = run.again(text="a,b1")  # the same question, now on row 1
    assert [place(w) for w in run.waiting] == ["q@0"] and [place(w) for w in moved.waiting] == ["q@1"]
    assert [f.type for f in _refused(lambda: moved.answer("q@0", "B1"))] == ["not_waiting"]


def test_a_saved_waiting_row_the_graph_never_made_is_refused_on_load():
    gate, _ = _counted()
    graph, run = _asking(gate, text="a,b")
    saved = run.to_json().replace('"keys":[[0],[1]]', '"keys":[[0],[7]]')
    assert saved != run.to_json()
    (fault,) = _refused(lambda: graph.load_run(saved))
    assert fault.type == "state_unreadable"


def test_a_question_record_is_json_a_host_draws_from_with_no_graph_loaded():
    @node("assign")
    def assign(ticket: str) -> Annotated[str, Param(title="Assignee")]:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B.")], default="ida", widget=Textarea())

    _, run = _once(assign, ticket="T-1")
    written = run.waiting[0].model_dump(mode="json")
    assert re.fullmatch(r"q~[0-9a-f]{8}", written.pop("id"))  # the place, and a fingerprint of the question asked there
    assert written == {
        "node_id": "q", "row": [], "prompt": "Who takes T-1?", "inputs": {"ticket": "T-1"},
        "questions": [{
            "name": "result", "title": "Assignee", "shape": "one", "required": False, "type": "str", "schema": {"type": "string"},
            "choices": [{"id": "ida", "title": "Ida B."}], "default": "ida", "widget": {"type": "textarea", "rows": 4},
            "columns": None, "description": None, "exclusive_group": None, "setting": False,
        }],
    }


def test_a_literal_says_the_type_of_its_values_on_an_input_and_on_a_question():
    @node("priority")
    def priority(level: Literal["a", "b"], ticket: str) -> Literal["low", "high"]:
        raise Ask(f"How urgent is {ticket}?")

    assert [field.type for field in priority.describe().inputs] == ["str", "str"]
    run = registry_of(priority).load(make_graph(priority, values={"priority.level": "a"})).run(ticket="T-1")
    (question,) = run.waiting[0].questions
    assert question.type == "str" and question.json_schema == {"enum": ["low", "high"], "type": "string"}


def test_one_row_answered_twice_in_one_call_is_refused_and_nothing_is_answered():
    gate, _ = _counted()
    _, run = _asking(gate, text="b1,b2")
    faults = _refused(lambda: run.answer({"q@0": "one", run.waiting[0].id: "two"}))
    assert [(f.type, f.loc) for f in faults] == [("invalid_answer", ("q", 0))]
    assert [place(w) for w in run.waiting] == ["q@0", "q@1"]


def test_an_id_whose_row_is_not_a_row_answers_nothing_not_even_a_node_that_runs_once():
    @node("once")
    def once(text: str) -> str:
        raise Ask("ok?")

    run = registry_of(once).load(make_graph(once)).run(text="x")
    assert [(f.type, f.loc) for f in _refused(lambda: run.answer("once@banana", "y"))] == [("not_waiting", ("once",))]
    assert run.answer("once", "y").status == "waiting"  # answered, not yet resumed


async def test_a_cancelled_run_with_questions_waiting_takes_no_answer_until_it_is_resumed():
    import asyncio

    gate, called = _counted()
    _, run = _asking(gate, text="b1,b2")
    cancel = asyncio.Event()
    cancel.set()
    async for event in run.stream(cancel=cancel):
        if event.type == "run_end":
            cancelled = event.run
    assert cancelled.status == "cancelled" and [place(w) for w in cancelled.waiting] == ["q@0", "q@1"]

    (problem,) = _refused(lambda: cancelled.answer("q@0", "B1"))
    assert problem.type == "run_cancelled"

    resumed = cancelled.resume()
    assert resumed.status == "waiting" and resumed.answer("q@0", "B1").status == "waiting"


def test_an_embeddings_questions_are_named_by_the_outputs_it_offers():
    from conductor import GraphRecord, NodeRecord, NodeRegistry, make_graph, node

    @node("ask")
    def ask(text: str) -> str:
        raise Ask(f"{text}?")

    @node("ask2")
    def ask2(text: str) -> str:
        raise Ask(f"again {text}?")

    registry = NodeRegistry()
    registry.add(ask, ask2, node("both")(GraphRecord(nodes=[NodeRecord(id="q", node="ask"), NodeRecord(id="q2", node="ask2")])))
    run = registry.load(make_graph(registry["both"], values={"both.q/text": "a", "both.q2/text": "b"})).run()
    assert sorted(q.name for w in run.waiting for q in w.questions) == ["q/result", "q2/result"]
    done = run.answer({w.id: "yes" for w in run.waiting}).resume()
    assert done.outputs == {"both.q/result": "yes", "both.q2/result": "yes"}


def test_a_form_inside_an_embedding_is_answered_by_the_names_the_embedding_offers():
    from dataclasses import dataclass

    from conductor import GraphRecord, NodeRecord, NodeRegistry, node

    @dataclass(frozen=True)
    class Two:
        a: str
        b: int = 0

    @node("form")
    def form(t: str) -> Two:
        raise Ask("two?")

    registry = NodeRegistry()
    registry.add(form, node("both")(GraphRecord(nodes=[NodeRecord(id="z1", node="form"), NodeRecord(id="z2", node="form")])))
    run = registry.load(GraphRecord(nodes=[NodeRecord(id="e", node="both")])).run({"z1/t": "1", "z2/t": "2"})
    answers = {w.id: {q.name: "A" if q.name.endswith("a") else 5 for q in w.questions} for w in run.waiting}
    assert sorted(name for given in answers.values() for name in given) == ["z1/a", "z1/b", "z2/a", "z2/b"]
    assert run.answer(answers).resume().results == {"e": {"z1/a": "A", "z1/b": 5, "z2/a": "A", "z2/b": 5}}
    (wrong,) = _refused(lambda: run.answer({run.waiting[0].id: {"z1/a": 3}}))
    assert wrong.ctx["question"] == "z1/a"


def test_an_answer_to_a_run_with_nothing_waiting_says_how_the_run_stands():
    @node("plain")
    def plain(text: str) -> str:
        return text

    run = registry_of(plain).load(make_graph(plain)).run(text="a")
    (problem,) = _refused(lambda: run.answer("plain", "b"))
    assert (problem.type, problem.ctx["status"]) == ("not_waiting", "complete") and "a run that is complete" in problem.msg


@pytest.mark.parametrize("choices", [[True, 2], ["1"]], ids=["a bool", "a number as text"])
def test_a_choice_is_read_as_its_answer_would_be_so_one_a_number_refuses_fails_the_node(choices: list[object]) -> None:
    @node("pick")
    def pick() -> int:
        raise Ask("?", choices=choices)

    _, run = _once(pick)
    assert (run.status, run.cause.type) == ("failed", "invalid_ask")
    assert run.cause.ctx["reason"] == "a choice for 'result' is not what it returns: Input should be a valid integer"


def test_a_model_that_isnt_a_choice_is_a_value_to_param_and_to_ask_alike() -> None:
    from pydantic import BaseModel

    class Option(BaseModel):  # a host's own option record, not a Choice
        id: str
        title: str

    ida = Option(id="ida", title="Ida B.")
    with pytest.raises(NodeDeclarationError) as declared:

        @node("assign_by_param")
        def assign_by_param(who: Annotated[str, Param(choices=[ida])]) -> str:
            return who

    assert declared.value.code == "choices_invalid"

    @node("assign")
    def assign() -> str:
        raise Ask("Who?", choices=[ida])

    _, run = _once(assign)
    assert (run.status, run.cause.type) == ("failed", "invalid_ask")
    assert run.cause.ctx["reason"] == "a choice for 'result' is not what it returns: Input should be a valid string"
