"""Tests asking a person for a value while a graph runs.

A node that needs a person raises ``Ask``, and its ``@node`` says ``asks=True``. Its call stops and waits. The
node's return type says what kind of answer it takes. The ``Ask`` can add what is only known as the node runs, such
as choices to pick from or a proposed default.

A host sees each call that waits as a ``Waiting``. Its ``questions`` are one per output of the node. For example,
``approve(line: str) -> bool`` raising ``Ask("Approve b?")`` asks one question, ``result``, a yes or no.

``run.answer`` checks an answer against those questions. It reports every fault at once, and it calls no node. A
node that needs the answer gets it through an edge from the node that asked.
"""

import datetime
import json
import re
import time
from dataclasses import FrozenInstanceError, dataclass, fields
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Choice,
    ExternalError,
    FieldDescription,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRegistry,
    Param,
    Policy,
    Run,
    Series,
    Table,
    Waiting,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Textarea
from pydantic import MISSING, AfterValidator, ValidationError

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
    """Runs ``definition`` once for each comma-separated piece of ``text``, and returns the graph and the run.

    The graph has two nodes. ``split`` cuts ``text`` at the commas, and ``definition``, as node ``q``, is called
    once per piece. The outputs of ``q`` are the graph's outputs.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"constant": text}}},
                    "q": {
                        "node": definition.id,
                        "inputs": {
                            next(iter(definition.inputs)): {
                                "sources": [{"node": "s", "output": "result"}]
                            }
                        },
                    },
                },
                "outputs": _outputs_of(definition),
            }
        ),
        registry_of(split, definition, *more),
    )
    return graph, graph.run()


def _once(definition: Any, **values: Any) -> Any:
    """Runs ``definition`` once, on its own, and returns the graph and the run.

    The node is ``q``, ``values`` are its constants, and its outputs are the graph's outputs.
    """
    inputs = {name: {"constant": value} for name, value in values.items()}
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"q": {"node": definition.id, "inputs": inputs}},
                "outputs": _outputs_of(definition),
            }
        ),
        registry_of(definition),
    )
    return graph, graph.run()


def _outputs_of(definition: Any) -> dict[str, Any]:
    """Returns a graph output for each output of ``definition``, placed as node ``q``, under the same name."""
    return {name: {"node": "q", "output": name} for name in definition.outputs}


def _refused(call: Any) -> list[Any]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    return raised.value.problems


# -- the return type says how to ask ---------------------------------------------------------------


def test_a_text_return_with_a_bare_ask_is_a_free_text_question():
    @node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask(f"Why was {ticket} closed?")

    _, run = _once(reason, ticket="T-1")
    (waiting,) = run.waiting
    (question,) = waiting.questions
    assert (
        waiting.prompt,
        question.type,
        question.choices,
        question.default,
        question.required,
    ) == ("Why was T-1 closed?", "str", None, MISSING, True)
    assert run.answer(waiting, "a duplicate").resume().outputs == {"result": "a duplicate"}


def test_how_a_question_looks_is_its_types_so_a_hook_can_give_each_output_a_widget():
    def outputs(names: str) -> dict[str, Any]:
        return {name: Annotated[str, Param(widget=Textarea(rows=2))] for name in names.split(",")}

    @node("notes", title="Notes", description="For a test.", compute_outputs=outputs, asks=True)
    def notes(names: str) -> dict[str, Any]:
        raise Ask("Notes", default={"a": "none"})

    _, run = _once(notes, names="a,b")
    assert [(q.name, q.widget, q.default, q.required) for q in run.waiting[0].questions] == [
        ("a", {"type": "textarea", "rows": 2}, "none", False),
        ("b", {"type": "textarea", "rows": 2}, MISSING, True),
    ]


def test_a_lone_branch_is_asked_as_required_as_its_answer_must_give_it():
    """A node's only output is asked as required, even when it is a branch.

    A node of one output is answered with its value, never with a mapping that could leave it out. So the question
    says what the answer check does. A ``compute_outputs`` hook can give a node one branch.
    """

    def outputs() -> dict[str, Any]:
        return {"way": (str, SKIPPED)}

    @node("lone", title="Lone", description="For a test.", compute_outputs=outputs, asks=True)
    def lone(text: str) -> dict[str, Any]:
        raise Ask("Which way?")

    _, run = _once(lone, text="a")
    [question] = run.waiting[0].questions
    assert (question.name, question.required) == ("way", True)
    assert (
        _refused(lambda: run.answer("q", None))[0].ctx["reason"] == "Input should be a valid string"
    )
    assert run.answer("q", "left").resume().results["q"] == {"way": "left"}


def test_a_node_that_returns_nothing_is_answered_with_none_to_go_on():
    @node("ack", title="Ack", description="For a test.", asks=True)
    def ack(text: str) -> None:
        raise Ask(f"Seen {text}?")

    _, run = _once(ack, text="x")
    assert run.waiting[0].questions == []
    assert run.answer("q", None).resume().status == "complete"


def test_an_ask_that_sets_nothing_saves_only_its_prompt():
    @node("plain", title="Plain", description="For a test.", asks=True)
    def plain(text: str) -> str:
        raise Ask("?")

    _, run = _once(plain, text="x")
    assert '"questions":{"0":{"prompt":"?"}}' in run.to_json()


def test_a_bool_is_a_yes_or_no_and_a_word_for_it_is_refused():
    @node("confirm", title="Confirm", description="For a test.", asks=True)
    def confirm(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph, run = _once(confirm, name="F-1")
    (waiting,) = run.waiting
    (question,) = waiting.questions
    assert (place(waiting), waiting.prompt, question.json_schema, question.required) == (
        "q",
        "Approve F-1?",
        {"type": "boolean"},
        True,
    )
    (fault,) = _refused(lambda: run.answer(waiting, "maybe"))
    assert (fault.type, fault.loc, fault.ctx["waiting_id"]) == (
        "invalid_answer",
        ("q",),
        waiting.id,
    )
    assert run.answer(waiting, True).resume().outputs == {"result": True}


def test_a_literal_offers_its_values_and_an_answer_outside_them_is_refused():
    @node("urgency", title="Urgency", description="For a test.", asks=True)
    def urgency(name: str) -> Literal["low", "high"]:
        raise Ask("How urgent?")

    _, run = _once(urgency, name="x")
    (question,) = run.waiting[0].questions
    assert question.choices == (Choice("low", "low"), Choice("high", "high"))
    assert _refused(lambda: run.answer("q", "urgent"))[0].type == "invalid_answer"
    assert run.answer("q", "high").resume().outputs == {"result": "high"}


@dataclass(frozen=True)
class _Closing:
    reason: str
    hours: float = 1.0
    comment: str | None = None
    escalate_to: Literal["ida", "bo"] = SKIPPED


def test_a_branch_typed_as_a_literal_offers_its_values_and_left_out_is_skipped():
    """Fixed choices on a form's field are a ``Literal`` on the dataclass, and the ``Ask`` stays bare.

    ``describe()`` shows them before any run. The branch is not required: left out of the answer, it is skipped.
    """

    @node("close", title="Close", description="For a test.", asks=True)
    def close(ticket: str) -> _Closing:
        raise Ask(f"Close {ticket}")

    described = {output.name: output for output in close.describe().outputs}
    assert described["escalate_to"].choices == (Choice("ida", "ida"), Choice("bo", "bo"))
    _, run = _once(close, ticket="T-1")
    (question,) = [q for q in run.waiting[0].questions if q.name == "escalate_to"]
    assert (question.choices, question.required) == (described["escalate_to"].choices, False)
    (fault,) = _refused(lambda: run.answer("q", {"reason": "done", "escalate_to": "zed"}))
    assert fault.type == "invalid_answer"
    taken = run.answer("q", {"reason": "done", "escalate_to": "ida"}).resume()
    assert taken.outputs["escalate_to"] == "ida"
    skipped = run.answer("q", {"reason": "done"}).resume()
    assert (skipped.status, skipped.outputs["escalate_to"]) == ("complete", None)


def test_choices_the_run_knows_limit_the_answer_and_a_choice_gives_a_title():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), "bo"])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == (Choice("ida", "Ida B."), Choice("bo", "bo"))
    (fault,) = _refused(lambda: run.answer("q", "carl"))
    assert fault.msg == "That isn't a valid answer: the value is not one of the choices."
    assert run.answer("q", "bo").resume().outputs == {"result": "bo"}


def test_an_answer_picks_a_choice_by_the_value_its_type_reads_so_1_picks_the_float_choice_1_0():
    """An answer is compared with the choices after its type has read it.

    The comparison is by JSON text, and it uses the value the type gives back, not the value as given.
    """

    @node("rate", title="Rate", description="For a test.", asks=True)
    def rate(ticket: str) -> float:
        raise Ask(f"Rate {ticket}", choices=[1.0, 2.5])

    _, run = _once(rate, ticket="T-1")
    # A browser sends 1.0 in JSON as 1.
    assert run.answer("q", 1).resume().outputs == {"result": 1.0}
    assert (
        _refused(lambda: run.answer("q", 2))[0].msg
        == "That isn't a valid answer: the value is not one of the choices."
    )


def test_an_input_is_checked_against_its_choices_by_their_json_as_an_answer_is():
    @node("take", title="Take", description="For a test.")
    def take(x: Annotated[Any, Param(choices=[1, {"a": 1, "b": 2}])]) -> str:
        return "taken"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"q": {"node": "take", "inputs": {"x": {"graph_input": "x"}}}},
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(take),
    )
    for taken in (1, {"b": 2, "a": 1}, None):
        assert graph.run(x=taken).outputs == {"result": "taken"}
    for refused in (True, 1.0, {"a": True, "b": 2}):
        with pytest.raises(InvalidCallError) as raised:
            graph.run(x=refused)
        assert "the value is not one of the choices" in raised.value.problems[0].msg


def test_two_fields_whose_choices_python_takes_as_equal_each_accept_only_their_own():
    """``True == 1`` in Python, but their JSON differs, so ``choices=[True]`` and ``choices=[1]`` stay apart."""

    def taking(choice: Any) -> Any:
        @node(f"take_{choice}".lower(), title="Take", description="For a test.")
        def take(x: Annotated[Any, Param(choices=[choice])]) -> str:
            return "taken"

        return Graph.compile(
            GraphRecord.model_validate(
                {
                    "nodes": {"q": {"node": take.id, "inputs": {"x": {"graph_input": "x"}}}},
                    "outputs": {"result": {"node": "q", "output": "result"}},
                }
            ),
            registry_of(take),
        )

    for choice, other in ((1, True), (True, 1)):
        graph = taking(choice)
        assert graph.run(x=choice).outputs == {"result": "taken"}
        with pytest.raises(InvalidCallError):
            graph.run(x=other)


def test_a_series_whose_items_take_none_is_answered_with_none_beside_its_picks():
    @node("tags", title="Tags", description="For a test.", asks=True)
    def tags(text: str) -> Series[str | None]:
        raise Ask(f"Tag {text}", choices=["a", "b"])

    _, run = _once(tags, text="x")
    assert run.answer("q", ["a", None, "b", None]).resume().outputs == {
        "result": ["a", None, "b", None]
    }
    assert (
        _refused(lambda: run.answer("q", ["a", None, "a"]))[0].msg
        == "That isn't a valid answer: a choice is picked twice."
    )


def test_a_choice_that_is_not_a_finite_float_is_picked_by_its_json_text():
    @node("cap", title="Cap", description="For a test.", asks=True)
    def cap(ticket: str) -> float:
        raise Ask(f"Cap {ticket}", choices=[1.5, float("inf")])

    _, run = _once(cap, ticket="T-1")
    assert run.waiting[0].questions[0].choices == (Choice(1.5, "1.5"), Choice("Infinity", "inf"))
    assert run.answer("q", float("inf")).resume().outputs == {"result": float("inf")}


def test_a_series_that_takes_none_is_answered_none_beside_its_choices():
    @node("tags", title="Tags", description="For a test.", asks=True)
    def tags(text: str) -> Series[str] | None:
        raise Ask(f"Tag {text}", choices=["a", "b"])

    _, run = _once(tags, text="x")
    assert run.answer("q", None).resume().outputs == {"result": None}
    assert run.answer("q", ["a"]).resume().outputs == {"result": ["a"]}


def test_an_ask_proposing_none_for_a_series_that_takes_none_waits_with_that_default():
    @node("tags", title="Tags", description="For a test.", asks=True)
    def tags(text: str) -> Series[str] | None:
        raise Ask(f"Tag {text}", choices=["a"], default=None)

    _, run = _once(tags, text="x")
    (question,) = run.waiting[0].questions
    assert (question.default, question.required) == (None, True)


class Person(Choice):
    team: str = ""


def test_a_field_whose_choices_are_a_hosts_own_keeps_their_class_beside_one_with_the_same_json():
    @node("plain", title="Plain", description="For a test.")
    def plain(x: Annotated[str, Param(choices=[Choice("ida", "Ida", team="s")])]) -> str:
        return x

    @node("own", title="Own", description="For a test.")
    def own(x: Annotated[str, Param(choices=[Person("ida", "Ida", team="s")])]) -> str:
        return x

    assert [type(c) for c in plain.describe().inputs[0].choices or ()] == [Choice]
    assert [type(c) for c in own.describe().inputs[0].choices or ()] == [Person]


def test_a_hosts_own_choice_keeps_its_fields_in_the_question():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Person("ida", "Ida B.", team="sales")])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == (Person("ida", "Ida B.", team="sales"),)
    assert run.answer("q", "ida").resume().outputs == {"result": "ida"}


def test_a_hosts_own_choice_keeps_its_fields_in_a_description_and_a_saved_question_read_back():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(
        ticket: str,
        team: Annotated[str, Param(choices=[Person("sales", "Sales", team="north")])] = "sales",
    ) -> str:
        raise Ask(f"Who takes {ticket}?", choices=[Person("ida", "Ida B.", team="sales")])

    described = assign.describe().inputs[1]
    assert described.choices == (Person("sales", "Sales", team="north"),)
    assert type(described).model_validate_json(described.model_dump_json()).choices[
        0
    ].model_dump() == {"id": "sales", "title": "Sales", "team": "north"}
    graph, run = _once(assign, ticket="T-1")
    loaded = Run.from_json(run.to_json(), graph)
    assert [choice.model_dump() for choice in loaded.waiting[0].questions[0].choices] == [
        {"id": "ida", "title": "Ida B.", "team": "sales"}
    ]
    assert loaded.answer("q", "ida").resume().outputs == {"result": "ida"}


def test_a_choice_is_read_by_the_type_it_is_a_value_of_so_1_on_a_float_is_1_0():
    @node("scale", title="Scale", description="For a test.", asks=True)
    def scale(by: Annotated[float, Param(choices=[1, Choice(2, "Double")])] = 1.0) -> float:
        raise Ask("How much?", choices=[1, 2])

    declared = scale.describe().inputs[0].choices
    _, run = _once(scale)
    asked = run.waiting[0].questions[0].choices
    for choices in (declared, asked):
        assert choices is not None and [(choice.id, type(choice.id)) for choice in choices] == [
            (1.0, float),
            (2.0, float),
        ]
    assert (
        '"choices":[{"id":1.0,"title":"1"},{"id":2.0,"title":"Double"}]'
        in scale.describe().model_dump_json()
    )


@pytest.mark.parametrize("choices", [[], [1, 1]], ids=["none", "one twice"])
def test_no_choices_or_one_twice_are_refused_by_param_when_declared_and_by_ask_when_raised(
    choices: list[int],
) -> None:
    with pytest.raises(NodeDeclarationError) as declared:

        @node("pick_declared", title="Pick", description="For a test.")
        def pick_declared(n: Annotated[int, Param(choices=choices)]) -> int:
            return n

    assert declared.value.problems[0].type == "choices_invalid"

    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> int:
        raise Ask("Which?", choices=choices)

    cause = _once(pick)[1].cause
    assert (cause.type, cause.ctx["reason"]) == (
        "invalid_ask",
        "the choices for 'result' are wrong ("
        + ("none is given" if not choices else "one is given twice")
        + ")",
    )


def test_two_dict_choices_with_their_keys_in_another_order_are_one_choice_given_twice():
    twice = [{"a": 1, "b": 2}, {"b": 2, "a": 1}]
    with pytest.raises(NodeDeclarationError) as declared:

        @node("pick_declared", title="Pick", description="For a test.")
        def pick_declared(n: Annotated[dict[str, int], Param(choices=twice)]) -> int:
            return 0

    assert declared.value.problems[0].type == "choices_invalid"

    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> dict[str, int]:
        raise Ask("Which?", choices=twice)

    cause = _once(pick)[1].cause
    assert (cause.type, cause.ctx["reason"]) == (
        "invalid_ask",
        "the choices for 'result' are wrong (one is given twice)",
    )


def test_the_choices_an_ask_offers_a_series_are_each_read_as_one_item():
    @node("numbers", title="Numbers", description="For a test.", asks=True)
    def numbers() -> Series[float]:
        raise Ask("Which?", choices=[1, Choice(2, "Two")])

    @node("lists", title="Lists", description="For a test.", asks=True)
    def lists() -> Series[float]:
        raise Ask("Which?", choices=[[1.0, 2.0]])

    _, run = _once(numbers)
    assert run.waiting[0].questions[0].choices == (Choice(1.0, "1"), Choice(2.0, "Two"))
    assert run.answer("q", [2, 1]).resume().outputs == {"result": [2.0, 1.0]}
    assert (
        _once(lists)[1].cause.ctx["reason"]
        == "the choices for 'result' are wrong (Input should be a valid number)"
    )


def test_a_number_and_a_date_are_read_from_their_json_form_into_the_outputs_type():
    @node("when", title="When", description="For a test.", asks=True)
    def when(name: str) -> datetime.date:
        raise Ask("When?")

    @node("weekday", title="Weekday", description="For a test.")
    def weekday(day: datetime.date) -> int:
        return day.isoweekday()

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "q": {"node": "when", "inputs": {"name": {"constant": "x"}}},
                    "w": {
                        "node": "weekday",
                        "inputs": {"day": {"sources": [{"node": "q", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "w", "output": "result"}},
            }
        ),
        registry_of(when, weekday),
    )
    run = graph.run()
    assert run.waiting[0].questions[0].json_schema == {"format": "date", "type": "string"}
    done = Run.from_json(run.answer("q", "2026-11-10").to_json(), graph).resume()
    assert done.results["q"]["result"] == datetime.date(2026, 11, 10) and done.outputs == {
        "result": 2
    }
    assert _refused(lambda: run.answer("q", "next tuesday"))[0].type == "invalid_answer"


def test_a_lone_output_is_titled_by_its_own_param_never_by_the_node():
    @node("ask", description="For a test.", title="Approve the invoice", asks=True)
    def ask() -> str:
        raise Ask("?")

    @node("named", title="Named", description="For a test.", asks=True)
    def named() -> Annotated[str, Param(title="Decision", description="What you decided")]:
        raise Ask("?")

    assert _once(ask)[1].waiting[0].questions[0].title is None
    question = _once(named)[1].waiting[0].questions[0]
    assert (question.title, question.description) == ("Decision", "What you decided")


def test_a_question_whose_type_takes_none_may_be_answered_with_none():
    @node("note", title="Note", description="For a test.", asks=True)
    def note(name: str) -> str | None:
        raise Ask("A note?")

    _, run = _once(note, name="x")
    assert run.waiting[0].questions[0].required is True
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
    assert (approved.required, amount.default, amount.required, note.required, note.widget) == (
        True,
        100.0,
        False,
        False,
        {"type": "textarea", "rows": 6},
    )
    assert (due.default, due.required) == ("2026-12-01", False)  # Its own default is proposed.


def test_a_form_answered_in_part_gives_each_field_left_out_its_default_or_none_and_refuses_one_with_neither():
    graph, run = _waiting_outcomes(review)
    answered = run.answer(
        {
            "q@0": {"approved": True},
            "q@1": {"approved": False, "amount": 5, "note": "late", "due": "2027-01-01"},
        }
    ).resume()
    assert answered.results["q"] == {
        "approved": [True, False],
        "amount": [100.0, 5.0],
        "note": [None, "late"],
        "due": [datetime.date(2026, 12, 1), datetime.date(2027, 1, 1)],
    }
    faults = _refused(lambda: run.answer("q@0", {"amount": "lots", "colour": "red"}))
    assert [(f.ctx.get("question"), f.msg) for f in faults] == [
        ("colour", "That isn't a valid answer: there is no question 'colour'."),
        ("approved", "That isn't a valid answer: 'approved' has no answer."),
        ("amount", "That isn't a valid answer: 'amount': Input should be a valid number."),
    ]
    assert (
        _refused(lambda: run.answer("q@0", True))[0].msg
        == "That isn't a valid answer: the answer isn't a mapping of approved, amount, note, due, one value per question."
    )


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
    assert (note.required, escalate.required, escalate.default) == (
        True,
        False,
        MISSING,
    )  # No default, so leaving it out means the branch isn't taken.
    assert "default" not in escalate.model_dump(mode="json")
    run.to_json()

    answered = run.answer(
        {"q@0": {"note": "fine"}, "q@1": {"note": "late", "escalate": "boss"}}
    ).resume()

    assert answered.results["q"] == {"note": ["fine", "late"], "escalate": [None, "boss"]}
    assert answered.results["q"] == Run.from_json(answered.to_json(), run.graph).results["q"]


@dataclass(frozen=True)
class Triage:
    note: str
    comment: str | None
    escalate: str | None
    keep: str = SKIPPED
    drop: str = SKIPPED


@node("triage", title="Triage", description="For a test.", asks=True)
def triage(text: str) -> Triage:
    raise Ask(f"Triage {text}")


def test_default_in_json_is_what_leaving_the_question_out_gives_and_null_is_only_none():
    _, run = _once(triage, text="a")
    written = {
        q["name"]: (q["required"], q.get("default", "no default"))
        for q in run.waiting[0].model_dump(mode="json")["questions"]
    }
    assert written == {
        "note": (True, "no default"),  # Must be answered.
        "comment": (False, None),  # Left out, it is None.
        "escalate": (False, None),  # Not a branch, so it may have a value beside keep or drop.
        "keep": (False, "no default"),  # Left out, the branch is skipped.
        "drop": (False, "no default"),
    }
    assert Waiting.model_validate_json(run.waiting[0].model_dump_json()) == run.waiting[0]
    answered = run.answer(run.waiting[0], {"note": "n", "escalate": "e", "keep": "k"}).resume()
    assert answered.results["q"] == {
        "note": "n",
        "comment": None,
        "escalate": "e",
        "keep": "k",
        "drop": None,
    }


def test_an_output_in_the_palette_is_the_question_its_type_sets():
    @node("maybe", title="Maybe", description="For a test.", asks=True)
    def maybe(text: str) -> str | None:
        raise Ask("Anything?")

    (in_palette,) = maybe.describe().outputs
    _, run = _once(maybe, text="a")
    assert (in_palette.required, in_palette.default) == (False, None)
    assert in_palette == run.graph.describe().nodes["q"].outputs[0]
    # One output is answered with a value, so its question is required.
    assert run.waiting[0].questions[0] == in_palette.model_copy(update={"required": True})


@dataclass(frozen=True)
class Route:
    keep: str = SKIPPED
    drop: str = SKIPPED


def test_an_ask_proposing_defaults_for_two_branches_fails_the_node():
    """An ``Ask``'s defaults fill an answer in, so they give a value to at most one branch, as an answer does."""

    @node("route", title="Route", description="For a test.", asks=True)
    def route(text: str) -> Route:
        raise Ask(f"Keep {text}?", default={"keep": "a", "drop": "b"})

    _, run = _once(route, text="a")
    assert (run.status, run.cause.type, run.waiting) == ("failed", "invalid_ask", [])
    assert run.cause.ctx["reason"] == (
        "the default fills keep, drop, but they default to SKIPPED, so they are branches of one node"
    )


def test_an_ask_may_propose_a_default_for_one_branch_which_another_answered_branch_skips():
    """A default on one branch fills the answer in. Answering another branch skips it, never gives it the default."""

    @node("route", title="Route", description="For a test.", asks=True)
    def route(text: str) -> Route:
        raise Ask(f"Keep {text}?", default={"keep": "a"})

    _, run = _once(route, text="a")
    keep, drop = run.waiting[0].questions
    assert (keep.default, keep.required, drop.default, drop.required) == (
        "a",
        False,
        MISSING,
        False,
    )
    assert run.answer("q", {}).resume().results["q"] == {"keep": "a", "drop": None}
    assert run.answer("q", {"drop": "b"}).resume().results["q"] == {"keep": None, "drop": "b"}


@dataclass(frozen=True)
class Maybe:
    value: str = SKIPPED


@pytest.mark.parametrize("returned", [Maybe, Routed], ids=["the only output", "beside an output"])
def test_an_ask_may_propose_a_default_for_a_lone_branch(returned):
    """A node's only branch may be filled in, whether it is the node's only output or not."""
    branch = next(field.name for field in fields(returned) if field.default is SKIPPED)

    @node("route", title="Route", description="For a test.", asks=True)
    def route(text: str) -> returned:  # type: ignore[valid-type]
        raise Ask(f"Keep {text}?", default={branch: "a"} if len(fields(returned)) > 1 else "a")

    _, run = _once(route, text="a")
    assert run.status == "waiting", run.cause
    assert [q.default for q in run.waiting[0].questions if q.name == branch] == ["a"]


@pytest.mark.parametrize(
    ("written", "error"),
    [
        ({"required": False, "default": None, "loops": ["s.result"]}, "loops_without_series"),
    ],
)
def test_a_field_whose_parts_contradict_each_other_is_refused(written, error):
    one = {"name": "x", "is_series": False, "type": "str", "json_schema": {"type": "string"}}
    with pytest.raises(ValidationError) as refused:
        FieldDescription.model_validate({**one, **written})
    assert refused.value.errors()[0]["type"] == error


def test_a_field_whose_parts_agree_reads_back():
    one = {"name": "x", "is_series": False, "type": "str", "json_schema": {"type": "string"}}
    assert (
        FieldDescription.model_validate({**one, "required": False, "default": None}).default is None
    )
    assert FieldDescription.model_validate({**one, "required": True, "default": "a"}).default == "a"
    assert FieldDescription.model_validate({**one, "required": False}).default is MISSING
    assert FieldDescription.model_validate(
        {**one, "required": False, "is_series": True, "loops": ["s.result"]}
    ).loops == ["s.result"]


@dataclass(frozen=True)
class Noted:
    note: str
    escalate: str | None


@node("route_to", title="Route to", description="For a test.", asks=True)
def route_to(text: str) -> Noted:
    raise Ask(f"Route {text}", choices={"escalate": ["boss", "team"]}, default={"escalate": "team"})


def test_a_waiting_call_shows_the_choices_and_default_its_answer_is_checked_against():
    _, run = _once(route_to, text="a")
    note, escalate = run.waiting[0].questions
    assert (note.required, note.choices, note.default) == (True, None, MISSING)
    assert ([c.id for c in escalate.choices], escalate.default, escalate.required) == (
        ["boss", "team"],
        "team",
        False,
    )

    assert run.answer("q", {"note": "fine"}).resume().outputs == {
        "note": "fine",
        "escalate": "team",
    }  # The Ask's default, not None.
    assert run.answer("q", {"note": "fine", "escalate": "boss"}).resume().outputs == {
        "note": "fine",
        "escalate": "boss",
    }
    assert [
        f.msg for f in _refused(lambda: run.answer("q", {"note": "fine", "escalate": "ceo"}))
    ] == [
        "That isn't a valid answer: 'escalate': the value is not one of the choices.",
    ]


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
    """Returns a graph where node ``q`` asks for a table of order lines.

    ``q`` is the ``lines`` node. The table is then split into one record per line, and ``each`` reads each record.
    """
    return Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "q": {"node": "lines", "inputs": {"order": {"constant": "O-1"}}},
                    "i": {
                        "node": "line-items",
                        "inputs": {"table": {"sources": [{"node": "q", "output": "result"}]}},
                    },
                    "e": {
                        "node": "each",
                        "inputs": {
                            "sku": {"sources": [{"node": "i", "output": "result", "field": "sku"}]},
                            "qty": {"sources": [{"node": "i", "output": "result", "field": "qty"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "e", "output": "result"}},
            }
        ),
        registry_of(lines, line_items, each),
    )


def test_a_table_is_answered_as_records_which_are_its_rows_and_keep_through_json():
    graph = _lines_graph()
    run = graph.run()
    (question,) = run.waiting[0].questions
    assert (
        not question.is_series
        and question.type.startswith("Table")
        and [
            c["properties"]["name"]["const"]
            for c in question.json_schema["properties"]["columns"]["prefixItems"]
        ]
        == ["sku", "qty"]
    )
    assert question.default == {
        "columns": [
            {"name": "sku", "type": "str", "values": ["A"]},
            {"name": "qty", "type": "int", "values": [1]},
        ]
    }  # A table in the JSON form conductor writes.
    answered = Run.from_json(
        Run.from_json(run.to_json(), graph)
        .answer("q", [{"sku": "B", "qty": 2}, {"sku": "C", "qty": 3}])
        .to_json(),
        graph,
    )
    assert answered.resume().outputs == {"result": ["BB", "CCC"]}


def test_a_table_default_survives_saving_and_answers_like_any_value():
    graph = _lines_graph()
    run = Run.from_json(graph.run().to_json(), graph)
    assert run.answer("q", run.waiting[0].questions[0].default).resume().outputs == {
        "result": ["A"]
    }


def test_an_empty_list_answers_a_table_with_no_rows_and_a_wrong_record_is_refused():
    graph = _lines_graph()
    run = graph.run()
    done = run.answer("q", []).resume()
    assert (
        done.status == "complete"
        and done.results["q"]["result"].columns == ["sku", "qty"]
        and done.results["e"]["result"] == []
    )
    assert [f.msg for f in _refused(lambda: run.answer("q", [{"sku": "B"}]))] == [
        "That isn't a valid answer: the column 'qty' is missing, and the row type needs it.",
    ]
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": "two"}]))[0].msg.startswith(
        "That isn't a valid answer: column 'qty':"
    )
    # Read strictly, like an answer of one value, so "2" is not an int.
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": "2"}]))[0].msg.startswith(
        "That isn't a valid answer: column 'qty':"
    )
    assert (
        _refused(lambda: run.answer("q", "B,2"))[0].msg
        == "That isn't a valid answer: a table must be a frame, or a list of records with one object per row."
    )


def test_a_table_answer_is_read_as_a_callers_records_are_with_its_cells_read_strictly():
    graph = _lines_graph()
    run = graph.run()
    extra = run.answer("q", [{"sku": "B", "qty": 2, "colour": "red"}]).resume()
    assert extra.results["q"]["result"].columns == ["sku", "qty"] and extra.outputs == {
        "result": ["BB"]
    }
    assert _refused(lambda: run.answer("q", [{"sku": "B", "qty": True}, {"sku": "C", "qty": 1}]))[
        0
    ].msg == ("That isn't a valid answer: column 'qty': a yes or no is not a number.")


def test_a_hooks_outputs_are_asked_by_their_types():
    def outputs(names: str) -> dict[str, Any]:
        return {name: bool for name in names.split(",")}

    @node("checks", title="Checks", description="For a test.", compute_outputs=outputs, asks=True)
    def checks(names: str) -> dict[str, Any]:
        raise Ask("Check each")

    outputs_declared = {
        name: {"node": "q", "output": name} for name in ("signed", "paid")
    }  # The graph outputs name the outputs the hook returns.
    record = GraphRecord.model_validate(
        {
            "nodes": {"q": {"node": "checks", "inputs": {"names": {"constant": "signed,paid"}}}},
            "outputs": outputs_declared,
        }
    )
    run = Graph.compile(record, registry_of(checks)).run()
    assert [(q.name, q.json_schema) for q in run.waiting[0].questions] == [
        ("signed", {"type": "boolean"}),
        ("paid", {"type": "boolean"}),
    ]
    assert run.answer("q", {"signed": True, "paid": False}).resume().outputs == {
        "signed": True,
        "paid": False,
    }


def test_answering_one_branch_skips_the_other_and_answering_none_skips_both():
    @node("route", title="Route", description="For a test.", asks=True)
    def route(text: str) -> Route:
        raise Ask(f"Keep {text}?")

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "q": {"node": "route", "inputs": {"text": {"constant": "a"}}},
                    "k": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "q", "output": "keep"}]}},
                    },
                    "d": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "q", "output": "drop"}]}},
                    },
                },
                "outputs": {
                    "d_result": {"node": "d", "output": "result"},
                    "k_result": {"node": "k", "output": "result"},
                },
            }
        ),
        registry_of(route, upper),
    )
    run = graph.run()
    assert [(q.name, q.required, q.default) for q in run.waiting[0].questions] == [
        ("keep", False, MISSING),
        ("drop", False, MISSING),
    ]
    done = run.answer("q", {"keep": "a"}).resume()
    assert (
        done.status == "complete"
        and done.results["k"]["result"] == "A"
        and done.results["d"]["result"] is None
    )
    assert (
        _refused(lambda: run.answer("q", {"keep": "a", "drop": "a"}))[0].msg
        == "That isn't a valid answer: keep, drop default to SKIPPED, so they are branches, and at most one of them can be answered."
    )
    neither = run.answer("q", {}).resume()
    assert neither.status == "complete" and neither.results["k"] == neither.results["d"] == {
        "result": None
    }


@dataclass(frozen=True)
class Side:
    left: str = SKIPPED
    right: str = SKIPPED


def test_two_ways_to_go_are_two_nodes_and_an_answer_to_each_is_checked_on_its_own():
    """A node's branches are one set: a call gives a value to at most one of them.

    Two independent ways to go, such as keep or drop and left or right, are two nodes. Answering two branches of
    each is two faults, one per node.
    """

    @node("route", title="Route", description="For a test.", asks=True)
    def route() -> Route:
        raise Ask("Keep or drop?")

    @node("side", title="Side", description="For a test.", asks=True)
    def side() -> Side:
        raise Ask("Left or right?")

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"r": {"node": "route"}, "s": {"node": "side"}},
                "outputs": {
                    "keep": {"node": "r", "output": "keep"},
                    "left": {"node": "s", "output": "left"},
                },
            }
        ),
        registry_of(route, side),
    ).run()

    problems = _refused(
        lambda: run.answer({"r": {"keep": "a", "drop": "b"}, "s": {"left": "a", "right": "b"}})
    )
    assert sorted(problem.ctx["reason"] for problem in problems) == [
        "keep, drop default to SKIPPED, so they are branches, and at most one of them can be answered",
        "left, right default to SKIPPED, so they are branches, and at most one of them can be answered",
    ]
    done = run.answer({"r": {"keep": "a"}, "s": {"left": "b"}}).resume()
    assert done.outputs == {"keep": "a", "left": "b"}


# -- picking several -----------------------------------------------------------------------------


@node("colours", title="Colours", description="For a test.", asks=True)
def colours(name: str) -> Series[Literal["red", "green", "blue"]]:
    raise Ask(f"Colours for {name}?", default=["red"])


def test_a_series_of_a_literal_is_a_pick_of_several_run_end_to_end_through_json():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"constant": "Ida,Bo"}}},
                    "q": {
                        "node": "colours",
                        "inputs": {"name": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}},
                    },
                    "j": {
                        "node": "join",
                        "inputs": {"texts": {"sources": [{"node": "u", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "j", "output": "result"}},
            }
        ),
        registry_of(split, colours, upper, join),
    )
    run = Run.from_json(graph.run().to_json(), graph)
    (question,) = run.waiting[0].questions
    assert (question.is_series, question.default, [c.id for c in question.choices]) == (
        True,
        ["red"],
        ["red", "green", "blue"],
    )
    faults = _refused(lambda: run.answer({"q@0": ["red", "red"], "q@1": ["pink"]}))
    assert [f.msg for f in faults] == [
        "That isn't a valid answer: a choice is picked twice.",
        "That isn't a valid answer: Input should be 'red', 'green' or 'blue'.",
    ]
    answered = Run.from_json(run.answer({"q@0": ["red", "blue"], "q@1": []}).to_json(), graph)
    assert answered.results["u"]["result"] == [
        [None, None],
        [],
    ]  # One list of picks per item. u hasn't run on them yet, so its results are None.
    done = answered.resume()
    assert (
        done.status == "complete"
        and done.results["q"]["result"] == [["red", "blue"], []]
        and done.results["j"]["result"] == ["RED+BLUE", ""]
    )


def test_a_series_with_choices_the_run_knows_is_a_pick_of_several_of_those():
    @node("people", title="People", description="For a test.", asks=True)
    def people(team: str) -> Series[str]:
        raise Ask(
            f"Who in {team}?",
            choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K."), Choice("carl", "Carl")],
        )

    _, run = _once(people, team="sales")
    (question,) = run.waiting[0].questions
    assert (question.is_series, [c.title for c in question.choices]) == (
        True,
        ["Ida B.", "Bo K.", "Carl"],
    )
    assert (
        _refused(lambda: run.answer("q", ["ida", "eve"]))[0].msg
        == "That isn't a valid answer: the value is not one of the choices."
    )
    assert run.answer("q", ["carl", "ida"]).resume().outputs == {"result": ["carl", "ida"]}


def test_a_choice_whose_id_is_a_dict_or_a_list_is_picked_by_its_json():
    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick(ticket: str) -> dict[str, int]:
        raise Ask("Which?", choices=[{"a": 1, "b": 2}, {"c": 3}])

    @node("picks", title="Picks", description="For a test.", asks=True)
    def picks(ticket: str) -> Series[list[int]]:
        raise Ask("Which?", choices=[[1, 2], [3]])

    _, one = _once(pick, ticket="t")
    assert one.answer("q", {"b": 2, "a": 1}).resume().outputs == {"result": {"b": 2, "a": 1}}
    assert (
        _refused(lambda: one.answer("q", {"c": 4}))[0].msg
        == "That isn't a valid answer: the value is not one of the choices."
    )
    _, several = _once(picks, ticket="t")
    assert several.answer("q", [[3], [1, 2]]).resume().outputs == {"result": [[3], [1, 2]]}
    assert (
        _refused(lambda: several.answer("q", [[3], [3]]))[0].msg
        == "That isn't a valid answer: a choice is picked twice."
    )
    assert (
        _refused(lambda: several.answer("q", [[3], [3], [4]]))[0].msg
        == "That isn't a valid answer: the value is not one of the choices."
    )


def test_a_node_per_item_of_a_list_answered_empty_shows_an_empty_list_before_and_after_resume():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "q": {"node": "colours", "inputs": {"name": {"constant": "x"}}},
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        registry_of(colours, upper),
    )
    answered = graph.run().answer("q", [])
    assert answered.results["u"]["result"] == []
    assert answered.resume().outputs == {"result": []}


# -- a wrong Ask fails the node --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ask", "reason"),
    [
        (
            lambda: Ask("?", default=3),
            "the default for 'result' isn't an answer a person could give (Input should be a valid string)",
        ),
        (
            lambda: Ask("?", default=SKIPPED),
            "the default for 'result' isn't an answer a person could give (the value can't be written as JSON)",
        ),
        (
            lambda: Ask("?", choices=["a", 2]),
            "the choices for 'result' are wrong (Input should be a valid string)",
        ),
        (
            lambda: Ask("?", choices=["a"], default="b"),
            "the default for 'result' isn't an answer a person could give (the value is not one of the choices)",
        ),
        (
            lambda: Ask("?", choices=["a", object()]),
            "the choices for 'result' are wrong (one can't be written as JSON)",
        ),
        (
            lambda: Ask("?", choices=[Choice(object(), "Thing")]),
            "the choices for 'result' are wrong (one can't be written as JSON)",
        ),
        (
            lambda: Ask("?", choices=["a", "a"]),
            "the choices for 'result' are wrong (one is given twice)",
        ),
        (lambda: Ask("?", choices=[]), "the choices for 'result' are wrong (none is given)"),
        (
            lambda: Ask("?", choices={"result": ["a"]}),
            "the node has one output, so choices= is a list, not a mapping",
        ),
    ],
)
def test_a_node_asking_what_its_output_cant_be_answered_with_fails_naming_why(ask, reason):
    @node("wrong", title="Wrong", description="For a test.", asks=True)
    def wrong() -> str:
        raise ask()

    _, run = _once(wrong)
    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx["reason"]) == (
        "failed",
        "invalid_ask",
        ("q",),
        reason,
    )


@pytest.mark.parametrize("stream", [False, True])
def test_a_question_on_inputs_with_no_json_form_fails_its_node_naming_the_input(stream):
    @node("make", title="Make", description="For a test.")
    def make(text: str) -> Any:
        return object()

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(label: str, value: Any) -> bool:
        raise Ask("ok?")

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "m": {"node": "make", "inputs": {"text": {"graph_input": "text"}}},
                    "a": {
                        "node": "approve",
                        "inputs": {
                            "label": {"constant": "x"},
                            "value": {"sources": [{"node": "m", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "a", "output": "result"}},
            }
        ),
        registry_of(make, approve),
    )

    async def streamed() -> Any:
        return [e async for e in graph.with_inputs(text="x").stream()][-1].run

    import asyncio

    run = asyncio.run(streamed()) if stream else graph.run(text="x")
    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx) == (
        "failed",
        "question_unwritable",
        ("a",),
        {"input": "value", "error": "PydanticSerializationError"},
    )


@pytest.mark.parametrize(
    ("ask", "reason"),
    [
        (
            lambda: Ask("?", default="a"),
            "the node has 2 outputs, so default= must be a mapping by output name",
        ),
        (
            lambda: Ask("?", default={"stay": "a"}),
            "default= names 'stay', which isn't an output. The outputs are keep, drop",
        ),
        (
            lambda: Ask("?", choices=["a"]),
            "the node has 2 outputs, so choices= must be a mapping by output name",
        ),
    ],
)
def test_a_node_of_several_outputs_asks_by_output_name(ask, reason):
    @node("wrong", title="Wrong", description="For a test.", asks=True)
    def wrong() -> Route:
        raise ask()

    assert _once(wrong)[1].cause.ctx["reason"] == reason


def test_a_default_obeys_what_an_answer_must_and_none_is_no_pick():
    @node("blank", title="Blank", description="For a test.", asks=True)
    def blank() -> str | None:
        raise Ask("?", choices=["a", "b"], default=None)

    @node("twice", title="Twice", description="For a test.", asks=True)
    def twice() -> Series[str]:
        raise Ask("?", choices=["t1", "t2"], default=["t1", "t1"])

    # None counts as no pick. A type that takes None takes it on top of its choices, as an input's Param does.
    (question,) = _once(blank)[1].waiting[0].questions
    assert (question.default, question.required) == (None, True)
    assert (
        _once(twice)[1].cause.ctx["reason"]
        == "the default for 'result' isn't an answer a person could give (a choice is picked twice)"
    )


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
    assert (
        _refused(lambda: run.answer("q", {}))[0].msg
        == "That isn't a valid answer: 'who' has no answer."
    )
    assert run.answer("q", {"who": "a"}).resume().outputs == {"who": "a", "note": None}


@pytest.mark.parametrize(
    ("returns", "wrong", "right"),
    [
        (bool, ["yes", 1, "true"], True),
        (int, ["3", True, 3.0 + 0.5], 3),
        (float, ["nan", True, "1.5"], 1.5),
        (datetime.date, [20261110, "10/11/2026"], "2026-11-10"),
    ],
)
def test_an_answer_is_read_strictly_from_its_json_form(returns, wrong, right):
    _, run = _once(
        node("strict", title="Asks", description="For a test.", asks=True)(_returning(returns))
    )
    for value in wrong:
        assert [f.type for f in _refused(lambda value=value: run.answer("q", value))] == [
            "invalid_answer"
        ], value
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

    assert (
        _once(wrong)[1].cause.ctx["reason"]
        == "the choices for 'result' are wrong (a table must be a frame, or a list of records with one object per row)"
    )


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
    assert (
        called == calls
        and run.to_json() == before
        and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    )
    assert [place(w) for w in answered.waiting] == ["q@2"]


def test_a_run_cant_be_changed_and_a_run_made_from_it_doesnt_share_its_inputs():
    """A ``Run`` is a value: its attributes can't be set, and its ``inputs`` can't be changed in place.

    The methods that move a run on return a new one, and a run that could still be changed would make that rule
    leaky. Setting ``max_items`` would skip the check ``with_max_items`` makes. The run ``answer`` returns would share
    its inputs with the run it was made from, so changing one would change both.
    """
    run = _approve_then_pay().run(text="A:100,B:900")
    answered = run.answer(run.waiting[0], True)
    with pytest.raises(FrozenInstanceError):
        run.max_items = -3  # type: ignore[misc]
    with pytest.raises(TypeError):
        answered.inputs["text"] = "changed"  # type: ignore[index]
    assert (run.max_items, dict(run.inputs)) == (None, {"text": "A:100,B:900"})


def test_a_run_whose_every_question_is_answered_is_complete_when_nothing_is_left_to_call():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="a,b1,b2")
    half = run.answer("q@1", "B1")
    assert (half.status, half.cause) == ("waiting", None)
    assert [p.ctx["status"] for p in _refused(lambda: half.outputs)] == ["waiting"]
    answered = half.answer("q@2", "B2")
    assert (answered.status, answered.waiting, answered.outputs) == (
        "complete",
        [],
        {"result": ["a", "B1", "B2"]},
    )


def test_a_run_whose_every_question_is_answered_is_ready_while_a_node_has_calls_to_make():
    run = _approve_then_pay().run(text="A:100,B:900")
    answered = run.answer(run.waiting[0], True)
    assert (answered.status, answered.waiting) == ("ready", [])
    assert [p.ctx["status"] for p in _refused(lambda: answered.outputs)] == ["ready"]
    assert answered.resume().outputs == {"result": "paid A; paid B"}


def _ask_or_wait_or_fail() -> Any:
    """Returns a node ``q`` that asks when given ``"ask"``, sleeps 60 ms on ``"slow"`` and raises on ``"boom"``."""

    @node("q", title="Q", description="For a test.", asks=True)
    def q(text: str) -> str:
        if text == "ask":
            raise Ask("?")
        if text == "slow":
            time.sleep(0.06)
        if text == "boom":
            raise ValueError("no")
        return text

    return q


def test_a_run_that_timed_out_and_is_answered_is_ready_as_its_outcomes_say_also_read_back():
    graph, _ = _waiting_outcomes(_ask_or_wait_or_fail(), text="ask,slow,slow")
    timed_out = graph.with_inputs().resume(timeout=0.03)
    assert (timed_out.status, timed_out.cause.type) == ("timed_out", "run_timed_out")
    answered = timed_out.answer(timed_out.waiting[0], "A")
    assert (answered.status, answered.cause, answered.waiting) == ("ready", None, [])
    assert Run.from_json(answered.to_json(), graph).status == "ready"
    assert answered.resume().outputs == {"result": ["A", "slow", "slow"]}


def test_an_empty_answer_changes_nothing_so_a_run_that_timed_out_stays_timed_out():
    graph, _ = _waiting_outcomes(_ask_or_wait_or_fail(), text="ask,slow,slow")
    timed_out = graph.with_inputs().resume(timeout=0.03)
    assert timed_out.status == "timed_out"

    answered = timed_out.answer({})

    assert (answered.status, answered.cause) == ("timed_out", timed_out.cause)
    assert answered.to_json() == timed_out.to_json()


def test_a_failed_call_keeps_the_run_failed_after_an_answer_to_another_call():
    graph, run = _waiting_outcomes(_ask_or_wait_or_fail(), text="ask,boom")
    answered = run.answer(run.waiting[0], "A")
    assert (run.status, answered.status, answered.cause) == ("failed", "failed", run.cause)
    assert Run.from_json(answered.to_json(), graph).cause == run.cause


def test_every_fault_of_one_answer_call_is_refused_at_once_and_nothing_is_answered():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="a,b1,b2")
    faults = _refused(lambda: run.answer({"q@1": 1, "q@0": "A", "nope@3": "x", "q@2": "B2"}))
    assert [(f.type, f.loc, f.ctx["waiting_id"]) for f in faults] == [
        ("invalid_answer", ("q", 1), "q@1"),
        ("not_waiting", ("q", 0), "q@0"),
        ("not_waiting", ("nope", 3), "nope@3"),
    ]
    assert [place(w) for w in run.waiting] == ["q@1", "q@2"]


def test_answering_with_ids_that_are_no_waiting_id_is_refused_as_problems():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    faults = _refused(lambda: run.answer({"0": "B1", "q@x": "B", "q": "B", "q@0@1": "B"}))
    assert [(f.type, f.loc) for f in faults] == [
        ("not_waiting", ("0",)),
        ("not_waiting", ("q",)),
        ("not_waiting", ("q",)),
        ("not_waiting", ("q", 0, 1)),
    ]


def test_a_waiting_id_has_one_spelling_and_any_other_names_nothing():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    first = run.waiting[0].id
    other_spellings = ["q@01", "q@+1", "q@ 1", "q@-0", "q@0~", f"/{first}"]
    faults = _refused(lambda: run.answer(dict.fromkeys(other_spellings, "B")))
    assert [f.type for f in faults] == ["not_waiting"] * len(other_spellings)
    assert (
        run.answer({"q@1": "B", first: "A"}).waiting == []
    )  # An id is found with or without its fingerprint.


def test_answer_takes_one_waiting_call_and_its_value_or_a_mapping_of_several():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    assert (
        run.answer(run.waiting[0], "1").answer("q@1", "2").to_json()
        == run.answer({"q@0": "1", "q@1": "2"}).to_json()
    )
    with pytest.raises(TypeError):
        run.answer("q@0")  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        run.answer({"q@0": "1"}, "2")  # type: ignore[call-overload]


def test_an_answer_is_given_once():
    gate, _ = _counted()
    _, run = _waiting_outcomes(gate, text="b")
    answered = run.answer("q@0", "B")
    assert [f.type for f in _refused(lambda: answered.answer("q@0", "B"))] == ["not_waiting"]


def test_a_value_no_question_can_read_is_refused_as_a_problem_never_an_exception_of_its_own():
    _, run = _waiting_outcomes(review)
    for value in (
        object(),
        {"approved": object()},
        {"approved": True, "amount": float("nan"), "due": object()},
    ):
        assert {f.type for f in _refused(lambda value=value: run.answer("q@0", value))} == {
            "invalid_answer"
        }
    graph = _lines_graph()
    for value in ([object()], [{"sku": object(), "qty": 1}], pl.DataFrame({"sku": [1]})):
        assert [f.type for f in _refused(lambda value=value: graph.run().answer("q", value))] == [
            "invalid_answer"
        ]


def test_after_an_answer_resume_calls_only_what_waited_on_it_and_an_item_left_unanswered_waits_on():
    gate, called = _counted()
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"constant": "a,b1,b2"}}},
                    "q": {
                        "node": "gate",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        registry_of(split, gate, upper),
    )
    run = graph.run()
    called.clear()
    resumed = Run.from_json(run.answer("q@1", "yes").to_json(), graph).resume()
    assert (
        called == []
        and resumed.status == "waiting"
        and [place(w) for w in resumed.waiting] == ["q@2"]
    )
    assert resumed.results["u"]["result"] == ["A", "YES", None]
    assert resumed.waiting[0] == run.waiting[1]  # The question is the same as when it was asked.


def test_a_node_that_reads_the_world_is_not_called_again_when_a_run_goes_on_from_waiting():
    read: list[str] = []

    @node(
        "now", title="Now", description="For a test.", policy=Policy(reuse_when_inputs_match=False)
    )
    def now(text: str) -> str:
        read.append(text)
        return text

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(text: str) -> str:
        raise Ask("?")

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "n": {"node": "now", "inputs": {"text": {"constant": "x"}}},
                    "q": {
                        "node": "ask",
                        "inputs": {"text": {"sources": [{"node": "n", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(now, ask),
    )
    graph.run().answer("q", "y").resume()
    assert read == ["x"]


def test_an_item_skipped_above_never_asks_and_the_others_complete_when_answered():
    @node("only-b", title="Only b", description="For a test.")
    def only_b(text: str) -> str:
        return text if text.startswith("b") else SKIPPED

    gate, called = _counted()
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"constant": "a,b1,b2"}}},
                    "o": {
                        "node": "only-b",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "q": {
                        "node": "gate",
                        "inputs": {"text": {"sources": [{"node": "o", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(split, only_b, gate),
    )
    run = graph.run()
    assert called == ["b1", "b2"] and [place(w) for w in run.waiting] == ["q@1", "q@2"]
    assert run.answer({"q@1": "1", "q@2": "2"}).resume().results["q"]["result"] == [None, "1", "2"]


def test_waiting_calls_are_listed_in_node_then_item_order_whatever_order_they_asked_in():
    @node(
        "slow-ask",
        title="Slow ask",
        description="For a test.",
        policy=Policy(concurrency=4),
        asks=True,
    )
    def slow_ask(text: str) -> str:
        time.sleep(0.02 * (3 - int(text)))
        raise Ask(text)

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(text: str) -> str:
        raise Ask(text)

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"constant": "0,1,2"}}},
                    "a": {
                        "node": "slow-ask",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "b": {
                        "node": "ask",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
                "outputs": {
                    "a_result": {"node": "a", "output": "result"},
                    "b_result": {"node": "b", "output": "result"},
                },
            }
        ),
        registry_of(split, slow_ask, ask),
    )
    run = graph.run()
    assert [place(w) for w in run.waiting] == ["a@0", "a@1", "a@2", "b@0", "b@1", "b@2"]
    assert [place(w) for w in Run.from_json(run.to_json(), graph).waiting] == [
        place(w) for w in run.waiting
    ]


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

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "r": {"node": "rows"},
                    "i": {
                        "node": "records",
                        "inputs": {"table": {"sources": [{"node": "r", "output": "result"}]}},
                    },
                    "q": {
                        "node": "look",
                        "inputs": {
                            "names": {
                                "sources": [{"node": "i", "output": "result", "field": "name"}]
                            },
                            "table": {"sources": [{"node": "r", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(rows, records, look),
    )
    run = graph.run()
    assert run.waiting[0].inputs == {
        "names": ["Ida", "Bo"],
        "table": {"columns": [{"name": "name", "type": "str", "values": ["Ida", "Bo"]}]},
    }
    assert Run.from_json(run.to_json(), graph).waiting == run.waiting


def test_a_waiting_id_is_the_same_when_a_store_gives_the_saved_inputs_back_with_their_keys_in_another_order():
    @node("check", title="Check", description="For a test.", asks=True)
    def check(data: dict[str, int]) -> bool:
        raise Ask("Right?")

    record = {
        "nodes": {"q": {"node": "check", "inputs": {"data": {"graph_input": "data"}}}},
        "outputs": _outputs_of(check),
    }
    graph = Graph.compile(GraphRecord.model_validate(record), registry_of(check))
    run = graph.run(data={"a": 1, "b": 2})
    saved = json.loads(run.to_json())
    saved["inputs"]["data"] = {"b": 2, "a": 1}
    loaded = Run.from_json(json.dumps(saved), graph)
    assert [w.id for w in loaded.waiting] == [w.id for w in run.waiting]
    assert loaded.answer(run.waiting[0].id, True).resume().outputs == {"result": True}


def test_items_made_of_what_a_caller_gives_pause_save_answer_and_finish():
    gate, _ = _counted()
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
                    "q": {
                        "node": "gate",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(split, gate),
    )
    run = Run.from_json(graph.run(text="a,b").to_json(), graph)
    assert [place(w) for w in run.waiting] == ["q@1"]
    assert Run.from_json(run.answer("q@1", "B").to_json(), graph).resume().outputs == {
        "result": ["a", "B"]
    }


def test_an_answer_names_its_item_by_index_so_one_for_an_item_that_no_longer_waits_is_refused():
    gate, _ = _counted()
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
                    "q": {
                        "node": "gate",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "q", "output": "result"}},
            }
        ),
        registry_of(split, gate),
    )
    run = graph.run(text="b1,a")
    moved = run.again(text="a,b1")  # The same question, now on item 1.
    assert [place(w) for w in run.waiting] == ["q@0"] and [place(w) for w in moved.waiting] == [
        "q@1"
    ]
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


def test_a_saved_waiting_call_with_its_question_taken_out_is_loaded_as_a_call_not_made_yet():
    """A saved call without its question is loaded as a call not made yet.

    A saved call counts as waiting only because its question is saved with it. Take the question out, and the run
    calls the node again when it resumes.
    """
    gate, _ = _counted()
    graph, run = _waiting_outcomes(gate, text="a,b")
    saved = json.loads(run.to_json())
    assert saved["outcomes"]["q"]["states"] == ["done", None] and list(
        saved["outcomes"]["q"]["questions"]
    ) == ["1"]
    del saved["outcomes"]["q"]["questions"]
    loaded = Run.from_json(json.dumps(saved), graph)
    assert loaded.waiting == []
    assert [place(w) for w in loaded.resume().waiting] == [
        "q@1"
    ]  # The node is called again, and asks again.


def test_a_question_record_is_json_a_host_draws_from_with_no_graph_loaded():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> Annotated[str, Param(title="Assignee", widget=Textarea())]:
        raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B.")], default="ida")

    _, run = _once(assign, ticket="T-1")
    written = run.waiting[0].model_dump(mode="json")
    assert re.fullmatch(
        r"q~[0-9a-f]{8}", written.pop("id")
    )  # Where the call is, then a fingerprint of its question.
    assert written == {
        "loc": ["q"],
        "prompt": "Who takes T-1?",
        "inputs": {"ticket": "T-1"},
        "questions": [
            {
                "name": "result",
                "title": "Assignee",
                "is_series": False,
                "required": True,
                "type": "str",
                "type_variables": [],
                "json_schema": {"type": "string", "enum": ["ida"]},
                "loops": [],
                "choices": [{"id": "ida", "title": "Ida B."}],
                "default": "ida",
                "widget": {"type": "textarea", "rows": 4},
                "description": None,
                "constant_only": False,
            }
        ],
    }
    # A host that stored the record reads it back, though its one question is required and has a default.
    assert Waiting.model_validate_json(run.waiting[0].model_dump_json()) == run.waiting[0]


def test_a_literal_says_the_type_of_its_values_on_an_input_and_on_a_question():
    @node("priority", title="Priority", description="For a test.", asks=True)
    def priority(level: Literal["a", "b"], ticket: str) -> Literal["low", "high"]:
        raise Ask(f"How urgent is {ticket}?")

    assert [field.type for field in priority.describe().inputs] == ["str", "str"]
    run = Graph.compile(
        make_graph(priority, constants={"priority.level": "a"}), registry_of(priority)
    ).run(ticket="T-1")
    (question,) = run.waiting[0].questions
    assert question.type == "str" and question.json_schema == {
        "enum": ["low", "high"],
        "type": "string",
    }


@pytest.mark.parametrize(
    ("hint", "asked", "enum"),
    [
        (Annotated[str, Param(choices=["ida", "bo"])], None, ["ida", "bo"]),
        (str, ["ida", "bo"], ["ida", "bo"]),
        (Literal["ida", "bo"], ["ida"], ["ida"]),
        (Literal["ida", "bo"] | None, ["ida"], ["ida", None]),
        (Annotated[str | None, Param(choices=["ida", "bo"])], None, ["ida", "bo", None]),
        (Series[Literal["ida", "bo"]], ["bo"], ["bo"]),
    ],
)
def test_a_questions_schema_lists_its_choices_as_its_enum(hint: Any, asked: Any, enum: Any):
    """A host that draws a form from a question's JSON schema offers the values its ``choices`` list.

    The ``enum`` is written from the choices: the type's own, or the ``Ask``'s when it narrows them. ``None`` is in
    it when the type takes ``None``.
    """

    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> hint:  # type: ignore[valid-type]
        raise Ask(f"Who takes {ticket}?", choices=asked)

    _, run = _once(assign, ticket="T-1")
    (question,) = run.waiting[0].questions
    assert question.json_schema["enum"] == enum
    assert [choice.id for choice in question.choices] == [i for i in enum if i is not None]
    if asked is None:
        assert assign.describe().outputs[0].json_schema == question.json_schema


class _Chosen(TypedDict):
    who: Annotated[str, Param(choices=["ida", "bo"])]


def _column(schema: dict[str, Any], name: str) -> dict[str, Any]:
    (column,) = [
        c
        for c in schema["properties"]["columns"]["prefixItems"]
        if c["properties"]["name"]["const"] == name
    ]
    return column["properties"]["values"]["items"]


@pytest.mark.parametrize(
    ("hint", "enum_of"),
    [
        (
            list[Annotated[str, Param(choices=["ida", "bo"])]],
            lambda schema: schema["items"]["enum"],
        ),
        (_Chosen, lambda schema: schema["properties"]["who"]["enum"]),
        (Table[_Chosen], lambda schema: _column(schema, "who")["enum"]),
    ],
)
def test_choices_below_the_top_are_their_schemas_enum_too(hint: Any, enum_of: Any):
    """A ``Param(choices=)`` on a list's item, a record's field or a table's column is that part's ``enum``.

    So a form drawn from the schema offers only what the run takes, at any depth.
    """

    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(ticket: str) -> hint:  # type: ignore[valid-type]
        raise Ask(f"Who takes {ticket}?")

    (output,) = assign.describe().outputs
    assert enum_of(output.json_schema) == ["ida", "bo"]


@host_type("shade", title="Shade")
class _Shade(Enum):
    RED = "red"
    BLUE = "blue"


def test_an_enum_is_described_asked_and_answered_by_its_values():
    """An enum's members are its choices, titled by their values, as a ``Literal``'s values are.

    The question lists them in its schema's ``enum``, and an answer ``"red"`` is ``Shade.RED`` and reads back.
    """

    @node("shade", title="Shade", description="For a test.", asks=True)
    def shade(ticket: str) -> _Shade:
        raise Ask(f"Shade {ticket}?")

    (output,) = shade.describe().outputs
    graph, run = _once(shade, ticket="T-1")
    (question,) = run.waiting[0].questions
    answered = run.answer(run.waiting[0], "red")
    answered.resume()

    assert [(choice.id, choice.title) for choice in output.choices] == [
        ("red", "red"),
        ("blue", "blue"),
    ]
    assert [choice.id for choice in question.choices] == ["red", "blue"]
    assert question.json_schema["enum"] == ["red", "blue"]
    assert answered.outputs == {"result": _Shade.RED}
    assert Run.from_json(answered.to_json(), graph).outputs == {"result": _Shade.RED}


def test_an_ask_narrows_an_enum_and_param_titles_its_members():
    """``Ask(choices=)`` offers some of an enum's members, and ``Param(choices=)`` gives them titles."""

    @node("shade", title="Shade", description="For a test.", asks=True)
    def shade(
        ticket: str,
    ) -> Annotated[_Shade, Param(choices=[Choice(_Shade.RED, "Red"), Choice("blue", "Blue")])]:
        raise Ask(f"Shade {ticket}?", choices=[_Shade.RED])

    (output,) = shade.describe().outputs
    _, run = _once(shade, ticket="T-1")
    (question,) = run.waiting[0].questions

    assert [(choice.id, choice.title) for choice in output.choices] == [
        ("red", "Red"),
        ("blue", "Blue"),
    ]
    assert [(choice.id, choice.title) for choice in question.choices] == [("red", "Red")]
    assert question.json_schema["enum"] == ["red"]


@host_type("weight", title="Weight")
class _Weight(Enum):
    HALF = 0.5
    ONE = 1.0


def test_an_answer_to_an_enum_is_read_as_the_enum_reads_it():
    """JavaScript writes ``1.0`` as ``1``, which ``Weight`` reads as ``Weight.ONE``."""

    @node("weigh", title="Weigh", description="For a test.", asks=True)
    def weigh(ticket: str) -> _Weight:
        raise Ask(f"Weigh {ticket}?")

    _, run = _once(weigh, ticket="T-1")
    answered = run.answer(run.waiting[0], 1)
    answered.resume()

    assert answered.outputs == {"result": _Weight.ONE}


def _shouted(text: str) -> str:
    return text + "!"


def test_an_ask_narrowing_a_type_whose_check_converts_shows_its_question():
    """The choices are read through the check once. Read again, ``"x"`` would become ``"x!!"``, refused by its own
    narrowing, and ``run.waiting`` would raise for every waiting call of the run."""

    @node("bang", title="Bang", description="For a test.", asks=True)
    def bang(ticket: str) -> Annotated[str, AfterValidator(_shouted)]:
        raise Ask(f"Bang {ticket}?", choices=["x"])

    _, run = _once(bang, ticket="T-1")
    (question,) = run.waiting[0].questions

    assert [(choice.id, choice.title) for choice in question.choices] == [("x!", "x")]
    with pytest.raises(InvalidCallError) as raised:
        run.answer(run.waiting[0], "x!")
    assert raised.value.problems[0].type == "invalid_answer"


def test_a_choice_that_is_not_finite_leaves_the_schema_with_no_enum():
    """JSON Schema would read ``inf``'s ``"Infinity"`` as text, which a number refuses, so there is no ``enum``.

    A schema may say less than the run checks, but never list a value the run refuses.
    """

    @node("cap", title="Cap", description="For a test.", asks=True)
    def cap(ticket: str) -> Annotated[float, Param(choices=[1.0, float("inf")])]:
        raise Ask(f"Cap {ticket}")

    asked_inf = node("cap-asked", title="Cap", description="For a test.", asks=True)(
        _ask_cap_with_inf
    )
    for definition in (cap, asked_inf):
        _, run = _once(definition, ticket="T-1")
        assert run.waiting[0].questions[0].json_schema == {"type": "number"}


def _ask_cap_with_inf(ticket: str) -> float:
    raise Ask(f"Cap {ticket}", choices=[1.5, float("inf")])


def test_an_ask_that_offers_some_of_a_types_titled_choices_by_id_keeps_their_titles():
    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign(
        ticket: str,
    ) -> Annotated[
        Literal["ida", "bo"], Param(choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])
    ]:
        raise Ask(f"Who takes {ticket}?", choices=["bo"])

    _, run = _once(assign, ticket="T-1")
    assert run.waiting[0].questions[0].choices == (Choice("bo", "Bo K."),)


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
    assert [(f.type, f.loc) for f in _refused(lambda: run.answer("once@banana", "y"))] == [
        ("not_waiting", ("once",))
    ]
    assert run.answer("once", "y").outputs == {
        "result": "y"
    }  # Answered, and there is no call left to make.


async def test_a_cancelled_run_takes_an_answer_to_what_waits_in_it_as_every_run_that_ended_early_does():
    from asyncio import Event

    gate, called = _counted()
    _, run = _waiting_outcomes(gate, text="b1,b2")
    cancel = Event()
    cancel.set()
    async for event in run.stream(cancel=cancel):
        if event.type == "run_end":
            cancelled = event.run
    assert cancelled.status == "cancelled" and [place(w) for w in cancelled.waiting] == [
        "q@0",
        "q@1",
    ]

    answered = cancelled.answer("q@0", "B1")
    assert (answered.status, answered.cause) == (
        "waiting",
        None,
    )  # The cancel ended that stream only, so the answered run has no cause. q@1 still waits.
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
    registry.add(
        ask,
        ask2,
        subgraph("both", title="Both", description="For a test.")(
            GraphRecord.model_validate(
                {
                    "nodes": {
                        "q": {"node": "ask", "inputs": {"text": {"graph_input": "first"}}},
                        "q2": {"node": "ask2", "inputs": {"text": {"graph_input": "second"}}},
                    },
                    "outputs": {
                        "first": {"node": "q", "output": "result"},
                        "second": {"node": "q2", "output": "result"},
                    },
                }
            )
        ),
    )
    run = Graph.compile(
        make_graph(registry["both"], constants={"both.first": "a", "both.second": "b"}), registry
    ).run()
    assert [
        (w.id.partition("~")[0], w.loc, [q.name for q in w.questions]) for w in run.waiting
    ] == [("both/q", ("both", "q"), ["result"]), ("both/q2", ("both", "q2"), ["result"])]
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
    registry.add(
        form,
        subgraph("both", title="Both", description="For a test.")(
            GraphRecord.model_validate(
                {
                    "nodes": {
                        "z1": {"node": "form", "inputs": {"t": {"graph_input": "z1_t"}}},
                        "z2": {"node": "form", "inputs": {"t": {"graph_input": "z2_t"}}},
                    },
                    "outputs": {
                        "z1_a": {"node": "z1", "output": "a"},
                        "z1_b": {"node": "z1", "output": "b"},
                        "z2_a": {"node": "z2", "output": "a"},
                        "z2_b": {"node": "z2", "output": "b"},
                    },
                }
            )
        ),
    )
    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "e": {
                        "node": "both",
                        "inputs": {
                            "z1_t": {"graph_input": "z1_t"},
                            "z2_t": {"graph_input": "z2_t"},
                        },
                    }
                },
                "outputs": {
                    "z1_a": {"node": "e", "output": "z1_a"},
                    "z1_b": {"node": "e", "output": "z1_b"},
                    "z2_a": {"node": "e", "output": "z2_a"},
                    "z2_b": {"node": "e", "output": "z2_b"},
                },
            }
        ),
        registry,
    ).run(z1_t="1", z2_t="2")
    answers = {
        w.id: {q.name: "A" if q.name == "a" else 5 for q in w.questions} for w in run.waiting
    }
    assert {w.loc: sorted(answers[w.id]) for w in run.waiting} == {
        ("e", "z1"): ["a", "b"],
        ("e", "z2"): ["a", "b"],
    }
    assert run.answer(answers).resume().results == {
        "e": {"z1_a": "A", "z1_b": 5, "z2_a": "A", "z2_b": 5}
    }
    (wrong,) = _refused(lambda: run.answer({run.waiting[0].id: {"a": 3}}))
    assert (wrong.type, wrong.loc, wrong.ctx["question"]) == ("invalid_answer", ("e", "z1"), "a")


def test_an_answer_to_a_run_with_nothing_waiting_says_the_runs_status():
    @node("plain", title="Plain", description="For a test.")
    def plain(text: str) -> str:
        return text

    run = Graph.compile(make_graph(plain), registry_of(plain)).run(text="a")
    (problem,) = _refused(lambda: run.answer("plain", "b"))
    assert (problem.type, problem.ctx["status"]) == (
        "not_waiting",
        "complete",
    ) and "The run is complete" in problem.msg


@pytest.mark.parametrize(
    ("choices", "reason"),
    [
        ([True, 2], "a yes or no is not a number"),
        (["1"], "Input should be a valid integer"),
    ],
    ids=["a bool", "a number as text"],
)
def test_a_choice_is_read_as_its_answer_would_be_so_one_a_number_refuses_fails_the_node(
    choices: list[object], reason: str
) -> None:
    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> int:
        raise Ask("?", choices=choices)

    _, run = _once(pick)
    assert (run.status, run.cause.type) == ("failed", "invalid_ask")
    assert run.cause.ctx["reason"] == f"the choices for 'result' are wrong ({reason})"


def test_an_asks_choice_of_a_yes_or_no_is_refused_on_a_literal_of_numbers_as_an_answer_is() -> None:
    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick() -> Literal[1, 2]:
        raise Ask("?", choices=[True])

    cause = _once(pick)[1].cause
    assert (cause.type, cause.ctx["reason"]) == (
        "invalid_ask",
        "the choices for 'result' are wrong (a yes or no is not a number)",
    )


def test_a_model_that_isnt_a_choice_is_a_value_to_param_and_to_ask_alike() -> None:
    from pydantic import BaseModel

    class Option(BaseModel):  # A host's own option class, not a Choice.
        id: str
        title: str

    ida = Option(id="ida", title="Ida B.")
    with pytest.raises(NodeDeclarationError) as declared:

        @node("assign_by_param", title="Assign by param", description="For a test.")
        def assign_by_param(who: Annotated[str, Param(choices=[ida])]) -> str:
            return who

    assert declared.value.problems[0].type == "choices_invalid"

    @node("assign", title="Assign", description="For a test.", asks=True)
    def assign() -> str:
        raise Ask("Who?", choices=[ida])

    _, run = _once(assign)
    assert (run.status, run.cause.type) == ("failed", "invalid_ask")
    assert (
        run.cause.ctx["reason"]
        == "the choices for 'result' are wrong (Input should be a valid string)"
    )


def test_a_lone_output_is_asked_under_the_title_describe_shows_for_it():
    @node("reason", title="Closing reason", description="Asks why a ticket was closed.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why was it closed?")

    graph = Graph.compile(make_graph(reason), registry_of(reason))
    (output,) = graph.describe().nodes["reason"].outputs
    (waiting,) = graph.run(ticket="t").waiting

    assert waiting.questions[0].title is output.title is None


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
        assert run.answer(waiting, shown[0].id).resume().outputs == {"result": read}


def test_an_ask_pickled_as_another_process_raises_it_still_proposes_no_default() -> None:
    import pickle

    @node("who", title="Who", description="Asks who takes a ticket.", asks=True)
    def who(ticket: str) -> str:
        raise pickle.loads(
            pickle.dumps(Ask("Who takes it?"))
        )  # As if the node's body ran in another process.

    _, run = _once(who, ticket="T-1")
    assert run.status == "waiting" and run.waiting[0].questions[0].required


# -- a node that asks, and a node that gets the answer through an edge -----------------------------------
#
# "Approve, then pay" is two nodes. ``approval(invoice) -> bool`` returns ``True`` for a small invoice and asks a
# person about a large one. ``pay(invoice, ok: bool)`` reads its ``result``. A host that wants to offer the two as one
# node in its palette puts them in a ``subgraph``.


class Invoice(TypedDict):
    number: str
    amount: int


calls_made: list[tuple[Any, ...]] = []


@node("invoices", title="Invoices", description="For a test.")
def invoices(text: str) -> Series[Invoice]:
    found = [line.split(":") for line in text.split(",") if line]
    return [Invoice(number=number, amount=int(amount)) for number, amount in found]


@node("approval", title="Approval", description="For a test.", asks=True)
def approval(invoice: Invoice) -> bool:
    calls_made.append(("approval", invoice["number"]))
    if invoice["amount"] <= 500:
        return True
    raise Ask(f"Pay {invoice['number']}?")


@node("pay", title="Pay", description="For a test.")
def pay(invoice: Invoice, ok: bool) -> str:
    calls_made.append(("pay", invoice["number"], ok))
    return f"{'paid' if ok else 'held'} {invoice['number']}"


@node("summary", title="Summary", description="For a test.")
def summary(lines: Series[str]) -> str:
    return "; ".join(lines)


_EDGES = [
    ("invoices.result", "approval.invoice"),
    ("invoices.result", "pay.invoice"),
    ("approval.result", "pay.ok"),
    ("pay.result", "summary.lines"),
]


def _approve_then_pay(paying: Any = pay) -> Graph:
    return Graph.compile(
        make_graph(invoices, approval, paying, summary, edges=_EDGES),
        registry_of(invoices, approval, paying, summary),
    )


@pytest.fixture(autouse=True)
def _no_calls() -> None:
    calls_made.clear()


# -- a node that asks says so ------------------------------------------------------------------------


def test_a_raise_of_ask_from_a_node_that_does_not_declare_asks_fails_the_run_naming_the_node():
    @node("reason", title="Reason", description="For a test.")
    def reason(ticket: str) -> str:
        raise Ask(f"Why was {ticket} closed?")

    run = Graph.compile(make_graph(reason), registry_of(reason)).run(ticket="T-1")
    assert run.status == "failed"
    assert (run.cause.type, run.cause.loc) == ("ask_not_declared", ("reason",))
    assert "asks=True" in run.cause.msg


def test_an_undeclared_ask_on_a_thread_fails_the_same_way():
    @node("reason", title="Reason", description="For a test.", policy=Policy(concurrency=2))
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    run = Graph.compile(make_graph(reason), registry_of(reason)).run(ticket="T-1")
    assert run.cause.type == "ask_not_declared"


def test_describe_says_whether_a_node_asks_and_a_registry_node_takes_asks_as_node_does():
    registry = NodeRegistry()

    @registry.node("reason", title="Reason", description="For a test.", asks=True)
    def reason(ticket: str) -> str:
        raise Ask("Why?")

    assert (registry["reason"].describe().asks, approval.describe().asks, pay.describe().asks) == (
        True,
        True,
        False,
    )


def test_a_subgraph_asks_when_a_node_inside_it_asks_however_deep():
    registry = registry_of(approval, pay)
    inner = registry.subgraph("inner", title="Inner", description="For a test.")(
        make_graph(approval)
    )
    registry.subgraph("outer", title="Outer", description="For a test.")(make_graph(inner))
    registry.subgraph("pays", title="Pays", description="For a test.")(make_graph(pay))
    asks = {entry.id: entry.asks for entry in registry.describe().nodes}
    assert asks == {"approval": True, "pay": False, "inner": True, "outer": True, "pays": False}


# -- approve, then pay -------------------------------------------------------------------------------


def test_approval_then_pay_waits_only_when_approval_asks():
    run = _approve_then_pay().run(text="A:100,B:200")
    assert run.outputs == {"result": "paid A; paid B"}
    assert calls_made == [
        ("approval", "A"),
        ("approval", "B"),
        ("pay", "A", True),
        ("pay", "B", True),
    ]

    calls_made.clear()
    run = _approve_then_pay().run(text="A:100,B:900")
    (waiting,) = run.waiting
    assert waiting.prompt == "Pay B?" and waiting.id.startswith("approval@1~")
    assert [(question.name, question.type) for question in waiting.questions] == [
        ("result", "bool")
    ]
    assert waiting.inputs == {"invoice": {"number": "B", "amount": 900}}
    assert ("pay", "A", True) in calls_made and not any(
        call[:2] == ("pay", "B") for call in calls_made
    )


def test_an_answer_resumes_pay_with_ok_and_asks_nothing_again():
    run = _approve_then_pay().run(text="A:100,B:900")
    calls_made.clear()
    done = run.answer(run.waiting[0], False).resume()
    assert done.outputs == {"result": "paid A; held B"}
    assert calls_made == [("pay", "B", False)]


def test_an_answer_saved_before_resume_and_read_back_in_another_process_resumes_pay_with_it():
    graph = _approve_then_pay()
    run = Run.from_json(graph.run(text="B:900").to_json(), graph)
    answered = Run.from_json(run.answer(run.waiting[0], True).to_json(), graph)
    assert answered.waiting == []
    calls_made.clear()
    assert answered.resume().outputs == {"result": "paid B"}
    assert calls_made == [("pay", "B", True)]


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

    graph = _approve_then_pay(flaky_pay)
    run = graph.run(text="B:900")
    failed = run.answer(run.waiting[0], True).resume()
    assert failed.status == "failed" and attempts == [True, True]
    calls_made.clear()
    assert Run.from_json(failed.to_json(), graph).resume().outputs == {"result": "paid B"}
    assert attempts == [True, True, True] and calls_made == []


# -- declining a question ------------------------------------------------------------------------------


def test_a_declined_question_fails_the_run_with_the_reason_in_ctx_and_calls_nothing_after_it():
    run = _approve_then_pay().run(text="A:100,B:900")
    calls_made.clear()
    declined = run.decline(run.waiting[0], reason="Over budget")
    assert (declined.status, declined.waiting) == ("failed", [])
    assert (declined.cause.type, declined.cause.loc, declined.cause.ctx) == (
        "answer_declined",
        ("approval", 1),
        {"reason": "Over budget"},
    )
    assert "Over budget" not in declined.cause.msg
    assert declined.results["pay"]["result"] == ["paid A", None]
    assert run.status == "waiting"  # The run declined is left as it was.
    assert calls_made == []


def test_resume_after_a_decline_asks_again_under_the_same_id():
    run = _approve_then_pay().run(text="B:900")
    declined = run.decline(run.waiting[0].id)
    assert declined.cause.ctx.get("reason") is None
    calls_made.clear()
    again = declined.resume()
    assert (again.status, again.cause, [w.id for w in again.waiting]) == (
        "waiting",
        None,
        [run.waiting[0].id],
    )
    assert calls_made == [("approval", "B")]
    assert again.answer(again.waiting[0], True).resume().outputs == {"result": "paid B"}


def test_a_declined_run_saved_and_read_back_is_failed_with_the_same_cause_and_asks_again():
    graph = _approve_then_pay()
    run = graph.run(text="B:900")
    declined = run.decline(run.waiting[0], reason="Over budget")
    read = Run.from_json(declined.to_json(), graph)
    assert (read.status, read.cause, read.waiting) == ("failed", declined.cause, [])
    assert [w.id for w in read.resume().waiting] == [run.waiting[0].id]


def test_a_decline_leaves_the_other_questions_waiting_and_an_answer_to_them_keeps_the_run_failed():
    run = _approve_then_pay().run(text="A:900,B:900")
    first, second = run.waiting
    declined = run.decline(first)
    assert [w.id for w in declined.waiting] == [second.id]
    answered = declined.answer(second, True)
    assert (answered.status, answered.cause.type) == ("failed", "answer_declined")
    assert [w.id for w in answered.resume().waiting] == [first.id]


def test_a_decline_names_its_call_by_the_rules_an_answer_does():
    graph = _approve_then_pay()
    run = graph.run(text="A:900")
    (waiting,) = run.waiting
    declined = run.decline(waiting)
    (twice,) = _refused(lambda: declined.decline(waiting))
    (nothing,) = _refused(lambda: run.decline("pay@0"))
    assert (twice.type, twice.ctx["status"], nothing.type) == (
        "not_waiting",
        "failed",
        "not_waiting",
    )
    moved = graph.with_inputs(text="A:950", previous=run).resume()
    (stale,) = _refused(lambda: moved.decline(waiting))
    assert (stale.type, stale.ctx["current"]) == ("answer_stale", moved.waiting[0].id)


def test_a_decline_whose_reason_isnt_text_is_refused_and_the_run_is_left_as_it_was():
    """A reason that isn't a ``str`` JSON can write would be saved as some other JSON, or not at all, so it is refused.

    A lone surrogate is a ``str``, but JSON can't write it, so the run would fail when it is saved.
    """
    run = _approve_then_pay().run(text="A:900")
    (waiting,) = run.waiting
    for reason in (123, object(), "\ud800"):
        (problem,) = _refused(lambda reason=reason: run.decline(waiting, reason=reason))
        assert (problem.type, problem.loc, problem.ctx) == (
            "invalid_answer",
            ("approval", 0),
            {"variant": "reason", "waiting_id": waiting.id, "got": type(reason).__name__},
        )
    assert (run.status, [w.id for w in run.waiting]) == ("waiting", [waiting.id])


# -- answers queued to give a run later ----------------------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    name: str
    count: int


@node("pair", title="Pair", description="For a test.", asks=True)
def _ask_for_a_pair(x: int) -> _Pair:
    raise Ask("Which pair?")


def test_answers_and_declines_queued_for_later_are_refused_as_the_run_refuses_them():
    run = Graph.compile(make_graph(_ask_for_a_pair), registry_of(_ask_for_a_pair)).run(x=1)
    (waiting,) = run.waiting
    for answer in (
        "\ud800",
        {"name": "\ud800", "count": 1},
        _Pair(name="\ud800", count=1),
        b"\xff",
    ):
        queued = _refused(lambda answer=answer: Run.queued_as_json({waiting.id: answer}))
        assert queued == _refused(lambda answer=answer: run.answer(waiting, answer)), answer
    # A mapping is answered by question name, so each question whose value JSON can't write is named.
    (named,) = _refused(lambda: run.answer(waiting, {"name": "\ud800", "count": 1}))
    assert (named.ctx["question"], named.ctx["reason"]) == (
        "name",
        "'name': the value can't be written as JSON",
    )
    queued = _refused(lambda: Run.queued_as_json(declines={waiting.id: "\ud800"}))
    assert queued == _refused(lambda: run.decline(waiting, reason="\ud800"))
    # An id that names no call is refused at the call it names, as not_waiting would name it.
    (problem,) = _refused(lambda: Run.queued_as_json({"approval@x/pay": object()}))
    assert (problem.type, problem.loc) == ("invalid_answer", ("approval",))


def test_an_answer_queued_for_later_is_read_by_the_run_as_the_answer_itself():
    run = Graph.compile(make_graph(_ask_for_a_pair), registry_of(_ask_for_a_pair)).run(x=1)
    (waiting,) = run.waiting
    for answer in (_Pair(name="bolt", count=3), {"name": "bolt", "count": 3}, {"name": 3}):
        queued = Run.queued_as_json({waiting.id: answer})[0][waiting.id]
        direct, later = (
            _outcome(lambda value=value: run.answer(waiting, value)) for value in (answer, queued)
        )
        assert direct == later, answer
    answers, reasons = Run.queued_as_json(
        {waiting.id: {"name": "bolt", "count": 3}, "pair~0": datetime.date(2026, 11, 15)},
        declines={waiting.id: "Over budget"},
    )
    assert (
        json.loads(json.dumps(answers))
        == answers
        == {waiting.id: {"name": "bolt", "count": 3}, "pair~0": "2026-11-15"}
    )
    assert reasons == {waiting.id: "Over budget"}


def _outcome(answer: Any) -> Any:
    """What a run does with an answer: the outputs once resumed, or the problems it refuses the answer with."""
    try:
        return answer().resume().outputs
    except InvalidCallError as refused:
        return refused.problems


# -- with_inputs reuses the answer ---------------------------------------------------------------------


def test_with_inputs_reuses_the_answer_and_asks_only_for_the_invoice_that_changed():
    graph = _approve_then_pay()
    run = graph.run(text="A:900,B:900")
    old_a, old_b = (waiting.id for waiting in run.waiting)
    run = run.answer({old_a: True, old_b: False}).resume()
    calls_made.clear()
    again = graph.with_inputs(text="A:900,B:950", previous=run).resume()
    assert [waiting.prompt for waiting in again.waiting] == ["Pay B?"]
    assert calls_made == [("approval", "B")]  # pay for A is reused, its arguments unchanged.
    (problem,) = _refused(lambda: again.answer(old_b, True))
    assert problem.type == "answer_stale"


def test_two_identical_invoices_answered_differently_keep_their_answers_when_one_is_put_in_front():
    graph = _approve_then_pay()
    run = graph.run(text="A:900,A:900")
    first, second = run.waiting
    run = run.answer({first.id: True, second.id: False}).resume()
    assert run.outputs == {"result": "paid A; held A"}
    calls_made.clear()
    again = graph.with_inputs(text="C:100,A:900,A:900", previous=run).resume()
    assert again.outputs == {"result": "paid C; paid A; held A"}
    assert calls_made == [("approval", "C"), ("pay", "C", True)]


def test_a_saved_run_keeps_two_identical_invoices_answered_differently_apart():
    graph = _approve_then_pay()
    run = graph.run(text="A:900,A:900")
    first, second = run.waiting
    saved = Run.from_json(run.answer({first.id: True, second.id: False}).resume().to_json(), graph)
    again = graph.with_inputs(text="C:100,A:900,A:900", previous=saved).resume()
    assert again.outputs == {"result": "paid C; paid A; held A"}


def test_an_answer_given_and_not_yet_resumed_is_reused_by_a_run_of_changed_inputs():
    graph = _approve_then_pay()
    run = graph.run(text="A:900")
    answered = run.answer(run.waiting[0], True)
    moved = graph.with_inputs(text="A:900,B:100", previous=answered).resume()
    assert moved.outputs == {"result": "paid A; paid B"}


# -- one palette entry ---------------------------------------------------------------------------------


def test_approval_and_pay_as_a_subgraph_is_one_node_that_waits_where_approval_asks():
    record = make_graph(
        approval,
        pay,
        edges=[("approval.result", "pay.ok")],
        inputs={"invoice": ["approval.invoice", "pay.invoice"]},
    )
    registry = registry_of(invoices, approval, pay, summary)
    approve_and_pay = registry.subgraph(
        "approve_and_pay", title="Approve and pay", description="For a test."
    )(record)
    edges = [
        ("invoices.result", "approve_and_pay.invoice"),
        ("approve_and_pay.result", "summary.lines"),
    ]
    graph = Graph.compile(make_graph(invoices, approve_and_pay, summary, edges=edges), registry)
    assert sorted(graph.describe().nodes) == ["approve_and_pay", "invoices", "summary"]
    assert "approve_and_pay" in {entry.id for entry in registry.describe().nodes}

    run = graph.run(text="A:100,B:900")
    (waiting,) = run.waiting
    assert waiting.loc == ("approve_and_pay", 1, "approval") and waiting.prompt == "Pay B?"
    assert run.answer(waiting, True).resume().outputs == {"result": "paid A; paid B"}


def test_results_show_one_none_per_item_of_a_branch_inside_a_loop_while_a_question_outside_waits():
    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return text.split(",")

    @node("gate", title="Gate", description="For a test.", asks=True)
    def gate(text: str) -> str:
        raise Ask("Which mark?")

    @dataclass(frozen=True)
    class Picked:
        yes: str = SKIPPED
        no: str = SKIPPED

    @node("pick", title="Pick", description="For a test.")
    def pick(line: str, mark: str) -> Picked:
        return Picked(yes=line + mark) if line == "a" else Picked(no=line)

    @node("loud", title="Loud", description="For a test.")
    def loud(line: str) -> str:
        return line.upper()

    def edge(node_id: str, output: str) -> dict[str, Any]:
        return {"sources": [{"node": node_id, "output": output}]}

    record = {
        "nodes": {
            "lines": {"node": "lines", "inputs": {"text": {"graph_input": "text"}}},
            "gate": {"node": "gate", "inputs": {"text": {"graph_input": "text"}}},
            "pick": {
                "node": "pick",
                "inputs": {"line": edge("lines", "result"), "mark": edge("gate", "result")},
            },
            "loud": {"node": "loud", "inputs": {"line": edge("pick", "yes")}},
        },
        "outputs": {"result": {"node": "loud", "output": "result"}},
    }
    graph = Graph.compile(json.dumps(record), registry_of(lines, gate, pick, loud))

    waiting = graph.run(text="a,b")

    assert waiting.status == "waiting"
    assert {name: waiting.results[name] for name in ("pick", "loud")} == {
        "pick": {"yes": [None, None], "no": [None, None]},
        "loud": {"result": [None, None]},
    }
    answered = waiting.answer(waiting.waiting[0], "!").resume()
    assert answered.results["loud"] == {"result": ["A!", None]}
