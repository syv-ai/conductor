"""One call of a node, with no event loop: every way it can end, as a value."""

import asyncio
import threading
from typing import Annotated, TypedDict

import polars as pl
from conductor import (
    ROW_ID_COLUMN,
    AskPerson,
    CompiledGraph,
    DType,
    ExternalFailure,
    Graph,
    GraphNode,
    NodeError,
    NodeRegistry,
    Policy,
    Result,
    Table,
    node,
)
from conductor._internal.execution.call_node import CallFailed, call_node
from conductor._internal.execution.run_state import WaitingUnit
from conductor.events import NodeComplete, RunEnd
from conductor.widgets import Textarea
from pydantic import TypeAdapter, ValidationError


class Contact(TypedDict):
    name: str
    email: str


Contacts = Table[Contact]


def _compiled_node(function):
    graph = Graph(nodes=[GraphNode(id="n", type=function.id, version=1)])
    return CompiledGraph.from_graph(graph, NodeRegistry([function])).node("n")


def _failure(outcome) -> tuple[str, str, bool]:
    assert isinstance(outcome, CallFailed), outcome
    return outcome.problem.type, outcome.problem.msg, outcome.retryable


@node("double", category="t")
def double(x: int) -> int:
    return x * 2


def test_a_good_call_returns_the_cells_it_wrote():
    assert call_node(_compiled_node(double), (), {"x": 21}, {}, "hash") == {"result": 42}


def test_an_input_that_does_not_fit_is_invalid_input_with_the_engines_reason_text():
    outcome = call_node(_compiled_node(double), (2,), {"x": "not a number"}, {}, "hash")

    assert _failure(outcome)[0] == "invalid_input"
    assert outcome.problem.loc == ("n", 2)
    assert outcome.problem.msg.startswith("Invalid input — ")
    assert outcome.problem.ctx == {"reason": outcome.problem.msg}


def test_a_table_without_a_column_it_needs_says_which():
    @node("needs_email", category="t")
    def needs_email(contacts: Contacts) -> str:
        return "ok"

    outcome = call_node(_compiled_node(needs_email), (), {"contacts": pl.DataFrame({"name": ["Ida"]})}, {}, "hash")

    assert _failure(outcome)[0] == "table_column_missing"
    assert outcome.problem.ctx == {"column": "email"}


def test_a_node_error_keeps_the_nodes_message_and_is_not_retryable():
    @node("refuses", category="t")
    def refuses() -> int:
        raise NodeError("Dokumentet kunne ikke læses.")

    assert _failure(call_node(_compiled_node(refuses), (), {}, {}, "hash")) == ("failed", "Dokumentet kunne ikke læses.", False)


def test_a_node_error_with_a_code_of_its_own_keeps_it():
    @node("refuses", category="t")
    def refuses() -> int:
        raise NodeError("for stort", code="result_too_large", ctx={"limit": 3})

    outcome = call_node(_compiled_node(refuses), (), {}, {}, "hash")

    assert _failure(outcome) == ("result_too_large", "for stort", False)
    assert outcome.problem.ctx == {"limit": 3}


def test_an_external_failure_may_be_retried():
    @node("flaky", category="t")
    def flaky() -> int:
        raise ExternalFailure("nede")

    assert _failure(call_node(_compiled_node(flaky), (), {}, {}, "hash")) == ("external_failed", "nede", True)


def test_a_class_the_policy_names_is_external_and_its_text_is_not_kept():
    @node("flaky", category="t", policy=Policy(retry_on=(ConnectionError,)))
    def flaky() -> int:
        raise ConnectionError("secret-token")

    outcome = call_node(_compiled_node(flaky), (), {}, {}, "hash")

    assert _failure(outcome) == ("external_failed", "An outside service did not answer.", True)
    assert outcome.problem.ctx == {"exception": "ConnectionError"}


def test_anything_else_a_body_raises_is_execution_failed_even_a_validation_error():
    @node("buggy", category="t")
    def buggy() -> int:
        return TypeAdapter(int).validate_python("not a number")

    outcome = call_node(_compiled_node(buggy), (), {}, {}, "hash")

    assert _failure(outcome) == ("execution_failed", "The node failed.", False)
    assert outcome.problem.ctx == {"exception": ValidationError.__name__}


def test_a_return_that_is_not_the_declared_type_is_invalid_output_with_its_errors():
    @node("liar", category="t")
    def liar() -> int:
        return "not a number"

    outcome = call_node(_compiled_node(liar), (), {}, {}, "hash")

    assert _failure(outcome)[:2] == ("invalid_output", "The node returned a value that is not what it declared.")
    assert outcome.problem.ctx["reason"] == "'result' must be a int, not a str"
    assert outcome.problem.ctx["errors"]


def test_returning_something_from_a_node_with_no_outputs_is_invalid_output():
    @node("silent", category="t")
    def silent() -> None:
        return 3

    outcome = call_node(_compiled_node(silent), (), {}, {}, "hash")

    assert _failure(outcome)[0] == "invalid_output"
    assert "declares no outputs" in outcome.problem.ctx["reason"]


def test_a_table_that_lost_its_rows_ids_fails_when_they_must_survive():
    @node("rebuild", category="t")
    def rebuild(contacts: Contacts) -> Annotated[Contacts, Result(title="Contacts")]:
        return contacts.drop(ROW_ID_COLUMN)

    compiled_node = _compiled_node(rebuild)
    held = pl.DataFrame({"name": ["Ida"], "email": ["i@x"], ROW_ID_COLUMN: ["s:0"]})

    outcome = call_node(compiled_node, (), {"contacts": held}, {}, "hash", frozenset({compiled_node._wiring.fields["result"].ref}))

    assert _failure(outcome)[0] == "row_ids_lost"


def test_a_returned_table_is_given_its_rows_ids():
    @node("csv", category="t")
    def csv() -> Annotated[Contacts, Result(title="Contacts")]:
        return pl.DataFrame({"name": ["Ida", "Bo"], "email": ["i@x", "b@x"]})

    cells = call_node(_compiled_node(csv), (), {}, {}, "hash")

    assert cells["result"][ROW_ID_COLUMN].to_list() == ["n.result:0", "n.result:1"]


def test_an_ask_is_the_waiting_unit_it_makes():
    @node("asks", category="t")
    def asks(x: int) -> str | AskPerson:
        return AskPerson(prompt="Hvad?")

    outcome = call_node(_compiled_node(asks), (1,), {"x": 1}, {}, "the-hash")

    assert isinstance(outcome, WaitingUnit)
    assert (outcome.row, outcome.prompt, outcome.input_hash, outcome.inputs) == ((1,), "Hvad?", "the-hash", {"x": 1})


def test_an_ask_that_is_not_sound_is_invalid_output():
    @node("asks", category="t")
    def asks() -> str | AskPerson:
        return AskPerson(defaults={"nope": 1})

    outcome = call_node(_compiled_node(asks), (), {}, {}, "hash")

    assert _failure(outcome)[0] == "invalid_output"
    assert "not one of this node's outputs" in outcome.problem.ctx["reason"]


def test_an_ask_proposing_a_widget_that_is_no_widget_is_invalid_output():
    @node("asks", category="t")
    def asks() -> str | AskPerson:
        return AskPerson(widgets={"result": Textarea})

    assert _failure(call_node(_compiled_node(asks), (), {}, {}, "hash"))[0] == "invalid_output"


# -- a call's output check is off the event loop --------------------------------------------


class _SlowOutput(DType, str):
    """An output whose check blocks until the other unit's completion has been read off the stream."""

    id = "call-node-slow-output"
    display_title = "Slow"
    check_began = threading.Event()
    other_unit_reported = threading.Event()
    other_unit_reported_in_time: bool | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        def slow_check(value):
            cls.check_began.set()
            cls.other_unit_reported_in_time = cls.other_unit_reported.wait(timeout=2)
            return cls(value)

        return core_schema.no_info_plain_validator_function(slow_check)


def test_a_slow_output_check_does_not_stall_the_other_units():
    """The other unit only finishes after the check has begun, and the check
    only passes once the stream has reported that unit: a check on the event
    loop would hold up the event, so the wait would run out."""
    @node("slow_output", category="t")
    def slow_output() -> Annotated[_SlowOutput, Result(title="Slow")]:
        return "x"  # a value that is not yet the type, so its check runs

    @node("other", category="t")
    def other() -> int:
        assert _SlowOutput.check_began.wait(timeout=5)
        return 1

    graph = Graph(nodes=[GraphNode(id="s", type="slow_output", version=1), GraphNode(id="o", type="other", version=1)])
    compiled = CompiledGraph.from_graph(graph, NodeRegistry([slow_output, other]))

    async def run() -> str:
        async with compiled.stream() as events:
            async for event in events:
                if isinstance(event, NodeComplete) and event.node_id == "o":
                    _SlowOutput.other_unit_reported.set()
                if isinstance(event, RunEnd):
                    return event.state.status
        raise AssertionError("no RunEnd")

    assert asyncio.run(run()) == "complete"
    assert _SlowOutput.other_unit_reported_in_time is True
