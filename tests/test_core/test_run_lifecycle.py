"""A run is its ``RunState``: the scenarios a host writes, through the public API only."""

import asyncio
import pathlib
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeDefinition,
    NodeError,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    node,
    table_with_columns,
)
from conductor.events import RunEnd, RunStart, UnitWaiting
from conductor.widgets import Textarea
from helpers import answer, read_table, waiting_id


class Txt(DType, str):
    id = "lifecycle-txt"
    display_title = "Tekst"


class Whole(DType, int):
    id = "lifecycle-int"
    display_title = "Tal"


Contacts = table_with_columns({"name": Txt, "email": Txt})
Out = Annotated[Txt, Result(title="Svar")]
calls: dict[str, int] = {}


def _called(name: str) -> None:
    calls[name] = calls.get(name, 0) + 1


@pytest.fixture(autouse=True)
def _no_calls():
    calls.clear()


def graph_node(cls_id, run_fn, **attrs):
    return type(cls_id.title().replace("-", ""), (NodeDefinition,), {"id": cls_id, "title": cls_id, "description": "d", "category": "test", "run": run_fn, **attrs})


@node("split", title="Split", description="d", category="test")
def split_node(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    _called("split")
    return [Txt(line) for line in text.split("\n")]


@node("words", title="Words", description="d", category="test")
def words(text: Annotated[Txt, Param(title="T")]) -> Annotated[Whole, Result(title="N")]:
    _called("words")
    return Whole(len(text.split()))


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out:
    _called("upper")
    return Txt(text.upper())


@node("approve", title="Approve", description="d", category="test")
def approve(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    _called("approve")
    return AskPerson(prompt=f"ok? {text}", defaults={"result": text})


@dataclass(frozen=True)
class Verdict:
    yes: Annotated[Txt, Result(title="Ja", exclusive_group="verdict")]
    no: Annotated[Txt, Result(title="Nej", exclusive_group="verdict")]
    note: Annotated[Txt, Result(title="Note")]


@node("decide", title="Decide", description="d", category="test")
def decide() -> Verdict | AskPerson:
    _called("decide")
    return AskPerson()


@node("ask-table", title="Ask table", description="d", category="test")
def ask_table() -> Annotated[Contacts, Result(title="Rettelser")] | AskPerson:
    _called("ask-table")
    return AskPerson(defaults={"result": read_table(Contacts, [{"name": "a", "email": "a@x"}])})


@node("ask-many", title="Ask many", description="d", category="test")
def ask_many(text: Annotated[Txt, Param(title="T")]) -> Annotated[Series[Txt], Result(title="Dele")] | AskPerson:
    return AskPerson()


@node("read-emails", title="Read", description="d", category="test")
def read_emails(email: Annotated[Txt, Param(title="E")]) -> Out:
    return Txt(email.upper())


@node("boom", title="Boom", description="d", category="test")
def boom(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Out:
    raise NodeError("det gik galt")


@node("wrong-type", title="Wrong", description="d", category="test")
def wrong_type() -> Out:
    return 42  # type: ignore[return-value]


@node("ask-stray", title="Ask stray", description="d", category="test")
def ask_stray() -> Out | AskPerson:
    return AskPerson(defaults={"nope": Txt("x")})


@node("slow", title="Slow", description="d", category="test")
def slow() -> Out:
    import time

    time.sleep(0.4)
    return Txt("late")


REGISTRY = NodeRegistry()
for cls in (split_node, words, upper_node, approve, decide, ask_table, ask_many, read_emails, boom, wrong_type, ask_stray, slow):
    REGISTRY = REGISTRY.extended_with([cls])


def compiled_of(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), REGISTRY)
    assert compiled.is_runnable, compiled.problems
    return compiled


def n(id: str, type: str | None = None, **bindings: Any) -> GraphNode:
    return GraphNode(id=id, type=type or id, version=1, bindings=bindings)


def roundtrip(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


LINES = (n("split", text=Static("one two\nthree")), n("count", "words", text=From("split.result")))


# -- scenario 1: run to the end, read the outputs -------------------------------------


def test_run_returns_the_state_and_outputs_read_it():
    compiled = compiled_of(*LINES)
    state = compiled.run_sync()
    assert (state.status, state.failure, state.waiting) == ("complete", None, ())
    counts = compiled.outputs(state)["count.result"]
    assert counts == {"rows": [(0,), (1,)], "values": [2, 1]}
    assert isinstance(counts["values"][0], Whole)


def test_the_same_read_works_on_a_state_back_from_json():
    compiled = compiled_of(*LINES)
    assert compiled.outputs(roundtrip(compiled.run_sync()))["count.result"]["values"] == [2, 1]


def test_in_a_running_loop_run_sync_points_at_run():
    compiled = compiled_of(*LINES)

    async def inside():
        with pytest.raises(RuntimeError, match="await compiled.run"):
            compiled.run_sync()
        return await compiled.run()

    assert asyncio.run(inside()).status == "complete"


# -- a failed run is a status, reading it is loud --------------------------------------


def test_a_failed_node_is_a_status_with_its_cause():
    compiled = compiled_of(n("boom"))
    state = compiled.run_sync()
    assert state.status == "failed"
    assert (state.failure.type, state.failure.msg, state.failure.loc[0], state.failure.loc[1:]) == ("failed", "det gik galt", "boom", ())


def test_outputs_of_a_state_that_is_not_complete_raises_with_the_cause():
    compiled = compiled_of(n("boom"))
    state = compiled.run_sync()
    with pytest.raises(InvalidCallError, match="det gik galt") as raised:
        compiled.outputs(state)
    (fault,) = raised.value.errors()
    assert fault.type == "run_not_complete"
    assert fault.ctx["status"] == "failed" and fault.ctx["failure"]["type"] == "failed"


def test_a_node_returning_the_wrong_type_ends_the_run_failed_not_refused():
    state = compiled_of(n("wrong-type")).run_sync()
    assert state.status == "failed" and state.failure.type == "invalid_output"
    assert "42" not in str(state.failure.ctx["reason"])


def test_asking_about_an_output_the_node_does_not_have_fails_the_run():
    state = compiled_of(n("ask-stray")).run_sync()
    assert state.status == "failed" and state.failure.type == "invalid_output"


def test_results_reads_what_a_stopped_run_did_produce():
    compiled = compiled_of(n("first", "upper", text=Static("a")), n("ask", "approve", text=From("first.result")))
    state = compiled.run_sync()
    assert state.status == "waiting" and compiled.results(state) == {"first": {"result": "A"}}


# -- scenario 2: stream events -------------------------------------------------------


def test_stream_starts_with_run_start_ends_with_run_end():
    compiled = compiled_of(n("ask", "approve", text=Static("a")))

    async def events():
        async with compiled.stream() as stream:
            return [event async for event in stream]

    seen = asyncio.run(events())
    assert isinstance(seen[0], RunStart) and isinstance(seen[-1], RunEnd)
    waiting = [event for event in seen if isinstance(event, UnitWaiting)]
    assert [event.unit.id.split("#")[0] for event in waiting] == ["ask"] and seen[-1].state.status == "waiting"


def test_a_refused_stream_raises_at_the_call_not_inside_the_block():
    broken = CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="x", type="nope", version=1),)), REGISTRY)
    with pytest.raises(InvalidCallError) as raised:
        broken.stream()
    (fault,) = raised.value.errors()
    assert fault.type == "unknown_node_type" and fault.loc == ("x",)


def test_leaving_the_block_stops_the_units_and_writes_no_state():
    compiled = compiled_of(n("slow"))

    async def leave():
        async with compiled.stream() as stream:
            async for event in stream:
                if isinstance(event, RunStart):
                    return "left"

    assert asyncio.run(leave()) == "left"


def test_cancel_and_timeout_end_the_leg_with_a_state():
    compiled = compiled_of(n("slow"))
    state = compiled.run_sync(timeout=0.05)
    assert state.status == "timed_out" and state.failure.type == "run_timed_out"
    assert state.failure.ctx["timeout_seconds"] == 0.05

    async def cancelled():
        cancel = asyncio.Event()
        cancel.set()
        return await compiled.run(cancel=cancel)

    state = asyncio.run(cancelled())
    assert state.status == "cancelled" and state.failure.type == "run_cancelled"


# -- scenario 3: a run stops for a person; another process goes on -----------------------


PER_ROW = (n("split", text=Static("a\nb\nc")), n("ask", "approve", text=From("split.result")), n("up", "upper", text=From("ask.result")))


def test_the_waiting_units_are_in_the_state_with_their_questions_and_load_with_no_graph():
    state = roundtrip(compiled_of(*PER_ROW).run_sync())
    assert [(unit.id.split("#")[0], unit.node_id, unit.row, unit.prompt) for unit in state.waiting] == [
        ("ask@0", "ask", (0,), "ok? a"), ("ask@1", "ask", (1,), "ok? b"), ("ask@2", "ask", (2,), "ok? c"),
    ]
    (question,) = state.waiting[0].questions
    assert question.name == "result" and question.dtype == {"id": "lifecycle-txt"} and question.title == "Svar"
    assert question.json_schema == {"type": "string"}
    assert (question.default, question.optional) == ("a", True)


def test_an_answer_finishes_its_unit_and_a_node_that_asked_is_not_called_again():
    compiled = compiled_of(*PER_ROW)
    first = compiled.run_sync()
    assert calls["approve"] == 3
    answered = answer(compiled, roundtrip(first), {"ask@1": {"result": "svar"}})
    assert [unit.id.split("#")[0] for unit in answered.waiting] == ["ask@0", "ask@2"]
    assert answered.status == "waiting" and "upper" not in calls

    second = compiled.run_sync(answered)
    assert second.status == "waiting" and calls["approve"] == 3 and calls["upper"] == 1
    assert [unit.id.split("#")[0] for unit in second.waiting] == ["ask@0", "ask@2"]

    third = compiled.run_sync(answer(compiled, second, {"ask@0": {"result": "x"}, "ask@2": None}))
    assert third.status == "complete" and calls["approve"] == 3
    assert compiled.outputs(third)["up.result"]["values"] == ["X", "SVAR"]


def test_answer_changes_no_run_and_leaves_the_state_it_was_given():
    compiled = compiled_of(n("ask", "approve", text=Static("a")))
    state = compiled.run_sync()
    dumped = state.model_dump_json()
    answer(compiled, state, {"ask": {"result": "b"}})
    assert state.model_dump_json() == dumped and calls["approve"] == 1


def test_a_question_with_a_default_may_be_left_blank():
    compiled = compiled_of(n("ask", "approve", text=Static("foreslået")))
    state = compiled.run_sync(answer(compiled, compiled.run_sync(), {"ask": {}}))
    assert compiled.outputs(state)["ask.result"] == "foreslået"


def test_every_fault_of_a_call_is_reported_at_once_and_the_state_is_untouched():
    compiled = compiled_of(n("d", "decide"), n("ask", "approve", text=Static("a")))
    state = compiled.run_sync()
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, state, {
            "ghost": {},
            "ask": {"stray": "x"},
            "d": {"yes": "a", "no": "b", "note": 1},
        })
    ask_id, d_id = waiting_id(state, "ask"), waiting_id(state, "d")
    assert sorted((fault.type, fault.loc) for fault in raised.value.errors()) == sorted([
        ("choice_unanswered", ("answers", d_id, "verdict")),
        ("invalid_value", ("answers", d_id, "note")),
        ("not_asked", ("answers", ask_id, "stray")),
        ("not_waiting", ("answers", "ghost")),
    ])
    assert len(state.waiting) == 2


def test_a_question_with_no_default_left_out_is_missing():
    compiled = compiled_of(n("d", "decide"))
    state = compiled.run_sync()
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, state, {"d": {"yes": "a"}})
    assert [(fault.type, fault.loc) for fault in raised.value.errors()] == [("missing", ("answers", waiting_id(state, "d"), "note"))]


def test_a_choice_output_left_out_means_that_branch_was_not_taken():
    compiled = compiled_of(n("d", "decide"), n("up", "upper", text=From("d.no")))
    state = compiled.run_sync(answer(compiled, compiled.run_sync(), {"d": {"yes": "ja", "note": "n"}}))
    assert state.status == "complete" and compiled.results(state)["d"] == {"yes": "ja", "note": "n"}
    assert "up" not in compiled.results(state)


def test_an_answer_for_a_unit_already_answered_is_not_waiting():
    compiled = compiled_of(n("ask", "approve", text=Static("a")))
    answered = answer(compiled, compiled.run_sync(), {"ask": {"result": "b"}})
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, answered, {"ask": {"result": "c"}})
    assert raised.value.errors()[0].type == "not_waiting"


def test_a_series_output_is_answered_as_a_plain_list_per_row():
    compiled = compiled_of(n("split", text=Static("a\nb")), n("ask", "ask-many", text=From("split.result")))
    state = compiled.run_sync()
    assert state.waiting[0].questions[0].json_schema == {"type": "array", "items": {"type": "string"}}
    state = compiled.run_sync(answer(compiled, state, {"ask@0": {"result": ["x", "y"]}, "ask@1": {"result": []}}))
    assert state.status == "complete"
    assert compiled.results(state)["ask"]["result"]["rows"] == [(0, 0), (0, 1)]


def test_a_table_is_answered_as_records_and_keeps_its_rows_across_json():
    compiled = compiled_of(n("t", "ask-table"), n("mail", "read-emails", email=From("t.result.email")))
    state = compiled.run_sync()
    assert state.waiting[0].questions[0].json_schema["type"] == "array"
    answered = answer(compiled, roundtrip(state), {"t": {"result": [{"name": "a", "email": "a@x"}, {"name": "b", "email": "b@x"}]}})
    done = compiled.run_sync(roundtrip(answered))
    assert done.status == "complete"
    assert compiled.results(done)["mail"]["result"]["values"] == ["A@X", "B@X"]


def test_adding_a_row_to_an_answered_table_keeps_the_other_rows_ids():
    compiled = compiled_of(n("t", "ask-table"))
    state = compiled.run_sync()
    wire = {"columns": [{"name": "name", "dtype": "String"}, {"name": "email", "dtype": "String"}],
            "data": {"name": ["a", "b", "new"], "email": ["a@x", "b@x", "n@x"]},
            "rows": ["t.result:1", "t.result:2", None]}
    done = compiled.run_sync(answer(compiled, state, {"t": {"result": wire}}))
    assert compiled.results(done)["t"]["result"]["__row"].to_list() == ["t.result:1", "t.result:2", "t.result:0"]


def test_a_table_form_with_rows_that_do_not_line_up_is_a_refusal():
    compiled = compiled_of(n("t", "ask-table"))
    wire = {"columns": [{"name": "name", "dtype": "String"}, {"name": "email", "dtype": "String"}],
            "data": {"name": ["a"], "email": ["a"]}, "rows": ["x", "y"]}
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, compiled.run_sync(), {"t": {"result": wire}})
    assert raised.value.errors()[0].type == "invalid_value"


def test_a_table_default_comes_back_with_the_waiting_unit():
    state = compiled_of(n("t", "ask-table")).run_sync()
    assert roundtrip(state).waiting[0].questions[0].default["name"].to_list() == ["a"]


def test_a_changed_node_asks_again_and_its_old_unit_is_gone():
    compiled = compiled_of(n("ask", "approve", text=Static("a")))
    earlier = compiled.run_sync()
    changed = compiled_of(n("ask", "approve", text=Static("b")))
    with pytest.raises(InvalidCallError) as raised:
        answer(changed, earlier, {"ask": {"result": "x"}})
    assert raised.value.errors()[0].type == "not_waiting"
    again = changed.run_sync(earlier)
    assert again.status == "waiting" and again.waiting[0].prompt == "ok? b" and calls["approve"] == 2


# -- scenario 4: re-run with one input changed ------------------------------------------


CHAIN = (n("a", "upper", text=Static("x")), n("b", "upper", text=From("a.result")))


def test_a_changed_input_runs_it_and_what_reads_it_and_nothing_else():
    compiled = compiled_of(*CHAIN, n("other", "upper", text=Static("y")))
    earlier = compiled.run_sync()
    calls.clear()
    edited = compiled_of(n("a", "upper", text=Static("z")), CHAIN[1], n("other", "upper", text=Static("y")))
    assert edited._nodes_changed_since(earlier) == ["a", "b"]
    state = edited.run_sync(earlier)
    assert calls["upper"] == 2 and edited.outputs(state)["b.result"] == "Z"


def test_run_start_says_what_was_reused_and_what_changed():
    compiled = compiled_of(*CHAIN)
    earlier = compiled.run_sync()
    edited = compiled_of(CHAIN[0], n("b", "upper", text=From("a.result")), n("c", "upper", text=Static("n")))

    async def first():
        async with edited.stream(earlier) as stream:
            return await anext(stream)

    start = asyncio.run(first())
    assert start.reused_nodes == ["a", "b"] and start.changed_nodes == ["c"]


def test_without_runs_one_node_again_though_nothing_changed():
    compiled = compiled_of(*CHAIN)
    state = compiled.run_sync()
    calls.clear()
    assert compiled._nodes_changed_since(state) == []
    compiled.run_sync(state.without_nodes("b"))
    assert calls["upper"] == 1


def test_a_node_gone_from_the_graph_is_named_and_dropped():
    state = compiled_of(*CHAIN).run_sync()
    shorter = compiled_of(CHAIN[0])
    assert shorter._nodes_changed_since(state) == ["b"]
    assert shorter.run_sync(state).status == "complete"


# -- restoring reads every value at the door ---------------------------------------------


def test_a_stored_value_that_no_longer_reads_is_refused_naming_where():
    compiled = compiled_of(n("count", "words", text=Static("a b")))
    stored = compiled.run_sync().model_dump(mode="json")
    stored["nodes"]["count"]["outputs"]["result"]["values"] = ["ikke et tal"]
    for verb in (compiled.run_sync, compiled.results, compiled.outputs):
        with pytest.raises(InvalidCallError) as raised:
            verb(RunState.model_validate(stored))
        (fault,) = raised.value.errors()
        assert fault.type == "state_unreadable" and fault.loc == ("state", "count.result")


def test_a_value_at_a_row_its_field_cannot_have_is_refused():
    compiled = compiled_of(n("count", "words", text=Static("a b")))
    stored = compiled.run_sync().model_dump(mode="json")
    stored["nodes"]["count"]["outputs"]["result"]["rows"] = [[3]]
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync(RunState.model_validate(stored))
    assert raised.value.errors()[0].loc == ("state", "count.result", 3)


def test_a_blank_is_a_legal_stored_value_on_any_output():
    compiled = compiled_of(n("a", "upper", text=Static("x")))
    stored = compiled.run_sync().model_dump(mode="json")
    stored["nodes"]["a"]["outputs"]["result"]["values"] = [None]
    assert compiled.results(RunState.model_validate(stored)) == {"a": {"result": None}}


def test_a_node_returning_none_on_a_typed_output_still_fails():
    @node("returns-none", title="x", description="d", category="test")
    def returns_none() -> Out:
        return None  # type: ignore[return-value]

    registry = NodeRegistry([returns_none])
    compiled = CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="r", type="returns-none", version=1),)), registry)
    assert compiled.run_sync().failure.type == "invalid_output"


def test_a_dump_of_a_value_with_no_json_form_names_the_output():
    class Opaque(DType):
        id = "lifecycle-opaque"
        display_title = "x"

    @node("makes-opaque", title="x", description="d", category="test")
    def makes_opaque() -> Annotated[Opaque, Result(title="O")]:
        return Opaque()

    registry = NodeRegistry([makes_opaque])
    state = CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="o", type="makes-opaque", version=1),)), registry).run_sync()
    with pytest.raises(Exception, match=r"o\.result"):
        state.model_dump_json()


# -- with_inputs --------------------------------------------------------------------------


def test_with_inputs_refuses_every_name_it_does_not_offer():
    compiled = compiled_of(n("a", "upper", text=Static("x")))
    with pytest.raises(InvalidCallError) as raised:
        compiled.with_inputs(nope=1, ghost=2)
    assert [(fault.type, fault.loc) for fault in raised.value.errors()] == [
        ("not_offered", ("inputs", "nope")), ("not_offered", ("inputs", "ghost")),
    ]


def test_no_series_constructor_is_needed_for_skips_in_a_many_valued_output():
    @node("sparse", title="x", description="d", category="test")
    def sparse() -> Annotated[Series[Txt], Result(title="L")]:
        return [Txt("a"), SKIPPED, Txt("c")]

    registry = NodeRegistry([sparse])
    compiled = CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="s", type="sparse", version=1),)), registry)
    assert compiled.outputs(roundtrip(compiled.run_sync()))["s.result"]["rows"] == [(0,), (2,)]


# -- staleness reaches a reader of a table column; a partly answered node reads so far ---


@node("extract", title="Extract", description="d", category="test")
def extract(text: Annotated[Txt, Param(title="T", widget=Textarea())] = Txt("")) -> Annotated[Contacts, Result(title="R")]:
    return read_table(Contacts, [{"name": text, "email": "a@x"}])


REGISTRY = REGISTRY.extended_with([extract])


def test_a_change_above_a_column_reader_drops_its_answer_and_it_asks_again():
    def graph(text):
        return compiled_of(n("x", "extract", text=Static(text)), n("ask", "approve", text=From("x.result.name")))

    compiled = graph("a")
    state = compiled.run_sync(answer(compiled, compiled.run_sync(), {"ask@0": {"result": "ja"}}))
    assert state.status == "complete"
    calls.clear()

    edited = graph("b")
    assert edited._nodes_changed_since(state) == ["x", "ask"]
    again = edited.run_sync(state)
    assert again.status == "waiting" and calls["approve"] == 1 and again.waiting[0].prompt == "ok? b"


def test_results_show_the_rows_answered_so_far():
    compiled = compiled_of(*PER_ROW)
    partly = compiled.run_sync(answer(compiled, compiled.run_sync(), {"ask@1": {"result": "svar"}}))
    assert partly.status == "waiting"
    result = compiled.results(partly)["ask"]["result"]
    assert result == {"rows": [(1,)], "values": ["svar"]}


def test_the_state_module_imports_nothing_from_the_graph():
    """A state loads, and a form renders from it, with no graph or registry in sight."""
    import conductor._internal.execution.run_state as module

    assert "conductor._internal.graph" not in pathlib.Path(module.__file__).read_text()


def test_a_signature_that_moves_a_node_to_other_rows_changes_its_fingerprint():
    @node("words", title="x", description="d", category="test")
    def whole_(text: Annotated[Series[Txt], Param(title="T")]) -> Annotated[Whole, Result(title="N")]:
        return Whole(len(text))

    per_row = compiled_of(*LINES)
    registry = NodeRegistry()
    for cls in (split_node, whole_):
        registry = registry.extended_with([cls])
    whole = CompiledGraph.from_graph(Graph(nodes=LINES), registry)
    assert per_row.node("count").fingerprint != whole.node("count").fingerprint
    assert whole._nodes_changed_since(per_row.run_sync()) == ["count"]


def test_an_input_that_starts_taking_a_blank_changes_the_fingerprint_and_is_called_on_the_blank_it_was_not_before():
    @node("nothing", title="x", description="d", category="test")
    def nothing() -> Annotated[Whole | None, Result(title="N")]:
        return None

    @node("counted", title="x", description="d", category="test")
    def refuses_a_blank(x: Annotated[Whole, Param(title="X")]) -> Annotated[Whole, Result(title="N")]:
        _called("counted")
        return Whole(x)

    @node("counted", title="x", description="d", category="test")
    def takes_a_blank(x: Annotated[Whole | None, Param(title="X")]) -> Annotated[Whole, Result(title="N")]:
        _called("counted")
        return Whole(0 if x is None else x)

    nodes = [GraphNode(id="n", type="nothing", version=1), GraphNode(id="c", type="counted", version=1, bindings={"x": From("n.result")})]
    before = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry().extended_with([nothing, refuses_a_blank]))
    after = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry().extended_with([nothing, takes_a_blank]))
    state = before.run_sync()
    assert calls == {}

    assert before.node("c").fingerprint != after.node("c").fingerprint
    assert after._nodes_changed_since(state) == ["c"]
    after.run_sync(state)
    assert calls == {"counted": 1}
