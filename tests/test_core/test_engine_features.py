"""Events and pipelining, Policy, hooks, a host's own types, FromRun, exclusive groups and describe(), through the public door."""
import asyncio
import base64
import json
import queue
import re
import time as t
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeAliasType

import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FromRun,
    GraphRecord,
    InvalidCallError,
    Param,
    Policy,
    Series,
    host_type,
    node,
    stored_graph,
)
from pydantic import AfterValidator, GetCoreSchemaHandler, JsonValue
from pydantic_core import core_schema

from tests.test_core.nodes import place, registry_of


def collect(run, **kw):
    async def go():
        return [e async for e in run.stream(**kw)]

    return asyncio.run(go())


# -- 1. events, concurrency, pipelining ----------------------------------------------------------


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",")


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("slow", title="Slow", description="For a test.", policy=Policy(concurrency=8))
def slow(text: str) -> str:
    t.sleep(0.1)
    return text * 2


@node("slow2", title="Slow2", description="For a test.", policy=Policy(concurrency=8))
def slow2(text: str) -> str:
    t.sleep(0.1)
    return text + "!"


@node("gate", title="Gate", description="For a test.")
def gate(text: str) -> str:
    if text == "b":
        raise Ask("ok?")
    return text


@node("never", title="Never", description="For a test.")
def never(text: str) -> str:
    return SKIPPED


@node("boom", title="Boom", description="For a test.", policy=Policy(concurrency=4))
def boom(text: str) -> str:
    if text == "c":
        raise ValueError("no c")
    t.sleep(0.05)
    return text


reg = registry_of(split, upper, join, slow, slow2, gate, never, boom)


def chain(*ids, gather=None):
    nodes = [{"id": i, "node": i} for i in ids]
    edges = [[f"{a}.result", f"{b}.text"] for a, b in zip(ids, ids[1:])]
    if gather:
        nodes.append({"id": "join", "node": "join"})
        edges.append([f"{ids[-1]}.result", "join.texts"])
    return {"nodes": nodes, "edges": edges}


def test_events_in_order():
    graph = reg.load(GraphRecord.model_validate(chain("split", "upper", gather=True)))
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and events[-1].status == "complete", kinds
    complete = {e.loc[0]: e.units for e in events if e.type == "node_complete"}
    assert complete == {"split": 1, "upper": 3, "join": 1}, complete
    assert kinds.index("node_start") < kinds.index("node_complete")
    assert events[-1].run.outputs == {"join.result": "A+B+C"}
    assert all(isinstance(e.model_dump(mode="json"), dict) for e in events)


def test_concurrency_eight_items_in_one_sleep():
    graph = reg.load(GraphRecord.model_validate(chain("split", "slow")))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    took = t.monotonic() - start
    assert took < 0.18 and run.results["slow"]["result"] == [c * 2 for c in "abcdefgh"], took


def test_pipelining_two_slow_nodes():
    """16 items through two slow nodes of 8 at once: node by node is 4 sleeps, pipelined 3."""
    graph = reg.load(GraphRecord.model_validate(chain("split", "slow", "slow2", gather=True)))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefghijklmnop"))
    took = t.monotonic() - start
    assert run.outputs["join.result"] == "+".join(c * 2 + "!" for c in "abcdefghijklmnop")
    assert took < 0.36, took


def test_stream_leaves_the_event_loop_free():
    graph = reg.load(GraphRecord.model_validate(chain("split", "slow")))
    ticks = []

    async def go():
        async def tick():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        events = [e async for e in graph.with_inputs(text="a,b").stream()]
        ticker.cancel()
        return events

    events = asyncio.run(go())
    assert len(ticks) >= 5 and events[-1].status == "complete", len(ticks)


def test_waiting_and_skipped_events():
    graph = reg.load(GraphRecord.model_validate({"nodes": [{"id": "split", "node": "split"}, {"id": "gate", "node": "gate"}, {"id": "never", "node": "never"}],
                      "edges": [["split.result", "gate.text"], ["split.result", "never.text"]]}))
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert "unit_waiting" in kinds and "node_skipped" in kinds and events[-1].status == "waiting", kinds
    waiting = [e.unit for e in events if e.type == "unit_waiting"]
    assert [place(w) for w in waiting] == ["gate@1"]
    assert not any(e.type == "node_complete" and e.loc[0] == "gate" for e in events)


def test_failure_keeps_calls_already_made():
    graph = reg.load(GraphRecord.model_validate(chain("split", "boom")))
    run = graph.run(text="a,b,c,d")
    assert run.status == "failed" and run.state.cause.loc == ("boom", 2), run.state.cause
    assert all(run._outcomes["boom"].get((i,)) is not None for i in (0, 1, 3))  # d finished on its thread after c failed
    seen = []

    @node("boom", title="Fixed", description="For a test.", policy=Policy(concurrency=4))
    def fixed(text: str) -> str:
        seen.append(text)
        return text

    again = registry_of(split, fixed).load(GraphRecord.model_validate(chain("split", "boom"))).load_run(run.to_json()).resume()
    assert again.status == "complete" and seen == ["c"], seen  # nothing finished runs twice


def test_cancel_and_timeout():
    graph = reg.load(GraphRecord.model_validate(chain("split", "slow")))
    run = graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).resume(timeout=0.15)
    assert run.status == "timed_out", run.status

    async def go():
        cancel = asyncio.Event()
        events = []
        async for e in graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).stream(cancel=cancel):
            events.append(e)
            if e.type == "node_start":
                cancel.set()
        return events

    events = asyncio.run(go())
    assert events[-1].status == "cancelled", events[-1].status
    assert len(events[-1].run._outcomes["slow"]) < 24


def test_gather_per_document_while_items_run():
    """documents -> lines -> slow per line -> per-document join: each document gets its own lines."""

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    reg2 = registry_of(docs, split, slow, join)
    graph = reg2.load(GraphRecord.model_validate({"nodes": [{"id": "docs", "node": "docs"}, {"id": "split", "node": "split"}, {"id": "slow", "node": "slow"}, {"id": "join", "node": "join"}],
                       "edges": [["docs.result", "split.text"], ["split.result", "slow.text"], ["slow.result", "join.texts"]]}))
    run = graph.run(text="a,b|c|d,e,f")
    assert run.results["join"]["result"] == ["aa+bb", "cc", "dd+ee+ff"], run.results


# -- 2. Policy ------------------------------------------------------------------------------------


class Flaky(OSError):
    pass


def test_a_leg_waiting_on_a_timer_looks_at_its_queue_without_waiting_before_it_waits_with_a_timeout(monkeypatch: pytest.MonkeyPatch):
    """CPython 3.12's ``SimpleQueue.get(timeout=)`` can stay in the call for ever when it is the first look at an
    empty queue, so a leg that waits for a retry never makes one: every timed wait follows a look that found nothing."""
    looks: list[str] = []

    class Watched(queue.SimpleQueue[Any]):
        def get(self, block: bool = True, timeout: float | None = None) -> Any:
            looks.append("no wait" if not block else "timed" if timeout is not None else "wait")
            assert timeout is None or timeout > 0
            return super().get(block, timeout)

    monkeypatch.setattr(queue, "SimpleQueue", Watched)

    @node("fetch", title="Fetch", description="For a test.", policy=Policy(retries=1, retry_delay=0.01))
    def fetch(text: str) -> str:
        if not looks:
            raise ExternalError("busy")
        return text

    assert registry_of(fetch).load(GraphRecord.model_validate(chain("fetch"))).run(text="a").outputs == {"fetch.result": "a"}
    assert "timed" in looks
    assert all(looks[i - 1] == "no wait" for i, look in enumerate(looks) if look == "timed")
    assert looks[0] == "no wait"


@pytest.mark.timeout(10)
@pytest.mark.parametrize("policy", [Policy(), Policy(concurrency=2)], ids=["on the leg's thread", "on a pool thread"])
def test_an_exit_a_node_raises_ends_the_run_with_it_wherever_the_node_is_called(policy: Policy):
    @node("leave", title="Leave", description="For a test.", policy=policy)
    def leave(text: str) -> str:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        registry_of(leave).load(GraphRecord.model_validate(chain("leave"))).run(text="a")


@pytest.mark.timeout(10)
@pytest.mark.parametrize("policy", [Policy(), Policy(concurrency=2)], ids=["on the leg's thread", "on a pool thread"])
def test_a_defect_in_an_output_types_own_check_ends_the_run_with_it_wherever_the_node_is_called(policy: Policy):
    def explode(value: str) -> str:
        raise RuntimeError("the type's own check has a bug")

    @node("work", title="Work", description="For a test.", policy=policy)
    def work(text: str) -> Annotated[str, AfterValidator(explode)]:
        return text

    with pytest.raises(RuntimeError, match="has a bug"):
        registry_of(work).load(GraphRecord.model_validate(chain("work"))).run(text="a")


def test_retries_external_failures_then_succeeds():
    tries: dict[str, int] = {}

    @node("fetch", title="Fetch", description="For a test.", policy=Policy(retries=2, retry_delay=0.01))
    def fetch(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if tries[text] < 3:
            raise ExternalError("busy")
        return text

    @node("client", title="Client", description="For a test.", policy=Policy(retries=1, retry_delay=0.01, retry_on=(Flaky,), concurrency=2))
    def client(text: str) -> str:
        tries["c" + text] = tries.get("c" + text, 0) + 1
        if tries["c" + text] < 2:
            raise Flaky("reset")
        return text + "!"

    graph = registry_of(split, fetch, client).load(GraphRecord.model_validate(chain("split", "fetch", "client")))
    events = collect(graph.with_inputs(text="a,b"))
    retries = [e for e in events if e.type == "node_retry"]
    assert events[-1].status == "complete" and events[-1].run.results["client"]["result"] == ["a!", "b!"], events[-1].run.cause
    assert sorted((e.loc[0], e.loc[1:], e.attempt) for e in retries) == [("client", (0,), 1), ("client", (1,), 1), ("fetch", (0,), 1), ("fetch", (0,), 2), ("fetch", (1,), 1), ("fetch", (1,), 2)]
    assert retries[0].cause.type == "external_failed" and retries[0].delay == 0.01


def test_retries_run_out_and_plain_errors_never_retry():
    calls = []

    @node("down", title="Down", description="For a test.", policy=Policy(retries=1, retry_delay=0.01))
    def down(text: str) -> str:
        calls.append(text)
        raise ExternalError("down")

    @node("bug", title="Bug", description="For a test.", policy=Policy(retries=3, retry_delay=0.01))
    def bug(text: str) -> str:
        calls.append("bug")
        raise KeyError(text)

    run = registry_of(down).load(GraphRecord.model_validate({"nodes": [{"id": "down", "node": "down"}]})).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "external_failed" and calls == ["x", "x"], (run.state.cause, calls)
    calls.clear()
    run = registry_of(bug).load(GraphRecord.model_validate({"nodes": [{"id": "bug", "node": "bug"}]})).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "execution_failed" and calls == ["bug"]


def test_attempt_timeout_is_final():
    calls = []

    @node("hang", title="Hang", description="For a test.", policy=Policy(attempt_timeout=0.05, retries=2, retry_delay=0.01))
    def hang(text: str) -> str:
        calls.append(text)
        t.sleep(0.3)
        return text

    start = t.monotonic()
    run = registry_of(hang).load(GraphRecord.model_validate({"nodes": [{"id": "hang", "node": "hang"}]})).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "timeout" and calls == ["x"] and t.monotonic() - start < 0.25, (run.state.cause, calls)


def test_reuse_when_inputs_match_false_calls_again():
    calls = []

    @node("rate", title="Rate", description="For a test.", policy=Policy(reuse_when_inputs_match=False))
    def rate(text: str) -> str:
        calls.append(text)
        return text

    graph = registry_of(split, rate, upper).load(GraphRecord.model_validate(chain("split", "rate", "upper")))
    first = graph.run(text="a,b")
    calls.clear()
    first.again(text="a,b")
    assert calls == ["a", "b"], calls


# -- 3. hooks ------------------------------------------------------------------------------------



def _formula_inputs(formula: str) -> dict[str, Any]:
    return {name: float for name in re.findall(r"{(\w+)}", formula)}


@node("formula", title="Formula", description="For a test.", compute_added_inputs=_formula_inputs)
def formula(formula: str = "", **values: float) -> float:
    return eval(formula.format(**values))  # a check, not a calculator


def _branch_outputs(cases: list[str], value: type) -> dict[str, Any]:
    from typing import Annotated

    from conductor import Param

    return {name: Annotated[value, Param(title=name.title(), exclusive_group="case")] for name in [*cases, "otherwise"]}


@node("route", title="Route", description="For a test.", compute_outputs=_branch_outputs)
def route(value: Any, cases: list[str] = []) -> Mapping[str, Any]:  # noqa: B006
    taken = next((c for c in cases if c in str(value)), "otherwise")
    return {name: value if name == taken else SKIPPED for name in [*cases, "otherwise"]}


@node("number", title="Number", description="For a test.")
def number(text: str) -> int:
    return len(text)


@node("half", title="Half", description="For a test.")
def half(x: float) -> float:
    return x / 2


def test_added_inputs_from_typed_value():
    reg3 = registry_of(number, formula)
    graph = reg3.load(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "number"}, {"id": "f", "node": "formula", "values": {"formula": "{a} * 10 + {b}", "b": 2}}],
                       "edges": [["n.result", "f.a"]]}))
    assert not graph.problems, graph.problems
    assert list(graph.inputs) == ["text"]
    run = graph.run(text="abc")
    assert run.outputs["f.result"] == 32.0, run.outputs
    broken = reg3.load(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "formula", "values": {"formula": "{a}"}}, {"id": "g", "node": "formula"}], "edges": [["f.result", "g.zz"]]}))
    assert [p.type for p in broken.problems] == ["unknown_input"], broken.problems


def test_outputs_typed_by_what_arrives():
    reg3 = registry_of(split, number, route, half, upper)
    graph = reg3.load(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "n", "node": "number"}, {"id": "r", "node": "route", "values": {"cases": ["3"]}},
                                 {"id": "h", "node": "half"}],
                       "edges": [["s.result", "n.text"], ["n.result", "r.value"], ["r.3", "h.x"], ["r.otherwise", "h.x"]]}))
    assert not graph.problems, graph.problems
    assert graph.run(text="abc,de").results["h"]["result"] == [1.5, 1.0]
    wrong = reg3.load(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "number"}, {"id": "r", "node": "route"}, {"id": "u", "node": "upper"}],
                       "edges": [["n.result", "r.value"], ["r.otherwise", "u.text"]]}))
    assert [p.type for p in wrong.problems] == ["type_mismatch"], wrong.problems


# -- 4. a host's own types ------------------------------------------------------------------------




@host_type("integer", title="Integer")
class Integer(int):
    pass


@host_type("number", title="Number", accepts=[Integer])
class Number(float):
    """Reads the Danish decimal comma; takes an Integer."""


    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v.replace(",", ".")) if isinstance(v, str) else cls(v), core_schema.float_schema())



@host_type("file", title="File")
class File:

    def __init__(self, name: str, content: bytes) -> None:
        self.name, self.content = name, content

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, File) else File(v["name"], base64.b64decode(v["content"])),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda f: {"name": f.name, "content": base64.b64encode(f.content).decode()}, when_used="json"))


@node("count", title="Count", description="For a test.")
def count(text: str) -> Integer:
    return Integer(len(text))


@node("double", title="Double", description="For a test.")
def double(x: Number) -> Number:
    return Number(x * 2)


@node("attach", title="Attach", description="For a test.")
def attach(x: Number) -> File:
    return File("x.txt", str(x).encode())


@node("size", title="Size", description="For a test.")
def size(f: File) -> int:
    return len(f.content)


def test_own_types_accept_and_round_trip():
    reg4 = registry_of(count, double, attach, size, upper)
    graph = reg4.load(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "count"}, {"id": "d", "node": "double"}, {"id": "a", "node": "attach"}, {"id": "s", "node": "size"}],
                       "edges": [["c.result", "d.x"], ["d.result", "a.x"], ["a.result", "s.f"]]}))
    assert not graph.problems, graph.problems
    run = graph.run(text="abc")
    assert run.outputs == {"s.result": 3}
    again = graph.load_run(run.to_json())
    assert again.results["a"]["result"].content == b"6.0"
    assert reg4.load(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double", "values": {"x": "1,5"}}]})).run().outputs == {"d.result": 3.0}
    wrong = reg4.load(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double"}, {"id": "c", "node": "count"}], "edges": [["d.result", "c.text"]]}))
    assert [p.type for p in wrong.problems] == ["type_mismatch"]


Amount = TypeAliasType("Amount", int | float)


@node("as-text", title="As text", description="For a test.")
def as_text(text: str) -> str:
    return text


@node("as-yes", title="As yes", description="For a test.")
def as_yes(text: str) -> bool:
    return bool(text)


@node("as-count", title="As count", description="For a test.")
def as_count(text: str) -> int:
    return len(text)


@node("as-low", title="As low", description="For a test.")
def as_low(text: str) -> Literal["low"]:
    return "low"


@node("show-json", title="Show json", description="For a test.")
def show_json(value: JsonValue) -> str:
    return json.dumps(value)


@node("show-number", title="Show number", description="For a test.")
def show_number(value: Amount) -> str:
    return repr(value)


@node("show-level", title="Show level", description="For a test.")
def show_level(value: Literal["low", "high"]) -> str:
    return value


@node("parse-json", title="Parse json", description="For a test.")
def parse_json(text: str) -> JsonValue:
    return json.loads(text)


def _edge(source: Any, target: Any) -> Any:
    return registry_of(source, target).load(GraphRecord.model_validate({
        "nodes": [{"id": "a", "node": source.id}, {"id": "b", "node": target.id}], "edges": [["a.result", "b.value"]],
    }))


@pytest.mark.parametrize(("source", "target", "fits"), [
    (as_text, show_json, True), (as_yes, show_json, True), (as_low, show_json, True),
    (as_count, show_number, True), (as_yes, show_number, False), (as_text, show_number, False),
    (as_low, show_level, True), (as_text, show_level, False),
])
def test_an_edge_into_a_type_alias_or_a_literal_takes_what_its_values_may_be(source, target, fits):
    graph = _edge(source, target)

    assert [p.type for p in graph.problems] == ([] if fits else ["type_mismatch"])
    if fits:
        assert graph.run(text="x").status == "complete"


def test_a_json_null_is_a_value_of_a_json_input_and_never_a_blank():
    run = _edge(parse_json, show_json).run(text="null")

    assert run.outputs == {"b.result": "null"} and run.not_called == []


# -- 5. FromRun -----------------------------------------------------------------------------------



class Session:
    def __init__(self, who: str) -> None:
        self.who = who


@node("greet", title="Greet", description="For a test.")
def greet(text: str, session: FromRun[Session]) -> str:
    return f"{text} from {session.who}"


def test_from_run_is_supplied_not_wired_nor_fingerprinted():
    graph = registry_of(split, greet).load(GraphRecord.model_validate(chain("split", "greet")))
    assert list(graph.inputs) == ["text"] and "session" not in graph._nodes["greet"].definition.inputs
    try:
        graph.run(text="a")
        raise AssertionError
    except InvalidCallError as error:
        assert error.errors()[0].type == "from_run_missing"
    run = graph.with_inputs(text="a,b").resume(from_run={Session: Session("Bo")})
    assert run.results["greet"]["result"] == ["a from Bo", "b from Bo"]
    events = collect(graph.with_inputs(text="a"), from_run={Session: Session("Al")})
    assert events[-1].run.outputs == {"greet.result": ["a from Al"]}
    bad = reg.load(GraphRecord.model_validate({"nodes": [{"id": "g", "node": "greet"}]})) if "greet" in reg else None
    assert bad is None


# -- 6. exclusive_group -----------------------------------------------------------------------------




@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("pair-lists", title="Pair lists", description="For a test.")
def pair_lists(a: Series[str], b: Series[str]) -> str:
    return f"{a}|{b}"


@node("pair-optional", title="Pair optional", description="For a test.")
def pair_optional(a: Series[str | None], b: Series[str | None]) -> str:
    return f"{a}|{b}"


def test_two_series_on_opposite_branches_refused():
    reg6 = registry_of(split, decide, upper, pair_lists, pair_optional)
    record = {"nodes": [{"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "u", "node": "upper"}, {"id": "p", "node": "pair-lists"}],
              "edges": [["s.result", "d.text"], ["d.yes", "u.text"], ["u.result", "p.a"], ["d.no", "p.b"]]}
    graph = reg6.load(GraphRecord.model_validate(record))
    assert [p.type for p in graph.problems] == ["series_inputs_never_share_an_item"], graph.problems
    record["nodes"][3]["node"] = "pair-optional"
    graph = reg6.load(GraphRecord.model_validate(record))
    assert not graph.problems and graph.exclusive_groups == {"d": {"answer": ["yes", "no"]}}
    assert graph.run(text="yes,no").outputs == {"p.result": "['YES', None]|[None, 'no']"}
    try:
        @node("bad", title="Bad", description="For a test.")
        def bad(x: Annotated[str, Param(exclusive_group="g")]) -> str:
            return x
        raise AssertionError
    except TypeError:
        pass


# -- 7. describe -----------------------------------------------------------------------------------



def test_describe_for_a_palette_and_a_drawing():
    reg7 = registry_of(count, double, attach, size, formula, route, decide, greet, slow)
    palette = json.loads(reg7.describe().model_dump_json())
    nodes = {n["id"]: n for n in palette["nodes"]}
    assert nodes["slow"]["policy"]["concurrency"] == 8 and nodes["formula"]["takes_added_inputs"] and nodes["route"]["computes_outputs"]
    assert [f["name"] for f in nodes["greet"]["inputs"]] == ["text"]  # FromRun is no input
    assert nodes["decide"]["outputs"][0]["exclusive_group"] == "answer"
    types = {t["id"]: t["accepted_as"] for t in palette["types"]}
    assert "number" in types["integer"] and "integer" not in types["number"], types


def test_a_drawing_describes_the_added_inputs_of_a_node_one_of_which_nothing_feeds_yet():
    reg7 = registry_of(count, formula)
    drawing = reg7.load(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "count"}, {"id": "f", "node": "formula", "values": {"formula": "{a}+{b}"}}], "edges": [["c.result", "f.a"]]}))
    placed = json.loads(drawing.describe().model_dump_json())["nodes"]
    assert [f["field"]["name"] for f in placed["f"]["inputs"]] == ["formula", "a", "b"], placed["f"]


def test_a_hook_reading_a_value_nobody_typed_in_is_a_problem_and_its_node_does_not_vanish():
    def added(n: int) -> dict[str, Any]:
        return {f"x{i}": str for i in range(n)}

    @node("hooky", title="Hooky", description="For a test.", compute_added_inputs=added)
    def hooky(n: int, **values: str) -> str:
        return "called"

    graph = registry_of(hooky, upper).load(GraphRecord.model_validate({"nodes": [{"id": "h", "node": "hooky"}, {"id": "u", "node": "upper"}], "edges": [["h.result", "u.text"]]}))

    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("h", "n"))]


def test_a_hook_naming_a_field_no_address_can_say_is_a_problem():
    @node("dotted", title="Dotted", description="For a test.", compute_outputs=lambda: {"a.b": str})
    def dotted() -> Mapping[str, Any]:
        return {"a.b": "x"}

    graph = registry_of(dotted).load(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "dotted"}]}))

    assert [(p.type, p.loc) for p in graph.problems] == [("parameter_name_invalid", ("d", "a.b"))]


def test_a_timed_out_or_cancelled_leg_keeps_the_calls_already_on_threads():
    sent: list[str] = []

    @node("mail4", title="Mail4", description="For a test.", policy=Policy(concurrency=4))
    def mail4(text: str) -> str:
        t.sleep(0.2)
        sent.append(text)
        return text

    graph = registry_of(split, mail4).load(GraphRecord.model_validate(chain("split", "mail4")))
    stopped = graph.with_inputs(text="a,b,c,d").resume(timeout=0.05)
    finished = graph.load_run(stopped.to_json()).resume()

    assert stopped.status == "timed_out" and sorted(sent) == ["a", "b", "c", "d"]
    assert finished.status == "complete" and sorted(sent) == ["a", "b", "c", "d"]


def test_what_a_signature_says_wrong_is_refused_at_node_with_its_code():
    from conductor import NodeDeclarationError

    cases = {
        "default_invalid": lambda: node("a", title="A", description="For a test.")(_fn("x: int = None")),
        "nested_series": lambda: node("b", title="B", description="For a test.")(_fn("x: Series[Series[int]]")),
        "bound_invalid": lambda: node("d", title="D", description="For a test.")(_fn('x: Annotated[int, Param(ge="a")]')),
    }
    for code, declare in cases.items():
        try:
            declare()
        except NodeDeclarationError as error:
            assert error.code == code, (code, error.code)
        else:
            raise AssertionError(code)


def _fn(parameter: str, **names: Any) -> Any:
    """A node function taking ``parameter``, written from source so its annotation is read as written."""
    namespace: dict[str, Any] = {"Annotated": Annotated, "Param": Param, "Series": Series, **names}
    exec(f"def f({parameter}) -> str:\n    return repr(x)", namespace)
    return namespace["f"]


def test_a_field_reads_as_its_author_wrote_it():
    from conductor import Table

    @host_type("Whole", title="Whole")
    class Whole(int):
        pass

    @node("kinds", title="Kinds", description="For a test.")
    def kinds(
        whole: Whole = 3, maybe: Annotated[str, Param(title="Maybe")] | None = None, anything: Any = None,
        many: Series = (), some: Series[int] | None = None,
    ) -> str:
        return type(whole).__name__

    fields = {f.name: f for f in kinds.describe().inputs}
    assert fields["maybe"].title == "Maybe" and not fields["maybe"].many
    assert fields["many"].many and fields["some"].many
    graph = registry_of(kinds).load(GraphRecord.model_validate({"nodes": [{"id": "k", "node": "kinds"}]}))
    assert graph.run().outputs == {"k.result": "Whole"}
    assert Table is not None


def test_a_failed_run_has_no_outputs_but_its_results_say_what_it_did():
    from conductor import InvalidCallError

    @node("numbers", title="Numbers", description="For a test.")
    def numbers(n: list[int]) -> Series[int]:
        return n

    @node("each", title="Each", description="For a test.")
    def each(n: int) -> int:
        if n == 1:
            raise ValueError("no")
        return n

    run = registry_of(numbers, each).load(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "numbers"}, {"id": "e", "node": "each"}], "edges": [["n.result", "e.n"]]})).run(n=[0, 1, 2])

    assert run.status == "failed" and run.results["e"]["result"][0] == 0
    try:
        run.outputs
    except InvalidCallError as error:
        assert error.errors()[0].type == "run_not_complete"
    else:
        raise AssertionError("outputs of a failed run")


def test_the_last_progress_of_a_node_is_always_said():
    graph = registry_of(split, upper).load(GraphRecord.model_validate(chain("split", "upper")))
    events = collect(graph.with_inputs(text=",".join(str(i) for i in range(3000))))

    progress = [(e.done, e.total) for e in events if e.type == "node_progress"]
    assert progress[-1] == (3000, 3000)


# -- an embedded graph's nodes are reused as they would be drawn in the graph itself --------------


def test_again_with_one_of_an_embeddings_inputs_changed_calls_only_the_inner_node_that_reads_it_and_asks_nothing_again():
    from conductor import GraphRecord, NodeRecord

    calls = {"mark": 0, "confirm": 0}

    @node("mark", title="Mark", description="For a test.")
    def mark(a: str) -> str:
        calls["mark"] += 1
        return a + "!"

    @node("confirm", title="Confirm", description="For a test.")
    def confirm(b: str) -> bool:
        calls["confirm"] += 1
        raise Ask(f"ok {b}?")

    box = stored_graph("two", title="Two", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="x", node="mark"), NodeRecord(id="y", node="confirm")]))
    graph = registry_of(mark, confirm, box).load(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "two"}]}))
    first = graph.run({"e.a": "1", "e.b": "2"})
    first = first.answer(first.waiting[0], True).resume()
    before = dict(calls)

    again = first.again({"e.a": "changed"})

    assert again.status == "complete" and again.waiting == []
    assert (calls["mark"] - before["mark"], calls["confirm"] - before["confirm"]) == (1, 0)
