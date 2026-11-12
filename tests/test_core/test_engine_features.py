"""Events and pipelining, ``Policy``, hooks, a host's own types, branches (``exclusive_group``) and ``describe()``."""
import asyncio
import base64
import json
import re
import time as t
from asyncio import Event
from collections.abc import Mapping
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Annotated, Any, Literal, TypeAliasType

import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    Graph,
    GraphRecord,
    InvalidCallError,
    Param,
    Policy,
    Run,
    Series,
    host_type,
    node,
    subgraph,
)
from conductor._internal.scheduler import type as scheduler_module
from pydantic import AfterValidator, GetCoreSchemaHandler, JsonValue
from pydantic_core import core_schema
from pytest import MonkeyPatch

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


def chain(*ids, then_join=None):
    """A graph record of these nodes, each wired to the next one's ``text``; with ``then_join``, a ``join`` collects the last."""
    nodes = [{"id": i, "node": i} for i in ids]
    edges = [[f"{a}.result", f"{b}.text"] for a, b in zip(ids, ids[1:])]
    if then_join:
        nodes.append({"id": "join", "node": "join"})
        edges.append([f"{ids[-1]}.result", "join.texts"])
    return {"nodes": nodes, "edges": edges}


def test_events_in_order():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "upper", then_join=True)), reg)
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and events[-1].status == "complete", kinds
    complete = {e.loc[0]: e.outcomes for e in events if e.type == "node_complete"}
    assert complete == {"split": 1, "upper": 3, "join": 1}, complete
    assert kinds.index("node_start") < kinds.index("node_complete")
    assert events[-1].run.outputs == {"join.result": "A+B+C"}
    assert all(isinstance(e.model_dump(mode="json"), dict) for e in events)


def test_concurrency_eight_items_in_one_sleep():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "slow")), reg)
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    took = t.monotonic() - start
    assert took < 0.18 and run.results["slow"]["result"] == [c * 2 for c in "abcdefgh"], took


def test_pipelining_two_slow_nodes():
    """16 items through two slow nodes of 8 at once: node by node is 4 sleeps, pipelined 3."""
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "slow", "slow2", then_join=True)), reg)
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefghijklmnop"))
    took = t.monotonic() - start
    assert run.outputs["join.result"] == "+".join(c * 2 + "!" for c in "abcdefghijklmnop")
    assert took < 0.36, took


def test_stream_leaves_the_event_loop_free():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "slow")), reg)
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
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "split", "node": "split"}, {"id": "gate", "node": "gate"}, {"id": "never", "node": "never"}],
                      "edges": [["split.result", "gate.text"], ["split.result", "never.text"]]}), reg)
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert "node_waiting" in kinds and "node_skipped" in kinds and events[-1].status == "waiting", kinds
    waiting = [e.waiting for e in events if e.type == "node_waiting"]
    assert [place(w) for w in waiting] == ["gate@1"]
    assert not any(e.type == "node_complete" and e.loc[0] == "gate" for e in events)


def test_failure_keeps_calls_already_made():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "boom")), reg)
    run = graph.run(text="a,b,c,d")
    assert run.status == "failed" and run.state.cause.loc == ("boom", 2), run.state.cause
    assert all(run._outcomes["boom"].get((i,)) is not None for i in (0, 1, 3))  # d finished on its thread after c failed
    seen = []

    @node("boom", title="Fixed", description="For a test.", policy=Policy(concurrency=4))
    def fixed(text: str) -> str:
        seen.append(text)
        return text

    again = Run.from_json(run.to_json(), Graph.compile(GraphRecord.model_validate(chain("split", "boom")), registry_of(split, fixed))).resume()
    assert again.status == "complete" and seen == ["c"], seen  # nothing finished runs twice


def test_cancel_and_timeout():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "slow")), reg)
    run = graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).resume(timeout=0.15)
    assert run.status == "timed_out", run.status

    async def go():
        cancel = Event()
        events = []
        async for e in graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).stream(cancel=cancel):
            events.append(e)
            if e.type == "node_start":
                cancel.set()
        return events

    events = asyncio.run(go())
    assert events[-1].status == "cancelled", events[-1].status
    assert len(events[-1].run._outcomes["slow"]) < 24


def test_a_nested_loop_collects_each_documents_lines_into_its_own_join():
    """documents -> lines -> slow per line -> per-document join: each document gets its own lines."""

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    reg2 = registry_of(docs, split, slow, join)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "docs", "node": "docs"}, {"id": "split", "node": "split"}, {"id": "slow", "node": "slow"}, {"id": "join", "node": "join"}],
                       "edges": [["docs.result", "split.text"], ["split.result", "slow.text"], ["slow.result", "join.texts"]]}), reg2)
    run = graph.run(text="a,b|c|d,e,f")
    assert run.results["join"]["result"] == ["aa+bb", "cc", "dd+ee+ff"], run.results


# -- 2. Policy ------------------------------------------------------------------------------------


class Flaky(OSError):
    pass


def test_a_run_waiting_for_a_retry_looks_at_its_queue_without_blocking_before_each_timed_wait(monkeypatch: MonkeyPatch):
    """CPython 3.12's ``SimpleQueue.get(timeout=)`` can block for ever when it is the first look at an empty queue,
    so a run that waits for a retry would never retry: every timed wait follows a look that found nothing."""
    looks: list[str] = []

    class Watched(SimpleQueue[Any]):
        def get(self, block: bool = True, timeout: float | None = None) -> Any:
            looks.append("no wait" if not block else "timed" if timeout is not None else "wait")
            assert timeout is None or timeout > 0
            return super().get(block, timeout)

    monkeypatch.setattr(scheduler_module, "SimpleQueue", Watched)

    @node("fetch", title="Fetch", description="For a test.", policy=Policy(retries=1, retry_delay=0.01))
    def fetch(text: str) -> str:
        if not looks:
            raise ExternalError("busy")
        return text

    assert Graph.compile(GraphRecord.model_validate(chain("fetch")), registry_of(fetch)).run(text="a").outputs == {"fetch.result": "a"}
    assert "timed" in looks
    assert all(looks[i - 1] == "no wait" for i, look in enumerate(looks) if look == "timed")
    assert looks[0] == "no wait"


@pytest.mark.timeout(10)
@pytest.mark.parametrize("policy", [Policy(), Policy(concurrency=2)], ids=["on the run's thread", "on a pool thread"])
def test_an_exit_a_node_raises_ends_the_run_with_it_wherever_the_node_is_called(policy: Policy):
    @node("leave", title="Leave", description="For a test.", policy=policy)
    def leave(text: str) -> str:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        Graph.compile(GraphRecord.model_validate(chain("leave")), registry_of(leave)).run(text="a")


@pytest.mark.timeout(10)
@pytest.mark.parametrize("policy", [Policy(), Policy(concurrency=2)], ids=["on the run's thread", "on a pool thread"])
def test_a_defect_in_an_output_types_own_check_ends_the_run_with_it_wherever_the_node_is_called(policy: Policy):
    def explode(value: str) -> str:
        raise RuntimeError("the type's own check has a bug")

    @node("work", title="Work", description="For a test.", policy=policy)
    def work(text: str) -> Annotated[str, AfterValidator(explode)]:
        return text

    with pytest.raises(RuntimeError, match="has a bug"):
        Graph.compile(GraphRecord.model_validate(chain("work")), registry_of(work)).run(text="a")


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

    graph = Graph.compile(GraphRecord.model_validate(chain("split", "fetch", "client")), registry_of(split, fetch, client))
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

    run = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "down", "node": "down"}]}), registry_of(down)).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "external_failed" and calls == ["x", "x"], (run.state.cause, calls)
    calls.clear()
    run = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "bug", "node": "bug"}]}), registry_of(bug)).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "execution_failed" and calls == ["bug"]


def test_attempt_timeout_is_final():
    calls = []

    @node("hang", title="Hang", description="For a test.", policy=Policy(attempt_timeout=0.05, retries=2, retry_delay=0.01))
    def hang(text: str) -> str:
        calls.append(text)
        t.sleep(0.3)
        return text

    start = t.monotonic()
    run = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "hang", "node": "hang"}]}), registry_of(hang)).run(text="x")
    assert run.status == "failed" and run.state.cause.type == "timeout" and calls == ["x"] and t.monotonic() - start < 0.25, (run.state.cause, calls)


def test_reuse_when_inputs_match_false_calls_again():
    calls = []

    @node("rate", title="Rate", description="For a test.", policy=Policy(reuse_when_inputs_match=False))
    def rate(text: str) -> str:
        calls.append(text)
        return text

    graph = Graph.compile(GraphRecord.model_validate(chain("split", "rate", "upper")), registry_of(split, rate, upper))
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


def test_added_inputs_follow_the_formula_constant():
    reg3 = registry_of(number, formula)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "number"}, {"id": "f", "node": "formula", "constants": {"formula": "{a} * 10 + {b}", "b": 2}}],
                       "edges": [["n.result", "f.a"]]}), reg3)
    assert not graph.problems, graph.problems
    assert list(graph.inputs) == ["text"]
    run = graph.run(text="abc")
    assert run.outputs["f.result"] == 32.0, run.outputs
    broken = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "formula", "constants": {"formula": "{a}"}}, {"id": "g", "node": "formula"}], "edges": [["f.result", "g.zz"]]}), reg3)
    assert [p.type for p in broken.problems] == ["unknown_input"], broken.problems


def test_computed_outputs_take_the_type_that_arrives_on_their_input():
    reg3 = registry_of(split, number, route, half, upper)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "n", "node": "number"}, {"id": "r", "node": "route", "constants": {"cases": ["3"]}},
                                 {"id": "h", "node": "half"}],
                       "edges": [["s.result", "n.text"], ["n.result", "r.value"], ["r.3", "h.x"], ["r.otherwise", "h.x"]]}), reg3)
    assert not graph.problems, graph.problems
    assert graph.run(text="abc,de").results["h"]["result"] == [1.5, 1.0]
    wrong = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "number"}, {"id": "r", "node": "route"}, {"id": "u", "node": "upper"}],
                       "edges": [["n.result", "r.value"], ["r.otherwise", "u.text"]]}), reg3)
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
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "count"}, {"id": "d", "node": "double"}, {"id": "a", "node": "attach"}, {"id": "s", "node": "size"}],
                       "edges": [["c.result", "d.x"], ["d.result", "a.x"], ["a.result", "s.f"]]}), reg4)
    assert not graph.problems, graph.problems
    run = graph.run(text="abc")
    assert run.outputs == {"s.result": 3}
    again = Run.from_json(run.to_json(), graph)
    assert again.results["a"]["result"].content == b"6.0"
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double", "constants": {"x": "1,5"}}]}), reg4).run().outputs == {"d.result": 3.0}
    wrong = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double"}, {"id": "c", "node": "count"}], "edges": [["d.result", "c.text"]]}), reg4)
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
    return Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "a", "node": source.id}, {"id": "b", "node": target.id}], "edges": [["a.result", "b.value"]],
    }), registry_of(source, target))


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


def test_a_json_null_on_a_json_input_is_a_value_and_its_node_is_called():
    run = _edge(parse_json, show_json).run(text="null")

    assert run.outputs == {"b.result": "null"} and run.not_called == []


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
    graph = Graph.compile(GraphRecord.model_validate(record), reg6)
    assert [p.type for p in graph.problems] == ["lists_from_different_loops"], graph.problems
    record["nodes"][3]["node"] = "pair-optional"
    graph = Graph.compile(GraphRecord.model_validate(record), reg6)
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



def test_a_registrys_description_has_each_nodes_policy_hooks_branches_and_the_types_each_type_accepts():
    reg7 = registry_of(count, double, attach, size, formula, route, decide, slow)
    palette = json.loads(reg7.describe().model_dump_json())
    nodes = {n["id"]: n for n in palette["nodes"]}
    assert nodes["slow"]["policy"]["concurrency"] == 8 and nodes["formula"]["takes_added_inputs"] and nodes["route"]["computes_outputs"]
    assert nodes["decide"]["outputs"][0]["exclusive_group"] == "answer"
    types = {t["id"]: t["accepted_as"] for t in palette["types"]}
    assert "number" in types["integer"] and "integer" not in types["number"], types


def test_a_compiled_graph_describes_a_nodes_added_inputs_with_one_that_no_edge_reaches():
    reg7 = registry_of(count, formula)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "count"}, {"id": "f", "node": "formula", "constants": {"formula": "{a}+{b}"}}], "edges": [["c.result", "f.a"]]}), reg7)
    described = json.loads(graph.describe().model_dump_json())["nodes"]
    assert [f["field"]["name"] for f in described["f"]["inputs"]] == ["formula", "a", "b"], described["f"]


def test_a_hook_reading_an_input_that_has_no_constant_is_the_only_problem_of_its_node():
    def added(n: int) -> dict[str, Any]:
        return {f"x{i}": str for i in range(n)}

    @node("hooky", title="Hooky", description="For a test.", compute_added_inputs=added)
    def hooky(n: int, **values: str) -> str:
        return "called"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "h", "node": "hooky"}, {"id": "u", "node": "upper"}], "edges": [["h.result", "u.text"]]}), registry_of(hooky, upper))

    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("h", "n"))]


@pytest.mark.parametrize("name", ["a.b", "a/b", "a@b", "a~b", ""])
def test_a_hook_naming_a_field_with_a_character_of_an_address_is_a_problem_as_a_node_id_is(name: str):
    """An output name follows the node id rule: not empty, and none of the characters that addresses, locs and
    waiting ids use."""
    @node("dotted", title="Dotted", description="For a test.", compute_outputs=lambda: {name: str})
    def dotted() -> Mapping[str, Any]:
        return {name: "x"}

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "dotted"}]}), registry_of(dotted))

    assert [(p.type, p.loc) for p in graph.problems] == [("parameter_name_invalid", ("d", name))]
    assert [(p.type, p.loc) for p in Graph.compile(GraphRecord.model_validate({"nodes": [{"id": name, "node": "dotted"}]}), registry_of(dotted)).problems] == [("invalid_node_id", (name,))]


def test_a_timed_out_run_keeps_the_calls_already_on_threads():
    sent: list[str] = []

    @node("mail4", title="Mail4", description="For a test.", policy=Policy(concurrency=4))
    def mail4(text: str) -> str:
        t.sleep(0.2)
        sent.append(text)
        return text

    graph = Graph.compile(GraphRecord.model_validate(chain("split", "mail4")), registry_of(split, mail4))
    stopped = graph.with_inputs(text="a,b,c,d").resume(timeout=0.05)
    finished = Run.from_json(stopped.to_json(), graph).resume()

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
    assert fields["maybe"].title == "Maybe" and not fields["maybe"].is_series
    assert fields["many"].is_series and fields["some"].is_series
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "k", "node": "kinds"}]}), registry_of(kinds))
    assert graph.run().outputs == {"k.result": "Whole"}
    assert Table is not None


def test_a_failed_run_has_no_outputs_but_its_results_say_what_it_did():

    @node("numbers", title="Numbers", description="For a test.")
    def numbers(n: list[int]) -> Series[int]:
        return n

    @node("each", title="Each", description="For a test.")
    def each(n: int) -> int:
        if n == 1:
            raise ValueError("no")
        return n

    run = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "numbers"}, {"id": "e", "node": "each"}], "edges": [["n.result", "e.n"]]}), registry_of(numbers, each)).run(n=[0, 1, 2])

    assert run.status == "failed" and run.results["e"]["result"][0] == 0
    try:
        run.outputs
    except InvalidCallError as error:
        assert error.errors()[0].type == "run_not_complete"
    else:
        raise AssertionError("outputs of a failed run")


def test_the_last_progress_of_a_node_is_always_said():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "upper")), registry_of(split, upper))
    events = collect(graph.with_inputs(text=",".join(str(i) for i in range(3000))))

    progress = [(e.finished, e.total) for e in events if e.type == "node_progress"]
    assert progress[-1] == (3000, 3000)


# -- a subgraph's inner nodes are reused as they would be in the outer graph itself --------------


def test_again_with_one_subgraph_input_changed_calls_only_the_inner_node_that_reads_it_and_asks_nothing_again():
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

    box = subgraph("two", title="Two", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="x", node="mark"), NodeRecord(id="y", node="confirm")]))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "two"}]}), registry_of(mark, confirm, box))
    first = graph.run({"e.a": "1", "e.b": "2"})
    first = first.answer(first.waiting[0], True).resume()
    before = dict(calls)

    again = first.again({"e.a": "changed"})

    assert again.status == "complete" and again.waiting == []
    assert (calls["mark"] - before["mark"], calls["confirm"] - before["confirm"]) == (1, 0)
