"""Events and pipelining, Policy, hooks, a host's own types, FromRun, exclusive groups and describe(), through the public door."""
import asyncio
import base64
import json
import re
import time as t
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from conductor import (
    SKIPPED,
    Ask,
    ExternalFailure,
    FromRun,
    InvalidCallError,
    Param,
    Policy,
    Series,
    node,
)
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from tests.test_core.nodes import place, registry_of


def collect(run, **kw):
    async def go():
        return [e async for e in run.stream(**kw)]

    return asyncio.run(go())


# -- 1. events, concurrency, pipelining ----------------------------------------------------------


@node("split")
def split(text: str) -> Series[str]:
    return text.split(",")


@node("upper")
def upper(text: str) -> str:
    return text.upper()


@node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("slow", policy=Policy(concurrency=8))
def slow(text: str) -> str:
    t.sleep(0.1)
    return text * 2


@node("slow2", policy=Policy(concurrency=8))
def slow2(text: str) -> str:
    t.sleep(0.1)
    return text + "!"


@node("gate")
def gate(text: str) -> str:
    if text == "b":
        raise Ask("ok?")
    return text


@node("never")
def never(text: str) -> str:
    return SKIPPED


@node("boom", policy=Policy(concurrency=4))
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
    graph = reg.load(chain("split", "upper", gather=True))
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and events[-1].status == "complete", kinds
    complete = {e.node_id: e.rows for e in events if e.type == "node_complete"}
    assert complete == {"split": 1, "upper": 3, "join": 1}, complete
    assert kinds.index("node_start") < kinds.index("node_complete")
    assert events[-1].run.outputs == {"join.result": "A+B+C"}
    assert all(isinstance(e.model_dump(mode="json"), dict) for e in events)


def test_concurrency_eight_rows_in_one_sleep():
    graph = reg.load(chain("split", "slow"))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    took = t.monotonic() - start
    assert took < 0.18 and run.results["slow"]["result"] == [c * 2 for c in "abcdefgh"], took


def test_pipelining_two_slow_nodes():
    """16 rows through two slow nodes of 8 at once: node by node is 4 sleeps, pipelined 3."""
    graph = reg.load(chain("split", "slow", "slow2", gather=True))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefghijklmnop"))
    took = t.monotonic() - start
    assert run.outputs["join.result"] == "+".join(c * 2 + "!" for c in "abcdefghijklmnop")
    assert took < 0.36, took


def test_stream_leaves_the_event_loop_free():
    graph = reg.load(chain("split", "slow"))
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
    graph = reg.load({"nodes": [{"id": "split", "node": "split"}, {"id": "gate", "node": "gate"}, {"id": "never", "node": "never"}],
                      "edges": [["split.result", "gate.text"], ["split.result", "never.text"]]})
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert "unit_waiting" in kinds and "node_skipped" in kinds and events[-1].status == "waiting", kinds
    waiting = [e.unit for e in events if e.type == "unit_waiting"]
    assert [place(w) for w in waiting] == ["gate@1"]
    assert not any(e.type == "node_complete" and e.node_id == "gate" for e in events)


def test_failure_keeps_calls_already_made():
    graph = reg.load(chain("split", "boom"))
    run = graph.run(text="a,b,c,d")
    assert run.status == "failed" and run.cause.loc == ("boom", 2), run.cause
    assert all(run._units["boom"].get((i,)) is not None for i in (0, 1, 3))  # d finished on its thread after c failed
    seen = []

    @node("boom", policy=Policy(concurrency=4))
    def fixed(text: str) -> str:
        seen.append(text)
        return text

    again = registry_of(split, fixed).load(chain("split", "boom")).load_run(run.to_json()).resume()
    assert again.status == "complete" and seen == ["c"], seen  # nothing finished runs twice


def test_cancel_and_timeout():
    graph = reg.load(chain("split", "slow"))
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
    assert len(events[-1].run._units["slow"]) < 24


def test_gather_per_document_while_rows_run():
    """documents -> lines -> slow per line -> per-document join: each document gets its own lines."""

    @node("docs")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    reg2 = registry_of(docs, split, slow, join)
    graph = reg2.load({"nodes": [{"id": "docs", "node": "docs"}, {"id": "split", "node": "split"}, {"id": "slow", "node": "slow"}, {"id": "join", "node": "join"}],
                       "edges": [["docs.result", "split.text"], ["split.result", "slow.text"], ["slow.result", "join.texts"]]})
    run = graph.run(text="a,b|c|d,e,f")
    assert run.results["join"]["result"] == ["aa+bb", "cc", "dd+ee+ff"], run.results


# -- 2. Policy ------------------------------------------------------------------------------------


class Flaky(OSError):
    pass


def test_retries_external_failures_then_succeeds():
    tries: dict[str, int] = {}

    @node("fetch", policy=Policy(retries=2, retry_delay=0.01))
    def fetch(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if tries[text] < 3:
            raise ExternalFailure("busy")
        return text

    @node("client", policy=Policy(retries=1, retry_delay=0.01, retry_on=(Flaky,), concurrency=2))
    def client(text: str) -> str:
        tries["c" + text] = tries.get("c" + text, 0) + 1
        if tries["c" + text] < 2:
            raise Flaky("reset")
        return text + "!"

    graph = registry_of(split, fetch, client).load(chain("split", "fetch", "client"))
    events = collect(graph.with_inputs(text="a,b"))
    retries = [e for e in events if e.type == "node_retry"]
    assert events[-1].status == "complete" and events[-1].run.results["client"]["result"] == ["a!", "b!"], events[-1].run.cause
    assert sorted((e.node_id, e.row, e.attempt) for e in retries) == [("client", (0,), 1), ("client", (1,), 1), ("fetch", (0,), 1), ("fetch", (0,), 2), ("fetch", (1,), 1), ("fetch", (1,), 2)]
    assert retries[0].cause.type == "external_failed" and retries[0].delay == 0.01


def test_retries_run_out_and_plain_errors_never_retry():
    calls = []

    @node("down", policy=Policy(retries=1, retry_delay=0.01))
    def down(text: str) -> str:
        calls.append(text)
        raise ExternalFailure("down")

    @node("bug", policy=Policy(retries=3, retry_delay=0.01))
    def bug(text: str) -> str:
        calls.append("bug")
        raise KeyError(text)

    run = registry_of(down).load({"nodes": [{"id": "down", "node": "down"}]}).run(text="x")
    assert run.status == "failed" and run.cause.type == "external_failed" and calls == ["x", "x"], (run.cause, calls)
    calls.clear()
    run = registry_of(bug).load({"nodes": [{"id": "bug", "node": "bug"}]}).run(text="x")
    assert run.status == "failed" and run.cause.type == "execution_failed" and calls == ["bug"]


def test_attempt_timeout_is_final():
    calls = []

    @node("hang", policy=Policy(attempt_timeout=0.05, retries=2, retry_delay=0.01))
    def hang(text: str) -> str:
        calls.append(text)
        t.sleep(0.3)
        return text

    start = t.monotonic()
    run = registry_of(hang).load({"nodes": [{"id": "hang", "node": "hang"}]}).run(text="x")
    assert run.status == "failed" and run.cause.type == "timeout" and calls == ["x"] and t.monotonic() - start < 0.25, (run.cause, calls)


def test_reuse_when_inputs_match_false_calls_again():
    calls = []

    @node("rate", policy=Policy(reuse_when_inputs_match=False))
    def rate(text: str) -> str:
        calls.append(text)
        return text

    graph = registry_of(split, rate, upper).load(chain("split", "rate", "upper"))
    first = graph.run(text="a,b")
    calls.clear()
    first.again(text="a,b")
    assert calls == ["a", "b"], calls


def test_policy_bounds_checked():
    try:
        Policy(concurrency=0)
        raise AssertionError
    except ValueError:
        pass


# -- 3. hooks ------------------------------------------------------------------------------------



def _formula_inputs(formula: str) -> dict[str, Any]:
    return {name: float for name in re.findall(r"{(\w+)}", formula)}


@node("formula", compute_added_inputs=_formula_inputs)
def formula(formula: str = "", **values: float) -> float:
    return eval(formula.format(**values))  # a check, not a calculator


def _branch_outputs(cases: list[str], value: type) -> dict[str, Any]:
    from typing import Annotated

    from conductor import Param

    return {name: Annotated[value, Param(title=name.title(), exclusive_group="case")] for name in [*cases, "otherwise"]}


@node("route", compute_outputs=_branch_outputs)
def route(value: Any, cases: list[str] = []) -> Mapping[str, Any]:  # noqa: B006
    taken = next((c for c in cases if c in str(value)), "otherwise")
    return {name: value if name == taken else SKIPPED for name in [*cases, "otherwise"]}


@node("number")
def number(text: str) -> int:
    return len(text)


@node("half")
def half(x: float) -> float:
    return x / 2


def test_added_inputs_from_typed_value():
    reg3 = registry_of(number, formula)
    graph = reg3.load({"nodes": [{"id": "n", "node": "number"}, {"id": "f", "node": "formula", "values": {"formula": "{a} * 10 + {b}", "b": 2}}],
                       "edges": [["n.result", "f.a"]]})
    assert not graph.problems, graph.problems
    assert list(graph.inputs) == ["text"]
    run = graph.run(text="abc")
    assert run.outputs["f.result"] == 32.0, run.outputs
    broken = reg3.load({"nodes": [{"id": "f", "node": "formula", "values": {"formula": "{a}"}}, {"id": "g", "node": "formula"}], "edges": [["f.result", "g.zz"]]})
    assert [p.type for p in broken.problems] == ["unknown_input"], broken.problems


def test_outputs_typed_by_what_arrives():
    reg3 = registry_of(split, number, route, half, upper)
    graph = reg3.load({"nodes": [{"id": "s", "node": "split"}, {"id": "n", "node": "number"}, {"id": "r", "node": "route", "values": {"cases": ["3"]}},
                                 {"id": "h", "node": "half"}],
                       "edges": [["s.result", "n.text"], ["n.result", "r.value"], ["r.3", "h.x"], ["r.otherwise", "h.x"]]})
    assert not graph.problems, graph.problems
    assert graph.run(text="abc,de").results["h"]["result"] == [1.5, 1.0]
    wrong = reg3.load({"nodes": [{"id": "n", "node": "number"}, {"id": "r", "node": "route"}, {"id": "u", "node": "upper"}],
                       "edges": [["n.result", "r.value"], ["r.otherwise", "u.text"]]})
    assert [p.type for p in wrong.problems] == ["type_mismatch"], wrong.problems


# -- 4. a host's own types ------------------------------------------------------------------------




class Integer(int):
    id = "integer"


class Number(float):
    """Reads the Danish decimal comma; takes an Integer."""

    id = "number"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v.replace(",", ".")) if isinstance(v, str) else cls(v), core_schema.float_schema())

    @classmethod
    def accepts(cls, source: type) -> bool:
        return issubclass(source, Integer)


class File:
    id = "file"

    def __init__(self, name: str, content: bytes) -> None:
        self.name, self.content = name, content

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, File) else File(v["name"], base64.b64decode(v["content"])),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda f: {"name": f.name, "content": base64.b64encode(f.content).decode()}, when_used="json"))


@node("count")
def count(text: str) -> Integer:
    return Integer(len(text))


@node("double")
def double(x: Number) -> Number:
    return Number(x * 2)


@node("attach")
def attach(x: Number) -> File:
    return File("x.txt", str(x).encode())


@node("size")
def size(f: File) -> int:
    return len(f.content)


def test_own_types_accept_and_round_trip():
    reg4 = registry_of(count, double, attach, size, upper)
    graph = reg4.load({"nodes": [{"id": "c", "node": "count"}, {"id": "d", "node": "double"}, {"id": "a", "node": "attach"}, {"id": "s", "node": "size"}],
                       "edges": [["c.result", "d.x"], ["d.result", "a.x"], ["a.result", "s.f"]]})
    assert not graph.problems, graph.problems
    run = graph.run(text="abc")
    assert run.outputs == {"s.result": 3}
    again = graph.load_run(run.to_json())
    assert again.results["a"]["result"].content == b"6.0"
    assert reg4.load({"nodes": [{"id": "d", "node": "double", "values": {"x": "1,5"}}]}).run().outputs == {"d.result": 3.0}
    wrong = reg4.load({"nodes": [{"id": "d", "node": "double"}, {"id": "c", "node": "count"}], "edges": [["d.result", "c.text"]]})
    assert [p.type for p in wrong.problems] == ["type_mismatch"]


# -- 5. FromRun -----------------------------------------------------------------------------------



class Session:
    def __init__(self, who: str) -> None:
        self.who = who


@node("greet")
def greet(text: str, session: FromRun[Session]) -> str:
    return f"{text} from {session.who}"


def test_from_run_is_supplied_not_wired_nor_fingerprinted():
    graph = registry_of(split, greet).load(chain("split", "greet"))
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
    bad = reg.load({"nodes": [{"id": "g", "node": "greet"}]}) if "greet" in reg else None
    assert bad is None


# -- 6. exclusive_group -----------------------------------------------------------------------------




@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("pair-lists")
def pair_lists(a: Series[str], b: Series[str]) -> str:
    return f"{a}|{b}"


@node("pair-optional")
def pair_optional(a: Series[str | None], b: Series[str | None]) -> str:
    return f"{a}|{b}"


def test_two_series_on_opposite_branches_refused():
    reg6 = registry_of(split, decide, upper, pair_lists, pair_optional)
    record = {"nodes": [{"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "u", "node": "upper"}, {"id": "p", "node": "pair-lists"}],
              "edges": [["s.result", "d.text"], ["d.yes", "u.text"], ["u.result", "p.a"], ["d.no", "p.b"]]}
    graph = reg6.load(record)
    assert [p.type for p in graph.problems] == ["series_inputs_never_share_a_row"], graph.problems
    record["nodes"][3]["node"] = "pair-optional"
    graph = reg6.load(record)
    assert not graph.problems and graph.exclusive_groups == {"d": {"answer": ["yes", "no"]}}
    assert graph.run(text="yes,no").outputs == {"p.result": "['YES', None]|[None, 'no']"}
    try:
        @node("bad")
        def bad(x: Annotated[str, Param(exclusive_group="g")]) -> str:
            return x
        raise AssertionError
    except TypeError:
        pass


# -- 7. describe -----------------------------------------------------------------------------------



def test_describe_for_a_palette_and_a_drawing():
    reg7 = registry_of(count, double, attach, size, formula, route, decide, greet, slow)
    palette = json.loads(json.dumps(reg7.describe()))
    nodes = {n["id"]: n for n in palette["nodes"]}
    assert nodes["slow"]["policy"]["concurrency"] == 8 and nodes["formula"]["takes_added_inputs"] and nodes["route"]["computes_outputs"]
    assert [f["name"] for f in nodes["greet"]["inputs"]] == ["text"]  # FromRun is no input
    assert nodes["decide"]["outputs"][0]["exclusive_group"] == "answer"
    types = {t["id"]: t["accepted_as"] for t in palette["types"]}
    assert "number" in types["integer"] and "integer" not in types["number"], types
    drawing = reg7.load({"nodes": [{"id": "c", "node": "count"}, {"id": "f", "node": "formula", "values": {"formula": "{a}+{b}"}}], "edges": [["c.result", "f.a"]]})
    placed = json.loads(json.dumps(drawing.describe()))["nodes"]
    assert [f["name"] for f in placed["f"]["inputs"]] == ["formula", "a", "b"], placed["f"]


def test_a_hook_reading_a_value_nobody_typed_in_is_a_problem_and_its_node_does_not_vanish():
    def added(n: int) -> dict[str, Any]:
        return {f"x{i}": str for i in range(n)}

    @node("hooky", compute_added_inputs=added)
    def hooky(n: int, **values: str) -> str:
        return "called"

    graph = registry_of(hooky, upper).load({"nodes": [{"id": "h", "node": "hooky"}, {"id": "u", "node": "upper"}], "edges": [["h.result", "u.text"]]})

    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("h", "n"))]


def test_a_hook_naming_a_field_no_address_can_say_is_a_problem():
    @node("dotted", compute_outputs=lambda: {"a.b": str})
    def dotted() -> Mapping[str, Any]:
        return {"a.b": "x"}

    graph = registry_of(dotted).load({"nodes": [{"id": "d", "node": "dotted"}]})

    assert [(p.type, p.loc) for p in graph.problems] == [("parameter_name_invalid", ("d", "a.b"))]


def test_a_timed_out_or_cancelled_leg_keeps_the_calls_already_on_threads():
    sent: list[str] = []

    @node("mail4", policy=Policy(concurrency=4))
    def mail4(text: str) -> str:
        t.sleep(0.2)
        sent.append(text)
        return text

    graph = registry_of(split, mail4).load(chain("split", "mail4"))
    stopped = graph.with_inputs(text="a,b,c,d").resume(timeout=0.05)
    finished = graph.load_run(stopped.to_json()).resume()

    assert stopped.status == "timed_out" and sorted(sent) == ["a", "b", "c", "d"]
    assert finished.status == "complete" and sorted(sent) == ["a", "b", "c", "d"]


def test_what_a_signature_says_wrong_is_refused_at_node_with_its_code():
    from conductor import NodeDeclarationError, Table
    from conductor.widgets import TextInput

    cases = {
        "default_invalid": lambda: node("a")(_fn("x: int = None")),
        "nested_series": lambda: node("b")(_fn("x: Series[Series[int]]")),
        "annotated_extra_unknown": lambda: node("c")(_fn("x: Annotated[int, TextInput()]", TextInput=TextInput)),
        "bound_invalid": lambda: node("d")(_fn('x: Annotated[int, Param(ge="a")]')),
    }
    for code, declare in cases.items():
        try:
            declare()
        except NodeDeclarationError as error:
            assert error.code == code, (code, error.code)
        else:
            raise AssertionError(code)
    try:
        Table()
    except TypeError:
        pass
    else:
        raise AssertionError("Table() built an instance")


def _fn(parameter: str, **names: Any) -> Any:
    """A node function taking ``parameter``, written from source so its annotation is read as written."""
    namespace: dict[str, Any] = {"Annotated": Annotated, "Param": Param, "Series": Series, **names}
    exec(f"def f({parameter}) -> str:\n    return repr(x)", namespace)
    return namespace["f"]


def test_a_field_reads_as_its_author_wrote_it():
    from conductor import Table

    class Whole(int):
        pass

    @node("kinds")
    def kinds(
        whole: Whole = 3, maybe: Annotated[str, Param(title="Maybe")] | None = None, anything: Any = None,
        many: Series = (), some: Series[int] | None = None,
    ) -> str:
        return type(whole).__name__

    fields = {f["name"]: f for f in kinds.describe()["inputs"]}
    assert fields["maybe"]["title"] == "Maybe" and fields["maybe"]["shape"] == "one"
    assert fields["many"]["shape"] == "many" and fields["some"]["shape"] == "many"
    graph = registry_of(kinds).load({"nodes": [{"id": "k", "node": "kinds"}]})
    assert graph.run().outputs == {"k.result": "Whole"}
    assert Table is not None


def test_a_failed_run_has_no_outputs_but_its_results_say_what_it_did():
    from conductor import InvalidCallError

    @node("each")
    def each(n: int) -> int:
        if n == 1:
            raise ValueError("no")
        return n

    run = registry_of(each).load({"nodes": [{"id": "e", "node": "each", "many": ["n"]}]}).run(n=[0, 1, 2])

    assert run.status == "failed" and run.results["e"]["result"][0] == 0
    try:
        run.outputs
    except InvalidCallError as error:
        assert error.errors()[0].type == "run_not_complete"
    else:
        raise AssertionError("outputs of a failed run")


def test_the_last_progress_of_a_node_is_always_said():
    graph = registry_of(split, upper).load(chain("split", "upper"))
    events = collect(graph.with_inputs(text=",".join(str(i) for i in range(3000))))

    progress = [(e.done, e.total) for e in events if e.type == "node_progress"]
    assert progress[-1] == (3000, 3000)
