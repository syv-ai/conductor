"""The engine's larger features, each in a section of its own.

The sections are: events and pipelining, ``Policy``, hooks, a host's own types, branches,
``describe()``, what a node's signature declares, and reuse inside a subgraph.
"""

import asyncio
import base64
import json
import re
import time as t
from asyncio import Event
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from threading import Barrier
from threading import Event as ThreadEvent
from typing import Annotated, Any, Literal, TypeAliasType, TypedDict

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
    Table,
    host_type,
    node,
    subgraph,
)
from pydantic import AfterValidator, BaseModel, GetCoreSchemaHandler, JsonValue
from pydantic_core import core_schema

from tests.test_core.nodes import place, registry_of

#: How many seconds a test waits for another thread. Past this it fails instead of hanging.
_BOUND = 10.0


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


@node("gate", title="Gate", description="For a test.", asks=True)
def gate(text: str) -> str:
    if text == "b":
        raise Ask("ok?")
    return text


@node("never", title="Never", description="For a test.")
def never(text: str) -> str:
    return SKIPPED


#: Holds each of ``boom``'s four calls until all have started, so ``d`` is on its thread when ``c`` fails.
_all_four_started = Barrier(4)


@node("boom", title="Boom", description="For a test.", policy=Policy(concurrency=4))
def boom(text: str) -> str:
    _all_four_started.wait(10)
    if text == "c":
        raise ValueError("no c")
    t.sleep(0.05)
    return text


reg = registry_of(split, upper, join, slow, gate, never, boom)


def chain(*ids, then_join=None):
    """Returns a graph record that chains the nodes ``ids``, each one's output wired to the next one's ``text``.

    With ``then_join``, a ``join`` node is added at the end to collect the last node's results.
    """
    nodes: dict[str, Any] = {ids[0]: {"node": ids[0], "inputs": {"text": {"graph_input": "text"}}}}
    nodes |= {
        b: {"node": b, "inputs": {"text": {"sources": [{"node": a, "output": "result"}]}}}
        for a, b in zip(ids, ids[1:])
    }
    if then_join:
        nodes["join"] = {
            "node": "join",
            "inputs": {"texts": {"sources": [{"node": ids[-1], "output": "result"}]}},
        }
    return {
        "nodes": nodes,
        "outputs": {"result": {"node": "join" if then_join else ids[-1], "output": "result"}},
    }


def test_events_in_order():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "upper", then_join=True)), reg)
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and events[-1].status == "complete", (
        kinds
    )
    complete = {e.loc[0]: e.count.finished for e in events if e.type == "node_complete"}
    assert complete == {"split": 1, "upper": 3, "join": 1}, complete
    assert kinds.index("node_start") < kinds.index("node_complete")
    assert events[-1].run.outputs == {"result": "A+B+C"}
    assert all(isinstance(e.model_dump(mode="json"), dict) for e in events)


def test_concurrency_eight_calls_of_a_node_run_at_once():
    """A node with ``concurrency=8`` runs eight calls at once.

    Each call waits until all eight have started, so the test fails if fewer run together.
    """
    eight = Barrier(8)

    @node("together", title="Together", description="For a test.", policy=Policy(concurrency=8))
    def together(text: str) -> str:
        eight.wait(_BOUND)  # times out and fails the call if fewer than eight run at once
        return text * 2

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "together")), registry_of(split, together)
    )
    run = graph.run(text=",".join("abcdefgh"))
    assert run.status == "complete", run.cause
    assert run.results["together"]["result"] == [c * 2 for c in "abcdefgh"]


def test_pipelining_two_nodes():
    """An item goes on to the second node as soon as the first node is done with it.

    The second node doesn't wait for the first to finish every item. The test proves it: the first node's last
    item waits until the second node has been called.
    """
    second_called = ThreadEvent()
    seen_before_the_last: list[bool] = []

    @node("first", title="First", description="For a test.", policy=Policy(concurrency=8))
    def first(text: str) -> str:
        if text == "p":
            seen_before_the_last.append(second_called.wait(_BOUND))
        return text * 2

    @node("second", title="Second", description="For a test.", policy=Policy(concurrency=8))
    def second(text: str) -> str:
        second_called.set()
        return text + "!"

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "first", "second", then_join=True)),
        registry_of(split, first, second, join),
    )
    run = graph.run(text=",".join("abcdefghijklmnop"))
    assert run.outputs["result"] == "+".join(c * 2 + "!" for c in "abcdefghijklmnop")
    assert seen_before_the_last == [True]


def test_stream_leaves_the_event_loop_free():
    """While ``stream()`` waits on a node's call, other tasks on the event loop run.

    The node waits until a task on the loop releases it, and that task only does so once the node has started. If
    the stream held the loop during the call, the task could never run, and the node would give up waiting. The
    timeout is only there so that a stream that holds the loop fails the test instead of hanging it.
    """
    started, released = ThreadEvent(), ThreadEvent()

    @node("held", title="Held", description="For a test.")
    def held(text: str) -> bool:
        started.set()
        return released.wait(timeout=30)

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "held")), registry_of(split, held)
    )

    async def go():
        async def release():
            while not started.is_set():
                await asyncio.sleep(0)
            released.set()

        releaser = asyncio.create_task(release())
        events = [e async for e in graph.with_inputs(text="a").stream()]
        await releaser
        return events[-1].run

    run = asyncio.run(go())
    assert (run.status, run.results["held"]["result"]) == ("complete", [True])


def test_waiting_and_skipped_events():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "split": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
                    "gate": {
                        "node": "gate",
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                    "never": {
                        "node": "never",
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                },
                "outputs": {
                    "gate_result": {"node": "gate", "output": "result"},
                    "never_result": {"node": "never", "output": "result"},
                },
            }
        ),
        reg,
    )
    events = collect(graph.with_inputs(text="a,b,c"))
    kinds = [e.type for e in events]
    assert "node_waiting" in kinds and events[-1].status == "waiting", kinds
    assert [
        (e.count.finished, e.count.skipped)
        for e in events
        if e.type == "node_complete" and e.loc == ("never",)
    ] == [(3, 3)]
    waiting = [e.waiting for e in events if e.type == "node_waiting"]
    assert [place(w) for w in waiting] == ["gate@1"]
    assert not any(e.type == "node_complete" and e.loc[0] == "gate" for e in events)


def test_failure_keeps_calls_already_made():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "boom")), reg)
    run = graph.run(text="a,b,c,d")
    assert run.status == "failed" and run.cause.loc == ("boom", 2), run.cause
    # d was already running on its thread when c failed, and it finished
    assert run.results["boom"]["result"] == ["a", "b", None, "d"]
    seen = []

    @node("boom", title="Fixed", description="For a test.", policy=Policy(concurrency=4))
    def fixed(text: str) -> str:
        seen.append(text)
        return text

    again = Run.from_json(
        run.to_json(),
        Graph.compile(
            GraphRecord.model_validate(chain("split", "boom")), registry_of(split, fixed)
        ),
    ).resume()
    assert again.status == "complete" and seen == ["c"], seen  # only the failed call runs again


def test_cancel_and_timeout():
    graph = Graph.compile(GraphRecord.model_validate(chain("split", "slow")), reg)
    run = graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).resume(timeout=0.15)
    assert run.status == "timed_out", run.status

    async def go():
        cancel = Event()
        events = []
        async for e in graph.with_inputs(text=",".join("abcdefghijklmnopqrstuvwx")).stream(
            cancel=cancel
        ):
            events.append(e)
            if e.type == "node_start":
                cancel.set()
        return events

    events = asyncio.run(go())
    assert events[-1].status == "cancelled", events[-1].status
    assert events[-1].run.progress.nodes["slow"].finished < 24


def test_a_nested_loop_collects_each_documents_lines_into_its_own_join():
    """Documents are split into lines, and each document's lines are joined on their own.

    The graph goes documents, then lines, then ``slow`` per line, then a ``join`` per document.
    """

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    reg2 = registry_of(docs, split, slow, join)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "docs": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}},
                    "split": {
                        "node": "split",
                        "inputs": {"text": {"sources": [{"node": "docs", "output": "result"}]}},
                    },
                    "slow": {
                        "node": "slow",
                        "inputs": {"text": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                    "join": {
                        "node": "join",
                        "inputs": {"texts": {"sources": [{"node": "slow", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "join", "output": "result"}},
            }
        ),
        reg2,
    )
    run = graph.run(text="a,b|c|d,e,f")
    assert run.results["join"]["result"] == ["aa+bb", "cc", "dd+ee+ff"], run.results


def test_a_timed_out_run_keeps_the_calls_already_on_threads():
    sent: list[str] = []

    @node("mail4", title="Mail4", description="For a test.", policy=Policy(concurrency=4))
    def mail4(text: str) -> str:
        t.sleep(0.2)
        sent.append(text)
        return text

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "mail4")), registry_of(split, mail4)
    )
    # four calls are running on threads when time runs out, and four never start
    stopped = graph.with_inputs(text="a,b,c,d,e,f,g,h").resume(timeout=0.05)
    assert stopped.status == "timed_out" and sorted(sent) == ["a", "b", "c", "d"]

    finished = Run.from_json(stopped.to_json(), graph).resume()
    assert finished.status == "complete" and sorted(sent) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
    ]


def test_a_failed_run_has_no_outputs_but_its_results_say_what_it_did():

    @node("numbers", title="Numbers", description="For a test.")
    def numbers(n: list[int]) -> Series[int]:
        return n

    @node("each", title="Each", description="For a test.")
    def each(n: int) -> int:
        if n == 1:
            raise ValueError("no")
        return n

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "n": {"node": "numbers", "inputs": {"n": {"graph_input": "n"}}},
                    "e": {
                        "node": "each",
                        "inputs": {"n": {"sources": [{"node": "n", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "e", "output": "result"}},
            }
        ),
        registry_of(numbers, each),
    ).run(n=[0, 1, 2])

    assert run.status == "failed" and run.results["e"]["result"][0] == 0
    try:
        run.outputs
    except InvalidCallError as error:
        assert error.problems[0].type == "run_not_complete"
    else:
        raise AssertionError("outputs of a failed run")


def test_the_last_progress_event_of_a_node_is_always_sent():
    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "upper")), registry_of(split, upper)
    )
    events = collect(graph.with_inputs(text=",".join(str(i) for i in range(3000))))

    progress = [(e.count.finished, e.count.total) for e in events if e.type == "node_progress"]
    assert progress[-1] == (3000, 3000)


# -- 2. Policy ------------------------------------------------------------------------------------


class Flaky(OSError):
    pass


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "policy", [Policy(), Policy(concurrency=2)], ids=["on the run's thread", "on a pool thread"]
)
def test_an_exit_a_node_raises_ends_the_run_with_it_wherever_the_node_is_called(policy: Policy):
    @node("leave", title="Leave", description="For a test.", policy=policy)
    def leave(text: str) -> str:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        Graph.compile(GraphRecord.model_validate(chain("leave")), registry_of(leave)).run(text="a")


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "policy", [Policy(), Policy(concurrency=2)], ids=["on the run's thread", "on a pool thread"]
)
def test_a_defect_in_an_output_types_own_check_ends_the_run_with_it_wherever_the_node_is_called(
    policy: Policy,
):
    def explode(value: str) -> str:
        raise RuntimeError("the type's own check has a bug")

    @node("work", title="Work", description="For a test.", policy=policy)
    def work(text: str) -> Annotated[str, AfterValidator(explode)]:
        return text

    with pytest.raises(RuntimeError, match="has a bug"):
        Graph.compile(GraphRecord.model_validate(chain("work")), registry_of(work)).run(text="a")


def test_retries_external_failures_then_succeeds():
    tries: dict[str, int] = {}

    @node(
        "fetch",
        title="Fetch",
        description="For a test.",
        policy=Policy(retries=2, retry_delay=0.01),
    )
    def fetch(text: str) -> str:
        tries[text] = tries.get(text, 0) + 1
        if tries[text] < 3:
            raise ExternalError("busy")
        return text

    @node(
        "client",
        title="Client",
        description="For a test.",
        policy=Policy(retries=1, retry_delay=0.01, retry_on=(Flaky,), concurrency=2),
    )
    def client(text: str) -> str:
        tries["c" + text] = tries.get("c" + text, 0) + 1
        if tries["c" + text] < 2:
            raise Flaky("reset")
        return text + "!"

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "fetch", "client")),
        registry_of(split, fetch, client),
    )
    events = collect(graph.with_inputs(text="a,b"))
    retries = [e for e in events if e.type == "node_retry"]
    assert events[-1].status == "complete" and events[-1].run.results["client"]["result"] == [
        "a!",
        "b!",
    ], events[-1].run.cause
    assert {e.loc for e in retries} == {
        ("client",),
        ("fetch",),
    }  # the event names the node, and its cause names the call
    assert sorted((e.cause.loc[0], e.cause.loc[1:], e.attempt) for e in retries) == [
        ("client", (0,), 1),
        ("client", (1,), 1),
        ("fetch", (0,), 1),
        ("fetch", (0,), 2),
        ("fetch", (1,), 1),
        ("fetch", (1,), 2),
    ]
    assert retries[0].cause.type == "external_failed" and retries[0].delay == 0.01


def test_retries_run_out_and_plain_errors_never_retry():
    calls = []

    @node(
        "down", title="Down", description="For a test.", policy=Policy(retries=1, retry_delay=0.01)
    )
    def down(text: str) -> str:
        calls.append(text)
        raise ExternalError("down")

    @node("bug", title="Bug", description="For a test.", policy=Policy(retries=3, retry_delay=0.01))
    def bug(text: str) -> str:
        calls.append("bug")
        raise KeyError(text)

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"down": {"node": "down", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "down", "output": "result"}},
            }
        ),
        registry_of(down),
    ).run(text="x")
    assert run.status == "failed" and run.cause.type == "external_failed" and calls == ["x", "x"], (
        run.cause,
        calls,
    )
    calls.clear()
    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"bug": {"node": "bug", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "bug", "output": "result"}},
            }
        ),
        registry_of(bug),
    ).run(text="x")
    assert run.status == "failed" and run.cause.type == "execution_failed" and calls == ["bug"]


def test_attempt_timeout_is_final():
    calls = []

    @node(
        "hang",
        title="Hang",
        description="For a test.",
        policy=Policy(attempt_timeout=0.05, retries=2, retry_delay=0.01),
    )
    def hang(text: str) -> str:
        calls.append(text)
        t.sleep(0.3)
        return text

    start = t.monotonic()
    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"hang": {"node": "hang", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "hang", "output": "result"}},
            }
        ),
        registry_of(hang),
    ).run(text="x")
    assert (
        run.status == "failed"
        and run.cause.type == "timeout"
        and calls == ["x"]
        and t.monotonic() - start < 0.25
    ), (run.cause, calls)
    # the failure says why the retries=2 the policy sets weren't made
    assert "a timed-out call isn't retried" in run.cause.msg


def test_reuse_when_inputs_match_false_calls_again():
    calls = []

    @node(
        "rate",
        title="Rate",
        description="For a test.",
        policy=Policy(reuse_when_inputs_match=False),
    )
    def rate(text: str) -> str:
        calls.append(text)
        return text

    graph = Graph.compile(
        GraphRecord.model_validate(chain("split", "rate", "upper")), registry_of(split, rate, upper)
    )
    first = graph.run(text="a,b")
    calls.clear()
    first.again(text="a,b")
    assert calls == ["a", "b"], calls


def test_a_saved_run_is_held_to_the_reuse_policy_its_node_has_when_it_is_read():
    """``Run.from_json`` drops what a saved run could reuse for a node whose ``Policy`` now says not to reuse.

    The run is made from a previous one while ``rate`` reuses, and saved before it resumes. Read back where ``rate``
    says ``reuse_when_inputs_match=False``, it calls ``rate`` for every item, as a run made there would.
    """
    calls = []

    def rate(text: str) -> str:
        calls.append(text)
        return text

    reuses = node("rate", title="Rate", description="For a test.")(rate)
    never = node(
        "rate",
        title="Rate",
        description="For a test.",
        policy=Policy(reuse_when_inputs_match=False),
    )(rate)
    record = GraphRecord.model_validate(chain("split", "rate", "upper"))
    reusing = Graph.compile(record, registry_of(split, reuses, upper))
    saved = reusing.with_inputs(text="a,b", previous=reusing.run(text="a,b")).to_json()

    calls.clear()
    Run.from_json(saved, reusing).resume()
    assert calls == []
    Run.from_json(saved, Graph.compile(record, registry_of(split, never, upper))).resume()
    assert calls == ["a", "b"]


# -- 3. hooks ------------------------------------------------------------------------------------


def _formula_inputs(formula: str) -> dict[str, Any]:
    return {name: float for name in re.findall(r"{(\w+)}", formula)}


@node("formula", title="Formula", description="For a test.", compute_added_inputs=_formula_inputs)
def formula(formula: str = "", **values: float) -> float:
    return eval(formula.format(**values))  # just enough to test the hook; not a real calculator


def _branch_outputs(cases: list[str], value: type) -> dict[str, Any]:
    from typing import Annotated

    from conductor import SKIPPED, Param

    return {
        name: (Annotated[value, Param(title=name.title())], SKIPPED)
        for name in [*cases, "otherwise"]
    }


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
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "n": {"node": "number", "inputs": {"text": {"graph_input": "text"}}},
                    "f": {
                        "node": "formula",
                        "inputs": {
                            "formula": {"constant": "{a} * 10 + {b}"},
                            "b": {"constant": 2},
                            "a": {"sources": [{"node": "n", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "f", "output": "result"}},
            }
        ),
        reg3,
    )
    assert not graph.problems, graph.problems
    assert list(graph.inputs) == ["text"]
    run = graph.run(text="abc")
    assert run.outputs["result"] == 32.0, run.outputs
    broken = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "f": {
                        "node": "formula",
                        "inputs": {"formula": {"constant": "{a}"}, "a": {"graph_input": "a"}},
                    },
                    "g": {
                        "node": "formula",
                        "inputs": {"zz": {"sources": [{"node": "f", "output": "result"}]}},
                    },
                }
            }
        ),
        reg3,
    )
    assert [p.type for p in broken.problems] == ["unknown_input"], broken.problems


def test_computed_outputs_take_the_type_that_arrives_on_their_input():
    reg3 = registry_of(split, number, route, half, upper)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
                    "n": {
                        "node": "number",
                        "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                    "r": {
                        "node": "route",
                        "inputs": {
                            "cases": {"constant": ["3"]},
                            "value": {"sources": [{"node": "n", "output": "result"}]},
                        },
                    },
                    "h": {
                        "node": "half",
                        "inputs": {
                            "x": {
                                "sources": [
                                    {"node": "r", "output": "3"},
                                    {"node": "r", "output": "otherwise"},
                                ]
                            }
                        },
                    },
                },
                "outputs": {"result": {"node": "h", "output": "result"}},
            }
        ),
        reg3,
    )
    assert not graph.problems, graph.problems
    assert graph.run(text="abc,de").results["h"]["result"] == [1.5, 1.0]
    wrong = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "n": {"node": "number", "inputs": {"text": {"graph_input": "text"}}},
                    "r": {
                        "node": "route",
                        "inputs": {"value": {"sources": [{"node": "n", "output": "result"}]}},
                    },
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "r", "output": "otherwise"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        reg3,
    )
    assert [p.type for p in wrong.problems] == ["type_mismatch"], wrong.problems


def test_a_hook_reading_an_input_that_has_no_constant_is_the_only_problem_of_its_node():
    def added(n: int) -> dict[str, Any]:
        return {f"x{i}": str for i in range(n)}

    @node("hooky", title="Hooky", description="For a test.", compute_added_inputs=added)
    def hooky(n: int, **values: str) -> str:
        return "called"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "h": {"node": "hooky"},
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "h", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        registry_of(hooky, upper),
    )

    assert [(p.type, p.loc) for p in graph.problems] == [("input_has_no_value", ("h", "n"))]


@pytest.mark.parametrize("name", ["a.b", "a/b", "a@b", "a~b", ""])
def test_a_hook_naming_a_field_with_a_character_of_a_source_is_a_problem_as_a_node_id_is(name: str):
    """A field name a hook returns, an output's or an added input's, follows the same rule as a node id.

    It can't be empty, and it can't use the characters that sources, locs and waiting ids are written with.
    """

    @node("dotted", title="Dotted", description="For a test.", compute_outputs=lambda: {name: str})
    def dotted() -> Mapping[str, Any]:
        return {name: "x"}

    graph = Graph.compile(
        GraphRecord.model_validate({"nodes": {"d": {"node": "dotted"}}, "outputs": {}}),
        registry_of(dotted),
    )

    assert [(p.type, p.loc, p.ctx["error"]) for p in graph.problems] == [
        ("compute_hook_failed", ("d", name), "not_a_name")
    ]

    @node("adds", title="Adds", description="For a test.", compute_added_inputs=lambda: {name: int})
    def adds(**kwargs: Any) -> int:
        return 1

    added = Graph.compile(
        GraphRecord.model_validate({"nodes": {"a": {"node": "adds"}}, "outputs": {}}),
        registry_of(adds),
    )
    assert [(p.type, p.loc, p.ctx["error"]) for p in added.problems] == [
        ("compute_hook_failed", ("a", name), "not_a_name")
    ]
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {name: {"node": "dotted"}}}), registry_of(dotted))
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_record", ("record", "nodes", name, "[key]"))
    ]


# -- 4. a host's own types ------------------------------------------------------------------------


@host_type("integer", title="Integer")
class Integer(int):
    pass


@host_type("number", title="Number", accepts=[Integer])
class Number(float):
    """A number that may be written with a decimal comma, as in ``"2,5"``. It also accepts an ``Integer``."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            lambda v: cls(v.replace(",", ".")) if isinstance(v, str) else cls(v),
            core_schema.float_schema(),
        )


@host_type("file", title="File")
class File:
    def __init__(self, name: str, content: bytes) -> None:
        self.name, self.content = name, content

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, File) else File(v["name"], base64.b64decode(v["content"])),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda f: {"name": f.name, "content": base64.b64encode(f.content).decode()},
                when_used="json",
            ),
        )


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
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "c": {"node": "count", "inputs": {"text": {"graph_input": "text"}}},
                    "d": {
                        "node": "double",
                        "inputs": {"x": {"sources": [{"node": "c", "output": "result"}]}},
                    },
                    "a": {
                        "node": "attach",
                        "inputs": {"x": {"sources": [{"node": "d", "output": "result"}]}},
                    },
                    "s": {
                        "node": "size",
                        "inputs": {"f": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "s", "output": "result"}},
            }
        ),
        reg4,
    )
    assert not graph.problems, graph.problems
    run = graph.run(text="abc")
    assert run.outputs == {"result": 3}
    again = Run.from_json(run.to_json(), graph)
    assert again.results["a"]["result"].content == b"6.0"
    assert Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"d": {"node": "double", "inputs": {"x": {"constant": "1,5"}}}},
                "outputs": {"result": {"node": "d", "output": "result"}},
            }
        ),
        reg4,
    ).run().outputs == {"result": 3.0}
    wrong = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "d": {"node": "double", "inputs": {"x": {"graph_input": "x"}}},
                    "c": {
                        "node": "count",
                        "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}},
                    },
                }
            }
        ),
        reg4,
    )
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
    return Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "a": {"node": source.id, "inputs": {"text": {"graph_input": "text"}}},
                    "b": {
                        "node": target.id,
                        "inputs": {"value": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        registry_of(source, target),
    )


@pytest.mark.parametrize(
    ("source", "target", "fits"),
    [
        (as_text, show_json, True),
        (as_yes, show_json, True),
        (as_low, show_json, True),
        (as_count, show_number, True),
        (as_yes, show_number, False),
        (as_text, show_number, False),
        (as_low, show_level, True),
    ],
)
def test_an_edge_into_a_type_alias_or_a_literal_takes_what_its_values_may_be(source, target, fits):
    graph = _edge(source, target)

    assert [p.type for p in graph.problems] == ([] if fits else ["type_mismatch"])
    if fits:
        assert graph.run(text="x").status == "complete"


@node("as-low-or-zed", title="As low or zed", description="For a test.")
def as_low_or_zed(text: str) -> Literal["low", "zed"]:
    return "low" if text == "low" else "zed"


@node("as-x", title="As x", description="For a test.")
def as_x(text: str) -> Literal["x"]:
    return "x"


@node("show-chosen-level", title="Show chosen level", description="For a test.")
def show_chosen_level(value: Annotated[str, Param(choices=["low", "high"])]) -> str:
    return value


@pytest.mark.parametrize("target", [show_level, show_chosen_level])
@pytest.mark.parametrize("source", [as_text, as_low_or_zed])
def test_an_edge_that_may_give_a_choice_compiles_and_each_value_is_checked_on_its_call(
    source, target
):
    """An input's choices are a ``Literal``'s values or ``Param(choices=)``, and an edge into either follows one rule.

    A ``str``, or a ``Literal`` with values beside the choices, may give one of them. So the edge compiles, and a
    value that isn't one fails the call it goes into.
    """
    graph = _edge(source, target)

    assert graph.problems == []
    assert graph.run(text="low").outputs == {"result": "low"}
    failed = graph.run(text="zed")
    assert (failed.status, failed.cause.type) == ("failed", "invalid_edge_value")


@pytest.mark.parametrize("target", [show_level, show_chosen_level])
def test_an_edge_that_never_gives_a_choice_is_a_type_mismatch_that_names_the_choices(target):
    """A ``Literal`` that shares no value with an input's choices could never give one of them."""
    (problem,) = _edge(as_x, target).problems

    assert problem.type == "type_mismatch"
    assert problem.msg.startswith("""'a.result' gives "x", but 'b.value' takes "low" or "high".""")
    assert [p.type for p in _edge(as_count, target).problems] == ["type_mismatch"]


@node("as-words", title="As words", description="For a test.")
def as_words(text: str) -> list[str]:
    return text.split(",")


@node("each-word", title="Each word", description="For a test.")
def each_word(text: str) -> Series[str]:
    return text.split(",")


@node("show-levels", title="Show levels", description="For a test.")
def show_levels(value: list[Literal["low", "high"]]) -> str:
    return "+".join(value)


@node("show-all-levels", title="Show all levels", description="For a test.")
def show_all_levels(value: Series[Literal["low", "high"]]) -> str:
    return "+".join(value)


@pytest.mark.parametrize(
    ("source", "target"), [(as_words, show_levels), (each_word, show_all_levels)]
)
def test_a_list_of_text_into_a_list_of_choices_compiles_and_each_item_is_checked_on_its_call(
    source, target
):
    """The rule for choices holds on the items of a list, and of a ``Series`` an input collects."""
    graph = _edge(source, target)

    assert graph.problems == []
    assert graph.run(text="low,high").outputs == {"result": "low+high"}
    assert graph.run(text="low,zed").cause.type == "invalid_edge_value"


def test_one_value_into_a_series_of_choices_names_the_values_in_its_type_mismatch():
    """``type_mismatch``'s ``one_value`` variant names a type with choices by its values, as the plain one does."""
    (problem,) = _edge(as_low, show_all_levels).problems

    assert (problem.type, problem.ctx["variant"]) == ("type_mismatch", "one_value")
    assert problem.msg.startswith(
        """'a.result' gives each call of 'b' one "low", but 'b.value' takes a Series of "low" or "high"."""
    )


def _wired(source_hint: Any, value: Any, target_hint: Any) -> Graph:
    """Compiles ``a``, which returns ``value`` typed ``source_hint``, wired into ``b``, which takes ``target_hint``."""

    def give(text: str) -> Any:
        return value

    def take(value: Any) -> str:
        return repr(value)

    give.__annotations__ = {"text": str, "return": source_hint}
    take.__annotations__ = {"value": target_hint, "return": str}
    return _edge(
        node("give", title="Give", description="For a test.")(give),
        node("take", title="Take", description="For a test.")(take),
    )


class _Text(TypedDict):
    who: str


class _Chosen(TypedDict):
    who: Annotated[str, Param(choices=["ida", "bo"])]


class _X(TypedDict):
    who: Literal["x"]


@pytest.mark.parametrize(
    ("source", "value", "target"),
    [
        (Literal[1], 1, Annotated[float, Param(choices=[1.0, 2.0])]),
        (Literal["x", None], None, Annotated[str | None, Param(choices=["ida", "bo"])]),
        (Literal["x"] | None, None, Annotated[str | None, Param(choices=["ida", "bo"])]),
        (Literal["x"] | Literal["ida"], "ida", Literal["ida", "bo"]),
        (int | float, 1.5, float),
    ],
)
def test_a_source_whose_values_the_input_may_take_compiles_and_runs(source, value, target):
    """Whether an edge's values may be taken is asked of the input's own check, the one each value gets on its call.

    So ``1`` is the float choice ``1.0``, ``None`` is taken beside the choices of a type that takes ``None``, and
    a union of ``Literal``s is read as the one ``Literal`` of all their values.
    """
    graph = _wired(source, value, target)

    assert graph.problems == []
    assert graph.run(text="x").status == "complete"


@pytest.mark.parametrize(
    ("source", "value", "target"),
    [
        (list[str], ["zed"], list[Annotated[str, Param(choices=["ida", "bo"])]]),
        (list[int], [12], list[Annotated[int, Param(le=3)]]),
        (_Text, {"who": "zed"}, _Chosen),
        (Table[_Text], [{"who": "zed"}], Table[_Chosen]),
    ],
)
def test_choices_and_limits_below_the_top_are_checked_on_each_call(source, value, target):
    """A ``Param`` on a list's item, a record's field or a table's column is checked as one on the input is."""
    run = _wired(source, value, target).run(text="x")

    assert (run.status, run.cause.type) == ("failed", "invalid_edge_value")


def test_a_literal_column_that_shares_no_choice_with_its_input_column_is_refused_without_contradicting_itself():
    """Two tables whose columns read alike, differing only in their choices, aren't named as one type twice."""
    (problem,) = _wired(Table[_X], [{"who": "x"}], Table[_Chosen]).problems

    assert problem.type == "type_mismatch"
    assert "with other choices or limits" in problem.msg
    assert "conductor.Table" not in problem.msg and "__main__" not in problem.msg


def test_a_value_an_inputs_check_converts_below_the_top_reaches_the_node_converted():
    """The input's own check runs on a list's items as on the input, so what it converts is what the node gets."""
    run = _wired(list[str], ["ABC"], list[Annotated[str, AfterValidator(str.lower)]]).run(text="x")

    assert run.outputs == {"result": "['abc']"}


@pytest.mark.parametrize(
    ("source", "value", "target"),
    [
        (Literal[1], 1, Literal[True]),
        (Literal[1.0], 1.0, Literal[1, 2]),
        (Literal[1.0], 1.0, Annotated[int, Param(choices=[1, 2])]),
        (float, 1.0, Literal[1]),
    ],
)
def test_a_literal_source_whose_values_classes_dont_fit_is_a_type_mismatch(source, value, target):
    """A ``Literal``'s values must be taken by the target's check and their classes must fit it: pydantic takes
    ``1`` as ``True`` and ``1.0`` as ``1``, but an ``int`` is no yes or no and a ``float`` is no ``int``."""
    (problem,) = _wired(source, value, target).problems

    assert problem.type == "type_mismatch"


type _Team = Literal["ida", "bo"]


@pytest.mark.parametrize(
    ("source", "value", "target"),
    [
        (Literal["x"], "x", JsonValue),
        (list[Literal["x"]], ["x"], JsonValue),
        (Literal["ida"], "ida", _Team),
        (Literal[1], 1, float),
    ],
)
def test_a_literal_source_lands_on_an_alias_or_a_wider_class(source, value, target):
    """The rule for a source that names its values holds through an alias, JsonValue's tagged union too."""
    graph = _wired(source, value, target)

    assert graph.problems == []
    assert graph.run(text="x").status == "complete"


def test_a_validator_on_a_unions_member_checks_only_that_member():
    """In ``Annotated[str, AfterValidator(str.lower)] | None`` the validator is the ``str``'s, so ``None`` passes it
    by."""
    graph = _wired(str | None, None, Annotated[str, AfterValidator(str.lower)] | None)

    assert graph.run(text="x").outputs == {"result": "None"}


@pytest.mark.parametrize(
    ("source", "value", "target", "named"),
    [
        (Literal["red"], "red", "_Tint", "takes tint."),
        (str, "red", "_Tint", "takes tint."),
        (Literal["zed"], "zed", Literal[tuple(f"v{n}" for n in range(300))], '"v4" or 295 more.'),
        (list[Literal["x"]], ["x"], list[Literal["ida", "bo"]], "with other choices or limits"),
    ],
)
def test_a_type_mismatch_names_an_enum_by_its_id_and_a_long_list_of_values_in_short(
    source, value, target, named
):
    """An enum's class is what an edge must give, so it keeps its id; a list of values is cut short after five; and
    two types that read alike aren't named as one."""
    (problem,) = _wired(source, value, _Tint if target == "_Tint" else target).problems

    assert problem.type == "type_mismatch"
    assert named in problem.msg
    assert (
        len(problem.msg) < 400 and "builtins" not in problem.msg and "Literal[" not in problem.msg
    )


def _raises_a_bug(value: Any) -> Any:
    raise KeyError("a lookup the host's check forgot to guard")


def test_a_hosts_check_that_raises_on_every_value_of_a_literal_source_is_a_type_mismatch():
    """Whatever a host's own check raises for a value is a refusal of it, at compile as for ``optional``."""
    (problem,) = _wired(Literal["x"], "x", Annotated[str, AfterValidator(_raises_a_bug)]).problems

    assert problem.type == "type_mismatch"


@host_type("plain-tint", title="Plain tint")
class _PlainTint(Enum):
    RED = "red"


@host_type("tint-model", title="Tint model")
class _TintModel(BaseModel):
    tint: Literal[_PlainTint.RED]


@node("give-tint-model", title="Give tint model", description="For a test.")
def give_tint_model(text: str) -> _TintModel:
    return _TintModel(tint=_PlainTint.RED)


@node("take-int", title="Take int", description="For a test.")
def take_int(value: int) -> str:
    return str(value)


def test_an_edge_from_a_host_models_field_that_doesnt_fit_is_a_type_mismatch():
    """A host model's own fields are its own to check, so ``@node`` doesn't read them, and naming such a field's type
    in a message never raises: compiling reports, as for any other edge."""
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "a": {"node": "give-tint-model", "inputs": {"text": {"graph_input": "text"}}},
                    "b": {
                        "node": "take-int",
                        "inputs": {
                            "value": {
                                "sources": [{"node": "a", "output": "result", "field": "tint"}]
                            }
                        },
                    },
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        registry_of(give_tint_model, take_int),
    )

    assert [problem.type for problem in graph.problems] == ["type_mismatch"]
    assert "plain-tint" in graph.problems[0].msg


@host_type("whole-tint", title="Whole tint")
class _WholeTint(IntEnum):
    ONE = 1
    TWO = 2


@pytest.mark.parametrize(("source", "value"), [(Literal[1], 1), (int, 1)])
def test_a_literal_of_an_enums_members_is_named_by_the_enums_id(source, value):
    """The class is what an edge into ``Literal[WholeTint.ONE]`` must give, so the message names it and gives no advice
    a value would meet."""
    (problem,) = _wired(source, value, Literal[_WholeTint.ONE]).problems

    assert problem.type == "type_mismatch"
    assert "takes whole-tint." in problem.msg
    assert "with other choices or limits" not in problem.msg


@host_type("tint", title="Tint")
class _Tint(Enum):
    RED = "red"
    BLUE = "blue"


def _must_be_a_tint(value: Any) -> Any:
    assert isinstance(value, _Tint)
    return value


@pytest.mark.parametrize(
    "target",
    [
        _Tint,
        Annotated[_Tint, Param(choices=[_Tint.RED])],
        Annotated[Any, AfterValidator(_must_be_a_tint)],
    ],
)
def test_an_enum_is_wired_run_and_read_back(target):
    """An enum is the type for a fixed set of its members: an edge of it compiles, runs and its run reads back."""
    graph = _wired(_Tint, _Tint.RED, target)
    run = graph.run(text="x")

    assert graph.problems == []
    assert run.outputs == {"result": "<_Tint.RED: 'red'>"}
    assert Run.from_json(run.to_json(), graph).results == run.results


def test_a_json_null_on_a_json_input_is_a_value_and_its_node_is_called():
    run = _edge(parse_json, show_json).run(text="null")

    assert run.outputs == {"result": "null"} and run.not_called == []


# -- 5. branches ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class YesNo:
    yes: str = SKIPPED
    no: str = SKIPPED


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("pair-lists", title="Pair lists", description="For a test.")
def pair_lists(a: Series[str], b: Series[str]) -> str:
    return f"{a}|{b}"


@node("pair-optional", title="Pair optional", description="For a test.")
def pair_optional(a: Series[str | None], b: Series[str | None]) -> str:
    return f"{a}|{b}"


def test_two_series_on_different_branches_refused():
    reg6 = registry_of(split, decide, upper, pair_lists, pair_optional)
    record = {
        "nodes": {
            "s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
            "d": {
                "node": "decide",
                "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}},
            },
            "u": {
                "node": "upper",
                "inputs": {"text": {"sources": [{"node": "d", "output": "yes"}]}},
            },
            "p": {
                "node": "pair-lists",
                "inputs": {
                    "a": {"sources": [{"node": "u", "output": "result"}]},
                    "b": {"sources": [{"node": "d", "output": "no"}]},
                },
            },
        },
        "outputs": {"result": {"node": "p", "output": "result"}},
    }
    graph = Graph.compile(GraphRecord.model_validate(record), reg6)
    assert [p.type for p in graph.problems] == ["lists_from_different_loops"], graph.problems
    record["nodes"]["p"]["node"] = "pair-optional"
    graph = Graph.compile(GraphRecord.model_validate(record), reg6)
    assert not graph.problems
    assert graph.run(text="yes,no").outputs == {"result": "['YES', None]|[None, 'no']"}


# -- 6. describe -----------------------------------------------------------------------------------


def test_a_registrys_description_has_each_nodes_hooks_branches_and_the_types_each_type_accepts():
    reg7 = registry_of(count, double, attach, size, formula, route, decide)
    palette = json.loads(reg7.describe().model_dump_json())
    nodes = {n["id"]: n for n in palette["nodes"]}
    assert nodes["formula"]["takes_added_inputs"] and nodes["route"]["computes_outputs"]
    assert [(o["name"], o["required"], "default" in o) for o in nodes["decide"]["outputs"]] == [
        ("yes", False, False),
        ("no", False, False),
    ]
    types = {t["id"]: t["accepted_as"] for t in palette["types"]}
    assert "number" in types["integer"] and "integer" not in types["number"], types


def test_a_compiled_graph_describes_a_nodes_added_inputs_with_one_that_no_edge_reaches():
    reg7 = registry_of(count, formula)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "c": {"node": "count", "inputs": {"text": {"graph_input": "text"}}},
                    "f": {
                        "node": "formula",
                        "inputs": {
                            "formula": {"constant": "{a}+{b}"},
                            "a": {"sources": [{"node": "c", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "f", "output": "result"}},
            }
        ),
        reg7,
    )
    described = json.loads(graph.describe().model_dump_json())["nodes"]
    assert [f["field"]["name"] for f in described["f"]["inputs"]] == ["formula", "a", "b"], (
        described["f"]
    )


# -- 7. what a node's signature declares ------------------------------------------------------------


def test_a_fault_in_a_signature_is_refused_at_node_with_its_problem_type():
    from conductor import NodeDeclarationError

    cases = {
        "default_invalid": lambda: node("a", title="A", description="For a test.")(
            _fn("x: int = None")
        ),
        "nested_series": lambda: node("b", title="B", description="For a test.")(
            _fn("x: Series[Series[int]]")
        ),
        "limit_not_supported": lambda: node("d", title="D", description="For a test.")(
            _fn('x: Annotated[int, Param(ge="a")]')
        ),
    }
    for code, declare in cases.items():
        try:
            declare()
        except NodeDeclarationError as error:
            assert error.problems[0].type == code, (code, error.problems[0].type)
        else:
            raise AssertionError(code)


def _fn(parameter: str, **names: Any) -> Any:
    """Returns a function ``f`` that takes ``parameter``, built from source text.

    Building it from text lets a test give an annotation exactly as written, even one a type checker refuses.
    """
    namespace: dict[str, Any] = {"Annotated": Annotated, "Param": Param, "Series": Series, **names}
    exec(f"def f({parameter}) -> str:\n    return repr(x)", namespace)
    return namespace["f"]


def test_an_input_hint_written_as_optional_annotated_any_or_bare_series_is_described_and_run_as_written():
    from conductor import Table

    @host_type("Whole", title="Whole")
    class Whole(int):
        pass

    @node("kinds", title="Kinds", description="For a test.")
    def kinds(
        whole: Whole = 3,
        maybe: Annotated[str, Param(title="Maybe")] | None = None,
        anything: Any = None,
        many: Series = (),
        some: Series[int] | None = None,
    ) -> str:
        return type(whole).__name__

    fields = {f.name: f for f in kinds.describe().inputs}
    assert fields["maybe"].title == "Maybe" and not fields["maybe"].is_series
    assert fields["many"].is_series and fields["some"].is_series
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"k": {"node": "kinds", "inputs": {}}},
                "outputs": {"result": {"node": "k", "output": "result"}},
            }
        ),
        registry_of(kinds),
    )
    assert graph.run().outputs == {"result": "Whole"}
    assert Table is not None


# -- 8. a subgraph's inner nodes are reused just as the outer graph's own nodes are ------------


def test_again_with_one_subgraph_input_changed_calls_only_the_inner_node_that_reads_it_and_asks_nothing_again():
    from conductor import GraphRecord

    calls = {"mark": 0, "confirm": 0}

    @node("mark", title="Mark", description="For a test.")
    def mark(a: str) -> str:
        calls["mark"] += 1
        return a + "!"

    @node("confirm", title="Confirm", description="For a test.", asks=True)
    def confirm(b: str) -> bool:
        calls["confirm"] += 1
        raise Ask(f"ok {b}?")

    box = subgraph("two", title="Two", description="For a test.")(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "x": {"node": "mark", "inputs": {"a": {"graph_input": "a"}}},
                    "y": {"node": "confirm", "inputs": {"b": {"graph_input": "b"}}},
                },
                "outputs": {
                    "x_result": {"node": "x", "output": "result"},
                    "y_result": {"node": "y", "output": "result"},
                },
            }
        )
    )
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "e": {
                        "node": "two",
                        "inputs": {"a": {"graph_input": "a"}, "b": {"graph_input": "b"}},
                    }
                },
                "outputs": {
                    "x_result": {"node": "e", "output": "x_result"},
                    "y_result": {"node": "e", "output": "y_result"},
                },
            }
        ),
        registry_of(mark, confirm, box),
    )
    first = graph.run(a="1", b="2")
    first = first.answer(first.waiting[0], True).resume()
    before = dict(calls)

    again = first.again(a="changed")

    assert again.status == "complete" and again.waiting == []
    assert (calls["mark"] - before["mark"], calls["confirm"] - before["confirm"]) == (1, 0)
