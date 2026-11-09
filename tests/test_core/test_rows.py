"""The engine runs rows."""

import asyncio
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from conductor import InvalidCallError, NodeRegistry, node
from conductor._sentinel import SKIPPED, AskPerson
from conductor.dtype import DType
from conductor.errors import (
    NodeExecutionError,
)
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Input, Output, Param, Result
from conductor.node_decorator import Policy
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea
from helpers import stream_of

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "rows-test-txt"
    title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Navne")]


@dataclass(frozen=True)
class Length:
    long: Annotated[Txt, Result(title="If long")]
    short: Annotated[Txt, Result(title="If short")]


@node("docs", title="Docs", description="d", category="test")
def docs(text: Annotated[Txt, Param(title="Texts", widget=Textarea())] = Txt("")) -> Documents:
    parts = [Txt(p) for p in text.split(",")] if text else []
    return Documents(texts=parts, names=[Txt(f"doc{i}") for i in range(len(parts))])


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("pair", title="Pair", description="d", category="test")
def pair(a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
    return Txt(f"{a}:{b}")


@node("lines", title="Lines", description="d", category="test")
def lines_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return [Txt(p) for p in text.split("/")]


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


@node("long-only", title="Long only", description="d", category="test")
def long_only(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Length:
    """A decision: keeps a text on `If long` when it has more than two characters."""
    return Length(long=text, short=SKIPPED) if len(text) > 2 else Length(long=SKIPPED, short=text)


@node("fails-on", title="Fails on", description="d", category="test")
def fails_on(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    if text == "boom":
        raise NodeExecutionError("kunne ikke", code="boom")
    return text


slow_state = SimpleNamespace(lock=threading.Lock(), active=0, peak=0, seen=[], calls=0)


@node("slow", title="Slow", description="d", category="test", policy=Policy(concurrency=1))
def slow(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    with slow_state.lock:
        slow_state.active += 1
        slow_state.peak = max(slow_state.peak, slow_state.active)
    time.sleep(0.02)
    with slow_state.lock:
        slow_state.active -= 1
        slow_state.seen.append(text)
    return text


wide_state = SimpleNamespace(lock=threading.Lock(), active=0, peak=0, seen=[], calls=0)


@node("wide", title="Wide", description="d", category="test")
def wide(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    with wide_state.lock:
        wide_state.active += 1
        wide_state.peak = max(wide_state.peak, wide_state.active)
    time.sleep(0.02)
    with wide_state.lock:
        wide_state.active -= 1
    return text


def columns_outputs(spec=""):
    spec = spec
    return tuple(Output(name, Txt, title=name.title()) for name in spec.split(",") if name)

@node("columns", title="Columns", description="d", category="test", compute_outputs=columns_outputs)
def columns(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""), spec: Annotated[Txt, Param(title="Kolonner", widget=Textarea())] = Txt("")) -> Mapping[str, Any]:
    """The unfold shape: the author names the columns; each is an output
    on the interface and nothing else. run hands them back by name."""
    return {name: Txt(f"{name}:{text}") for name in spec.split(",") if name}


def _registry():
    registry = NodeRegistry()
    for node_cls in (docs, upper_node, pair, lines_node, join_node, long_only, fails_on, slow, wide, columns):
        registry = registry.extended_with([node_cls])
    return registry


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


def _docs(texts):
    return GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(texts)})


def _run(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())


def _events(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())

    async def gathered():
        return [e async for e in stream_of(compiled)]

    return asyncio.run(gathered())


# --- a computed interface, by name -----------------------------------------------------


def test_a_computed_roster_runs_by_name_and_edges_like_any_field():
    results = _run([
        _docs("a,b"),
        GraphNode(id="c", type="columns", version=1, bindings={"text": _edge(("docs", "texts")), "spec": Static("navn,email")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("c", "email"))}),
    ])

    assert list(results["c"]["navn"]) == ["navn:a", "navn:b"]
    assert results["c"]["email"].index == Index("docs.texts")
    assert list(results["up"]["result"]) == ["EMAIL:A", "EMAIL:B"]


def test_a_computed_roster_with_no_outputs_is_compiles_problem():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="c", type="columns", version=1, bindings={"spec": Static("")})]), _registry())

    assert [(p.type, p.loc[0], p.fatal) for p in compiled.problems] == [("no_outputs", "c", False)]
    assert compiled.is_runnable


# --- iteration end to end -----------------------------------------------------------


def test_an_iterating_chain_runs_per_row_and_a_reduction_collapses_it():
    results = _run([
        _docs("a,b,c"),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])

    up = results["up"]["result"]
    assert isinstance(up, Series)
    assert up.index == Index("docs.texts")
    assert up.rows == ((0,), (1,), (2,))
    assert list(up) == ["A", "B", "C"]
    assert results["j"]["result"] == "A+B+C"


def test_a_scalar_broadcasts_beside_a_series():
    results = _run([
        _docs("a,b"),
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static("p")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("prefix", "result")), "b": _edge(("docs", "texts"))}),
    ])

    assert list(results["p"]["result"]) == ["P:a", "P:b"]


def test_an_iterating_node_returning_a_series_unfolds_and_a_reduction_refolds():
    """Unfold and gather with no special node: lines per document, then
    one text per document again."""
    results = _run([
        _docs("a/b,c"),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])

    lines = results["lines"]["result"]
    assert lines.index == Index("lines.result", parent=Index("docs.texts"))
    assert lines.rows == ((0, 0), (0, 1), (1, 0))
    assert results["up"]["result"].rows == ((0, 0), (0, 1), (1, 0))
    per_doc = results["j"]["result"]
    assert per_doc.index == Index("docs.texts")
    assert list(per_doc) == ["A+B", "C"]


def test_a_parent_series_broadcasts_down_to_a_child_row():
    results = _run([
        _docs("a/b,c"),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("lines", "result")), "b": _edge(("docs", "texts"))}),
    ])

    assert list(results["p"]["result"]) == ["a:a/b", "b:a/b", "c:c"]


def test_a_gather_of_scalars_arrives_as_a_series():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert results["j"]["result"] == "A+B"


def test_a_list_set_on_a_node_running_per_row_runs_once_per_value_in_each_row():
    """Questions set on «Spørg», which runs per document: each document asks each question."""
    results = _run([
        _docs("a,b,c"),
        GraphNode(id="ask", type="pair", version=1, bindings={"a": _edge(("docs", "texts")), "b": Static(["q1", "q2", "q3"])}),
    ])

    asked = results["ask"]["result"]
    assert asked.rows == tuple((doc, question) for doc in range(3) for question in range(3))
    assert list(asked) == [f"{doc}:{question}" for doc in "abc" for question in ("q1", "q2", "q3")]


def test_no_documents_ask_no_questions():
    results = _run([
        _docs(""),
        GraphNode(id="ask", type="pair", version=1, bindings={"a": _edge(("docs", "texts")), "b": Static(["q1", "q2"])}),
    ])

    assert list(results["ask"]["result"]) == []


def test_a_document_skipped_above_asks_none_of_its_questions():
    results = _run([
        _docs("abc,d,efg"),
        GraphNode(id="g", type="long-only", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="ask", type="pair", version=1, bindings={"a": _edge(("g", "long")), "b": Static(["q1", "q2"])}),
    ])

    assert results["ask"]["result"].rows == ((0, 0), (0, 1), (2, 0), (2, 1))


# --- sparse series and skip ---------------------------------------------


def test_an_iterating_gate_masks_rows_and_downstream_runs_on_fewer():
    results = _run([
        _docs("abc,d,efg"),
        GraphNode(id="g", type="long-only", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="long", type="upper", version=1, bindings={"text": _edge(("g", "long"))}),
        GraphNode(id="short", type="upper", version=1, bindings={"text": _edge(("g", "short"))}),
    ])

    assert results["long"]["result"].rows == ((0,), (2,))
    assert list(results["long"]["result"]) == ["ABC", "EFG"]
    assert results["short"]["result"].rows == ((1,),)


def test_a_reduction_over_a_sparse_series_sees_only_the_rows_present():
    results = _run([
        _docs("abc,d,efg"),
        GraphNode(id="g", type="long-only", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("g", "long"))}),
    ])

    assert results["j"]["result"] == "abc+efg"


def test_a_skipped_scalar_skips_what_hangs_off_it():
    results = _run([
        GraphNode(id="g", type="long-only", version=1, bindings={"text": Static("hey")}),
        GraphNode(id="long", type="upper", version=1, bindings={"text": _edge(("g", "long"))}),
        GraphNode(id="short", type="upper", version=1, bindings={"text": _edge(("g", "short"))}),
    ])

    assert results["long"]["result"] == "HEY"
    assert "short" not in results


def test_two_branches_merge_back_by_wiring_and_the_chain_stays_on_the_index():
    """Merging by edges, end to end: a decision running per row, each branch through its own node, both connected into one input."""
    results = _run([
        _docs("hey,x,world"),
        GraphNode(id="g", type="long-only", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="long", type="upper", version=1, bindings={"text": _edge(("g", "long"))}),
        GraphNode(id="short", type="pair", version=1, bindings={"a": _edge(("g", "short")), "b": Static("!")}),
        GraphNode(id="all", type="upper", version=1, bindings={"text": _edge(("long", "result"), ("short", "result"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("all", "result"))}),
    ])

    assert list(results["all"]["result"]) == ["HEY", "X:!", "WORLD"]
    assert results["all"]["result"].rows == ((0,), (1,), (2,))
    assert results["j"]["result"] == "HEY+X:!+WORLD"


def test_a_skipped_node_skips_the_series_it_would_have_made():
    results = _run([
        GraphNode(id="g", type="long-only", version=1, bindings={"text": Static("hey")}),
        GraphNode(id="docs", type="docs", version=1, bindings={"text": _edge(("g", "short"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])

    assert "docs" not in results
    assert "up" not in results
    assert "j" not in results


# --- failure -------------------------------------------------------------------


def test_a_failed_row_fails_the_node_and_the_cause_names_the_row():
    events = _events([
        _docs("ok,boom,ok"),
        GraphNode(id="f", type="fails-on", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("f", "result"))}),
    ])

    (error,) = [e for e in events if e.type == "node_failed"]
    assert error.node_id == "f"
    assert error.cause.type == "boom"
    assert error.cause.loc[1:] == (1,)
    assert events[-1].state.status == "failed"
    assert not any(e.type == "node_start" and e.node_id == "j" for e in events)


def test_run_sync_returns_the_error_ending_with_its_cause():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[_docs("boom"), GraphNode(id="f", type="fails-on", version=1, bindings={"text": _edge(("docs", "texts"))})]),
        _registry(),
    )

    ending = compiled.run_sync()
    assert ending.status == "failed"
    assert ending.failure.loc[0] == "f"
    assert ending.failure.loc[1:] == (0,)


def test_a_defect_in_the_engine_fails_the_leg_instead_of_hanging_it(monkeypatch):
    """An exception the engine did not expect, raised while running a unit,
    ends the leg as a failure of that unit. A task that died silently would
    leave the loop waiting for a unit that never reports."""
    from conductor.execution.live_run_values import LiveRunValues

    def broken(self, unit):
        raise RuntimeError("the ledger lost a value")

    monkeypatch.setattr(LiveRunValues, "inputs_for", broken)
    compiled = CompiledGraph.from_graph(Graph(nodes=[_docs("a,b")]), _registry())

    async def gathered():
        return [e async for e in stream_of(compiled, timeout=5)]

    events = asyncio.run(gathered())

    assert events[-1].state.status == "failed"
    assert events[-1].state.failure.type == "engine_error"
    assert events[-1].state.failure.ctx == {"exception": "RuntimeError"}
    assert "the ledger lost a value" not in str(events[-1].state.failure)


def test_a_graph_compile_rejected_is_refused_with_its_problems():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="gone", version=1)]), _registry())

    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync()
    assert [fault.type for fault in raised.value.errors()] == ["unknown_node_type"]


# --- scheduling -----------------------------------------------------------------


def test_rows_run_concurrently_under_the_engines_cap():
    wide_state.peak = 0
    _run([_docs("a,b,c,d"), GraphNode(id="w", type="wide", version=1, bindings={"text": _edge(("docs", "texts"))})])

    assert wide_state.peak > 1


def test_a_policy_of_one_runs_rows_one_at_a_time():
    slow_state.peak, slow_state.seen = 0, []
    _run([_docs("a,b,c"), GraphNode(id="s", type="slow", version=1, bindings={"text": _edge(("docs", "texts"))})])

    assert slow_state.peak == 1
    assert sorted(slow_state.seen) == ["a", "b", "c"]


def test_progress_is_per_row_and_the_total_is_known_once_the_index_is_sealed():
    events = _events([_docs("a,b,c"), GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    progress = [(e.done, e.total) for e in events if e.type == "node_progress" and e.node_id == "up"]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [e.type for e in events if getattr(e, "node_id", None) == "up"][0] == "node_start"
    assert [e.type for e in events if getattr(e, "node_id", None) == "up"][-1] == "node_complete"


def test_a_unit_standing_in_for_rows_never_made_reports_no_progress():
    """``a/b`` is long and splits into two lines; ``x`` is short, so its lines are
    never made and ``up`` stands in at ``(1,)``, which counts as no row."""
    events = _events([
        _docs("a/b,x"),
        GraphNode(id="long", type="long-only", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("long", "long"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
    ])

    progress = [(e.done, e.total) for e in events if e.type == "node_progress" and e.node_id == "up"]
    assert progress == [(1, 2), (2, 2)]


def test_computed_inputs_reach_the_node_as_keywords():
    """End to end: the template's placeholders are inputs the author
    edges, and the node receives them by name."""
    import re

    from conductor.widgets import TextWidget

    def template_node_inputs(template=""):
        names = re.findall(r"\{(\w+)\}", str(template))
        return tuple(Input(n, Txt, widget=TextWidget()) for n in dict.fromkeys(names))

    @node("template", title="Template", description="d", category="test", compute_inputs=template_node_inputs)
    def template_node(template: Annotated[Txt, Param(title="Template", connectable=False, widget=Textarea())] = Txt(""), **values: Txt) -> Out:
        return Txt(re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), template))

    registry = _registry()
    registry = registry.extended_with([template_node])
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs("Ida,Bo"),
            GraphNode(id="case", type="upper", version=1, bindings={"text": Static("24-1")}),
            GraphNode(id="t", type="template", version=1, bindings={
                "template": Static("Dear {name} ({case})"),
                "name": _edge(("docs", "texts")),
                "case": _edge(("case", "result")),
            }),
        ]),
        registry,
    )
    assert compiled.is_runnable, compiled.problems

    assert list(compiled.results(compiled.run_sync())["t"]["result"]) == ["Dear Ida (24-1)", "Dear Bo (24-1)"]


def test_node_complete_carries_the_series():
    events = _events([_docs("a,b"), GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    (done,) = [e for e in events if e.type == "node_complete" and e.node_id == "up"]
    assert list(done.result["result"]) == ["A", "B"]
    assert events[-1].state.status == "complete"


# --- every ending is one shape ----------------------------------------------------


def test_a_cancelled_leg_carries_its_state():
    """`graph_cancelled` and `graph_timeout` end a leg the way `graph_complete`
    and `graph_pending` do — the run's state beside their reason — so a host can start a new run from any ending."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
            GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        ]),
        _registry(),
    )
    stop = asyncio.Event()
    stop.set()

    async def leg(**kwargs):
        return [e async for e in stream_of(compiled, **kwargs)]

    (ending,) = [e.state for e in asyncio.run(leg(cancel=stop)) if e.type == "run_end"]
    assert ending.status == "cancelled" and compiled.results(ending) == {}

    seeded = asyncio.run(leg(state=ending))
    assert seeded[-1].state.status == "complete"
    assert compiled.results(seeded[-1].state)["b"]["result"] == "B"


# --- a leg ends pending ---------------------------------------------------


@node("asks", title="Ask", description="d", category="test")
def ask_node(proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    """An asking node: packages its proposal into the question and
    returns AskPerson. Its output is what the person supplies."""
    return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})


def test_a_run_that_may_ask_says_so_and_still_declares_its_outputs():
    """`-> Out | AskPerson` is the truthful annotation: the pause is read past, the
    record declares the one output the person's answer lands on."""
    interface = ask_node.versions[1].interface
    assert [(out.name, out.dtype) for out in interface.outputs] == [("result", Txt)]


def _registry_with_asks(*extra):
    registry = _registry()
    for node_cls in (ask_node, *extra):
        registry = registry.extended_with([node_cls])
    return registry


def _leg(compiled, **kwargs):
    async def collect_all():
        return [e async for e in stream_of(compiled, **kwargs)]

    return asyncio.run(collect_all())


def test_a_node_that_asks_ends_the_leg_pending_with_its_question_named_by_address():
    """The leg runs to quiescence — the independent node completes in
    the same leg — and ends pending with the question, not paused at it."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="ask", type="asks", version=1, bindings={"proposal": Static("proposal")}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
            GraphNode(id="other", type="upper", version=1, bindings={"text": Static("x")}),
        ]),
        _registry_with_asks(),
    )
    events = _leg(compiled)

    assert events[-1].state.status == "waiting"
    (waiting,) = events[-1].state.waiting
    assert (waiting.node_id, waiting.row) == ("ask", ())
    (question,) = waiting.questions
    assert question.name == "result" and question.default == "proposal"
    assert compiled.results(events[-1].state)["other"]["result"] == "X"
    assert "up" not in compiled.results(events[-1].state)
    assert not any(e.type == "node_start" and e.node_id == "up" for e in events)


def test_answering_is_the_next_leg_from_the_state_and_the_cache():
    """The answer becomes the asking node's output; the state carries
    what the first leg produced, so nothing done is done twice."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="ask", type="asks", version=1, bindings={"proposal": Static("proposal")}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
            GraphNode(id="other", type="upper", version=1, bindings={"text": Static("x")}),
        ]),
        _registry_with_asks(),
    )
    first = _leg(compiled)[-1].state

    second = _leg(compiled, state=first, answers={"ask": {"result": Txt("yes")}})

    assert second[-1].state.status == "complete"
    assert compiled.results(second[-1].state)["up"]["result"] == "YES"
    assert compiled.results(second[-1].state)["ask"]["result"] == "yes"
    assert not any(e.type == "node_start" and e.node_id in ("other", "ask") for e in second)


def test_an_iterating_asking_node_pends_once_per_row_and_is_answered_as_a_series():
    """Ten units blocked are ten questions in one set; each unit
    answers for itself, by its own id."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs("a,b"),
            GraphNode(id="ask", type="asks", version=1, bindings={"proposal": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
        ]),
        _registry_with_asks(),
    )
    first = _leg(compiled)[-1].state

    assert first.status == "waiting"
    assert [(w.node_id, w.row) for w in first.waiting] == [("ask", (0,)), ("ask", (1,))]
    assert [w.questions[0].default for w in first.waiting] == ["a", "b"]

    second = _leg(compiled, state=first, answers={"ask@0": {"result": Txt("x")}, "ask@1": {"result": Txt("y")}})

    assert list(compiled.results(second[-1].state)["up"]["result"]) == ["X", "Y"]


@node("asks-if-long", title="Ask if long", description="d", category="test")
def asks_if_long(proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    """AskPerson about a text longer than two characters, and passes a short one through."""
    if len(proposal) <= 2:
        return proposal
    return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})


def _asks_if_long(texts):
    return CompiledGraph.from_graph(
        Graph(nodes=[
            _docs(texts),
            GraphNode(id="ask", type="asks-if-long", version=1, bindings={"proposal": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
        ]),
        _registry_with_asks(asks_if_long),
    )


def test_an_answer_fills_the_rows_it_names_and_the_rows_already_done_stay():
    """Rows 0 and 2 ran in the first leg; only row 1 asked. The answer
    names row 1, and rows 0 and 2 keep what they produced."""
    compiled = _asks_if_long("a,long,b")
    first = _leg(compiled)[-1].state
    assert [(w.node_id, w.row) for w in first.waiting] == [("ask", (1,))]

    second = _leg(compiled, state=first, answers={"ask@1": {"result": Txt("yes")}})

    assert second[-1].state.status == "complete"
    assert list(compiled.results(second[-1].state)["ask"]["result"]) == ["a", "yes", "b"]
    assert list(compiled.results(second[-1].state)["up"]["result"]) == ["A", "YES", "B"]
    (start,) = [e for e in second if e.type == "run_start"]
    assert "docs" in start.reused_nodes


def test_a_row_left_unanswered_goes_on_waiting_with_its_question_frozen():
    compiled = _asks_if_long("long,longer")
    first = _leg(compiled)[-1].state

    second = _leg(compiled, state=first, answers={"ask@0": {"result": Txt("yes")}})

    assert second[-1].state.status == "waiting"
    assert [(w.node_id, w.row) for w in second[-1].state.waiting] == [("ask", (1,))]
    assert not any(e.type == "node_start" and e.node_id == "ask" for e in second)
    assert not any(e.type == "node_complete" and e.node_id == "ask" for e in second)


def test_two_asking_nodes_in_parallel_are_one_pending_set():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="asks", version=1, bindings={"proposal": Static("1")}),
            GraphNode(id="b", type="asks", version=1, bindings={"proposal": Static("2")}),
        ]),
        _registry_with_asks(),
    )
    (ending,) = [e.state for e in _leg(compiled) if e.type == "run_end"]

    assert sorted(w.node_id for w in ending.waiting) == ["a", "b"]


def test_run_sync_returns_the_pending_ending():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="ask", type="asks", version=1, bindings={"proposal": Static("p")})]),
        _registry_with_asks(),
    )

    pending = compiled.run_sync()
    assert pending.status == "waiting"
    assert [w.node_id for w in pending.waiting] == ["ask"]

    answered = _leg(compiled, state=pending, answers={"ask": {"result": Txt("yes")}})
    assert answered[-1].state.status == "complete"
    assert compiled.results(answered[-1].state)["ask"]["result"] == "yes"


# --- an embedded graph runs as nodes of the one run --------------------------------


#: An embedded graph whose inner node pairs what enters with a list the author typed.
TypedInside = embedded_graph_node("typed-inside", (
    GraphNode(id="h", type="upper", version=1),
    GraphNode(id="t", type="pair", version=1, bindings={"a": _edge(("h", "result")), "b": Static(["p", "q"])}),
    GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("t", "result"))}),
), _registry(), "Indlejret")


@pytest.mark.parametrize(("texts", "joined"), [("a,b", ["A:p+A:q", "B:p+B:q"]), ("", [])])
def test_a_typed_in_list_inside_an_iterating_embedded_graph_is_a_child_row_under_each_outer_row(texts, joined):
    """Once per outer row, the inner node runs once per typed value, and the
    inner reduction gathers those under the outer row. With no outer rows
    there are no typed rows either, and the run still completes."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs(texts),
            GraphNode(id="emb", type="typed-inside", version=1, bindings={"h.text": _edge(("docs", "texts"))}),
        ]),
        _registry_with_asks(TypedInside),
    )
    assert compiled.node("emb/t").runs_per_row_of.parent == Index("docs.texts")

    results = compiled.results(compiled.run_sync())

    assert list(results["emb/j"]["result"]) == joined
    assert results["emb/t"]["result"].rows == tuple((i, n) for i in range(len(joined)) for n in range(2))


def test_a_restore_leaves_a_list_set_on_a_node_under_a_skipped_outer_row_complete():
    """A skip above the embedded graph leaves its list set on a node with no rows
    under the outer row, and a restore works that out again, so the inner
    node that stood in is still complete on the next leg."""
    from conductor.execution.live_run_values import LiveRunValues, Skip

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[_docs("a,b"), GraphNode(id="emb", type="typed-inside", version=1, bindings={"h.text": _edge(("docs", "texts"))})]),
        _registry_with_asks(TypedInside),
    )
    live = LiveRunValues(compiled)
    live.record(("docs", ()), Skip(at=()))
    for node_id in compiled.execution_order[1:]:
        live.record((node_id, ()), live.inputs_for((node_id, ())))

    restored = LiveRunValues.restore(compiled, live.state("complete"))

    assert restored.units("emb/t") == live.units("emb/t") == [("emb/t", ())]
    assert restored.completed_nodes() == live.completed_nodes() == set(compiled.execution_order)


def test_an_inner_reduction_over_the_entering_series_runs_once_per_outer_row():
    """Gathering end to end: three texts in, three joined texts out —
    the ledger groups by the iteration index's depth, so the fold of one is one
    rule, not a second kind of index."""

    embedded = embedded_graph_node("inner-graph", (
        GraphNode(id="holder", type="upper", version=1, bindings={"text": Static("inner")}),
        GraphNode(id="gather", type="join", version=1, bindings={"texts": _edge(("holder", "result"))}),
    ), _registry(), "Indlejret")

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs("a,b,c"),
            GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.text": _edge(("docs", "texts"))}),
            GraphNode(id="after", type="join", version=1, bindings={"texts": _edge(("emb", "gather.result"))}),
        ]),
        _registry_with_asks(embedded),
    )
    assert compiled.is_runnable, compiled.problems
    results = compiled.results(compiled.run_sync())

    assert list(results["emb/holder"]["result"]) == ["A", "B", "C"]
    assert list(results["emb/gather"]["result"]) == ["A", "B", "C"]
    assert results["emb/gather"]["result"].index == Index("docs.texts")
    assert results["after"]["result"] == "A+B+C"
