"""The engine runs rows."""

import asyncio
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pytest
from conductor import NodeRegistry, run_sync
from conductor._sentinel import SKIPPED, Asks
from conductor.dtype import DType
from conductor.errors import (
    CompilationError,
    ErrorCause,
    NodeExecutionError,
)
from conductor.execution.engine import execute
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.interface import Interface
from conductor.metadata import Input, Output, Param, Result
from conductor.node import GraphVersion, NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea


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


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Texts", widget=Textarea())] = Txt("")) -> Documents:
        parts = [Txt(p) for p in text.split(",")] if text else []
        return Documents(texts=parts, names=[Txt(f"doc{i}") for i in range(len(parts))])


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Pair(NodeDefinition):
    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"

    def run(self, a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
        return Txt(f"{a}:{b}")


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return [Txt(p) for p in text.split("/")]


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class LongOnly(NodeDefinition):
    """A decision: keeps a text on `If long` when it has more than two characters."""

    id = "long-only"
    title = "Long only"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Length:
        return Length(long=text, short=SKIPPED) if len(text) > 2 else Length(long=SKIPPED, short=text)


class FailsOn(NodeDefinition):
    id = "fails-on"
    title = "Fails on"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        if text == "boom":
            raise NodeExecutionError("kunne ikke", cause=ErrorCause(code="boom", message="Boom."))
        return text


class Slow(NodeDefinition):
    id = "slow"
    title = "Slow"
    description = "d"
    category = "test"
    seen: list = []
    lock = threading.Lock()
    active = 0
    peak = 0

    @version(1, policy=Policy(concurrency=1))
    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        with Slow.lock:
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
        time.sleep(0.02)
        with Slow.lock:
            Slow.active -= 1
            Slow.seen.append(text)
        return text


class Wide(NodeDefinition):
    id = "wide"
    title = "Wide"
    description = "d"
    category = "test"
    lock = threading.Lock()
    active = 0
    peak = 0

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        with Wide.lock:
            Wide.active += 1
            Wide.peak = max(Wide.peak, Wide.active)
        time.sleep(0.02)
        with Wide.lock:
            Wide.active -= 1
        return text


class Columns(NodeDefinition):
    """The unfold shape: the author names the columns; each is an output
    on the interface and nothing else. run hands them back by name."""

    id = "columns"
    title = "Columns"
    description = "d"
    category = "test"

    def compute_outputs(self, declared, values, arriving):
        spec = values.get("spec", "")
        return tuple(Output(name=name, dtype=Txt, title=name.title()) for name in spec.split(",") if name)

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""), spec: Annotated[Txt, Param(title="Kolonner", widget=Textarea())] = Txt("")) -> Mapping[str, Any]:
        return {name: Txt(f"{name}:{text}") for name in spec.split(",") if name}


def _registry():
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Pair, Lines, Join, LongOnly, FailsOn, Slow, Wide, Columns):
        registry.register(node_cls)
    return registry


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


def _docs(texts):
    return GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(texts)})


def _run(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return run_sync(compiled).results


def _events(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())

    async def gathered():
        return [e async for e in execute(compiled)]

    return asyncio.run(gathered())


# --- a computed interface, by name -----------------------------------------------------


def test_a_computed_roster_runs_by_name_and_edges_like_any_field():
    results = _run([
        _docs("a,b"),
        GraphNode(id="c", type="columns", version=1, bindings={"text": _edge(("docs", "texts")), "spec": Static("navn,email")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("c", "email"))}),
    ])

    assert list(results["c"]["navn"]) == ["navn:a", "navn:b"]
    assert results["c"]["email"].index == Index("docs")
    assert list(results["up"]["result"]) == ["EMAIL:A", "EMAIL:B"]


def test_a_computed_roster_with_no_outputs_is_compiles_problem():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="c", type="columns", version=1, bindings={"spec": Static("")})]), _registry())

    assert [(p.code, p.node_id, p.fatal) for p in compiled.problems] == [("no_outputs", "c", False)]
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
    assert up.index == Index("docs")
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
    assert lines.index == Index("lines", parent=Index("docs"))
    assert lines.rows == ((0, 0), (0, 1), (1, 0))
    assert results["up"]["result"].rows == ((0, 0), (0, 1), (1, 0))
    per_doc = results["j"]["result"]
    assert per_doc.index == Index("docs")
    assert list(per_doc) == ["A+B", "C"]


def test_a_parent_series_broadcasts_down_to_a_child_row():
    results = _run([
        _docs("a/b,c"),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("lines", "result")), "b": _edge(("docs", "names"))}),
    ])

    assert list(results["p"]["result"]) == ["a:doc0", "b:doc0", "c:doc1"]


def test_a_gather_of_scalars_arrives_as_a_series():
    results = _run([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert results["j"]["result"] == "A+B"


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


def test_a_skipped_node_skips_the_series_it_would_have_born():
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

    (error,) = [e for e in events if e.type == "node_error"]
    assert error.node_id == "f"
    assert error.cause.code == "boom"
    assert error.cause.row == (1,)
    assert events[-1].type == "graph_error"
    assert not any(e.type == "node_start" and e.node_id == "j" for e in events)


def test_run_sync_returns_the_error_ending_with_its_cause():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[_docs("boom"), GraphNode(id="f", type="fails-on", version=1, bindings={"text": _edge(("docs", "texts"))})]),
        _registry(),
    )

    ending = run_sync(compiled)
    assert ending.type == "graph_error"
    assert ending.node_id == "f"
    assert ending.cause.row == (0,)


def test_a_defect_in_the_engine_fails_the_leg_instead_of_hanging_it(monkeypatch):
    """An exception the engine did not expect, raised while running a unit,
    ends the leg as a failure of that unit. A task that died silently would
    leave the loop waiting for a unit that never reports."""
    from conductor.execution.ledger import Ledger

    def broken(self, unit):
        raise RuntimeError("the ledger lost a cell")

    monkeypatch.setattr(Ledger, "inputs_for", broken)
    compiled = CompiledGraph.from_graph(Graph(nodes=[_docs("a,b")]), _registry())

    async def gathered():
        return [e async for e in execute(compiled, timeout=5)]

    events = asyncio.run(gathered())

    assert events[-1].type == "graph_error"
    assert events[-1].cause.code == "engine_error"
    assert events[-1].cause.details == {"exception": "RuntimeError"}
    assert "the ledger lost a cell" not in events[-1].error


def test_a_graph_compile_rejected_is_refused_with_its_problems():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="gone", version=1)]), _registry())

    with pytest.raises(CompilationError) as raised:
        run_sync(compiled)
    assert [p.code for p in raised.value.problems] == ["unknown_node_type"]


# --- scheduling -----------------------------------------------------------------


def test_rows_run_concurrently_under_the_engines_cap():
    Wide.peak = 0
    _run([_docs("a,b,c,d"), GraphNode(id="w", type="wide", version=1, bindings={"text": _edge(("docs", "texts"))})])

    assert Wide.peak > 1


def test_a_policy_of_one_runs_rows_one_at_a_time():
    Slow.peak, Slow.seen = 0, []
    _run([_docs("a,b,c"), GraphNode(id="s", type="slow", version=1, bindings={"text": _edge(("docs", "texts"))})])

    assert Slow.peak == 1
    assert sorted(Slow.seen) == ["a", "b", "c"]


def test_progress_is_per_row_and_the_total_is_known_once_the_index_is_sealed():
    events = _events([_docs("a,b,c"), GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    progress = [(e.done, e.total) for e in events if e.type == "node_progress" and e.node_id == "up"]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [e.type for e in events if getattr(e, "node_id", None) == "up"][0] == "node_start"
    assert [e.type for e in events if getattr(e, "node_id", None) == "up"][-1] == "node_complete"


def test_computed_inputs_reach_the_node_as_keywords():
    """End to end: the template's placeholders are inputs the author
    edges, and the node receives them by name."""
    import re

    from conductor.metadata import Input
    from conductor.widgets import TextWidget

    class Template(NodeDefinition):
        id = "template"
        title = "Template"
        description = "d"
        category = "test"

        def run(self, template: Annotated[Txt, Param(title="Template", show_handle=False, widget=Textarea())] = Txt(""), **values: Txt) -> Out:
            return Txt(re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]), template))

        def compute_inputs(self, declared, values):
            names = re.findall(r"\{(\w+)\}", str(values.get("template", "")))
            return (*declared, *(Input(name=n, dtype=Txt, title=n, widget=TextWidget()) for n in dict.fromkeys(names)))

    registry = _registry()
    registry.register(Template)
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

    assert list(run_sync(compiled).results["t"]["result"]) == ["Dear Ida (24-1)", "Dear Bo (24-1)"]


def test_node_complete_carries_the_series():
    events = _events([_docs("a,b"), GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    (done,) = [e for e in events if e.type == "node_complete" and e.node_id == "up"]
    assert list(done.result["result"]) == ["A", "B"]
    assert events[-1].type == "graph_complete"


# --- every ending is one shape ----------------------------------------------------


def test_a_cancelled_leg_carries_its_results_and_record():
    """`graph_cancelled` and `graph_timeout` end a leg the way `graph_complete`
    and `graph_pending` do — the results so far and the ledger's cells beside
    their reason — so a host can start a new run from any ending."""
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
        return [e async for e in execute(compiled, **kwargs)]

    (ending,) = [e for e in asyncio.run(leg(cache={"a": {"result": Txt("klar")}}, cancel=stop)) if e.type == "graph_cancelled"]
    assert ending.results["a"]["result"] == "klar"
    assert "completed_nodes" not in ending

    seeded = asyncio.run(leg(record=ending.record))
    assert seeded[-1].type == "graph_complete"
    assert seeded[-1].results["b"]["result"] == "B"
    assert not any(e.type == "node_start" and e.node_id == "a" for e in seeded)


# --- a leg ends pending ---------------------------------------------------


class AskNode(NodeDefinition):
    """An asking node: packages its proposal into the question and
    returns Asks. Its output is what the person supplies."""

    id = "asks"
    title = "Ask"
    description = "d"
    category = "test"

    def run(self, proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | Asks:
        return Asks(questions=(Input(name="result", dtype=Txt, title="Answer", widget=Textarea(), default=proposal, optional=True),))


def test_a_run_that_may_ask_says_so_and_still_declares_its_outputs():
    """`-> Out | Asks` is the truthful annotation: the pause is read past, the
    record declares the one output the person's answer lands on."""
    interface = AskNode.versions[1].interface
    assert [(out.name, out.dtype) for out in interface.outputs] == [("result", Txt)]
    assert interface.returns is Txt  # `Out` is one Txt output; the union declared nothing


def _registry_with_asks(*extra):
    registry = _registry()
    for node_cls in (AskNode, *extra):
        registry.register(node_cls)
    return registry


def _leg(compiled, **kwargs):
    async def collect_all():
        return [e async for e in execute(compiled, **kwargs)]

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

    assert events[-1].type == "graph_pending"
    (waiting,) = events[-1].pending
    assert (waiting.node_id, waiting.row) == ("ask", None)
    (question,) = waiting.questions
    assert question.name == "ask.result" and question.default == "proposal"
    assert events[-1].results["other"]["result"] == "X"
    assert "up" not in events[-1].results
    assert not any(e.type == "node_start" and e.node_id == "up" for e in events)


def test_answering_is_the_next_leg_from_the_record_and_the_cache():
    """The answer is the asking node's output in ``cache``; the cells carry
    what the first leg produced, so nothing done is done twice."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="ask", type="asks", version=1, bindings={"proposal": Static("proposal")}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
            GraphNode(id="other", type="upper", version=1, bindings={"text": Static("x")}),
        ]),
        _registry_with_asks(),
    )
    first = _leg(compiled)[-1]

    second = _leg(compiled, record=first.record, cache={"ask": {"result": Txt("yes")}})

    assert second[-1].type == "graph_complete"
    assert second[-1].results["up"]["result"] == "YES"
    assert second[-1].results["ask"]["result"] == "yes"
    assert not any(e.type == "node_start" and e.node_id in ("other", "ask") for e in second)


def test_an_iterating_asking_node_pends_once_per_row_and_is_answered_as_a_series():
    """Ten units blocked are ten questions in one set; the answer is
    the node's output — a series on its iteration index — in one cache entry."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs("a,b"),
            GraphNode(id="ask", type="asks", version=1, bindings={"proposal": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
        ]),
        _registry_with_asks(),
    )
    first = _leg(compiled)[-1]

    assert first.type == "graph_pending"
    assert [(w.node_id, w.row) for w in first.pending] == [("ask", (0,)), ("ask", (1,))]
    assert [w.questions[0].default for w in first.pending] == ["a", "b"]

    second = _leg(compiled, record=first.record, cache={"ask": {"result": Series(Index("docs"), [Txt("x"), Txt("y")])}})

    assert list(second[-1].results["up"]["result"]) == ["X", "Y"]


class AsksIfLong(NodeDefinition):
    """Asks about a text longer than two characters, and passes a short one through."""

    id = "asks-if-long"
    title = "Ask if long"
    description = "d"
    category = "test"

    def run(self, proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | Asks:
        if len(proposal) <= 2:
            return proposal
        return Asks(questions=(Input(name="result", dtype=Txt, title="Answer", widget=Textarea(), default=proposal, optional=True),))


def _asks_if_long(texts):
    return CompiledGraph.from_graph(
        Graph(nodes=[
            _docs(texts),
            GraphNode(id="ask", type="asks-if-long", version=1, bindings={"proposal": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
        ]),
        _registry_with_asks(AsksIfLong),
    )


def test_an_answer_fills_the_rows_it_names_and_the_rows_already_done_stay():
    """Rows 0 and 2 ran in the first leg; only row 1 asked. The answer
    names row 1, and rows 0 and 2 keep what they produced."""
    compiled = _asks_if_long("a,long,b")
    first = _leg(compiled)[-1]
    assert [(w.node_id, w.row) for w in first.pending] == [("ask", (1,))]

    second = _leg(compiled, record=first.record, cache={"ask": {"result": Series(Index("docs"), [Txt("yes")], rows=[(1,)])}})

    assert second[-1].type == "graph_complete"
    assert list(second[-1].results["ask"]["result"]) == ["a", "yes", "b"]
    assert list(second[-1].results["up"]["result"]) == ["A", "YES", "B"]
    (cached,) = [e for e in second if e.type == "node_complete" and e.node_id == "ask"]
    assert cached.cached is True and list(cached.result["result"]) == ["a", "yes", "b"]


def test_a_row_left_unanswered_asks_again_in_the_next_leg():
    compiled = _asks_if_long("long,longer")
    first = _leg(compiled)[-1]

    second = _leg(compiled, record=first.record, cache={"ask": {"result": Series(Index("docs"), [Txt("yes")], rows=[(0,)])}})

    assert second[-1].type == "graph_pending"
    assert [(w.node_id, w.row) for w in second[-1].pending] == [("ask", (1,))]
    assert [(e.done, e.total) for e in second if e.type == "node_progress" and e.node_id == "ask"] == [(1, 2)]
    assert not any(e.type == "node_complete" and e.node_id == "ask" for e in second)


def test_a_unit_already_done_cannot_be_given_a_result():
    """The cells say what a leg produced; a cache entry that contradicts
    them is refused rather than laid over them."""
    compiled = _asks_if_long("a,long")
    first = _leg(compiled)[-1]

    with pytest.raises(ValueError, match="'docs' is already done"):
        _leg(compiled, record=first.record, cache={"docs": {"texts": Series(Index("docs"), [Txt("z")]), "names": Series(Index("docs"), [Txt("n")])}})
    with pytest.raises(ValueError, match=r"'ask' at row \[0\] is already done"):
        _leg(compiled, record=first.record, cache={"ask": {"result": Series(Index("docs"), [Txt("z"), Txt("yes")])}})


def test_a_row_the_run_has_not_produced_cannot_be_answered():
    compiled = _asks_if_long("long")

    with pytest.raises(ValueError, match=r"'ask' has no row \[0\]"):
        _leg(compiled, cache={"ask": {"result": Series(Index("docs"), [Txt("yes")])}})


def test_two_asking_nodes_in_parallel_are_one_pending_set():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="asks", version=1, bindings={"proposal": Static("1")}),
            GraphNode(id="b", type="asks", version=1, bindings={"proposal": Static("2")}),
        ]),
        _registry_with_asks(),
    )
    (ending,) = [e for e in _leg(compiled) if e.type == "graph_pending"]

    assert sorted(w.node_id for w in ending.pending) == ["a", "b"]


def test_run_sync_returns_the_pending_ending():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="ask", type="asks", version=1, bindings={"proposal": Static("p")})]),
        _registry_with_asks(),
    )

    pending = run_sync(compiled)
    assert pending.type == "graph_pending"
    assert [w.node_id for w in pending.pending] == ["ask"]

    answered = _leg(compiled, record=pending.record, cache={"ask": {"result": Txt("yes")}})
    assert answered[-1].type == "graph_complete"
    assert answered[-1].results["ask"]["result"] == "yes"


# --- an embedded graph runs as nodes of the one run --------------------------------


class TypedInside(NodeDefinition):
    """An embedded graph whose inner node pairs what enters with a list the author typed."""

    id = "typed-inside"
    title = "Indlejret"
    description = "d"
    category = "test"
    versions: ClassVar[dict[int, GraphVersion]] = {
        1: GraphVersion(
            graph=(
                GraphNode(id="h", type="upper", version=1),
                GraphNode(id="t", type="pair", version=1, bindings={"a": _edge(("h", "result")), "b": Static(["p", "q"])}),
                GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("t", "result"))}),
            ),
            interface=Interface(
                inputs=(Input(name="h.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
                outputs=(Output(name="j.result", dtype=Txt, title="Result"),),
                returns=Mapping,
            ),
        )
    }


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
    assert compiled.node("emb/t").iterates_on.parent == Index("docs")

    results = run_sync(compiled).results

    assert list(results["emb/j"]["result"]) == joined
    assert results["emb/t"]["result"].rows == tuple((i, n) for i in range(len(joined)) for n in range(2))


def test_an_inner_reduction_over_the_entering_series_runs_once_per_outer_row():
    """Gathering end to end: three texts in, three joined texts out —
    the ledger groups by the iteration index's depth, so the fold of one is one
    rule, not a second kind of index."""

    class Embedded(NodeDefinition):
        id = "inner-graph"
        title = "Indlejret"
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, GraphVersion]] = {
            1: GraphVersion(
                graph=(
                    GraphNode(id="holder", type="upper", version=1, bindings={"text": Static("inner")}),
                    GraphNode(id="gather", type="join", version=1, bindings={"texts": _edge(("holder", "result"))}),
                ),
                interface=Interface(
                    inputs=(Input(name="holder.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt("inner"), optional=True),),
                    outputs=(Output(name="gather.result", dtype=Txt, title="Result"),),
                    returns=Mapping,
                ),
            )
        }

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            _docs("a,b,c"),
            GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.text": _edge(("docs", "texts"))}),
            GraphNode(id="after", type="join", version=1, bindings={"texts": _edge(("emb", "gather.result"))}),
        ]),
        _registry_with_asks(Embedded),
    )
    assert compiled.is_runnable, compiled.problems
    results = run_sync(compiled).results

    assert list(results["emb/holder"]["result"]) == ["A", "B", "C"]
    assert list(results["emb/gather"]["result"]) == ["A", "B", "C"]
    assert results["emb/gather"]["result"].index == Index("docs")
    assert results["after"]["result"] == "A+B+C"
