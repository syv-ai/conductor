"""The ledger, for what a run through the public doors can't show: a skip's reach, what deciding a group costs, and the ledger agreeing with a restored copy of itself and with asking every unit, on random graphs."""

import random
from dataclasses import dataclass
from typing import Annotated, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Param,
    Ref,
    Result,
    Series,
    Static,
    Table,
    node,
)
from conductor._internal.execution.run_ledger import (
    BlankInputsStopCall,
    RunLedger,
    SkippedAtRow,
    Unit,
)
from conductor._internal.execution.validate_unit_outputs import validate_unit_outputs
from conductor._internal.execution.waiting_units import waiting_unit_for
from conductor._internal.node_signature import split_return_into_outputs
from conductor.widgets import Textarea


def _results_of(ledger: RunLedger) -> dict:
    """What a host reads off the state this ledger would write now."""
    return ledger._compiled.results(ledger.state("running"))


class Txt(DType, str):
    id = "ledger-test-txt"
    display_title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Names")]


@dataclass(frozen=True)
class Column:
    full: Annotated[Series[Txt], Result(title="Full")]
    empty: Annotated[Series[Txt], Result(title="Empty")]


@node("docs", title="Docs", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Documents:
    return Documents(texts=(), names=())


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("pair", title="Pair", description="d", category="test")
def pair(a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
    return Txt(a + b)


@node("lines", title="Lines", description="d", category="test")
def lines_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return text.splitlines()


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


@node("join-per-text", title="Join per text", description="d", category="test")
def join_per_text(text: Annotated[Txt, Param(title="Text")], texts: Annotated[Series[Txt], Param(title="Texts")]) -> Out:
    return Txt(text + "+".join(texts))


@node("decide", title="Gate", description="d", category="test")
def gate(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Column:
    """A decision over a whole column: the column goes out on ``full`` or ``empty``."""
    return Column(full=texts, empty=SKIPPED) if len(texts) else Column(full=SKIPPED, empty=texts)


def _registry():
    registry = NodeRegistry()
    for node_cls in (docs, upper_node, pair, lines_node, join_node, join_per_text, gate):
        registry = registry.extended_with([node_cls])
    return registry


def _ledger(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return RunLedger(compiled)


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


DOCS = GraphNode(id="docs", type="docs", version=1)


def test_a_decision_over_a_whole_column_gates_what_hangs_off_the_branch_not_taken():
    """An "if the list is empty" decision: the untaken branch is skipped at the top,
    not an empty series, so a reduction downstream of it does not run, and a node
    running per row of it has no units and finishes with it."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="decide", type="decide", version=1, bindings={"texts": _edge(("docs", "texts"))}),
        GraphNode(id="full", type="join", version=1, bindings={"texts": _edge(("decide", "full"))}),
        GraphNode(id="empty", type="join", version=1, bindings={"texts": _edge(("decide", "empty"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("decide", "empty"))}),
        GraphNode(id="after", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.units_ready_to_start()
    ledger.record(("docs", ()), {"texts": ["a", "b"], "names": ["x", "y"]})
    recorded = ledger.record(("decide", ()), {"full": ["a", "b"], "empty": SKIPPED})

    assert set(recorded.units_ready) == {("full", ()), ("empty", ()), ("after", ())}
    assert "up" in recorded.nodes_completed  # a node with no rows finishes when its index gets none
    assert list(ledger.inputs_for(("full", ()))["texts"]) == ["a", "b"]
    assert ledger.inputs_for(("empty", ())) == SkippedAtRow(row=())
    assert ledger.units("up") == []
    assert ledger.inputs_for(("after", ())) == SkippedAtRow(row=())
    ledger.record(("empty", ()), SkippedAtRow(row=()))
    ledger.record(("after", ()), SkippedAtRow(row=()))
    results = _results_of(ledger)
    assert "empty" not in results and "up" not in results and "after" not in results
    assert "empty" not in ledger.result_of("decide")  # the branch not taken is absent, never SKIPPED
    assert ledger.result_of("up") is None  # the node stands for a skipped row as a whole


def test_a_skip_written_on_a_longer_row_reaches_the_group_above_it():
    """A skip on a parent row carries on down the rows made under it, so a
    reduction of those rows is skipped too; a skip the tree can't tell about."""

    @dataclass(frozen=True)
    class Gate:
        open: Annotated[Txt, Result(title="Open", choice="c")]
        shut: Annotated[Txt, Result(title="Shut", choice="c")]

    @node("gate", title="Gate", description="d", category="test")
    def gate_row(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Gate:
        return Gate(open=text, shut=SKIPPED)

    compiled = CompiledGraph.from_graph(Graph(nodes=[
        DOCS,
        GraphNode(id="g", type="gate", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="line", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("line", "result")), "b": _edge(("g", "open"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("p", "result"))}),
    ]), _registry().extended_with([gate_row]))
    assert compiled.is_runnable, compiled.problems
    ledger = RunLedger(compiled)
    ledger.record(("docs", ()), {"texts": ["x", "y"], "names": ["n", "m"]})
    ledger.record(("g", (0,)), {"open": "x", "shut": SKIPPED})
    ledger.record(("g", (1,)), {"open": SKIPPED, "shut": "y"})
    ledger.record(("line", (0,)), {"result": ["a"]})
    ledger.record(("line", (1,)), {"result": ["b", "c"]})
    ledger.record(("p", (0, 0)), {"result": "ax"})
    assert ledger.inputs_for(("p", (1, 0))) == ledger.inputs_for(("p", (1, 1))) == SkippedAtRow(row=(1,))
    ledger.record(("p", (1, 0)), SkippedAtRow(row=(1,)))
    ledger.record(("p", (1, 1)), SkippedAtRow(row=(1,)))

    assert ledger.inputs_for(("j", (1,))) == SkippedAtRow(row=(1,))
    assert list(ledger.inputs_for(("j", (0,)))["texts"]) == ["ax"]
    state = ledger.state("running")
    assert state.output_columns["p.result"].skipped == [(1,)]
    assert RunLedger.restore(compiled, state).inputs_for(("j", (1,))) == SkippedAtRow(row=(1,))


def test_a_skip_above_a_maker_reaches_every_index_under_it():
    """A document whose gate is shut gets no paragraphs and so no lines. A reduction per
    document over the lines is skipped there, though the node that makes the lines never had a
    row to run on, while the open document is reduced as usual."""
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        DOCS,
        GraphNode(id="g", type="per-row-gate", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="paras", type="lines", version=1, bindings={"text": _edge(("g", "open"))}),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("paras", "result"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
        GraphNode(id="j", type="join-per-text", version=1, bindings={"text": _edge(("docs", "texts")), "texts": _edge(("up", "result"))}),
    ]), _registry().extended_with([per_row_gate]))
    assert compiled.is_runnable, compiled.problems
    ledger = RunLedger(compiled)
    ledger.units_ready_to_start()
    ledger.record(("docs", ()), {"texts": ["x", "y"], "names": ["n", "m"]})
    ledger.record(("g", (0,)), {"open": "x", "shut": SKIPPED})
    recorded = ledger.record(("g", (1,)), {"open": SKIPPED, "shut": "y"})
    assert ("paras", (1,)) in recorded.units_ready
    ledger.record(("paras", (0,)), {"result": ["p"]})
    ledger.record(("lines", (0, 0)), {"result": ["l"]})
    assert ledger.record(("up", (0, 0, 0)), {"result": "L"}).units_ready == [("j", (0,))]

    recorded = ledger.record(("paras", (1,)), ledger.inputs_for(("paras", (1,))))

    assert recorded.units_ready == [("j", (1,))]
    assert ledger.units("lines") == [("lines", (0, 0))]
    assert ledger.inputs_for(("j", (1,))) == SkippedAtRow(row=(1,))
    assert list(ledger.inputs_for(("j", (0,)))["texts"]) == ["L"]
    assert sorted(recorded.nodes_completed) == ["lines", "paras", "up"]


def test_a_node_over_an_empty_list_has_no_units_and_is_not_announced():
    """An empty list is rows that were made, none of them. The node running per row
    of it never starts, so it is complete but nothing reports it; the reduction
    over it runs, on an empty series."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.units_ready_to_start()

    recorded = ledger.record(("docs", ()), {"texts": [], "names": []})

    assert recorded.nodes_completed == ["docs"]
    assert recorded.units_ready == [("j", ())]
    assert ledger.node_is_complete("up") and "up" not in ledger.completed_nodes()
    assert list(ledger.inputs_for(("j", ()))["texts"]) == []


def test_a_group_reduction_is_asked_a_handful_of_times_however_long_the_group():
    """What a reduction reads is one thing, its writer's rows; it is parked on that
    and asked again when the last of them is written, not once per write."""
    asks: list[int] = []
    for lines in (100, 800):
        ledger = _ledger([
            DOCS,
            GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
            GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
        ])
        ledger.units_ready_to_start()
        calls = 0
        original = ledger.first_thing_unit_still_needs

        def counted(unit):
            nonlocal calls
            if unit[0] == "j":
                calls += 1
            return original(unit)

        ledger.first_thing_unit_still_needs = counted  # type: ignore[method-assign]
        ledger.record(("docs", ()), {"texts": ["text"], "names": ["n"]})
        ledger.record(("lines", (0,)), {"result": ["l"] * lines})
        for line in range(lines):
            ledger.record(("up", (0, line)), {"result": "L"})
        asks.append(calls)

    assert asks[0] == asks[1] <= 3, asks


@dataclass(frozen=True)
class _Gated:
    open: Annotated[Txt, Result(title="Open", choice="c")]
    shut: Annotated[Txt, Result(title="Shut", choice="c")]


@node("per-row-gate", title="Gate", description="d", category="test")
def per_row_gate(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> _Gated:
    return _Gated(open=text, shut=SKIPPED)


# -- the ledger against itself ----------------------------------------------------


class Rows(TypedDict):
    name: Txt


@node("split-table", title="Split", description="d", category="test")
def split_table(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Table[Rows], Result(title="Rows")]:
    return pl.DataFrame({"name": list(text)})


@node("branch-if", title="Branch", description="d", category="test")
def branch_if(text: Annotated[Txt, Param(title="Text")]) -> Annotated[Txt, Result(title="Kept")]:
    return SKIPPED if ord(text[:1] or "b") % 2 else text


@node("chars", title="Chars", description="d", category="test")
def chars(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Chars")]:
    return [] if text[:1] in ("", "b", "B") else [Txt(char) for char in text * 2]


@node("ask-per-row", title="Ask", description="d", category="test")
def ask_per_row(text: Annotated[Txt, Param(title="Text")]) -> Out | AskPerson:
    return text if ord(text[:1] or "a") % 3 else AskPerson(prompt=text)


def _random_graph(rng: random.Random, registry: NodeRegistry) -> CompiledGraph:
    """A runnable graph of the catalogue above, wired at random, one per seed tried."""
    while True:
        refs = ["src.result"]
        nodes = [GraphNode(id="src", type="chars", version=1, bindings={"text": Static(Txt(rng.choice(["ab", "ba", "cab", "bcd", "acbd"])))})]
        for i in range(rng.randint(2, 7)):
            node_id = f"n{i}"
            source = From(rng.choice(refs))
            kind = rng.choice(["upper", "branch-if", "branch-if", "chars", "chars", "chars", "join", "split-table", "pair", "list-on-node", "join-per-text", "ask-per-row"])
            bindings: dict = {"text": source}
            if kind == "join":
                bindings = {"texts": source}
            elif kind == "pair":
                bindings = {"a": source, "b": From(rng.choice(refs))}
            elif kind == "list-on-node":
                kind, bindings = "pair", {"a": source, "b": Static([Txt("p"), Txt("q")])}
            elif kind == "join-per-text":
                bindings = {"text": source, "texts": From(rng.choice(refs))}
            nodes.append(GraphNode(id=node_id, type=kind, version=1, bindings=bindings))
            if kind == "split-table":
                refs.extend([f"{node_id}.result.name"])
            else:
                refs.append(f"{node_id}.result")
        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        if compiled.is_runnable:
            return compiled


def _canon(value):
    """A unit's inputs, comparable: a series by its rows and values, a table by its rows."""
    if isinstance(value, Series):
        return ("series", value.index.id, value.rows, tuple(_canon(item) for item in value.values))
    if isinstance(value, pl.DataFrame):
        return ("table", tuple(value.columns), tuple(value.rows()))
    if isinstance(value, dict):
        return {key: _canon(item) for key, item in value.items()}
    return value


def _unit_outputs(ledger: RunLedger, compiled: CompiledGraph, unit: Unit, inputs: dict):
    """What this unit returns, really: the node called and its return read as the engine does."""
    node_id, row = unit
    compiled_node = compiled.node(node_id)
    value = compiled_node.version.run(**compiled_node._validate_call_inputs(inputs))
    if isinstance(value, AskPerson):
        return value
    by_output = split_return_into_outputs(compiled_node.version.return_shape, value, compiled_node.interface.outputs, compiled_node.version.interface.outputs)
    cells, problems = validate_unit_outputs(compiled_node, row, by_output)
    assert not problems, problems
    return cells


def _drive_graph(seed: int, check_after_each_record) -> None:
    rng = random.Random(seed)
    registry = _registry().extended_with([split_table, branch_if, chars, ask_per_row])
    compiled = _random_graph(rng, registry)
    ledger = RunLedger(compiled)
    in_flight = set(ledger.units_ready_to_start())
    parked_answers: list[tuple[Unit, dict]] = []
    check_after_each_record(ledger, compiled, in_flight, [])
    for _ in range(5000):
        if in_flight:
            unit = rng.choice(sorted(in_flight))
            in_flight.discard(unit)
            inputs = ledger.inputs_for(unit)
            if isinstance(inputs, (SkippedAtRow, BlankInputsStopCall)):
                recorded = ledger.record(unit, inputs)
            else:
                outputs = _unit_outputs(ledger, compiled, unit, inputs)
                if isinstance(outputs, AskPerson):
                    waiting_unit = waiting_unit_for(compiled.node(unit[0]), unit[1], outputs, inputs, "0" * 32)
                    ledger.mark_waiting(unit, waiting_unit)
                    parked_answers.append((unit, {"result": Txt("answered")}))
                    check_after_each_record(ledger, compiled, in_flight, [])
                    continue
                recorded = ledger.record(unit, outputs)
        elif parked_answers:
            unit, cells = parked_answers.pop(rng.randrange(len(parked_answers)))
            recorded = ledger.record(unit, cells)
        else:
            break
        in_flight.update(recorded.units_ready)
        check_after_each_record(ledger, compiled, in_flight, recorded.units_ready)
    assert not in_flight, "a unit never finished"
    assert not ledger.waiting()
    assert all(ledger.node_is_complete(node_id) for node_id in compiled.execution_order), "a node did not finish"


@pytest.mark.parametrize("seed", range(120))
def test_a_restore_after_every_write_asks_answers_and_states_the_same(seed):
    """Restoring from a state written at any moment gives the ledger the same answers."""

    def check(ledger: RunLedger, compiled: CompiledGraph, in_flight: set[Unit], _released: list[Unit]) -> None:
        state = ledger.state("running")
        restored = RunLedger.restore(compiled, state)
        ready = set(restored.units_ready_to_start())
        assert ready == in_flight
        for unit in ready:
            assert _canon(restored.inputs_for(unit)) == _canon(ledger.inputs_for(unit))
        assert restored.state("running").model_dump_json() == state.model_dump_json()

    _drive_graph(seed, check)


@pytest.mark.parametrize("seed", range(120, 240))
def test_what_record_returns_is_exactly_what_it_released(seed):
    """What each ``record`` returned plus what is in flight is every unit found ready by asking them all."""

    def check(ledger: RunLedger, compiled: CompiledGraph, in_flight: set[Unit], _released: list[Unit]) -> None:
        assert set(RunLedger.restore(compiled, ledger.state("running")).units_ready_to_start()) == in_flight

    _drive_graph(seed, check)
