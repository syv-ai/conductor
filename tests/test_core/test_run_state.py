"""A leg ends with a typed ``RunState``, and a node the graph changed runs again.

The state is what a host stores between legs and hands back: every value, what is done, and a fingerprint per node of
how the graph uses it. It survives JSON, and the next leg's results are
typed like the first's. On restore, a node whose fingerprint differs from
the graph it is restored into — a static edited, a version bumped, a
binding moved — is dropped with everything downstream of it, so those
units run again; a node the graph no longer has is dropped the same way,
and ``without`` drops one on purpose (a host's "run from here").
"""

import asyncio
import json
from typing import Annotated

import pytest
from conductor import AskPerson, CompiledGraph, GraphNode, NodeRegistry, OlderVersion, Upgrade, node
from conductor.dtype import DType
from conductor.execution.run_state import RunState
from conductor.graph.binding import From, Static
from conductor.graph.model import Graph
from conductor.metadata import Param, Result
from conductor.series import Series
from conductor.widgets import Textarea
from helpers import stream_of
from pydantic import ValidationError


class Txt(DType, str):
    id = "state-test-text"
    title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]

calls: list[str] = []


@node("split", title="Split", description="d", category="test")
def split_node(text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
    calls.append("split")
    return [Txt(part) for part in text.split(",")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: In = Txt("")) -> Out:
    calls.append(f"upper:{text}")
    return Txt(text.upper())


def wrap_v1(text: In = Txt("")) -> Out:
    calls.append(f"wrap:{text}")
    return Txt(f"[{text}]")

@node("wrap", title="Wrap", description="d", category="test", older=[OlderVersion(wrap_v1, upgrade=Upgrade())])
def wrap(text: In = Txt("")) -> Out:
    calls.append(f"wrap2:{text}")
    return Txt(f"<{text}>")


@node("echo", title="Echo", description="d", category="test")
def echo(text: In = Txt("")) -> Out:
    calls.append(f"echo:{text}")
    return text


@node("ask", title="Ask", description="d", category="test")
def ask(text: In = Txt("")) -> Out | AskPerson:
    calls.append(f"ask:{text}")
    return AskPerson(defaults={"result": text}, widgets={"result": Textarea()})


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for cls in (split_node, upper_node, wrap, echo, ask):
        reg = reg.extended_with([cls])
    return reg


def _compiled(nodes: list[GraphNode]) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _leg(compiled: CompiledGraph, **kw) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in stream_of(compiled, **kw)]

    return asyncio.run(run())


def _edge(graph_node: str, field: str = "result") -> From:
    return From(f"{graph_node}.{field}")


def _through_json(state: RunState) -> RunState:
    return RunState.model_validate(json.loads(state.model_dump_json()))


# -- the state survives JSON ---------------------------------------------------------


def test_a_paused_legs_state_survives_json_and_the_next_leg_returns_typed_results():
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="ask", type="ask", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])

    first = _leg(compiled)
    assert first[-1].state.status == "waiting"
    state = first[-1].state
    assert isinstance(state, RunState)

    restored = _through_json(state)
    second = _leg(compiled, state=restored, answers={"ask@0": {"result": Txt("x")}, "ask@1": {"result": Txt("y")}})

    assert second[-1].state.status == "complete"
    parts = compiled.results(second[-1].state)["split"]["result"]
    assert isinstance(parts, Series) and all(type(v) is Txt for v in parts)
    assert list(compiled.results(second[-1].state)["up"]["result"]) == ["X", "Y"]
    assert calls.count("split") == 1


def test_the_state_carries_a_fingerprint_per_node():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
    ])

    state = _leg(compiled)[-1].state

    assert set(state.node_fingerprints) == {"a", "b"}
    assert state.node_fingerprints["a"] == compiled.node("a").fingerprint
    assert all(len(fp) == 64 for fp in state.node_fingerprints.values())


# -- a changed node runs again -------------------------------------------------------


def test_a_static_edited_between_legs_reruns_that_node_and_its_readers_and_nothing_else():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="wrap", version=1, bindings={"text": _edge("b")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static("alone")}),
    ]
    state = _leg(_compiled(nodes))[-1].state
    calls.clear()

    edited = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"text": Static("y")}), *nodes[1:]])
    ending = _leg(edited, state=_through_json(state))[-1].state

    assert ending.status == "complete"
    assert edited.results(ending)["c"]["result"] == "[Y]"
    assert edited.results(ending)["d"]["result"] == "alone"
    assert sorted(calls) == ["echo:y", "upper:y", "wrap:Y"]


def test_a_node_removed_between_legs_is_dropped_with_its_readers():
    calls.clear()
    state = _leg(_compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="echo", version=1, bindings={"text": Static("c")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("c")}),
    ]))[-1].state
    calls.clear()

    # ``c`` is gone and ``d`` now reads ``a``: ``c``'s values are dropped, ``d`` runs again, ``a`` and ``b`` do not.
    smaller = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("a")}),
    ])
    ending = _leg(smaller, state=_through_json(state))[-1].state

    assert ending.status == "complete"
    assert set(smaller.results(ending)) == {"a", "b", "d"}
    assert smaller.results(ending)["d"]["result"] == "[x]"
    assert calls == ["wrap:x"]


def test_a_nodes_shape_changed_between_legs_reruns_it():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("a")}),
    ]
    state = _leg(_compiled(nodes))[-1].state
    calls.clear()

    bumped = _compiled([nodes[0], GraphNode(id="w", type="wrap", version=2, bindings={"text": _edge("a")})])
    ending = _leg(bumped, state=_through_json(state))[-1].state

    assert bumped.results(ending)["w"]["result"] == "<x>"
    assert calls == ["wrap2:x"]


def test_without_drops_a_node_on_purpose_with_everything_downstream():
    """Run from here: the host drops one node from the state, and the
    restore drops what reads it, so they run again while the rest is kept."""
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("up")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static("alone")}),
    ])
    state = _leg(compiled)[-1].state
    calls.clear()

    ending = _leg(compiled, state=_through_json(state.without("up")))[-1].state

    assert ending.status == "complete"
    assert list(compiled.results(ending)["w"]["result"]) == ["[A]", "[B]"]
    assert sorted(calls) == ["upper:a", "upper:b", "wrap:A", "wrap:B"]
    assert "up" not in state.without("up").node_fingerprints and "split" in state.without("up").node_fingerprints


# -- what a node returns and what a person answers are checked ------------------------


def test_a_return_of_the_wrong_type_fails_the_returning_node_with_invalid_output():
    """``-> Text`` returning ``42`` is the node's fault, not the next node's:
    the unit that returned it fails with ``invalid_output``, naming the
    output, and nothing is written."""
    calls.clear()

    @node("wrong", title="Wrong", description="d", category="test")
    def wrong(text: In = Txt("")) -> Out:
        return 42  # type: ignore[return-value]

    reg = NodeRegistry()
    for cls in (wrong, upper_node):
        reg = reg.extended_with([cls])
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="w", type="wrong", version=1, bindings={"text": Static("x")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("w")}),
    ]), reg)

    events = _leg(compiled)

    error = next(e for e in events if e.type == "node_failed")
    assert error.node_id == "w"
    assert error.cause.type == "invalid_output"
    assert error.cause.msg == "The node returned a value that is not what it declared."
    assert "result" in error.cause.ctx["reason"] and "42" not in error.cause.ctx["reason"]
    assert events[-1].state.status == "failed" and "w.result" not in events[-1].state.output_columns
    assert calls == []


def test_a_state_value_that_does_not_read_back_as_its_type_is_refused():
    import pytest
    from conductor import InvalidCallError

    compiled = _compiled([GraphNode(id="e", type="echo", version=1, bindings={"text": Static("x")})])
    state = _leg(compiled)[-1].state
    dumped = state.model_dump(mode="json")
    dumped["output_columns"]["e.result"]["values"] = [{"not": "text"}]

    with pytest.raises(InvalidCallError, match=r"e\.result"):
        _leg(compiled, state=RunState.model_validate(dumped))


def test_a_node_that_runs_once_is_at_the_empty_row_on_the_wire():
    """The top is the empty row, ``[]`` in JSON, for a value and a done unit
    alike; a state read back from that JSON runs nothing again."""
    compiled = _compiled([GraphNode(id="e", type="echo", version=1, bindings={"text": Static("x")})])
    wire = _leg(compiled)[-1].state.model_dump(mode="json")
    assert wire["output_columns"]["e.result"]["rows"] == [[]] and wire["finished_rows"] == {"e": [[]]}

    calls.clear()
    assert _leg(compiled, state=RunState.model_validate(wire))[-1].state.status == "complete"
    assert calls == []


def _once_state() -> dict:
    compiled = _compiled([GraphNode(id="e", type="echo", version=1, bindings={"text": Static("x")})])
    return _leg(compiled)[-1].state.model_dump(mode="json")


@pytest.mark.parametrize(("edit", "refusal"), [
    (lambda d: d["output_columns"]["e.result"].update(rows=[[0]]), r"e\.result is at row \[0\]"),
    (lambda d: d["finished_rows"].update(e=[[0]]), r"'e' at row \[0\]"),
    (lambda d: d["output_columns"].update({"e.reslt": d["output_columns"].pop("e.result")}), r"e\.reslt, which the graph does not have"),
])
def test_a_state_entry_a_node_that_runs_once_has_no_place_for_is_refused(edit, refusal):
    from conductor import InvalidCallError

    compiled = _compiled([GraphNode(id="e", type="echo", version=1, bindings={"text": Static("x")})])
    dumped = _leg(compiled)[-1].state.model_dump(mode="json")
    edit(dumped)

    with pytest.raises(InvalidCallError, match=refusal):
        _leg(compiled, state=RunState.model_validate(dumped))


@pytest.mark.parametrize("column", [
    {"rows": [[]], "values": []},
    {"rows": None, "values": []},
    {"rows": [[]], "values": ["A"], "stray": 1},
    {"rows": "x", "values": ["A"]},
])
def test_a_column_that_does_not_line_up_is_refused_where_it_arrives(column):
    with pytest.raises(ValidationError):
        RunState.model_validate({"status": "complete", "output_columns": {"e.result": column}})


def test_a_state_whose_key_is_not_an_address_is_refused_where_it_arrives():
    with pytest.raises(ValidationError):
        RunState.model_validate({"status": "complete", "output_columns": {"e": {"rows": [], "values": []}}})


def test_an_answer_reads_through_the_outputs_type():
    """A host may hand an answer in wire form, as it came over HTTP, or as the typed value."""
    calls.clear()
    once = _compiled([
        GraphNode(id="ask", type="ask", version=1, bindings={"text": Static("q")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])
    first = _leg(once)[-1].state
    ending = _leg(once, state=first, answers={"ask": {"result": "typed in"}})[-1].state
    assert ending.status == "complete" and once.results(ending)["up"]["result"] == "TYPED IN"
    assert type(once.results(ending)["ask"]["result"]) is Txt


def test_a_unit_whose_second_output_is_invalid_writes_nothing():
    """A unit's outputs are checked before any is written: the state holds
    all of them or none, never a half-written row."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Both:
        head: Annotated[Txt, Result(title="Head")]
        parts: Annotated[Series[Txt], Result(title="Parts")]

    @node("splits", title="Splits", description="d", category="test")
    def splits(text: In = Txt("")) -> Both:
        return Both(head=Txt("h"), parts=3)  # type: ignore[arg-type]

    reg = NodeRegistry([splits])
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="splits", version=1, bindings={"text": Static("x")})]), reg)

    state = compiled.run_sync()

    assert state.status == "failed" and state.failure.type == "invalid_output"
    assert state.output_columns == {} and state.finished_rows == {}


# -- edits between legs, and a state read after the ledger moved on ----------------------


def test_a_list_set_on_a_node_edited_between_legs_reruns_on_the_new_rows():
    """The rows of a list set on a node are the graph's, never the state's: grown,
    shrunk or added between legs, the node runs on what the author typed now."""
    calls.clear()
    state = _leg(_compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(["a", "b"])})]))[-1].state
    calls.clear()

    longer = _compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(["a", "b", "c"])})])
    grown = _leg(longer, state=_through_json(state))[-1].state
    assert grown.status == "complete" and list(longer.results(grown)["e"]["result"]) == ["A", "B", "C"]
    shorter = _compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(["a"])})])
    shrunk = _leg(shorter, state=_through_json(state))[-1].state
    assert shrunk.status == "complete" and list(shorter.results(shrunk)["e"]["result"]) == ["A"]
    wider = _compiled([
        GraphNode(id="e", type="upper", version=1, bindings={"text": Static(["a", "b"])}),
        GraphNode(id="f", type="upper", version=1, bindings={"text": Static(["x", "y"])}),
    ])
    added = _leg(wider, state=_through_json(state))[-1].state
    assert added.status == "complete" and list(wider.results(added)["f"]["result"]) == ["X", "Y"]
    assert "upper:a" not in calls[-2:]  # ``e`` was kept on the last leg


def test_a_static_hashes_as_its_type_writes_it_not_as_the_author_spelled_it():
    class Num(DType, float):
        id = "state-test-number"
        title = "Number"

    @node("half", title="Half", description="d", category="test")
    def half(n: Annotated[Num, Param(title="N", widget=Textarea())] = Num(0)) -> Annotated[Num, Result(title="Out")]:
        return Num(n / 2)

    def fingerprint(value):
        reg = NodeRegistry([half])
        return CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="h", type="half", version=1, bindings={"n": Static(value)})]), reg).node("h").fingerprint

    assert fingerprint(2) == fingerprint(2.0) == fingerprint(Num(2))
    assert fingerprint(2) != fingerprint(3)


def test_series_outputs_of_two_lengths_both_land_each_on_its_own_index():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Both:
        a: Annotated[Series[Txt], Result(title="A")]
        b: Annotated[Series[Txt], Result(title="B")]

    @node("uneven", title="Uneven", description="d", category="test")
    def uneven(text: In = Txt("")) -> Both:
        return Both(a=[Txt("x")], b=[Txt("y"), Txt("z")])

    reg = NodeRegistry([uneven])
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="u", type="uneven", version=1, bindings={"text": Static("x")})]), reg)

    ending = list(_leg(compiled))[-1].state
    result = compiled.results(ending)["u"]

    assert ending.status == "complete"
    assert (list(result["a"]), list(result["b"])) == (["x"], ["y", "z"])


def test_the_state_an_ending_carries_reads_as_a_stored_one_does():
    """A state a leg just built holds its values typed; the same state stored
    and read back holds their wire form. Both read the same."""
    compiled = _compiled([
        GraphNode(id="s", type="split", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("s")}),
    ])
    state = _leg(compiled)[-1].state
    live, stored = compiled.results(state), compiled.results(_through_json(state))

    assert list(live["up"]["result"]) == list(stored["up"]["result"]) == ["A", "B"]
    assert live["up"]["result"].rows == stored["up"]["result"].rows


def test_a_live_state_read_over_an_edited_graph_leaves_out_what_changed():
    """Over an edited graph, the edited node is left out of what a state reads as."""
    nodes = [GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x")})]
    state = _leg(_compiled(nodes))[-1].state
    edited = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"text": Static("y")})])

    assert "a" not in edited.results(state)
