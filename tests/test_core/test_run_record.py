"""A leg ends with a typed run record, and a node the graph changed runs again.

The record is what a host stores between legs and hands back: every cell
in wire form, the rows born, what is done, and a fingerprint per node of
how the graph placed it. It survives JSON, and the next leg's results are
typed like the first's. On restore, a node whose fingerprint differs from
the graph it is restored into — a static edited, a version bumped, a
binding moved — is dropped with everything downstream of it, so those
units run again; a node the graph no longer has is dropped the same way,
and ``without`` drops one on purpose (a host's "run from here").
"""

import asyncio
import json
from typing import Annotated

from conductor import Asks, CompiledGraph, GraphNode, NodeRegistry
from conductor.dtype import DType
from conductor.execution.engine import execute
from conductor.execution.record import RunRecord
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.metadata import Input, Param, Result
from conductor.node import NodeDefinition, upgrade, version
from conductor.ref import Ref
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "record-test-text"
    title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]

calls: list[str] = []


class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
        calls.append("split")
        return [Txt(part) for part in text.split(",")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"upper:{text}")
        return Txt(text.upper())


class Wrap(NodeDefinition):
    id = "wrap"
    title = "Wrap"
    description = "d"
    category = "test"

    @upgrade(1, 2)
    def _v1_to_v2(values):
        return values

    @version(1)
    def run_v1(self, text: In = Txt("")) -> Out:
        calls.append(f"wrap:{text}")
        return Txt(f"[{text}]")

    @version(2)
    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"wrap2:{text}")
        return Txt(f"<{text}>")


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out:
        calls.append(f"echo:{text}")
        return text


class Ask(NodeDefinition):
    id = "ask"
    title = "Ask"
    description = "d"
    category = "test"

    def run(self, text: In = Txt("")) -> Out | Asks:
        calls.append(f"ask:{text}")
        return Asks(questions=(Input(name="result", dtype=Txt, title="Answer", widget=Textarea(), default=text, optional=True),))


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for cls in (Split, Upper, Wrap, Echo, Ask):
        reg.register(cls)
    return reg


def _compiled(nodes: list[GraphNode]) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled


def _leg(compiled: CompiledGraph, **kw) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in execute(compiled, **kw)]

    return asyncio.run(run())


def _edge(node: str, field: str = "result") -> Edges:
    return Edges(refs=(Ref(node, field),))


def _through_json(record: RunRecord) -> RunRecord:
    return RunRecord.model_validate(json.loads(json.dumps(record.model_dump())))


# -- the record survives JSON ---------------------------------------------------------


def test_a_paused_legs_record_survives_json_and_the_next_leg_returns_typed_results():
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="ask", type="ask", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])

    first = _leg(compiled)
    assert first[-1]["type"] == "graph_pending"
    record = first[-1]["record"]
    assert isinstance(record, RunRecord)

    restored = _through_json(record)
    second = _leg(compiled, record=restored, cache={"ask": {"result": Series(compiled.field(Ref("split", "result")).index, [Txt("x"), Txt("y")])}})

    assert second[-1]["type"] == "graph_complete"
    parts = second[-1]["results"]["split"]["result"]
    assert isinstance(parts, Series) and all(type(v) is Txt for v in parts)
    assert list(second[-1]["results"]["up"]["result"]) == ["X", "Y"]
    assert calls.count("split") == 1


def test_the_record_carries_a_fingerprint_per_node():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
    ])

    record = _leg(compiled)[-1]["record"]

    assert set(record.node_fingerprints) == {"a", "b"}
    assert record.node_fingerprints["a"] == compiled.node("a").fingerprint
    assert all(len(fp) == 64 for fp in record.node_fingerprints.values())


# -- a changed node runs again -------------------------------------------------------


def test_a_static_edited_between_legs_reruns_that_node_and_its_readers_and_nothing_else():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="wrap", version=1, bindings={"text": _edge("b")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static(value="alone")}),
    ]
    record = _leg(_compiled(nodes))[-1]["record"]
    calls.clear()

    edited = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="y")}), *nodes[1:]])
    ending = _leg(edited, record=_through_json(record))[-1]

    assert ending["type"] == "graph_complete"
    assert ending["results"]["c"]["result"] == "[Y]"
    assert ending["results"]["d"]["result"] == "alone"
    assert sorted(calls) == ["echo:y", "upper:y", "wrap:Y"]


def test_a_node_removed_between_legs_is_dropped_with_its_readers():
    calls.clear()
    record = _leg(_compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="c", type="echo", version=1, bindings={"text": Static(value="c")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("c")}),
    ]))[-1]["record"]
    calls.clear()

    # ``c`` is gone and ``d`` now reads ``a``: ``c``'s cells are dropped, ``d`` runs again, ``a`` and ``b`` do not.
    smaller = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge("a")}),
        GraphNode(id="d", type="wrap", version=1, bindings={"text": _edge("a")}),
    ])
    ending = _leg(smaller, record=_through_json(record))[-1]

    assert ending["type"] == "graph_complete"
    assert set(ending["results"]) == {"a", "b", "d"}
    assert ending["results"]["d"]["result"] == "[x]"
    assert calls == ["wrap:x"]


def test_a_nodes_shape_changed_between_legs_reruns_it():
    calls.clear()
    nodes = [
        GraphNode(id="a", type="echo", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("a")}),
    ]
    record = _leg(_compiled(nodes))[-1]["record"]
    calls.clear()

    bumped = _compiled([nodes[0], GraphNode(id="w", type="wrap", version=2, bindings={"text": _edge("a")})])
    ending = _leg(bumped, record=_through_json(record))[-1]

    assert ending["results"]["w"]["result"] == "<x>"
    assert calls == ["wrap2:x"]


def test_without_drops_a_node_on_purpose_with_everything_downstream():
    """Run from here: the host drops one node from the record, and the
    restore drops what reads it, so they run again while the rest is kept."""
    calls.clear()
    compiled = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="w", type="wrap", version=1, bindings={"text": _edge("up")}),
        GraphNode(id="d", type="echo", version=1, bindings={"text": Static(value="alone")}),
    ])
    record = _leg(compiled)[-1]["record"]
    calls.clear()

    ending = _leg(compiled, record=_through_json(record.without("up")))[-1]

    assert ending["type"] == "graph_complete"
    assert list(ending["results"]["w"]["result"]) == ["[A]", "[B]"]
    assert sorted(calls) == ["upper:a", "upper:b", "wrap:A", "wrap:B"]
    assert "up" not in record.without("up").node_fingerprints and "split" in record.without("up").node_fingerprints


# -- what a node returns and what a person answers are checked ------------------------


def test_a_return_of_the_wrong_type_fails_the_returning_node_with_invalid_output():
    """``-> Text`` returning ``42`` is the node's fault, not the next node's:
    the unit that returned it fails with ``invalid_output``, naming the
    output, and nothing is written."""
    calls.clear()

    class Wrong(NodeDefinition):
        id = "wrong"
        title = "Wrong"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            return 42  # type: ignore[return-value]

    reg = NodeRegistry()
    for cls in (Wrong, Upper):
        reg.register(cls)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="w", type="wrong", version=1, bindings={"text": Static(value="x")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("w")}),
    ]), reg)

    events = _leg(compiled)

    error = next(e for e in events if e["type"] == "node_error")
    assert error["node_id"] == "w"
    assert error["cause"].code == "invalid_output"
    assert error["cause"].message == "The node returned a value that is not what it declared."
    assert "result" in error["cause"].details["reason"] and "42" not in error["cause"].details["reason"]
    assert events[-1]["type"] == "graph_error" and events[-1]["record"].cells == []
    assert calls == []


def test_a_misspelled_output_in_the_cache_names_the_node_and_the_output():
    compiled = _compiled([GraphNode(id="e", type="echo", version=1, bindings={"text": Static(value="x")})])

    import pytest

    with pytest.raises(ValueError, match=r"'e' has no output 'reslt'"):
        _leg(compiled, cache={"e": {"reslt": Txt("y")}})
    with pytest.raises(ValueError, match=r"'e'.*'result'"):
        _leg(compiled, cache={"e": {}})


def test_an_answer_decodes_through_the_codec_by_the_outputs_type():
    """A host may hand an answer in wire form — as it came over HTTP — or as
    the typed value; both land as the type the output declares, and a
    per-row answer as a series on the node's own index."""
    calls.clear()
    once = _compiled([
        GraphNode(id="ask", type="ask", version=1, bindings={"text": Static(value="q")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])
    ending = _leg(once, cache={"ask": {"result": "typed in"}})[-1]
    assert ending["type"] == "graph_complete" and ending["results"]["up"]["result"] == "TYPED IN"
    assert type(ending["results"]["ask"]["result"]) is Txt

    per_row = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="ask", type="ask", version=1, bindings={"text": _edge("split")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge("ask")}),
    ])
    first = _leg(per_row)
    second = _leg(per_row, record=first[-1]["record"], cache={"ask": {"result": {"rows": [[0], [1]], "values": ["x", "y"]}}})
    assert list(second[-1]["results"]["up"]["result"]) == ["X", "Y"]


def test_a_per_row_answer_that_leaves_a_row_out_of_one_output_is_refused():
    """Two outputs answered for different rows is a half answer, not a skip."""
    from dataclasses import dataclass

    import pytest
    from conductor.execution.ledger import Ledger

    @dataclass(frozen=True)
    class Two:
        a: Annotated[Txt, Result(title="A")]
        b: Annotated[Txt, Result(title="B")]

    class Pair(NodeDefinition):
        id = "pair"
        title = "Pair"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Two:
            return Two(a=text, b=text)

    reg = NodeRegistry()
    for cls in (Split, Pair):
        reg.register(cls)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="p", type="pair", version=1, bindings={"text": _edge("split")}),
    ]), reg)
    ledger = Ledger(compiled)
    ledger.record(("split", None), {"result": [Txt("a"), Txt("b")]})
    index = compiled.field(Ref("split", "result")).index

    with pytest.raises(ValueError, match=r"'p'.*'b'.*\[1\]"):
        ledger.inject("p", {"a": Series(index, [Txt("A"), Txt("B")]), "b": Series(index, [Txt("A")], rows=[(0,)])})


def test_a_unit_whose_second_output_is_invalid_writes_nothing():
    """A unit's outputs are checked before any is written: the ledger holds
    all of them or none, never a half-written row."""
    from dataclasses import dataclass

    from conductor.execution.ledger import Ledger

    @dataclass(frozen=True)
    class Both:
        head: Annotated[Txt, Result(title="Head")]
        parts: Annotated[Series[Txt], Result(title="Parts")]

    class Splits(NodeDefinition):
        id = "splits"
        title = "Splits"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Both:
            return Both(head=Txt("h"), parts=[Txt("p")])

    reg = NodeRegistry()
    reg.register(Splits)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="splits", version=1, bindings={"text": Static(value="x")})]), reg)
    ledger = Ledger(compiled)

    import pytest

    with pytest.raises(ValueError):
        ledger.record(("s", None), {"head": Txt("h"), "parts": 3})  # a series output that is not a sequence

    assert ledger.cells().cells == [] and not ledger.is_done(("s", None))


# -- the review's own cases -------------------------------------------------------------


def test_a_typed_in_list_edited_between_legs_reruns_on_the_new_rows():
    """The rows of a typed-in list are the graph's, never the record's: grown,
    shrunk or added between legs, the node runs on what the author typed now."""
    calls.clear()
    record = _leg(_compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(value=["a", "b"])})]))[-1]["record"]
    calls.clear()

    grown = _leg(_compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(value=["a", "b", "c"])})]), record=_through_json(record))[-1]
    assert grown["type"] == "graph_complete" and list(grown["results"]["e"]["result"]) == ["A", "B", "C"]
    shrunk = _leg(_compiled([GraphNode(id="e", type="upper", version=1, bindings={"text": Static(value=["a"])})]), record=_through_json(record))[-1]
    assert shrunk["type"] == "graph_complete" and list(shrunk["results"]["e"]["result"]) == ["A"]
    added = _leg(_compiled([
        GraphNode(id="e", type="upper", version=1, bindings={"text": Static(value=["a", "b"])}),
        GraphNode(id="f", type="upper", version=1, bindings={"text": Static(value=["x", "y"])}),
    ]), record=_through_json(record))[-1]
    assert added["type"] == "graph_complete" and list(added["results"]["f"]["result"]) == ["X", "Y"]
    assert "upper:a" not in calls[-2:]  # ``e`` was kept on the last leg


def test_a_static_hashes_as_its_type_writes_it_not_as_the_author_spelled_it():
    class Num(DType, float):
        id = "record-test-number"
        title = "Number"

    class Half(NodeDefinition):
        id = "half"
        title = "Half"
        description = "d"
        category = "test"

        def run(self, n: Annotated[Num, Param(title="N", widget=Textarea())] = Num(0)) -> Annotated[Num, Result(title="Out")]:
            return Num(n / 2)

    def fingerprint(value):
        reg = NodeRegistry()
        reg.register(Half)
        return CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="h", type="half", version=1, bindings={"n": Static(value=value)})]), reg).node("h").fingerprint

    assert fingerprint(2) == fingerprint(2.0) == fingerprint(Num(2))
    assert fingerprint(2) != fingerprint(3)


def test_series_outputs_of_two_lengths_fail_the_returning_node_with_invalid_output():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Both:
        a: Annotated[Series[Txt], Result(title="A")]
        b: Annotated[Series[Txt], Result(title="B")]

    class Uneven(NodeDefinition):
        id = "uneven"
        title = "Uneven"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Both:
            return Both(a=[Txt("x")], b=[Txt("y"), Txt("z")])

    reg = NodeRegistry()
    reg.register(Uneven)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="u", type="uneven", version=1, bindings={"text": Static(value="x")})]), reg)

    error = next(e for e in _leg(compiled) if e["type"] == "node_error")

    assert error["cause"].code == "invalid_output" and "differ in length" in error["cause"].details["reason"]


def test_an_answer_of_the_wrong_shape_or_naming_a_row_twice_names_the_node_and_output():
    import pytest
    from conductor.execution.ledger import Ledger

    once = _compiled([GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")})])
    with pytest.raises(ValueError, match=r"'split' was given a value for 'result'"):
        Ledger(once).inject("split", {"result": "not a series"})

    per_row = _compiled([
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="ask", type="ask", version=1, bindings={"text": _edge("split")}),
    ])
    ledger = Ledger(per_row)
    ledger.record(("split", None), {"result": [Txt("a"), Txt("b")]})
    with pytest.raises(ValueError, match=r"'ask': 'result' names a row twice"):
        ledger.inject("ask", {"result": {"rows": [[0], [0]], "values": ["x", "y"]}})
