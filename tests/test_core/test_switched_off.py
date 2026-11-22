"""Tests for a node switched off in a graph record, ``{"node": "shout", "off": true}``.

A node switched off compiles as any node does, and its problems are reported. A run never calls it: each of its calls
is skipped, so what reads only from it is skipped too, as on a branch not taken. A graph input only nodes switched off
take is no graph input, and an output of a node switched off is no graph output.
"""

from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    Ask,
    CallCount,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    Param,
    Series,
    Upgrade,
    node,
    subgraph,
)
from conductor.testing import check_registry

from tests.test_core.nodes import registry_of

CALLED: list[str] = []


@node("off-upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    CALLED.append(f"upper {text}")
    return text.upper()


@node("off-exclaim", title="Exclaim", description="For a test.")
def exclaim(text: str, mark: str = "!") -> str:
    CALLED.append(f"exclaim {text}")
    return text + mark


@node("off-split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split()


@node("off-join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Size:
    long: str = SKIPPED
    short: str = SKIPPED


@node("off-route", title="Route", description="For a test.")
def route(text: str) -> Size:
    return Size(long=text) if len(text) > 3 else Size(short=text)


@node("off-confirm", title="Confirm", description="For a test.", asks=True)
def confirm(text: str) -> str:
    raise Ask(f"Keep {text}?")


REGISTRY = registry_of(upper, exclaim, split, join, route, confirm)


def _edge(node_id: str, output: str = "result") -> dict:
    return {"sources": [{"node": node_id, "output": output}]}


def _chain(*, off: bool) -> Graph:
    """``text -> upper -> exclaim``, with ``upper`` switched off when ``off``. ``upper`` alone takes ``text``."""
    upper_record = {"node": "off-upper", "inputs": {"text": {"graph_input": "text"}}}
    if off:
        upper_record["off"] = True
    return Graph.compile(
        {
            "inputs": {"text": {}, "mark": {"default": "!"}},
            "nodes": {
                "upper": upper_record,
                "exclaim": {
                    "node": "off-exclaim",
                    "inputs": {"text": _edge("upper"), "mark": {"graph_input": "mark"}},
                },
            },
            "outputs": {
                "upper": {"node": "upper", "output": "result"},
                "result": {"node": "exclaim", "output": "result"},
            },
        },
        REGISTRY,
    )


def test_a_record_reads_off_and_saves_it_only_when_true():
    switched = NodeRecord.model_validate({"node": "off-upper", "off": True})
    assert switched.off
    assert switched.model_dump(mode="json") == {"node": "off-upper", "off": True}
    assert NodeRecord(node="off-upper").model_dump(mode="json") == {"node": "off-upper"}


def test_off_is_read_strictly_as_a_yes_or_no():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile({"nodes": {"u": {"node": "off-upper", "off": 1}}}, REGISTRY)
    [problem] = refused.value.problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == ("invalid_record", "bool_type")


def test_a_node_switched_off_is_never_called_and_what_reads_it_is_skipped():
    CALLED.clear()
    graph = _chain(off=True)
    assert graph.is_runnable
    run = graph.run({})
    assert CALLED == []
    assert run.status == "complete"
    assert run.results == {"upper": {"result": None}, "exclaim": {"result": None}}
    assert run.outputs == {"result": None}
    assert run.progress.nodes["upper"] == CallCount(
        finished=1, waiting=0, skipped=1, none_input=0, known=1, total=1
    )
    assert run.progress.nodes["exclaim"].skipped == 1


def test_a_graph_input_only_a_node_switched_off_takes_is_no_graph_input():
    graph = _chain(off=True)
    assert list(graph.inputs) == ["mark"]
    assert list(graph.describe().inputs) == ["mark"]
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs({"text": "a"})
    assert [problem.type for problem in refused.value.problems] == ["unknown_graph_input"]
    assert list(_chain(off=False).inputs) == ["text", "mark"]


def test_an_output_of_a_node_switched_off_is_no_graph_output():
    graph = _chain(off=True)
    assert list(graph.outputs) == ["result"]
    assert list(graph.describe().outputs) == ["result"]
    assert list(_chain(off=False).outputs) == ["result", "upper"]


def test_a_graph_output_downstream_of_a_node_switched_off_may_be_skipped():
    off, on = _chain(off=True).outputs["result"], _chain(off=False).outputs["result"]
    assert (on.required, off.required) == (True, False)
    assert off == on.model_copy(update={"required": False})


def test_a_subgraph_output_downstream_of_a_node_switched_off_inside_may_be_skipped():
    inner = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "upper": {
                    "node": "off-upper",
                    "inputs": {"text": {"graph_input": "text"}},
                    "off": True,
                },
                "exclaim": {"node": "off-exclaim", "inputs": {"text": _edge("upper")}},
            },
            "outputs": {"result": {"node": "exclaim", "output": "result"}},
        }
    )
    part = subgraph("off-part-2", title="Part", description="For a test.")(inner)
    # the subgraph's text feeds only the node switched off inside, so it is no input of the subgraph node
    graph = Graph.compile(
        {
            "nodes": {
                "part": {"node": "off-part-2"},
                "after": {"node": "off-upper", "inputs": {"text": _edge("part")}},
            },
            "outputs": {
                "part": {"node": "part", "output": "result"},
                "after": {"node": "after", "output": "result"},
            },
        },
        registry_of(upper, exclaim, part),
    )
    assert [port.required for port in graph.describe().nodes["part"].outputs] == [False]
    assert (graph.outputs["part"].required, graph.outputs["after"].required) == (False, False)
    assert graph.run({}).outputs == {"part": None, "after": None}


def _part_record(*, off: bool) -> GraphRecord:
    """``upper`` gives the output ``result``, and ``exclaim``, after it, the output ``loud``."""
    exclaim_record = {"node": "off-exclaim", "inputs": {"text": _edge("upper")}}
    if off:
        exclaim_record["off"] = True
    return GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "upper": {"node": "off-upper", "inputs": {"text": {"graph_input": "text"}}},
                "exclaim": exclaim_record,
            },
            "outputs": {
                "result": {"node": "upper", "output": "result"},
                "loud": {"node": "exclaim", "output": "result"},
            },
        }
    )


def test_a_subgraph_version_that_switches_off_the_node_behind_an_output_needs_an_upgrade():
    first = subgraph("off-part-3", title="Part", description="For a test.")(_part_record(off=False))
    with pytest.raises(NodeDeclarationError) as refused:
        subgraph("off-part-3", title="Part", description="For a test.", replaces=first)(
            _part_record(off=True)
        )
    [problem] = refused.value.problems
    assert (problem.type, problem.ctx["gone"]) == ("upgrade_missing", ["loud"])
    second = subgraph(
        "off-part-3",
        title="Part",
        description="For a test.",
        replaces=first,
        upgrade=Upgrade(renamed_outputs={"loud": "result"}),
    )(_part_record(off=True))
    assert second.output_names == ("result",)
    assert check_registry(registry_of(upper, exclaim, second)) == []


def test_a_node_switched_off_is_described_as_any_node():
    off, on = _chain(off=True).describe(), _chain(off=False).describe()
    assert off.nodes["upper"] == on.nodes["upper"]


@dataclass(frozen=True)
class Pair:
    left: str
    right: str


@node("off-pair", title="Pair", description="For a test.")
def pair(text: str) -> Pair:
    return Pair(text, text)


@node("off-count", title="Count", description="For a test.")
def count(text: str, times: int = 1) -> int:
    return len(text) * times


def _problems(nodes: dict) -> list[str]:
    graph = Graph.compile({"nodes": nodes}, registry_of(upper, exclaim, pair, count))
    return [problem.type for problem in graph.problems]


def test_a_node_switched_off_needs_no_value_for_its_inputs():
    graph = Graph.compile({"nodes": {"u": {"node": "off-upper", "off": True}}}, REGISTRY)
    assert graph.problems == []
    assert graph.run({}).status == "complete"
    assert _problems({"u": {"node": "off-upper"}}) == ["input_has_no_value"]


@node("off-cut", title="Cut", description="For a test.")
def cut(at: Annotated[int, Param(constant_only=True)], text: str = "") -> str:
    return text[:at]


def test_an_input_switched_off_that_takes_only_a_constant_still_needs_one():
    graph = Graph.compile({"nodes": {"c": {"node": "off-cut", "off": True}}}, registry_of(cut))
    assert [problem.type for problem in graph.problems] == ["input_has_no_value_constant_only"]


def test_every_other_problem_of_a_node_switched_off_is_still_reported():
    assert _problems({"gone": {"node": "no-such-node", "off": True}}) == ["unknown_node"]
    assert _problems({"u": {"node": "off-upper", "version": 7, "off": True}}) == ["unknown_version"]
    bad_constant = {"node": "off-count", "inputs": {"text": {"constant": "a"}}, "off": True}
    bad_constant["inputs"]["times"] = {"constant": "x"}
    assert _problems({"c": bad_constant}) == ["invalid_value"]
    leaving = {
        "c": {"node": "off-count", "inputs": {"text": {"constant": "a"}}, "off": True},
        "u": {"node": "off-upper", "inputs": {"text": _edge("c")}},
    }
    assert _problems(leaving) == ["type_mismatch"]
    # two outputs of one node into one input are refused, switched off or not
    two = {
        "p": {"node": "off-pair", "inputs": {"text": {"constant": "a"}}},
        "u": {
            "node": "off-upper",
            "inputs": {
                "text": {
                    "sources": [
                        {"node": "p", "output": "left"},
                        {"node": "p", "output": "right"},
                    ]
                }
            },
        },
    }
    assert _problems(two) == ["several_edges_into_one_input"]
    two["p"]["off"] = True
    assert _problems(two) == ["several_edges_into_one_input"]


@dataclass(frozen=True)
class Noted:
    note: str
    short: str = SKIPPED


@node("off-noted", title="Noted", description="For a test.")
def noted(text: str) -> Noted:
    return Noted(text, text)


def test_switching_a_node_off_never_makes_its_branch_and_another_output_exclusive():
    nodes = {
        "n": {"node": "off-noted", "inputs": {"text": {"constant": "a"}}, "off": True},
        "u": {
            "node": "off-upper",
            "inputs": {
                "text": {
                    "sources": [{"node": "n", "output": "note"}, {"node": "n", "output": "short"}]
                }
            },
        },
    }
    graph = Graph.compile({"nodes": nodes}, registry_of(upper, noted))
    assert [problem.type for problem in graph.problems] == ["several_edges_into_one_input"]


def test_a_node_switched_off_in_a_subgraph_used_per_item_makes_what_collects_it_one_that_may_be_skipped():
    inner = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "upper": {
                    "node": "off-upper",
                    "inputs": {"text": {"graph_input": "text"}},
                    "off": True,
                },
                "exclaim": {
                    "node": "off-exclaim",
                    "inputs": {"text": _edge("upper"), "mark": {"graph_input": "text"}},
                },
            },
            "outputs": {"result": {"node": "exclaim", "output": "result"}},
        }
    )
    part = subgraph("off-part-4", title="Part", description="For a test.")(inner)
    graph = Graph.compile(
        {
            "inputs": {"text": {}},
            "nodes": {
                "split": {"node": "off-split", "inputs": {"text": {"graph_input": "text"}}},
                "part": {"node": "off-part-4", "inputs": {"text": _edge("split")}},
                "join": {"node": "off-join", "inputs": {"texts": _edge("part")}},
            },
            "outputs": {"result": {"node": "join", "output": "result"}},
        },
        registry_of(upper, exclaim, split, join, part),
    )
    assert not graph.outputs["result"].required
    assert graph.run({"text": "a b"}).outputs == {"result": None}


def test_an_input_with_several_edges_takes_the_source_that_is_on():
    record = {
        "inputs": {"text": {}},
        "nodes": {
            "route": {"node": "off-route", "inputs": {"text": {"graph_input": "text"}}},
            "loud": {"node": "off-upper", "inputs": {"text": _edge("route", "long")}, "off": True},
            "quiet": {"node": "off-exclaim", "inputs": {"text": _edge("route", "short")}},
            "last": {
                "node": "off-exclaim",
                "inputs": {
                    "text": {
                        "sources": [
                            {"node": "loud", "output": "result"},
                            {"node": "quiet", "output": "result"},
                        ]
                    },
                    "mark": {"constant": "?"},
                },
            },
        },
        "outputs": {"result": {"node": "last", "output": "result"}},
    }
    graph = Graph.compile(record, REGISTRY)
    assert graph.run({"text": "ab"}).outputs == {"result": "ab!?"}
    assert graph.run({"text": "abcd"}).outputs == {"result": None}


def test_a_node_switched_off_inside_a_loop_skips_each_call_and_what_collects_it_which_may_be_skipped():
    CALLED.clear()
    graph = Graph.compile(
        {
            "inputs": {"text": {}},
            "nodes": {
                "split": {"node": "off-split", "inputs": {"text": {"graph_input": "text"}}},
                "upper": {"node": "off-upper", "inputs": {"text": _edge("split")}, "off": True},
                "join": {"node": "off-join", "inputs": {"texts": _edge("upper")}},
            },
            "outputs": {"result": {"node": "join", "output": "result"}},
        },
        REGISTRY,
    )
    assert not graph.outputs["result"].required
    run = graph.run({"text": "a b c"})
    assert CALLED == []
    assert run.status == "complete"
    # no item can have a value, so the list is shown as None, as for a node skipped by a branch outside the loop
    assert run.results["upper"] == {"result": None}
    assert run.progress.nodes["upper"] == CallCount(
        finished=3, waiting=0, skipped=3, none_input=0, known=3, total=3
    )
    assert run.progress.nodes["join"].skipped == 1


def test_a_node_that_asks_switched_off_never_waits():
    graph = Graph.compile(
        {
            "inputs": {"text": {}},
            "nodes": {
                "upper": {"node": "off-upper", "inputs": {"text": {"graph_input": "text"}}},
                "confirm": {
                    "node": "off-confirm",
                    "inputs": {"text": _edge("upper")},
                    "off": True,
                },
            },
            "outputs": {"result": {"node": "upper", "output": "result"}},
        },
        REGISTRY,
    )
    assert not graph.asks
    run = graph.run({"text": "a"})
    assert run.status == "complete"
    assert run.waiting == []
    assert run.outputs == {"result": "A"}


def _loud():
    """The subgraph ``upper -> exclaim``: its graph input ``text`` and its output ``result``."""
    return subgraph("off-loud", title="Loud", description="For a test.")(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "upper": {"node": "off-upper", "inputs": {"text": {"graph_input": "text"}}},
                    "exclaim": {"node": "off-exclaim", "inputs": {"text": _edge("upper")}},
                },
                "outputs": {"result": {"node": "exclaim", "output": "result"}},
            }
        )
    )


def test_a_subgraph_node_switched_off_calls_no_node_inside_it():
    CALLED.clear()
    registry = registry_of(upper, exclaim, _loud())
    graph = Graph.compile(
        {
            "inputs": {"text": {}},
            "nodes": {
                "loud": {
                    "node": "off-loud",
                    "inputs": {"text": {"graph_input": "text"}},
                    "off": True,
                },
                "after": {"node": "off-upper", "inputs": {"text": _edge("loud")}},
            },
            "outputs": {"result": {"node": "after", "output": "result"}},
        },
        registry,
    )
    assert graph.is_runnable
    assert list(graph.inputs) == []
    run = graph.run({})
    assert CALLED == []
    assert run.status == "complete"
    assert run.outputs == {"result": None}
    assert run.progress.nodes["loud"].skipped == 2


def test_a_subgraph_whose_node_is_switched_off_offers_no_port_only_that_node_serves():
    inner = GraphRecord.model_validate(
        {
            "inputs": {"text": {}, "other": {}},
            "nodes": {
                "upper": {"node": "off-upper", "inputs": {"text": {"graph_input": "text"}}},
                "exclaim": {
                    "node": "off-exclaim",
                    "inputs": {"text": {"graph_input": "other"}},
                    "off": True,
                },
            },
            "outputs": {
                "result": {"node": "upper", "output": "result"},
                "loud": {"node": "exclaim", "output": "result"},
            },
        }
    )
    part = subgraph("off-part", title="Part", description="For a test.")(inner)
    assert part.input_names == ("text",)
    assert part.output_names == ("result",)
    graph = Graph.compile(
        {
            "inputs": {"text": {}},
            "nodes": {"part": {"node": "off-part", "inputs": {"text": {"graph_input": "text"}}}},
            "outputs": {"result": {"node": "part", "output": "result"}},
        },
        registry_of(upper, exclaim, part),
    )
    assert graph.is_runnable
    assert [port.name for port in graph.describe().nodes["part"].outputs] == ["result"]
    assert graph.run({"text": "a"}).outputs == {"result": "A"}


def test_switched_back_on_a_node_is_called_fresh():
    CALLED.clear()
    first = _chain(off=False).run({"text": "a"})
    switched_off = _chain(off=True).with_inputs({}, previous=first).resume()
    assert CALLED == ["upper a", "exclaim A"]
    CALLED.clear()
    again = _chain(off=False).with_inputs({"text": "a"}, previous=switched_off).resume()
    assert CALLED == ["upper a", "exclaim A"]
    assert again.outputs == {"result": "A!", "upper": "A"}


def test_a_run_with_a_node_switched_off_saves_and_reads_back():
    graph = _chain(off=True)
    run = graph.run({})
    read = type(run).from_json(run.to_json(), graph)
    assert read.status == "complete"
    assert read.results == run.results
    with pytest.raises(InvalidCallError) as refused:
        type(run).from_json(run.to_json(), _chain(off=False))
    assert refused.value.problems[0].type == "saved_run_of_another_graph"
