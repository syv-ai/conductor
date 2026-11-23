"""Tests for one value or many: the declared type decides which a node gets.

A node written for one value never has to care whether there are many. The tests follow a user of an editor. They
upload contracts and appendices, one file or many, to a graph. The graph's author picked either a node that reads
one file or one that reads them all.

The upload gives a ``Series`` of files. So one file and eleven files run the same graph. A list given to an input
that takes one value is refused.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from conductor import (
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Run,
    Series,
    make_graph,
    subgraph,
)

CALLS: Counter[str] = Counter()
registry = NodeRegistry()


@registry.node("upload", title="Upload", description="The files a user uploaded, one or many.")
def upload(files: list[str]) -> Series[str]:
    return files


@registry.node("read", title="Read", description="Reads one file: run once per file of a Series.")
def read(file: str) -> str:
    """Reads one file: run once per file of a Series."""
    CALLS["read"] += 1
    return file.upper()


@registry.node("review", title="Review", description="Reviews one contract.")
def review(text: str) -> str:
    """Reviews one contract."""
    CALLS["review"] += 1
    return f"rev({text})"


@registry.node("overall", title="Overall", description="Reviews every contract in one call.")
def overall(texts: Series[str]) -> str:
    """Reviews every contract in one call."""
    CALLS["overall"] += 1
    return f"overall({len(texts)})"


@registry.node("compare", title="Compare", description="For a test.")
def compare(contract: str, appendix: str) -> str:
    CALLS["compare"] += 1
    return f"{contract}~{appendix}"


@registry.node("with_appendices", title="With appendices", description="For a test.")
def with_appendices(contract: str, appendices: Series[str]) -> str:
    CALLS["with_appendices"] += 1
    return f"{contract}+{len(appendices)}"


@registry.node("tags", title="Tags", description="For a test.")
def tags(tags: list[str]) -> int:
    return len(tags)


@registry.node("anything", title="Anything", description="For a test.")
def anything(value: Any) -> str:
    return type(value).__name__


@registry.node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",")


@registry.node("approve", title="Approve", description="For a test.", asks=True)
def approve(text: str) -> bool:
    raise Ask(f"approve {text}?")


def record_of(
    *nodes: tuple[str, str] | tuple[str, str, dict[str, Any]],
    edges: tuple[tuple[str, str], ...] = (),
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    reg: NodeRegistry = registry,
) -> GraphRecord:
    """Builds a graph record with ``make_graph``.

    Args:
        nodes: Each node as ``(id, node)`` or ``(id, node, constants)``. ``node`` is a definition id in ``reg``.
        edges: Each edge as ``(source, target)``, such as ``("read.result", "review.text")``.
        inputs: The graph's inputs, each a name mapped to the input it feeds, such as ``"read.file"``.
        outputs: The graph's outputs, each a name mapped to the output it returns, such as ``"review.result"``.
        reg: The registry the definitions come from.
    """
    return make_graph(
        *((n[0], reg[n[1]]) for n in nodes),
        edges=list(edges),
        constants={
            f"{n[0]}.{name}": value
            for n in nodes
            for name, value in (n[2] if len(n) == 3 else {}).items()
        },
        inputs=inputs or {},
        outputs=outputs or {},
    )


def ran(
    record: GraphRecord, reg: NodeRegistry = registry, max_items: int | None = None, **inputs: Any
) -> Any:
    """Compiles a graph record, checks it has no problems, and runs it.

    ``CALLS`` is cleared first, so it counts only this run's calls.
    """
    CALLS.clear()
    graph = Graph.compile(record, reg)
    assert graph.problems == [], graph.problems
    return graph.run(inputs, max_items=max_items)


FILES = [f"k{i}" for i in range(11)]
UPLOAD = ("upload.result", "read.file")
FILES_IN = {"files": "upload.files"}
PER = record_of(
    ("upload", "upload"),
    ("read", "read"),
    ("review", "review"),
    edges=(UPLOAD, ("read.result", "review.text")),
    inputs=FILES_IN,
    outputs={"result": "review.result"},
)
TOGETHER = record_of(
    ("upload", "upload"),
    ("read", "read"),
    ("overall", "overall"),
    edges=(UPLOAD, ("read.result", "overall.texts")),
    inputs=FILES_IN,
    outputs={"result": "overall.result"},
)
BOTH = record_of(
    ("upload", "upload"),
    ("read", "read"),
    ("review", "review"),
    ("overall", "overall"),
    edges=(UPLOAD, ("read.result", "review.text"), ("review.result", "overall.texts")),
    inputs=FILES_IN,
    outputs={"result": "overall.result"},
)


# -- the scenario weighted highest: 11 contracts, one review each, all in one, or both ----------------------------


def test_eleven_contracts_reviewed_one_by_one_make_eleven_reviews_and_one_file_makes_one_in_the_same_shape():
    run = ran(PER, files=FILES)
    assert CALLS == {"read": 11, "review": 11}
    assert run.outputs == {"result": [f"rev(K{i})" for i in range(11)]}
    assert run.graph.describe().nodes["review"].per_item_of == "upload.result"

    run = ran(PER, files=FILES[:1])
    assert CALLS == {"read": 1, "review": 1} and run.outputs == {
        "result": ["rev(K0)"]
    }  # still a list, of one
    assert run.graph.describe().nodes["review"].per_item_of == "upload.result"
    assert ran(PER, files=[]).outputs == {"result": []} and CALLS == {}


@pytest.mark.parametrize(
    ("record", "calls"),
    [
        (TOGETHER, {"read": 11, "overall": 1}),
        (BOTH, {"read": 11, "review": 11, "overall": 1}),
    ],
    ids=[
        "all-together-make-one-call",
        "both-in-one-graph-review-each-contract-and-then-all-of-them-once",
    ],
)
def test_eleven_contracts_reviewed_together_or_each_and_then_together_make_one_overall_call_and_one_file_is_a_list_of_one(
    record: Any, calls: dict[str, int]
):
    run = ran(record, files=FILES)
    assert CALLS == calls and run.outputs == {"result": "overall(11)"}
    assert ran(record, files=FILES[:1]).outputs == {"result": "overall(1)"}


def test_a_compiled_graph_stays_the_same_graph_however_many_files_a_run_is_given():
    graph = Graph.compile(PER, registry)
    fingerprint = graph.fingerprint
    for files in ([], ["a"], ["a", "b"]):
        run = graph.run(json.loads(json.dumps({"files": files})))
        assert run.graph is graph and run.outputs == {
            "result": [f"rev({f.upper()})" for f in files]
        }
    assert graph.fingerprint == fingerprint


# -- two uploads ---------------------------------------------------------------------------------------------------


TWO = record_of(
    ("c", "upload"),
    ("a", "upload"),
    ("cmp", "compare"),
    edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")),
    inputs={"contracts": "c.files", "appendices": "a.files"},
    outputs={"result": "cmp.result"},
)
ONE_CONTRACT = record_of(
    ("c", "read"),
    ("a", "upload"),
    ("cmp", "compare"),
    edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")),
    inputs={"file": "c.file", "files": "a.files"},
    outputs={"result": "cmp.result"},
)
COLLECTED = record_of(
    ("c", "upload"),
    ("a", "upload"),
    ("w", "with_appendices"),
    edges=(("c.result", "w.contract"), ("a.result", "w.appendices")),
    inputs={"contracts": "c.files", "appendices": "a.files"},
    outputs={"result": "w.result"},
)


def test_two_uploads_meeting_at_a_node_are_a_problem_and_one_contract_meets_each_appendix():
    """Conductor never guesses which contract goes with which appendix.

    Two uploads that meet at one node are a problem. The graph's author has to wire which goes with which.
    """
    (problem,) = Graph.compile(TWO, registry).problems
    assert (problem.type, problem.loc, problem.ctx["series"]) == (
        "series_not_nested",
        ("cmp",),
        ["c.result", "a.result"],
    )
    assert ran(ONE_CONTRACT, file="c", files=["a1", "a2"]).outputs == {"result": ["C~a1", "C~a2"]}
    assert CALLS == {"read": 1, "compare": 2}


def test_each_contract_is_called_with_every_appendix_collected_whole():
    assert ran(COLLECTED, contracts=["c"], appendices=[]).outputs == {"result": ["c+0"]}
    run = ran(COLLECTED, contracts=["c1", "c2"], appendices=["a1", "a2", "a3"])
    assert run.outputs == {"result": ["c1+3", "c2+3"]}
    assert CALLS == {"with_appendices": 2}


def test_a_series_longer_than_max_items_stops_the_run_before_any_file_is_read():
    run = ran(PER, max_items=10, files=FILES)
    assert run.status == "stopped" and run.cause.type == "item_limit"
    assert run.cause.ctx == {"series": "upload.result", "items": 11, "max_items": 10}
    assert CALLS == {}


# -- a list at an input of one value -------------------------------------------------------------------------------


def _refused(call: Any) -> list[tuple[str, tuple[str, ...]]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.loc) for p in refused.value.problems]


def test_a_list_given_or_set_as_a_constant_at_an_input_of_one_value_is_refused_and_starts_no_loop():
    graph = Graph.compile(
        record_of(("r", "review"), inputs={"text": "r.text"}, outputs={"result": "r.result"}),
        registry,
    )
    assert _refused(lambda: graph.run({"text": ["a", "b"]})) == [
        ("invalid_input_list_for_one_value", ("inputs", "text"))
    ]
    assert _refused(lambda: graph.run({"text": ["a"]})) == [
        ("invalid_input_list_for_one_value", ("inputs", "text"))
    ]
    assert graph.run({"text": "a"}).outputs == {"result": "rev(a)"}

    with_constant = Graph.compile(
        record_of(
            ("cmp", "compare", {"contract": ["c1", "c2"], "appendix": "a"}),
            outputs={"result": "cmp.result"},
        ),
        registry,
    )
    assert [(p.type, p.loc) for p in with_constant.problems] == [
        ("invalid_value", ("cmp", "contract"))
    ]


def test_an_input_whose_type_takes_a_list_takes_it_as_one_value_and_a_series_input_takes_only_a_list():
    assert ran(
        record_of(("t", "tags"), inputs={"tags": "t.tags"}, outputs={"result": "t.result"}),
        tags=["x", "y"],
    ).outputs == {"result": 2}
    assert ran(
        record_of(("v", "anything"), inputs={"value": "v.value"}, outputs={"result": "v.result"}),
        value=["x", "y"],
    ).outputs == {"result": "list"}
    overall_alone = Graph.compile(
        record_of(("o", "overall"), inputs={"texts": "o.texts"}, outputs={"result": "o.result"}),
        registry,
    )
    assert overall_alone.run({"texts": ["one"]}).outputs == {"result": "overall(1)"}
    assert _refused(lambda: overall_alone.run({"texts": "one"})) == [
        ("invalid_input", ("inputs", "texts"))
    ]


def test_a_run_waiting_per_item_is_read_back_onto_the_same_items():
    graph = Graph.compile(
        record_of(
            ("s", "split"),
            ("q", "approve"),
            edges=(("s.result", "q.text"),),
            inputs={"text": "s.text"},
            outputs={"result": "q.result"},
        ),
        registry,
    )
    run = graph.run({"text": "a,b"})
    assert [w.loc[1:] for w in run.waiting] == [(0,), (1,)]
    saved = Run.from_json(run.to_json(), graph)
    assert saved.graph is graph and saved.graph.describe().nodes["q"].per_item_of == "s.result"
    done = saved.with_answers({w.id: w.loc[1:] == (1,) for w in saved.waiting}).resume()
    assert done.outputs == {"result": [False, True]}


# -- a subgraph, per outer item -----------------------------------------------------------------------------


def test_a_subgraph_given_one_file_per_outer_item_runs_per_item_and_one_that_collects_its_one_text_is_refused():
    reg = NodeRegistry()
    reg.add_subregistry(registry)
    per = record_of(
        ("read", "read"),
        ("review", "review"),
        edges=(("read.result", "review.text"),),
        inputs={"file": "read.file"},
        outputs={"result": "review.result"},
    )
    together = record_of(
        ("read", "read"),
        ("overall", "overall"),
        edges=(("read.result", "overall.texts"),),
        inputs={"file": "read.file"},
        outputs={"result": "overall.result"},
    )
    reg.add(
        subgraph("per", title="Per", description="For a test.")(per),
        subgraph("together", title="Together", description="For a test.")(together),
    )
    outer = {
        inner: record_of(
            ("s", "split"),
            ("e", inner),
            edges=(("s.result", "e.file"),),
            inputs={"text": "s.text"},
            outputs={"result": "e.result"},
            reg=reg,
        )
        for inner in ("per", "together")
    }
    assert list(ran(outer["per"], reg, text="x,y,z").outputs.values()) == [
        ["rev(X)", "rev(Y)", "rev(Z)"]
    ]
    assert [(p.type, p.loc) for p in Graph.compile(outer["together"], reg).problems] == [
        ("type_mismatch_series_given_one_value", ("e", "overall", "texts"))
    ]  # read runs once per call of the subgraph, so overall would get one text
    assert ran(
        record_of(("e", "per"), inputs={"file": "e.file"}, outputs={"result": "e.result"}, reg=reg),
        reg,
        file="a",
    ).outputs == {"result": "rev(A)"}
