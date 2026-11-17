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
from typing import Annotated, Any, Literal, TypeAliasType

import pytest
from conductor import (
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Param,
    Run,
    Series,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import BaseModel, ValidationError

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


def test_eleven_contracts_reviewed_all_together_make_one_call_and_one_file_is_collected_as_a_list_of_one():
    run = ran(TOGETHER, files=FILES)
    assert CALLS == {"read": 11, "overall": 1} and run.outputs == {"result": "overall(11)"}
    assert ran(TOGETHER, files=FILES[:1]).outputs == {"result": "overall(1)"}


def test_both_in_one_graph_review_each_contract_and_then_all_of_them_once():
    run = ran(BOTH, files=FILES)
    assert CALLS == {"read": 11, "review": 11, "overall": 1} and run.outputs == {
        "result": "overall(11)"
    }
    assert ran(BOTH, files=FILES[:1]).outputs == {"result": "overall(1)"}


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
        ("invalid_input_make_a_series", ("inputs", "text"))
    ]
    assert _refused(lambda: graph.run({"text": ["a"]})) == [
        ("invalid_input_make_a_series", ("inputs", "text"))
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
    done = saved.answer({w.id: w.loc[1:] == (1,) for w in saved.waiting}).resume()
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
        ("type_mismatch_one_value", ("e", "overall", "texts"))
    ]  # read runs once per call of the subgraph, so overall would get one text
    assert ran(
        record_of(("e", "per"), inputs={"file": "e.file"}, outputs={"result": "e.result"}, reg=reg),
        reg,
        file="a",
    ).outputs == {"result": "rev(A)"}


# -- a yes or no is never taken as a number, and a limit stays on its value --------------------------------------


def _one(node: str, graph_input: str, reg: NodeRegistry) -> GraphRecord:
    """Builds a graph of one node, ``x``.

    The caller gives its input ``graph_input``, and the graph returns its output ``result``.
    """
    return record_of(
        ("x", node),
        inputs={graph_input: f"x.{graph_input}"},
        outputs={"result": "x.result"},
        reg=reg,
    )


def _checked_registry() -> NodeRegistry:
    local_registry = NodeRegistry()

    @local_registry.node("level", title="Level", description="For a test.")
    def level(n: Literal[1, 2, 3]) -> int:
        return n

    return local_registry


def _refused_as(call: Any) -> list[tuple[str, str]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.ctx["errors"][0]["type"]) for p in refused.value.problems]


def test_a_yes_or_no_is_never_taken_as_a_number_of_a_literal():
    checked = _checked_registry()
    graph = Graph.compile(_one("level", "n", checked), checked)

    assert _refused_as(lambda: graph.run({"n": True})) == [
        ("invalid_input", "bool_is_not_a_number")
    ]
    assert graph.run({"n": 2}).outputs == {"result": 2}


def test_a_yes_or_no_is_never_taken_as_a_number_of_a_union_without_a_yes_or_no():
    local_registry = NodeRegistry()

    @local_registry.node("either", title="Either", description="For a test.")
    def either(n: int | float) -> str:
        return repr(n)

    @local_registry.node("or-text", title="Or text", description="For a test.")
    def or_text(n: int | str | None = None) -> str:
        return repr(n)

    @local_registry.node("or-flag", title="Or flag", description="For a test.")
    def or_flag(n: int | bool) -> str:
        return repr(n)

    @local_registry.node("returns-yes", title="Returns yes", description="For a test.")
    def returns_yes(n: int) -> Literal[1, 2] | float:
        return True  # type: ignore[return-value]

    for node_id in ("either", "or-text"):
        graph = Graph.compile(_one(node_id, "n", local_registry), local_registry)
        assert _refused_as(lambda: graph.run({"n": True})) == [
            ("invalid_input", "bool_is_not_a_number")
        ]
        assert graph.run({"n": 2}).outputs == {"result": "2"}
    assert Graph.compile(_one("or-flag", "n", local_registry), local_registry).run(
        {"n": True}
    ).outputs == {"result": "True"}
    assert (
        Graph.compile(_one("returns-yes", "n", local_registry), local_registry).run({"n": 1}).status
        == "failed"
    )


class Order(BaseModel):
    count: Annotated[int, Param(title="Count", ge=0)]


def test_a_param_keeps_its_limits_inside_a_hosts_own_model_and_its_schema_says_them():
    with pytest.raises(ValidationError) as refused:
        Order(count=-1)
    assert refused.value.errors()[0]["type"] == "greater_than_equal"
    assert Order.model_json_schema()["properties"]["count"] == {
        "type": "integer",
        "minimum": 0,
        "title": "Count",
    }


Tree = TypeAliasType("Tree", "dict[str, Tree] | list[Tree] | int")


def test_a_list_given_to_a_type_alias_whose_values_may_be_lists_is_one_value():
    local_registry = NodeRegistry()

    @local_registry.node("depth", title="Depth", description="For a test.")
    def depth(tree: Tree) -> str:
        return repr(tree)

    graph = Graph.compile(_one("depth", "tree", local_registry), local_registry)

    assert graph.run({"tree": [1, [2]]}).outputs == {"result": "[1, [2]]"}


def _taking(hint: Any) -> tuple[Graph[None], NodeRegistry]:
    """Compiles a graph of one node that takes a graph input ``value`` of type ``hint`` and returns it.

    Returns:
        The graph, and the registry it was compiled with.
    """

    def take(value: Any) -> Any:
        return value

    take.__annotations__ = {"value": hint, "return": hint}
    definition = node("take", title="Take", description="For a test.")(take)
    local_registry = NodeRegistry()
    local_registry.add(definition)
    return Graph.compile(make_graph(definition), local_registry), local_registry


def _refused_reasons(hint: Any, value: Any) -> list[str]:
    """Runs a graph whose input of type ``hint`` is given ``value``, and returns why it was refused.

    Returns:
        The type of each pydantic error in the ``invalid_input`` problem.
    """
    graph, _ = _taking(hint)
    with pytest.raises(InvalidCallError) as refused:
        graph.run({"value": value})
    (problem,) = refused.value.problems
    assert problem.type == "invalid_input"
    return [error["type"] for error in problem.ctx["errors"]]


def _taken(hint: Any, value: Any) -> Any:
    """Gives ``value`` to a graph input of type ``hint``, and returns the value the node got."""
    graph, _ = _taking(hint)
    return graph.run({"value": value}).outputs["result"]


def test_true_is_refused_for_a_number_alike_in_a_record_a_list_of_records_and_a_table():
    """``True`` is refused for an ``int`` field, wherever the field is.

    The field can be in a ``TypedDict``, a list or dict of them, or a ``Table``. ``True`` is refused the same way when
    it's given to a graph input and when it's a constant in a graph record read from JSON.
    """
    from typing import TypedDict

    from conductor import Table

    class Row(TypedDict):
        n: int
        share: float

    wrong, right = {"n": True, "share": 0.5}, {"n": 1, "share": 0.5}
    for hint, value in (
        (Row, wrong),
        (list[Row], [right, wrong]),
        (dict[str, Row], {"a": wrong}),
        (Table[Row], [right, wrong]),
        (Row, {"n": 1, "share": False}),
    ):
        assert _refused_reasons(hint, value) == ["bool_is_not_a_number"], hint
        _, local_registry = _taking(hint)
        record = {"nodes": {"take": {"node": "take", "inputs": {"value": {"constant": value}}}}}
        (problem,) = Graph.compile(
            GraphRecord.model_validate_json(json.dumps(record)), local_registry
        ).problems
        assert [error["type"] for error in problem.ctx["errors"]] == ["bool_is_not_a_number"], hint
    assert _taken(Row, right) == right and _taken(list[Row], [right]) == [right]


def test_an_enum_of_numbers_is_a_type_and_refuses_true_alone_and_in_a_list():
    from enum import Enum, IntEnum

    @host_type("level", title="Level")
    class Level(IntEnum):
        LOW = 1

    @host_type("share", title="Share")
    class Share(float, Enum):
        HALF = 0.5

    assert (
        _taken(Level, 1) is Level.LOW
        and _taken(list[Level], [1]) == [Level.LOW]
        and _taken(Share, 0.5) is Share.HALF
    )
    for hint, value in ((Level, True), (list[Level], [True])):
        assert _refused_reasons(hint, value) == ["bool_is_not_a_number"], hint
