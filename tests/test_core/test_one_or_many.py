"""One value or many: the declared type decides, and a node written for one value never sees the difference.

The scenarios are an editor's: a user uploads contracts and appendices, one or many, to a graph whose author picked
a node that reads one file or one that reads them all. The upload gives a ``Series`` of files, so one file and
eleven run the same graph; a list handed to an input of one value is refused.
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
) -> dict[str, Any]:
    """A graph record as data: each node ``(id, node)`` or ``(id, node, constants)``, edges ``(source, target)``, and
    the graph's inputs and outputs, each a name mapped to its place ``"node.name"``."""
    entries: dict[str, dict[str, Any]] = {
        n[0]: {"node": n[1], "inputs": {name: {"constant": value} for name, value in (n[2] if len(n) == 3 else {}).items()}} for n in nodes
    }
    for source, target in edges:
        (source_node, output), (target_node, name) = source.split("."), target.split(".")
        entries[target_node]["inputs"].setdefault(name, {"sources": []})["sources"].append({"node": source_node, "output": output})
    for graph_input, place in (inputs or {}).items():
        node_id, name = place.split(".")
        entries[node_id]["inputs"][name] = {"graph_input": graph_input}
    declared = {name: dict(zip(("node", "output"), place.split("."), strict=True)) for name, place in (outputs or {}).items()}
    return {"nodes": entries, "outputs": declared}


def ran(record: dict[str, Any], reg: NodeRegistry = registry, max_items: int | None = None, **inputs: Any) -> Any:
    """The run of a graph record that compiles with no problems, its calls counted afresh in ``CALLS``."""
    CALLS.clear()
    graph = Graph.compile(GraphRecord.model_validate(record), reg)
    assert graph.problems == [], graph.problems
    return graph.run(inputs, max_items=max_items)


FILES = [f"k{i}" for i in range(11)]
UPLOAD = ("upload.result", "read.file")
FILES_IN = {"files": "upload.files"}
PER = record_of(("upload", "upload"), ("read", "read"), ("review", "review"), edges=(UPLOAD, ("read.result", "review.text")),
                inputs=FILES_IN, outputs={"result": "review.result"})
TOGETHER = record_of(("upload", "upload"), ("read", "read"), ("overall", "overall"), edges=(UPLOAD, ("read.result", "overall.texts")),
                     inputs=FILES_IN, outputs={"result": "overall.result"})
BOTH = record_of(("upload", "upload"), ("read", "read"), ("review", "review"), ("overall", "overall"),
                 edges=(UPLOAD, ("read.result", "review.text"), ("review.result", "overall.texts")), inputs=FILES_IN, outputs={"result": "overall.result"})


# -- the scenario weighted highest: 11 contracts, one review each, all in one, or both ----------------------------


def test_eleven_contracts_reviewed_one_by_one_make_eleven_reviews_and_one_file_makes_one_in_the_same_shape():
    run = ran(PER, files=FILES)
    assert CALLS == {"read": 11, "review": 11}
    assert run.outputs == {"result": [f"rev(K{i})" for i in range(11)]}
    assert run.graph.runs_on["review"].series == "upload.result"

    run = ran(PER, files=FILES[:1])
    assert CALLS == {"read": 1, "review": 1} and run.outputs == {"result": ["rev(K0)"]}  # a list of one
    assert run.graph.runs_on["review"].series == "upload.result"
    assert ran(PER, files=[]).outputs == {"result": []} and CALLS == {}


def test_eleven_contracts_reviewed_all_together_make_one_call_and_one_file_is_collected_as_a_list_of_one():
    run = ran(TOGETHER, files=FILES)
    assert CALLS == {"read": 11, "overall": 1} and run.outputs == {"result": "overall(11)"}
    assert ran(TOGETHER, files=FILES[:1]).outputs == {"result": "overall(1)"}


def test_both_in_one_graph_review_each_contract_and_then_all_of_them_once():
    run = ran(BOTH, files=FILES)
    assert CALLS == {"read": 11, "review": 11, "overall": 1} and run.outputs == {"result": "overall(11)"}
    assert ran(BOTH, files=FILES[:1]).outputs == {"result": "overall(1)"}


def test_a_compiled_graph_stays_the_same_graph_however_many_files_a_run_is_given():
    graph = Graph.compile(GraphRecord.model_validate(PER), registry)
    fingerprint = graph.fingerprint
    for files in ([], ["a"], ["a", "b"]):
        run = graph.run(json.loads(json.dumps({"files": files})))
        assert run.graph is graph and run.outputs == {"result": [f"rev({f.upper()})" for f in files]}
    assert graph.fingerprint == fingerprint


# -- two uploads ---------------------------------------------------------------------------------------------------


TWO = record_of(("c", "upload"), ("a", "upload"), ("cmp", "compare"), edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")),
                inputs={"contracts": "c.files", "appendices": "a.files"}, outputs={"result": "cmp.result"})
ONE_CONTRACT = record_of(("c", "read"), ("a", "upload"), ("cmp", "compare"), edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")),
                         inputs={"file": "c.file", "files": "a.files"}, outputs={"result": "cmp.result"})
COLLECTED = record_of(("c", "upload"), ("a", "upload"), ("w", "with_appendices"), edges=(("c.result", "w.contract"), ("a.result", "w.appendices")),
                      inputs={"contracts": "c.files", "appendices": "a.files"}, outputs={"result": "w.result"})


def test_two_uploads_meeting_at_a_node_are_a_problem_and_one_contract_meets_each_appendix():
    """Which contract goes with which appendix is for the graph's author to wire, never guessed."""
    (problem,) = Graph.compile(GraphRecord.model_validate(TWO), registry).problems
    assert (problem.type, problem.loc, problem.ctx["series"]) == ("series_not_nested", ("cmp",), ["c.result", "a.result"])
    assert ran(ONE_CONTRACT, file="c", files=["a1", "a2"]).outputs == {"result": ["C~a1", "C~a2"]}
    assert CALLS == {"read": 1, "compare": 2}


def test_each_contract_is_called_with_every_appendix_collected_whole():
    assert ran(COLLECTED, contracts=["c"], appendices=[]).outputs == {"result": ["c+0"]}
    run = ran(COLLECTED, contracts=["c1", "c2"], appendices=["a1", "a2", "a3"])
    assert run.outputs == {"result": ["c1+3", "c2+3"]}
    assert CALLS == {"with_appendices": 2}


def test_a_series_longer_than_max_items_stops_the_run_before_any_file_is_read():
    run = ran(PER, max_items=10, files=FILES)
    assert run.status == "stopped" and run.state.cause.type == "item_limit"
    assert run.state.cause.ctx == {"series": "upload.result", "items": 11, "max_items": 10}
    assert CALLS == {}


# -- a list at an input of one value -------------------------------------------------------------------------------


def _refused(call: Any) -> list[tuple[str, tuple[str, ...]]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.loc) for p in refused.value.errors()]


def test_a_list_given_or_set_as_a_constant_at_an_input_of_one_value_is_refused_and_starts_no_loop():
    graph = Graph.compile(GraphRecord.model_validate(record_of(("r", "review"), inputs={"text": "r.text"}, outputs={"result": "r.result"})), registry)
    assert _refused(lambda: graph.run(text=["a", "b"])) == [("invalid_input", ("inputs", "text"))]
    assert _refused(lambda: graph.run(text=["a"])) == [("invalid_input", ("inputs", "text"))]
    assert graph.run(text="a").outputs == {"result": "rev(a)"}

    with_constant = Graph.compile(GraphRecord.model_validate(record_of(("cmp", "compare", {"contract": ["c1", "c2"], "appendix": "a"}), outputs={"result": "cmp.result"})), registry)
    assert [(p.type, p.loc) for p in with_constant.problems] == [("invalid_value", ("cmp", "contract"))]


def test_an_input_whose_type_takes_a_list_takes_it_as_one_value_and_a_series_input_takes_only_a_list():
    assert ran(record_of(("t", "tags"), inputs={"tags": "t.tags"}, outputs={"result": "t.result"}), tags=["x", "y"]).outputs == {"result": 2}
    assert ran(record_of(("v", "anything"), inputs={"value": "v.value"}, outputs={"result": "v.result"}), value=["x", "y"]).outputs == {"result": "list"}
    overall_alone = Graph.compile(GraphRecord.model_validate(record_of(("o", "overall"), inputs={"texts": "o.texts"}, outputs={"result": "o.result"})), registry)
    assert overall_alone.run(texts=["one"]).outputs == {"result": "overall(1)"}
    assert _refused(lambda: overall_alone.run(texts="one")) == [("invalid_input", ("inputs", "texts"))]


def test_a_run_waiting_per_item_is_read_back_onto_the_same_items():
    graph = Graph.compile(GraphRecord.model_validate(record_of(("s", "split"), ("q", "approve"), edges=(("s.result", "q.text"),), inputs={"text": "s.text"}, outputs={"result": "q.result"})), registry)
    run = graph.run(text="a,b")
    assert [w.loc[1:] for w in run.waiting] == [(0,), (1,)]
    saved = Run.from_json(run.to_json(), graph)
    assert saved.graph is graph and saved.graph.runs_on["q"].mode == "per_item"
    done = saved.answer({w.id: w.loc[1:] == (1,) for w in saved.waiting}).resume()
    assert done.outputs == {"result": [False, True]}


# -- a subgraph, per outer item -----------------------------------------------------------------------------


def test_a_subgraph_given_one_file_per_outer_item_runs_per_item_whether_it_reviews_each_or_all():
    reg = NodeRegistry()
    reg.add_subregistry(registry)
    per = record_of(("read", "read"), ("review", "review"), edges=(("read.result", "review.text"),), inputs={"file": "read.file"}, outputs={"result": "review.result"})
    together = record_of(("read", "read"), ("overall", "overall"), edges=(("read.result", "overall.texts"),), inputs={"file": "read.file"}, outputs={"result": "overall.result"})
    reg.add(subgraph("per", title="Per", description="For a test.")(GraphRecord.model_validate(per)), subgraph("together", title="Together", description="For a test.")(GraphRecord.model_validate(together)))
    for inner, results in (("per", ["rev(X)", "rev(Y)", "rev(Z)"]), ("together", ["overall(1)", "overall(1)", "overall(1)"])):
        outer = record_of(("s", "split"), ("e", inner), edges=(("s.result", "e.file"),), inputs={"text": "s.text"}, outputs={"result": "e.result"})
        assert list(ran(outer, reg, text="x,y,z").outputs.values()) == [results]
    assert ran(record_of(("e", "per"), inputs={"file": "e.file"}, outputs={"result": "e.result"}), reg, file="a").outputs == {"result": "rev(A)"}


# -- what a yes or no, and a limit, land on ------------------------------------------------------------------------


def _one(node: str, graph_input: str) -> dict[str, Any]:
    """A graph of one node ``x``, its input ``graph_input`` given by the caller and its output ``result`` returned."""
    return record_of(("x", node), inputs={graph_input: f"x.{graph_input}"}, outputs={"result": "x.result"})


def _checked_registry() -> NodeRegistry:
    local_registry = NodeRegistry()

    @local_registry.node("level", title="Level", description="For a test.")
    def level(n: Literal[1, 2, 3]) -> int:
        return n

    return local_registry


def _refused_as(call: Any) -> list[tuple[str, str]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.ctx["errors"][0]["type"]) for p in refused.value.errors()]


def test_a_yes_or_no_never_lands_on_a_literal_of_numbers():
    graph = Graph.compile(GraphRecord.model_validate(_one("level", "n")), _checked_registry())

    assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
    assert graph.run(n=2).outputs == {"result": 2}


def test_a_yes_or_no_never_lands_on_a_union_a_number_is_in_and_a_yes_or_no_is_not():
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
        graph = Graph.compile(GraphRecord.model_validate(_one(node_id, "n")), local_registry)
        assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
        assert graph.run(n=2).outputs == {"result": "2"}
    assert Graph.compile(GraphRecord.model_validate(_one("or-flag", "n")), local_registry).run(n=True).outputs == {"result": "True"}
    assert Graph.compile(GraphRecord.model_validate(_one("returns-yes", "n")), local_registry).run(n=1).status == "failed"


class Order(BaseModel):
    count: Annotated[int, Param(title="Count", ge=0)]


def test_a_param_keeps_its_limits_inside_a_hosts_own_model_and_its_schema_says_them():
    with pytest.raises(ValidationError) as refused:
        Order(count=-1)
    assert refused.value.errors()[0]["type"] == "greater_than_equal"
    assert Order.model_json_schema()["properties"]["count"] == {"type": "integer", "minimum": 0, "title": "Count"}


Tree = TypeAliasType("Tree", "dict[str, Tree] | list[Tree] | int")


def test_a_list_given_to_a_type_alias_whose_values_may_be_lists_is_one_value():
    local_registry = NodeRegistry()

    @local_registry.node("depth", title="Depth", description="For a test.")
    def depth(tree: Tree) -> str:
        return repr(tree)

    graph = Graph.compile(GraphRecord.model_validate(_one("depth", "tree")), local_registry)

    assert graph.run(tree=[1, [2]]).outputs == {"result": "[1, [2]]"}


def test_true_is_refused_for_a_number_alike_in_a_record_a_list_of_records_and_a_table():
    """``True`` for an ``int`` field is refused alike in a ``TypedDict``, a list or dict of them, and a ``Table``."""
    from typing import TypedDict

    from conductor import Table
    from conductor._internal.gate import validator_of
    from conductor._internal.types import Type

    class Row(TypedDict):
        n: int
        share: float

    wrong, right = {"n": True, "share": 0.5}, {"n": 1, "share": 0.5}
    for hint, value in ((Row, wrong), (list[Row], [right, wrong]), (dict[str, Row], {"a": wrong}), (Table[Row], [right, wrong]), (Row, {"n": 1, "share": False})):
        with pytest.raises(ValidationError) as refused:
            validator_of(Type(hint)).validate_python(value)
        assert refused.value.errors()[0]["type"] == "bool_is_not_a_number", hint
    assert validator_of(Type(Row)).validate_python(right) == right and validator_of(Type(list[Row])).validate_python([right]) == [right]
    assert validator_of(Type(Row)).validate_json(json.dumps(right), strict=True) == right
    with pytest.raises(ValidationError, match="valid integer"):  # from JSON, strictly, the refusal is pydantic's own
        validator_of(Type(Row)).validate_json(json.dumps(wrong), strict=True)


def test_an_enum_of_numbers_is_a_type_and_refuses_true_alone_and_in_a_list():
    from enum import Enum, IntEnum

    from conductor._internal.gate import validator_of
    from conductor._internal.types import Type

    class Level(IntEnum):
        LOW = 1

    class Share(float, Enum):
        HALF = 0.5

    def read(hint: Any, value: Any) -> Any:
        return validator_of(Type(hint)).validate_python(value)

    assert read(Level, 1) is Level.LOW and read(list[Level], [1]) == [Level.LOW] and read(Share, 0.5) is Share.HALF
    for hint, value in ((Level, True), (list[Level], [True])):
        with pytest.raises(ValidationError) as refused:
            read(hint, value)
        assert refused.value.errors()[0]["type"] == "bool_is_not_a_number"
