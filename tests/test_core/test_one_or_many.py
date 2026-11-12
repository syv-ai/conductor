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


@registry.node("approve", title="Approve", description="For a test.")
def approve(text: str) -> bool:
    raise Ask(f"approve {text}?")


def drawn(*nodes: tuple[str, str] | dict[str, Any], edges: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
    """A record: each node ``(id, node)`` or a node record of its own, and edges ``(from, to)``."""
    return {
        "nodes": [n if isinstance(n, dict) else {"id": n[0], "node": n[1]} for n in nodes],
        "edges": [list(e) for e in edges],
    }


def ran(record: dict[str, Any], reg: NodeRegistry = registry, max_items: int | None = None, **inputs: Any) -> Any:
    """The run of a record that loads clean, its calls counted afresh in ``CALLS``."""
    CALLS.clear()
    graph = Graph.compile(GraphRecord.model_validate(record), reg)
    assert graph.problems == [], graph.problems
    return graph.run(inputs, max_items=max_items)


FILES = [f"k{i}" for i in range(11)]
UPLOAD = ("upload.result", "read.file")
PER = drawn(("upload", "upload"), ("read", "read"), ("review", "review"), edges=(UPLOAD, ("read.result", "review.text")))
TOGETHER = drawn(("upload", "upload"), ("read", "read"), ("overall", "overall"), edges=(UPLOAD, ("read.result", "overall.texts")))
BOTH = drawn(("upload", "upload"), ("read", "read"), ("review", "review"), ("overall", "overall"),
             edges=(UPLOAD, ("read.result", "review.text"), ("review.result", "overall.texts")))


# -- the scenario weighted highest: 11 contracts, one review each, all in one, or both ----------------------------


def test_eleven_contracts_reviewed_one_by_one_make_eleven_reviews_and_one_file_makes_one_in_the_same_shape():
    run = ran(PER, files=FILES)
    assert CALLS == {"read": 11, "review": 11}
    assert run.outputs == {"review.result": [f"rev(K{i})" for i in range(11)]}
    assert run.graph.runs_on["review"].series == "upload.result"

    run = ran(PER, files=FILES[:1])
    assert CALLS == {"read": 1, "review": 1} and run.outputs == {"review.result": ["rev(K0)"]}  # a Series of one
    assert run.graph.runs_on["review"].series == "upload.result"
    assert ran(PER, files=[]).outputs == {"review.result": []} and CALLS == {}


def test_eleven_contracts_reviewed_all_together_make_one_call_and_one_file_gathers_as_a_series_of_one():
    run = ran(TOGETHER, files=FILES)
    assert CALLS == {"read": 11, "overall": 1} and run.outputs == {"overall.result": "overall(11)"}
    assert ran(TOGETHER, files=FILES[:1]).outputs == {"overall.result": "overall(1)"}


def test_both_in_one_graph_review_each_contract_and_then_all_of_them_once():
    run = ran(BOTH, files=FILES)
    assert CALLS == {"read": 11, "review": 11, "overall": 1} and run.outputs == {"overall.result": "overall(11)"}
    assert ran(BOTH, files=FILES[:1]).outputs == {"overall.result": "overall(1)"}


def test_a_graph_is_the_one_graph_that_was_loaded_however_many_files_a_run_is_given():
    graph = Graph.compile(GraphRecord.model_validate(PER), registry)
    fingerprint = graph.fingerprint
    for files in ([], ["a"], ["a", "b"]):
        run = graph.run(json.loads(json.dumps({"files": files})))
        assert run.graph is graph and run.outputs == {"review.result": [f"rev({f.upper()})" for f in files]}
    assert graph.fingerprint == fingerprint


# -- two uploads ---------------------------------------------------------------------------------------------------


TWO = drawn(("c", "upload"), ("a", "upload"), ("cmp", "compare"), edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")))
ONE_CONTRACT = drawn(("c", "read"), ("a", "upload"), ("cmp", "compare"), edges=(("c.result", "cmp.contract"), ("a.result", "cmp.appendix")))
GATHERED = drawn(("c", "upload"), ("a", "upload"), ("w", "with_appendices"), edges=(("c.result", "w.contract"), ("a.result", "w.appendices")))


def test_two_uploads_meeting_at_a_node_are_refused_at_load_and_one_contract_is_there_beside_each_appendix():
    """Which contract goes with which appendix is the author's to draw, never guessed."""
    (problem,) = Graph.compile(GraphRecord.model_validate(TWO), registry).problems
    assert (problem.type, problem.loc, problem.ctx["series"]) == ("series_not_nested", ("cmp",), ["c.result", "a.result"])
    assert ran(ONE_CONTRACT, file="c", files=["a1", "a2"]).outputs == {"cmp.result": ["C~a1", "C~a2"]}
    assert CALLS == {"read": 1, "compare": 2}


def test_each_contract_with_all_the_appendices_gathers_the_appendices_whole_beside_each_contract():
    assert ran(GATHERED, **{"c.files": ["c"], "a.files": []}).outputs == {"w.result": ["c+0"]}
    run = ran(GATHERED, **{"c.files": ["c1", "c2"], "a.files": ["a1", "a2", "a3"]})
    assert run.outputs == {"w.result": ["c1+3", "c2+3"]}
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


def test_a_list_given_or_typed_at_an_input_of_one_value_is_refused_and_makes_no_items():
    graph = Graph.compile(GraphRecord.model_validate(drawn(("r", "review"))), registry)
    assert _refused(lambda: graph.run(text=["a", "b"])) == [("invalid_input", ("inputs", "text"))]
    assert _refused(lambda: graph.run(text=["a"])) == [("invalid_input", ("inputs", "text"))]
    assert graph.run(text="a").outputs == {"r.result": "rev(a)"}

    typed = Graph.compile(GraphRecord.model_validate(drawn({"id": "cmp", "node": "compare", "constants": {"contract": ["c1", "c2"], "appendix": "a"}})), registry)
    assert [(p.type, p.loc) for p in typed.problems] == [("invalid_value", ("cmp", "contract"))]


def test_an_input_whose_type_takes_a_list_takes_it_as_one_value_and_a_series_input_takes_only_a_list():
    assert ran(drawn(("t", "tags")), tags=["x", "y"]).outputs == {"t.result": 2}
    assert ran(drawn(("v", "anything")), value=["x", "y"]).outputs == {"v.result": "list"}
    overall_alone = Graph.compile(GraphRecord.model_validate(drawn(("o", "overall"))), registry)
    assert overall_alone.run(texts=["one"]).outputs == {"o.result": "overall(1)"}
    assert _refused(lambda: overall_alone.run(texts="one")) == [("invalid_input", ("inputs", "texts"))]


def test_a_run_that_waits_per_item_saves_and_loads_back_onto_the_same_items():
    graph = Graph.compile(GraphRecord.model_validate(drawn(("s", "split"), ("q", "approve"), edges=(("s.result", "q.text"),))), registry)
    run = graph.run(text="a,b")
    assert [w.loc[1:] for w in run.waiting] == [(0,), (1,)]
    saved = Run.from_json(run.to_json(), graph)
    assert saved.graph is graph and saved.graph.runs_on["q"].mode == "per_item"
    done = saved.answer({w.id: w.loc[1:] == (1,) for w in saved.waiting}).resume()
    assert done.outputs == {"q.result": [False, True]}


# -- a subgraph, per outer item -----------------------------------------------------------------------------


def test_an_embedded_graph_fed_one_file_per_outer_item_runs_per_item_whether_it_reviews_each_or_all():
    reg = NodeRegistry()
    reg.add_subregistry(registry)
    per = drawn(("read", "read"), ("review", "review"), edges=(("read.result", "review.text"),))
    together = drawn(("read", "read"), ("overall", "overall"), edges=(("read.result", "overall.texts"),))
    reg.add(subgraph("per", title="Per", description="For a test.")(GraphRecord.model_validate(per)), subgraph("together", title="Together", description="For a test.")(GraphRecord.model_validate(together)))
    for inner, results in (("per", ["rev(X)", "rev(Y)", "rev(Z)"]), ("together", ["overall(1)", "overall(1)", "overall(1)"])):
        outer = drawn(("s", "split"), ("e", inner), edges=(("s.result", "e.file"),))
        assert list(ran(outer, reg, text="x,y,z").outputs.values()) == [results]
    assert ran(drawn(("e", "per")), reg, file="a").outputs == {"e.review.result": "rev(A)"}


# -- what a yes or no, and a limit, land on ------------------------------------------------------------------------


def _checked_registry() -> NodeRegistry:
    held = NodeRegistry()

    @held.node("level", title="Level", description="For a test.")
    def level(n: Literal[1, 2, 3]) -> int:
        return n

    return held


def _refused_as(call: Any) -> list[tuple[str, str]]:
    with pytest.raises(InvalidCallError) as refused:
        call()
    return [(p.type, p.ctx["errors"][0]["type"]) for p in refused.value.errors()]


def test_a_yes_or_no_never_lands_on_a_literal_of_numbers():
    graph = Graph.compile(GraphRecord(nodes=[{"id": "x", "node": "level"}]), _checked_registry())

    assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
    assert graph.run(n=2).outputs == {"x.result": 2}


def test_a_yes_or_no_never_lands_on_a_union_a_number_is_in_and_a_yes_or_no_is_not():
    held = NodeRegistry()

    @held.node("either", title="Either", description="For a test.")
    def either(n: int | float) -> str:
        return repr(n)

    @held.node("or-text", title="Or text", description="For a test.")
    def or_text(n: int | str | None = None) -> str:
        return repr(n)

    @held.node("or-flag", title="Or flag", description="For a test.")
    def or_flag(n: int | bool) -> str:
        return repr(n)

    @held.node("returns-yes", title="Returns yes", description="For a test.")
    def returns_yes(n: int) -> Literal[1, 2] | float:
        return True  # type: ignore[return-value]

    for node_id in ("either", "or-text"):
        graph = Graph.compile(GraphRecord(nodes=[{"id": "x", "node": node_id}]), held)
        assert _refused_as(lambda: graph.run(n=True)) == [("invalid_input", "bool_is_not_a_number")]
        assert graph.run(n=2).outputs == {"x.result": "2"}
    assert Graph.compile(GraphRecord(nodes=[{"id": "x", "node": "or-flag"}]), held).run(n=True).outputs == {"x.result": "True"}
    assert Graph.compile(GraphRecord(nodes=[{"id": "x", "node": "returns-yes"}]), held).run(n=1).status == "failed"


class Order(BaseModel):
    count: Annotated[int, Param(title="Count", ge=0)]


def test_a_param_holds_its_limits_inside_a_hosts_own_model_and_its_schema_says_them():
    with pytest.raises(ValidationError) as refused:
        Order(count=-1)
    assert refused.value.errors()[0]["type"] == "greater_than_equal"
    assert Order.model_json_schema()["properties"]["count"] == {"type": "integer", "minimum": 0, "title": "Count"}


Tree = TypeAliasType("Tree", "dict[str, Tree] | list[Tree] | int")


def test_a_list_given_to_a_type_alias_whose_values_may_be_lists_is_one_value():
    held = NodeRegistry()

    @held.node("depth", title="Depth", description="For a test.")
    def depth(tree: Tree) -> str:
        return repr(tree)

    graph = Graph.compile(GraphRecord(nodes=[{"id": "d", "node": "depth"}]), held)

    assert graph.run(tree=[1, [2]]).outputs == {"d.result": "[1, [2]]"}


def test_true_is_refused_for_a_number_alike_in_a_record_a_list_of_records_and_a_table():
    """``True`` for an ``int`` field of a ``TypedDict`` was read as 1, where the same row in a ``Table`` was refused."""
    from typing import TypedDict

    from conductor import Table
    from conductor._internal.value_type import FieldValueError, Type

    class Row(TypedDict):
        n: int
        share: float

    wrong, right = {"n": True, "share": 0.5}, {"n": 1, "share": 0.5}
    for hint, value in ((Row, wrong), (list[Row], [right, wrong]), (dict[str, Row], {"a": wrong}), (Table[Row], [right, wrong]), (Row, {"n": 1, "share": False})):
        with pytest.raises(FieldValueError) as refused:
            Type(hint).read(value)
        assert refused.value.errors[0]["type"] == "bool_is_not_a_number", hint
    assert Type(Row).read(right) == right and Type(list[Row]).read([right]) == [right]
    assert Type(Row).read_json(right) == right
    with pytest.raises(FieldValueError, match="valid integer"):  # from JSON, strictly, the refusal is pydantic's own
        Type(Row).read_json(wrong)


def test_an_enum_of_numbers_is_a_type_and_refuses_true_alone_and_in_a_list():
    from enum import Enum, IntEnum

    from conductor._internal.value_type import FieldValueError, Type

    class Level(IntEnum):
        LOW = 1

    class Share(float, Enum):
        HALF = 0.5

    assert Type(Level).read(1) is Level.LOW and Type(list[Level]).read([1]) == [Level.LOW] and Type(Share).read(0.5) is Share.HALF
    for hint, value in ((Level, True), (list[Level], [True])):
        with pytest.raises(FieldValueError) as refused:
            Type(hint).read(value)
        assert refused.value.errors[0]["type"] == "bool_is_not_a_number"
