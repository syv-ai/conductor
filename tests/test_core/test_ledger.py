"""The ledger: what a run has produced, and what that makes ready."""

from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import NodeExecutionError
from conductor.execution.ledger import Ledger, Skip
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import ConnectionList, Textarea


class Txt(DType, str):
    id = "ledger-test-txt"
    title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Names")]


@dataclass(frozen=True)
class Column:
    full: Annotated[Series[Txt], Result(title="Full")]
    empty: Annotated[Series[Txt], Result(title="Empty")]


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Textarea(title="Folder")] = Txt("")) -> Documents:
        return Documents(texts=(), names=())


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return Txt(text.upper())


class Pair(NodeDefinition):
    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"

    def run(self, a: Annotated[Txt, Textarea(title="A")] = Txt(""), b: Annotated[Txt, Textarea(title="B")] = Txt("")) -> Out:
        return Txt(a + b)


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Gate(NodeDefinition):
    """A decision over a whole column: the column goes out on ``full`` or ``empty``."""

    id = "gate"
    title = "Gate"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Column:
        return Column(full=texts, empty=SKIPPED) if len(texts) else Column(full=SKIPPED, empty=texts)


def _registry():
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Pair, Lines, Join, Gate):
        registry.register(node_cls)
    return registry


def _ledger(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return Ledger(compiled)


def _edge(*refs):
    return Edges(refs=tuple(Ref(n, f) for n, f in refs))


DOCS = GraphNode(id="docs", type="docs", version=1)


# --- units ------------------------------------------------------------------


def test_a_node_that_runs_once_is_one_unit_and_is_ready_with_no_edges():
    ledger = _ledger([DOCS])

    assert ledger.units("docs") == [("docs", None)]
    assert ledger.ready(("docs", None))
    assert not ledger.complete("docs")


def test_an_iterating_node_has_no_units_until_its_index_has_rows():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    assert ledger.units("up") == []


def test_a_root_index_is_born_sealed_by_the_unit_that_produces_it():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})

    assert ledger.complete("docs")
    assert ledger.units("up") == [("up", (0,)), ("up", (1,))]
    assert all(ledger.ready(unit) for unit in ledger.units("up"))


def test_series_outputs_of_one_unit_must_agree_in_length():
    ledger = _ledger([DOCS])

    with pytest.raises(ValueError, match="length"):
        ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x"]})


def test_a_unit_that_skips_every_series_output_gates_its_index():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])

    ledger.record(("docs", None), {"texts": SKIPPED, "names": SKIPPED})

    assert ledger.units("up") == [("up", None)]
    assert ledger.inputs_for(("up", None)) == Skip(at=None)


# --- what a unit receives ----------------------------------------------------------


def test_an_iterating_unit_receives_its_row():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})

    assert ledger.inputs_for(("up", (1,))) == {"text": "b"}


def test_a_scalar_beside_a_series_broadcasts():
    ledger = _ledger([
        DOCS,
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static(value="p")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("prefix", "result")), "b": _edge(("docs", "texts"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    assert not ledger.ready(("p", (0,)))
    ledger.record(("prefix", None), {"result": "P"})

    assert ledger.ready(("p", (0,)))
    assert ledger.inputs_for(("p", (1,))) == {"a": "P", "b": "b"}


def test_contagion_reads_the_upstream_row_and_nothing_waits_on_other_rows():
    """Row 0 of `up` makes row 0 of `n` ready while row 1 is still running."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="n", type="upper", version=1, bindings={"text": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("up", (0,)), {"result": "A"})

    assert ledger.ready(("n", (0,)))
    assert not ledger.ready(("n", (1,)))
    assert ledger.inputs_for(("n", (0,))) == {"text": "A"}


def test_an_iterating_unit_returning_a_series_births_rows_under_its_row():
    ledger = _ledger([DOCS, GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))})])
    ledger.record(("docs", None), {"texts": ["a\nb", "c"], "names": ["x", "y"]})
    ledger.record(("lines", (1,)), {"result": ["c"]})
    ledger.record(("lines", (0,)), {"result": ["a", "b"]})

    assert ledger.complete("lines")
    lines = ledger.results()["lines"]["result"]
    assert lines.index == Index("lines", parent=Index("docs"))
    assert lines.rows == ((0, 0), (0, 1), (1, 0))
    assert list(lines) == ["a", "b", "c"]


def test_a_reduction_on_a_child_receives_the_group_under_its_parent_row():
    ledger = _ledger([
        DOCS,
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("lines", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a\nb", "c"], "names": ["x", "y"]})
    ledger.record(("lines", (0,)), {"result": ["a", "b"]})

    assert ledger.ready(("j", (0,)))
    assert not ledger.ready(("j", (1,)))
    group = ledger.inputs_for(("j", (0,)))["texts"]
    assert isinstance(group, Series)
    assert group.rows == ((0, 0), (0, 1))
    assert list(group) == ["a", "b"]


def test_a_parent_index_series_broadcasts_down_to_a_child_row():
    ledger = _ledger([
        DOCS,
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("lines", "result")), "b": _edge(("docs", "names"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a\nb", "c"], "names": ["x", "y"]})
    ledger.record(("lines", (0,)), {"result": ["a", "b"]})

    assert ledger.inputs_for(("p", (0, 1))) == {"a": "b", "b": "x"}


def test_a_reduction_on_a_root_waits_for_the_whole_series():
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("up", (0,)), {"result": "A"})
    assert not ledger.ready(("j", None))
    ledger.record(("up", (1,)), {"result": "B"})

    assert ledger.ready(("j", None))
    texts = ledger.inputs_for(("j", None))["texts"]
    assert texts.index == Index("docs")
    assert list(texts) == ["A", "B"]


def test_a_gather_lands_on_the_inputs_own_index_and_drops_skipped_sources():
    ledger = _ledger([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static(value="b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])
    ledger.record(("a", None), {"result": "A"})
    ledger.record(("b", None), Skip(at=None))

    texts = ledger.inputs_for(("j", None))["texts"]
    assert texts.index == Index("j.texts")
    assert list(texts) == ["A"]


def test_an_unbound_series_input_receives_its_default_on_its_own_index():
    ledger = _ledger([GraphNode(id="j", type="join", version=1)])

    texts = ledger.inputs_for(("j", None))["texts"]
    assert isinstance(texts, Series)
    assert texts.index == Index("j.texts")
    assert list(texts) == []


def test_a_typed_list_on_a_series_input_lands_on_the_inputs_own_index():
    ledger = _ledger([GraphNode(id="j", type="join", version=1, bindings={"texts": Static(value=["a", "b"])})])

    texts = ledger.inputs_for(("j", None))["texts"]
    assert texts.index == Index("j.texts")
    assert texts.rows == ((0,), (1,))


def test_a_static_list_on_a_scalar_input_births_the_rows_compile_named():
    """Compile stored the index a static list iterates on (`field(ref).index`); the
    ledger births its rows there — before anything runs,
    sealed — and reads each unit's element off it. The ledger tests the
    value with nothing of its own."""
    ledger = _ledger([GraphNode(id="u", type="upper", version=1, bindings={"text": Static(value=["a", "b"])})])

    assert ledger._compiled.field(Ref("u", "text")).index == Index("u.text")
    assert ledger.units("u") == [("u", (0,)), ("u", (1,))]
    assert ledger.progress("u") == (0, 2)
    assert ledger.inputs_for(("u", (1,))) == {"text": "b"}


# --- skip and sparsity ---------------------------------------------------


def test_a_skipped_scalar_input_skips_the_unit():
    ledger = _ledger([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("a", "result"))}),
    ])
    ledger.record(("a", None), Skip(at=None))

    assert ledger.inputs_for(("b", None)) == Skip(at=None)


def test_a_skipped_row_leaves_the_series_sparse():
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="n", type="upper", version=1, bindings={"text": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b", "c"], "names": ["x", "y", "z"]})
    ledger.record(("up", (0,)), {"result": "A"})
    ledger.record(("up", (1,)), Skip(at=(1,)))
    ledger.record(("up", (2,)), {"result": "C"})

    up = ledger.results()["up"]["result"]
    assert up.rows == ((0,), (2,))
    assert list(up) == ["A", "C"]
    assert ledger.inputs_for(("n", (1,))) == Skip(at=(1,))
    assert ledger.inputs_for(("n", (2,))) == {"text": "C"}


def test_a_skipped_node_skips_what_it_births():
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), Skip(at=None))

    assert ledger.units("up") == [("up", None)]
    assert ledger.ready(("up", None))
    assert ledger.inputs_for(("up", None)) == Skip(at=None)
    ledger.record(("up", None), Skip(at=None))
    assert ledger.complete("up")
    assert ledger.inputs_for(("j", None)) == Skip(at=None)


def test_an_iterating_node_with_every_row_skipped_is_an_empty_series_not_skipped():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])
    ledger.record(("docs", None), {"texts": ["a"], "names": ["x"]})
    ledger.record(("up", (0,)), Skip(at=(0,)))

    assert "up" in ledger.results()
    assert list(ledger.results()["up"]["result"]) == []


def test_a_decision_over_a_whole_column_gates_what_hangs_off_the_branch_not_taken():
    """An "if the list is empty" decision: the untaken branch is a gate,
    not an empty series, so a reduction downstream of it does not run."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="gate", type="gate", version=1, bindings={"texts": _edge(("docs", "texts"))}),
        GraphNode(id="full", type="join", version=1, bindings={"texts": _edge(("gate", "full"))}),
        GraphNode(id="empty", type="join", version=1, bindings={"texts": _edge(("gate", "empty"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("gate", "empty"))}),
        GraphNode(id="after", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("gate", None), {"full": ["a", "b"], "empty": SKIPPED})

    assert ledger.ready(("full", None)) and ledger.ready(("empty", None))
    assert list(ledger.inputs_for(("full", None))["texts"]) == ["a", "b"]
    assert ledger.inputs_for(("empty", None)) == Skip(at=None)
    # the gate reaches through a node running per row: its rows exist (born by the full branch), each is gated
    assert ledger.units("up") == [("up", (0,)), ("up", (1,))]
    assert ledger.inputs_for(("up", (0,))) == Skip(at=None)
    assert not ledger.ready(("after", None))
    ledger.record(("up", (0,)), Skip(at=None))
    ledger.record(("up", (1,)), Skip(at=None))
    assert ledger.inputs_for(("after", None)) == Skip(at=None)
    ledger.record(("empty", None), Skip(at=None))
    ledger.record(("after", None), Skip(at=None))
    results = ledger.results()
    assert "empty" not in results and "up" not in results and "after" not in results
    assert ledger.result_of("gate")["empty"] is SKIPPED


def test_a_gate_at_a_parent_row_reaches_only_the_rows_under_it():
    """An iterating node skipping its series output at row 1 gates the child
    index under (1,): the reduction for parent row 1 is skipped, row 0 runs."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a\nb", "c"], "names": ["x", "y"]})
    ledger.record(("lines", (0,)), {"result": ["a", "b"]})
    ledger.record(("lines", (1,)), {"result": SKIPPED})

    assert ledger.units("up") == [("up", (1,)), ("up", (0, 0)), ("up", (0, 1))]
    assert ledger.inputs_for(("up", (1,))) == Skip(at=(1,))
    ledger.record(("up", (1,)), Skip(at=(1,)))
    assert ledger.ready(("j", (1,)))
    assert ledger.inputs_for(("j", (1,))) == Skip(at=(1,))
    ledger.record(("up", (0, 0)), {"result": "A"})
    ledger.record(("up", (0, 1)), {"result": "B"})
    assert list(ledger.inputs_for(("j", (0,)))["texts"]) == ["A", "B"]
    ledger.record(("j", (0,)), {"result": "A+B"})
    ledger.record(("j", (1,)), Skip(at=(1,)))

    joined = ledger.results()["j"]["result"]
    assert joined.rows == ((0,),)
    assert list(joined) == ["A+B"]
    assert ledger.results()["up"]["result"].rows == ((0, 0), (0, 1))


def test_a_mask_and_a_gate_differ_only_in_depth():
    """Every row of `up` masked: `j` runs on an empty series. `up` gated
    from above: `j` does not run."""
    nodes = [
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ]
    masked = _ledger(nodes)
    masked.record(("docs", None), {"texts": ["a"], "names": ["x"]})
    masked.record(("up", (0,)), Skip(at=(0,)))
    assert list(masked.inputs_for(("j", None))["texts"]) == []

    gated = _ledger(nodes)
    gated.record(("docs", None), Skip(at=None))
    gated.record(("up", None), Skip(at=None))
    assert gated.inputs_for(("j", None)) == Skip(at=None)


# --- what the run produced ----------------------------------------------------------------


def test_results_hold_scalars_and_series_on_their_indexes():
    ledger = _ledger([
        DOCS,
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("up", (1,)), {"result": "B"})
    ledger.record(("up", (0,)), {"result": "A"})
    ledger.record(("j", None), {"result": "A+B"})

    results = ledger.results()
    assert results["j"] == {"result": "A+B"}
    assert results["up"]["result"].index == Index("docs")
    assert list(results["up"]["result"]) == ["A", "B"]
    assert results["docs"]["names"].rows == ((0,), (1,))


def test_two_refs_on_one_index_into_a_scalar_input_read_whichever_covers_the_row():
    """Merging is edges. Per row the one source that covers it; none is a mask; two fail the node."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="m", type="upper", version=1, bindings={"text": _edge(("a", "result"), ("b", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["x", "y", "z"], "names": ["1", "2", "3"]})
    ledger.record(("a", (0,)), {"result": "A0"})
    ledger.record(("b", (0,)), Skip(at=(0,)))
    ledger.record(("a", (1,)), Skip(at=(1,)))
    ledger.record(("b", (1,)), {"result": "B1"})
    ledger.record(("a", (2,)), Skip(at=(2,)))
    ledger.record(("b", (2,)), Skip(at=(2,)))

    assert ledger.inputs_for(("m", (0,))) == {"text": "A0"}
    assert ledger.inputs_for(("m", (1,))) == {"text": "B1"}
    assert ledger.inputs_for(("m", (2,))) == Skip(at=(2,))


def test_a_row_two_sources_cover_fails_the_node_with_the_row():
    ledger = _ledger([
        DOCS,
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="m", type="upper", version=1, bindings={"text": _edge(("a", "result"), ("b", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["x"], "names": ["1"]})
    ledger.record(("a", (0,)), {"result": "A0"})
    ledger.record(("b", (0,)), {"result": "B0"})

    with pytest.raises(NodeExecutionError) as failed:
        ledger.inputs_for(("m", (0,)))
    assert failed.value.cause.code == "row_covered_twice" and failed.value.cause.row == (0,)


def test_two_refs_on_one_index_into_a_series_input_are_one_series_on_it():
    ledger = _ledger([
        DOCS,
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["x", "y"], "names": ["1", "2"]})
    ledger.record(("a", (0,)), {"result": "A0"})
    ledger.record(("b", (0,)), Skip(at=(0,)))
    ledger.record(("a", (1,)), Skip(at=(1,)))
    ledger.record(("b", (1,)), {"result": "B1"})

    texts = ledger.inputs_for(("j", None))["texts"]
    assert texts.index == Index("docs") and texts.rows == ((0,), (1,))
    assert list(texts) == ["A0", "B1"]


def test_a_skipped_node_is_absent_from_the_results():
    ledger = _ledger([GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")})])
    ledger.record(("a", None), Skip(at=None))

    assert ledger.results() == {}


def test_progress_counts_rows_and_knows_the_total_once_sealed():
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])
    assert ledger.progress("up") == (0, None)
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("up", (0,)), {"result": "A"})

    assert ledger.progress("up") == (1, 2)


def test_progress_counts_rows_not_cover_units():
    """A cover unit stands in for rows a gate never let be born; "4 of 10"
    counts rows, so covers are not in either number."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
    ])
    ledger.record(("docs", None), {"texts": ["a\nb", "c"], "names": ["x", "y"]})
    ledger.record(("lines", (0,)), {"result": ["a", "b"]})
    ledger.record(("lines", (1,)), {"result": SKIPPED})

    assert ledger.units("up") == [("up", (1,)), ("up", (0, 0)), ("up", (0, 1))]
    ledger.record(("up", (1,)), Skip(at=(1,)))
    ledger.record(("up", (0, 0)), {"result": "A"})

    assert ledger.progress("up") == (1, 2)


def test_the_ledger_restores_from_its_cells():
    """The cells are the record; the next leg starts from them, nothing pruned."""
    ledger = _ledger([DOCS, GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))})])
    ledger.record(("docs", None), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("up", (0,)), Skip(at=(0,)))

    restored = Ledger.restore(ledger._compiled, ledger.cells())

    assert restored.units("up") == ledger.units("up")
    assert restored.is_done(("up", (0,))) and not restored.is_done(("up", (1,)))
    assert restored.inputs_for(("up", (1,))) == {"text": "b"}
    assert restored.results()["docs"]["texts"].rows == ((0,), (1,))


def test_a_pending_unit_waits_and_so_does_what_reads_it():
    """A unit whose node asked a person is neither done nor
    failed; its node is incomplete, its consumers are not ready, and the
    leg's pending set names the question by address."""
    from conductor.metadata import Input
    from conductor.widgets import Textarea

    ledger = _ledger([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("a", "result"))}),
    ])
    ledger.pend(("a", None), (Input(name="result", dtype=Txt, title="Svar", widget=Textarea(title="Svar")),))

    assert ledger.is_pending(("a", None)) and not ledger.is_done(("a", None))
    assert not ledger.complete("a")
    assert not ledger.ready(("b", None))
    (waiting,) = ledger.pending()
    assert (waiting["node_id"], waiting["row"]) == ("a", None)
    assert waiting["questions"][0].name == Ref("a", "result")
    assert ("a", None) not in {(n, None if r is None else tuple(r)) for n, r in Ledger.restore(ledger._compiled, ledger.cells())._done}
