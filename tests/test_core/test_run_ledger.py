"""The ledger, for the two things a run through the public doors can't show: a skip reaching what hangs off an untaken branch, whatever the execution order hides, and what deciding a group costs."""

from dataclasses import dataclass
from typing import Annotated

from conductor import (
    SKIPPED,
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
    node,
)
from conductor._internal.execution.run_ledger import RunLedger, SkippedAtRow
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
    not an empty series, so a reduction downstream of it does not run."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="decide", type="decide", version=1, bindings={"texts": _edge(("docs", "texts"))}),
        GraphNode(id="full", type="join", version=1, bindings={"texts": _edge(("decide", "full"))}),
        GraphNode(id="empty", type="join", version=1, bindings={"texts": _edge(("decide", "empty"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("decide", "empty"))}),
        GraphNode(id="after", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])
    ledger.record(("docs", ()), {"texts": ["a", "b"], "names": ["x", "y"]})
    ledger.record(("decide", ()), {"full": ["a", "b"], "empty": SKIPPED})

    assert ledger.unit_is_ready(("full", ())) and ledger.unit_is_ready(("empty", ()))
    assert list(ledger.inputs_for(("full", ()))["texts"]) == ["a", "b"]
    assert ledger.inputs_for(("empty", ())) == SkippedAtRow(row=())
    # the branch not taken has an index of its own with no rows, so the node running per row of it stands in once, at the top
    assert ledger.units("up") == [("up", ())]
    assert ledger.inputs_for(("up", ())) == SkippedAtRow(row=())
    assert not ledger.unit_is_ready(("after", ()))
    ledger.record(("up", ()), SkippedAtRow(row=()))
    assert ledger.inputs_for(("after", ())) == SkippedAtRow(row=())
    ledger.record(("empty", ()), SkippedAtRow(row=()))
    ledger.record(("after", ()), SkippedAtRow(row=()))
    results = _results_of(ledger)
    assert "empty" not in results and "up" not in results and "after" not in results
    assert "empty" not in ledger.result_of("decide")  # the branch not taken is absent, never SKIPPED


class _CountedSet(set):
    """A set that counts how often it is asked whether it holds something."""

    asked = 0

    def __contains__(self, item: object) -> bool:
        self.asked += 1
        return super().__contains__(item)


def _groups_asked_with_lines_written_between_splits(paragraphs: int) -> int:
    """How often a document's reduction looks at the made groups while the
    document's paragraphs are split one at a time, with the lines of each
    written before the next is split."""
    ledger = _ledger([
        DOCS,
        GraphNode(id="paras", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("paras", "result"))}),
        GraphNode(id="work", type="upper", version=1, bindings={"text": _edge(("lines", "result"))}),
        GraphNode(id="j", type="join-per-text", version=1, bindings={"text": _edge(("docs", "texts")), "texts": _edge(("work", "result"))}),
    ])
    ledger.record(("docs", ()), {"texts": ["\n".join("p" * paragraphs)], "names": ["x"]})
    ledger.record(("paras", (0,)), {"result": ["p"] * paragraphs})
    ledger._groups_with_all_rows_made = _CountedSet(ledger._groups_with_all_rows_made)
    for paragraph in range(paragraphs):
        ledger.record(("lines", (0, paragraph)), {"result": ["l", "l"]})
        for line in range(2):
            ledger.record(("work", (0, paragraph, line)), {"result": "L"})
    return ledger._groups_with_all_rows_made.asked


def test_a_group_two_levels_above_its_rows_looks_at_each_parent_row_once():
    """A reduction over lines per document is woken by every write under the
    document; deciding that the group is made costs what its size costs, not its square."""
    small, large = (_groups_asked_with_lines_written_between_splits(n) for n in (100, 400))

    assert large <= 6 * small, (small, large)
