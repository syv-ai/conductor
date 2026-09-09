"""Lifting, alignment and reduction, derived at compile."""

from dataclasses import dataclass, replace
from typing import Annotated, Any

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType, Single
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Output
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import ConnectionList, Textarea


class Txt(DType, str):
    id = "lifting-test-txt"
    title = "Text"


class Num(DType, float):
    id = "lifting-test-num"
    title = "Number"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    filenames: Annotated[Series[Txt], Result(title="Filenames")]


class Docs(NodeDefinition):
    """A series is born here: one text per document."""

    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Textarea(title="Folder")] = Txt("")) -> Documents:
        return Documents(texts=(), filenames=())


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper case"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return Txt(text.upper())


class Pair(NodeDefinition):
    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"

    def run(
        self,
        a: Annotated[Txt, Textarea(title="A")] = Txt(""),
        b: Annotated[Txt, Textarea(title="B")] = Txt(""),
    ) -> Out:
        return Txt(a + b)


class Join(NodeDefinition):
    """A reduction: declares the whole series."""

    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Out:
        return Txt("\n".join(texts))


class Lines(NodeDefinition):
    """Opens one text into many: lifted, it is an unfold."""

    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


class Count(NodeDefinition):
    id = "count"
    title = "Count"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Annotated[Num, Result(title="Count")]:
        return Num(len(text))


class Route(NodeDefinition):
    """A pass-through: it routes what it does not read, so it declares
    ``Any``, and its hook types the output from what arrives."""

    id = "route"
    title = "Pass on"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Any, ConnectionList(title="Value")]) -> Annotated[Any, Result(title="Pass on")]:
        return value

    def compute_outputs(self, declared, values, arriving):
        dtype = arriving.get("value", Any)
        return tuple(replace(out, dtype=dtype) for out in declared)


class Only(NodeDefinition):
    """A reduction over whatever arrives: `Series[Any] -> Any`."""

    id = "only"
    title = "Only one"
    description = "d"
    category = "test"

    def run(self, values: Annotated[Series[Any], ConnectionList(title="Values")]) -> Annotated[Any, Result(title="The value")]:
        return values[0]

    def compute_outputs(self, declared, values, arriving):
        series = arriving.get("values")
        dtype = series.element if series is not None else Any
        return tuple(replace(out, dtype=dtype) for out in declared)


class Script(NodeDefinition):
    """An open roster: every connected name is a parameter, received whole."""

    id = "script"
    title = "Python"
    description = "d"
    category = "test"

    def run(self, code: Annotated[Txt, Textarea(title="Code", show_handle=False)] = Txt(""), **inputs: Single) -> Out:
        return Txt(",".join(sorted(inputs)))


def _registry():
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Pair, Join, Lines, Count, Route, Only, Script):
        registry.register(node_cls)
    return registry


def _compiled(nodes, **kw):
    return CompiledGraph.from_graph(Graph(nodes=nodes, **kw), _registry())


def _edge(*refs):
    return Edges(refs=tuple(Ref(n, f) for n, f in refs))


# --- a series into a scalar input lifts the node --------------------------------


def test_a_scalar_node_fed_scalars_is_not_lifted():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("a", "result"))}),
    ])

    assert compiled.lifted_on("b") is None
    assert compiled.type_of(Ref("b", "result")) is Txt
    assert compiled.index_of(Ref("b", "result")) is None


def test_a_series_is_born_at_the_node_that_produces_it():
    compiled = _compiled([GraphNode(id="docs", type="docs", version=1)])

    texts_type, texts_index = compiled.type_of(Ref("docs", "texts")), compiled.index_of(Ref("docs", "texts"))
    assert texts_type is Series[Txt]
    assert texts_index == Index("docs")
    assert texts_index.parent is None


def test_every_series_a_node_produces_shares_its_index():
    """One node, one index: its series outputs are columns of one table."""
    compiled = _compiled([GraphNode(id="docs", type="docs", version=1)])

    assert compiled.index_of(Ref("docs", "texts")) == compiled.index_of(Ref("docs", "filenames"))


def test_a_series_into_a_scalar_input_lifts_the_node():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("up") == Index("docs")
    assert compiled.index_of(Ref("up", "text")) == Index("docs")


def test_a_lifted_nodes_outputs_are_series_on_the_same_index():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
    ])

    result_type, result_index = compiled.type_of(Ref("up", "result")), compiled.index_of(Ref("up", "result"))
    assert result_type is Series[Txt]
    assert result_index == Index("docs")


def test_lifting_is_contagious_and_needs_no_construct():
    """Per-row over many nodes: each node downstream receives a series and
    lifts in turn. Nothing is stored, marked or grouped."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="n", type="count", version=1, bindings={"text": _edge(("up", "result"))}),
    ])

    assert compiled.lifted_on("n") == Index("docs")
    assert compiled.type_of(Ref("n", "result")) is Series[Num]


def test_a_scalar_beside_a_series_broadcasts():
    """The same value every row — and no mode on the wire says so."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static(value="Attachment: ")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("prefix", "result")), "b": _edge(("docs", "texts"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("p") == Index("docs")
    assert compiled.index_of(Ref("p", "a")) is None
    assert compiled.index_of(Ref("p", "b")) == Index("docs")


# --- admission: the one type question, asked here ---------------------------------


def test_a_mismatched_edge_is_a_fatal_problem_on_the_target_field():
    compiled = _compiled([
        GraphNode(id="n", type="count", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("n", "result"))}),
    ])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("type_mismatch", True, "up", "text")
    assert problem.details["source"] == "n.result"
    assert set(problem.details) == {"source", "source_type", "target_type", "source_said", "target_said"}
    assert "id" in problem.details["target_type"]  # a description record, not a name
    assert not compiled.is_runnable


def test_a_mismatched_series_is_judged_by_its_element():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="count", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("n", "result"))}),
    ])

    assert [p.code for p in compiled.problems_for()] == ["type_mismatch"]


def test_a_pass_through_binds_its_type_from_the_edge():
    """`Any` in takes what arrived, the hook types the output from
    `arriving`, and the roster is concrete — nothing downstream sees a
    lost type."""
    compiled = _compiled([
        GraphNode(id="n", type="count", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="r", type="route", version=1, bindings={"value": _edge(("n", "result"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("r", "result"))}),
    ])

    assert compiled.interface_of("r").inputs[0].dtype is Num
    assert compiled.interface_of("r").outputs[0].dtype is Num
    assert compiled.type_of(Ref("r", "result")) is Num
    # A node aware of the type it inherits: the next edge is checked against the bound type.
    assert [p.code for p in compiled.problems_for()] == ["type_mismatch"]


def test_a_lifted_pass_through_binds_the_element_and_lifts():
    """A pass-through fed a series is lifted like any scalar node, and
    takes the element as its type."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="r", type="route", version=1, bindings={"value": _edge(("docs", "texts"))}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("r", "result"))}),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.type_of(Ref("r", "result")) is Series[Txt]
    assert compiled.lifted_on("r") == Index("docs")
    assert compiled.lifted_on("up") == Index("docs")


def test_a_reduction_over_the_variable_binds_from_the_series_element():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="one", type="only", version=1, bindings={"values": _edge(("docs", "texts"))}),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.interface_of("one").inputs[0].dtype is Series[Txt]
    assert compiled.type_of(Ref("one", "result")) is compiled.type_of(Ref("one", "result"))
    assert compiled.type_of(Ref("one", "result")) is Txt
    assert compiled.lifted_on("one") is None


def test_an_unconnected_any_input_is_unbound_required_and_the_node_has_no_shape():
    """The honest mid-edit state: fatal, on the field, and nothing
    is guessed about the node. There is no second code for an untyped
    field — unconnected is unconnected."""
    compiled = _compiled([GraphNode(id="r", type="route", version=1)])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("unbound_required", True, "r", "value")
    with pytest.raises(KeyError):
        compiled.lifted_on("r")


def test_a_static_on_an_any_input_is_unbound_too():
    """Only an edge can say what the type is."""
    compiled = _compiled([GraphNode(id="r", type="route", version=1, bindings={"value": Static(value="hi")})])

    assert [p.code for p in compiled.problems_for()] == ["unbound_required"]


def test_a_source_may_refuse_to_be_received_whole_naming_the_fix():
    """A type stated incompletely — a table with no columns — cannot
    be handed whole to a node that will read it by name. The type says so
    (`refuses_whole`), and compile refuses the edge on the field with the
    type's own sentence; routed through an `Any` input the same value passes."""
    from typing import Annotated

    from conductor import NodeRegistry
    from conductor.dtype import DType, Single
    from conductor.node import NodeDefinition
    from conductor.returns import Result

    class Half(DType):
        id = "half"
        title = "Half"

        @classmethod
        def refuses_whole(cls):
            return ("columns_unknown", "The columns are unknown; state them.")

    class Halves(NodeDefinition):
        id = "halves"
        title = "H"
        description = "d"
        category = "test"

        def run(self) -> Annotated[Half, Result(title="Half")]:
            return Half()

    class Reads(NodeDefinition):
        id = "reads"
        title = "R"
        description = "d"
        category = "test"

        def run(self, **inputs: Single) -> Annotated[Txt, Result(title="Text")]:
            return Txt("")

    class Routes(NodeDefinition):
        id = "routes"
        title = "R"
        description = "d"
        category = "test"

        def run(self, value: Annotated[Any, ConnectionList(title="V")]) -> Annotated[Any, Result(title="V")]:
            return value

        def compute_outputs(self, declared, values, arriving):
            dtype = arriving.get("value", Any)
            return tuple(replace(out, dtype=dtype) for out in declared)

    registry = NodeRegistry()
    for node_cls in (Halves, Reads, Routes):
        registry.register(node_cls)
    edge = {"h": GraphNode(id="h", type="halves", version=1)}
    read = CompiledGraph.from_graph(Graph(nodes=[edge["h"], GraphNode(id="r", type="reads", version=1, bindings={"x": Edges(refs=(Ref("h", "result"),))})]), registry)
    routed = CompiledGraph.from_graph(Graph(nodes=[edge["h"], GraphNode(id="r", type="routes", version=1, bindings={"value": Edges(refs=(Ref("h", "result"),))})]), registry)

    (problem,) = read.problems_for(node_id="r")
    assert (problem.code, problem.fatal, problem.field) == ("columns_unknown", True, "x")
    assert "state them" in problem.message
    assert routed.is_runnable and routed.type_of(Ref("r", "result")) is Half


def test_an_open_roster_takes_one_input_per_edge_received_whole():
    """Two edges, two parameters, each typed by what arrives — a
    series arrives whole and lifts nothing."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="count", version=1, bindings={"text": Static(value="hi")}),
        GraphNode(id="s", type="script", version=1, bindings={
            "code": Static(value="return {}"),
            "antal": _edge(("n", "result")),
            "tekster": _edge(("docs", "texts")),
        }),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    roster = compiled.interface_of("s")
    assert [(i.name, i.dtype) for i in roster.inputs] == [("code", Txt), ("antal", Num), ("tekster", Series[Txt])]
    assert type(roster.inputs[2].widget).__name__ == "ConnectionList"
    assert compiled.lifted_on("s") is None
    assert compiled.index_of(Ref("s", "tekster")) == Index("docs")


def test_an_open_roster_parameter_takes_one_edge():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static(value="b")}),
        GraphNode(id="s", type="script", version=1, bindings={"code": Static(value=""), "x": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert [p.code for p in compiled.problems_for()] == ["one_edge_per_parameter"]


def test_an_open_roster_parameter_is_named_like_a_keyword_argument():
    """`**inputs` is Python's own spelling: a parameter's name is an
    identifier, or the node has no signature to receive it in."""
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="s", type="script", version=1, bindings={"code": Static(value=""), "my value": _edge(("a", "result"))}),
    ])

    assert [(p.code, p.field) for p in compiled.problems_for()] == [("parameter_name_invalid", "my value")]


def test_a_node_with_a_broken_edge_has_no_shape_and_says_so_once():
    compiled = _compiled([
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("ghost", "result"))}),
        GraphNode(id="c", type="upper", version=1, bindings={"text": _edge(("b", "result"))}),
    ])

    assert [p.code for p in compiled.problems_for()] == ["unknown_ref_node"]
    with pytest.raises(KeyError):
        compiled.lifted_on("b")
    with pytest.raises(KeyError):
        compiled.type_of(Ref("c", "result"))


def test_a_node_with_no_outputs_yet_is_the_ordinary_mid_edit_state():
    """Outputs born of what the author has not yet given are not a fault:
    non-fatal, the flow still runs, the node is still derived — there is
    simply nothing to edge from it yet."""

    class Sheet(NodeDefinition):
        id = "sheet"
        title = "Sheet"
        description = "d"
        category = "test"

        def run(self, header: Annotated[Txt, Textarea(title="Header")] = Txt("")) -> Out:
            return header

        def compute_outputs(self, declared, values, arriving):
            return tuple(
                Output(name=col, dtype=Txt, title=col)
                for col in str(values.get("header", "")).split(",")
                if col
            )

    registry = _registry()
    registry.register(Sheet)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="sheet", version=1)]), registry)

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id) == ("no_outputs", False, "s")
    assert compiled.is_runnable
    assert compiled.lifted_on("s") is None
    assert compiled.interface_of("s").outputs == ()


# --- alignment is lineage -----------------------------------------------------


def test_compute_outputs_sees_the_dtype_each_connected_input_receives():
    """An opener's columns come from what arrives, per unit — the element of a series."""

    class Opener(NodeDefinition):
        id = "opener"
        title = "Opener"
        description = "d"
        category = "test"

        def run(self, value: Annotated[Txt, Textarea(title="Value")] = Txt("")) -> Out:
            return value

        def compute_outputs(self, declared, values, arriving):
            if "value" not in arriving:
                return declared
            return (Output(name=f"from_{arriving['value'].id}", dtype=Txt, title="Opened"),)

    registry = _registry()
    registry.register(Opener)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="o", type="opener", version=1, bindings={"value": _edge(("docs", "texts"))}),
        GraphNode(id="bare", type="opener", version=1),
    ]), registry)

    assert compiled.is_runnable, compiled.problems_for()
    assert [o.name for o in compiled.interface_of("o").outputs] == ["from_lifting-test-txt"]
    assert compiled.index_of(Ref("o", "from_lifting-test-txt")) == Index("docs")
    assert [o.name for o in compiled.interface_of("bare").outputs] == ["result"]


def test_a_hook_that_cannot_answer_refuses_and_the_refusal_is_the_placements_problem():
    """`Refuses(code, message)` from a roster hook — asked once, with real
    arrivals — lands as the node's one fatal `Problem`: the host
    names the code and writes the sentence, compile only anchors it, and
    there is no `no_outputs` echo beside it."""
    from conductor.node import Refuses

    class Fussy(NodeDefinition):
        id = "fussy"
        title = "Fussy"
        description = "d"
        category = "test"

        def run(self, value: Annotated[Txt, Textarea(title="Value")] = Txt("")) -> Out:
            return value

        def compute_outputs(self, declared, values, arriving):
            raise Refuses("wrong_shape", "What arrives does not fit.")

    registry = _registry()
    registry.register(Fussy)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="f", type="fussy", version=1, bindings={"value": _edge(("docs", "texts"))}),
    ]), registry)

    (problem,) = compiled.problems_for(node_id="f")
    assert (problem.code, problem.fatal, problem.node_id) == ("wrong_shape", True, "f")
    assert problem.message == "What arrives does not fit."
    assert compiled.interface_of("f").outputs == ()


def test_a_hook_reads_a_defaulted_static_nothing_bound():
    """The edges pass hands `compute_outputs` the declaration's defaults
    under the typed statics, so a hook indexes `values[...]` with no
    guard and an unbound defaulted field means what the declaration
    states."""

    class Suffixer(NodeDefinition):
        id = "suffixer"
        title = "Suffixes"
        description = "d"
        category = "test"

        def run(self, suffix: Annotated[Txt, Textarea(title="Suffix")] = Txt("out")) -> Out:
            return suffix

        def compute_outputs(self, declared, values, arriving):
            return (Output(name=str(values["suffix"]), dtype=Txt, title="Out"),)

    registry = _registry()
    registry.register(Suffixer)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="suffixer", version=1)]), registry)

    assert compiled.is_runnable, compiled.problems_for()
    assert [o.name for o in compiled.interface_of("s").outputs] == ["out"]



def test_two_series_on_one_index_align():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("up", "result")), "b": _edge(("docs", "filenames"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("p") == Index("docs")


def test_two_series_on_unrelated_indexes_are_a_fatal_problem_naming_both():
    """Same length is never the test: compile cannot know lengths."""
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("a", "texts")), "b": _edge(("b", "texts"))}),
    ])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("misaligned", True, "p", None)
    assert "p.a" in problem.message and "p.b" in problem.message
    assert problem.details == {"a": "p.a", "b": "p.b"}
    with pytest.raises(KeyError):
        compiled.lifted_on("p")


# --- reduction ---------------------------------------------------------------------


def test_a_reduction_on_a_root_yields_a_scalar_and_is_not_lifted():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("up", "result"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("j") is None
    assert compiled.index_of(Ref("j", "texts")) == Index("docs")
    assert compiled.type_of(Ref("j", "result")) is compiled.type_of(Ref("j", "result"))
    assert compiled.type_of(Ref("j", "result")) is Txt
    assert compiled.index_of(Ref("j", "result")) is None


def test_a_lifted_node_returning_a_series_gives_birth_to_a_child_index():
    """An unfold with no special node: one text in, many lines out, each
    line knowing which document it came from."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
    ])

    lines_type, lines_index = compiled.type_of(Ref("lines", "result")), compiled.index_of(Ref("lines", "result"))
    assert compiled.lifted_on("lines") == Index("docs")
    assert lines_type is Series[Txt]
    assert lines_index == Index("lines")
    assert lines_index.parent == Index("docs")


def test_a_reduction_on_a_child_is_a_lift_on_the_parent():
    """One per document again: the index supplies the grouping, and there
    is no key column to pick."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("lines", "result"))}),
    ])

    assert compiled.lifted_on("j") == Index("docs")
    assert compiled.type_of(Ref("j", "result")) is Series[Txt]
    assert compiled.index_of(Ref("j", "result")) == Index("docs")


def test_a_parent_index_series_broadcasts_down_to_a_child_index_node():
    """The firma is the same for each of its employees."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("lines", "result")), "b": _edge(("docs", "filenames"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("p") == Index("lines")


def test_two_children_of_one_parent_do_not_align():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="l1", type="lines", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="l2", type="lines", version=1, bindings={"text": _edge(("docs", "filenames"))}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": _edge(("l1", "result")), "b": _edge(("l2", "result"))}),
    ])

    assert [p.code for p in compiled.problems_for()] == ["misaligned"]


# --- gather ---------------------------------------------------------------------------


def test_n_scalar_refs_into_a_series_input_gather_onto_a_fresh_index():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static(value="b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])

    texts_type, texts_index = compiled.type_of(Ref("j", "texts")), compiled.index_of(Ref("j", "texts"))
    assert compiled.lifted_on("j") is None
    assert texts_type is Series[Txt]
    assert texts_index == Index("j.texts")
    assert texts_index.parent is None


def test_n_series_refs_concatenate_and_lineage_is_gone():
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "texts"), ("b", "texts"))}),
    ])

    assert compiled.is_runnable
    assert compiled.lifted_on("j") is None
    assert compiled.index_of(Ref("j", "texts")) == Index("j.texts")


def test_two_refs_on_one_index_into_a_scalar_input_are_a_union_on_it():
    """A merge is edges. The node lifts on the shared index and reads per row."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="m", type="upper", version=1, bindings={"text": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.lifted_on("m") == Index("docs")
    assert (compiled.type_of(Ref("m", "result")), compiled.index_of(Ref("m", "result"))) == (Series[Txt], Index("docs"))


def test_two_refs_on_different_indexes_into_a_scalar_input_is_fatal():
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="m", type="upper", version=1, bindings={"text": _edge(("a", "texts"), ("b", "texts"))}),
    ])

    (problem,) = compiled.problems_for(node_id="m")
    assert (problem.code, problem.field) == ("union_needs_one_index", "text")


def test_two_series_refs_on_one_index_into_a_series_input_read_that_index_not_a_pile():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert compiled.lifted_on("j") is None
    assert compiled.index_of(Ref("j", "texts")) == Index("docs")


def test_one_scalar_ref_into_a_series_input_is_a_gather_of_one():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"))}),
    ])

    assert compiled.index_of(Ref("j", "texts")) == Index("j.texts")


def test_a_typed_list_and_a_default_live_on_the_inputs_own_index():
    compiled = _compiled([
        GraphNode(id="typed", type="join", version=1, bindings={"texts": Static(value=["a", "b"])}),
        GraphNode(id="absent", type="join", version=1),
    ])

    assert compiled.index_of(Ref("typed", "texts")) == Index("typed.texts")
    assert compiled.index_of(Ref("absent", "texts")) == Index("absent.texts")


def test_a_gathered_series_judges_each_ref_by_the_element():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="n", type="count", version=1, bindings={"text": Static(value="a")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("a", "result"), ("n", "result"))}),
    ])

    assert [p.code for p in compiled.problems_for()] == ["type_mismatch"]


# --- compile knows which index, never which rows ----------------------------------


def test_compile_stores_no_rows_and_no_mask():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("docs", "texts"))}),
    ])

    assert not hasattr(compiled.lifted_on("up"), "rows")
    assert not hasattr(compiled, "rows")
    assert not hasattr(compiled, "mask")
