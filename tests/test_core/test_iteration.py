"""Iteration, alignment and reduction, derived at compile."""

from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    ArrivingType,
    AsItArrives,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    Input,
    Output,
    Param,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor.widgets import Textarea
from helpers import registry_of, with_added


class Txt(DType, str):
    id = "iteration-test-txt"
    display_title = "Text"


class Num(DType, float):
    id = "iteration-test-num"
    display_title = "Number"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    filenames: Annotated[Series[Txt], Result(title="Filenames")]


@node("docs", title="Documents", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Documents:
    """A series is made here: one text per document."""
    return Documents(texts=(), filenames=())


@node("upper", title="Upper case", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("pair", title="Pair", description="d", category="test")
def pair(
    a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
    b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(a + b)


@node("triple", title="Triple", description="d", category="test")
def triple(
    a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
    b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
    c: Annotated[Txt, Param(title="C", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(a + b + c)


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    """A reduction: declares the whole series."""
    return Txt("\n".join(texts))


@node("lines", title="Lines", description="d", category="test")
def lines(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    """Opens one text into many: iterating, it is an unfold."""
    return text.splitlines()


@node("count", title="Count", description="d", category="test")
def count(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Num, Result(title="Count")]:
    return Num(len(text))


def route_outputs(value: ArrivingType):
    dtype = value
    return (Output("result", dtype, title="Pass on"),)

@node("route", title="Pass on", description="d", category="test", compute_outputs=route_outputs)
def route(value: Annotated[Any, Param(title="Value")]) -> Annotated[Any, Result(title="Pass on")]:
    """A pass-through: it routes what it does not read, so it declares
    ``Any``, and its hook types the output from what arrives."""
    return value


def only_outputs(**arriving: ArrivingType):
    series = arriving.get("values")
    dtype = series if series is not None else Any
    return (Output("result", dtype, title="The value"),)

@node("only", title="Only one", description="d", category="test", compute_outputs=only_outputs)
def only(values: Annotated[Series[Any], Param(title="Values")]) -> Annotated[Any, Result(title="The value")]:
    """A reduction over whatever arrives: `Series[Any] -> Any`."""
    return values[0]


@node("script", title="Python", description="d", category="test")
def script(code: Annotated[Txt, Param(title="Code", connectable=False, widget=Textarea())] = Txt(""), **inputs: AsItArrives) -> Out:
    """An open interface: every connected name is a parameter, received whole."""
    return Txt(",".join(sorted(inputs)))


def _registry():
    return registry_of(docs, upper_node, pair, triple, join_node, lines, count, route, only, script)


def _compiled(nodes, **kw):
    return CompiledGraph.from_graph(Graph(nodes=nodes, **kw), _registry())


# --- a series into a scalar input makes the node iterate --------------------------------


def test_a_scalar_node_fed_scalars_does_not_iterate():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("a.result")}),
    ])

    assert compiled.run_plan.nodes["b"].runs_per_row_of is None
    assert compiled.field(Ref("b", "result")).dtype is Txt


def test_a_series_is_made_at_the_node_that_produces_it():
    compiled = _compiled([GraphNode(id="docs", type="docs", version=1)])

    assert compiled.field(Ref("docs", "texts")).dtype is Txt and compiled.field(Ref("docs", "texts")).many
    assert Ref("docs", "texts") in compiled.run_plan.nodes["docs"].makes_rows
    assert compiled.run_plan.nodes["docs"].runs_per_row_of is None


def test_each_series_output_makes_rows_on_an_index_of_its_own():
    """Each output that returns rows owns its table: two series outputs of one node are two indexes, named after the outputs."""
    compiled = _compiled([GraphNode(id="docs", type="docs", version=1)])

    assert compiled.run_plan.nodes["docs"].makes_rows == (Ref("docs", "texts"), Ref("docs", "filenames"))


def test_a_series_into_a_scalar_input_makes_the_node_iterate():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["up"].runs_per_row_of == "docs.texts"
    assert compiled.run_plan.nodes["up"].inputs["text"].rows_of == "docs.texts"


def test_an_iterating_nodes_outputs_are_series_on_the_same_index():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
    ])

    assert compiled.field(Ref("up", "result")).dtype is Txt and compiled.field(Ref("up", "result")).many
    assert compiled.run_plan.nodes["up"].runs_per_row_of == "docs.texts"


def test_iteration_is_contagious_and_needs_no_construct():
    """Per-row over many nodes: each node downstream receives a series and
    iterates in turn. Nothing is stored, marked or grouped."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="n", type="count", version=1, bindings={"text": From("up.result")}),
    ])

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "docs.texts"
    assert compiled.field(Ref("n", "result")).dtype is Num and compiled.field(Ref("n", "result")).many


def test_a_scalar_beside_a_series_broadcasts():
    """The same value every row — and no mode on the wire says so."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="prefix", type="upper", version=1, bindings={"text": Static("Attachment: ")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("prefix.result"), "b": From("docs.texts")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["p"].runs_per_row_of == "docs.texts"
    assert compiled.run_plan.nodes["p"].inputs["a"].rows_of is None
    assert compiled.run_plan.nodes["p"].inputs["b"].rows_of == "docs.texts"


# --- admission: the one type question, asked here ---------------------------------


def test_a_mismatched_edge_is_a_fatal_problem_on_the_target_field():
    compiled = _compiled([
        GraphNode(id="n", type="count", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("n.result")}),
    ])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0], make_problem.loc[1]) == ("type_mismatch", True, "up", "text")
    assert make_problem.ctx["source"] == "n.result"
    assert set(make_problem.ctx) == {"source", "source_type", "source_many", "target_type", "target_many", "source_said", "target_said"}
    assert "id" in make_problem.ctx["target_type"]  # a description record, not a name
    assert not compiled.is_runnable


def test_a_mismatched_series_is_judged_by_its_element():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="count", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("n.result")}),
    ])

    assert [p.type for p in compiled.problems] == ["type_mismatch"]


def test_a_pass_through_binds_its_type_from_the_edge():
    """`Any` in takes what arrived, the hook types the output from
    `arriving`, and the node's interface is concrete — nothing downstream sees a
    lost type."""
    compiled = _compiled([
        GraphNode(id="n", type="count", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="r", type="route", version=1, bindings={"value": From("n.result")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("r.result")}),
    ])

    assert compiled.node("r").interface.inputs[0].dtype is Num
    assert compiled.node("r").interface.outputs[0].dtype is Num
    assert compiled.field(Ref("r", "result")).dtype is Num
    # A node aware of the type it inherits: the next edge is checked against the bound type.
    assert [p.type for p in compiled.problems] == ["type_mismatch"]


def test_an_iterating_pass_through_binds_the_element_and_iterates():
    """A pass-through fed a series iterates like any scalar node, and
    takes the element as its type."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="r", type="route", version=1, bindings={"value": From("docs.texts")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("r.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("r", "result")).dtype is Txt and compiled.field(Ref("r", "result")).many
    assert compiled.run_plan.nodes["r"].runs_per_row_of == "docs.texts"
    assert compiled.run_plan.nodes["up"].runs_per_row_of == "docs.texts"


def test_a_reduction_over_the_variable_binds_from_the_series_element():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="one", type="only", version=1, bindings={"values": From("docs.texts")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("one").interface.inputs[0].dtype is Txt and compiled.node("one").interface.inputs[0].many
    assert compiled.field(Ref("one", "result")).dtype is compiled.field(Ref("one", "result")).dtype
    assert compiled.field(Ref("one", "result")).dtype is Txt
    assert compiled.run_plan.nodes["one"].runs_per_row_of is None


def test_an_unconnected_any_input_is_unbound_required_and_the_node_has_no_shape():
    """The honest mid-edit state: fatal, on the field, and nothing
    is guessed about the node. There is no second code for an untyped
    field — unconnected is unconnected."""
    compiled = _compiled([GraphNode(id="r", type="route", version=1)])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0], make_problem.loc[1]) == ("unbound_required", True, "r", "value")
    assert compiled.node("r")._fields is None


def test_a_static_on_an_any_input_is_unbound_too():
    """Only an edge can say what the type is."""
    compiled = _compiled([GraphNode(id="r", type="route", version=1, bindings={"value": Static("hi")})])

    assert [p.type for p in compiled.problems] == ["unbound_required"]


def test_an_open_inputs_takes_one_input_per_edge_received_whole():
    """Two edges, two parameters, each typed by what arrives — a
    series arrives whole and makes nothing iterate."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="count", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="s", type="script", version=1, bindings={
            "code": Static("return {}"),
            "antal": From("n.result"),
            "tekster": From("docs.texts"),
        }),
    ])

    assert compiled.is_runnable, compiled.problems
    interface = compiled.node("s").interface
    assert [(i.name, i.dtype, i.many) for i in interface.inputs] == [("code", Txt, False), ("antal", Num, False), ("tekster", Txt, True)]
    assert interface.inputs[2].widget is None
    assert compiled.run_plan.nodes["s"].runs_per_row_of is None
    assert compiled.run_plan.nodes["s"].inputs["tekster"].receive_mode == "entire_value"


def test_an_open_inputs_parameter_takes_one_edge():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="s", type="script", version=1, bindings={"code": Static(""), "x": From("a.result", "b.result")}),
    ])

    assert [p.type for p in compiled.problems] == ["one_edge_per_parameter"]


def test_an_open_inputs_parameter_is_named_like_a_keyword_argument():
    """`**inputs` is Python's own spelling: a parameter's name is an
    identifier, or the node has no signature to receive it in."""
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="s", type="script", version=1, bindings={"code": Static(""), "my value": From("a.result")}),
    ])

    assert [(p.type, p.loc[1]) for p in compiled.problems] == [("parameter_name_invalid", "my value")]


def test_a_node_with_a_broken_edge_has_no_shape_and_says_so_once():
    compiled = _compiled([
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("ghost.result")}),
        GraphNode(id="c", type="upper", version=1, bindings={"text": From("b.result")}),
    ])

    assert [p.type for p in compiled.problems] == ["unknown_ref_node"]
    assert compiled.node("b")._fields is None
    with pytest.raises(KeyError):
        compiled.field(Ref("c", "result")).dtype


def test_a_node_with_no_outputs_yet_is_the_ordinary_mid_edit_state():
    """Outputs made from what the author has not yet given are not a fault:
    non-fatal, the graph still runs, the node is still derived — there is
    simply nothing to edge from it yet."""

    def sheet_outputs(header=""):
        return tuple(
            Output(col, Txt)
            for col in str(header).split(",")
            if col
        )

    @node("sheet", title="Sheet", description="d", category="test", compute_outputs=sheet_outputs)
    def sheet(header: Annotated[Txt, Param(title="Header", widget=Textarea())] = Txt("")) -> Out:
        return header

    registry = with_added(_registry(), sheet)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="sheet", version=1)]), registry)

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0]) == ("no_outputs", False, "s")
    assert compiled.is_runnable
    assert compiled.run_plan.nodes["s"].runs_per_row_of is None
    assert compiled.node("s").interface.outputs == ()


# --- alignment is lineage -----------------------------------------------------


def test_compute_outputs_sees_the_dtype_each_connected_input_receives():
    """An opener's columns come from what arrives, per unit — the element of a series."""

    def opener_outputs(value: ArrivingType):
        if value.id == "any":
            return (Output("result", Txt, title="R"),)
        return (Output(f"from_{value.id}", Txt, title="Opened"),)

    @node("opener", title="Opener", description="d", category="test", compute_outputs=opener_outputs)
    def opener(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    registry = with_added(_registry(), opener)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="o", type="opener", version=1, bindings={"value": From("docs.texts")}),
        GraphNode(id="bare", type="opener", version=1),
    ]), registry)

    assert compiled.is_runnable, compiled.problems
    assert [o.name for o in compiled.node("o").interface.outputs] == ["from_iteration-test-txt"]
    assert compiled.run_plan.nodes["o"].runs_per_row_of == "docs.texts"
    assert [o.name for o in compiled.node("bare").interface.outputs] == ["result"]


def test_a_hook_that_cannot_answer_refuses_and_the_refusal_is_the_nodes_problem():
    """`CompileProblem(code, message)` from a field hook — asked once, with real
    arrivals — lands as the node's one fatal `Problem`: the host
    names the code and writes the sentence, compile only anchors it, and
    there is no `no_outputs` echo beside it."""
    from conductor import ComputeHookError

    def fussy_outputs():
        raise ComputeHookError("What arrives does not fit.", code="wrong_shape")

    @node("fussy", title="Fussy", description="d", category="test", compute_outputs=fussy_outputs)
    def fussy(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    registry = with_added(_registry(), fussy)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="f", type="fussy", version=1, bindings={"value": From("docs.texts")}),
    ]), registry)

    (make_problem,) = [p for p in compiled.problems if p.loc[0] == "f"]
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0]) == ("wrong_shape", True, "f")
    assert make_problem.msg == "What arrives does not fit."
    assert compiled.node("f").interface.outputs == ()


def test_an_inputs_hook_that_cannot_answer_refuses_and_compile_does_not_raise():
    """`CompileProblem` from `compute_added_inputs` is the node's one fatal `Problem`
    too, asked before any edge is walked: compile never raises for a fault
    in the graph, and a node reading the refused one carries no echo."""
    from conductor import ComputeHookError

    def fussy_inputs():
        raise ComputeHookError("That mode is not one this node has.", code="wrong_mode")

    @node("fussy-inputs", title="Fussy", description="d", category="test", compute_added_inputs=fussy_inputs)
    def fussy(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt(""), **more: Any) -> Out:
        return value

    registry = with_added(_registry(), fussy)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="f", type="fussy-inputs", version=1, bindings={"value": Static("x")}),
        GraphNode(id="u", type="upper", version=1, bindings={"text": From("f.result")}),
    ]), registry)

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0]) == ("wrong_mode", True, "f")
    assert make_problem.msg == "That mode is not one this node has."
    assert not compiled.is_runnable


def test_a_hook_reads_a_defaulted_static_nothing_bound():
    """The edges pass hands `compute_outputs` the declaration's defaults
    under the typed statics, so a hook indexes `values[...]` with no
    guard and an unbound defaulted field means what the declaration
    states."""

    def suffixer_outputs(suffix):
        return (Output(str(suffix), Txt, title="Out"),)

    @node("suffixer", title="Suffixes", description="d", category="test", compute_outputs=suffixer_outputs)
    def suffixer(suffix: Annotated[Txt, Param(title="Suffix", widget=Textarea())] = Txt("out")) -> Out:
        return suffix

    registry = with_added(_registry(), suffixer)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="suffixer", version=1)]), registry)

    assert compiled.is_runnable, compiled.problems
    assert [o.name for o in compiled.node("s").interface.outputs] == ["out"]



def test_two_series_on_one_index_align():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("up.result"), "b": From("docs.texts")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["p"].runs_per_row_of == "docs.texts"


def test_two_series_outputs_of_one_node_do_not_line_up():
    """Each output owns its rows, so two of one node's series are two tables: a node fed both per row can't tell which goes with which."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("docs.texts"), "b": From("docs.filenames")}),
    ])

    assert [make_problem.type for make_problem in compiled.problems] == ["misaligned"]


def test_two_series_on_unrelated_indexes_are_a_fatal_problem_naming_both():
    """Same length is never the test: compile cannot know lengths."""
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("a.texts"), "b": From("b.texts")}),
    ])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc) == ("misaligned", True, ("p",))
    assert "p.a" in make_problem.msg and "p.b" in make_problem.msg
    assert make_problem.ctx == {"a": "p.a", "b": "p.b"}
    assert compiled.node("p")._fields is None


# --- reduction ---------------------------------------------------------------------


def test_a_reduction_on_a_root_yields_a_scalar_and_does_not_iterate():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("up.result")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["j"].runs_per_row_of is None
    assert compiled.run_plan.nodes["j"].inputs["texts"].receive_mode == "entire_value"
    assert compiled.field(Ref("j", "result")).dtype is Txt


def test_a_node_running_per_row_returning_a_series_makes_rows_of_a_child_index():
    """An unfold with no special node: one text in, many lines out, each
    line knowing which document it came from."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": From("docs.texts")}),
    ])

    assert compiled.run_plan.nodes["lines"].runs_per_row_of == "docs.texts"
    assert compiled.field(Ref("lines", "result")).dtype is Txt and compiled.field(Ref("lines", "result")).many
    assert compiled.run_plan.nodes["lines"].makes_rows == (Ref("lines", "result"),)


def test_a_reduction_on_a_child_runs_per_row_of_the_parent():
    """One per document again: the index supplies the grouping, and there
    is no key column to pick."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("lines.result")}),
    ])

    assert compiled.run_plan.nodes["j"].runs_per_row_of == "docs.texts"
    assert compiled.field(Ref("j", "result")).dtype is Txt and compiled.field(Ref("j", "result")).many


def test_a_parent_index_series_broadcasts_down_to_a_child_index_node():
    """The firma is the same for each of its employees."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="lines", type="lines", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("lines.result"), "b": From("docs.texts")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["p"].runs_per_row_of == "lines.result"


def test_two_children_of_one_parent_do_not_align():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="l1", type="lines", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="l2", type="lines", version=1, bindings={"text": From("docs.filenames")}),
        GraphNode(id="p", type="pair", version=1, bindings={"a": From("l1.result"), "b": From("l2.result")}),
    ])

    assert [p.type for p in compiled.problems] == ["misaligned"]


# --- gather ---------------------------------------------------------------------------


def test_n_scalar_refs_into_a_series_input_gather_onto_a_fresh_index():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static("b")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result", "b.result")}),
    ])

    assert compiled.run_plan.nodes["j"].runs_per_row_of is None
    assert compiled.field(Ref("j", "texts")).dtype is Txt and compiled.field(Ref("j", "texts")).many
    assert compiled.run_plan.nodes["j"].inputs["texts"].receive_mode == "collected"


def test_n_series_refs_concatenate_and_lineage_is_gone():
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.texts", "b.texts")}),
    ])

    assert compiled.is_runnable
    assert compiled.run_plan.nodes["j"].runs_per_row_of is None
    assert compiled.run_plan.nodes["j"].inputs["texts"].receive_mode == "collected"


def test_two_refs_on_one_index_into_a_scalar_input_are_a_union_on_it():
    """A merge is edges. The node runs per row of the shared index and reads per row."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="a", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="m", type="upper", version=1, bindings={"text": From("a.result", "b.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["m"].runs_per_row_of == "docs.texts"
    assert compiled.field(Ref("m", "result")).dtype is Txt and compiled.field(Ref("m", "result")).many


def test_two_refs_on_different_indexes_into_a_scalar_input_is_fatal():
    compiled = _compiled([
        GraphNode(id="a", type="docs", version=1),
        GraphNode(id="b", type="docs", version=1),
        GraphNode(id="m", type="upper", version=1, bindings={"text": From("a.texts", "b.texts")}),
    ])

    (make_problem,) = [p for p in compiled.problems if p.loc[0] == "m"]
    assert (make_problem.type, make_problem.loc[1]) == ("union_needs_one_index", "text")


def test_two_series_refs_on_one_index_into_a_series_input_read_that_index_not_a_pile():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="a", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result", "b.result")}),
    ])

    assert compiled.run_plan.nodes["j"].runs_per_row_of is None
    assert compiled.run_plan.nodes["j"].inputs["texts"].receive_mode == "entire_value"


def test_one_scalar_ref_into_a_series_input_is_a_gather_of_one():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result")}),
    ])

    assert compiled.run_plan.nodes["j"].inputs["texts"].receive_mode == "collected"


def test_a_typed_list_and_a_default_live_on_the_inputs_own_index():
    compiled = _compiled([
        GraphNode(id="typed", type="join", version=1, bindings={"texts": Static(["a", "b"])}),
        GraphNode(id="absent", type="join", version=1),
    ])

    assert compiled.run_plan.nodes["typed"].inputs["texts"].receive_mode == "entire_value"
    assert compiled.run_plan.nodes["absent"].inputs["texts"].receive_mode == "entire_value"


def test_a_gathered_series_judges_each_ref_by_the_element():
    compiled = _compiled([
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("a")}),
        GraphNode(id="n", type="count", version=1, bindings={"text": Static("a")}),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result", "n.result")}),
    ])

    assert [p.type for p in compiled.problems] == ["type_mismatch"]


# --- compile knows which index, never which rows ----------------------------------


def test_compile_stores_no_rows_and_no_mask():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
    ])

    assert not hasattr(compiled, "rows")
    assert not hasattr(compiled, "mask")


# --- what a hook may answer ---------------------------------------------------------


def test_a_hook_that_adds_an_input_of_a_type_no_edge_can_carry_closes_it():
    """A type no edge can carry closes the input it makes."""

    def adds_inputs():
        return (Input("raw", list[int], title="Raw"),)

    @node("adds", title="Adds", description="d", category="test", compute_added_inputs=adds_inputs)
    def adds(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt(""), **more: Any) -> Out:
        return value

    registry = with_added(_registry(), adds)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="n", type="count", version=1, bindings={"text": Static("hi")}),
        GraphNode(id="a", type="adds", version=1, bindings={"raw": From("n.result")}),
    ]), registry)

    assert [(p.type, p.loc[1]) for p in compiled.problems] == [("edge_into_unconnectable_input", "raw")]


def test_a_hook_that_leaves_a_connected_output_untyped_is_a_problem_not_a_crash():
    """An output typed ``Any`` after the hook has answered can carry nothing
    on an edge. Reading it is a problem, not a crash of the walk."""

    def vague_outputs():
        return (Output("result", Any, title="Out"),)  # never types it

    @node("vague", title="Vague", description="d", category="test", compute_outputs=vague_outputs)
    def vague(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Annotated[Any, Result(title="Out")]:
        return value

    registry = with_added(_registry(), vague)
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="v", type="vague", version=1, bindings={"value": Static("hi")}),
        GraphNode(id="u", type="upper", version=1, bindings={"text": From("v.result")}),
    ]), registry)

    assert [(p.type, p.fatal, p.loc[0], p.loc[1]) for p in compiled.problems] == [("output_has_no_type", True, "v", "result")]
    assert not compiled.is_runnable


def test_a_refusal_from_a_node_whose_source_is_broken_is_dropped_for_the_sources_fault():
    """The broken edge carries the fault; the hook's refusal about the arrival
    it never got would report the same fact twice. The node still gets an
    interface an editor can draw, with no outputs."""
    from conductor import ComputeHookError

    def fussy_outputs():
        raise ComputeHookError("What arrives does not fit.", code="wrong_shape")

    @node("fussy", title="Fussy", description="d", category="test", compute_outputs=fussy_outputs)
    def fussy(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    registry = with_added(_registry(), fussy)

    def codes(source_field):
        compiled = CompiledGraph.from_graph(Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="f", type="fussy", version=1, bindings={"value": From(f"docs.{source_field}")}),
        ]), registry)
        return [p.type for p in compiled.problems if p.loc[0] == "f"], compiled.node("f")

    broken, graph_node = codes("nope")
    assert broken == ["unknown_ref_output"]
    assert (graph_node.status, graph_node.interface.outputs) == ("wiring_failed", ())
    assert codes("texts")[0] == ["wrong_shape"]


# --- a list set on a node --------------------------------------------------------


def test_a_list_set_on_a_node_running_per_row_hangs_under_its_rows():
    """Questions set on a node that runs per document: a questions index under the documents index, and the node runs per question per document."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="ask", type="pair", version=1, bindings={"a": From("docs.texts"), "b": Static(["q1", "q2"])}),
    ])

    assert compiled.is_runnable, compiled.problems
    questions = "ask.b"
    assert compiled.run_plan.nodes["ask"].inputs["b"].receive_mode == "value_at_row" and compiled.run_plan.nodes["ask"].inputs["b"].rows_of == questions
    assert compiled.run_plan.nodes["ask"].runs_per_row_of == questions


def test_a_list_set_on_a_node_that_runs_once_is_a_root():
    compiled = _compiled([GraphNode(id="ask", type="pair", version=1, bindings={"b": Static(["q1", "q2"])})])

    assert compiled.run_plan.nodes["ask"].runs_per_row_of == "ask.b"


def test_two_lists_on_a_node_running_per_row_are_refused():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="ask", type="triple", version=1, bindings={
            "a": From("docs.texts"), "b": Static(["q1", "q2"]), "c": Static(["x", "y"]),
        }),
    ])

    (raised,) = compiled.problems
    assert (raised.type, raised.fatal, raised.loc[1]) == ("lists_on_a_per_row_node", True, "c")
    assert raised.ctx == {"a": "ask.b", "b": "ask.c"}


def test_two_lists_on_a_node_that_runs_once_are_the_list_refusal():
    compiled = _compiled([GraphNode(id="ask", type="pair", version=1, bindings={"a": Static(["p", "q"]), "b": Static(["x", "y"])})])

    assert [make_problem.type for make_problem in compiled.problems] == ["lists_on_a_per_row_node"]
