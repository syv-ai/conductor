"""One record for what goes wrong, and who raises what."""

import ast
from pathlib import Path
from typing import Annotated, Any

import pytest
from conductor import (
    PROBLEM_CODES,
    AsItArrives,
    CompiledGraph,
    GraphNode,
    NodeRegistry,
    Param,
    Problem,
    Result,
    Static,
)
from conductor.dtype import DType
from conductor.errors import (
    CompileProblem,
    ConductorError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeExecutionError,
    problem,
)
from conductor.graph.model import Graph
from conductor.node import NodeDefinition


class Txt(DType, str):
    id = "errors_txt"
    title = "Txt"


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "c"

    def run(self, text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
        return text


def _registry(*nodes: Any) -> NodeRegistry:
    return NodeRegistry([Echo, *nodes])


def test_a_problem_is_the_four_keys_and_its_fatality_is_its_codes():
    told = problem("no_outputs", "a")

    assert (told.type, told.loc, told.ctx) == ("no_outputs", ("a",), {})
    assert told.fatal is False
    assert problem("unbound_required", "a", "x").fatal is True


def test_a_code_the_catalogue_does_not_know_is_fatal():
    assert Problem(type="the_hosts_own", msg="m").fatal is True


def test_fatality_cannot_be_set_on_a_record():
    """No value is stored, so nothing can disagree with the code's row."""
    told = Problem(type="no_outputs", msg="m", loc=("a",))

    assert told.model_copy(update={"msg": "n"}).fatal is False
    assert Problem.model_validate({"type": "no_outputs", "msg": "m", "fatal": True}).fatal is False


def test_a_dump_has_fatal_and_reads_back():
    told = problem("unknown_locked_field", "a", "x")
    dumped = told.model_dump(mode="json")

    assert dumped == {"type": "unknown_locked_field", "msg": told.msg, "loc": ["a", "x"], "ctx": {}, "fatal": False}
    assert Problem.model_validate(dumped) == told


def test_a_none_place_is_left_out_of_loc():
    assert problem("unbound_required", "a", None).loc == ("a",)


def test_every_code_a_site_writes_is_in_the_catalogue_and_every_row_is_written():
    """``PROBLEM_CODES`` is what a host translating by code covers, so it
    equals the codes written at the sites: a ``problem(...)`` and a ``code=``
    in the engine and compile. ``failed`` is the engine's name for a
    ``NodeError`` the node raised with no code."""
    import conductor.execution
    import conductor.graph

    written: set[str] = {"failed"}

    def literals(expression: ast.expr) -> set[str]:
        return {node.value for node in ast.walk(expression) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    for package in (conductor.execution, conductor.graph):
        for path in Path(package.__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == "problem" and node.args:
                    written |= literals(node.args[0])
                written |= {
                    value for keyword in node.keywords if keyword.arg == "code" for value in literals(keyword.value)
                    if keyword.value is not None
                }

    assert written == PROBLEM_CODES


def test_the_errors_a_caller_catches_sit_under_conductor_error():
    """A failed node is a ``NodeError``; a refused call is a ``ValueError``
    as well, so a host's ``except ValueError`` catches it."""
    assert issubclass(NodeExecutionError, NodeError) and issubclass(NodeError, ConductorError)
    assert issubclass(InvalidCallError, ConductorError) and issubclass(InvalidCallError, ValueError)


def test_a_node_failure_takes_its_code_and_ctx_from_what_the_node_raised():
    failure = NodeExecutionError("went wrong", code="rate_limited", ctx={"retry_after": 30})

    assert (failure.code, failure.ctx, str(failure)) == ("rate_limited", {"retry_after": 30}, "went wrong")


# --- a refused call ------------------------------------------------------------------


def test_a_graph_that_cannot_run_is_refused_with_its_own_problems():
    graph = Graph(nodes=[GraphNode(id="a", type="echo", version=1)])
    compiled = CompiledGraph.from_graph(graph, _registry())

    with pytest.raises(InvalidCallError) as refused:
        compiled.run_sync()

    assert refused.value.errors() == [p for p in compiled.problems if p.fatal]
    assert refused.value.errors()[0].loc == ("a", "text")


# --- hooks -----------------------------------------------------------------------


def _hooked(raised: Exception) -> Any:
    class Hooked(NodeDefinition):
        id = "hooked"
        title = "H"
        description = "d"
        category = "c"

        def compute_inputs(self, declared: Any, values: Any) -> Any:
            raise raised

        def run(self, text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
            return text

    return Hooked


def test_a_compile_problem_from_a_hook_lands_on_the_field_it_names():
    registry = _registry(_hooked(CompileProblem("wrong_mode", "Not that mode.", field="text")))
    graph = Graph(nodes=[GraphNode(id="h", type="hooked", version=1)])

    (told,) = CompiledGraph.from_graph(graph, registry).problems

    assert (told.type, told.msg, told.loc, told.fatal) == ("wrong_mode", "Not that mode.", ("h", "text"), True)


def test_a_hook_may_not_raise_a_code_the_catalogue_says_can_still_run():
    with pytest.raises(ValueError, match="no_outputs"):
        CompileProblem("no_outputs", "m")


def test_a_hook_that_raises_anything_else_is_a_bug_and_leaves_compile():
    """Compile never raises for a fault in the graph; a hook's own bug is not one."""
    registry = _registry(_hooked(KeyError("mode")))
    graph = Graph(nodes=[GraphNode(id="h", type="hooked", version=1)])

    with pytest.raises(KeyError):
        CompiledGraph.from_graph(graph, registry)


# --- problems of their own ---------------------------------------------------------


def test_a_binding_on_a_field_a_hook_took_away_is_unused_and_a_typo_is_fatal():
    class Comes(NodeDefinition):
        id = "comes"
        title = "Comes"
        description = "d"
        category = "c"

        def compute_inputs(self, declared: Any, values: Any) -> Any:
            return declared

        def run(self, text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
            return text

    bound = {"gone": Static(1)}
    (stale,) = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="echo", version=1, bindings={"text": Static("x"), **bound})]), _registry()).problems
    (unused,) = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="comes", version=1, bindings={"text": Static("x"), **bound})]), _registry(Comes)).problems

    assert (stale.type, stale.fatal) == ("stale_binding", True)
    assert (unused.type, unused.fatal) == ("unused_binding", False)


def test_an_unknown_node_type_lists_what_there_is():
    graph = Graph(nodes=[GraphNode(id="a", type="ecko", version=1)])

    (told,) = CompiledGraph.from_graph(graph, _registry()).problems

    assert (told.type, told.ctx["choices"]) == ("unknown_node_type", "echo")
    assert "echo" in told.msg


# --- at import ----------------------------------------------------------------------


def test_a_wrong_declaration_is_a_type_error_with_a_code_and_the_node():
    with pytest.raises(NodeDeclarationError) as wrong:
        class NoReturn(NodeDefinition):
            id = "no-return"
            title = "T"
            description = "d"
            category = "c"

            def run(self, text: Annotated[Txt, Param()]):  # no return type
                return text

    assert isinstance(wrong.value, TypeError)
    assert (wrong.value.code, wrong.value.node_id) == ("missing_return_type", "no-return")


def test_a_wrong_type_has_a_code_and_no_node():
    with pytest.raises(NodeDeclarationError) as wrong:
        class NoId(DType, str):
            title = "x"

    assert (wrong.value.code, wrong.value.node_id) == ("type_without_id", None)


def test_a_registry_refusal_names_the_node():
    class Other(NodeDefinition):
        id = "echo"
        title = "Other"
        description = "d"
        category = "c"

        def run(self, text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
            return text

    with pytest.raises(NodeDeclarationError) as wrong:
        NodeRegistry([Echo, Other])

    assert (wrong.value.code, wrong.value.node_id) == ("duplicate_node_id", "echo")


def test_a_single_annotation_is_still_refused_on_a_named_parameter():
    with pytest.raises(NodeDeclarationError) as wrong:
        class Bad(NodeDefinition):
            id = "bad"
            title = "T"
            description = "d"
            category = "c"

            def run(self, x: AsItArrives) -> Annotated[Txt, Result(title="R")]:
                return Txt("x")

    assert wrong.value.code == "as_it_arrives_on_named_parameter"
