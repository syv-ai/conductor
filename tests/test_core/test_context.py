"""The run's context: one object the host passes for a run, which a node takes as its first parameter.

The host names the class once, on the registry (``NodeRegistry(context_type=AppContext)``), and passes an instance
of it for each run (``graph.run(context=AppContext(...))``). A node asks for it by its first parameter,
``def sign(ctx: Context[AppContext], text: str)``. Compiling checks that the registry's class has what each node
asks for; starting a run checks that the object is of that class. The context is never saved with a run, and it
plays no part in which outcomes of the previous run a run reuses.
"""

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
from conductor import (
    Context,
    Graph,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    Run,
    Series,
    make_graph,
    node,
    subgraph,
)
from conductor.testing import check_registry


@dataclass(frozen=True)
class Caller:
    name: str


class HttpClient:
    def get(self, url: str) -> str:
        return f"<{url}>"


@dataclass(frozen=True)
class AppContext:
    caller: Caller
    http_client: HttpClient


IDA = AppContext(caller=Caller("Ida"), http_client=HttpClient())


class HasHttpClient(Protocol):
    @property
    def http_client(self) -> HttpClient: ...


@node("sign", title="Sign", description="Signs a text with the name of the person running it.")
def sign(ctx: Context[AppContext], text: str) -> str:
    return f"{text}, {ctx.caller.name}"


@node("fetch", title="Fetch", description="Reads a page.")
def fetch(ctx: Context[HasHttpClient], url: str) -> str:
    return ctx.http_client.get(url)


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@node("lines", title="Lines", description="For a test.")
def lines(text: str) -> Series[str]:
    return text.split("\n")


def app_registry(*definitions: Any) -> NodeRegistry[AppContext]:
    registry = NodeRegistry(context_type=AppContext)
    registry.add(*definitions)
    return registry


def problems_of(graph: Graph[Any]) -> list[tuple[str, tuple[str | int, ...]]]:
    return [(p.type, p.loc) for p in graph.problems]


# -- a node declares it --------------------------------------------------------------------------


def test_a_node_takes_the_context_as_its_first_parameter_and_a_test_calls_it_with_the_object_first():
    assert sign(IDA, "Done") == "Done, Ida"
    assert list(sign.inputs) == ["text"]
    assert sign.context is not None and (sign.context.name, sign.context.cls) == ("ctx", AppContext)
    assert upper.context is None


def test_the_context_is_recognised_by_its_metadata_whatever_the_parameter_is_called():
    @node("signed", title="Signed", description="For a test.")
    def signed(app: Context[AppContext], /, text: str) -> str:
        return f"{text}, {app.caller.name}"

    assert signed.context is not None and signed.context.name == "app" and list(signed.inputs) == ["text"]
    assert Graph.compile(make_graph(signed), app_registry(signed)).run(text="Done", context=IDA).outputs == {"result": "Done, Ida"}


def _second(text: str, ctx: Context[AppContext]) -> str:
    return text


def _keyword_only(*, ctx: Context[AppContext], text: str) -> str:
    return text


def _optional(ctx: Context[AppContext] | None = None) -> str:
    return "x"


def _in_a_list(ctx: list[Context[AppContext]]) -> str:
    return "x"


def _returned() -> Context[AppContext]:
    return IDA


@pytest.mark.parametrize("function", [_second, _keyword_only, _optional, _in_a_list, _returned])
def test_the_context_anywhere_but_the_whole_first_parameter_is_refused_at_declaration(function: Any):
    with pytest.raises(NodeDeclarationError, match="Context") as raised:
        node("x", title="X", description="For a test.")(function)
    assert raised.value.code == "context_misplaced"


def test_the_context_names_a_class():
    with pytest.raises(NodeDeclarationError, match="Context names the class") as raised:

        @node("bare", title="Bare", description="For a test.")
        def bare(ctx: Context[list[str]], text: str) -> str:
            return text

    assert raised.value.code == "context_not_a_class"


def test_an_added_input_named_as_the_context_parameter_is_the_hooks_failure():
    def added(formula: str) -> dict[str, Any]:
        return {"ctx": str}

    @node("adds", title="Adds", description="For a test.", compute_added_inputs=added)
    def adds(ctx: Context[AppContext], formula: str, **values: str) -> str:
        return formula

    [problem] = Graph.compile(make_graph(adds, constants={"adds.formula": "x"}), app_registry(adds)).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("adds",), "added_input_taken")


# -- the registry names the class --------------------------------------------------------------


def test_a_registry_names_the_class_of_the_context_and_a_filtered_registry_keeps_it():
    registry = app_registry(sign, upper)
    assert registry.context_type is AppContext and NodeRegistry().context_type is None
    assert registry.filter(lambda definition: definition.id == "sign").context_type is AppContext


def test_a_subregistry_naming_the_same_class_or_none_is_added():
    registry = app_registry(sign)
    registry.add_subregistry(app_registry(fetch))
    prebuilt = NodeRegistry()
    prebuilt.add(upper)
    registry.add_subregistry(prebuilt)
    assert registry.context_type is AppContext and [definition.id for definition in registry] == ["sign", "fetch", "upper"]


def test_a_subregistry_naming_another_class_or_one_a_registry_with_none_lacks_is_refused():
    @dataclass(frozen=True)
    class OtherContext:
        caller: Caller

    with pytest.raises(NodeDeclarationError, match="OtherContext") as raised:
        app_registry().add_subregistry(NodeRegistry(context_type=OtherContext))  # type: ignore[arg-type]
    assert raised.value.code == "context_type_differs"
    with pytest.raises(NodeDeclarationError, match="AppContext") as raised:
        NodeRegistry().add_subregistry(app_registry(sign))  # type: ignore[arg-type]
    assert raised.value.code == "context_type_differs"


# -- compiling checks the class against what each node asks for ---------------------------------


def test_a_node_may_ask_for_the_class_a_base_of_it_or_a_protocol_it_satisfies():
    @dataclass(frozen=True)
    class TenantContext(AppContext):
        tenant: str

    @node("caller_of", title="Caller of", description="For a test.")
    def caller_of(ctx: Context[AppContext]) -> str:
        return ctx.caller.name

    registry = NodeRegistry(context_type=TenantContext)
    registry.add(caller_of, fetch)
    record = make_graph(caller_of, fetch, constants={"fetch.url": "a"}, edges=[], outputs={"caller": "caller_of.result", "fetched": "fetch.result"})
    graph = Graph.compile(record, registry)
    assert graph.problems == []
    assert graph.run(context=TenantContext(caller=Caller("Bo"), http_client=HttpClient(), tenant="t")).outputs == {
        "caller": "Bo", "fetched": "<a>",
    }


def test_a_node_asking_for_what_the_class_lacks_is_a_problem_at_the_node_and_parameter():
    class HasDatabase(Protocol):
        @property
        def database(self) -> str: ...

    @node("count", title="Count", description="For a test.")
    def count(ctx: Context[HasDatabase], text: str) -> int:
        return len(text)

    graph = Graph.compile(make_graph(count), app_registry(count))
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("context_does_not_fit", ("count", "ctx"))
    assert problem.ctx == {"node_id": "count", "parameter": "ctx", "needs": "HasDatabase", "context_type": "AppContext", "lacks": ["database"], "variant": "lacks"}


def test_a_protocol_is_checked_by_its_members_names_only_so_a_member_of_another_type_fits():
    """Whether the member's type fits is the host's type checker's job, not compiling's."""
    @dataclass(frozen=True)
    class TextContext:
        http_client: str

    registry = NodeRegistry(context_type=TextContext)
    registry.add(fetch)
    assert Graph.compile(make_graph(fetch), registry).problems == []


def test_a_class_that_is_not_the_registrys_nor_a_base_of_it_does_not_fit():
    @node("caller_of", title="Caller of", description="For a test.")
    def caller_of(ctx: Context[Caller]) -> str:
        return ctx.name

    [problem] = Graph.compile(make_graph(caller_of), app_registry(caller_of)).problems
    assert (problem.type, problem.ctx) == ("context_does_not_fit", {"node_id": "caller_of", "parameter": "ctx", "needs": "Caller", "context_type": "AppContext"})
    assert "AppContext is not a Caller" in problem.msg


def test_a_node_that_takes_a_context_on_a_registry_that_names_no_class_is_a_problem():
    registry = NodeRegistry()
    registry.add(sign)
    [problem] = Graph.compile(make_graph(sign), registry).problems
    assert (problem.type, problem.loc, problem.ctx["variant"]) == ("context_does_not_fit", ("sign", "ctx"), "no_context_type")
    assert "NodeRegistry(context_type=" in problem.msg


def test_a_node_inside_a_subgraph_is_checked_with_the_rest_and_its_problem_is_at_the_subgraph_node():
    class HasDatabase(Protocol):
        @property
        def database(self) -> str: ...

    @node("count", title="Count", description="For a test.")
    def count(ctx: Context[HasDatabase], text: str) -> int:
        return len(text)

    counted = subgraph("counted", title="Counted", description="For a test.")(make_graph(count))
    assert problems_of(Graph.compile(make_graph(counted), app_registry(count, counted))) == [("context_does_not_fit", ("counted", "count", "ctx"))]


# -- starting a run checks the object -----------------------------------------------------------


def test_every_node_that_takes_the_context_gets_the_one_object_inside_a_subgraph_too():
    signed = subgraph("signed", title="Signed", description="For a test.")(make_graph(lines, sign, edges=[("lines.result", "sign.text")]))
    record = make_graph(signed, sign, edges=[], inputs={"inner": "signed.text", "outer": "sign.text"}, outputs={"inner": "signed.result", "outer": "sign.result"})
    run = Graph.compile(record, app_registry(lines, sign, signed)).run(inner="a\nb", outer="c", context=IDA)
    assert run.outputs == {"inner": ["a, Ida", "b, Ida"], "outer": "c, Ida"}


def test_a_run_whose_nodes_take_the_context_and_is_given_none_is_refused_before_any_call():
    signed = subgraph("signed", title="Signed", description="For a test.")(make_graph(sign))
    record = make_graph(signed, sign, edges=[], inputs={"inner": "signed.text", "outer": "sign.text"}, outputs={"inner": "signed.result", "outer": "sign.result"})
    graph = Graph.compile(record, app_registry(sign, signed))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(inner="a", outer="b")
    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        ("context_missing", ("sign", "ctx"), {"parameter": "ctx", "needs": "AppContext"}),
        ("context_missing", ("signed", "sign", "ctx"), {"parameter": "ctx", "needs": "AppContext"}),
    ]


def test_a_context_of_another_class_is_refused_at_the_start_of_a_run():
    graph = Graph.compile(make_graph(sign), app_registry(sign))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text="a", context=Caller("Ida"))  # type: ignore[arg-type]
    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        ("invalid_context", ("context",), {"expected": "AppContext", "got": "Caller"}),
    ]
    assert "Ida" not in refused.value.problems[0].msg


def test_a_context_given_to_a_graph_whose_registry_names_no_class_is_refused():
    registry = NodeRegistry()
    registry.add(upper)
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(make_graph(upper), registry).run(text="a", context=IDA)  # type: ignore[arg-type]
    assert [(p.type, p.ctx) for p in refused.value.problems] == [("invalid_context", {"got": "AppContext", "variant": "no_context_type"})]


def test_a_graph_none_of_whose_nodes_takes_the_context_runs_with_it_or_without():
    graph = Graph.compile(make_graph(upper), app_registry(upper, sign))
    assert graph.run(text="a").outputs == graph.run(text="a", context=IDA).outputs == {"result": "A"}


def test_resume_and_stream_are_checked_as_run_is():
    ready = Graph.compile(make_graph(sign), app_registry(sign)).with_inputs(text="a")
    with pytest.raises(InvalidCallError) as refused:
        ready.stream()
    assert [p.type for p in refused.value.problems] == ["context_missing"]

    async def events() -> list[Any]:
        return [event async for event in ready.stream(context=IDA)]

    assert asyncio.run(events())[-1].run.outputs == ready.resume(context=IDA).outputs == {"result": "a, Ida"}


# -- never saved, and no part of reuse ---------------------------------------------------------


def test_the_context_is_never_saved_and_is_passed_again_on_each_resume():
    calls: list[str] = []

    @node("ask_once", title="Ask once", description="For a test.")
    def ask_once(ctx: Context[AppContext], text: str) -> str:
        calls.append(ctx.caller.name)
        return text

    graph = Graph.compile(make_graph(ask_once), app_registry(ask_once))
    saved = graph.with_inputs(text="a").to_json()
    assert "Ida" not in saved
    assert Run.from_json(saved, graph).resume(context=IDA).outputs == {"result": "a"} and calls == ["Ida"]


def test_a_run_again_with_another_context_reuses_what_the_node_returned():
    calls: list[str] = []

    @node("signed", title="Signed", description="For a test.")
    def signed(ctx: Context[AppContext], text: str) -> str:
        calls.append(ctx.caller.name)
        return f"{text} by {ctx.caller.name}"

    first = Graph.compile(make_graph(signed), app_registry(signed)).run(text="hi", context=IDA)
    again = first.again(context=dataclasses.replace(IDA, caller=Caller("Bo")))
    assert calls == ["Ida"] and again.outputs == {"result": "hi by Ida"}


# -- check_registry ------------------------------------------------------------------------------


def test_check_registry_runs_each_example_with_the_context_it_is_given():
    registry = app_registry(sign, upper)
    assert check_registry(registry, examples={"sign": [{"text": "a"}]}, context=IDA) == []
    [problem] = check_registry(registry, examples={"sign": [{"text": "a"}]})
    assert (problem.type, problem.loc, [one["type"] for one in problem.ctx["problems"]]) == ("run_failed", ("sign", 1), ["context_missing"])
