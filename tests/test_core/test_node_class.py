from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Annotated, Any, ClassVar

import pytest
from conductor import (
    ArrivingType,
    AsItArrives,
    CompiledGraph,
    Deprecation,
    DType,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeDefinition,
    NodeDescription,
    NodeRegistry,
    Output,
    Param,
    Policy,
    Result,
    Series,
    Static,
    Upgrade,
    node,
)
from conductor.widgets import Choice, Dropdown, Textarea
from pydantic import TypeAdapter


class Txt(DType, str):
    id = "node-class-test-txt"
    display_title = "Text"


#: The single-output return, spelled once so the fixtures below do not each
#: repeat it. It IS a `Result`; a record appears in the multi-output test.
Out = Annotated[Txt, Result(title="R")]


def test_a_node_declares_its_identity_and_implements_run():
    @node("greeter", title="Greeter", description="Says hello", category="test")
    def greeter(name: Annotated[Txt, Param(title="Name", widget=Textarea())] = Txt("")) -> Out:
        return Txt(f"hi {name}")

    assert greeter(name=Txt("ida")) == "hi ida"


def test_the_interface_is_derived_when_the_class_is_defined():
    """Subclassing is the trigger, so it cannot be forgotten."""

    @node("greeter-2", title="Greeter", description="d", category="test")
    def greeter(name: Annotated[Txt, Param(title="Name", widget=Textarea())] = Txt("")) -> Out:
        return name

    iface = greeter.versions[greeter.current_version].interface
    assert [i.name for i in iface.inputs] == ["name"]
    assert [h.name for h in iface.outputs] == ["result"]


def test_a_bare_widget_fails_at_import():
    """The widget goes on the ``Param``; a bare one is an error, not a fallback."""
    with pytest.raises(TypeError, match="Param"):

        @node("bare", title="Bare", description="d", category="test")
        def bare(x: Annotated[Txt, Textarea()] = Txt("")) -> Out:
            return x


def test_a_class_says_nothing_about_what_the_engine_must_do():
    """There is no role. ``SKIPPED`` and ``AskPerson`` are values a
    ``run`` returns, and the engine acts on the value. A flag per
    capability is the same fact under another name."""

    @node("greeter-3", title="Greeter", description="d", category="test")
    def greeter(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    for gone in ("role", "is_decision", "is_signal", "is_input", "asks", "dynamic_handles"):
        assert not hasattr(greeter, gone), gone


def test_an_intermediate_base_declares_nothing():
    """A shared base that adds no `run` is not a node."""

    class Shared(NodeDefinition):
        category: ClassVar[str] = "io"

    assert not hasattr(Shared, "versions")
    assert not hasattr(Shared, "current")


def test_a_node_may_be_deprecated_and_the_notice_is_content():
    """The record's presence is the fact. Nothing is wrong with a
    graph that uses a deprecated node — it is not a Problem."""

    @node("old", title="Old", description="d", category="test", deprecation=Deprecation(header="Retired", alternative="greeter", migration="Use Greeter instead."))
    def old_node(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    @node("live", title="Live", description="d", category="test")
    def live(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert old_node.versions[1].deprecation == Deprecation(header="Retired", alternative="greeter", migration="Use Greeter instead.")
    assert live.versions[1].deprecation is None
    assert not hasattr(old_node, "deprecated"), "no boolean beside the record"


def test_a_bare_deprecated_means_going_away_with_no_details_yet():
    @node("old-bare", title="Old", description="d", category="test", deprecation=Deprecation())
    def old_node(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert old_node.versions[1].deprecation == Deprecation()




def test_a_version_with_no_policy_gets_the_default():
    @node("plain-policy", title="Plain", description="d", category="test")
    def plain(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert plain.versions[1].policy == Policy()
    assert plain.versions[1].policy.retries == 0
    assert plain.versions[1].policy.concurrency == 8


def test_policy_is_declared_per_version():
    def fetch_v1(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    @node("fetch", title="Fetch", description="d", category="test", policy=Policy(attempt_timeout=30.0, concurrency=1), replaces=node("fetch", policy=Policy(retries=3, retry_delay=1.0))(fetch_v1), upgrade=Upgrade())
    def fetch(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert fetch.versions[1].policy.retries == 3
    assert fetch.versions[2].policy.retries == 0
    assert fetch.versions[2].policy.attempt_timeout == 30.0
    assert fetch.versions[2].policy.concurrency == 1


def test_the_registry_gives_back_the_class():
    @node("translate", title="Translation", description="Translates text", category="test")
    def translate(
        text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
        language: Annotated[Txt, Param(title="Language", widget=Dropdown(choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English"))))] = Txt("da"),
    ) -> Out:
        return Txt(f"{language}:{text}")

    registry = NodeRegistry([translate])

    assert registry["translate"] is translate


def test_a_caller_asks_the_class_rather_than_a_copy_of_it():
    """There is no record, so there is nothing to keep in step."""

    @node("translate-2", title="Translation", description="d", category="test")
    def translate(
        text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
        language: Annotated[Txt, Param(title="Language", widget=Dropdown(choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English"))))] = Txt("da"),
    ) -> Out:
        return text

    registry = NodeRegistry([translate])
    found = registry["translate-2"]

    assert found.title == "Translation"
    iface = found.versions[1].interface
    assert [i.name for i in iface.inputs] == ["text", "language"]
    assert iface.inputs[0].connectable is True


def test_every_declared_version_is_registered():
    def two_v1(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    @node("two-reg", title="Two", description="d", category="test", replaces=node("two-reg")(two_v1), upgrade=Upgrade())
    def two(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    registry = NodeRegistry([two])

    assert set(registry["two-reg"].versions) == {1, 2}


def test_the_registry_keys_on_the_id():
    @node("echo-key", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    registry = NodeRegistry([echo])

    assert "echo-key" in registry
    assert "echo-nothing" not in registry
    assert registry.nodes == (echo,)


def test_the_flattened_record_is_gone():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conductor._internal.registry.definition")




def test_a_hook_that_cannot_answer_raises_refuses():
    """`CompileProblem(code, message)` is the one refusal a field hook has: the
    host names the code and writes the sentence, and the compiler anchors both
    as the node's fatal `Problem` (the graph plans)."""
    from conductor import ComputeHookError
    from conductor._internal.errors import ConductorError

    refusal = ComputeHookError("What arrives does not fit.", code="wrong_shape")
    assert (refusal.code, refusal.message) == ("wrong_shape", "What arrives does not fit.")
    assert isinstance(refusal, ConductorError)


def test_a_reduction_declares_a_series_input():
    """The dtype is the only thing that says 'do not broadcast me'."""

    @node("join", title="Join text", description="d", category="test")
    def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("\n".join(texts))

    (texts,) = join_node.versions[1].interface.inputs
    assert texts.dtype is Txt and texts.many




def test_describe_is_the_class_as_a_record():
    def translate_v1(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return text

    @node("describe-translate", title="Translation", description="Translates", category="test", tags=("Language model",), policy=Policy(retries=2), replaces=node("describe-translate")(translate_v1), upgrade=Upgrade())
    def translate(
        text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
        language: Annotated[Txt, Param(title="Language", widget=Dropdown(choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English"))))] = Txt("da"),
    ) -> Out:
        return text

    d = translate.describe()

    assert isinstance(d, NodeDescription)
    assert (d.id, d.title, d.tags) == ("describe-translate", "Translation", ("Language model",))
    assert d.current_version == 2
    assert [i.name for i in d.versions[2].interface.inputs] == ["text", "language"]
    assert d.versions[2].policy.retries == 2
    assert d.versions[1].interface.outputs[0].name == "result"
    assert d.versions[1].open_input_type is None
    assert not hasattr(d, "role")


def test_describe_carries_each_versions_notice():
    """Two scopes, two true things. The palette renders both when both
    exist, node first — a precedence rule would show strictly less."""

    def old_node_v1(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    @node("describe-old", title="Old", description="d", category="test", deprecation=Deprecation(header="Node retired"), replaces=node("describe-old", deprecation=Deprecation(header="v1 retired", migration="Use v2."))(old_node_v1), upgrade=Upgrade())
    def old_node(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    d = old_node.describe()

    assert d.versions[1].deprecation == Deprecation(header="v1 retired", migration="Use v2.")
    assert d.versions[2].deprecation == Deprecation(header="Node retired")
    data = TypeAdapter(NodeDescription).dump_python(d, mode="json")
    assert "deprecation" not in data, "one level: the newest version's notice is the node's"
    assert "deprecated" not in data


def test_describe_dumps_as_the_palette_payload():
    @node("describe-echo", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea(rows=2))] = Txt("")) -> Out:
        return x

    data = TypeAdapter(NodeDescription).dump_python(echo.describe(), mode="json")

    assert data["id"] == "describe-echo"
    assert data["category"] == "test"
    assert data["versions"]["1"]["interface"]["inputs"][0]["widget"] == {
        "type": "textarea", "rows": 2,
    }
    assert data["versions"]["1"]["interface"]["inputs"][0]["dtype"] == {"id": "node-class-test-txt"}
    assert data["versions"]["1"]["interface"]["outputs"][0]["dtype"] == {"id": "node-class-test-txt"}
    for gone in ("base_id", "width", "is_decision", "has_dynamic_outputs", "latest_version", "deprecated"):
        assert gone not in data


def test_describe_publishes_a_json_schema():
    schema = TypeAdapter(NodeDescription).json_schema(mode="serialization")

    assert "Input" in schema["$defs"]
    assert "Output" in schema["$defs"]
    assert "Policy" in schema["$defs"]


def test_describe_is_computed_not_stored():
    @node("describe-echo-2", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert echo.describe() == echo.describe()
    assert "description_record" not in vars(echo)




class Flag(DType):
    """A boolean dtype for the test — conductor ships none."""

    id = "node-class-test-flag"
    display_title = "Yes/No"

    def __init__(self, value: bool = False) -> None:
        self.value = bool(value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Flag) and other.value == self.value


def test_a_category_is_a_string_on_the_definition():
    """A category is presentation: where the palette files the node. No
    object, no registration side-channel."""
    import conductor

    @node("echo-cat", title="Echo", description="Returns input", category="tools")
    def echo(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return text

    assert echo.category == "tools"
    assert echo.describe().category == "tools"
    assert not hasattr(conductor, "NodeCategory")
    assert not hasattr(NodeRegistry, "include")


def test_a_class_node_executes_in_a_graph():
    def _decorate(text: str) -> str:
        return text.upper() + "!"

    @node("shout", title="Shout", description="d", category="test")
    def shout(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(_decorate(text))

    registry = NodeRegistry([shout])

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="shout", version=1, bindings={"text": Static("hi")})]), registry)
    results = compiled.results(compiled.run_sync())
    assert results["a"]["result"] == "HI!"


def test_two_versions_of_one_class_execute_independently():
    def suffix_v1(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return text

    @node("suffix", title="Suffix", description="d", category="test", replaces=node("suffix")(suffix_v1), upgrade=Upgrade())
    def suffix(
        text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
        mark: Annotated[Txt, Param(title="Mark", widget=Textarea())] = Txt("?"),
    ) -> Out:
        return Txt(text + mark)

    registry = NodeRegistry([suffix])

    for pinned, expected in ((1, "hi"), (2, "hi?")):
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="suffix", version=pinned, bindings={"text": Static("hi")})]), registry)
        assert compiled.results(compiled.run_sync())["a"]["result"] == expected


def test_a_version_a_class_does_not_declare_is_refused():
    """The pin is what a graph selects with, so a graph naming a version
    that is not there must not resolve to the nearest one."""

    @node("once-exec", title="Once", description="d", category="test")
    def once(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    registry = NodeRegistry([once])
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="a", type="once-exec", version=2),
        GraphNode(id="b", type="never-registered", version=1),
    ]), registry)

    assert [p.type for p in compiled.problems] == ["unknown_node_version", "unknown_node_type"]
    assert not hasattr(NodeRegistry, "runner_for")


# --- a node is a function ---------------------------------------------------------


def _x(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
    return x


def test_a_node_is_called_like_its_function():
    @node("upper")
    def upper(text: Txt) -> Txt:
        """Upper-cases a text."""
        return Txt(text.upper())

    assert upper(Txt("a")) == "A"
    assert (upper.id, upper.title, upper.description, upper.category, upper.current_version) == ("upper", "Upper", "Upper-cases a text.", "", 1)
    (input_,) = upper.versions[1].interface.inputs
    (out,) = upper.versions[1].interface.outputs
    assert (input_.name, input_.dtype, input_.title, input_.required) == ("text", Txt, "Text", True)
    assert (out.name, out.dtype, out.title) == ("result", Txt, "Result")


def test_a_title_defaults_to_the_function_name_read_as_words():
    @node("x-y")
    def split_lines(x: Txt) -> Txt:
        return x

    assert split_lines.title == "Split lines"


def test_a_sink_node_declares_no_outputs_and_runs():
    seen = []

    @node("sink")
    def sink(text: Txt) -> None:
        seen.append(text)

    registry = NodeRegistry([sink])
    assert sink.versions[1].interface.outputs == ()
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="sink", version=1, bindings={"text": Static("hi")})]), registry)
    assert compiled.problems == ()
    assert compiled.run_sync().status == "complete" and seen == ["hi"]


def test_a_result_can_name_the_output():
    @node("n")
    def named(x: Txt) -> Annotated[Txt, Result(name="summary")]:
        return x

    assert [o.name for o in named.versions[1].interface.outputs] == ["summary"]


def test_a_version_number_is_its_place_and_each_keeps_its_policy():
    @node("two", replaces=node("two", replaces=node("two", policy=Policy(retries=2))(_x))(_x), policy=Policy(retries=5))
    def two(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return Txt(x + "!")

    assert set(two.versions) == {1, 2, 3} and two.current_version == 3
    assert [two.versions[n].policy.retries for n in (1, 2, 3)] == [2, 0, 5]
    assert two.versions[1].body is _x and two(Txt("a")) == "a!"


def test_an_older_version_carries_its_own_notice():
    @node("two", replaces=node("two", deprecation=Deprecation(header="Old"))(_x), deprecation=None)
    def two(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Out:
        return x

    assert two.versions[1].deprecation.header == "Old" and two.versions[2].deprecation is None


def test_an_async_function_is_refused():
    with pytest.raises(NodeDeclarationError, match="async") as raised:
        @node("waits")
        async def waits(x: Txt) -> Txt:
            return x
    assert raised.value.code == "async_run" and "waits" in str(raised.value)


def test_a_definition_is_frozen():
    @node("fixed")
    def fixed(x: Txt) -> Txt:
        return x

    with pytest.raises(FrozenInstanceError):
        fixed.title = "Other"


# --- the registry is built whole --------------------------------------------------


def test_two_definitions_with_one_id_are_refused():
    @node("same")
    def first(x: Txt) -> Txt:
        return x

    @node("same")
    def second(x: Txt) -> Txt:
        return x

    with pytest.raises(NodeDeclarationError, match="declared twice"):
        NodeRegistry([first, second])


def test_something_that_is_not_a_node_is_refused():
    with pytest.raises(NodeDeclarationError, match="declare one with @node"):
        NodeRegistry([_x])


def test_an_alternative_may_come_later_in_the_list_and_must_exist():
    @node("old", deprecation=Deprecation(alternative="new"))
    def old(x: Txt) -> Txt:
        return x

    @node("new")
    def new(x: Txt) -> Txt:
        return x

    assert len(NodeRegistry([old, new])) == 2
    with pytest.raises(NodeDeclarationError, match="alternative"):
        NodeRegistry([old])


# --- hooks read what their signature names ----------------------------------------


def test_a_hook_gets_the_typed_value_the_declared_default_included():
    def header_outputs(header: Txt) -> list[Output]:
        return [Output(name, Txt) for name in header.split(",")]

    @node("sheet", compute_outputs=header_outputs)
    def sheet(header: Annotated[Txt, Param(connectable=False)] = Txt("a,b")) -> Mapping[str, Any]:
        return {name: Txt(name) for name in header.split(",")}

    registry = NodeRegistry([sheet])
    default = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="sheet", version=1)]), registry)
    typed = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="sheet", version=1, bindings={"header": Static("c")})]), registry)
    assert [o.name for o in default.node("s").interface.outputs] == ["a", "b"]
    assert [o.name for o in typed.node("s").interface.outputs] == ["c"]


def test_a_hook_is_not_called_when_a_value_it_needs_has_none():
    def names_outputs(names: Txt) -> list[Output]:
        raise AssertionError("not called")

    @node("n", compute_outputs=names_outputs)
    def needs_names(names: Txt) -> Annotated[Any, Result(title="R")]:
        return names

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="n", version=1)]), NodeRegistry([needs_names]))
    assert [o.name for o in compiled.node("n").interface.outputs] == ["result"]


def test_compute_added_inputs_adds_to_what_the_node_declares():
    def more_inputs(mode: Txt) -> list[Input]:
        return [] if mode == "a" else [Input("extra", Txt, widget=Textarea(), default=Txt(""))]

    @node("n", compute_added_inputs=more_inputs)
    def modes(mode: Txt = Txt("a"), **more: Txt) -> Txt:
        return mode

    registry = NodeRegistry([modes])
    def inputs(mode):
        graph = Graph(nodes=[GraphNode(id="n", type="n", version=1, bindings={"mode": Static(mode)})])
        return [i.name for i in CompiledGraph.from_graph(graph, registry).node("n").interface.inputs]

    assert inputs("a") == ["mode"] and inputs("b") == ["mode", "extra"]


def test_each_version_has_its_own_hooks():
    def v1_outputs() -> list[Output]:
        return [Output("one", Txt)]

    def v2_outputs() -> list[Output]:
        return [Output("two", Txt)]

    def older_run(x: Txt) -> Mapping[str, Any]:
        return {"one": x}

    @node("n", compute_outputs=v2_outputs, replaces=node("n", compute_outputs=v1_outputs)(older_run), upgrade=Upgrade(renamed_outputs={"one": "two"}))
    def newer(x: Txt) -> Mapping[str, Any]:
        return {"two": x}

    registry = NodeRegistry([newer])
    def outs(v):
        graph = Graph(nodes=[GraphNode(id="n", type="n", version=v, bindings={"x": Static("a")})])
        return [o.name for o in CompiledGraph.from_graph(graph, registry).node("n").interface.outputs]

    assert outs(1) == ["one"] and outs(2) == ["two"]


@pytest.mark.parametrize("hook, code", [
    (lambda nope: [], "hook_parameter_unknown"),
    (lambda *a: [], "hook_parameter_unknown"),
    (lambda **k: [], "hook_parameter_unknown"),
])
def test_a_hook_parameter_must_name_an_input(hook, code):
    with pytest.raises(NodeDeclarationError) as raised:
        def run(x: Txt) -> Mapping[str, Any]:
            return {}
        node("n", compute_outputs=hook)(run)
    assert raised.value.code == code


def test_a_mapping_or_an_any_output_needs_a_hook():
    def mapping(x: Txt) -> Mapping[str, Any]:
        return {}

    def anything(x: Txt) -> Any:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        node("n")(mapping)
    assert raised.value.code == "mapping_return_without_hook"
    with pytest.raises(NodeDeclarationError) as raised:
        node("n")(anything)
    assert raised.value.code == "any_output_without_hook"


def test_compute_added_inputs_needs_somewhere_for_its_inputs_to_land():
    with pytest.raises(NodeDeclarationError) as raised:
        node("n", compute_added_inputs=lambda: [])(_x)
    assert raised.value.code == "compute_added_inputs_without_var_keyword"


def test_as_it_arrives_beside_compute_added_inputs_is_refused():
    def run(**inputs: AsItArrives) -> Txt:
        return Txt("")

    with pytest.raises(NodeDeclarationError) as raised:
        node("n", compute_added_inputs=lambda: [])(run)
    assert raised.value.code == "as_it_arrives_beside_compute_added_inputs"


def test_an_unknown_annotation_extra_is_refused():
    def run(x: Annotated[Txt, object()]) -> Txt:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        node("n")(run)
    assert raised.value.code == "annotated_extra_unknown"


def test_a_definition_may_hold_graphs_by_number_with_holes():
    @node("shout")
    def shout(text: Annotated[Txt, Param(widget=Textarea())] = Txt("")) -> Out:
        return text

    inner = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="inner", type="shout", version=1)]), NodeRegistry([shout]))
    wrapped = NodeDefinition.from_compiled_graphs("wrapped", title="Embedded", description="d", versions={3: inner})

    assert set(wrapped.versions) == {3} and wrapped.current_version == 3
    assert [i.name for i in wrapped.versions[3].interface.inputs] == ["text"]
    described = wrapped.describe().versions[3]
    assert [i.name for i in described.interface.inputs] == ["text"]
    assert described.policy is None and described.deprecation is None
    assert "wrapped" in NodeRegistry([wrapped])
    with pytest.raises(TypeError, match="graph"):
        wrapped()


def test_the_contract_is_importable_from_the_package_root():
    import conductor

    for name in ("node", "NodeDefinition", "Upgrade", "Policy", "Deprecation", "NodeRegistry", "upgrade_node", "ArrivingType", "AsItArrives", "FromRun"):
        assert hasattr(conductor, name)


def test_arriving_is_the_edges_type_or_the_typed_in_values():
    class Num(DType, float):
        id = "node-class-num"

    def echoed_outputs(x: ArrivingType) -> list[Output]:
        return [Output("result", x)]

    @node("echoed", compute_outputs=echoed_outputs)
    def echoed(x: Txt) -> Mapping[str, Any]:
        return {"result": x}

    @node("num")
    def num() -> Num:
        return Num(1)

    registry = NodeRegistry([echoed, num])
    typed_in = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="echoed", version=1, bindings={"x": Static("a")})]), registry)
    nothing = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="echoed", version=1)]), registry)

    assert typed_in.node("n").interface.outputs[0].dtype is Txt
    assert nothing.node("n").interface.outputs[0].dtype.id == "any"
