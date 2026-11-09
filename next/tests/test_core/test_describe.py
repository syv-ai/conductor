"""``compiled.describe()``: what an editor reads to draw a compiled graph, as plain JSON."""

import json
from typing import Annotated

from conductor import (
    CompiledGraph,
    From,
    FromRun,
    Graph,
    GraphNode,
    Input,
    NodeDefinition,
    NodeRegistry,
    Param,
    Ref,
    Series,
    Static,
    node,
)


class Caller:
    pass


@node("shout")
def shout(text: str, caller: FromRun[Caller]) -> str:
    return text.upper()


@node("split")
def split(text: str) -> Series[str]:
    return text.split()


def _extra_input(language: str) -> list[Input]:
    return [Input(language, str, title=language.title())]


@node("translate", compute_added_inputs=_extra_input)
def translate(language: Annotated[str, Param(connectable=False)] = "da", **texts: str) -> str:
    return language


_inner = NodeRegistry([shout])
_embedded = NodeDefinition.from_compiled_graphs(
    "wrapped", title="Wrapped", description="d", category="test",
    versions={1: CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="all", type="shout", version=1)]), _inner)},
)
_registry = NodeRegistry([shout, split, translate, _embedded])


def _compiled() -> CompiledGraph:
    return CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="emb", type="wrapped", version=1),
            GraphNode(id="words", type="split", version=1, bindings={"text": From(Ref("emb", "result"))}),
            GraphNode(id="hook", type="translate", version=1, bindings={"language": Static("en"), "en": From(Ref("words", "result"))}),
            GraphNode(id="gone", type="nothing-here", version=1),
        ]),
        _registry,
    )


def test_a_description_is_plain_json_and_says_what_compile_said():
    compiled = _compiled()
    described = compiled.describe()

    written = json.loads(described.model_dump_json(), parse_constant=lambda token: pytest_fail(token))

    assert written == described.model_dump(mode="json")
    assert written["is_runnable"] is compiled.is_runnable
    assert [problem["type"] for problem in written["problems"]] == [problem.type for problem in compiled.problems]
    assert written["run_plan"] == compiled.run_plan.model_dump(mode="json")
    assert written["interface"]["needs"] == ["Caller"]


def pytest_fail(token: str) -> None:
    raise AssertionError(f"{token} is not JSON")


def test_every_key_of_fields_is_an_address_compiled_field_accepts():
    compiled = _compiled()
    described = compiled.describe()

    assert set(described.fields) == {"emb.text", "emb.result", "words.text", "words.result", "hook.language", "hook.en", "hook.result"}
    for address, field in described.fields.items():
        assert field.dtype == compiled.field(address).type.describe()
        assert field.many is compiled.field(address).many
    assert described.fields[Ref("words", "result")].many is True
    assert described.fields[Ref("emb", "result")].many is False


def test_an_address_on_a_node_that_is_a_graph_is_the_field_inside_it():
    compiled = _compiled()

    assert compiled.field("emb.result") is compiled.field("emb/all.result")
    assert compiled.field("emb.text") is compiled.field("emb/all.text")


def test_a_node_compile_could_not_resolve_is_there_with_its_state_alone():
    described = _compiled().describe()

    assert list(described.nodes) == ["emb", "words", "hook", "gone"]
    assert described.nodes["gone"].state == "resolution_failed"
    assert described.nodes["gone"].interface is None
    assert not any(address.node_id == "gone" for address in described.fields)


def test_a_node_says_whether_the_graph_takes_inputs_on_it_and_a_hook_changes_its_fields():
    described = _compiled().describe()

    assert described.nodes["emb"].takes_graph_inputs is True
    assert described.nodes["words"].takes_graph_inputs is False
    hook = described.nodes["hook"]
    assert hook.state == "ready" and [input_.name for input_ in hook.interface.inputs] == ["language", "en"]
    assert described.nodes["emb"].interface.needs == ("Caller",)
