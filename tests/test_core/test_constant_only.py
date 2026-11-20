"""Tests inputs that take only a constant.

Some inputs must be fixed when the graph is built, not fed while it runs. An input declared with
``Param(constant_only=True)`` is one. So is every input a ``compute_...`` hook reads, since the hook runs when the
graph is compiled.

Such an input refuses an edge and a graph input. ``describe()`` tells an editor which inputs these are, and
``make_graph`` never turns one into a graph input.
"""

from typing import Annotated, Any

import pytest
from conductor import (
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Series,
    make_graph,
    node,
)


@node("word", title="Word", description="For a test.")
def word(text: str) -> str:
    return text


@node("pad", title="Pad", description="For a test.")
def pad(text: str, width: Annotated[int, Param(constant_only=True)] = 4) -> str:
    return text.ljust(width)


def _by_width(width: int) -> dict[str, Any]:
    return {f"part{i}": str for i in range(width)}


@node("parts", title="Parts", description="For a test.", compute_outputs=_by_width)
def parts(text: str, width: int) -> dict[str, Any]:
    return {f"part{i}": text for i in range(width)}


def _registry(*definitions: Any) -> NodeRegistry:
    registry = NodeRegistry()
    registry.add(*definitions)
    return registry


REGISTRY = _registry(word, pad, parts)


def _problems(
    inputs: dict[str, Any], node_id: str = "pad", declared: dict[str, Any] | None = None
) -> list[tuple[str, tuple[str | int, ...]]]:
    record = GraphRecord.model_validate(
        {
            "inputs": {"text": {}, **(declared or {})},
            "nodes": {
                "w": {"node": "word", "inputs": {"text": {"graph_input": "text"}}},
                "p": {"node": node_id, "inputs": {"text": {"constant": "a"}, **inputs}},
            },
        }
    )
    return [(p.type, p.loc) for p in Graph.compile(record, REGISTRY).problems]


@pytest.mark.parametrize("node_id", ["pad", "parts"])
@pytest.mark.parametrize(
    ("entry", "declared"),
    [
        ({"sources": [{"node": "w", "output": "result"}]}, {}),
        ({"graph_input": "width"}, {"width": {}}),
        ({"graph_input": "width"}, {"width": {"default": 3}}),
    ],
    ids=["edge", "graph_input", "graph_input_with_default"],
)
def test_an_input_that_takes_only_a_constant_refuses_an_edge_and_a_graph_input(
    node_id: str, entry: dict[str, Any], declared: dict[str, Any]
):
    assert _problems({"width": entry}, node_id, declared) == [
        ("input_takes_only_a_constant", ("p", "width"))
    ]


def test_the_problem_says_what_the_record_gave_and_what_to_give():
    record = GraphRecord.model_validate(
        {
            "inputs": {"w": {}},
            "nodes": {
                "p": {
                    "node": "parts",
                    "inputs": {"text": {"constant": "a"}, "width": {"graph_input": "w"}},
                }
            },
        }
    )
    [problem] = Graph.compile(record, REGISTRY).problems
    assert problem.ctx == {"node_id": "p", "input": "width", "given": "graph_input"}
    assert (
        problem.msg
        == "'width' only takes a constant, not an edge or a graph input. Give it a constant."
    )


def test_a_constant_and_the_declared_default_are_taken():
    assert _problems({"width": {"constant": 2}}) == []
    assert _problems({}) == []
    assert _problems({"width": {"constant": 2}}, "parts") == []


def test_an_input_a_hook_reads_with_no_value_has_no_value_and_is_told_to_take_a_constant():
    [problem] = Graph.compile(
        GraphRecord.model_validate(
            {"nodes": {"p": {"node": "parts", "inputs": {"text": {"constant": "a"}}}}}
        ),
        REGISTRY,
    ).problems
    assert (problem.type, problem.loc) == ("input_has_no_value_constant_only", ("p", "width"))
    assert problem.msg == "'width' gets no value, and it only takes a constant. Give it one."


def test_describe_says_which_inputs_take_only_a_constant():
    for described in (pad.describe(), parts.describe()):
        assert {f.name: f.constant_only for f in described.inputs} == {"text": False, "width": True}
    graph = Graph.compile(make_graph(word, pad), REGISTRY)
    assert [i.field.constant_only for i in graph.describe().nodes["pad"].inputs] == [False, True]
    assert pad.describe().model_dump(mode="json")["inputs"][1]["constant_only"] is True


def test_make_graph_never_makes_it_a_graph_input():
    @node("cut", title="Cut", description="For a test.")
    def cut(text: str, at: Annotated[int, Param(constant_only=True)]) -> str:
        return text[:at]

    record = make_graph(cut, constants={"cut.at": 2})
    graph = Graph.compile(record, _registry(cut))
    assert list(graph.inputs) == ["text"] and graph.run({"text": "abc"}).outputs == {"result": "ab"}
    left_open = make_graph(cut)
    assert list(left_open.nodes["cut"].inputs) == ["text"]
    assert [(p.type, p.loc) for p in Graph.compile(left_open, _registry(cut)).problems] == [
        ("input_has_no_value_constant_only", ("cut", "at"))
    ]


def test_make_graph_never_chains_into_it():
    @node("cut", title="Cut", description="For a test.")
    def cut(at: Annotated[int, Param(constant_only=True)]) -> str:
        return "x" * at

    with pytest.raises(InvalidCallError) as refused:
        make_graph(word, cut)
    problem = refused.value.problems[0]
    assert (problem.type, problem.ctx) == (
        "not_a_chain_constant_only",
        {"node_id": "cut", "before": "word", "inputs": ["at"]},
    )


def test_the_chain_problem_reads_right_for_several_inputs():
    @node("cut", title="Cut", description="For a test.")
    def cut(
        at: Annotated[int, Param(constant_only=True)], to: Annotated[int, Param(constant_only=True)]
    ) -> str:
        return "x" * (to - at)

    with pytest.raises(InvalidCallError) as refused:
        make_graph(word, cut)
    assert refused.value.problems[0].msg == (
        "make_graph can't wire 'cut' after 'word' by itself, because only a constant can give at, to a value. "
        "Give at, to a constant, and pass edges= to make_graph."
    )


Setting = Annotated[int, Param(constant_only=True)]


@pytest.mark.parametrize(
    ("hint", "constant_only"),
    [
        (Annotated[Setting, Param(title="Width")], True),
        (Annotated[Setting, Param(constant_only=False, title="Width")], False),
        (Annotated[Annotated[int, Param(title="Width")], Param(constant_only=True)], True),
    ],
    ids=["alias_wrapped_again", "later_turns_it_off", "later_turns_it_on"],
)
def test_stacked_params_merge_each_argument_the_last_one_given(hint: Any, constant_only: bool):
    def widen(text: str, width: int) -> str:
        return text.ljust(width)

    widen.__annotations__["width"] = hint
    [_, width] = node("widen", title="Widen", description="For a test.")(widen).describe().inputs
    assert (width.title, width.constant_only) == ("Width", constant_only)


def test_a_param_given_none_feeds_one_graph_input_with_one_given_nothing():
    @node("plain", title="Plain", description="For a test.")
    def plain(n: Annotated[int, Param()]) -> int:
        return n

    def said(n: int) -> int:
        return n

    said.__annotations__["n"] = Annotated[int, Param(title=None)]
    said_node = node("said", title="Said", description="For a test.")(said)
    record = GraphRecord.model_validate(
        {
            "inputs": {"n": {}},
            "nodes": {
                "a": {"node": "plain", "inputs": {"n": {"graph_input": "n"}}},
                "b": {"node": "said", "inputs": {"n": {"graph_input": "n"}}},
            },
        }
    )
    assert Graph.compile(record, _registry(plain, said_node)).problems == []


def test_a_series_item_can_take_only_a_constant():
    @node("pick", title="Pick", description="For a test.")
    def pick(text: str, at: Series[Annotated[int, Param(constant_only=True)]]) -> str:
        return "".join(text[i] for i in at)

    [_, at] = pick.describe().inputs
    assert at.constant_only and at.is_series
    record = GraphRecord.model_validate(
        {
            "nodes": {
                "w": {"node": "word", "inputs": {"text": {"constant": "1"}}},
                "p": {
                    "node": "pick",
                    "inputs": {
                        "text": {"constant": "abc"},
                        "at": {"sources": [{"node": "w", "output": "result"}]},
                    },
                },
            }
        }
    )
    assert [(p.type, p.loc) for p in Graph.compile(record, _registry(word, pick)).problems] == [
        ("input_takes_only_a_constant", ("p", "at"))
    ]


def _with_a_setting(count: int) -> dict[str, Any]:
    return {f"setting{i}": Annotated[int, Param(constant_only=True)] for i in range(count)}


@node("set", title="Set", description="For a test.", compute_added_inputs=_with_a_setting)
def settings(count: int, **settings: int) -> int:
    return sum(settings.values())


@pytest.mark.parametrize(
    ("entry", "found"),
    [
        (
            {"sources": [{"node": "w", "output": "result"}]},
            ("input_takes_only_a_constant", "sources"),
        ),
        ({"graph_input": "s"}, ("input_takes_only_a_constant", "graph_input")),
        (None, ("input_has_no_value_constant_only", None)),
        ({"constant": 2}, None),
    ],
    ids=["edge", "graph_input", "nothing", "constant"],
)
def test_an_input_a_hook_adds_can_take_only_a_constant(
    entry: dict[str, Any] | None, found: tuple[str, str | None] | None
):
    inputs = {"count": {"constant": 1}, **({} if entry is None else {"setting0": entry})}
    record = GraphRecord.model_validate(
        {
            "nodes": {
                "w": {"node": "word", "inputs": {"text": {"constant": "a"}}},
                "s": {"node": "set", "inputs": inputs},
            }
        }
    )
    graph = Graph.compile(record, _registry(word, settings))
    assert [(p.type, p.loc, p.ctx.get("given")) for p in graph.problems] == (
        [] if found is None else [(found[0], ("s", "setting0"), found[1])]
    )
    assert [i.field.constant_only for i in graph.describe().nodes["s"].inputs] == [True, True]


def test_a_refused_edge_decides_no_type():
    @node("number", title="Number", description="For a test.")
    def number(text: str) -> int:
        return len(text)

    @node("either", title="Either", description="For a test.")
    def either[T](x: Annotated[T, Param(constant_only=True)], y: T) -> T:
        return y

    record = GraphRecord.model_validate(
        {
            "nodes": {
                "w": {"node": "word", "inputs": {"text": {"constant": "a"}}},
                "n": {"node": "number", "inputs": {"text": {"constant": "a"}}},
                "e": {
                    "node": "either",
                    "inputs": {
                        "x": {"sources": [{"node": "w", "output": "result"}]},
                        "y": {"sources": [{"node": "n", "output": "result"}]},
                    },
                },
            }
        }
    )
    graph = Graph.compile(record, _registry(word, number, either))
    assert [p.type for p in graph.problems] == ["input_takes_only_a_constant"]
    described = graph.describe().nodes["e"]
    assert [f.field.type for f in described.inputs] == ["int", "int"]
    assert [f.type for f in described.outputs] == ["int"]


def test_an_output_cannot_take_only_a_constant():
    def loud(text: str) -> Annotated[str, Param(constant_only=True)]:
        return text

    with pytest.raises(NodeDeclarationError) as refused:
        node("loud", title="Loud", description="For a test.")(loud)
    assert refused.value.problems[0].type == "constant_only_on_output"
