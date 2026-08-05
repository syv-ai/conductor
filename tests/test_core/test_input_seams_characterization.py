"""Behaviour of every input-reading seam, pinned before ``compute_inputs``.

These tests describe what conductor does TODAY with statically declared
inputs. Adding a resolved-input overlay must not change any of it: a node
without a ``compute_inputs`` hook has to behave identically afterwards.

If a change to this file seems necessary, that is a behaviour change and
needs deciding on purpose — not a test edit.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import CompilationError
from conductor.execution.resolver import InputResolver
from conductor.registry.schema import serialize_registry
from conductor.widgets import ConnectionList, Output, Text


def _registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("src", version=1, name="Src", description="Source")
    def src() -> Annotated[str, Output(label="Ud")]:
        return "x"

    @reg.node("sink", version=1, name="Sink", description="Sink")
    def sink(
        text: Annotated[str, Text(label="Tekst")],
        items: Annotated[list[str], ConnectionList(label="Elementer")] = [],  # noqa: B006
    ) -> Annotated[str, Output(label="Ud")]:
        return text

    return reg


class TestStaticInputsAreDeclared:
    def test_signature_becomes_input_metadata(self) -> None:
        nd = _registry().get("sink@1")
        by_name = {i.name: i for i in nd.inputs}
        assert by_name["text"].type_str == "str"
        assert by_name["text"].label == "Tekst"
        assert by_name["items"].expects_list is True
        assert by_name["items"].uses_connection_list is True

    def test_palette_serializes_inputs_without_a_dynamic_flag(self) -> None:
        payload = {n["id"]: n for n in serialize_registry(_registry())}
        assert [i["name"] for i in payload["sink@1"]["inputs"]] == ["text", "items"]
        assert "has_dynamic_inputs" not in payload["sink@1"]


class TestEdgeIntoUnknownInput:
    """An edge into an undeclared input handle is SILENTLY IGNORED today.

    Not rejected, and not even warned about: ``_find_input`` returns None
    and the type check simply skips the edge. Worth knowing, because it
    means dynamic input handles already carry edges without
    ``compute_inputs`` — what the hook adds is that they get type-checked
    instead of waved through.
    """

    def test_an_edge_into_an_undeclared_handle_compiles_clean(self) -> None:
        compiled = compile(
            nodes=[GraphNode("a", "src@1", None), GraphNode("b", "sink@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "nope")],
            registry=_registry(),
        )
        assert "b" in compiled.execution_order
        assert compiled.type_warnings == ()

    def test_compile_accepts_an_edge_into_a_declared_handle(self) -> None:
        compiled = compile(
            nodes=[GraphNode("a", "src@1", None), GraphNode("b", "sink@1", None)],
            edges=[GraphEdge("e1", "a", "b", "result", "text")],
            registry=_registry(),
        )
        assert "b" in compiled.execution_order


class TestConsumeIntoUnknownInput:
    def test_consume_into_undeclared_input_is_a_compilation_error(self) -> None:
        nodes = [
            GraphNode("a", "src@1", None, produces={"result": "Delt"}),
            GraphNode("b", "sink@1", None, consumes={"nope": ("a", "result")}),
        ]
        with pytest.raises(CompilationError, match="unknown input"):
            compile(nodes=nodes, edges=[], registry=_registry())


class TestParamInfoCache:
    def test_param_info_reports_connection_list_and_list_flags(self) -> None:
        resolver = InputResolver(_registry())
        assert resolver._param_info("sink@1", "items") == (True, True)
        assert resolver._param_info("sink@1", "text") == (False, False)

    def test_unknown_param_falls_back_to_false_false(self) -> None:
        resolver = InputResolver(_registry())
        assert resolver._param_info("sink@1", "nope") == (False, False)


class TestVarKeywordAlreadyBypassesFiltering:
    """The runtime half is already solved — pin it so nobody 'fixes' it."""

    def test_var_keyword_function_keeps_undeclared_keys(self) -> None:
        from conductor.execution.engine import _filter_to_signature

        def fn(a: str, **kwargs: Any) -> str:
            return a

        out = _filter_to_signature(fn, {"a": "1", "surprise": "2"})
        assert out == {"a": "1", "surprise": "2"}

    def test_plain_function_drops_undeclared_keys(self) -> None:
        from conductor.execution.engine import _filter_to_signature

        def fn(a: str) -> str:
            return a

        assert _filter_to_signature(fn, {"a": "1", "surprise": "2"}) == {"a": "1"}
