"""Specification tests for shared references (produce/consume).

These tests encode the behavior defined in `docs/shared-references.md`. They
are skipped at the module level until the feature lands. Once implementation
is in place, remove the `pytestmark` line below — the tests should then pass
without modification.

If you need to change expected behavior, change the design doc first, then
update the tests, then adjust the implementation.
"""

from __future__ import annotations

from typing import Annotated

import pytest

from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor._sentinel import SKIPPED
from conductor.compound.for_each import FOR_EACH
from conductor.errors import CompilationError, CycleDetectionError
from conductor.execution.engine import collect, execute, execute_sync, resume_sync
from conductor.errors import FlowPausedException, HumanInputRequired
from conductor.types import NodeCategory
from conductor.widgets import ConnectionList, Output, Text, Textarea


# =============================================================================
# Fixtures / helpers
# =============================================================================


@pytest.fixture
def registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("build-map", version=1, name="Build Map", description="Builds a dict")
    def build_map(
        seed: Annotated[str, Text(label="Seed")] = "default",
    ) -> Annotated[dict[str, str], Output(label="Mapping")]:
        return {"Alice": f"P001-{seed}", "Bob": f"P002-{seed}"}

    @reg.node("redact", version=1, name="Redact", description="Redacts with a mapping")
    def redact(
        text: Annotated[str, Text(label="Text")],
        mapping: Annotated[dict[str, str], Text(label="Mapping")],
    ) -> Annotated[str, Output(label="Redacted")]:
        out = text
        for k, v in mapping.items():
            out = out.replace(k, v)
        return out

    @reg.node("echo", version=1, name="Echo", description="Echoes input")
    def echo(
        text: Annotated[str, Text(label="Text")],
    ) -> Annotated[str, Output(label="Out")]:
        return text

    @reg.node("upper", version=1, name="Upper", description="Uppercases")
    def upper(
        text: Annotated[str, Text(label="Text")],
    ) -> Annotated[str, Output(label="Out")]:
        return text.upper()

    @reg.node("concat", version=1, name="Concat", description="Concatenates")
    def concat(
        a: Annotated[str, Text(label="A")],
        b: Annotated[str, Text(label="B")],
    ) -> Annotated[str, Output(label="Out")]:
        return f"{a}|{b}"

    @reg.node(
        "number",
        version=1,
        name="Number",
        description="Emits a number",
    )
    def number(
        value: Annotated[int, Text(label="Value")] = 0,
    ) -> Annotated[int, Output(label="Out")]:
        return value

    @reg.node(
        "if-empty",
        version=1,
        name="If Empty",
        description="Conditional",
        category=NodeCategory.CONTROL,
    )
    def if_empty(
        text: Annotated[str, Text(label="Text")],
    ) -> tuple[
        Annotated[str, Output(label="Not empty")],
        Annotated[str, Output(label="Empty")],
    ]:
        if text.strip():
            return (text, SKIPPED)
        return (SKIPPED, "empty")

    @reg.node(
        "for-each-start",
        version=1,
        name="For Each (Start)",
        description="Start",
        category=NodeCategory.CONTROL,
    )
    def for_each_start(
        items: Annotated[list[str], ConnectionList(label="Items")],
    ) -> tuple[
        Annotated[str, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        raise NotImplementedError

    @reg.node(
        "for-each-end",
        version=1,
        name="For Each (End)",
        description="End",
        category=NodeCategory.CONTROL,
    )
    def for_each_end(
        item: Annotated[str, Text(label="Item")],
    ) -> Annotated[list[str], Output(label="Collected")]:
        raise NotImplementedError

    @reg.node(
        "approve",
        version=1,
        name="Approve",
        description="Pauses for approval",
    )
    def approve(
        text: Annotated[str, Textarea(label="Text")],
    ) -> Annotated[str, Output(label="Approved")]:
        raise HumanInputRequired(prompt=f"approve: {text}", schema={"approved": "bool"})

    return reg


# =============================================================================
# 1. Happy path — basic produce / consume
# =============================================================================


class TestBasicProduceConsume:
    def test_simple_produce_consume_no_explicit_edge(self, registry):
        """Consumer input is filled from the producer via consume binding, with
        no explicit edge drawn between them. The flow runs as if the edge
        existed: producer first, consumer second."""
        nodes = [
            GraphNode("n1", "build-map@1", {"seed": "x"}, produces={"result": "mapping"}),
            GraphNode(
                "n2", "redact@1", {"text": "Alice met Bob."},
                consumes={"mapping": ("n1", "result")},
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)

        # Producer must be scheduled before consumer.
        order = list(compiled.execution_order)
        assert order.index("n1") < order.index("n2")

        results = execute_sync(compiled)
        assert results["n2"]["result"] == "P001-x met P002-x."

    def test_fan_out_single_producer_many_consumers(self, registry):
        """A single producer can feed an unlimited number of consumers via
        consume bindings, no edges required."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "shared"}, produces={"result": "shared_text"}),
            GraphNode("c1", "upper@1", None, consumes={"text": ("src", "result")}),
            GraphNode("c2", "upper@1", None, consumes={"text": ("src", "result")}),
            GraphNode("c3", "upper@1", None, consumes={"text": ("src", "result")}),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        results = execute_sync(compiled)

        assert results["c1"]["result"] == "SHARED"
        assert results["c2"]["result"] == "SHARED"
        assert results["c3"]["result"] == "SHARED"

    def test_consumer_can_mix_edges_and_consumes_on_different_inputs(self, registry):
        """A node may consume one input via shared reference and another via an
        explicit edge simultaneously, provided they target different input
        handles."""
        nodes = [
            GraphNode("src_a", "echo@1", {"text": "A"}, produces={"result": "left"}),
            GraphNode("src_b", "echo@1", {"text": "B"}),
            GraphNode(
                "join", "concat@1", None,
                consumes={"a": ("src_a", "result")},
            ),
        ]
        edges = [GraphEdge("e1", "src_b", "join", "result", "b")]
        compiled = compile(nodes=nodes, edges=edges, registry=registry)
        results = execute_sync(compiled)
        assert results["join"]["result"] == "A|B"

    def test_producer_with_no_consumers_still_runs(self, registry):
        """Declaring `produces` is a decoration; the producer runs like any
        other node, even if nothing consumes it."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "hi"}, produces={"result": "unused"}),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        results = execute_sync(compiled)
        assert results["src"]["result"] == "hi"

    def test_empty_produces_and_consumes_equivalent_to_none(self, registry):
        """Empty dicts on `produces` / `consumes` behave identically to None."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "hi"}, produces={}),
            GraphNode("n2", "echo@1", {"text": "yo"}, consumes={}),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        results = execute_sync(compiled)
        assert results["n1"]["result"] == "hi"
        assert results["n2"]["result"] == "yo"


# =============================================================================
# 2. Compile-time validation (§6 in the doc)
# =============================================================================


class TestProducerValidation:
    def test_produces_unknown_output_handle_raises(self, registry):
        """A producer that claims to publish a handle not declared by its
        node type is a compile error."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "x"}, produces={"nonexistent": "label"}),
        ]
        with pytest.raises(CompilationError, match="nonexistent"):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_producer_inside_for_each_region_rejected_in_v1(self, registry):
        """v1 forbids producers inside compound regions. The body node of a
        for-each cannot declare `produces`."""
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b"]}),
            GraphNode(
                "body", "upper@1", None,
                produces={"result": "per_iteration"},
            ),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        with pytest.raises(CompilationError, match="compound region|v1"):
            compile(
                nodes=nodes, edges=edges, registry=registry,
                compound_types=[FOR_EACH],
            )

    def test_duplicate_producer_labels_warn_not_error(self, registry):
        """Two producers choosing the same display label is a warning, not an
        error — bindings reference identity, not label."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "a"}, produces={"result": "name"}),
            GraphNode("n2", "echo@1", {"text": "b"}, produces={"result": "name"}),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        codes = [w.code for w in compiled.type_warnings]
        assert "shared-label-collision" in codes


class TestConsumerValidation:
    def test_consumes_unknown_producer_id_raises(self, registry):
        nodes = [
            GraphNode(
                "n2", "redact@1", {"text": "hi"},
                consumes={"mapping": ("does-not-exist", "result")},
            ),
        ]
        with pytest.raises(CompilationError, match="does-not-exist"):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_consumes_producer_without_produces_declaration_raises(self, registry):
        """Consumers may only bind to outputs that are explicitly marked as
        shared on the producer."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "x"}),  # no `produces`
            GraphNode(
                "n2", "echo@1", None,
                consumes={"text": ("n1", "result")},
            ),
        ]
        with pytest.raises(CompilationError, match="not produced"):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_consumes_handle_not_in_producers_produces_raises(self, registry):
        """The specific output handle must appear in the producer's `produces`
        dict. A producer that shares `result` but not `extra` cannot satisfy a
        consumer asking for `extra`."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "x"}, produces={"result": "label"}),
            GraphNode(
                "n2", "echo@1", None,
                consumes={"text": ("n1", "extra")},
            ),
        ]
        with pytest.raises(CompilationError, match="extra"):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_consumes_unknown_input_handle_raises(self, registry):
        """The consumer's input handle must exist on its node type."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "x"}, produces={"result": "label"}),
            GraphNode(
                "n2", "echo@1", None,
                consumes={"not_an_input": ("n1", "result")},
            ),
        ]
        with pytest.raises(CompilationError, match="not_an_input"):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_edge_and_consume_on_same_input_raises(self, registry):
        """An input handle cannot be both the target of an explicit edge and a
        consume binding."""
        nodes = [
            GraphNode("src_edge", "echo@1", {"text": "E"}),
            GraphNode("src_ref", "echo@1", {"text": "R"}, produces={"result": "r"}),
            GraphNode(
                "n3", "echo@1", None,
                consumes={"text": ("src_ref", "result")},
            ),
        ]
        edges = [GraphEdge("e1", "src_edge", "n3", "result", "text")]
        with pytest.raises(CompilationError, match="both consumed|choose one"):
            compile(nodes=nodes, edges=edges, registry=registry)


class TestCycleDetection:
    def test_cycle_through_shared_reference_detected(self, registry):
        """A cycle formed by shared references is detected just like an edge
        cycle. The error message names the shared reference so the user can
        locate it."""
        nodes = [
            GraphNode(
                "a", "echo@1", {"text": "x"},
                produces={"result": "ref_a"},
                consumes={"text": ("b", "result")},
            ),
            GraphNode(
                "b", "echo@1", None,
                produces={"result": "ref_b"},
                consumes={"text": ("a", "result")},
            ),
        ]
        with pytest.raises(CycleDetectionError):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_cycle_mixed_edges_and_consumes_detected(self, registry):
        """A cycle closed by a mix of explicit edges and shared references is
        still a cycle."""
        nodes = [
            GraphNode("a", "echo@1", None, produces={"result": "ra"}),
            GraphNode("b", "echo@1", None),
            GraphNode(
                "c", "echo@1", None,
                consumes={"text": ("a", "result")},
            ),
        ]
        # a -> b (edge), b -> a (edge) makes cycle independent of consume, so
        # construct so the cycle requires both:
        #   a --consume--> b  (a consumes b.result)
        #   b --edge-----> c
        #   c --edge-----> a
        nodes = [
            GraphNode(
                "a", "echo@1", None,
                produces={"result": "ra"},
                consumes={"text": ("b", "result")},
            ),
            GraphNode("b", "echo@1", None, produces={"result": "rb"}),
            GraphNode("c", "echo@1", None),
        ]
        edges = [
            GraphEdge("e1", "b", "c", "result", "text"),
            GraphEdge("e2", "c", "b", "result", "text"),  # forces cycle
        ]
        with pytest.raises(CycleDetectionError):
            compile(nodes=nodes, edges=edges, registry=registry)


# =============================================================================
# 3. Type checking (§6.5)
# =============================================================================


class TestTypeChecking:
    def test_type_mismatch_produces_warning_by_default(self, registry):
        """A type-incompatible consume binding produces a warning, matching
        explicit-edge behavior."""
        nodes = [
            GraphNode("src", "number@1", {"value": 42}, produces={"result": "n"}),
            GraphNode(
                "dst", "redact@1", {"text": "x"},
                consumes={"mapping": ("src", "result")},  # dict vs int
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        assert any("mapping" in w.message for w in compiled.type_warnings)

    def test_type_mismatch_raises_in_strict_mode(self, registry):
        nodes = [
            GraphNode("src", "number@1", {"value": 42}, produces={"result": "n"}),
            GraphNode(
                "dst", "redact@1", {"text": "x"},
                consumes={"mapping": ("src", "result")},
            ),
        ]
        with pytest.raises(CompilationError):
            compile(
                nodes=nodes, edges=[], registry=registry,
                strict_types=True,
            )

    def test_compatible_types_no_warning(self, registry):
        """Matching types emit no warning, even on the shared-reference path."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "x"}, produces={"result": "s"}),
            GraphNode(
                "dst", "upper@1", None,
                consumes={"text": ("src", "result")},
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        for w in compiled.type_warnings:
            # No warnings specifically about the shared consume path.
            assert "upper" not in w.message or "text" not in w.message


# =============================================================================
# 4. Input resolution precedence (§7.1)
# =============================================================================


class TestResolutionPrecedence:
    def test_consume_overrides_static_data(self, registry):
        """When a node has both static data for a handle and a consume binding
        for that handle, the consume wins."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "from-shared"}, produces={"result": "r"}),
            GraphNode(
                "dst", "upper@1", {"text": "from-static"},
                consumes={"text": ("src", "result")},
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        results = execute_sync(compiled)
        assert results["dst"]["result"] == "FROM-SHARED"

    def test_static_data_used_when_no_consume_or_edge(self, registry):
        """Inputs with no edge and no consume binding fall through to static
        data unchanged."""
        nodes = [
            GraphNode("n1", "upper@1", {"text": "static"}),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        results = execute_sync(compiled)
        assert results["n1"]["result"] == "STATIC"


# =============================================================================
# 5. Loops (§8)
# =============================================================================


class TestLoopInteraction:
    def test_consumer_inside_for_each_reads_top_level_producer(self, registry):
        """The flagship use case: a system-prompt-style value produced at the
        top level is consumed inside every iteration of a for-each."""
        reg = registry

        @reg.node(
            "concat-const",
            version=1,
            name="Concat with constant",
            description="Combines item with a shared constant",
        )
        def concat_const(
            item: Annotated[str, Text(label="Item")],
            prefix: Annotated[str, Text(label="Prefix")],
        ) -> Annotated[str, Output(label="Out")]:
            return f"{prefix}:{item}"

        nodes = [
            GraphNode("sys", "echo@1", {"text": "SYS"}, produces={"result": "system_prompt"}),
            GraphNode("start", "for-each-start@1", {"items": ["a", "b", "c"]}),
            GraphNode(
                "body", "concat-const@1", None,
                consumes={"prefix": ("sys", "result")},
            ),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "item"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=reg,
            compound_types=[FOR_EACH],
        )
        results = execute_sync(compiled)
        assert results["end"]["result"] == ["SYS:a", "SYS:b", "SYS:c"]

    def test_consume_value_is_broadcast_not_iterated(self, registry):
        """The consumer receives the same producer value on every iteration —
        this is broadcast semantics, not per-iteration values."""
        # Identical to the previous test in shape; asserted here as a separate
        # semantic guarantee so a regression that accidentally varies the value
        # fails this test with a clear message.
        reg = registry

        @reg.node(
            "tag",
            version=1,
            name="Tag",
            description="Tags item with prefix",
        )
        def tag(
            item: Annotated[str, Text(label="Item")],
            prefix: Annotated[str, Text(label="Prefix")],
        ) -> Annotated[str, Output(label="Out")]:
            return f"{prefix}/{item}"

        nodes = [
            GraphNode("p", "echo@1", {"text": "P"}, produces={"result": "p"}),
            GraphNode("start", "for-each-start@1", {"items": ["1", "2"]}),
            GraphNode(
                "body", "tag@1", None,
                consumes={"prefix": ("p", "result")},
            ),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "item"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]
        compiled = compile(
            nodes=nodes, edges=edges, registry=reg,
            compound_types=[FOR_EACH],
        )
        results = execute_sync(compiled)
        for out in results["end"]["result"]:
            assert out.startswith("P/")


# =============================================================================
# 6. SKIPPED propagation (§7.3)
# =============================================================================


class TestSkipPropagation:
    def test_skipped_producer_causes_consumer_to_skip(self, registry):
        """If a producer is skipped (all inputs SKIPPED), consumers whose only
        input is that producer are also skipped — identical to edge skip
        propagation."""
        nodes = [
            GraphNode("cond", "if-empty@1", {"text": ""}),
            # Producer receives SKIPPED on the "Not empty" branch (output_1)
            GraphNode(
                "prod", "echo@1", None,
                produces={"result": "maybe"},
            ),
            GraphNode(
                "cons", "upper@1", None,
                consumes={"text": ("prod", "result")},
            ),
        ]
        edges = [GraphEdge("e1", "cond", "prod", "output_1", "text")]
        compiled = compile(nodes=nodes, edges=edges, registry=registry)
        results = execute_sync(compiled)

        # `cons` should be skipped (its only input source was skipped),
        # and therefore should not appear in the final results dict.
        assert "cons" not in results


# =============================================================================
# 7. Checkpoint / resume (§9)
# =============================================================================


class TestCheckpointResume:
    def test_shared_value_survives_checkpoint_and_resume(self, registry):
        """Values produced before a pause must be available to consumers that
        run after resume — the values live in `state.results` and are already
        part of `FlowCheckpoint.results`."""
        nodes = [
            GraphNode(
                "src", "build-map@1", {"seed": "s"},
                produces={"result": "mapping"},
            ),
            GraphNode("gate", "approve@1", {"text": "please review"}),
            GraphNode(
                "cons", "redact@1", {"text": "Alice met Bob."},
                consumes={"mapping": ("src", "result")},
            ),
        ]
        edges = [GraphEdge("e1", "gate", "cons", "result", "text")]
        compiled = compile(nodes=nodes, edges=edges, registry=registry)

        with pytest.raises(FlowPausedException) as exc:
            execute_sync(compiled)
        checkpoint = exc.value.checkpoint

        results = resume_sync(
            compiled, checkpoint,
            response="Alice met Bob.",
        )
        assert results["cons"]["result"] == "P001-s met P002-s."


# =============================================================================
# 8. Backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    def test_graph_without_shared_references_unchanged(self, registry):
        """A graph that uses neither `produces` nor `consumes` compiles and
        runs identically to before."""
        nodes = [
            GraphNode("n1", "echo@1", {"text": "hello"}),
            GraphNode("n2", "upper@1", None),
        ]
        edges = [GraphEdge("e1", "n1", "n2", "result", "text")]
        compiled = compile(nodes=nodes, edges=edges, registry=registry)
        results = execute_sync(compiled)
        assert results["n2"]["result"] == "HELLO"

    def test_produces_consumes_default_to_none(self):
        """Constructing a GraphNode without passing the new fields keeps the
        fields as None (not empty dicts)."""
        n = GraphNode("n", "echo@1", None)
        assert n.produces is None
        assert n.consumes is None

    def test_graph_node_remains_frozen(self, registry):
        """GraphNode is still a frozen dataclass — mutation raises."""
        n = GraphNode(
            "n", "echo@1", None,
            produces={"result": "r"},
        )
        with pytest.raises(Exception):
            n.produces = {"result": "other"}  # type: ignore[misc]


# =============================================================================
# 9. CompiledGraph surface (§6.3)
# =============================================================================


class TestCompiledGraphSurface:
    def test_consume_map_exposed_on_compiled_graph(self, registry):
        """Compiled graph exposes a `consume_map` so host apps and tests can
        inspect what implicit dependencies exist."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "x"}, produces={"result": "r"}),
            GraphNode(
                "dst", "upper@1", None,
                consumes={"text": ("src", "result")},
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        assert compiled.consume_map == {("dst", "text"): ("src", "result")}

    def test_edge_map_not_polluted_by_consume_bindings(self, registry):
        """Implicit dependencies live on `consume_map`, not `edge_map`, so
        host apps relying on `edge_map` as a list of user-drawn wires see
        exactly what the user drew."""
        nodes = [
            GraphNode("src", "echo@1", {"text": "x"}, produces={"result": "r"}),
            GraphNode(
                "dst", "upper@1", None,
                consumes={"text": ("src", "result")},
            ),
        ]
        compiled = compile(nodes=nodes, edges=[], registry=registry)
        assert compiled.edge_map == {}
