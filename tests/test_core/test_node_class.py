from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.graph.compiler import compile as compile_graph
from conductor.graph.model import GraphNode
from conductor.metadata import Input, Output
from conductor.node import (
    Deprecation,
    NodeDefinition,
    NodeDescription,
    Policy,
    deprecated,
    upgrade,
    version,
)
from conductor.registry import runner_for
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import Choice, ConnectionList, Dropdown, Switch, Textarea
from pydantic import TypeAdapter


class Txt(DType, str):
    id = "node-class-test-txt"
    title = "Text"


#: The single-output return, spelled once so the fixtures below do not each
#: repeat it. It IS a `Result`; a record appears in the multi-output test.
Out = Annotated[Txt, Result(title="R")]


def test_a_node_declares_its_identity_and_implements_run():
    class Greeter(NodeDefinition):
        id = "greeter"
        title = "Greeter"
        description = "Says hello"
        category = "test"

        def run(self, name: Annotated[Txt, Textarea(title="Name")] = Txt("")) -> Out:
            return Txt(f"hi {name}")

    assert Greeter().run(name=Txt("ida")) == "hi ida"


def test_the_interface_is_derived_when_the_class_is_defined():
    """Subclassing is the trigger, so it cannot be forgotten."""

    class Greeter(NodeDefinition):
        id = "greeter-2"
        title = "Greeter"
        description = "d"
        category = "test"

        def run(self, name: Annotated[Txt, Textarea(title="Name")] = Txt("")) -> Out:
            return name

    iface = Greeter.versions[Greeter.current].interface
    assert [i.name for i in iface.inputs] == ["name"]
    assert [h.name for h in iface.outputs] == ["result"]


def test_a_node_without_an_id_fails_at_import_not_at_run():
    with pytest.raises(TypeError, match="id"):

        class Nameless(NodeDefinition):
            title = "No id"
            description = "d"
            category = "test"

            def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
                return x


def test_a_node_without_a_category_fails_at_import():
    """Where the palette files a node is a fact the author states, not
    one a default answers for them."""
    with pytest.raises(TypeError, match="category"):

        class Unfiled(NodeDefinition):
            id = "unfiled"
            title = "Unfiled"
            description = "d"

            def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
                return x


def test_a_param_without_a_widget_fails_at_import():
    """An input without a widget is an error, not a fallback."""
    with pytest.raises(TypeError):

        class Bare(NodeDefinition):
            id = "bare"
            title = "Bare"
            description = "d"
            category = "test"

            def run(self, x: Txt = Txt("")) -> Out:
                return x


def test_a_class_says_nothing_about_what_the_engine_must_do():
    """There is no role. ``SKIPPED`` and ``Asks`` are values a
    ``run`` returns, and the engine acts on the value. A flag per
    capability is the same fact under another name."""

    class Greeter(NodeDefinition):
        id = "greeter-3"
        title = "Greeter"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    for gone in ("role", "is_decision", "is_signal", "is_input", "asks", "dynamic_handles"):
        assert not hasattr(Greeter, gone), gone


def test_the_dead_declaration_fields_are_gone():
    """No node in either repo sets any of these."""
    for gone in ("idempotency_key", "actor", "uses", "width"):
        assert not hasattr(NodeDefinition, gone)


def test_an_intermediate_base_declares_nothing():
    """A shared base that adds no `run` is not a node."""

    class Shared(NodeDefinition):
        category: ClassVar[str] = "io"

    assert not hasattr(Shared, "versions")
    assert not hasattr(Shared, "current")


def test_a_node_may_be_deprecated_and_the_notice_is_content():
    """The record's presence is the fact. Nothing is wrong with a
    graph that places a deprecated node — it is not a Problem."""

    @deprecated(header="Retired", alternative="greeter", migration="Use Greeter instead.")
    class Old(NodeDefinition):
        id = "old"
        title = "Old"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    class Live(NodeDefinition):
        id = "live"
        title = "Live"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Old.deprecation == Deprecation(header="Retired", alternative="greeter", migration="Use Greeter instead.")
    assert Live.deprecation is None
    assert not hasattr(Old, "deprecated"), "no boolean beside the record"


def test_a_bare_deprecated_means_going_away_with_no_details_yet():
    @deprecated()
    class Old(NodeDefinition):
        id = "old-bare"
        title = "Old"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Old.deprecation == Deprecation()




def test_an_undecorated_run_is_version_one():
    class Once(NodeDefinition):
        id = "once"
        title = "Once"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert set(Once.versions) == {1}
    assert Once.current == 1
    assert Once.versions[1].run.__name__ == "run"


def test_versions_live_together_in_one_class():
    class Two(NodeDefinition):
        id = "two"
        title = "Two"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, old: Annotated[Txt, Textarea(title="Old")] = Txt("")) -> Out:
            return old

        @version(2)
        def run(self, new: Annotated[Txt, Textarea(title="New")] = Txt("")) -> Out:
            return new

    assert set(Two.versions) == {1, 2}
    assert Two.current == 2
    assert [i.name for i in Two.versions[1].interface.inputs] == ["old"]
    assert [i.name for i in Two.versions[2].interface.inputs] == ["new"]


def test_the_current_version_is_the_one_called_run():
    class Two(NodeDefinition):
        id = "two-b"
        title = "Two"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

        @version(2)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return Txt(x.upper())

    assert Two.versions[Two.current].run.__name__ == "run"


def test_a_version_with_no_policy_gets_the_default():
    class Plain(NodeDefinition):
        id = "plain-policy"
        title = "Plain"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Plain.versions[1].policy == Policy()
    assert Plain.versions[1].policy.retries == 0
    assert Plain.versions[1].policy.concurrency == 8


def test_policy_is_declared_per_version():
    class Fetch(NodeDefinition):
        id = "fetch"
        title = "Fetch"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=1.0))
        def run_v1(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

        @version(2, policy=Policy(timeout=30.0, concurrency=1))
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Fetch.versions[1].policy.retries == 3
    assert Fetch.versions[2].policy.retries == 0
    assert Fetch.versions[2].policy.timeout == 30.0
    assert Fetch.versions[2].policy.concurrency == 1


def test_a_gap_in_the_versions_is_the_catalogs_rule_not_the_classs():
    """Defining {1, 3} is legal — a loaded definition (an embedded flow whose
    approved versions skip one) is whole as it is. Registering it is not:
    the catalog promises every version up to the current one."""

    class Gapped(NodeDefinition):
        id = "gapped"
        title = "Gapped"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

        @version(3)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert set(Gapped.versions) == {1, 3}
    from conductor import NodeRegistry

    with pytest.raises(ValueError, match="numbers from 1 with no hole"):
        NodeRegistry().register(Gapped)


def test_two_methods_claiming_one_version_are_refused():
    with pytest.raises(TypeError, match="version 1"):

        class Twice(NodeDefinition):
            id = "twice"
            title = "Twice"
            description = "d"
            category = "test"

            @version(1)
            def run_a(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
                return x

            @version(1)
            def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
                return x


def test_a_version_record_does_not_restate_its_number():
    import dataclasses

    from conductor.node import GraphVersion, NodeVersion

    assert [f.name for f in dataclasses.fields(NodeVersion)] == ["run", "interface", "policy", "deprecation"]
    assert [f.name for f in dataclasses.fields(GraphVersion)] == ["graph", "interface"]


def test_a_version_may_be_deprecated_on_its_own():
    """Where the decorator sits is the scope. On a run method the
    notice rides on that NodeVersion, beside its signature and policy —
    a per-version fact filed with the version, not keyed by its number on
    the class."""

    class Two(NodeDefinition):
        id = "two-dep"
        title = "Two"
        description = "d"
        category = "test"

        @deprecated(header="Use v2", migration="The field 'old' is now 'new'.")
        @version(1)
        def run_v1(self, old: Annotated[Txt, Textarea(title="Old")] = Txt("")) -> Out:
            return old

        @version(2)
        def run(self, new: Annotated[Txt, Textarea(title="New")] = Txt("")) -> Out:
            return new

    assert Two.versions[1].deprecation == Deprecation(header="Use v2", migration="The field 'old' is now 'new'.")
    assert Two.versions[2].deprecation is None
    assert Two.deprecation is None
    assert Two.current == 2, "deprecation never moves which version runs"


def test_an_undecorated_run_may_carry_a_deprecation():
    class Once(NodeDefinition):
        id = "once-dep"
        title = "Once"
        description = "d"
        category = "test"

        @deprecated()
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Once.versions[1].deprecation == Deprecation()




def test_the_registry_gives_back_the_class():
    class Translate(NodeDefinition):
        id = "translate"
        title = "Translation"
        description = "Translates text"
        category = "test"

        def run(
            self,
            text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
            language: Annotated[Txt, Dropdown(title="Language", choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English")))] = Txt("da"),
        ) -> Out:
            return Txt(f"{language}:{text}")

    registry = NodeRegistry()
    registry.register(Translate)

    assert registry.get("translate") is Translate


def test_a_caller_asks_the_class_rather_than_a_copy_of_it():
    """There is no record, so there is nothing to keep in step."""

    class Translate(NodeDefinition):
        id = "translate-2"
        title = "Translation"
        description = "d"
        category = "test"

        def run(
            self,
            text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
            language: Annotated[Txt, Dropdown(title="Language", choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English")))] = Txt("da"),
        ) -> Out:
            return text

    registry = NodeRegistry()
    registry.register(Translate)
    found = registry.get("translate-2")

    assert found.title == "Translation"
    iface = found.versions[1].interface
    assert [i.name for i in iface.inputs] == ["text", "language"]
    assert iface.inputs[0].show_handle is True


def test_every_declared_version_is_registered():
    class Two(NodeDefinition):
        id = "two-reg"
        title = "Two"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

        @version(2)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    registry = NodeRegistry()
    registry.register(Two)

    assert set(registry.get("two-reg").versions) == {1, 2}


def test_the_registry_keys_on_the_id():
    class Echo(NodeDefinition):
        id = "echo-key"
        title = "Echo"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    registry = NodeRegistry()
    registry.register(Echo)

    assert registry.contains("echo-key")
    assert not registry.contains("echo-nothing")
    assert registry.definitions() == (Echo,)


def test_registering_the_same_id_twice_is_refused():
    class A(NodeDefinition):
        id = "dupe"
        title = "A"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    class B(NodeDefinition):
        id = "dupe"
        title = "B"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    registry = NodeRegistry()
    registry.register(A)
    with pytest.raises(ValueError, match="dupe"):
        registry.register(B)


def test_a_registered_node_numbers_from_one():
    class Late(NodeDefinition):
        id = "late"
        title = "Late"
        description = "d"
        category = "test"

        @version(3)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert set(Late.versions) == {3}, "a class may declare any contiguous range"
    with pytest.raises(ValueError, match="numbers from 1"):
        NodeRegistry().register(Late)


def test_registering_something_that_is_not_a_node_is_refused():
    registry = NodeRegistry()
    with pytest.raises(TypeError, match="NodeDefinition"):
        registry.register(object)


def test_the_current_version_may_not_retire_alone():
    """A palette offering a version that announces its own death with
    no newer one to move to is incoherent. The class carrying a notice too
    is what makes it coherent."""

    class Dying(NodeDefinition):
        id = "dying"
        title = "Dying"
        description = "d"
        category = "test"

        @deprecated(header="Retired")
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    with pytest.raises(ValueError, match="current"):
        NodeRegistry().register(Dying)

    @deprecated(header="Whole node retired")
    class Retiring(NodeDefinition):
        id = "retiring"
        title = "Retiring"
        description = "d"
        category = "test"

        @deprecated(header="Retired")
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    NodeRegistry().register(Retiring)


def test_an_alternative_names_a_node_in_the_same_catalog():
    """An alternative resolved to nothing in the editor is a silent failure,
    so the registry refuses it — which means the replacement is
    registered before the node it replaces."""

    class New(NodeDefinition):
        id = "new-node"
        title = "New"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    @deprecated(alternative="new-node")
    class Old(NodeDefinition):
        id = "old-node"
        title = "Old"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    with pytest.raises(ValueError, match="new-node"):
        NodeRegistry().register(Old)

    registry = NodeRegistry()
    registry.register(New)
    registry.register(Old)
    assert registry.get("old-node").deprecation.alternative == "new-node"


def test_the_flattened_record_is_gone():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conductor.registry.definition")




def test_a_node_shapes_its_own_outputs_from_its_values():
    """The columns come from the value the node holds."""

    class OpenSheet(NodeDefinition):
        id = "open-sheet"
        title = "Open sheet"
        description = "Exposes a header row as outputs"
        category = "test"

        def run(self, header: Annotated[Txt, Textarea(title="Header")] = Txt("")) -> Out:
            return header

        def compute_outputs(self, declared, values, arriving):
            return tuple(
                Output(name=col, dtype=Txt, title=col)
                for col in str(values.get("header", "")).split(",")
                if col
            )

    declared = OpenSheet.versions[1].interface.outputs
    outputs = OpenSheet().compute_outputs(declared, {"header": "name,email"}, {})
    assert [o.name for o in outputs] == ["name", "email"]


def test_a_node_shapes_its_own_inputs_from_its_values():
    """A roster that depends on a value is the node's own answer."""

    class Modes(NodeDefinition):
        id = "modes"
        title = "Modes"
        description = "A second field only in one mode"
        category = "test"

        def run(
            self,
            mode: Annotated[Txt, Dropdown(title="Mode", choices=(Choice(id="a", title="A"), Choice(id="b", title="B")))] = Txt("a"),
            extra: Annotated[Txt, Textarea(title="Extra")] = Txt(""),
        ) -> Out:
            return mode

        def compute_inputs(self, declared, values):
            if values.get("mode") == "b":
                return declared
            return tuple(i for i in declared if i.name != "extra")

    declared = Modes.versions[1].interface.inputs
    assert [i.name for i in Modes().compute_inputs(declared, {"mode": "a"})] == ["mode"]
    assert [i.name for i in Modes().compute_inputs(declared, {"mode": "b"})] == ["mode", "extra"]


def test_a_hook_shapes_the_version_the_placement_pins_not_the_newest():
    """A fresh instance knows no version. The caller hands the hook the
    declaration of the version it is asking about, so an old placement keeps
    its old roster."""
    from conductor.node import version

    class Two(NodeDefinition):
        id = "two-hook"
        title = "Two"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, old: Annotated[Txt, Textarea(title="Old")] = Txt("")) -> Out:
            return old

        @version(2)
        def run(self, new: Annotated[Txt, Textarea(title="New")] = Txt("")) -> Out:
            return new

    pinned = Two.versions[1].interface.inputs
    assert [i.name for i in Two().compute_inputs(pinned, {})] == ["old"]


def test_a_node_with_no_shaping_declares_none():
    """The default hooks answer with the declaration, so nothing branches."""

    class Simple(NodeDefinition):
        id = "simple"
        title = "Simple"
        description = "No shaping"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    current = Simple.versions[Simple.current].interface
    assert Simple().compute_inputs(current.inputs, {}) == current.inputs
    assert Simple().compute_outputs(current.outputs, {}, {}) == current.outputs


def test_the_hook_contract_is_two_methods():
    """A value's constraints are its dtype's constructor rules, and a
    wiring problem is the compiler's — so a node has no `validate` and no
    `Problem` channel of its own."""
    assert not hasattr(NodeDefinition, "validate")


def test_a_hook_that_cannot_answer_raises_refuses():
    """`Refuses(code, message)` is the one refusal a roster hook has: the
    host names the code and writes the sentence, and the compiler anchors both
    as the placement's fatal `Problem` (the graph plans)."""
    from conductor.node import Refuses

    refusal = Refuses("wrong_shape", "What arrives does not fit.")
    assert (refusal.code, refusal.message) == ("wrong_shape", "What arrives does not fit.")
    assert isinstance(refusal, Exception)


def test_a_reduction_declares_a_series_input():
    """The dtype is the only thing that says 'do not broadcast me'."""

    class Join(NodeDefinition):
        id = "join"
        title = "Join text"
        description = "d"
        category = "test"

        def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Out:
            return Txt("\n".join(texts))

    (texts,) = Join.versions[1].interface.inputs
    assert texts.dtype is Series[Txt]




def test_describe_is_the_class_as_a_record():
    class Translate(NodeDefinition):
        id = "describe-translate"
        title = "Translation"
        description = "Translates"
        category = "test"
        tags = ("Language model",)

        @version(1)
        def run_v1(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
            return text

        @version(2, policy=Policy(retries=2))
        def run(
            self,
            text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
            language: Annotated[Txt, Dropdown(title="Language", choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English")))] = Txt("da"),
        ) -> Out:
            return text

    d = Translate.describe()

    assert isinstance(d, NodeDescription)
    assert (d.id, d.title, d.tags) == ("describe-translate", "Translation", ("Language model",))
    assert d.current == 2
    assert [i.name for i in d.versions[2].inputs] == ["text", "language"]
    assert d.versions[2].policy.retries == 2
    assert d.versions[1].outputs[0].name == "result"
    assert d.versions[1].open is None
    assert not hasattr(d, "role")


def test_describe_carries_both_notices():
    """Two scopes, two true things. The palette renders both when both
    exist, node first — a precedence rule would show strictly less."""

    @deprecated(header="Node retired")
    class Old(NodeDefinition):
        id = "describe-old"
        title = "Old"
        description = "d"
        category = "test"

        @deprecated(header="v1 retired", migration="Use v2.")
        @version(1)
        def run_v1(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

        @version(2)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    d = Old.describe()

    assert d.deprecation == Deprecation(header="Node retired")
    assert d.versions[1].deprecation == Deprecation(header="v1 retired", migration="Use v2.")
    assert d.versions[2].deprecation is None
    data = TypeAdapter(NodeDescription).dump_python(d, mode="json")
    assert data["deprecation"] == {"header": "Node retired", "description": None, "alternative": None, "migration": None}
    assert "deprecated" not in data


def test_describe_dumps_as_the_palette_payload():
    class Echo(NodeDefinition):
        id = "describe-echo"
        title = "Echo"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X", rows=2)] = Txt("")) -> Out:
            return x

    data = TypeAdapter(NodeDescription).dump_python(Echo.describe(), mode="json")

    assert data["id"] == "describe-echo"
    assert data["category"] == "test"
    assert data["versions"]["1"]["inputs"][0]["widget"] == {
        "kind": "textarea", "min_length": None, "max_length": None, "rows": 2,
    }
    assert data["versions"]["1"]["inputs"][0]["dtype"] == {"id": "node-class-test-txt", "accepted_as": ["node-class-test-txt"]}
    assert data["versions"]["1"]["outputs"][0]["dtype"] == {"id": "node-class-test-txt", "accepted_as": ["node-class-test-txt"]}
    for gone in ("base_id", "width", "is_decision", "has_dynamic_outputs", "latest_version", "deprecated"):
        assert gone not in data


def test_describe_publishes_a_json_schema():
    schema = TypeAdapter(NodeDescription).json_schema(mode="serialization")

    assert "Input" in schema["$defs"]
    assert "Output" in schema["$defs"]
    assert "Policy" in schema["$defs"]
    widget = schema["$defs"]["Input"]["properties"]["widget"]
    assert widget["discriminator"]["propertyName"] == "kind"


def test_describe_is_computed_not_stored():
    class Echo(NodeDefinition):
        id = "describe-echo-2"
        title = "Echo"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    assert Echo.describe() == Echo.describe()
    assert "description_record" not in vars(Echo)




class Flag(DType):
    """A boolean dtype for the test — conductor ships none."""

    id = "node-class-test-flag"
    title = "Yes/No"

    def __init__(self, value: bool = False) -> None:
        self.value = bool(value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Flag) and other.value == self.value


def test_an_upgrade_rewrites_saved_values_between_versions():
    class Docs(NodeDefinition):
        id = "docs"
        title = "Docs"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(
            self,
            files: Annotated[Txt, Textarea(title="Files")] = Txt(""),
            strategy: Annotated[Txt, Textarea(title="Strategy")] = Txt("whole"),
        ) -> Out:
            return Txt(f"{files}:{strategy}")

        @version(2)
        def run(
            self,
            files: Annotated[Txt, Textarea(title="Files")] = Txt(""),
            split: Annotated[Flag, Switch(title="Split")] = Flag(False),
        ) -> Out:
            return Txt(f"{files}:{split.value}")

        @upgrade(1, 2)
        def _v1_to_v2(values):
            rewritten = dict(values)
            rewritten["split"] = Flag(rewritten.pop("strategy", None) == "per_page")
            return rewritten

    registry = NodeRegistry()
    registry.register(Docs)

    rewrite = registry.upgrade_path("docs", 1, 2)
    assert rewrite is not None
    assert rewrite({"files": "a.pdf", "strategy": "per_page"}) == {
        "files": "a.pdf",
        "split": Flag(True),
    }


def test_an_upgrade_takes_values_and_not_an_instance():
    """It rewrites data. There is no placement to consult and no state."""

    class Docs2(NodeDefinition):
        id = "docs-2"
        title = "Docs"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, a: Annotated[Txt, Textarea(title="A")] = Txt("")) -> Out:
            return a

        @version(2)
        def run(self, b: Annotated[Txt, Textarea(title="B")] = Txt("")) -> Out:
            return b

        @upgrade(1, 2)
        def _v1_to_v2(values):
            return {"b": values["a"]}

    assert isinstance(vars(Docs2)["_v1_to_v2"], staticmethod)
    assert Docs2._v1_to_v2({"a": Txt("x")}) == {"b": "x"}


def test_a_missing_upgrade_path_is_none_not_an_error():
    class Plain(NodeDefinition):
        id = "plain"
        title = "Plain"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    registry = NodeRegistry()
    registry.register(Plain)
    assert registry.upgrade_path("plain", 1, 2) is None


def test_there_is_exactly_one_way_to_declare_a_node():
    """Gone, so they cannot come back by habit."""
    import conductor

    assert not hasattr(NodeRegistry, "node")
    assert not hasattr(NodeRegistry, "register_class")
    assert not hasattr(conductor, "BaseNode")


def test_a_category_is_a_string_on_the_definition():
    """A category is presentation: where the palette files the node. No
    object, no registration side-channel."""
    import conductor

    class Echo(NodeDefinition):
        id = "echo-cat"
        title = "Echo"
        description = "Returns input"
        category = "tools"

        def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
            return text

    assert Echo.category == "tools"
    assert Echo.describe().category == "tools"
    assert not hasattr(conductor, "NodeCategory")
    assert not hasattr(NodeRegistry, "include")


def test_the_contract_is_importable_from_the_package_root():
    import conductor

    for name in (
        "NodeDefinition", "NodeVersion", "GraphVersion", "Policy", "Deprecation",
        "NodeDescription", "Input", "Interface", "Provided", "AnyWidget",
    ):
        assert getattr(conductor, name) is not None
    assert callable(conductor.version)
    assert callable(conductor.upgrade)
    assert callable(conductor.deprecated)


def test_a_class_node_executes_in_a_graph():
    class Shout(NodeDefinition):
        id = "shout"
        title = "Shout"
        description = "d"
        category = "test"

        def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
            return Txt(self._decorate(text))

        def _decorate(self, text: str) -> str:
            return text.upper() + "!"

    registry = NodeRegistry()
    registry.register(Shout)

    compiled = compile_graph(
        nodes=[GraphNode(id="a", type="shout", version=1, data={"text": "hi"})],
        edges=[],
        registry=registry,
    )
    results = execute_sync(compiled)
    assert results["a"]["result"] == "HI!"


def test_two_versions_of_one_class_execute_independently():
    class Suffix(NodeDefinition):
        id = "suffix"
        title = "Suffix"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
            return text

        @version(2)
        def run(
            self,
            text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
            mark: Annotated[Txt, Textarea(title="Mark")] = Txt("?"),
        ) -> Out:
            return Txt(text + mark)

    registry = NodeRegistry()
    registry.register(Suffix)

    for pinned, expected in ((1, "hi"), (2, "hi?")):
        compiled = compile_graph(
            nodes=[GraphNode(id="a", type="suffix", version=pinned, data={"text": "hi"})],
            edges=[],
            registry=registry,
        )
        assert execute_sync(compiled)["a"]["result"] == expected


def test_a_version_a_class_does_not_declare_is_refused():
    """The pin is what a graph selects with, so a graph naming a version
    that is not there must not resolve to the nearest one."""

    class Once(NodeDefinition):
        id = "once-exec"
        title = "Once"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            return x

    registry = NodeRegistry()
    registry.register(Once)

    with pytest.raises(KeyError):
        runner_for(registry, "once-exec", 2)
    with pytest.raises(KeyError):
        runner_for(registry, "never-registered", 1)


def test_each_call_gets_a_fresh_instance():
    """Stateless by contract."""
    seen = []  # holds the instances, so CPython cannot reuse an id

    class Counting(NodeDefinition):
        id = "counting"
        title = "Counting"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Out:
            seen.append(self)
            return x

    registry = NodeRegistry()
    registry.register(Counting)
    runner = runner_for(registry, "counting", 1)
    runner(x=Txt("a"))
    runner(x=Txt("b"))

    assert seen[0] is not seen[1]


def test_a_definition_may_carry_its_versions_by_value():
    """A definition built from data (an embedded flow) sets ``versions`` in
    the class body, each a ``GraphVersion`` holding an interface and the
    placements the compiler expands it to. ``register()`` checks the
    numbering and ``describe()`` reads it, but ``runner_for`` refuses it:
    there is nothing to run, the compiler expanded it."""
    from collections.abc import Mapping

    from conductor.interface import Interface, model_of
    from conductor.node import GraphVersion

    class Wrapped(NodeDefinition):
        id = "wrapped"
        title = "Embedded"
        description = "d"
        category = "test"

        versions: ClassVar[dict[int, GraphVersion]] = {
            3: GraphVersion(
                graph=(),
                interface=Interface(
                    inputs=(Input(name="inner.text", dtype=Txt, title="Text", widget=Textarea(title="Text")),),
                    outputs=(Output(name="inner.result", dtype=Txt, title="Result"),),
                    returns=Mapping,
                ),
            )
        }

    assert set(Wrapped.versions) == {3} and Wrapped.current == 3
    assert [i.name for i in Wrapped.versions[3].interface.inputs] == ["inner.text"]
    assert isinstance(getattr(model_of(Wrapped.versions[3].interface.inputs)(**{"inner.text": "hi"}), "inner.text"), Txt)
    described = Wrapped.describe().versions[3]
    assert [i.name for i in described.inputs] == ["inner.text"]
    assert described.policy is None and described.deprecation is None
    with pytest.raises(ValueError, match="numbers from 1"):
        NodeRegistry().register(Wrapped)

    class Loaded(Wrapped):
        id = "loaded"
        versions: ClassVar[dict[int, GraphVersion]] = {1: Wrapped.versions[3]}

    registry = NodeRegistry()
    registry.register(Loaded)
    with pytest.raises(TypeError, match="graph"):
        runner_for(registry, "loaded", 1)
