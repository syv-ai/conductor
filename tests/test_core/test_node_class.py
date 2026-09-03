from typing import Annotated, ClassVar

import pytest
from conductor.dtype import DType
from conductor.node import Deprecation, NodeDefinition, deprecated
from conductor.returns import Result
from conductor.widgets import Textarea


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




