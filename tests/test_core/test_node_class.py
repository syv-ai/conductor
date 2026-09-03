from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.node import Deprecation, NodeDefinition, Policy, deprecated, version
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




