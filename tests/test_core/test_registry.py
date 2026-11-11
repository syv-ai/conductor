"""``NodeRegistry`` — the nodes a host offers, by id.

Nodes go on a registry as routes go on a FastAPI router: ``@registry.node``
declares and adds, ``add`` takes a plain ``@node`` from elsewhere, and
``add_subregistry`` copies another registry's nodes in. Every node that
enters takes the registry's defaults by one rule. ``replace``, ``remove``
and ``filter`` change or narrow what is offered; looping yields the
definitions.
"""

import itertools
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, TypedDict

import pytest
from conductor import (
    Deprecation,
    GraphRecord,
    NodeDeclarationError,
    NodeRecord,
    NodeRegistry,
    Param,
    Series,
    Table,
    Upgrade,
    node,
)
from conductor._internal.fields import REQUIRED
from conductor.widgets import Textarea

from tests.test_core.nodes import ReadAsItsBuiltin


class Txt(ReadAsItsBuiltin, str):
    type_id: ClassVar[str] = "registry-test-text"
    type_title: ClassVar[str] = "Text"


Out = Annotated[Txt, Param(title="Output")]


def echo_v1(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return text


@node("echo", title="Echo", description="Returns input unchanged", category="test", replaces=node("echo", title="Echo v1", description="")(echo_v1), upgrade=Upgrade())
def echo(
    text: Annotated[Txt, Param(title="Input", widget=Textarea())],
    prefix: Annotated[Txt, Param(title="Prefix", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(f"{prefix}{text}")


@node("upper", title="Uppercase", description="Uppercases text", category="test")
def upper_node(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return Txt(text.upper())


def _registry_with(*definitions: Any) -> NodeRegistry:
    registry = NodeRegistry()
    registry.add(*definitions)
    return registry


def _node(node_id: str, **declared: Any) -> Any:
    @node(node_id, **{"title": "Made", "description": "", **declared})
    def made(x: str = "") -> str:
        return x

    return made


# --- looking a node up ------------------------------------------------------


def test_a_node_is_looked_up_by_its_id_and_looping_yields_the_definitions_in_order():
    registry = NodeRegistry()
    registry.add(echo, upper_node)

    assert registry["echo"].function is echo.function
    assert "upper" in registry and "nonexistent" not in registry
    assert [d.function for d in registry] == [echo.function, upper_node.function]
    assert len(registry) == 2


def test_an_unknown_id_is_a_key_error_naming_the_ids_there_are():
    registry = NodeRegistry()
    registry.add(echo)

    with pytest.raises(KeyError, match="'nope' is not registered here; the ids are echo"):
        registry["nope"]


# --- the decorator ----------------------------------------------------------


def test_the_decorator_declares_as_node_does_and_adds_the_node():
    registry = NodeRegistry()

    @registry.node("shout", description="", title="Shout")
    def shout(text: str) -> str:
        return text.upper()

    assert registry["shout"] is shout
    assert shout.title == "Shout"
    assert shout("a") == "A"


def test_a_plain_node_registers_nothing():
    registry = NodeRegistry()

    @node("alone", title="Alone", description="")
    def alone(text: str) -> str:
        return text

    assert "alone" not in registry
    assert alone("a") == "a"


# --- add, replace, remove ---------------------------------------------------


def test_a_newer_version_declared_on_the_registry_takes_the_place_of_the_one_it_replaces():
    registry = NodeRegistry(category="text")

    @registry.node("upper", title="Upper", description="")
    def upper(text: str) -> str:
        return text.upper()

    @registry.node("upper", title="Upper v2", description="", replaces=upper, upgrade=Upgrade())
    def upper_v2(text: str, suffix: str = "") -> str:
        return text.upper() + suffix

    assert registry["upper"] is upper_v2
    assert sorted(upper_v2.versions) == [1, 2] and upper_v2.category == "text"
    assert len(registry) == 1


def test_replaces_naming_a_node_the_registry_does_not_hold_is_still_a_duplicate():
    registry = NodeRegistry()
    held = _node("upper")
    registry.add(held)

    with pytest.raises(NodeDeclarationError, match="two nodes have the id 'upper'"):

        @registry.node("upper", title="Other", description="", replaces=_node("upper"), upgrade=Upgrade())
        def other(x: str = "") -> str:
            return x

    assert registry["upper"].function is held.function


def test_two_nodes_built_from_graphs_under_one_id_are_told_apart_by_title():
    inner = GraphRecord(nodes=[NodeRecord(id="a", node="upper")])
    registry = NodeRegistry()
    registry.add(node("g", description="", title="First")(inner))

    with pytest.raises(NodeDeclarationError, match="titled 'First'.*titled 'Second'"):
        registry.add(node("g", description="", title="Second")(inner))


def test_a_second_node_under_one_id_is_refused_naming_where_both_were_written():
    registry = NodeRegistry()
    first, second = _node("dup"), _node("dup")
    registry.add(first)

    with pytest.raises(NodeDeclarationError, match="two nodes have the id 'dup'") as raised:
        registry.add(second)

    assert raised.value.code == "duplicate_node_id"
    assert str(raised.value).count("made") == 2
    assert registry["dup"].function is first.function


def test_the_same_node_added_twice_is_a_duplicate_too():
    registry = NodeRegistry()
    registry.add(echo)

    with pytest.raises(NodeDeclarationError, match="'echo'"):
        registry.add(echo)


def test_a_refused_add_adds_none_of_the_nodes_it_was_given():
    registry = NodeRegistry()
    registry.add(_node("taken"))

    with pytest.raises(NodeDeclarationError):
        registry.add(_node("fresh"), _node("taken"))

    assert "fresh" not in registry


def test_replace_puts_a_node_in_place_of_the_one_under_its_id():
    registry = NodeRegistry()
    registry.add(_node("a"), _node("b"))
    newer = _node("a", title="Newer")

    registry.replace(newer)

    assert registry["a"].function is newer.function and registry["a"].title == "Newer"
    assert [definition.id for definition in registry] == ["a", "b"]


def test_replace_and_remove_refuse_an_id_that_is_not_there():
    registry = NodeRegistry()

    with pytest.raises(KeyError, match="'a' is not registered here"):
        registry.replace(_node("a"))
    with pytest.raises(KeyError, match="'a' is not registered here"):
        registry.remove("a")


def test_remove_takes_a_node_out_and_the_types_only_it_declared():
    registry = NodeRegistry()
    registry.add(echo, _node("plain"))

    registry.remove("echo")

    assert "echo" not in registry
    assert "registry-test-text" not in registry.types


def test_a_deprecation_may_name_an_alternative_the_registry_does_not_hold():
    """Which nodes a registry offers is the host's choice, so narrowing one never breaks a deprecation notice."""
    registry = NodeRegistry()

    registry.add(_node("old", deprecation=Deprecation(alternative="new", migration="Use new.")))

    assert "old" in registry


# --- defaults ---------------------------------------------------------------


def test_a_node_takes_the_registrys_defaults_its_own_winning():
    registry = NodeRegistry(category="text", tags=["builtin"], extra={"tier": "free", "icon": "a"})

    @registry.node("own", title="Own", description="", category="mine", tags=["fast"], extra={"icon": "b"})
    def own(text: str) -> str:
        return text

    @registry.node("bare", title="Bare", description="")
    def bare(text: str) -> str:
        return text

    assert (own.category, own.tags, dict(own.extra)) == ("mine", ("builtin", "fast"), {"tier": "free", "icon": "b"})
    assert (bare.category, bare.tags, dict(bare.extra)) == ("text", ("builtin",), {"tier": "free", "icon": "a"})


def test_add_applies_the_defaults_to_a_plain_node_and_leaves_the_plain_one_as_it_was():
    plain = _node("plain")
    registry = NodeRegistry(category="text")

    registry.add(plain)

    assert registry["plain"].category == "text"
    assert plain.category == ""


def test_every_version_of_a_node_takes_the_registrys_defaults():
    @node("split", title="Split", description="")
    def split_v1(text: str) -> str:
        return text

    @node("split", title="Split", description="", replaces=split_v1)
    def split(text: str) -> str:
        return text

    registry = NodeRegistry(category="text", tags=["builtin"], extra={"icon": "scissors"})
    registry.add(split)

    for version in (1, 2):
        held = registry["split"].version_of(version)
        assert (held.category, held.tags, dict(held.extra)) == ("text", ("builtin",), {"icon": "scissors"})


def test_a_newer_version_declared_on_the_registry_replaces_the_one_it_holds():
    registry = NodeRegistry(category="text")

    @registry.node("up", title="Up", description="")
    def up_v1(text: str) -> str:
        return text

    @registry.node("up", title="Up", description="", replaces=up_v1)
    def up(text: str) -> str:
        return text

    assert registry["up"] is up
    assert registry["up"].version_of(1).function is up_v1.function


@pytest.mark.parametrize("defaults", [{}, {"category": "text"}])
def test_a_plain_added_version_is_never_the_one_held_so_declaring_its_successor_is_a_duplicate(defaults: dict[str, Any]):
    @node("up", title="Up", description="")
    def up_v1(text: str) -> str:
        return text

    registry = NodeRegistry(**defaults)
    registry.add(up_v1)

    with pytest.raises(NodeDeclarationError) as refused:
        @registry.node("up", title="Up", description="", replaces=up_v1)
        def up(text: str) -> str:
            return text

    assert refused.value.code == "duplicate_node_id"


@pytest.mark.parametrize("defaults", [{}, {"category": "text"}])
def test_the_version_a_decorator_returned_is_replaced_by_its_successor(defaults: dict[str, Any]):
    registry = NodeRegistry(**defaults)

    @registry.node("up", title="Up", description="")
    def up_v1(text: str) -> str:
        return text

    @registry.node("up", title="Up", description="", replaces=up_v1)
    def up(text: str) -> str:
        return text

    assert registry["up"] is up
    assert registry["up"].version_of(1).function is up_v1.function


def test_a_registry_holds_its_own_copy_of_a_node_it_is_given():
    registry = NodeRegistry()
    registry.add(echo)

    assert registry["echo"] is not echo
    assert registry["echo"].function is echo.function


# --- subregistries ----------------------------------------------------------


def test_a_subregistry_is_copied_in_with_both_registries_defaults_outermost_first():
    text = NodeRegistry(category="text", tags=["text"])

    @text.node("upper", title="Upper", description="")
    def upper(value: str) -> str:
        return value.upper()

    top = NodeRegistry(category="other", tags=["builtin"])
    top.add_subregistry(text)

    assert (top["upper"].category, top["upper"].tags) == ("text", ("builtin", "text"))


def test_a_change_to_a_subregistry_after_it_was_added_does_not_reach_the_registry():
    sub = NodeRegistry()
    sub.add(_node("a"))
    top = NodeRegistry()
    top.add_subregistry(sub)

    sub.add(_node("b"))
    sub.remove("a")

    assert [definition.id for definition in top] == ["a"]


def test_one_id_in_two_subregistries_is_refused():
    first, second = NodeRegistry(), NodeRegistry()
    first.add(_node("x"))
    second.add(_node("x"))
    top = NodeRegistry()
    top.add_subregistry(first)

    with pytest.raises(NodeDeclarationError, match="'x'"):
        top.add_subregistry(second)


def test_a_subregistrys_named_types_come_along():
    class Money(ReadAsItsBuiltin, float):
        type_id: ClassVar[str] = "money"
        type_title: ClassVar[str] = "Money"

    top = NodeRegistry()
    top.add_subregistry(NodeRegistry(types=[Money]))

    assert top.types == {"money": Money}


def test_a_registry_per_request_leaves_the_shared_one_as_it_was():
    shared = NodeRegistry()
    shared.add(_node("built-in"))

    per_request = NodeRegistry()
    per_request.add_subregistry(shared)
    per_request.add(_node("loaded"))

    assert "loaded" in per_request and "built-in" in per_request
    assert "loaded" not in shared


# --- filter -----------------------------------------------------------------


def test_filter_gives_a_new_registry_of_the_nodes_the_predicate_keeps():
    registry = NodeRegistry(tags=["builtin"])
    registry.add(_node("free", extra={"tier": "free"}), _node("paid", extra={"tier": "paid"}))

    free = registry.filter(lambda definition: definition.extra.get("tier") == "free")

    assert [definition.id for definition in free] == ["free"]
    assert len(registry) == 2
    free.add(_node("later"))
    assert free["later"].tags == ("builtin",)
    assert "later" not in registry


def test_a_filtered_registry_compiles_only_what_it_kept():
    registry = NodeRegistry()
    registry.add(_node("kept"), _node("dropped"))
    narrowed = registry.filter(lambda definition: definition.id == "kept")

    graph = narrowed.load(GraphRecord(nodes=[NodeRecord(id="d", node="dropped")]))

    assert [problem.type for problem in graph.problems] == ["unknown_node"]


# --- extra ------------------------------------------------------------------


def test_extra_is_json_kept_and_described_but_never_read():
    @node("tagged", title="Tagged", description="", extra={"icon": "star", "order": [1, 2], "beta": None})
    def tagged(text: str) -> str:
        return text

    assert dict(tagged.extra) == {"icon": "star", "order": [1, 2], "beta": None}
    assert tagged.describe().extra == {"icon": "star", "order": [1, 2], "beta": None}


# --- what a version declares ------------------------------------------------


def test_inputs_extracted_from_signature():
    older = echo.replaces
    assert older is not None and list(older.inputs) == ["text"]
    assert older.inputs["text"].description.title == "Input"


def test_outputs_extracted_from_return_type():
    assert list(echo.outputs) == ["result"]
    assert echo.outputs["result"].description.title == "Output"


def test_multi_input_node():
    assert list(echo.inputs) == ["text", "prefix"]


def test_default_values_captured():
    text, prefix = echo.inputs.values()
    assert text.default is REQUIRED
    assert prefix.default == ""
    assert prefix.default is not REQUIRED


def test_var_keyword_is_where_added_inputs_arrive_and_needs_the_hook():
    """``**values`` is where the inputs a ``compute_added_inputs`` hook adds arrive;
    the hook declares them, so without one it is refused."""

    @node("varkw", title="Var kw", description="", compute_added_inputs=lambda: {"extra_one": str})
    def var_kw(text: Annotated[Txt, Param(title="Text", widget=Textarea())], **values: Any) -> Out:
        return text

    assert list(var_kw.inputs) == ["text"]
    with pytest.raises(NodeDeclarationError) as raised:

        @node("varkw", title="No hook", description="")
        def no_hook(text: str, **values: Any) -> str:
            return text

    assert raised.value.code == "var_keyword_without_hook"


def test_var_positional_is_refused():
    """A ``*args`` parameter has no name an edge could land on."""
    with pytest.raises(NodeDeclarationError, match="passed by name"):

        @node("varpos", title="VarPos", description="Dynamic", category="test")
        def var_pos(text: Annotated[Txt, Param(title="Text", widget=Textarea())], *args: Any) -> Out:
            return text


# --- the types a registry holds ---------------------------------------------


class Text(ReadAsItsBuiltin, str):
    type_id: ClassVar[str] = "text"
    type_title: ClassVar[str] = "Text"


class Number(ReadAsItsBuiltin, float):
    type_id: ClassVar[str] = "number"
    type_title: ClassVar[str] = "Number"


class AnotherText(ReadAsItsBuiltin, str):
    """A second class claiming ``text``: what a second library, or a notebook cell run again, makes."""

    type_id: ClassVar[str] = "text"
    type_title: ClassVar[str] = "Text"


def _declare_text() -> type:
    """What a notebook cell does when it runs again: the same class body, a new class."""

    class Text(ReadAsItsBuiltin, str):
        type_id: ClassVar[str] = "text"
        type_title: ClassVar[str] = "Text"

    return Text


@node("weigh", title="Weigh", description="")
def weigh(text: Text, weights: Series[Number], tags: list[str] = []) -> Text:  # noqa: B006
    return text


def test_a_new_registry_holds_no_types():
    assert dict(NodeRegistry().types) == {}


def test_a_hosts_type_is_described_by_its_own_id_and_title():
    registry = NodeRegistry(types=[Text])

    assert [t.model_dump() for t in registry.describe().types] == [{"id": "text", "title": "Text", "accepted_as": ["text"]}]


def test_a_hosts_type_is_described_by_the_id_and_title_it_declares_and_conductors_by_its_table():
    class Money(ReadAsItsBuiltin, float):
        type_id: ClassVar[str] = "money"
        type_title: ClassVar[str] = "Beløb"

    @node("price", title="Price", description="")
    def price(text: str) -> Money:
        return Money(1)

    registry = NodeRegistry()
    registry.add(price)

    assert {t.id: t.title for t in registry.describe().types} == {"str": "Text", "money": "Beløb"}


@dataclass(frozen=True)
class Customer:
    type_id: ClassVar[str] = "customer"
    type_title: ClassVar[str] = "Customer"
    id: str = ""
    title: str = ""


@dataclass(frozen=True)
class Money:
    type_id: ClassVar[str] = "money"
    type_title: ClassVar[str] = "Money"
    amount: float = 0.0


def test_fields_named_id_and_title_are_the_records_own():
    @node("bill", title="Bill", description="")
    def bill(customer: Customer, price: Money) -> None:
        pass

    registry = NodeRegistry()
    registry.add(bill)

    assert dict(registry.types) == {"customer": Customer, "money": Money}
    assert Customer(id="c1", title="Ms").title == "Ms"


def test_adding_a_node_adds_the_types_it_declares_a_series_giving_its_item_and_a_generic_its_class():
    registry = NodeRegistry()
    registry.add(weigh)

    assert dict(registry.types) == {"text": Text, "number": Number, "list": list, "str": str}


def test_every_class_a_fields_type_names_is_one_the_registry_holds_and_a_table_names_none():
    """A union names each member, a ``Literal`` its values' types and a generic its class and its arguments'. ``None``
    names no class a value is made of, and a table is known by its columns, so its id is no class a registry holds."""

    class Cash(ReadAsItsBuiltin, float):
        type_id: ClassVar[str] = "cash"
        type_title: ClassVar[str] = "Cash"

    class Row(TypedDict):
        weight: Number

    @node("mix", title="Mix", description="")
    def mix(either: Cash | Text | None, level: Literal["low", "high"], weights: Series[Number], rows: Table[Row], tags: list[str] = []) -> Text:  # noqa: B006
        return Text("")

    registry = NodeRegistry()
    registry.add(mix)

    [described] = registry.describe().nodes
    shown = {named for field in (*described.inputs, *described.outputs) for named in field.type.split(" | ")}
    assert shown == {"cash", "text", "str", "number", "list[str]", "Table[weight: number]"}
    assert dict(registry.types) == {"cash": Cash, "text": Text, "str": str, "number": Number, "list": list}


def test_an_any_input_adds_no_type():
    @node("pass", title="Passes", description="")
    def passes(value: Any) -> Text:
        return Text(value)

    registry = NodeRegistry()
    registry.add(passes)

    assert dict(registry.types) == {"text": Text}
    assert [t.model_dump() for t in registry.describe().types] == [{"id": "text", "title": "Text", "accepted_as": ["text"]}]


def test_two_classes_may_claim_one_id_until_one_registry_holds_both():
    first, second = _declare_text(), _declare_text()
    assert first is not second

    assert dict(NodeRegistry(types=[first]).types) == {"text": first}
    assert dict(NodeRegistry(types=[second]).types) == {"text": second}
    with pytest.raises(NodeDeclarationError, match="type id 'text'") as raised:
        NodeRegistry(types=[first, second])
    assert raised.value.code == "duplicate_type_id"


def test_one_id_from_two_classes_is_refused_naming_both_and_one_class_twice_is_not():
    assert dict(NodeRegistry(types=[Text, Text]).types) == {"text": Text}

    with pytest.raises(NodeDeclarationError) as raised:
        NodeRegistry(types=[Text, AnotherText])

    assert f"{Text.__module__}.Text" in str(raised.value)
    assert f"{AnotherText.__module__}.AnotherText" in str(raised.value)


def test_one_node_declaring_two_classes_under_one_id_is_refused():
    @node("copy", title="Copy", description="")
    def copy(text: Text) -> AnotherText:
        return AnotherText(text)

    with pytest.raises(NodeDeclarationError, match="type id 'text'"):
        NodeRegistry().add(copy)


def test_a_type_clash_on_add_names_both_classes_and_adds_nothing():
    @node("whisper", title="Whisper", description="")
    def whisper(text: AnotherText) -> AnotherText:
        return AnotherText(text.lower())

    registry = NodeRegistry()
    registry.add(weigh)

    with pytest.raises(NodeDeclarationError) as raised:
        registry.add(whisper)

    assert "Text" in str(raised.value) and "AnotherText" in str(raised.value)
    assert "whisper" not in registry
    assert dict(registry.types) == {"text": Text, "number": Number, "list": list, "str": str}


def test_a_subclass_without_its_own_id_cannot_sit_beside_its_parent():
    """A subclass is a type of its own; under its parent's id it would travel as the parent."""

    class Short(Text):
        type_title: ClassVar[str] = "Short"

    @node("cut", title="Cut", description="")
    def cut(text: Short) -> Short:
        return text

    registry = NodeRegistry(types=[Text])

    with pytest.raises(NodeDeclarationError, match="type id 'text'") as raised:
        registry.add(cut)

    assert raised.value.code == "duplicate_type_id"
    assert dict(registry.types) == {"text": Text}


def test_the_types_are_read_only_and_filter_keeps_the_named_ones():
    class Score(ReadAsItsBuiltin, float):
        type_id: ClassVar[str] = "score"
        type_title: ClassVar[str] = "Score"

    registry = NodeRegistry(types=[Score])
    registry.add(weigh)

    with pytest.raises(TypeError):
        registry.types["score"] = Score  # type: ignore[index]

    assert registry.filter(lambda definition: True).types == registry.types
    assert dict(registry.filter(lambda definition: False).types) == {"score": Score}


# --- what a registry is -----------------------------------------------------


def test_a_registry_is_a_container_and_not_a_mapping():
    """``in``, ``[]``, ``len`` and looping over definitions, and no ``get`` answering a typo with ``None``."""
    from collections.abc import Container, Mapping

    registry = _registry_with(echo)

    assert isinstance(registry, Container) and not isinstance(registry, Mapping)
    assert not hasattr(registry, "get") and not hasattr(registry, "items")


def test_a_stored_graph_made_a_node_is_read_as_load_reads_a_record():
    from conductor import InvalidCallError, node

    with pytest.raises(InvalidCallError) as refused:
        node("e", title="E", description="")({"nodes": [{"id": "a", "node": "x", "vaules": {}}]})
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_record", ("record", "nodes", 0, "vaules"))]
    assert node("e", title="E", description="")('{"nodes": [{"id": "a", "node": "x"}]}').graph == GraphRecord(nodes=[NodeRecord(id="a", node="x")])


def test_an_infinite_default_writes_as_text():
    import json

    from conductor import node

    @node("far", title="Far", description="")
    def far(limit: float = float("inf")) -> float:
        return limit

    assert json.loads(far.describe().model_dump_json())["inputs"][0]["default"] == "Infinity"


def test_a_stored_graphs_fields_are_listed_by_the_registry_that_holds_its_nodes_not_by_its_definition():
    from conductor import make_graph, node

    @node("upper", title="Upper", description="")
    def upper(text: str) -> str:
        return text.upper()

    shout = node("shout", title="Shout", description="")(make_graph(upper))
    registry = NodeRegistry()
    registry.add(upper, shout)
    listed = next(n for n in registry.describe().nodes if n.id == "shout")
    assert listed.kind == "graph" and not hasattr(shout, "describe")
    assert not {"policy", "takes_added_inputs", "computes_outputs"} & listed.model_dump(mode="json").keys()
    assert [f.name for f in listed.inputs] == ["text"] and [f.name for f in listed.outputs] == ["upper.result"]


def test_a_stored_graph_is_described_once_until_the_registry_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """``describe()`` is what an editor's palette asks on every open, so a stored graph's fields are worked out once."""
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    registry.add(node("loud", title="Loud", description="")(GraphRecord(nodes=[NodeRecord(id="s", node="shout")])))
    loads = []
    load = NodeRegistry.load
    monkeypatch.setattr(NodeRegistry, "load", lambda self, record: loads.append(record) or load(self, record))

    first, second = registry.describe(), registry.describe()
    assert first == second and len(loads) == 1

    @node("shout", title="Shout in words", description="")
    def shout_in_words(words: str) -> str:
        return words.upper()

    registry.replace(shout_in_words)
    (loud,) = [n for n in registry.describe().nodes if n.id == "loud"]
    assert [field.name for field in loud.inputs] == ["words"] and len(loads) == 2


def test_the_modules_own_example_runs():
    from conductor._internal import registry as module

    after = module.__doc__.split("::\n", 1)[1].splitlines()
    block = "\n".join(line[4:] for line in itertools.takewhile(lambda line: not line or line.startswith("    "), after))
    scope: dict[str, Any] = {"NodeRegistry": NodeRegistry, "shout": _node("shout")}
    exec(block, scope)  # noqa: S102  the docstring's own code, as a reader runs it
    assert [definition.id for definition in scope["registry"]] == ["upper", "shout"]
