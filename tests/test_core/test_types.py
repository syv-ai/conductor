"""A type means the same wherever it is read: on an edge, in a list, under a limit, in an answer.

Each test names one place where reading an annotation a second way gave a second answer.
"""

import datetime
from typing import Annotated, Any, ClassVar, Literal

import pytest
from conductor import Ask, NodeDeclarationError, NodeRegistry, Param, Series, node
from conductor.widgets import Widget
from pydantic import BaseModel, JsonValue

from tests.test_core.nodes import registry_of


class Text(str):
    """A host's own text."""

    type_id: ClassVar[str] = "text"
    type_title: ClassVar[str] = "Text"


class Money(BaseModel):
    """A host's own record."""

    type_id: ClassVar[str] = "money"
    type_title: ClassVar[str] = "Money"
    amount: float = 0


def _run(definition: Any, *more: Any, edges: list[list[str]] | None = None, **values: Any) -> Any:
    """``definition`` as node ``q`` with ``values`` typed in, and each of ``more`` as a node named by its id; the run, resumed."""
    nodes = [{"id": "q", "node": definition.id, "values": values}, *({"id": other.id, "node": other.id} for other in more)]
    graph = registry_of(definition, *more).load({"nodes": nodes, "edges": edges or []})
    assert not graph.problems, graph.problems
    return graph.run().resume()


@pytest.mark.xfail(strict=True, reason="an answer to a Series is read as JSON under a Python-only rule")
def test_an_answer_to_a_series_of_dates_is_read_from_its_json() -> None:
    @node("days", title="Days", description="")
    def days() -> Series[datetime.date]:
        raise Ask("Which days?")

    run = _run(days)
    assert run.answer("q", ["2026-01-01"]).resume().outputs == {"q.result": [datetime.date(2026, 1, 1)]}


@pytest.mark.xfail(strict=True, reason="a limit on a subclass of a builtin is dropped")
def test_a_limit_on_a_hosts_text_is_held() -> None:
    @node("short", title="Short", description="")
    def short(text: Annotated[Text, Param(max_length=3)]) -> str:
        return text

    graph = registry_of(short).load({"nodes": [{"id": "q", "node": "short", "values": {"text": "too long"}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


@pytest.mark.xfail(strict=True, reason="a subclass of a builtin is only read at the top of a hint")
def test_a_list_of_a_hosts_text_is_a_type_like_any_other() -> None:
    @node("texts", title="Texts", description="")
    def texts(items: list[Text]) -> int:
        return sum(type(item) is Text for item in items)

    assert _run(texts, items=["a", "b"]).outputs == {"q.result": 2}


@pytest.mark.xfail(strict=True, reason="an alias is compared by its JSON kinds, which a model shares with a dict")
def test_a_model_does_not_land_on_a_json_value() -> None:
    @node("money", title="Money", description="")
    def money() -> Money:
        return Money()

    @node("json", title="JSON", description="")
    def json(value: JsonValue) -> str:
        return str(value)

    graph = registry_of(money, json).load({
        "nodes": [{"id": "m", "node": "money"}, {"id": "j", "node": "json"}],
        "edges": [["m.result", "j.value"]],
    })
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


@pytest.mark.xfail(strict=True, reason="a host's own method named accepts is called as a hook")
def test_a_hosts_own_method_named_accepts_is_nothing_to_conductor() -> None:
    class Offer(BaseModel):
        type_id: ClassVar[str] = "offer"
        type_title: ClassVar[str] = "Offer"
        price: float = 0

        def accepts(self, counter: float) -> bool:
            return counter >= self.price

    @node("money", title="Money", description="")
    def money() -> Money:
        return Money()

    @node("take", title="Take", description="")
    def take(offer: Offer) -> Offer:
        return offer

    registry = NodeRegistry()
    registry.add(money, take)
    registry.describe()
    graph = registry.load({"nodes": [{"id": "m", "node": "money"}, {"id": "t", "node": "take"}], "edges": [["m.result", "t.offer"]]})
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


@pytest.mark.xfail(strict=True, reason="any failure while a schema is built is reported as type_not_carried")
def test_a_bug_in_a_hosts_schema_hook_is_raised_as_itself() -> None:
    class Buggy:
        type_id: ClassVar[str] = "buggy"
        type_title: ClassVar[str] = "Buggy"

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            return cls.no_such_helper()  # type: ignore[attr-defined]

    with pytest.raises(AttributeError, match="no_such_helper"):

        @node("buggy", title="Buggy", description="")
        def buggy(value: Buggy) -> str:
            return ""


@pytest.mark.xfail(strict=True, reason="a widget is written with fallback=str, so an object's repr reaches an editor")
def test_a_widget_with_no_json_form_fails_where_the_node_is_declared() -> None:
    class Catalogue:
        pass

    class Picker(Widget):
        type: Literal["picker"] = "picker"
        source: Any = None

    with pytest.raises(NodeDeclarationError) as raised:

        @node("pick", title="Pick", description="")
        def pick(value: Annotated[str, Param(widget=Picker(source=Catalogue()))]) -> str:
            return value

    assert raised.value.code == "widget_invalid"


@pytest.mark.xfail(strict=True, reason="the bool rule is applied to a field's own type and not to what a list holds")
def test_a_yes_or_no_is_no_number_inside_a_list_either() -> None:
    @node("total", title="Total", description="")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    graph = registry_of(total).load({"nodes": [{"id": "q", "node": "total", "values": {"numbers": [True]}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]


@pytest.mark.xfail(strict=True, reason="generic arguments are compared for equality only")
def test_a_list_of_whole_numbers_lands_on_a_list_of_numbers_and_never_on_a_list_of_texts() -> None:
    @node("ints", title="Ints", description="")
    def ints() -> list[int]:
        return [1, 2]

    @node("mean", title="Mean", description="")
    def mean(numbers: list[float]) -> float:
        return sum(numbers) / len(numbers)

    @node("words", title="Words", description="")
    def words(texts: list[str]) -> int:
        return len(texts)

    assert _run(mean, ints, edges=[["ints.result", "q.numbers"]]).outputs == {"q.result": 1.5}
    graph = registry_of(ints, words).load({
        "nodes": [{"id": "i", "node": "ints"}, {"id": "w", "node": "words"}],
        "edges": [["i.result", "w.texts"]],
    })
    assert [problem.type for problem in graph.problems] == ["type_mismatch"]


@pytest.mark.xfail(strict=True, reason="Annotated metadata other than Param is refused")
def test_a_limit_spelled_as_pydantic_spells_it_is_carried() -> None:
    from pydantic import Field, PositiveInt

    @node("count", title="Count", description="")
    def count(times: PositiveInt, least: Annotated[int, Field(ge=2)] = 2) -> int:
        return times * least

    graph = registry_of(count).load({"nodes": [{"id": "q", "node": "count", "values": {"times": 0}}]})
    assert [problem.type for problem in graph.problems] == ["invalid_value"]
    assert _run(count, times=3).outputs == {"q.result": 6}
