"""``@host_type``, the decorator a host uses to declare a class of its own.

A host can pass its own classes along edges. ``@host_type`` gives such a class an id and a title. A wrong
declaration fails right away, at the class statement, rather than later when a graph is compiled.
"""

import dataclasses
import datetime
from collections import OrderedDict, deque
from enum import Enum
from fractions import Fraction
from http import HTTPStatus
from ipaddress import IPv4Address
from pathlib import Path
from re import Pattern
from typing import Any, Generic, TypeVar
from zoneinfo import ZoneInfo

import pytest
from conductor import (
    Ask,
    Graph,
    NodeDeclarationError,
    NodeRegistry,
    Run,
    host_type,
    make_graph,
    node,
)
from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ByteSize,
    FtpUrl,
    HttpUrl,
    IPvAnyAddress,
    NaiveDatetime,
    PastDate,
    PostgresDsn,
)
from pydantic_core import core_schema

from tests.test_core.test_nodes import declared


@host_type("text", title="Text")
class Text(str):
    def __new__(cls, value: str) -> "Text":
        if len(value) > 20:
            raise ValueError("a text is at most 20 characters")
        return super().__new__(cls, value)


@host_type("money", title="Money")
class Money(BaseModel):
    amount: float


@host_type("point", title="Point")
@dataclasses.dataclass
class Point:
    x: float
    y: float


@host_type("colour", title="Colour")
class Colour(Enum):
    RED = "red"


@host_type("upload", title="Upload")
class Upload:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: held.name),
        )


@host_type("celsius", title="Celsius")
class Celsius(BaseModel):
    degrees: float


@host_type("temperature", title="Temperature", accepts=[Celsius])
class Temperature(BaseModel):
    degrees: float


Status = host_type("http-status", title="HTTP status")(HTTPStatus)

T = TypeVar("T")


@host_type("box", title="Box")
class Box(BaseModel, Generic[T]):
    inside: T


@pytest.mark.parametrize(
    ("held", "type_id"),
    [
        (Text, "text"),
        (Money, "money"),
        (Point, "point"),
        (Colour, "colour"),
        (Upload, "upload"),
        (Temperature, "temperature"),
        (Status, "http-status"),
        (list[Point], "list[point]"),
    ],
)
def test_a_class_declared_with_host_type_is_a_fields_type_under_its_id(
    held: Any, type_id: str
) -> None:
    def f(value: held) -> str:  # type: ignore[valid-type]
        return ""

    (described,) = declared(f).describe().inputs
    assert described.type == type_id


def test_a_subclass_of_a_builtin_needs_no_schema_of_its_own_and_its_own_check_runs() -> None:
    @node("short", title="Short", description="For a test.")
    def short(text: Text) -> Text:
        return text

    registry = NodeRegistry()
    registry.add(short)
    graph = Graph.compile(make_graph(short), registry)

    assert type(graph.run({"text": "hello"}).outputs["result"]) is Text
    with pytest.raises(Exception) as refused:
        graph.run({"text": "x" * 21})
    assert refused.value.problems[0].type == "invalid_input"


def test_an_edge_may_join_a_type_to_one_that_accepts_it_and_not_the_other_way() -> None:
    @node("measure", title="Measure", description="For a test.")
    def measure(degrees: float) -> Celsius:
        return Celsius(degrees=degrees)

    @node("read", title="Read", description="For a test.")
    def read(temperature: Temperature) -> float:
        return temperature.degrees

    @node("warmth", title="Warmth", description="For a test.")
    def warmth(degrees: float) -> Temperature:
        return Temperature(degrees=degrees)

    @node("in-celsius", title="In Celsius", description="For a test.")
    def in_celsius(celsius: Celsius) -> float:
        return celsius.degrees

    registry = NodeRegistry()
    registry.add(measure, read, warmth, in_celsius)
    accepted = Graph.compile(make_graph(measure, read), registry)
    refused = Graph.compile(make_graph(warmth, in_celsius), registry)

    assert accepted.problems == []
    assert [problem.type for problem in refused.problems] == ["type_mismatch"]


def test_a_generic_pydantic_model_is_named_by_its_declared_id_and_its_arguments() -> None:
    def f(a: Box[int], b: Box[str]) -> str:
        return ""

    registry = NodeRegistry()
    registry.add(declared(f))
    assert [i.type for i in declared(f).describe().inputs] == ["box[int]", "box[str]"]
    assert {"box[int]", "box[str]"} <= set(registry.types)


def _refused(build: Any) -> NodeDeclarationError:
    with pytest.raises(NodeDeclarationError) as refused:
        build()
    return refused.value


def test_an_empty_title_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="")
        class Thing(BaseModel):
            pass

    (problem,) = _refused(build).problems
    assert (problem.type, problem.ctx["empty"]) == ("empty_title_or_description", "title")
    assert (
        str(_refused(build))
        == "@host_type('thing') has no title. Write what a person should read about it in a palette."
    )


@pytest.mark.parametrize(
    "wrong_id", ["str", "Series", "Table", "row", "list[int]", "a b", "a|b", ""], ids=repr
)
def test_an_id_that_is_no_word_or_is_one_of_conductors_own_is_an_invalid_node_id_as_on_a_node(
    wrong_id: Any,
) -> None:
    def build() -> None:
        @host_type(wrong_id, title="Thing")
        class Thing(BaseModel):
            pass

    (problem,) = _refused(build).problems
    assert (problem.type, problem.loc, problem.ctx["declared_with"]) == (
        "invalid_node_id",
        (),
        "@host_type",
    )


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        ({"id": 5, "title": "Thing"}, [("string_type", ["id"])]),
        ({"id": "thing", "title": None}, [("string_type", ["title"])]),
        ({"id": 5, "title": None}, [("string_type", ["id"]), ("string_type", ["title"])]),
    ],
)
def test_an_id_or_title_that_isnt_text_is_an_invalid_declaration_argument_as_on_a_node(
    arguments: dict[str, Any], faults: list[tuple[str, list[str]]]
) -> None:
    (problem,) = _refused(lambda: host_type(**arguments)).problems
    assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
        "invalid_declaration_argument",
        faults,
    )


def test_accepts_given_something_other_than_classes_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="Thing", accepts=["celsius"])  # type: ignore[list-item]
        class Thing(BaseModel):
            pass

    (problem,) = _refused(build).problems
    assert (problem.type, [(e["type"], e["loc"]) for e in problem.ctx["errors"]]) == (
        "invalid_declaration_argument",
        [("is_type", ["accepts", 0])],
    )


def test_a_plain_class_pydantic_cant_validate_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("handle", title="Handle")
        class Handle:
            def __init__(self, path: str) -> None:
                self.path = path

    refused = _refused(build)
    assert refused.problems[0].type == "host_type_not_supported" and "Handle" in str(refused)


def test_host_type_written_under_dataclass_fails_at_the_class_statement_and_says_to_put_it_on_top() -> (
    None
):
    def build() -> None:
        @dataclasses.dataclass
        @host_type("pair", title="Pair")
        class Pair:
            x: float

    refused = _refused(build)
    assert refused.problems[
        0
    ].type == "host_type_not_supported" and "@host_type above @dataclass" in str(refused)


def test_a_class_python_lets_nobody_write_on_is_refused_with_the_way_out() -> None:
    refused = _refused(lambda: host_type("ordered", title="Ordered")(OrderedDict))
    assert refused.problems[
        0
    ].type == "type_declaration_not_writable" and "a class of your own" in str(refused)


def test_a_class_that_is_not_declared_fails_at_node_naming_the_class_and_the_line_to_write() -> (
    None
):
    @dataclasses.dataclass
    class Plain:
        x: float

    def f(values: list[Plain]) -> str:
        return ""

    refused = _refused(lambda: declared(f))
    assert (
        refused.problems[0].type == "type_not_named"
        and "Plain" in str(refused)
        and '@host_type("plain", title="Plain")' in str(refused)
    )


@pytest.mark.parametrize(
    ("name", "line"),
    [
        ("InvoiceLine", '@host_type("invoice-line", title="Invoice line")'),
        ("IPAddress", '@host_type("ip-address", title="IP address")'),
        ("Row", '@host_type("row-type", title="Row")'),
    ],
)
def test_the_line_a_class_that_is_not_declared_is_told_to_write_declares_it(
    name: str, line: str
) -> None:
    """The refusal suggests an id and a title made from the class's name, and pasting them declares the class.

    ``Row`` would give ``"row"``, which is one of conductor's own ids, so it gets ``-type`` after it.
    """
    undeclared = dataclasses.make_dataclass(name, [("x", float)])

    def f(value: undeclared) -> str:  # type: ignore[valid-type]
        return ""

    refused = _refused(lambda: declared(f))
    assert line in str(refused)
    problem = refused.problems[0]
    host_type(problem.ctx["suggested_id"], title=problem.ctx["suggested_title"])(undeclared)
    assert declared(f).inputs["value"].type.id == problem.ctx["suggested_id"]


def test_a_subclass_does_not_inherit_its_parents_declaration() -> None:
    class Rich(Money):
        note: str = ""

    def f(value: Rich) -> str:
        return ""

    refused = _refused(lambda: declared(f))
    assert refused.problems[
        0
    ].type == "type_not_named" and "a subclass doesn't inherit its declaration" in str(refused)


def test_a_hook_that_returns_a_class_that_is_not_declared_is_the_hooks_failure() -> None:
    @dataclasses.dataclass
    class Plain:
        x: float

    @node(
        "made", title="Made", description="For a test.", compute_outputs=lambda: {"result": Plain}
    )
    def made() -> dict[str, Any]:
        return {}

    registry = NodeRegistry()
    registry.add(made)
    (problem,) = Graph.compile(
        make_graph(made, outputs={"result": "made.result"}), registry
    ).problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == (
        "compute_hook_returned_invalid",
        "type_not_named",
    )


@pytest.mark.parametrize("accepts", [None, int, 5], ids=repr)
def test_accepts_that_is_no_list_of_classes_fails_at_the_class_statement(accepts: Any) -> None:
    def build() -> None:
        @host_type("thing", title="Thing", accepts=accepts)
        class Thing(BaseModel):
            pass

    assert _refused(build).problems[0].type == "invalid_declaration_argument"


def test_a_title_of_only_spaces_fails_at_the_class_statement() -> None:
    def build() -> None:
        @host_type("thing", title="   ")
        class Thing(BaseModel):
            pass

    assert _refused(build).problems[0].type == "empty_title_or_description"


def test_host_type_on_something_that_is_no_class_or_is_a_typed_dict_is_refused() -> None:
    from typing import TypedDict

    class Row(TypedDict):
        n: int

    def a_function() -> None:
        pass

    for wrong in (42, list[int], a_function, Row):
        refused = _refused(lambda wrong=wrong: host_type("thing", title="Thing")(wrong))
        assert refused.problems[0].type == "type_declaration_not_on_a_class", wrong


@pytest.mark.parametrize(
    ("hint", "type_id", "given"),
    [
        (HttpUrl, "HttpUrl", "https://example.com/a"),
        (PostgresDsn, "PostgresDsn", "postgres://db/main"),
        (ByteSize, "ByteSize", "1KB"),
        (Path, "Path", "reports/2026.csv"),
        (IPv4Address, "IPv4Address", "10.0.0.1"),
        (ZoneInfo, "ZoneInfo", "Europe/Copenhagen"),
        (Fraction, "Fraction", "1/3"),
        (complex, "complex", "1+2j"),
        (Pattern, "Pattern", "a+"),
        (deque[int], "deque[int]", [1, 2]),
    ],
)
def test_pydantics_own_types_and_the_standard_librarys_need_no_declaration(
    hint: Any, type_id: str, given: Any
) -> None:
    """Conductor names them, by the name a host imports, so a node takes them with no ``@host_type``; ``Pattern``,
    ``complex`` and ``deque`` are written in C, so ``@host_type`` could never name them."""

    def passes(value: hint) -> hint:  # type: ignore[valid-type]
        return value

    definition = node("passes", title="Passes", description="For a test.")(passes)
    registry = NodeRegistry()
    registry.add(definition)
    graph = Graph.compile(make_graph(definition), registry)
    run = graph.run({"value": given})

    assert definition.describe().inputs[0].type == type_id
    assert run.status == "complete"
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


def test_a_marker_class_is_read_as_the_class_of_its_values_with_its_check() -> None:
    """``AwareDatetime`` gives a ``datetime``, so it has the id ``datetime``: an edge from it lands on a ``datetime``
    input, and an edge from a ``datetime`` lands on it with its check on each call."""

    @node("stamp", title="Stamp", description="For a test.")
    def stamp(naive: bool) -> datetime.datetime:
        moment = datetime.datetime(2026, 1, 2, 3, 4)
        return moment if naive else moment.replace(tzinfo=datetime.UTC)

    @node("aware", title="Aware", description="For a test.")
    def aware(moment: AwareDatetime) -> AwareDatetime:
        return moment

    @node("plain", title="Plain", description="For a test.")
    def plain(moment: datetime.datetime) -> str:
        return moment.isoformat()

    @node("closest", title="Closest", description="For a test.")
    def closest(address: IPvAnyAddress, day: PastDate) -> str:
        return f"{address} {day}"

    registry = NodeRegistry()
    registry.add(stamp, aware, plain, closest)
    graph = Graph.compile(
        make_graph(
            stamp,
            aware,
            plain,
            edges=[("stamp.result", "aware.moment"), ("aware.result", "plain.moment")],
        ),
        registry,
    )
    described = closest.describe().inputs

    assert aware.describe().inputs[0].type == "datetime"
    assert [field.type for field in described] == ["IPv4Address | IPv6Address", "date"]
    assert graph.problems == []
    assert graph.run({"naive": False}).outputs == {"result": "2026-01-02T03:04:00+00:00"}
    assert graph.run({"naive": True}).cause.type == "invalid_edge_value"


def test_a_value_of_pydantics_own_class_built_by_hand_is_still_checked() -> None:
    """``ByteSize(-5)`` can be built, and its own check refuses it, so only Python's own classes skip the check."""

    @node("size", title="Size", description="For a test.")
    def size() -> ByteSize:
        return ByteSize(-5)

    registry = NodeRegistry()
    registry.add(size)
    run = Graph.compile(make_graph(size), registry).run()

    assert (run.status, run.cause.type) == ("failed", "invalid_output")


@pytest.mark.parametrize("named", [HttpUrl, FtpUrl, int, Path, NaiveDatetime, IPvAnyAddress])
def test_host_type_on_a_class_conductor_names_is_refused_and_says_to_declare_a_subclass(
    named: Any,
) -> None:
    """A declaration is stored on the class, so on one of conductor's own it would rename the class for every
    registry in the process, depending on what read it first."""
    refused = _refused(lambda: host_type("mine", title="Mine")(named))

    assert refused.problems[0].type == "invalid_declaration_argument"
    assert [(e["type"], e["loc"]) for e in refused.problems[0].ctx["errors"]] == [
        ("conductors_own_type", ["cls"])
    ]
    assert "can't take its arguments: cls: " in str(refused)
    assert "declare a subclass" in str(refused)
    assert "the type the signature says" not in str(refused)


def test_a_hosts_subclass_of_a_class_conductor_names_is_its_own_type() -> None:
    """A subclass of a named class, or of a marker class, is the host's own type under the host's id, so it is not
    read as the marker's value class, and an edge from ``str`` lands on it only through ``accepts``. A saved run reads
    back under the host's ids."""

    @host_type("site", title="Site")
    class Site(AnyUrl):
        pass

    @host_type("local-time", title="Local time", accepts=[datetime.datetime])
    class LocalTime(NaiveDatetime):
        pass

    @node("fetch", title="Fetch", description="For a test.")
    def fetch(site: Site, at: LocalTime) -> Site:
        return site

    @node("now", title="Now", description="For a test.")
    def now() -> datetime.datetime:
        return datetime.datetime(2026, 1, 2, 3, 4)

    registry = NodeRegistry()
    registry.add(fetch, now)
    graph = Graph.compile(make_graph(fetch), registry)
    run = graph.run({"site": "https://a.example", "at": "2026-01-02T03:04:00"})
    titles = {described.id: described.title for described in registry.describe().types}
    wired = Graph.compile(make_graph(now, fetch, edges=[("now.result", "fetch.at")]), registry)

    assert [field.type for field in fetch.describe().inputs] == ["site", "local-time"]
    assert (titles["site"], titles["local-time"]) == ("Site", "Local time")
    assert Run.from_json(run.to_json(), graph).outputs == {"result": Site("https://a.example")}
    assert wired.problems == []


def test_a_node_returning_a_declared_dataclass_has_one_output_an_edge_reads_whole_or_by_field() -> (
    None
):
    """A declared dataclass is a type, so it travels as one value on the output ``result``; a dataclass with no
    declaration still lists the node's outputs. An edge reads the whole value or one field of it, a saved run reads
    it back, and a subgraph's port carries it."""

    @dataclasses.dataclass(frozen=True)
    class Parts:
        x: float
        y: float

    @node("locate", title="Locate", description="For a test.")
    def locate(name: str) -> Point:
        return Point(1.5, 2.5)

    @node("parts", title="Parts", description="For a test.")
    def parts(name: str) -> Parts:
        return Parts(1.5, 2.5)

    @node("north", title="North", description="For a test.")
    def north(y: float) -> float:
        return y + 1

    @node("plot", title="Plot", description="For a test.")
    def plot(point: Point) -> str:
        return f"{point.x},{point.y}"

    registry = NodeRegistry()
    registry.add(locate, parts, north, plot)
    record = make_graph(
        locate,
        north,
        plot,
        edges=[("locate.result.y", "north.y"), ("locate.result", "plot.point")],
        outputs={"north": "north.result", "plot": "plot.result"},
    )
    graph = Graph.compile(record, registry)
    run = graph.run({"name": "home"})
    place = registry.subgraph("place", title="Place", description="For a test.")(
        make_graph(locate, outputs={"where": "locate.result"})
    )
    outer = Graph.compile(make_graph(place, plot, edges=[("place.where", "plot.point")]), registry)

    assert [(field.name, field.type) for field in locate.describe().outputs] == [
        ("result", "point")
    ]
    assert [field.name for field in parts.describe().outputs] == ["x", "y"]
    assert run.outputs == {"north": 3.5, "plot": "1.5,2.5"}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs
    (described,) = (each for each in registry.describe().nodes if each.id == "place")
    assert [(field.name, field.type) for field in described.outputs] == [("where", "point")]
    assert outer.run({"name": "home"}).outputs == {"result": "1.5,2.5"}


def test_an_undeclared_subclass_of_a_declared_dataclass_is_refused_on_a_return_as_on_an_input() -> (
    None
):
    """A subclass doesn't inherit its parent's declaration, and a dataclass with a declaration anywhere among its
    parents is a type, not a dataclass of outputs, so returning one is ``type_not_named`` with the parent named, as
    taking one is."""

    @dataclasses.dataclass
    class Point3(Point):
        z: float = 0.0

    def returning() -> None:
        @node("locate", title="Locate", description="For a test.")
        def locate(name: str) -> Point3:
            return Point3(1.0, 2.0)

    def taking() -> None:
        @node("plot", title="Plot", description="For a test.")
        def plot(point: Point3) -> str:
            return "here"

    returned, taken = _refused(returning), _refused(taking)

    assert [(each.type, each.ctx["parent"]) for each in returned.problems] == [
        ("type_not_named", "Point")
    ]
    assert [(each.type, each.ctx["parent"]) for each in taken.problems] == [
        ("type_not_named", "Point")
    ]


def test_a_node_returning_a_declared_dataclass_is_answered_with_one_value() -> None:
    """A node that asks for a declared dataclass asks one question, ``result``, answered with the value or its
    fields as a mapping, and a queued answer is kept through a saved run."""

    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick(name: str) -> Point:
        raise Ask("Where is it?")

    registry = NodeRegistry()
    registry.add(pick)
    graph = Graph.compile(make_graph(pick), registry)
    run = graph.run({"name": "home"})
    (waiting,) = run.waiting
    queued = run.with_answers({waiting.id: Point(3, 4)})

    assert [question.name for question in waiting.questions] == ["result"]
    assert queued.resume().outputs == {"result": Point(3, 4)}
    assert run.with_answers({waiting.id: {"x": 3, "y": 4}}).resume().outputs == {
        "result": Point(3, 4)
    }
    assert Run.from_json(queued.to_json(), graph).resume().outputs == {"result": Point(3, 4)}
