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
from typing import Annotated, Any, Generic, Literal, NotRequired, TypedDict, TypeVar
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from conductor import (
    Ask,
    Graph,
    NodeDeclarationError,
    NodeRegistry,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
)
from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ByteSize,
    ConfigDict,
    Discriminator,
    Field,
    FtpUrl,
    HttpUrl,
    IPvAnyAddress,
    NaiveDatetime,
    PastDate,
    PostgresDsn,
    computed_field,
    field_serializer,
    field_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_core import core_schema

from tests.test_core.test_nodes import declared


@host_type("text", title="Text")
class Text(str):
    def __new__(cls, value: str) -> "Text":
        if len(value) > 20:
            raise ValueError("a text is at most 20 characters")
        return super().__new__(cls, value)


@host_type("money", title="Money")
class Money(BaseModel, frozen=True):
    amount: float


@host_type("point", title="Point")
@dataclasses.dataclass(frozen=True)
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
class Celsius(BaseModel, frozen=True):
    degrees: float


@host_type("temperature", title="Temperature", accepts=[Celsius])
class Temperature(BaseModel, frozen=True):
    degrees: float


Status = host_type("http-status", title="HTTP status")(HTTPStatus)

T = TypeVar("T")


@host_type("box", title="Box")
class Box(BaseModel, Generic[T], frozen=True):
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


def _faults(build: Any) -> list[tuple[str, list[str | int]]]:
    """The code and loc of every fault in the ``invalid_declaration_argument`` that ``build`` raises."""
    (problem,) = _refused(build).problems
    assert (problem.type, problem.loc) == ("invalid_declaration_argument", ())
    return [(e["type"], e["loc"]) for e in problem.ctx["errors"]]


@pytest.mark.parametrize(
    ("wrong_id", "code"),
    [
        ("str", "conductors_own_type_id"),
        ("Series", "conductors_own_type_id"),
        ("Table", "conductors_own_type_id"),
        ("date", "conductors_own_type_id"),
        ("list[int]", "type_id_not_one_word"),
        ("a b", "type_id_not_one_word"),
        ("a|b", "type_id_not_one_word"),
        ("", "type_id_not_one_word"),
    ],
    ids=repr,
)
def test_an_id_that_is_no_word_or_is_one_of_conductors_own_is_refused_by_its_own_rule(
    wrong_id: str, code: str
) -> None:
    def build() -> None:
        @host_type(wrong_id, title="Thing")
        class Thing(BaseModel, frozen=True):
            pass

    assert _faults(build) == [(code, ["id"])]
    # each message states only the rule the id broke
    (message,) = [e["msg"] for e in _refused(build).problems[0].ctx["errors"]]
    assert ("one word" in message) == (code == "type_id_not_one_word")


@pytest.mark.parametrize(
    ("arguments", "faults"),
    [
        ({"id": 5, "title": "Thing"}, [("string_type", ["id"])]),
        ({"id": "thing", "title": None}, [("string_type", ["title"])]),
        ({"id": 5, "title": None}, [("string_type", ["id"]), ("string_type", ["title"])]),
        ({"id": "thing", "title": ""}, [("blank_text", ["title"])]),
        ({"id": "thing", "title": "   "}, [("blank_text", ["title"])]),
        # accepts is a list of classes
        ({"id": "thing", "title": "Thing", "accepts": ["celsius"]}, [("is_type", ["accepts", 0])]),
        ({"id": "thing", "title": "Thing", "accepts": None}, [("tuple_type", ["accepts"])]),
        ({"id": "thing", "title": "Thing", "accepts": int}, [("tuple_type", ["accepts"])]),
        ({"id": "thing", "title": "Thing", "accepts": 5}, [("tuple_type", ["accepts"])]),
        # every fault in one pass, as on a node
        (
            {"id": "a b", "title": " ", "accepts": [1]},
            [
                ("type_id_not_one_word", ["id"]),
                ("blank_text", ["title"]),
                ("is_type", ["accepts", 0]),
            ],
        ),
    ],
)
def test_an_argument_it_cant_take_is_an_invalid_declaration_argument_as_on_a_node(
    arguments: dict[str, Any], faults: list[tuple[str, list[str]]]
) -> None:
    assert _faults(lambda: host_type(**arguments)) == faults


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
        @dataclasses.dataclass(frozen=True)
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
        ("Date", '@host_type("date-type", title="Date")'),
    ],
)
def test_the_line_a_class_that_is_not_declared_is_told_to_write_declares_it(
    name: str, line: str
) -> None:
    """The refusal suggests an id and a title made from the class's name, and pasting them declares the class.

    ``Date`` would give ``"date"``, which is one of conductor's own ids, so it gets ``-type`` after it.
    """
    undeclared = dataclasses.make_dataclass(name, [("x", float)], frozen=True)

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
    assert [port.type for port in described] == ["IPv4Address | IPv6Address", "date"]
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


class _OpenModel(BaseModel):
    x: int


class _ThawedModel(BaseModel, frozen=False):
    x: int


@dataclasses.dataclass
class _OpenDataclass:
    x: int


@pydantic_dataclass
class _OpenPydanticDataclass:
    x: int


@pytest.mark.parametrize(
    "open_class", [_OpenModel, _ThawedModel, _OpenDataclass, _OpenPydanticDataclass]
)
def test_host_type_on_a_dataclass_or_model_that_isnt_frozen_is_refused_and_says_how_to_freeze_it(
    open_class: type,
) -> None:
    """Every node that reads a value gets the same object, so a node never changes a value it is given, and a class
    that lets a node set its fields is refused where it is declared."""
    refused = _refused(lambda: host_type("open", title="Open")(open_class))

    assert refused.problems[0].type == "invalid_declaration_argument"
    assert [(e["type"], e["loc"]) for e in refused.problems[0].ctx["errors"]] == [
        ("type_not_frozen", ["cls"])
    ]
    assert f"cls: {open_class.__qualname__} isn't frozen" in str(refused)
    assert "@dataclass(frozen=True)" in str(refused)
    assert "ConfigDict(frozen=True)" in str(refused)
    assert "model_copy(update=...)" in str(refused)


class _FrozenModel(BaseModel, frozen=True):
    x: int


class _FrozenModelsChild(_FrozenModel):
    y: int


@dataclasses.dataclass(frozen=True)
class _FrozenDataclass:
    x: int


class _FrozenDataclassesChild(_FrozenDataclass):
    pass


@pydantic_dataclass(frozen=True)
class _FrozenPydanticDataclass:
    x: int


@pytest.mark.parametrize(
    "frozen_class",
    [
        _FrozenModel,
        _FrozenModelsChild,
        _FrozenDataclass,
        _FrozenDataclassesChild,
        _FrozenPydanticDataclass,
    ],
)
def test_host_type_takes_a_frozen_dataclass_or_model_and_a_subclass_that_inherits_it(
    frozen_class: type,
) -> None:
    assert host_type("frozen", title="Frozen")(frozen_class) is frozen_class


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

    assert [port.type for port in fetch.describe().inputs] == ["site", "local-time"]
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

    assert [(port.name, port.type) for port in locate.describe().outputs] == [("result", "point")]
    assert [port.name for port in parts.describe().outputs] == ["x", "y"]
    assert run.outputs == {"north": 3.5, "plot": "1.5,2.5"}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs
    (described,) = (each for each in registry.describe().nodes if each.id == "place")
    assert [(port.name, port.type) for port in described.outputs] == [("where", "point")]
    assert outer.run({"name": "home"}).outputs == {"result": "1.5,2.5"}


def test_an_undeclared_subclass_of_a_declared_dataclass_is_refused_on_a_return_as_on_an_input() -> (
    None
):
    """A subclass doesn't inherit its parent's declaration, and a dataclass with a declaration anywhere among its
    parents is a type, not a dataclass of outputs, so returning one is ``type_not_named`` with the parent named, as
    taking one is."""

    @dataclasses.dataclass(frozen=True)
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


@host_type("wallet", title="Wallet")
class Wallet(BaseModel, frozen=True):
    money: Money


class MoneyRow(TypedDict):
    money: Money


class PointRow(TypedDict):
    point: Point


class TwoMoneyRow(TypedDict):
    paid: Money
    owed: Money


class WalletRow(TypedDict):
    wallet: Wallet


@pytest.mark.parametrize(
    ("row", "value"),
    [
        (MoneyRow, {"money": Money(amount=1)}),
        (PointRow, {"point": Point(1.0, 2.0)}),
        (TwoMoneyRow, {"paid": Money(amount=1), "owed": Money(amount=2)}),
        (WalletRow, {"wallet": Wallet(money=Money(amount=1))}),
    ],
    ids=["model", "dataclass", "one model twice", "model in a model"],
)
@pytest.mark.parametrize("given", ["returned", "answered"])
def test_a_series_of_rows_with_a_hosts_class_in_a_cell_runs_and_its_saved_run_reads_back(
    row: Any, value: dict[str, Any], given: str
) -> None:
    """A row's cell may hold a host's own class, whether a node returns the rows or a person answers them."""

    def returns() -> Series[row]:  # type: ignore[valid-type]
        return [value]

    def asks() -> Series[row]:  # type: ignore[valid-type]
        raise Ask("Which rows?")

    rows = node("rows", title="Rows", description="For a test.", asks=given == "answered")(
        asks if given == "answered" else returns
    )
    registry = NodeRegistry()
    registry.add(rows)
    graph = Graph.compile(make_graph(rows), registry)
    run = graph.run({})
    if given == "answered":
        run = run.with_answers(run.waiting[0], [value]).resume()

    assert run.outputs == {"result": [value]}
    assert Run.from_json(run.to_json(), graph).outputs == {"result": [value]}


class MoniesRow(TypedDict):
    monies: list[Money]


class ColourCellRow(TypedDict):
    colour: Colour


class ColoursRow(TypedDict):
    colours: list[Colour]


class PurseRow(TypedDict):
    purse: dict[str, Money]


class PairRow(TypedDict):
    pair: tuple[Money, Money]


class ColourPairRow(TypedDict):
    colours: tuple[Colour, Colour]


class Paint(TypedDict):
    colour: Colour


class PaintRow(TypedDict):
    paint: Paint


#: A row type with a host's class in its cell, the row as a node returns it in a list, and the same row as a frame
#: keeps it: a struct of its fields, its value, or a list of structs.
_HOSTS_CLASS_CELLS = [
    (MoneyRow, {"money": Money(amount=1)}, {"money": {"amount": 1.0}}),
    (PointRow, {"point": Point(1.0, 2.0)}, {"point": {"x": 1.0, "y": 2.0}}),
    (ColourCellRow, {"colour": Colour.RED}, {"colour": "red"}),
    (MoniesRow, {"monies": [Money(amount=1)]}, {"monies": [{"amount": 1.0}]}),
    (ColoursRow, {"colours": [Colour.RED]}, {"colours": ["red"]}),
    (PurseRow, {"purse": {"a": Money(amount=1)}}, {"purse": {"a": {"amount": 1.0}}}),
    (
        PairRow,
        {"pair": (Money(amount=1), Money(amount=2))},
        {"pair": [{"amount": 1.0}, {"amount": 2.0}]},
    ),
    (PaintRow, {"paint": {"colour": Colour.RED}}, {"paint": {"colour": "red"}}),
    (ColourPairRow, {"colours": (Colour.RED, Colour.RED)}, {"colours": ["red", "red"]}),
]
_CELL_IDS = [
    "model",
    "dataclass",
    "enum",
    "list of models",
    "list of enums",
    "dict of models",
    "tuple of models",
    "enum in a row",
    "tuple of enums",
]


@pytest.mark.parametrize(
    ("row", "value", "kept", "returned"),
    [
        (*cells, returned)
        for cells in _HOSTS_CLASS_CELLS
        for returned in ("list", "frame")
        # a frame gives every row every key, so a dict's can't be kept in one
        if returned == "list" or cells[0] is not PurseRow
    ],
    ids=[
        f"{returned}-{name}"
        for name in _CELL_IDS
        for returned in ("list", "frame")
        if returned == "list" or name != "dict of models"
    ],
)
def test_a_series_of_rows_gives_a_hosts_class_in_a_cell_whether_the_node_returned_a_list_or_a_frame(
    row: Any, value: dict[str, Any], kept: dict[str, Any], returned: str
) -> None:
    """A node may return a Series of rows as a frame. A cell of a host's class is still the class, as when the node
    returns a list, for the run, for a node wired to it and in the saved run read back."""
    got: list[Any] = []

    def returns() -> Series[row]:  # type: ignore[valid-type]
        return pl.DataFrame([kept]) if returned == "frame" else [value]  # type: ignore[return-value]

    def takes(rows: Series[row]) -> int:  # type: ignore[valid-type]
        got.extend(rows)
        return len(rows)

    rows = node("rows", title="Rows", description="For a test.")(returns)
    counts = node("counts", title="Counts", description="For a test.")(takes)
    registry = NodeRegistry()
    registry.add(rows, counts)
    graph = Graph.compile(
        make_graph(rows, counts, edges=[("rows.result", "counts.rows")]), registry
    )
    run = graph.run({})

    assert run.results["rows"] == {"result": [value]}
    assert got == [value]
    assert Run.from_json(run.to_json(), graph).results == run.results


@pytest.mark.parametrize(
    ("row", "value", "kept"),
    [cells for cells in _HOSTS_CLASS_CELLS if cells[0] is not PurseRow],
    ids=[name for name in _CELL_IDS if name != "dict of models"],
)
@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_table_keeps_a_hosts_class_in_a_cell_as_its_fields_or_its_value(
    row: Any, value: dict[str, Any], kept: dict[str, Any], returned: str
) -> None:
    """Polars has no type for a host's class, so a table keeps such a cell as its fields or its value, whether the node returned
    rows or a frame. Each cell is checked by the row's type, and the saved run reads it back the same."""

    def returns() -> Table[row]:  # type: ignore[valid-type]
        return pl.DataFrame([kept]) if returned == "frame" else [value]  # type: ignore[return-value]

    table = node("table", title="Table", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(table)
    graph = Graph.compile(make_graph(table), registry)
    run = graph.run({})

    assert run.outputs["result"].to_dicts() == [kept]
    assert Run.from_json(run.to_json(), graph).outputs["result"].to_dicts() == [kept]


def test_a_frames_column_of_python_objects_is_read_by_each_objects_class() -> None:
    """A frame may hold a host's models as Python objects. Each is read by its own class's schema, so a model is
    taken and kept as its fields, and an object with no JSON form is refused as a column a saved run can't keep."""

    def returns(cells: list[Any]) -> Any:
        @node("table", title="Table", description="For a test.")
        def table() -> Table[MoneyRow]:
            return pl.DataFrame({"money": pl.Series(cells, dtype=pl.Object)})

        registry = NodeRegistry()
        registry.add(table)
        return Graph.compile(make_graph(table), registry).run({})

    refused = returns([object()])

    assert returns([Money(amount=1)]).outputs["result"].to_dicts() == [{"money": {"amount": 1.0}}]
    assert refused.cause is not None
    assert refused.cause.type == "invalid_output"
    assert refused.cause.ctx["reason"] == "column 'money' is Object, which can't be written as JSON"


@host_type("trimmed-name", title="Trimmed name")
class TrimmedName(BaseModel, frozen=True):
    text: str

    @field_validator("text")
    @classmethod
    def _trimmed(cls, text: str) -> str:
        return text.strip()


class TrimmedRow(TypedDict):
    name: TrimmedName


@pytest.mark.parametrize("declared", ["Series", "Table"])
@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_cell_a_model_changes_is_the_changed_value_in_the_table_the_series_and_the_saved_run(
    declared: str, returned: str
) -> None:
    """A model whose check strips its text gives ``"a"`` for ``"  a  "``. The node's output, a node with a ``Table``
    input wired to it, and the saved run all hold ``"a"``, whether the node returned rows or a frame."""
    got: list[Any] = []
    output: Any = Series if declared == "Series" else Table

    def returns() -> output[TrimmedRow]:  # type: ignore[valid-type]
        if returned == "frame":
            return pl.DataFrame({"name": [{"text": "  a  "}]})  # type: ignore[return-value]
        return [{"name": TrimmedName(text="  a  ")}]  # type: ignore[return-value]

    def takes(rows: Table[TrimmedRow]) -> int:
        got.extend(rows.to_dicts())
        return rows.height

    rows = node("rows", title="Rows", description="For a test.")(returns)
    counts = node("counts", title="Counts", description="For a test.")(takes)
    registry = NodeRegistry()
    registry.add(rows, counts)
    graph = Graph.compile(
        make_graph(rows, counts, edges=[("rows.result", "counts.rows")]), registry
    )
    run = graph.run({})
    result = run.results["rows"]["result"]

    if declared == "Series":
        assert result == [{"name": TrimmedName(text="a")}]
    else:
        assert result.to_dicts() == [{"name": {"text": "a"}}]
    assert got == [{"name": {"text": "a"}}]
    assert "  a  " not in run.to_json()
    assert Run.from_json(run.to_json(), graph).to_json() == run.to_json()


@pytest.mark.parametrize("in_a_list", [False, True], ids=["a model", "a list of models"])
@pytest.mark.parametrize("taken", ["Series", "Table"])
def test_a_frame_returned_for_a_series_of_rows_reads_each_cell_once(
    taken: str, in_a_list: bool
) -> None:
    """A frame's cell of a host's model is read by the model's check once, when the node returns it. A node wired to
    the Series, as a Series or as a table of the same rows, gets what that check gave."""
    calls: list[str] = []

    @host_type("counted-name", title="Counted name")
    class CountedName(BaseModel, frozen=True):
        text: str

        @field_validator("text")
        @classmethod
        def _counted(cls, text: str) -> str:
            calls.append(text)
            return text

    class CountedRow(TypedDict):
        name: CountedName

    class CountedListRow(TypedDict):
        name: list[CountedName]

    row: Any = CountedListRow if in_a_list else CountedRow
    cell: Any = [{"text": "a"}] if in_a_list else {"text": "a"}

    @node("rows", title="Rows", description="For a test.")
    def rows() -> Series[row]:  # type: ignore[valid-type]
        return pl.DataFrame({"name": [cell] * 10})  # type: ignore[return-value]

    taker: Any = Series if taken == "Series" else Table

    def takes(rows: taker[row]) -> int:  # type: ignore[valid-type]
        return len(rows)

    counts = node("counts", title="Counts", description="For a test.")(takes)
    registry = NodeRegistry()
    registry.add(rows, counts)
    graph = Graph.compile(
        make_graph(rows, counts, edges=[("rows.result", "counts.rows")]), registry
    )

    assert graph.run({}).outputs == {"result": 10}
    assert len(calls) == 10


class MaybeMoneyRow(TypedDict):
    money: Money | None


@pytest.mark.parametrize(
    "cells",
    [[Money(amount=1), None], [None, Money(amount=1)]],
    ids=["none later", "none first"],
)
@pytest.mark.parametrize(
    ("declared", "returned", "taken"),
    [
        ("Table", "list", None),
        ("Table", "frame", None),
        ("Series", "frame", None),
        ("Series", "list", "Table"),
        ("Series", "frame", "Table"),
    ],
    ids=[
        "table rows",
        "table frame",
        "series frame",
        "series list into a table",
        "series frame into a table",
    ],
)
def test_a_model_or_none_cell_holding_none_runs_and_reads_back(
    cells: list[Any], declared: str, returned: str, taken: str | None
) -> None:
    """A cell of ``Money | None`` may be ``None``, first or later in its column, whatever form the rows take."""
    output: Any = Series if declared == "Series" else Table
    got: list[Any] = []

    def returns() -> output[MaybeMoneyRow]:  # type: ignore[valid-type]
        if returned == "frame":
            return pl.DataFrame(  # type: ignore[return-value]
                {"money": [None if cell is None else {"amount": 1.0} for cell in cells]}
            )
        return [{"money": cell} for cell in cells]  # type: ignore[return-value]

    def takes(rows: Table[MaybeMoneyRow]) -> int:
        got.extend(rows.to_dicts())
        return rows.height

    rows = node("rows", title="Rows", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(rows)
    nodes, edges = [rows], []
    if taken:
        counts = node("counts", title="Counts", description="For a test.")(takes)
        registry.add(counts)
        nodes, edges = [rows, counts], [("rows.result", "counts.rows")]
    graph = Graph.compile(make_graph(*nodes, edges=edges), registry)
    run = graph.run({})
    result = run.results["rows"]["result"]
    kept = [{"money": None if cell is None else {"amount": 1.0}} for cell in cells]

    assert run.status == "complete"
    if declared == "Series":
        assert result == [{"money": cell} for cell in cells]
    else:
        assert result.to_dicts() == kept
    assert got == (kept if taken else [])
    assert Run.from_json(run.to_json(), graph).to_json() == run.to_json()


@host_type("priced", title="Priced")
class Priced(BaseModel, frozen=True):
    amount: float

    @field_serializer("amount")
    def _as_text(self, amount: float) -> str:
        return f"{amount:.2f}"


@host_type("doubled", title="Doubled")
class Doubled(BaseModel, frozen=True):
    a: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def double(self) -> int:
        return self.a * 2


@host_type("hidden", title="Hidden")
class Hidden(BaseModel, frozen=True):
    a: int
    secret: str = Field(exclude=True)


@host_type("noted-money", title="Noted money")
class NotedMoney(Money, frozen=True):
    note: str


class PricedRow(TypedDict):
    cell: Priced


class DoubledRow(TypedDict):
    cell: Doubled


class HiddenRow(TypedDict):
    cell: Hidden


#: A row type whose cell's class has a rule for how it is written, a value of it, and the cell as a table keeps it:
#: by the fields the class declares, as they are.
type _Wallet = Money


class MoniesPairRow(TypedDict):
    cell: tuple[Money, Money]


class MoneyListRow(TypedDict):
    cell: list[Money]


class WalletAliasRow(TypedDict):
    cell: _Wallet


_WRITTEN_BY_FIELDS = [
    (PricedRow, Priced(amount=1), {"amount": 1.0}),
    (DoubledRow, Doubled(a=1), {"a": 1}),
    (HiddenRow, Hidden(a=1, secret="s"), {"a": 1, "secret": "s"}),
    (MoneyRow, NotedMoney(amount=1, note="n"), {"amount": 1.0}),
]


@pytest.mark.parametrize(
    ("row", "value", "kept"),
    [
        *_WRITTEN_BY_FIELDS,
        (MoneyListRow, [NotedMoney(amount=1, note="n")], [{"amount": 1.0}]),
        (
            MoniesPairRow,
            (NotedMoney(amount=1, note="n"), Money(amount=2)),
            [{"amount": 1.0}, {"amount": 2.0}],
        ),
        (WalletAliasRow, NotedMoney(amount=1, note="n"), {"amount": 1.0}),
    ],
    ids=[
        "a field serializer",
        "a computed field",
        "an excluded field",
        "a subclass",
        "a subclass in a list",
        "a subclass in a tuple",
        "a subclass through an alias",
    ],
)
@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_table_keeps_a_cell_by_the_fields_its_type_declares_at_any_depth(
    row: Any, value: Any, kept: Any, returned: str
) -> None:
    """A cell is kept as its declared type's fields, as they are: a field serializer isn't applied, a computed field
    isn't added, an excluded field is kept, and a subclass is kept as the class its column's type declares, wherever
    the type puts it. Its saved run reads back the same."""
    (name,) = row.__annotations__

    def returns() -> Table[row]:  # type: ignore[valid-type]
        if returned == "frame":
            return pl.DataFrame({name: [kept]})  # type: ignore[return-value]
        return [{name: value}]  # type: ignore[return-value]

    table = node("table", title="Table", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(table)
    graph = Graph.compile(make_graph(table), registry)
    run = graph.run({})

    assert run.outputs["result"].to_dicts() == [{name: kept}]
    assert Run.from_json(run.to_json(), graph).to_json() == run.to_json()


class AnyRow(TypedDict):
    cell: Any


def _any_table(cells: list[Any]) -> Any:
    """A graph whose one node returns a ``Table[AnyRow]`` of ``cells``, one per row."""

    def returns() -> Table[AnyRow]:
        return [{"cell": one} for one in cells]  # type: ignore[return-value]

    table = node("table", title="Table", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(table)
    return Graph.compile(make_graph(table), registry)


@pytest.mark.parametrize(
    "cells",
    [
        [1, None, 2],
        ["a", "b"],
        [[1, 2], [3]],
        [{"x": 1, "y": None}, {"y": 2, "x": 3}],
        [datetime.date(2026, 1, 1)],
    ],
    ids=["whole numbers", "texts", "lists of one type", "dicts of the same keys", "dates"],
)
def test_an_any_cell_polars_gives_back_as_it_was_is_kept(cells: list[Any]) -> None:
    run = _any_table(cells).run({})

    assert run.outputs["result"]["cell"].to_list() == cells


@pytest.mark.parametrize(
    "cells",
    [
        [1, 1.5],
        [1, "x"],
        [True, 2],
        [[1], [1.5]],
        [{"x": 1}, {"y": 2}],
        [Money(amount=1)],
        [[Money(amount=1)]],
        [{"colour": Colour.RED}],
        [(1, 2)],
    ],
    ids=[
        "a whole number and a float",
        "a number and a text",
        "a yes or no and a number",
        "lists of different items",
        "dicts of different keys",
        "a model",
        "a list of models",
        "an enum in a dict",
        "a tuple",
    ],
)
def test_an_any_cell_polars_would_change_fails_the_run_naming_the_column(cells: list[Any]) -> None:
    """A column of ``Any`` gets the type polars picks for its values, so it is kept only if polars gives each value
    back as it was, of the same class: ``1`` beside ``1.5`` would come back as ``1.0``."""
    run = _any_table(cells).run({})

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("table",))
    assert [(e["type"], e["ctx"]["column"], e["ctx"]["cell"]) for e in run.cause.ctx["errors"]] == [
        ("table_cell_not_kept", "cell", "Any")
    ]
    assert "Give the column a type a table keeps, or pass the rows as a Series" in run.cause.msg


class _Part(TypedDict):
    a: int
    b: NotRequired[int]


class PartRow(TypedDict):
    part: _Part


class PartsRow(TypedDict):
    parts: list[_Part]


@pytest.mark.parametrize(
    ("row", "rows"),
    [
        (PartRow, [{"part": {"a": 1}}, {"part": {"a": 2, "b": 3}}]),
        (PartsRow, [{"parts": [{"a": 1}, {"a": 2, "b": 3}]}]),
    ],
    ids=["a row", "a list in a row"],
)
def test_a_series_of_rows_whose_cell_leaves_out_a_not_required_key_reads_back(
    row: Any, rows: list[dict[str, Any]]
) -> None:
    """A Series of rows given as a list may hold a ``TypedDict`` with a ``NotRequired`` field, which one value has
    and another leaves out. Each keeps its keys, so the saved run reads back as it was."""

    def returns() -> Series[row]:  # type: ignore[valid-type]
        return rows  # type: ignore[return-value]

    node_rows = node("rows", title="Rows", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(node_rows)
    graph = Graph.compile(make_graph(node_rows), registry)
    run = graph.run({})

    assert run.outputs == {"result": rows}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


@host_type("meowing", title="Meowing")
class Meowing(BaseModel, frozen=True):
    kind: Literal["meowing"] = "meowing"
    meow: str | None = None


@host_type("barking", title="Barking")
class Barking(BaseModel, frozen=True):
    kind: Literal["barking"] = "barking"
    bark: int


class PetRow(TypedDict):
    pet: Annotated[Meowing | Barking, Discriminator("kind")]


class ScoresRow(TypedDict):
    points: dict[str, int]


class MoneyColourRow(TypedDict):
    pair: tuple[Money, Colour]


def _as_plain_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, tuple):
        return [str(one) for one in value]
    return value


@pytest.mark.parametrize(
    ("row", "rows", "cell"),
    [
        (PetRow, [{"pet": Meowing(meow="m")}, {"pet": Barking(bark=1)}], "meowing | barking"),
        (ScoresRow, [{"points": {"x": 1}}, {"points": {"y": 2}}], "dict[str, int]"),
        (MoneyColourRow, [{"pair": (Money(amount=1), Colour.RED)}], "tuple[money, colour]"),
    ],
    ids=["a discriminated union of models", "a dict", "a tuple of different items"],
)
@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_series_of_rows_of_a_cell_a_table_does_not_keep_is_taken_as_a_list_and_refused_as_a_frame(
    row: Any, rows: list[Any], cell: str, returned: str
) -> None:
    """A Series of rows is a list, so its cells are kept as they are. A frame can't keep such a cell, so a frame for
    such rows is refused, naming the column, rather than read as other values."""

    def returns() -> Series[row]:  # type: ignore[valid-type]
        if returned == "frame":
            # each row as plain data polars holds: a model as its fields, a tuple's items as texts
            return pl.DataFrame(  # type: ignore[return-value]
                [{name: _as_plain_data(value) for name, value in one.items()} for one in rows]
            )
        return rows  # type: ignore[return-value]

    node_rows = node("rows", title="Rows", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(node_rows)
    graph = Graph.compile(make_graph(node_rows), registry)
    run = graph.run({})

    if returned == "list":
        assert run.outputs == {"result": rows}
        assert Run.from_json(run.to_json(), graph).outputs == run.outputs
    else:
        assert run.cause is not None
        assert (run.cause.type, [error["type"] for error in run.cause.ctx["errors"]]) == (
            "invalid_output",
            ["table_cell_not_kept"],
        )
        assert cell in run.cause.ctx["reason"]


type _Deep = list[_Deep]


class DeepRow(TypedDict):
    deep: _Deep


@pytest.mark.parametrize("returned", ["list", "frame"])
def test_a_table_cell_whose_type_holds_itself_is_kept_as_deep_as_its_value(returned: str) -> None:
    """``type Deep = list[Deep]`` holds itself. A cell of it is kept as deep as the value goes."""

    def returns() -> Table[DeepRow]:
        value = [[[]]]
        return pl.DataFrame({"deep": [value]}) if returned == "frame" else [{"deep": value}]  # type: ignore[return-value]

    table = node("table", title="Table", description="For a test.")(returns)
    registry = NodeRegistry()
    registry.add(table)
    graph = Graph.compile(make_graph(table), registry)
    run = graph.run({})

    assert run.outputs["result"].to_dicts() == [{"deep": [[[]]]}]
    assert Run.from_json(run.to_json(), graph).to_json() == run.to_json()


class NotedRow(TypedDict):
    text: Annotated[str, {"note": "a dict can't be hashed"}]


def test_a_table_column_whose_type_can_not_be_hashed_is_kept() -> None:
    """A column's type may hold what can't be hashed, such as a dict in its ``Annotated``. It is kept as any is."""

    @node("table", title="Table", description="For a test.")
    def table() -> Table[NotedRow]:
        return [{"text": "a"}]  # type: ignore[return-value]

    registry = NodeRegistry()
    registry.add(table)
    graph = Graph.compile(make_graph(table), registry)
    run = graph.run({})

    assert run.outputs["result"].to_dicts() == [{"text": "a"}]


class NotedMoneyRow(TypedDict):
    money: NotedMoney


@pytest.mark.parametrize("returned", ["list", "frame"])
def test_rows_of_a_wider_row_into_a_table_input_keep_only_the_tables_fields(returned: str) -> None:
    """A Series of rows whose cell is a subclass, wired to a table whose cell is the class, gives the table the
    class's fields only, from a list and from a frame alike."""
    got: list[Any] = []

    def returns() -> Series[NotedMoneyRow]:
        if returned == "frame":
            return pl.DataFrame({"money": [{"amount": 1.0, "note": "n"}]})  # type: ignore[return-value]
        return [{"money": NotedMoney(amount=1, note="n")}]

    def takes(rows: Table[MoneyRow]) -> int:
        got.extend(rows.to_dicts())
        return rows.height

    rows = node("rows", title="Rows", description="For a test.")(returns)
    counts = node("counts", title="Counts", description="For a test.")(takes)
    registry = NodeRegistry()
    registry.add(rows, counts)
    graph = Graph.compile(
        make_graph(rows, counts, edges=[("rows.result", "counts.rows")]), registry
    )
    run = graph.run({})

    assert (run.status, got) == ("complete", [{"money": {"amount": 1.0}}])


def test_a_frame_with_two_columns_of_python_objects_names_the_one_with_no_json_form() -> None:
    @node("table", title="Table", description="For a test.")
    def table() -> Table[TwoMoneyRow]:
        return pl.DataFrame(
            {
                "paid": pl.Series([Money(amount=1)], dtype=pl.Object),
                "owed": pl.Series([object()], dtype=pl.Object),
            }
        )

    registry = NodeRegistry()
    registry.add(table)
    refused = Graph.compile(make_graph(table), registry).run({})

    assert refused.cause is not None
    assert refused.cause.ctx["reason"] == "column 'owed' is Object, which can't be written as JSON"


@dataclasses.dataclass
class _PaidAndOwed:
    paid: TwoMoneyRow
    owed: TwoMoneyRow


def test_a_node_whose_two_outputs_hold_the_same_model_returns_both() -> None:
    """Two outputs whose types each hold the same model are checked together, as one return."""
    row: TwoMoneyRow = {"paid": Money(amount=1), "owed": Money(amount=2)}

    @node("accounts", title="Accounts", description="For a test.")
    def accounts() -> _PaidAndOwed:
        return _PaidAndOwed(paid=row, owed=row)

    registry = NodeRegistry()
    registry.add(accounts)

    assert Graph.compile(make_graph(accounts), registry).run({}).outputs == {
        "paid": row,
        "owed": row,
    }


class ColourRow(TypedDict):
    colour: Colour


class SizeRow(TypedDict):
    size: Literal["s", "m"]


@pytest.mark.parametrize(
    ("row", "returned", "rows"),
    [
        (ColourRow, [{"colour": Colour.RED}], [{"colour": Colour.RED}]),
        (SizeRow, [{"size": "s"}], [{"size": "s"}]),
        (SizeRow, pl.DataFrame({"size": ["s"]}), [{"size": "s"}]),
    ],
    ids=["enum", "literal", "literal from a table"],
)
def test_a_series_of_rows_with_named_values_in_a_cell_is_declared_and_runs(
    row: Any, returned: Any, rows: list[Any]
) -> None:
    """A row's cell may be an enum or a Literal. Both forms of a Series of rows are built in one build, so the cell's
    schema is built once."""

    @node("named", title="Named", description="For a test.")
    def named() -> Series[row]:  # type: ignore[valid-type]
        return returned  # type: ignore[no-any-return]

    registry = NodeRegistry()
    registry.add(named)
    graph = Graph.compile(make_graph(named), registry)
    run = graph.run({})

    assert run.outputs == {"result": rows}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


@host_type("label", title="Label")
@dataclasses.dataclass(frozen=True)
class Label:
    text: str


class StrippedRow(TypedDict):
    __pydantic_config__ = ConfigDict(str_strip_whitespace=True)
    label: Label


class LabelRow(TypedDict):
    label: Label


def test_a_plain_dataclass_in_a_row_is_read_by_the_rows_config_and_its_saved_run_reads_back_as_itself() -> (
    None
):
    """A plain dataclass takes the config of the row it sits in, whether a person answers the rows or a saved run
    reads them back."""

    @node("labels", title="Labels", description="For a test.", asks=True)
    def labels() -> Series[StrippedRow]:
        raise Ask("Which labels?")

    registry = NodeRegistry()
    registry.add(labels)
    graph = Graph.compile(make_graph(labels), registry)
    run = graph.run({})
    run = run.with_answers(run.waiting[0], [{"label": {"text": "  a  "}}]).resume()

    assert run.outputs == {"result": [{"label": Label("a")}]}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


def test_a_plain_dataclass_in_a_frame_for_a_series_of_rows_is_read_by_the_rows_config() -> None:
    """A frame's cell is read by the row's own schema of its field, so a plain dataclass takes the row's config there
    too."""

    @node("labels", title="Labels", description="For a test.")
    def labels() -> Series[StrippedRow]:
        return pl.DataFrame({"label": [{"text": "  a  "}]})  # type: ignore[return-value]

    registry = NodeRegistry()
    registry.add(labels)
    graph = Graph.compile(make_graph(labels), registry)
    run = graph.run({})

    assert run.outputs == {"result": [{"label": Label("a")}]}
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs


@dataclasses.dataclass
class _StrippedAndKept:
    stripped: StrippedRow
    kept: LabelRow


def test_a_return_whose_two_outputs_hold_one_plain_dataclass_under_two_configs_is_checked() -> None:
    """Each output of a return is checked by its own type, so two rows of different configs may hold one plain
    dataclass, which pydantic builds under each row's config."""
    stripped: StrippedRow = {"label": Label("a")}
    kept: LabelRow = {"label": Label("b")}

    @node("two-labels", title="Two labels", description="For a test.")
    def two_labels() -> _StrippedAndKept:
        return _StrippedAndKept(stripped=stripped, kept=kept)

    registry = NodeRegistry()
    registry.add(two_labels)
    graph = Graph.compile(make_graph(two_labels), registry)

    assert graph.run({}).outputs == {"stripped": stripped, "kept": kept}
