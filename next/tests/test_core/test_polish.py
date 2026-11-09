"""Silent wrong answers found by running real graphs, each pinned through the doors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    AskPerson,
    From,
    FromRun,
    Graph,
    InvalidCallError,
    NodeDeclarationError,
    Param,
    Policy,
    Result,
    Series,
    Static,
    Table,
    node,
    table_with_columns,
)
from pydantic import BaseModel


@dataclass(frozen=True)
class _RunBy:
    name: str


class _Amount(TypedDict):
    amount: int


@node("whole-amount")
def _whole_amount() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.75]})


@node("whole-number")
def _whole_number() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.0, None]})


@node("late-records")
def _late_records() -> Table:
    return [{"n": 1}] * 150 + [{"n": 2.5, "late": "x"}]  # type: ignore[return-value]


class _Inner(BaseModel):
    value: float


class _Reading(BaseModel):
    value: float
    maybe: float | None
    inner: _Inner


@node("reading")
def _reading() -> _Reading:
    return _Reading(value=math.nan, maybe=math.inf, inner=_Inner(value=-math.inf))


@node("read-reading")
def _read_reading(reading: _Reading) -> float:
    return reading.inner.value


class _Rule:
    """A host's own value, set on a closed input: no pydantic, no dataclass."""


@node("with-rule")
def _with_rule(rule: Annotated[_Rule, Param(connectable=False)] = _Rule()) -> str:  # noqa: B008
    return "x"


@dataclass
class _Branch:
    title: str


@node("with-branches")
def _with_branches(branches: Annotated[list[_Branch], Param(connectable=False)] = [_Branch("a")]) -> str:  # noqa: B006, B008
    return "x"


@node("people")
def _people() -> Series[str]:
    return ["Ida", "Bo", "Nina"]


@node("skip-bo")
def _skip_bo(name: str) -> str:
    return SKIPPED if name == "Bo" else name.upper()  # type: ignore[return-value]


@node("skip-nina")
def _skip_nina(name: str) -> str:
    return SKIPPED if name == "Nina" else name.lower()  # type: ignore[return-value]


@node("zip-pairs")
def _zip_pairs(upper: Series[str | None], lower: Series[str | None]) -> str:
    return ",".join(f"{a}/{b}" for a, b in zip(upper, lower, strict=True))


@node("zip-only-full")
def _zip_only_full(upper: Series[str], lower: Series[str]) -> str:
    return ",".join(f"{a}/{b}" for a, b in zip(upper, lower, strict=True))


@node("zip-open")
def _zip_open(**columns: Series) -> str:
    return ",".join("/".join(str(cell) for cell in row) for row in zip(*columns.values(), strict=True))


# -- a node that reads the world ----------------------------------------------


_FILE: dict[str, str] = {}
_CALLS: dict[str, int] = {}
_SEEN_ROWS: list[str] = []


@node("read-file", policy=Policy(reuse_when_inputs_match=False))
def _read_file(path: str) -> Series[str]:
    _CALLS["read"] = _CALLS.get("read", 0) + 1
    return _FILE[path].splitlines()


@node("mail-line")
def _mail_line(line: str) -> str:
    _SEEN_ROWS.append(line)
    return f"sent {line}"


@node("ask-line")
def _ask_line(line: str) -> str | AskPerson:
    return AskPerson()


# -- outputs that are alternatives ------------------------------------------------


class _Named(TypedDict):
    name: str


class _Route(TypedDict):
    route: str


@dataclass(frozen=True)
class _Routes:
    yes: Annotated[Table[_Named], Result(exclusive_group="r")]
    no: Annotated[Table[_Named], Result(exclusive_group="r")]


@node("route-once")
def _route_once(go: bool) -> _Routes:
    frame = pl.DataFrame({"name": ["taken"]})
    return _Routes(frame, SKIPPED) if go else _Routes(SKIPPED, frame)  # type: ignore[arg-type]


@node("flags")
def _flags() -> Series[bool]:
    return [True, False]


@node("show-table")
def _show_table(table: Table) -> str:
    return ",".join(table["name"].to_list()) + "|" + ",".join(name for name in table.columns if not name.startswith("__"))


@node("copy-table")
def _copy_table(table: Table[_Named]) -> Table[_Named]:
    return table


# -- what a merged column is called -----------------------------------------------


class _Contact(TypedDict):
    name: str
    email: str


@node("contacts")
def _contacts() -> Table[_Contact]:
    return pl.DataFrame({"name": ["Ida", "Bo", "Nina"], "email": ["i", "b", "n"]})


@node("first-two")
def _first_two(contacts: Table[_Contact]) -> Table[_Contact]:
    """Keeps two rows, and writes one blank and one new value into a column it carries on."""
    return contacts.head(2).with_columns(pl.Series("email", [None, "B"]))


@node("sent-to")
def _sent_to(email: str) -> Annotated[str, Result(title="Sendt")]:
    return f"to {email}"


@node("shout")
def _shout(name: str) -> str:
    return name.upper()


@node("sheet-of")
def _sheet_of(table: Table) -> Table:
    return table


# -- placing a node, in the checker's eye ---------------------------------------------


@node("open-text")
def _open_text(text: str, suffix: str) -> str:
    return text + suffix


@node("who-runs")
def _who_runs(greeting: str, who: FromRun[_RunBy]) -> str:
    return f"{greeting} {who.name}"


def test_with_values_types_a_value_in_anywhere_and_refuses_an_address_it_cant_set() -> None:
    graph = Graph(nodes=[_open_text.graph_node("a", text="x", suffix="!"), _open_text.graph_node("b", text=From("a.result"), suffix="?")])
    assert graph.with_values({"b.suffix": "!!"}).nodes[1].bindings["suffix"] == Static("!!")
    with pytest.raises(InvalidCallError) as refused:
        graph.with_values({"nope.text": 1, "b.text": 2, "bad": 3})
    assert [problem.type for problem in refused.value.errors()] == ["not_offered"] * 3


def test_a_table_typed_dict_may_say_a_column_can_be_blank() -> None:
    class Person(TypedDict):
        age: int | None

    assert Table[Person] is table_with_columns({"age": int})


# -- what an error says -------------------------------------------------------------


@node("three-letters")
def _three_letters() -> Series[str]:
    return ["a", "b", "c"]


@node("boom-at-one")
def _boom_at_one(n: str) -> str:
    if n == "b":
        raise ValueError("no")
    return n


def test_a_typo_in_graph_node_names_the_node_and_its_inputs() -> None:
    with pytest.raises(TypeError, match="'open-text' placed as 'x'.*text, suffix"):
        _open_text.graph_node("x", txt="y")  # type: ignore[call-arg]


def test_an_unknown_param_argument_is_called_unknown() -> None:
    with pytest.raises(NodeDeclarationError, match="no argument 'titel'"):
        Param(titel="Text")  # type: ignore[call-arg]


# -- series inputs on opposite branches -----------------------------------------------


@dataclass(frozen=True)
class _Parity:
    even: Annotated[int, Result(exclusive_group="parity")]
    odd: Annotated[int, Result(exclusive_group="parity")]


@node("numbers")
def _numbers() -> Series[int]:
    return [1, 2, 3, 4]


@node("parity")
def _parity(n: int) -> _Parity:
    return _Parity(n, SKIPPED) if n % 2 == 0 else _Parity(SKIPPED, n)  # type: ignore[arg-type]


@node("add-both")
def _add_both(a: Series[int], b: Series[int]) -> int:
    return sum(a) + sum(b)


@node("pair-both")
def _pair_both(a: Series[int | None], b: Series[int | None]) -> str:
    return ",".join(f"{x}/{y}" for x, y in zip(a, b, strict=True))


@node("add-all")
def _add_all(values: Series[int]) -> int:
    return sum(values)


_PARITY_NODES = [_numbers, _parity, _add_both, _pair_both, _add_all]


@node("plus-one")
def _plus_one(x: int) -> int:
    return x + 1


