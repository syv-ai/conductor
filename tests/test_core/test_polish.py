"""Silent wrong answers found by running real graphs, each pinned through the doors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
from conductor import (
    SKIPPED,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Param,
    RunState,
    Series,
    Table,
    node,
)
from pydantic import BaseModel


def _compiled(*placed: GraphNode, nodes: list[Any]) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(placed)), NodeRegistry(nodes))


class _Amount(TypedDict):
    amount: int


@node("whole-amount")
def _whole_amount() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.75]})


def test_a_fraction_in_an_int_column_fails_the_node_rather_than_being_cut_off() -> None:
    compiled = _compiled(_whole_amount.graph_node("a"), nodes=[_whole_amount])
    state = compiled.run_sync()
    assert state.status == "failed"
    assert state.failure is not None and state.failure.type == "invalid_output"


@node("whole-number")
def _whole_number() -> Table[_Amount]:
    return pl.DataFrame({"amount": [12.0, None]})


def test_a_whole_number_held_as_a_float_reads_as_an_int() -> None:
    compiled = _compiled(_whole_number.graph_node("a"), nodes=[_whole_number])
    assert compiled.results(compiled.run_sync())["a"]["result"]["amount"].to_list() == [12, None]


@node("late-records")
def _late_records() -> Table:
    return [{"n": 1}] * 150 + [{"n": 2.5, "late": "x"}]  # type: ignore[return-value]


def test_records_are_read_by_all_their_rows_not_the_first_hundred() -> None:
    compiled = _compiled(_late_records.graph_node("a"), nodes=[_late_records])
    frame = compiled.results(compiled.run_sync())["a"]["result"]
    assert frame["n"].to_list()[-1] == 2.5
    assert frame["late"].to_list()[-1] == "x"


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


def test_a_float_that_isnt_finite_in_a_model_survives_a_saved_state() -> None:
    compiled = _compiled(_reading.graph_node("a"), _read_reading.graph_node("b", reading=From("a.result")), nodes=[_reading, _read_reading])
    state = RunState.model_validate_json(compiled.run_sync().model_dump_json())
    restored = compiled.results(state)["a"]["result"]
    assert math.isnan(restored.value)
    assert restored.maybe == math.inf
    assert restored.inner.value == -math.inf
    assert compiled.run_sync(state).status == "complete"


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


def test_describe_writes_as_json_whatever_a_host_set_on_a_closed_input() -> None:
    compiled = _compiled(_with_rule.graph_node("a"), _with_branches.graph_node("b"), nodes=[_with_rule, _with_branches])
    written = compiled.describe().model_dump_json()
    assert '"title":"a"' in written
    NodeRegistry([_with_rule, _with_branches]).describe().model_dump_json()


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


def _two_columns_with_skips(reader: Any, **wired: Any) -> CompiledGraph:
    return _compiled(
        _people.graph_node("p"),
        _skip_bo.graph_node("up", name=From("p.result")),
        _skip_nina.graph_node("low", name=From("p.result")),
        reader,
        nodes=[_people, _skip_bo, _skip_nina, _zip_pairs, _zip_only_full, _zip_open],
    )


def test_two_series_with_skips_on_different_rows_keep_their_rows_as_blanks() -> None:
    compiled = _two_columns_with_skips(_zip_pairs.graph_node("z", upper=From("up.result"), lower=From("low.result")))
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida,None/bo,NINA/None"


def test_a_row_skipped_on_one_series_leaves_every_series_that_takes_no_blank() -> None:
    compiled = _two_columns_with_skips(_zip_only_full.graph_node("z", upper=From("up.result"), lower=From("low.result")))
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida"


def test_an_open_series_input_keeps_every_edge_in_step() -> None:
    placed = GraphNode(id="z", type="zip-open", version=1, bindings={"a": From("up.result"), "b": From("low.result")})
    compiled = _two_columns_with_skips(placed)
    assert compiled.results(compiled.run_sync())["z"]["result"] == "IDA/ida,None/bo,NINA/None"
