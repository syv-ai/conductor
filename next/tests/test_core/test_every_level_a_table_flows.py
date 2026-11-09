"""The flows and scenarios of the every-level-a-table model that took more than one rule to get right, each run end to end through the public doors."""

from dataclasses import dataclass
from typing import Annotated, Any, Mapping, TypedDict

import polars as pl
import pytest
from conductor import (
    ROW_ID_COLUMN,
    ArrivingType,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeDefinition,
    NodeRegistry,
    Output,
    Param,
    Result,
    Static,
    Table,
    node,
    table_with_columns,
)
from conductor.widgets import TextWidget


class Contact(TypedDict):
    name: str
    email: str


Contacts = Table[Contact]
CSV = [("Ida", "ida@x"), ("Bo", None), ("Nina", "nina@x")]


@node("csv", category="t")
def csv() -> Annotated[Contacts, Result(title="Contacts")]:
    return pl.DataFrame({"name": [name for name, _ in CSV], "email": [email for _, email in CSV]})


@node("send", category="t")
def send(email: str) -> Annotated[str, Result(title="result")]:
    return f"sent:{email}"


@node("export", category="t")
def export(table: Table) -> Annotated[Table, Result(title="Sheet")]:
    return table


@node("rebuild", category="t")
def rebuild(table: Contacts) -> Annotated[Contacts, Result(title="Contacts")]:
    """Returns the same rows without their ids: a node that rebuilt its table, as a join or a CSV round trip does."""
    return table.drop(ROW_ID_COLUMN)


@node("keep", category="t")
def keep(table: Contacts) -> Annotated[Contacts, Result(title="Contacts")]:
    return table.filter(pl.col("email").is_not_null())


def _catalogue(*extra: type[NodeDefinition]) -> NodeRegistry:
    return NodeRegistry([csv, send, export, rebuild, keep, *extra])


def _node(node_id: str, node_type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def _compiled(*nodes: GraphNode, registry: NodeRegistry | None = None) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _catalogue())
    return compiled


def _sheet(compiled: CompiledGraph, node_id: str = "x") -> pl.DataFrame:
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    return compiled.results(state)[node_id]["result"].drop(ROW_ID_COLUMN)


def _embedded(node_id: str, nodes: tuple[GraphNode, ...], registry: NodeRegistry) -> NodeDefinition:
    return NodeDefinition.from_compiled_graphs(node_id, title=node_id, category="t", description="d", versions={1: CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry)})


# -- a table rebuilt inside an embedded graph --------------------------------------


# -- what a Table input carries -----------------------------------------------------

def _table_then(table: ArrivingType) -> list[Output]:
    return [Output("result", table)]


@node("filter", category="t", compute_outputs=_table_then)
def filter_rows(table: Table) -> Mapping[str, Any]:
    return {"result": table.filter(pl.col("email").is_not_null())}


@node("sort", category="t", compute_outputs=_table_then)
def sort_rows(table: Table) -> Mapping[str, Any]:
    return {"result": table.sort("name", descending=True)}


def _chosen(table: ArrivingType, keep: str) -> list[Output]:
    return [Output("result", table_with_columns({name: dtype for name, dtype in table.columns if name in keep.split(",")}))]


@node("choose", category="t", compute_outputs=_chosen)
def choose_columns(table: Table, keep: Annotated[str, Param(title="Keep", connectable=False, widget=TextWidget())]) -> Mapping[str, Any]:
    return {"result": table.select(ROW_ID_COLUMN, *keep.split(","))}


def test_a_bare_table_source_arrives_bare() -> None:
    @node("bare", category="t")
    def bare() -> Table:
        return pl.DataFrame({"a": [1]})

    seen: list[Any] = []

    def outputs(table: ArrivingType) -> list[Output]:
        seen.append(table)
        return [Output("result", table)]

    @node("arrives", category="t", compute_outputs=outputs)
    def arrives(table: Table) -> Mapping[str, Any]:
        return {"result": table}

    _compiled(_node("b", "bare"), _node("c", "csv"), _node("x", "arrives", table=From("b.result", "c.result")), registry=_catalogue(bare, arrives))
    assert seen == [Table]


def test_a_typed_in_frame_arrives_with_the_columns_it_holds() -> None:
    seen: list[Any] = []

    def outputs(value: ArrivingType) -> list[Output]:
        seen.append(value)
        return [Output("result", value)]

    @node("hold", category="t", compute_outputs=outputs)
    def hold(value: Annotated[Table, Param(title="T", widget=TextWidget())]) -> Mapping[str, Any]:
        return {"result": value}

    compiled = _compiled(_node("h", "hold", value=Static([{"a": "1", "n": 2}])), _node("s", "send", email=From("h.result.a")), registry=_catalogue(hold))
    assert [(name, dtype.wrapped_type) for name, dtype in seen[0].columns] == [("a", str), ("n", int)]
    assert compiled.is_runnable, compiled.problems


# -- Flow 8a: a typed-in list is a column ----------------------------------------------

@dataclass(frozen=True)
class Doc:
    name: Annotated[str, Result(title="Name")]
    text: Annotated[str, Result(title="Text")]


@node("docs", category="t")
def docs(file: str) -> Doc:
    return Doc(name=file, text=f"text of {file}")


@node("ask", category="t")
def ask_question(doc: str, question: str) -> Annotated[str, Result(title="Answer")]:
    return f"{doc} / {question}"


# -- a table read from outside ---------------------------------------------------------

class Danish(DType, float):
    """A number written the way a Dane writes it, ``1,5``."""

    id = "danish-number"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        def read(value: Any) -> Danish:
            return cls(float(value.replace(",", ".")) if isinstance(value, str) else value)

        return core_schema.no_info_plain_validator_function(read)


class Priced(TypedDict):
    item: str
    price: Danish


@node("priced", category="t")
def priced(rows: Annotated[Table[Priced], Param(title="Rows", widget=TextWidget())]) -> Annotated[Table[Priced], Result(title="Rows")]:
    return rows


def test_a_cell_its_type_cannot_read_is_refused_naming_the_column() -> None:
    compiled = _compiled(_node("p", "priced", rows=Static([{"item": "a", "price": "abc"}])), registry=_catalogue(priced))
    (found,) = compiled.problems
    assert found.type == "invalid_static"
    assert "price" in found.msg


def test_scenario_21_a_missing_column_is_named_in_a_structured_code() -> None:
    @node("app-input", category="t")
    def app_input(contacts: Annotated[Contacts, Param(title="Contacts", widget=TextWidget())]) -> Annotated[Contacts, Result(title="C")]:
        return contacts

    compiled = _compiled(_node("h", "app-input"), registry=_catalogue(app_input))
    with pytest.raises(InvalidCallError) as raised:
        compiled.with_inputs(contacts=[{"name": "Ida"}])
    (found,) = raised.value.errors()
    assert (found.type, found.ctx["column"]) == ("table_column_missing", "email")
    assert "email" in found.msg


# -- two lists, and a waiting unit's inputs ---------------------------------------------

def test_two_lists_typed_into_a_node_are_the_one_list_refusal_wired_or_not() -> None:
    @node("two", category="t")
    def two(a: str, b: str) -> str:
        return a + b

    compiled = _compiled(_node("t", "two", a=Static(["1", "2"]), b=Static(["3", "4"])), registry=_catalogue(two))
    (found,) = compiled.problems
    assert found.type == "lists_on_a_per_row_node"


def _asks(vendor: str, note: str = "n") -> Annotated[str, Result(title="Approved by")] | AskPerson:
    return AskPerson(prompt=f"Approve {vendor}?")


