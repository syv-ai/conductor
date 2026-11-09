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
    Ref,
    Result,
    RunState,
    Series,
    Static,
    Table,
    node,
    table_with_columns,
)
from conductor.widgets import TextWidget
from helpers import answer


class Contact(TypedDict):
    name: str
    email: str


Contacts = Table[Contact]
CSV = [("Ida", "ida@x"), ("Bo", None), ("Nina", "nina@x")]


@node("csv", category="t")
def csv() -> Annotated[Contacts, Result(title="Contacts")]:
    return pl.DataFrame({"name": [name for name, _ in CSV], "email": [email for _, email in CSV]})


@node("send", category="t")
def send(email: str) -> Annotated[str, Result(title="sent")]:
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

@pytest.mark.parametrize("inner", ["rebuild", "keep"])
def test_a_table_rebuilt_inside_an_embedded_graph_fails_as_it_would_drawn_flat(inner: str) -> None:
    """The sub-flow that, wired beside its own source, used to give 6 rows with blanks on both sides and status complete."""
    registry = _catalogue()
    sub = _embedded("rb", (_node("k", inner), _node("s", "send", email=From("k.result.email"))), registry)
    registry = registry.extended_with([sub])
    compiled = _compiled(_node("c", "csv"), _node("sub", "rb", table=From("c.result")), _node("x", "export", table=From("c.result", "sub.result")), registry=registry)
    state = compiled.run_sync()
    flat = _compiled(_node("c", "csv"), _node("k", inner, table=From("c.result")), _node("s", "send", email=From("k.result.email")), _node("x", "export", table=From("c.result", "s.result")), registry=registry).run_sync()
    if inner == "rebuild":
        assert (state.status, state.failure.type, state.failure.loc[0]) == ("failed", "row_ids_lost", "sub/k")
        assert (flat.status, flat.failure.type) == ("failed", "row_ids_lost")
        assert state.failure.type == flat.failure.type
        return
    assert (state.status, flat.status) == ("complete", "complete")
    sheet = compiled.results(state)["x"]["result"].drop(ROW_ID_COLUMN)
    assert sheet.height == 3
    assert sheet["sent"].to_list() == ["sent:ida@x", None, "sent:nina@x"]


def test_rebuilt_table_nested_two_embeddings_deep_still_fails() -> None:
    registry = _catalogue()
    inner = _embedded("inner", (_node("k", "rebuild"),), registry)
    registry = registry.extended_with([inner])
    outer = _embedded("outer", (_node("i", "inner"), _node("s", "send", email=From("i.result.email"))), registry)
    registry = registry.extended_with([outer])
    compiled = _compiled(_node("c", "csv"), _node("sub", "outer", table=From("c.result")), _node("x", "export", table=From("c.result", "sub.result")), registry=registry)
    state = compiled.run_sync()
    assert (state.status, state.failure.type) == ("failed", "row_ids_lost")


def test_a_rebuilt_table_merged_with_a_column_made_per_row_of_it_is_fine() -> None:
    """M5's other side: the rebuilt table is merged with what was made from it, not with the table it came from."""
    registry = _catalogue()
    sub = _embedded("rb", (_node("k", "rebuild"), _node("s", "send", email=From("k.result.email")), _node("m", "export", table=From("k.result", "s.result"))), registry)
    registry = registry.extended_with([sub])
    compiled = _compiled(_node("c", "csv"), _node("sub", "rb", table=From("c.result")), registry=registry)
    sheet = _sheet(compiled, "sub/m")
    assert sheet.height == 3


def test_a_table_rebuilt_from_no_rows_inside_an_embedded_graph_loses_nothing() -> None:
    @node("no-rows", category="t")
    def no_rows() -> Annotated[Contacts, Result(title="Contacts")]:
        return pl.DataFrame(schema={"name": pl.String, "email": pl.String})

    registry = _catalogue(no_rows)
    sub = _embedded("rb", (_node("k", "rebuild"),), registry)
    registry = registry.extended_with([sub])
    compiled = _compiled(_node("c", "no-rows"), _node("sub", "rb", table=From("c.result")), _node("x", "export", table=From("c.result", "sub.result")), registry=registry)
    assert compiled.run_sync().status == "complete"


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


def test_scenario_6_a_filter_is_followed_by_a_per_row_node_on_its_columns() -> None:
    compiled = _compiled(
        _node("c", "csv"), _node("f", "filter", table=From("c.result")), _node("s", "send", email=From("f.result.email")),
        _node("x", "export", table=From("f.result", "s.result")), registry=_catalogue(filter_rows),
    )
    sheet = _sheet(compiled)
    assert sheet.to_dicts() == [
        {"name": "Ida", "email": "ida@x", "sent": "sent:ida@x"},
        {"name": "Nina", "email": "nina@x", "sent": "sent:nina@x"},
    ]


def test_scenario_7_sort_and_choose_columns_are_followed_by_per_row_nodes() -> None:
    compiled = _compiled(
        _node("c", "csv"), _node("o", "sort", table=From("c.result")), _node("p", "choose", table=From("o.result"), keep=Static("email")),
        _node("s", "send", email=From("p.result.email")), registry=_catalogue(sort_rows, choose_columns),
    )
    assert compiled.is_runnable, compiled.problems
    assert compiled.results(compiled.run_sync())["s"]["result"]["values"] == ["sent:nina@x", "sent:ida@x", None]


def test_arriving_merged_table_names_every_column_as_the_run_does() -> None:
    compiled = _compiled(
        _node("c", "csv"), _node("s", "send", email=From("c.result.email")), _node("s2", "send", email=From("c.result.name")),
        _node("x", "filter", table=From("c.result", "s.result", "s2.result")),
        registry=_catalogue(filter_rows),
    )
    declared = [name for name, _ in compiled.field(Ref("x", "result")).type.columns]
    assert declared == ["name", "email", "sent", "sent_s2"]
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    frame = compiled.results(state)["x"]["result"].drop(ROW_ID_COLUMN)
    assert frame.columns == declared
    assert frame["sent"].to_list() == ["sent:ida@x", "sent:nina@x"]


def test_a_table_arriving_on_an_input_with_columns_is_the_declared_type_the_call_receives() -> None:
    def outputs(table: ArrivingType) -> list[Output]:
        assert [name for name, _ in table.columns] == ["name", "email"]
        return [Output("result", table)]

    @node("typed-input", category="t", compute_outputs=outputs)
    def typed_input(table: Contacts) -> Mapping[str, Any]:
        return {"result": table}

    compiled = _compiled(_node("c", "csv"), _node("s", "send", email=From("c.result.email")), _node("x", "typed-input", table=From("c.result", "s.result")), registry=_catalogue(typed_input))
    assert compiled.run_sync().status == "complete"


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

    compiled = _compiled(_node("b", "bare"), _node("c", "csv"), _node("x", "arrives", table=From("b.result", "c.result")), registry=_catalogue(bare, arrives))
    assert compiled.is_runnable
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


def test_flow_2_merged_column_takes_the_output_title_a_clash_the_node_id() -> None:
    compiled = _compiled(
        _node("c", "csv"), _node("m", "send", email=From("c.result.email")), _node("x", "export", table=From("c.result", "m.result")),
    )
    assert _sheet(compiled).columns == ["name", "email", "sent"]


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


def test_flow_8a_answers_sit_beside_their_questions() -> None:
    files, questions = ["a.pdf", "b.pdf", "c.pdf"], ["who", "when"]
    compiled = _compiled(
        _node("d", "docs", file=Static(files)), _node("q", "ask", doc=From("d.text"), question=Static(questions)),
        _node("x", "export", table=From("q.question", "q.result")),
        registry=_catalogue(docs, ask_question),
    )
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    frame = compiled.results(state)["x"]["result"].drop(ROW_ID_COLUMN)
    assert frame.columns == ["question", "Answer"]
    assert frame.height == 6
    assert sorted(frame["question"].to_list()) == sorted(questions * 3)
    assert frame.filter(pl.col("question") == "who")["Answer"].to_list() == [f"text of {file} / who" for file in files]
    # Nothing of the typed-in list travels in the state.
    assert "question" not in state.nodes["q"].outputs
    restored = RunState.model_validate_json(state.model_dump_json())
    assert compiled.results(restored)["x"]["result"].drop(ROW_ID_COLUMN).equals(frame)


def test_a_series_of_typed_in_questions_read_under_each_document() -> None:
    @node("per-doc", category="t")
    def per_doc(doc: str, questions: Series[str]) -> Annotated[str, Result(title="Line")]:
        return f"{doc}:{','.join(questions)}"

    compiled = _compiled(
        _node("d", "docs", file=Static(["a.pdf", "b.pdf"])), _node("q", "ask", doc=From("d.text"), question=Static(["who", "when"])),
        _node("n", "per-doc", doc=From("d.name"), questions=From("q.question")),
        registry=_catalogue(docs, ask_question, per_doc),
    )
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert compiled.results(state)["n"]["result"]["values"] == ["a.pdf:who,when", "b.pdf:who,when"]


def test_a_typed_in_list_as_a_column_reads_by_a_per_row_node() -> None:
    compiled = _compiled(_node("q", "send", email=Static(["a", "b"])), _node("s", "send", email=From("q.email")))
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    assert compiled.results(state)["s"]["result"]["values"] == ["sent:a", "sent:b"]


def test_a_typed_in_list_read_by_a_run_that_restores() -> None:
    @node("wait", category="t")
    def wait(question: str) -> Annotated[str, Result(title="A")] | AskPerson:
        return AskPerson()

    compiled = _compiled(_node("w", "wait", question=Static(["a", "b"])), _node("x", "export", table=From("w.question", "w.result")), registry=_catalogue(wait))
    waiting = compiled.run_sync()
    assert waiting.status == "waiting"
    done = answer(compiled, RunState.model_validate_json(waiting.model_dump_json()), {unit.id: {"result": f"answer {unit.row[0]}"} for unit in waiting.waiting})
    finished = compiled.run_sync(done)
    assert compiled.results(finished)["x"]["result"].drop(ROW_ID_COLUMN).to_dicts() == [{"question": "a", "A": "answer 0"}, {"question": "b", "A": "answer 1"}]


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


def test_a_cell_polars_cannot_cast_goes_through_its_own_type() -> None:
    compiled = _compiled(_node("p", "priced", rows=Static([{"item": "a", "price": "1,5"}, {"item": "b", "price": "2,0"}])), registry=_catalogue(priced))
    assert compiled.is_runnable, compiled.problems
    frame = compiled.results(compiled.run_sync())["p"]["result"]
    assert frame["price"].dtype == pl.Float64
    assert frame["price"].to_list() == [1.5, 2.0]


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


def test_a_missing_column_at_run_time_is_named_in_a_structured_code() -> None:
    @node("bare-rows", category="t")
    def bare_rows() -> Table:
        return pl.DataFrame({"name": ["Ida"]})

    @node("needs-email", category="t")
    def needs_email(contacts: Contacts) -> str:
        return "ok"

    compiled = _compiled(_node("b", "bare-rows"), _node("n", "needs-email", contacts=From("b.result")), registry=_catalogue(bare_rows, needs_email))
    state = compiled.run_sync()
    assert (state.status, state.failure.type, state.failure.ctx["column"]) == ("failed", "table_column_missing", "email")


def test_scenario_20_an_empty_list_is_a_table_of_no_rows() -> None:
    calls: list[int] = []

    @node("count-rows", category="t")
    def count_rows(rows: Annotated[Contacts, Param(title="R", widget=TextWidget())]) -> Annotated[int, Result(title="N")]:
        calls.append(rows.height)
        return rows.height

    compiled = _compiled(_node("h", "count-rows"), registry=_catalogue(count_rows))
    filled = compiled.with_inputs(rows=[])
    state = filled.run_sync()
    assert state.status == "complete", state.failure
    assert filled.results(state)["h"]["result"] == 0
    assert calls == [0]


def test_a_list_of_records_is_one_table_not_many() -> None:
    compiled = _compiled(_node("h", "export", table=Static([{"a": "1"}, {"a": "2"}])))
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    result = compiled.results(state)["h"]["result"]
    assert isinstance(result, pl.DataFrame) and result.height == 2


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


def test_a_waiting_unit_carries_the_inputs_it_received_as_json() -> None:
    approve = node("approve", category="t")(_asks)
    compiled = _compiled(_node("c", "csv"), _node("a", "approve", vendor=From("c.result.name"), note=Static("hello")), registry=_catalogue(approve))
    state = compiled.run_sync()
    assert state.status == "waiting"
    unit = state.waiting[0]
    assert unit.prompt == "Approve Ida?"
    assert unit.inputs["vendor"] == "Ida"
    dumped = RunState.model_validate_json(state.model_dump_json())
    assert dumped.waiting[0].inputs == {"vendor": "Ida", "note": "hello"}


def test_a_waiting_unit_holds_a_table_and_a_series_it_received() -> None:
    @node("review", category="t")
    def review(values: Series[str], rows: Table) -> Annotated[str, Result(title="OK")] | AskPerson:
        return AskPerson()

    compiled = _compiled(_node("c", "csv"), _node("r", "review", values=From("c.result.name"), rows=From("c.result")), registry=_catalogue(review))
    state = compiled.run_sync()
    assert state.status == "waiting"
    written = RunState.model_validate_json(state.model_dump_json()).waiting[0].inputs
    assert written["values"] == ["Ida", "Bo", "Nina"]
    assert written["rows"]["name"].to_list() == ["Ida", "Bo", "Nina"]
