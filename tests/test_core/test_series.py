import dataclasses

import pytest
from conductor.dtype import DType
from conductor.series import Index, Series
from pydantic import BaseModel


class Text(DType, str):
    id = "series-test-text"
    title = "Text"


class Num(DType, float):
    id = "series-test-num"
    title = "Tal"


# --- a series reads as its values -----------------------------------------


def test_a_series_reads_as_its_values():
    s = Series[Text](Index("docs"), ["first", "second"])

    assert list(s) == ["first", "second"]
    assert s[0] == "first"
    assert len(s) == 2


def test_a_dense_root_series_covers_every_row_of_its_index():
    """A row is a path: ``(i,)`` on a root."""
    s = Series[Text](Index("docs"), ["a", "b", "c"])

    assert s.rows == ((0,), (1,), (2,))


def test_a_sparse_series_covers_a_subset():
    """What a lifted decision produces: fewer rows, no holes, same index."""
    s = Series[Text](Index("docs"), ["a", "c"], rows=((0,), (2,)))

    assert s.rows == ((0,), (2,))
    assert list(s) == ["a", "c"]


def test_rows_and_values_must_agree():
    with pytest.raises(ValueError, match="rows"):
        Series[Text](Index("docs"), ["a", "b"], rows=((0,),))


def test_a_row_is_a_path_of_ints():
    """A bare int where a path belongs is a mistake, not a shorthand."""
    with pytest.raises(TypeError, match="path"):
        Series[Text](Index("docs"), ["a"], rows=(0,))


def test_a_node_never_builds_a_series():
    """A node returns values; the engine says which index they live on.
    A fresh root exists for what the validator is handed with no index."""
    assert not hasattr(Series, "of")
    assert Index.fresh().parent is None
    assert Index.fresh() != Index.fresh()


# --- the index is lineage ---------------------------------------------------


def test_two_series_on_one_index_share_it():
    docs = Index("docs")
    a = Series[Text](docs, ["x", "y"])
    b = Series[Num](docs, [Num(1), Num(2)])

    assert a.index == b.index


def test_a_child_row_knows_its_parent_row():
    """What an unfold node creates: three employees from two applications. The
    row's path says which application, and nothing else has to."""
    docs = Index("docs")
    employees = Series[Text](Index("employees", parent=docs), ["a", "b", "c"], rows=((0, 0), (0, 1), (1, 0)))

    assert employees.index.parent == docs
    assert [row[:-1] for row in employees.rows] == [(0,), (0,), (1,)]


def test_a_root_index_has_no_parent():
    assert Index("docs").parent is None


def test_an_index_is_its_identity_and_lineage():
    """Compile names an index and the engine births rows on it
    — on the series, never on the index. One index, one object
    shape, compared by id."""
    docs = Index("docs")

    assert Index("unfold", parent=docs) == Index("unfold", parent=docs)
    assert len({Index("unfold", parent=docs), Index("unfold", parent=docs)}) == 1
    assert Index("docs") != Index("other")
    assert [f.name for f in dataclasses.fields(Index)] == ["id", "parent"]


# --- accepts: the gather direction -----------------------------------------


def test_a_scalar_into_a_series_input_gathers():
    """N scalar refs into one `Series[X]` input become a series."""
    assert Series[Text].accepts(Text) is True


def test_a_series_into_a_series_input_passes_whole():
    assert Series[Text].accepts(Series[Text]) is True


def test_a_series_input_judges_by_element():
    assert Series[Text].accepts(Num) is False
    assert Series[Text].accepts(Series[Num]) is False


def test_a_series_into_a_scalar_input_is_judged_by_its_element():
    """The type says whether the element fits. That the node then runs
    once per row is a fact about the shapes, and compile reads it
    — the type does not carry a second spelling of it."""
    assert Text.accepts(Series[Text]) is True
    assert Text.accepts(Series[Num]) is False


def test_a_series_of_any_is_a_reduction_typed_later():
    """``Series[Any]`` is what a reduction over whatever arrives declares —
    a "count" or "the only one" node receives the whole series. It is a real
    class like any parameterised series, its element is ``Any``, its wire
    form says so, and it is not in the vocabulary."""
    from typing import Any

    from conductor.dtype import registered_dtypes

    assert Series[Any].element is Any
    assert Series[Any].describe() == {"id": "series", "of": {"id": "any"}}
    assert Series[Any] not in registered_dtypes()


# --- a parameterised series is a real type ----------------------------------


def test_a_parameterised_series_is_a_real_type():
    """A real Python type, taken literally — and why `describe` can see its element.

    A `typing` alias forwards attribute access to its origin, so a
    classmethod on one cannot read what it was parameterised with.
    """
    assert isinstance(Series[Text], type)
    assert issubclass(Series[Text], Series)
    assert Series[Text] is Series[Text]
    assert Series[Text].element is Text


def test_a_series_declares_its_element_type_on_the_wire():
    assert Series[Text].describe() == {"id": "series", "of": {"id": "series-test-text", "accepted_as": ["series-test-text"]}}


def test_an_unparameterised_series_cannot_be_serialized():
    with pytest.raises(ValueError):
        Series.describe()


def test_a_series_of_series_does_not_exist():
    """Depth is a `Table` cell and unfolding, never nesting."""
    with pytest.raises(TypeError, match="does not exist"):
        Series[Series[Text]]


def test_a_parameterised_series_does_not_re_register_its_id():
    from conductor.dtype import registered_dtypes

    assert Series in registered_dtypes()
    assert Series[Text] not in registered_dtypes()


# --- pydantic -----------------------------------------------------------------


def test_pydantic_validates_the_elements_and_keeps_the_index():
    class M(BaseModel):
        kilder: Series[Text]

    docs = Index("docs")
    validated = M(kilder=Series[Text](docs, ["hej"], rows=((3,),))).kilder

    assert isinstance(validated[0], Text)
    assert validated.index == docs
    assert validated.rows == ((3,),)


def test_a_plain_sequence_validates_into_a_series_on_a_fresh_root():
    class M(BaseModel):
        kilder: Series[Text]

    validated = M(kilder=["a", "b"]).kilder

    assert list(validated) == ["a", "b"]
    assert validated.index.parent is None


def test_a_scalar_is_not_silently_wrapped_into_a_series():
    """A caller sending the wrong shape hears about it."""

    class M(BaseModel):
        kilder: Series[Text]

    with pytest.raises(Exception):
        M(kilder="not a series")


def test_a_bad_element_fails_validation_naming_the_field():
    class M(BaseModel):
        kilder: Series[Text]

    with pytest.raises(Exception) as exc:
        M(kilder=Series[Text](Index("docs"), [object()]))

    assert "kilder" in str(exc.value)


def test_a_series_dumps_with_its_index_and_rows():
    """A series on the wire is tagged with where its rows came from, so
    any consumer can group aligned series into one table. The index is
    carried whole — id, parent, parent rows — and never as a length."""
    from pydantic import TypeAdapter

    docs = Index("docs")
    s = Series[Text](Index("lines", parent=docs), ["a", "b"], rows=((0, 0), (1, 0)))

    assert TypeAdapter(Series[Text]).dump_python(s, mode="json") == {
        "index": {"id": "lines", "parent": {"id": "docs", "parent": None}},
        "rows": [[0, 0], [1, 0]],
        "values": ["a", "b"],
    }


def test_a_series_publishes_a_json_schema():
    from pydantic import TypeAdapter

    schema = TypeAdapter(Series[Text]).json_schema(mode="serialization")

    assert set(schema["properties"]) == {"index", "rows", "values"}


def test_a_series_of_wire_formed_values_dumps_each_through_its_own_schema():
    """A dtype with a wire form of its own — a file, a table — keeps it
    inside a series: the values travel through the element's schema."""
    from pydantic import TypeAdapter
    from pydantic_core import core_schema

    class Blob(DType):
        id = "series-test-blob"
        title = "Blob"
        __slots__ = ("data",)

        def __init__(self, data: str) -> None:
            self.data = data

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return core_schema.no_info_plain_validator_function(
                lambda v: v if isinstance(v, Blob) else Blob(v["data"]),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    lambda blob: {"data": blob.data}, when_used="json"
                ),
            )

    dumped = TypeAdapter(Series[Blob]).dump_python(Series[Blob](Index("x"), [Blob("a")]), mode="json")

    assert dumped["values"] == [{"data": "a"}]
