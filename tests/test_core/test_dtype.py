"""A type carries what it is and what it admits."""

from typing import Any

import pytest
from conductor.dtype import DType, Single, dtype_of, registered_dtypes
from pydantic import BaseModel


class Text(DType, str):
    """A host-side type, defined here because conductor ships none.

    The id is test-scoped: the dtype registry is process-global, and
    ``conductor_nodes.types.Text`` claims ``"text"`` when the whole suite
    runs in one process.
    """

    id = "dtype-test-text"
    title = "Text"


class Colour(DType):
    """A type built on nothing — validated by instance, not by a builtin."""

    id = "dtype-test-colour"
    title = "Colour"

    def __init__(self, name):
        self.name = name


def test_a_dtype_declares_its_id_and_its_title():
    assert Text.id == "dtype-test-text"
    assert Text.title == "Text"


def test_a_dtype_declares_no_widget():
    """A type does not get to choose the control an input renders as."""
    assert not hasattr(Text, "default_widget")


def test_a_dtype_has_no_conversions():
    """A value arrives as the type the wire carried."""
    assert not hasattr(Text, "converts_to")
    assert not hasattr(Text, "convert")


def test_a_type_renders_its_own_text():
    """The one hook a user-facing rendering calls. The default is ``str``;
    a host whose values read differently overrides it on the type."""
    assert Text.as_text(Text("hello")) == "hello"
    assert Colour.as_text(3.5) == "3.5"


def test_a_dtype_is_a_real_python_type():
    """So a type checker sees it and isinstance works."""
    assert isinstance(Text("hello"), str)
    assert Text("hello") == "hello"


def test_pydantic_validates_into_the_subclass_not_the_builtin():
    """The one piece of help pydantic needs: a str subclass is unknown to it."""

    class M(BaseModel):
        text: Text

    validated = M(text="hello").text
    assert isinstance(validated, Text)
    assert M(text=Text("hello")).text == "hello"


def test_a_dtype_built_on_nothing_validates_by_instance():
    class M(BaseModel):
        colour: Colour

    assert isinstance(M(colour=Colour("red")).colour, Colour)
    with pytest.raises(Exception):
        M(colour="red")


def test_a_dtype_must_declare_an_id_and_a_title():
    """Drift is prevented at definition, not at review."""
    with pytest.raises(TypeError, match="id"):

        class Nameless(DType, str):
            title = "No id"


def test_two_unrelated_dtypes_cannot_share_an_id():
    with pytest.raises(ValueError, match="already"):

        class Duplicate(DType, str):
            id = "dtype-test-text"
            title = "Another text"


def test_every_dtype_serializes_as_an_object():
    """Never a string — nothing downstream parses a type."""
    assert Text.describe() == {"id": "dtype-test-text", "accepted_as": ["dtype-test-text"]}


def test_accepted_as_is_every_type_that_admits_this_one_self_included():
    """The record precomputes the editor's applicability question — where
    may this value land? — off ``accepts`` itself, so an override that
    widens a type's welcome is served without a browser dtype catalog."""

    class Grade(DType, int):
        id = "dtype-test-grade"
        title = "Grade"

    class Score(DType, float):
        id = "dtype-test-score"
        title = "Score"

        @classmethod
        def accepts(cls, source: Any) -> bool:
            return super().accepts(source) or issubclass(source, Grade)

    assert Grade.describe() == {"id": "dtype-test-grade", "accepted_as": ["dtype-test-grade", "dtype-test-score"]}
    assert Score.describe() == {"id": "dtype-test-score", "accepted_as": ["dtype-test-score"]}


def test_a_type_is_not_authorable_unless_it_says_so():
    """The host's derived choice lists — what may a person type into a
    cell, a form answer, a schema field — read this flag and nothing else."""
    assert DType.authorable is False
    assert Text.authorable is False


def test_a_scalar_has_no_element():
    """What `accepts` reads to tell a scalar from a collection."""
    assert Text.element is None


def test_a_type_refuses_nothing_whole_unless_it_says_so():
    """The one question compile asks a source before binding it whole."""
    assert Text.refuses_whole() is None and Colour.refuses_whole() is None


def test_the_base_has_no_id():
    """Nothing on a wire ever carries "could not say". A pass-through
    declares ``Any``, which compile types from the wire — so the base is an
    ABC and nothing more, and a subclass that forgets its id inherits none."""
    assert not hasattr(DType, "id")
    with pytest.raises(TypeError, match="id"):

        class Inherits(DType, str):
            title = "Arver"


# --- accepts: the one question ------------------------------------------


def test_a_type_accepts_itself_and_its_subclasses():
    class Kort(Text):
        pass

    assert Text.accepts(Text) is True
    assert Text.accepts(Kort) is True


def test_a_type_rejects_an_unrelated_type():
    assert Text.accepts(Colour) is False
    assert Colour.accepts(Text) is False


def test_there_are_two_answers():
    """Nothing on a wire says "could not say" — a pass-through declares
    ``Any`` and compile types it from the wire — so the answer is a bool,
    never a verdict with a third state."""
    assert isinstance(Text.accepts(Text), bool)
    assert isinstance(Text.accepts(Colour), bool)


def test_the_base_type_itself_cannot_be_a_target():
    """An input declared as bare `DType` would accept anything, which is a
    declaration with no type in it."""
    with pytest.raises(TypeError, match="declare"):
        DType.accepts(Text)


# --- Any: the unconstrained input, and Single ------------------------


def test_any_is_the_unconstrained_marker_and_is_not_a_type():
    """``Any`` stands for "whatever arrives"; it is not a type, so it is
    not in the vocabulary and no value ever carries it."""
    from typing import Annotated, Any

    assert Any not in registered_dtypes()
    assert dtype_of(Annotated[Any, object()]) is Any


def test_single_is_a_bare_marker():
    """``**inputs: Single`` — the open roster. ``Single`` is the whole
    declaration; what it means is ``Interface.of`` to read."""
    assert Single() is not None


# --- dtype_of --------------------------------------------------------------


def test_dtype_of_reads_the_annotation():
    from typing import Annotated

    assert dtype_of(Text) is Text
    assert dtype_of(Annotated[Text, object()]) is Text


def test_dtype_of_returns_None_for_a_plain_python_type():
    """AKA's catalog contract test is what turns this into an error."""
    assert dtype_of(str) is None


def test_conductor_declares_no_domain_types_of_its_own():
    """The ABC is conductor's, the vocabulary is the host's.

    `Series` is the exception and is deliberate — a collection is structure,
    not domain: it means the same thing in every host.
    """
    from conductor.dtype import registered_dtypes
    from conductor.series import Series

    shipped = {
        d for d in registered_dtypes()
        if d.__module__.startswith("conductor.")
    }

    assert shipped == {Series}


def test_the_vocabulary_is_importable_from_the_root():
    import conductor

    for name in ("DType", "DTypeRef", "Single", "Series", "Index", "Ref", "Result"):
        assert getattr(conductor, name) is not None


def test_nothing_named_many_or_spread_is_exported():
    import conductor

    assert not hasattr(conductor, "Many")
    assert not hasattr(conductor, "Spread")
