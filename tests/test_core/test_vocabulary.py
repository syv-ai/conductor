"""A registry owns its type vocabulary.

Which value types exist is the host's decision, and the host makes it on
its ``NodeRegistry``: the types its nodes declare, plus what it adds with
``add_types``. Nothing is process-wide. So a notebook cell that declares
a type runs twice without complaint, two libraries in one process can
each name a ``text``, one id claimed by two classes on one registry is
refused where the second arrives, and "where may a value of this type
land?" (``accepted_as``) is answered over one registry's words and no
others.
"""

from __future__ import annotations

import pathlib
import re
from typing import Annotated, Any

import pytest
from conductor import NodeDefinition, NodeRegistry, Param, Result
from conductor.dtype import DType
from conductor.registry import RegistryDescription, TypeDescription
from conductor.series import Series
from conductor.widgets import Textarea


class Text(DType, str):
    id = "text"
    title = "Text"


class Number(DType, float):
    id = "number"
    title = "Number"


class AnotherText(DType, str):
    """A second class claiming ``text`` — what a second library, or a re-run cell, produces."""

    id = "text"
    title = "Text"


class Score(DType, float):
    """Admits a ``Number`` too, so its answer differs from the default."""

    id = "score"
    title = "Score"

    @classmethod
    def accepts(cls, source: Any) -> bool:
        return super().accepts(source) or issubclass(source, Number)


class Shout(NodeDefinition):
    """Declares ``Text`` per row, ``Number`` whole, a pass-through and a closed plain-typed input."""

    id = "shout"
    title = "Shout"
    description = "Uppercases"
    category = "test"

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        weights: Annotated[Series[Number], Param(title="Weights", widget=Textarea())],
        anything: Annotated[Any, Param(title="Anything", widget=Textarea())],
        tags: Annotated[list[str], Param(title="Tags", show_handle=False, widget=Textarea())] = [],
    ) -> Annotated[Text, Result(title="Loud")]:
        return Text(text.upper())


def _declare_text() -> type[DType]:
    """What a notebook cell does when it is run again: the same class body, a new class."""

    class Text(DType, str):
        id = "text"
        title = "Text"

    return Text


# --- the vocabulary is the registry's ---------------------------------------


def test_a_type_declared_twice_registers_on_two_registries():
    first, second = _declare_text(), _declare_text()
    assert first is not second and first.id == second.id == "text"

    a, b = NodeRegistry(), NodeRegistry()
    a.add_types(first)
    b.add_types(second)

    assert a.types == {"text": first}
    assert b.types == {"text": second}


def test_declaring_your_own_text_beside_the_standard_librarys_is_fine():
    """The README imports ``conductor_nodes.types.Text`` in one section and declares a ``Text`` in another."""
    from conductor_nodes.types import Text as StandardText

    class Text(DType, str):
        id = "text"
        title = "Text"

    assert Text.id == StandardText.id


def test_one_id_from_two_classes_on_one_registry_is_refused_naming_both():
    from conductor_nodes.types import Text as StandardText

    registry = NodeRegistry()
    registry.add_types(Text)
    registry.add_types(Text)  # the same class again is not a collision

    with pytest.raises(ValueError) as refused:
        registry.add_types(StandardText)

    message = str(refused.value)
    assert "'text'" in message
    assert "conductor_nodes.types.Text" in message
    assert f"{Text.__module__}.Text" in message
    assert registry.types == {"text": Text}


def test_two_classes_with_one_id_in_one_call_are_refused_together():
    """A batch is checked against itself as well as against what is filed."""
    registry = NodeRegistry()
    with pytest.raises(ValueError, match="'text'"):
        registry.add_types(Text, AnotherText)
    assert registry.types == {}

    class Echo(NodeDefinition):
        id = "echo"
        title = "Echo"
        description = "Returns its input"
        category = "test"

        def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[AnotherText, Result(title="Same")]:
            return AnotherText(text)

    with pytest.raises(ValueError, match="'text'"):
        registry.register(Echo)
    assert registry.types == {} and not registry.contains("echo")


def test_register_adds_the_types_a_node_declares():
    """A scalar is a word, a ``Series[X]`` contributes ``X``, and ``Any`` and a closed input's plain type contribute nothing."""
    registry = NodeRegistry()
    registry.register(Shout)

    assert registry.types == {"text": Text, "number": Number}


def test_a_collision_through_register_names_both_classes_and_files_nothing():
    class Whisper(NodeDefinition):
        id = "whisper"
        title = "Whisper"
        description = "Lowercases"
        category = "test"

        def run(self, text: Annotated[AnotherText, Param(title="Text", widget=Textarea())]) -> Annotated[AnotherText, Result(title="Quiet")]:
            return AnotherText(text.lower())

    registry = NodeRegistry()
    registry.register(Shout)
    with pytest.raises(ValueError) as refused:
        registry.register(Whisper)

    assert f"{Text.__module__}.Text" in str(refused.value)
    assert f"{AnotherText.__module__}.AnotherText" in str(refused.value)
    assert not registry.contains("whisper")
    assert registry.types == {"text": Text, "number": Number}


def test_a_series_is_not_a_word():
    """A series is many of something; the something is the word."""
    registry = NodeRegistry()
    with pytest.raises(TypeError, match="element"):
        registry.add_types(Series[Text])
    with pytest.raises(TypeError, match="element"):
        registry.add_types(Series)
    with pytest.raises(TypeError, match="concrete"):
        registry.add_types(DType)

    registry.register(Shout)
    assert Series not in registry.types.values()


def test_the_vocabulary_is_read_only_and_kept_by_extended_with():
    registry = NodeRegistry()
    registry.register(Shout)

    with pytest.raises(TypeError):
        registry.types["score"] = Score  # type: ignore[index]

    assert registry.extended_with({}).types == registry.types


# --- accepted_as -------------------------------------------------------------


def test_accepted_as_is_answered_over_the_registrys_words_self_included():
    both = NodeRegistry()
    both.add_types(Number, Score)
    assert both.accepted_as(Number) == ("number", "score")
    assert both.accepted_as(Score) == ("score",)
    assert both.accepted_as(Series[Number]) == ("number", "score")

    alone = NodeRegistry()
    alone.add_types(Number)
    assert alone.accepted_as(Number) == ("number",)


def test_a_bug_in_a_types_accepts_is_not_swallowed():
    class Broken(DType, str):
        id = "broken"
        title = "Broken"

        @classmethod
        def accepts(cls, source: Any) -> bool:
            raise TypeError("a bug in accepts")

    registry = NodeRegistry()
    registry.add_types(Text, Broken)
    with pytest.raises(TypeError, match="a bug in accepts"):
        registry.accepted_as(Text)


# --- describe: the palette ---------------------------------------------------


def test_a_type_record_is_its_id():
    """``accepted_as`` is the registry's answer, so the record a field carries is the id alone."""
    assert Text.describe() == {"id": "text"}
    assert Series[Text].describe() == {"id": "series", "of": {"id": "text"}}


def test_describe_is_the_palette_nodes_and_vocabulary():
    registry = NodeRegistry()
    registry.register(Shout)
    registry.add_types(Score)

    described = registry.describe()

    assert isinstance(described, RegistryDescription)
    assert described.nodes == (Shout.describe(),)
    assert described.types == (
        TypeDescription(id="text", title="Text", accepted_as=("text",)),
        TypeDescription(id="number", title="Number", accepted_as=("number", "score")),
        TypeDescription(id="score", title="Score", accepted_as=("score",)),
    )
    assert described.model_dump(mode="json")["types"][1] == {"id": "number", "title": "Number", "accepted_as": ["number", "score"]}


# --- a subclass names itself ------------------------------------------------


def test_a_subclass_without_its_own_id_is_refused():
    with pytest.raises(TypeError, match="own"):

        class Whole(Number):
            title = "Whole"

    class Whole(Number):  # noqa: F811 — the corrected declaration
        id = "whole"
        title = "Whole"

    assert Whole.id == "whole"


def test_a_parameterisation_is_marked_and_keeps_its_bases_id():
    """``Series[Text]`` is built by a marked factory: it carries ``series`` as its id on purpose."""
    assert Series[Text].id == "series"
    assert Series[Text].parameterises is Series
    assert Text.parameterises is None

    class Bag(DType):
        id = "bag"
        title = "Bag"

    class BagOfText(Bag, parameterises=Bag):
        pass

    assert BagOfText.id == "bag"
    registry = NodeRegistry()
    registry.add_types(BagOfText)
    assert registry.types == {"bag": Bag}


def test_the_mark_names_a_concrete_dtype_base_and_the_class_keeps_its_id():
    """No escape from naming yourself: the mark needs a real base, and a class
    with an id of its own is a type of its own. A self-named subclass of a
    parameterisation is a word again."""

    class Bag(DType):
        id = "bag"
        title = "Bag"

    with pytest.raises(TypeError, match="concrete"):

        class NoBase(DType, str, parameterises=str):
            title = "No base"

    with pytest.raises(TypeError, match="concrete"):

        class TheBase(DType, parameterises=DType):
            title = "The base"

    with pytest.raises(TypeError, match="own"):

        class Renamed(Bag, parameterises=Bag):
            id = "renamed"

    class BagOfText(Bag, parameterises=Bag):
        pass

    class Sack(BagOfText):
        id = "sack"
        title = "Sack"

    assert Sack.parameterises is None
    registry = NodeRegistry()
    registry.add_types(Sack)
    assert registry.types == {"sack": Sack}


# --- the README ---------------------------------------------------------------


def test_the_readmes_blocks_run_in_order_in_one_process():
    """A reader who runs the README top to bottom in one notebook hits no error from the library.

    A block that only illustrates — a fragment that names something the README
    never defines, or is not a complete statement — is skipped; every other
    block runs in one shared namespace, in order.
    """
    readme = pathlib.Path(__file__).parents[2] / "README.md"
    if not readme.exists():
        pytest.skip("the README is not in this tree")
    namespace: dict[str, Any] = {}
    for number, block in enumerate(re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)):
        try:
            code = compile(block, f"<README block {number}>", "exec")
        except SyntaxError:
            continue
        try:
            exec(code, namespace)
        except (NameError, ModuleNotFoundError):
            continue
