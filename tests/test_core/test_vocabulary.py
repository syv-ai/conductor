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
from conductor import (
    DType,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Result,
    Series,
    TypeDescription,
    node,
)
from conductor._internal.registry import RegistryDescription
from conductor.widgets import Textarea


class Text(DType, str):
    id = "text"
    display_title = "Text"


class Number(DType, float):
    id = "number"
    display_title = "Number"


class AnotherText(DType, str):
    """A second class claiming ``text`` — what a second library, or a re-run cell, produces."""

    id = "text"
    display_title = "Text"


class Score(DType, float):
    """Admits a ``Number`` too, so its answer differs from the default."""

    id = "score"
    display_title = "Score"

    @classmethod
    def accepts(cls, source: Any) -> bool:
        return super().accepts(source) or issubclass(source, Number)


@node("shout", title="Shout", description="Uppercases", category="test")
def shout(
    text: Annotated[Text, Param(title="Text", widget=Textarea())],
    weights: Annotated[Series[Number], Param(title="Weights", widget=Textarea())],
    anything: Annotated[Any, Param(title="Anything", widget=Textarea())],
    tags: Annotated[list[str], Param(title="Tags", connectable=False, widget=Textarea())] = [],
) -> Annotated[Text, Result(title="Loud")]:
    """Declares ``Text`` per row, ``Number`` whole, a pass-through and a closed plain-typed input."""
    return Text(text.upper())


def _declare_text() -> type[DType]:
    """What a notebook cell does when it is run again: the same class body, a new class."""

    class Text(DType, str):
        id = "text"
        display_title = "Text"

    return Text


# --- the vocabulary is the registry's ---------------------------------------


def test_a_type_declared_twice_registers_on_two_registries():
    first, second = _declare_text(), _declare_text()
    assert first is not second and first.id == second.id == "text"

    a, b = NodeRegistry(types=[first]), NodeRegistry(types=[second])

    assert a.types == {"text": first}
    assert b.types == {"text": second}


def test_one_id_from_two_classes_on_one_registry_is_refused_naming_both():
    class StandardText(DType, str):
        id = "text"

    assert NodeRegistry(types=[Text, Text]).types == {"text": Text}  # the same class twice is not a collision

    with pytest.raises(NodeDeclarationError) as raised:
        NodeRegistry(types=[Text, StandardText])

    message = str(raised.value)
    assert "'text'" in message
    assert "StandardText" in message
    assert f"{Text.__module__}.Text" in message


def test_two_classes_with_one_id_in_one_call_are_refused_together():
    """A batch is checked against itself as well as against what is filed."""
    with pytest.raises(NodeDeclarationError, match="'text'"):
        NodeRegistry(types=[Text, AnotherText])

    @node("echo", title="Echo", description="Returns its input", category="test")
    def echo(text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[AnotherText, Result(title="Same")]:
        return AnotherText(text)

    with pytest.raises(NodeDeclarationError, match="'text'"):
        NodeRegistry([echo])


def test_register_adds_the_types_a_node_declares():
    """A scalar is a word, a ``Series[X]`` contributes ``X``, and ``Any`` and a closed input's plain type contribute nothing."""
    registry = NodeRegistry([shout])

    assert registry.types == {"text": Text, "number": Number}


def test_a_collision_through_register_names_both_classes_and_files_nothing():
    @node("whisper", title="Whisper", description="Lowercases", category="test")
    def whisper(text: Annotated[AnotherText, Param(title="Text", widget=Textarea())]) -> Annotated[AnotherText, Result(title="Quiet")]:
        return AnotherText(text.lower())

    registry = NodeRegistry([shout])
    with pytest.raises(NodeDeclarationError) as raised:
        registry = registry.extended_with([whisper])

    assert f"{Text.__module__}.Text" in str(raised.value)
    assert f"{AnotherText.__module__}.AnotherText" in str(raised.value)
    assert "whisper" not in registry
    assert registry.types == {"text": Text, "number": Number}


def test_a_series_is_not_a_word():
    """A series is many of something; the something is the word."""
    with pytest.raises(NodeDeclarationError, match="not a DType"):
        NodeRegistry(types=[Series[Text]])
    with pytest.raises(NodeDeclarationError, match="not a DType"):
        NodeRegistry(types=[Series])
    with pytest.raises(TypeError, match="concrete"):
        NodeRegistry(types=[DType])
    registry = NodeRegistry()

    registry = registry.extended_with([shout])
    assert Series not in registry.types.values()


def test_the_vocabulary_is_read_only_and_kept_by_extended_with():
    registry = NodeRegistry([shout])

    with pytest.raises(TypeError):
        registry.types["score"] = Score  # type: ignore[index]

    assert registry.extended_with([]).types == registry.types


# --- accepted_as -------------------------------------------------------------


def test_accepted_as_is_answered_over_the_registrys_words_self_included():
    both = NodeRegistry([], types=[Number, Score])
    assert both.accepted_as(Number) == ("number", "score")
    assert both.accepted_as(Score) == ("score",)

    alone = NodeRegistry([], types=[Number])
    assert alone.accepted_as(Number) == ("number",)


def test_a_bug_in_a_types_accepts_is_not_swallowed():
    class Broken(DType, str):
        id = "broken"
        display_title = "Broken"

        @classmethod
        def accepts(cls, source: Any) -> bool:
            raise TypeError("a bug in accepts")

    registry = NodeRegistry([], types=[Text, Broken])
    with pytest.raises(TypeError, match="a bug in accepts"):
        registry.accepted_as(Text)


# --- describe: the palette ---------------------------------------------------


def test_a_type_record_is_its_id():
    """``accepted_as`` is the registry's answer, so the record a field carries is the id alone."""
    assert Text.describe() == {"id": "text"}


def test_describe_is_the_palette_nodes_and_vocabulary():
    registry = NodeRegistry([shout], types=[Score])

    described = registry.describe()

    assert isinstance(described, RegistryDescription)
    assert described.nodes == (shout.describe(),)
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
            display_title = "Whole"

    class Whole(Number):  # noqa: F811 — the corrected declaration
        id = "whole"
        display_title = "Whole"

    assert Whole.id == "whole"


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
