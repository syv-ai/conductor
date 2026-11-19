"""Declaring a kind of problem with a function.

Each kind of problem is one function under ``@catalogue.problem``, and its code is the function's name. Its
parameters say where the problem is and what it carries. A positional-only parameter is a part of ``loc``, and a keyword-only one is a key of ``ctx``. The
string the function returns is the message. A function whose signature doesn't fit these rules is refused as soon
as it is declared.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any

import pytest
from conductor import DECLARATION_PROBLEMS, PROBLEMS, Problem, ProblemCatalogue
from conductor.testing.problems import REGISTRY_CHECK_PROBLEMS

catalogue = ProblemCatalogue()


@catalogue.problem
def unknown_shelf(
    room: str, shelf: str, /, *, shelves: list[str], nearest: str | None = None
) -> str:
    """A book is put on a shelf the room doesn't have.

    Args:
        room: The room.
        shelf: The name that is no shelf of it.
        shelves: The shelves the room has.
        nearest: The shelf whose name is closest, when one is close.
    """
    return f"'{room}' has no shelf '{shelf}'; it has {', '.join(shelves)}."


@catalogue.problem
def room_closed(*loc: str | int, until: str) -> str:
    """A room is closed.

    Args:
        loc: The room and the door.
        until: When it opens.
    """
    return f"The room is closed until {until}."


def test_a_kind_makes_the_problem_its_function_declares() -> None:
    made = unknown_shelf("study", "top", shelves=["low", "high"])

    assert made == Problem(
        type="unknown_shelf",
        loc=("study", "top"),
        msg="'study' has no shelf 'top'; it has low, high.",
        ctx={"room": "study", "shelf": "top", "shelves": ["low", "high"]},
    )
    assert Problem.model_validate_json(made.model_dump_json()) == made


def test_ctx_keys_come_in_the_order_the_function_declares_them() -> None:
    declared = unknown_shelf("study", "top", shelves=["low"], nearest="low")
    reversed_keywords = unknown_shelf("study", "top", nearest="low", shelves=["low"])

    assert list(declared.ctx) == ["room", "shelf", "shelves", "nearest"]
    assert list(reversed_keywords.ctx) == list(declared.ctx)
    assert reversed_keywords.model_dump_json() == declared.model_dump_json()


def test_a_key_that_may_be_left_out_is_in_ctx_only_when_it_is_given() -> None:
    assert "nearest" not in unknown_shelf("study", "top", shelves=[]).ctx
    assert unknown_shelf("study", "top", shelves=[], nearest="tip").ctx["nearest"] == "tip"


def test_a_part_of_loc_is_given_by_position_only_so_loc_and_ctx_always_agree() -> None:
    """A part of ``loc`` can't be passed by name.

    If it could, it would land in ``ctx`` and be missing from ``loc``.
    """
    with pytest.raises(TypeError, match="positional-only"):
        unknown_shelf(room="study", shelf="top", shelves=[])  # type: ignore[call-arg]


def test_a_place_the_sentence_does_not_read_is_in_loc_alone() -> None:
    made = room_closed("study", 2, until="noon")

    assert (made.loc, made.ctx) == (("study", 2), {"until": "noon"})


def test_a_catalogue_lists_its_kinds_and_their_codes() -> None:
    assert catalogue.codes == {"unknown_shelf", "room_closed"}
    assert [kind.code for kind in catalogue] == ["unknown_shelf", "room_closed"]


def test_a_kind_has_a_sentence_with_a_replacement_field_for_each_ctx_key_and_a_type_for_its_ctx() -> (
    None
):
    assert unknown_shelf.sentence == "'{room}' has no shelf '{shelf}'; it has {shelves}."
    assert typing.get_type_hints(unknown_shelf.ctx_type) == {
        "room": str,
        "shelf": str,
        "shelves": list[str],
        "nearest": str,
    }
    assert unknown_shelf.ctx_type.__required_keys__ == {"room", "shelf", "shelves"}


def _place_by_name(room: str, *, shelf: str) -> str:
    return ""


def _var_keyword(*loc: str | int, **shelves: str) -> str:
    return ""


def _default_other_than_none(*loc: str | int, shelf: str = "top") -> str:
    return ""


def _key_without_a_type(*loc: str | int, shelf) -> str:  # type: ignore[no-untyped-def]
    return ""


def _place_without_a_type(room, /, *, shelf: str) -> str:  # type: ignore[no-untyped-def]
    return ""


def _place_with_a_default(room: str = "study", /, *, shelf: str) -> str:
    return ""


def _place_that_is_no_place(room: list[str], /, *, shelf: str) -> str:
    return ""


def _places_of_another_type(*loc: float, shelf: str) -> str:
    return ""


def _no_sentence(*loc: str | int, shelf: str) -> int:
    return 0


@pytest.mark.parametrize(
    ("say", "parameter"),
    [
        (_place_by_name, "room"),
        (_var_keyword, "shelves"),
        (_default_other_than_none, "shelf"),
        (_key_without_a_type, "shelf"),
        (_place_without_a_type, "room"),
        (_place_with_a_default, "room"),
        (_place_that_is_no_place, "room"),
        (_places_of_another_type, "loc"),
        (_no_sentence, "return"),
    ],
)
def test_a_signature_outside_the_three_rules_is_refused_where_it_is_declared(
    say: Callable[..., Any], parameter: str
) -> None:
    with pytest.raises(TypeError, match=rf"{say.__name__}.*'{parameter}'"):
        ProblemCatalogue().problem(say)


def test_a_second_kind_with_the_same_code_is_refused_where_it_is_declared() -> None:
    """The second kind raises right away, so the module that declares it fails to import.

    That way no problems page and no host ever sees two kinds it can't tell apart.
    """

    def shelf_missing(*loc: str | int) -> str:
        return ""

    declared = ProblemCatalogue()
    declared.problem(shelf_missing)

    with pytest.raises(
        TypeError, match="shelf_missing: a kind with this code is declared before it"
    ):
        declared.problem(shelf_missing)


def test_every_kinds_sentence_fills_from_its_ctx_keys_and_leaves_no_brace() -> None:
    for declared in (PROBLEMS, REGISTRY_CHECK_PROBLEMS):
        for kind in declared:
            filled = kind.sentence.format(**{key: "x" for key in kind.ctx_type.__annotations__})

            assert "{" not in filled and "}" not in filled, kind.code


def test_ctx_error_is_only_a_developers_code_or_an_exceptions_class() -> None:
    """A kind takes ``error`` only when its docstring says the key holds a ``NodeError``'s ``code`` or an exception's
    class name. Any other text, such as pydantic's reason for a schema it can't build, is a ``reason``, and a value
    refusal says why in ``ctx.errors``."""
    for declared in (PROBLEMS, DECLARATION_PROBLEMS, REGISTRY_CHECK_PROBLEMS):
        for kind in declared:
            if "error" not in kind.ctx_type.__annotations__:
                continue
            said = " ".join(
                kind.__doc__.split("        error: ", 1)[1].split("\n        ")[0].split()
            )
            assert "class name" in said or ("``NodeError``" in said and "``code``" in said), (
                kind.code,
                said,
            )


def test_no_kind_takes_both_error_and_errors() -> None:
    """``ctx.error`` is a developer's code or an exception's class, and a value refusal says why in ``ctx.errors``,
    with pydantic's codes. So no kind has both: a key means one thing on every kind that has it."""
    both = [
        kind.code
        for declared in (PROBLEMS, DECLARATION_PROBLEMS, REGISTRY_CHECK_PROBLEMS)
        for kind in declared
        if {"error", "errors"} <= kind.ctx_type.__annotations__.keys()
    ]

    assert both == []


def test_joined_sections_keep_their_order_titles_and_texts_and_refuse_a_kind_two_of_them_declare() -> (
    None
):
    """Joining two catalogues keeps each kind under the section it was declared in.

    So a kind is never printed under another section's title.
    """
    hall, study = (
        ProblemCatalogue("Hall", "Met in the hall."),
        ProblemCatalogue("Study", "Met in the study."),
    )

    def door_locked(*loc: str | int) -> str:
        return "The door is locked; ask for the key."

    def shelf_full(*loc: str | int) -> str:
        return "The shelf is full; use another."

    hall.problem(door_locked)
    study.problem(shelf_full)
    house = ProblemCatalogue.joined(hall, study)

    assert ([(kind.code, kind.section) for kind in house], house.sections) == (
        [("door_locked", "Hall"), ("shelf_full", "Study")],
        {"Hall": "Met in the hall.", "Study": "Met in the study."},
    )
    with pytest.raises(TypeError, match="door_locked: a kind with this code is declared before it"):
        ProblemCatalogue.joined(hall, hall)
