"""A kind of problem is declared by one function, and a signature the three parameter rules don't cover is refused there."""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any, Literal

import pytest
from conductor import PROBLEMS, Problem, ProblemCatalogue
from conductor.testing.problems import REGISTRY_CHECK_PROBLEMS
from conductor_providers.fastapi import ROUTER_PROBLEMS
from conductor_providers.fastapi.testing.problems import RUN_STORE_CHECK_PROBLEMS

catalogue = ProblemCatalogue()


@catalogue.problem()
def unknown_shelf(room: str, shelf: str, /, *, shelves: list[str], nearest: str | None = None) -> str:
    """A book is put on a shelf the room doesn't have.

    Args:
        room: The room.
        shelf: The name that is no shelf of it.
        shelves: The shelves the room has.
        nearest: The shelf whose name is closest, when one is close.
    """
    return f"'{room}' has no shelf '{shelf}'; it has {', '.join(shelves)}."


@catalogue.problem("unknown_shelf", variant="another_room")
def unknown_shelf_in_another_room(room: str, shelf: str, /, *, other: str) -> str:
    """A book is put on a shelf that is in another room.

    Args:
        room: The room.
        shelf: The name that is no shelf of it.
        other: The room that has the shelf.
    """
    return f"'{shelf}' is a shelf of '{other}', not of '{room}'."


@catalogue.problem()
def room_closed(*loc: str | int, until: str) -> str:
    """A room is closed.

    Args:
        loc: The room and the door.
        until: When it opens.
    """
    return f"The room is closed until {until}."


def test_a_kind_makes_the_problem_its_function_declares() -> None:
    made = unknown_shelf("study", "top", shelves=["low", "high"])

    assert made == Problem(type="unknown_shelf", loc=("study", "top"), msg="'study' has no shelf 'top'; it has low, high.", ctx={"room": "study", "shelf": "top", "shelves": ["low", "high"]})
    assert Problem.model_validate_json(made.model_dump_json()) == made


def test_a_key_that_may_be_left_out_is_in_ctx_only_when_it_is_given() -> None:
    assert "nearest" not in unknown_shelf("study", "top", shelves=[]).ctx
    assert unknown_shelf("study", "top", shelves=[], nearest="tip").ctx["nearest"] == "tip"


def test_a_variant_is_a_second_sentence_under_the_code_and_says_so_in_ctx() -> None:
    made = unknown_shelf_in_another_room("study", "top", other="hall")

    assert (made.type, made.msg, made.ctx) == ("unknown_shelf", "'top' is a shelf of 'hall', not of 'study'.", {"room": "study", "shelf": "top", "other": "hall", "variant": "another_room"})


def test_a_part_of_loc_is_given_by_position_only_so_loc_and_ctx_always_agree() -> None:
    """A part of ``loc`` given by name would be in ``ctx`` and missing from ``loc``; the signature refuses it."""
    with pytest.raises(TypeError, match="positional-only"):
        unknown_shelf(room="study", shelf="top", shelves=[])  # type: ignore[call-arg]


def test_a_place_the_sentence_does_not_read_is_in_loc_alone() -> None:
    made = room_closed("study", 2, until="noon")

    assert (made.loc, made.ctx) == (("study", 2), {"until": "noon"})


def test_a_catalogue_lists_its_kinds_and_their_codes() -> None:
    assert catalogue.codes == {"unknown_shelf", "room_closed"}
    assert [(kind.code, kind.variant) for kind in catalogue] == [("unknown_shelf", None), ("unknown_shelf", "another_room"), ("room_closed", None)]


def test_a_kind_has_a_sentence_with_a_replacement_field_for_each_ctx_key_and_a_type_for_its_ctx() -> None:
    assert unknown_shelf.sentence == "'{room}' has no shelf '{shelf}'; it has {shelves}."
    assert typing.get_type_hints(unknown_shelf.ctx_type) == {"room": str, "shelf": str, "shelves": list[str], "nearest": str}
    assert unknown_shelf.ctx_type.__required_keys__ == {"room", "shelf", "shelves"}
    assert typing.get_type_hints(unknown_shelf_in_another_room.ctx_type)["variant"] == Literal["another_room"]


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
def test_a_signature_outside_the_three_rules_is_refused_where_it_is_declared(say: Callable[..., Any], parameter: str) -> None:
    with pytest.raises(TypeError, match=rf"{say.__name__}.*'{parameter}'"):
        ProblemCatalogue().problem()(say)


def test_a_variant_for_a_code_the_catalogue_has_not_declared_is_refused() -> None:
    def lonely(*loc: str | int, shelf: str) -> str:
        return ""

    with pytest.raises(TypeError, match="lonely.*'no_such_code'"):
        ProblemCatalogue().problem("no_such_code", variant="near")(lonely)


def test_every_kinds_sentence_fills_from_its_ctx_keys_and_leaves_no_brace() -> None:
    for declared in (PROBLEMS, REGISTRY_CHECK_PROBLEMS, ROUTER_PROBLEMS, RUN_STORE_CHECK_PROBLEMS):
        for kind in declared:
            filled = kind.sentence.format(**{key: "x" for key in kind.ctx_type.__annotations__})

            assert "{" not in filled and "}" not in filled, kind.code
