"""One record for what goes wrong, and who raises what."""

from typing import Annotated, Any

import pytest
from conductor import (
    AsItArrives,
    ComputeHookError,
    DType,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRegistry,
    Param,
    Problem,
    Result,
    node,
)
from conductor._internal.errors import ConductorError, make_problem


class Txt(DType, str):
    id = "errors_txt"
    display_title = "Txt"


@node("echo", title="Echo", description="d", category="c")
def echo(text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
    return text


def _registry(*nodes: Any) -> NodeRegistry:
    return NodeRegistry([echo, *nodes])


def test_a_problem_is_the_four_keys_and_its_fatality_is_its_codes():
    told = make_problem("no_outputs", "a")

    assert (told.type, told.loc, told.ctx) == ("no_outputs", ("a",), {})
    assert told.fatal is False
    assert make_problem("unbound_required", "a", "x").fatal is True


def test_a_code_the_catalogue_does_not_know_is_fatal():
    assert Problem(type="the_hosts_own", msg="m").fatal is True


def test_fatality_cannot_be_set_on_a_record():
    """No value is stored, so nothing can disagree with the code's row."""
    told = Problem(type="no_outputs", msg="m", loc=("a",))

    assert told.model_copy(update={"msg": "n"}).fatal is False
    assert Problem.model_validate({"type": "no_outputs", "msg": "m", "fatal": True}).fatal is False


def test_a_dump_has_fatal_and_reads_back():
    told = make_problem("unknown_locked_field", "a", "x")
    dumped = told.model_dump(mode="json")

    assert dumped == {"type": "unknown_locked_field", "msg": told.msg, "loc": ["a", "x"], "ctx": {}, "fatal": False}
    assert Problem.model_validate(dumped) == told


def test_a_none_place_is_left_out_of_loc():
    assert make_problem("unbound_required", "a", None).loc == ("a",)


def test_the_errors_a_caller_catches_sit_under_conductor_error():
    """A failed node is a ``NodeError``; a refused call is a ``ValueError``
    as well, so a host's ``except ValueError`` catches it."""
    assert issubclass(NodeError, ConductorError)
    assert issubclass(InvalidCallError, ConductorError) and issubclass(InvalidCallError, ValueError)


def test_a_node_failure_takes_its_code_and_ctx_from_what_the_node_raised():
    failure = NodeError("went wrong", code="rate_limited", ctx={"retry_after": 30})

    assert (failure.code, failure.ctx, str(failure)) == ("rate_limited", {"retry_after": 30}, "went wrong")


# --- a refused call ------------------------------------------------------------------


# --- hooks -----------------------------------------------------------------------


def _hooked(raised: Exception) -> Any:
    def hooked_inputs():
        raise raised

    @node("hooked", title="H", description="d", category="c", compute_added_inputs=hooked_inputs)
    def hooked(text: Annotated[Txt, Param()], **more: Any) -> Annotated[Txt, Result(title="R")]:
        return text

    return hooked


def test_a_hook_may_not_raise_a_code_the_catalogue_says_can_still_run():
    with pytest.raises(ValueError, match="no_outputs"):
        ComputeHookError("m", code="no_outputs")


# --- problems of their own ---------------------------------------------------------


# --- at import ----------------------------------------------------------------------


def test_a_wrong_declaration_is_a_type_error_with_a_code_and_the_node():
    with pytest.raises(NodeDeclarationError) as wrong:
        @node("no-return", title="T", description="d", category="c")
        def no_return(text: Annotated[Txt, Param()]):  # no return type
            return text

    assert isinstance(wrong.value, TypeError)
    assert (wrong.value.code, wrong.value.node_id) == ("missing_return_type", "no-return")


def test_a_wrong_type_has_a_code_and_no_node():
    with pytest.raises(NodeDeclarationError) as wrong:
        class NoId(DType, str):
            display_title = "x"

    assert (wrong.value.code, wrong.value.node_id) == ("type_without_id", None)


def test_a_registry_refusal_names_the_node():
    @node("echo", title="Other", description="d", category="c")
    def other(text: Annotated[Txt, Param()]) -> Annotated[Txt, Result(title="R")]:
        return text

    with pytest.raises(NodeDeclarationError) as wrong:
        NodeRegistry([echo, other])

    assert (wrong.value.code, wrong.value.node_id) == ("duplicate_node_id", "echo")


def test_a_single_annotation_is_still_refused_on_a_named_parameter():
    with pytest.raises(NodeDeclarationError) as wrong:
        @node("bad", title="T", description="d", category="c")
        def bad(x: AsItArrives) -> Annotated[Txt, Result(title="R")]:
            return Txt("x")

    assert wrong.value.code == "as_it_arrives_on_named_parameter"
