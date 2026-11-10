"""``conductor.testing.check_registry``: each check passes on nodes that keep the contract and names each that doesn't."""

from typing import Annotated, Any

from conductor import (
    Ask,
    ExternalFailure,
    FromRun,
    NodeRegistry,
    Param,
    Series,
    Upgrade,
    make_graph,
    node,
)
from conductor.testing import check_registry


def registry_of(*definitions: Any) -> NodeRegistry:
    held = NodeRegistry()
    held.add(*definitions)
    return held


def found(registry: NodeRegistry, **options: Any) -> list[tuple[str, tuple[str | int, ...]]]:
    return [(p.type, p.loc) for p in check_registry(registry, **options)]


# -- nodes that keep the contract ------------------------------------------------------------------


@node("upper")
def upper(text: str, times: Annotated[int, Param(ge=1)] = 1) -> str:
    return text.upper() * times


def _columns(names: str) -> dict[str, Any]:
    return {name: str for name in names.split(",")}


@node("columns", compute_outputs=_columns)
def columns(text: str, names: str) -> dict[str, Any]:
    return dict.fromkeys(names.split(","), text)


@node("count")
def count_v1(n: int) -> int:
    return n


@node("count", replaces=count_v1, upgrade=Upgrade(renamed_inputs={"n": "number"}))
def count(number: int) -> int:
    return number


@node("approve")
def approve(text: str) -> bool:
    raise Ask(f"Approve {text}?")


class Session:
    pass


@node("who")
def who(session: FromRun[Session], text: str) -> str:
    return text


shout = node("shout")(make_graph(upper))

GOOD_EXAMPLES: dict[Any, list[dict[str, Any]]] = {
    "upper": [{"text": "a", "times": 2}],
    "approve": [{"text": "the plan"}],
    "who": [{"session": Session(), "text": "a"}],
    "shout": [{"text": "hi"}],
    ("count", 1): [{"n": 3}],
}


def test_nodes_that_keep_the_contract_give_no_problem():
    assert found(registry_of(upper, columns, count, approve, who, shout), examples=GOOD_EXAMPLES) == []


def test_no_body_is_called_without_an_example_and_each_example_calls_it_once():
    calls: list[str] = []

    @node("noted")
    def noted(text: str) -> str:
        calls.append(text)
        return text

    assert found(registry_of(noted)) == [] and calls == []
    assert found(registry_of(noted), examples={"noted": [{"text": "a"}, {"text": "b"}]}) == [] and calls == ["a", "b"]


# -- hooks -----------------------------------------------------------------------------------------


def _breaks(names: str) -> dict[str, Any]:
    raise KeyError(names)


@node("breaks", compute_outputs=_breaks)
def breaks(names: str) -> dict[str, Any]:
    return {}


def test_a_hook_that_fails_on_made_up_values_or_an_example_is_hook_failed():
    assert found(registry_of(breaks)) == [("hook_failed", ("breaks", 1))]
    assert found(registry_of(breaks), examples={"breaks": [{"names": "a"}]}) == [("hook_failed", ("breaks", 1))]


def test_a_hook_reading_an_input_no_value_can_be_made_up_for_needs_an_example():
    def _code(code: str) -> dict[str, Any]:
        return {"x": str}

    @node("coded", compute_outputs=_code)
    def coded(code: Annotated[str, Param(pattern=r"^\d{4}$")]) -> dict[str, Any]:
        return {"x": code}

    assert found(registry_of(coded)) == [("needs_example", ("coded", 1))]
    assert found(registry_of(coded), examples={"coded": [{"code": "1234"}]}) == []


# -- upgrades --------------------------------------------------------------------------------------


def test_an_upgrade_that_leaves_the_node_broken_or_raises_is_upgrade_broken():
    @node("size")
    def size_v1(n: int) -> int:
        return n

    @node("size", replaces=size_v1, upgrade=Upgrade(rewrite=lambda typed: {**typed, "n": "many"}))
    def size(n: int) -> int:
        return n

    @node("length")
    def length_v1(n: int) -> int:
        return n

    @node("length", replaces=length_v1, upgrade=Upgrade(rewrite=lambda typed: {"m": typed["missing"]}))
    def length(m: int) -> int:
        return m

    assert found(registry_of(size, length)) == [("upgrade_broken", ("size", 1)), ("upgrade_broken", ("length", 1))]
    [problem] = check_registry(registry_of(size))
    assert problem.ctx["cause"] == ["invalid_value"]


# -- calling the body ------------------------------------------------------------------------------


def test_a_body_that_changes_a_value_it_was_given_is_changed_its_inputs():
    @node("appends")
    def appends(items: Series[int]) -> int:
        items.append(1)
        return len(items)

    [problem] = check_registry(registry_of(appends), examples={"appends": [{"items": [1, 2]}]})
    assert (problem.type, problem.loc, problem.ctx["input"]) == ("changed_its_inputs", ("appends", 1), "items")


def test_an_example_that_does_not_run_to_the_end_or_a_pause_is_run_failed():
    @node("unsure")
    def unsure(text: str) -> bool:
        raise Ask("Sure?", default="maybe")  # type: ignore[arg-type]

    @node("offline")
    def offline(text: str) -> str:
        raise ExternalFailure("The service is down.")

    examples = {"unsure": [{"text": "a"}], "offline": [{"text": "a"}], "who": [{"text": "no session"}], "upper": [{"txt": "a"}]}
    problems = check_registry(registry_of(unsure, offline, who, upper), examples=examples)
    assert [(p.type, p.loc, p.ctx["cause"]) for p in problems] == [
        ("run_failed", ("unsure", 1), ["invalid_ask"]),
        ("run_failed", ("offline", 1), ["external_failed"]),
        ("run_failed", ("who", 1), ["from_run_missing"]),
        ("run_failed", ("upper", 1), ["unknown_input"]),
    ]


# -- pauses ----------------------------------------------------------------------------------------


def _confirm(text: str) -> str:
    raise Ask(f"Is {text} right?")


@node("confirmed")
def confirmed(text: str) -> str:
    return _confirm(text)


def test_a_body_that_raises_ask_itself_or_through_a_helper_needs_an_example_that_pauses():
    assert found(registry_of(approve, confirmed, upper)) == [("pause_untested", ("approve", 1)), ("pause_untested", ("confirmed", 1))]
    assert found(registry_of(approve, confirmed), examples={"approve": [{"text": "a"}], "confirmed": [{"text": "b"}]}) == []


def test_an_older_versions_pause_does_not_test_the_newest():
    @node("gate")
    def gate_v1(text: str) -> bool:
        raise Ask("Old?")

    @node("gate", replaces=gate_v1)
    def gate(text: str) -> bool:
        raise Ask("New?")

    assert found(registry_of(gate), examples={("gate", 1): [{"text": "a"}]}) == [("pause_untested", ("gate", 2))]


# -- the kit's own inputs --------------------------------------------------------------------------


def test_an_example_for_a_node_or_version_not_there_is_example_for_no_node():
    examples = {"nope": [{}], ("upper", 7): [{}]}
    assert found(registry_of(upper), examples=examples) == [("example_for_no_node", ("nope",)), ("example_for_no_node", ("upper", 7))]


def test_a_waiver_hides_its_failure_and_one_that_no_longer_fails_is_a_problem():
    assert found(registry_of(breaks), expected_failures={"breaks:hook_failed": "the service is mocked in CI"}) == []
    stale = check_registry(registry_of(upper), expected_failures={"upper:run_failed": "fixed since"})
    assert [(p.type, p.loc, p.ctx["waiver"]) for p in stale] == [("waiver_not_needed", ("upper",), "upper:run_failed")]
