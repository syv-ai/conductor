"""``conductor.testing.check_registry``: each check passes on nodes that keep the contract and names each that doesn't."""

from typing import Annotated, Any

import pytest
from conductor import (
    Ask,
    ExternalError,
    FromRun,
    InvalidCallError,
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


@node("upper", title="Upper", description="")
def upper(text: str, times: Annotated[int, Param(ge=1)] = 1) -> str:
    return text.upper() * times


def _columns(names: str) -> dict[str, Any]:
    return {name: str for name in names.split(",")}


@node("columns", title="Columns", description="", compute_outputs=_columns)
def columns(text: str, names: str) -> dict[str, Any]:
    return dict.fromkeys(names.split(","), text)


@node("count", title="Count v1", description="")
def count_v1(n: int) -> int:
    return n


@node("count", title="Count", description="", replaces=count_v1, upgrade=Upgrade(renamed_inputs={"n": "number"}))
def count(number: int) -> int:
    return number


@node("approve", title="Approve", description="")
def approve(text: str) -> bool:
    raise Ask(f"Approve {text}?")


class Session:
    pass


@node("who", title="Who", description="")
def who(session: FromRun[Session], text: str) -> str:
    return text


shout = node("shout", title="Shout", description="")(make_graph(upper))

GOOD_EXAMPLES: dict[Any, list[dict[str, Any]]] = {
    "upper": [{"text": "a", "times": 2}],
    "approve": [{"text": "the plan"}],
    "who": [{"text": "a"}],
    ("count", 1): [{"n": 3}],
}


def test_nodes_that_keep_the_contract_give_no_problem():
    assert found(registry_of(upper, columns, count, approve, who), examples=GOOD_EXAMPLES, from_run={Session: Session()}) == []


def test_a_node_built_from_a_graph_runs_its_example_as_any_node_does():
    assert found(registry_of(upper, shout), examples={"shout": [{"text": "hi"}]}) == []


def test_a_from_run_value_is_taken_by_the_class_the_node_names_and_a_subclass_key_does_not_fill_it():
    class TestSession(Session):
        pass

    registry = registry_of(who)
    assert found(registry, examples={"who": [{"text": "a"}]}, from_run={Session: TestSession()}) == []
    graph = registry.load(make_graph(who, values={"who.text": "a"}))
    assert graph.run(from_run={Session: TestSession()}).outputs == {"who.result": "a"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(from_run={TestSession: TestSession()})
    assert [(p.type, p.ctx["given"]) for p in refused.value.errors()] == [("from_run_missing", ["TestSession"])]
    with pytest.raises(InvalidCallError) as refused:
        graph.run(from_run={int: 3})
    assert [(p.type, p.ctx["given"]) for p in refused.value.errors()] == [("from_run_missing", ["int"])]
    assert "it gives int" in refused.value.errors()[0].msg


def test_no_body_is_called_without_an_example_and_each_example_calls_it_once():
    calls: list[str] = []

    @node("noted", title="Noted", description="")
    def noted(text: str) -> str:
        calls.append(text)
        return text

    assert found(registry_of(noted)) == [] and calls == []
    assert found(registry_of(noted), examples={"noted": [{"text": "a"}, {"text": "b"}]}) == [] and calls == ["a", "b"]


# -- hooks -----------------------------------------------------------------------------------------


def _breaks(names: str) -> dict[str, Any]:
    raise KeyError(names)


@node("breaks", title="Breaks", description="", compute_outputs=_breaks)
def breaks(names: str) -> dict[str, Any]:
    return {}


def test_a_hook_that_fails_on_made_up_values_or_an_example_is_hook_failed():
    assert found(registry_of(breaks)) == [("hook_failed", ("breaks", 1))]
    assert found(registry_of(breaks), examples={"breaks": [{"names": "a"}]}) == [("hook_failed", ("breaks", 1))]


def test_a_hook_reading_an_input_no_value_can_be_made_up_for_needs_an_example():
    def _code(code: str) -> dict[str, Any]:
        return {"x": str}

    @node("coded", title="Coded", description="", compute_outputs=_code)
    def coded(code: Annotated[str, Param(pattern=r"^\d{4}$")]) -> dict[str, Any]:
        return {"x": code}

    assert found(registry_of(coded)) == [("needs_example", ("coded", 1))]
    assert found(registry_of(coded), examples={"coded": [{"code": "1234"}]}) == []


def test_a_hook_reading_a_number_between_two_limits_is_checked_on_a_value_between_them():
    def _cut(threshold: float) -> dict[str, Any]:
        return {"x": bool}

    @node("cut", title="Cut", description="Tells whether a value passes a threshold.", compute_outputs=_cut)
    def cut(threshold: Annotated[float, Param(gt=0, lt=1)]) -> dict[str, Any]:
        return {"x": True}

    assert found(registry_of(cut)) == []


# -- upgrades --------------------------------------------------------------------------------------


def test_an_upgrade_that_leaves_the_node_broken_or_raises_is_upgrade_broken():
    @node("size", title="Size v1", description="")
    def size_v1(n: int) -> int:
        return n

    @node("size", title="Size", description="", replaces=size_v1, upgrade=Upgrade(rewrite=lambda typed: {**typed, "n": "many"}))
    def size(n: int) -> int:
        return n

    @node("length", title="Length v1", description="")
    def length_v1(n: int) -> int:
        return n

    @node("length", title="Length", description="", replaces=length_v1, upgrade=Upgrade(rewrite=lambda typed: {"m": typed["missing"]}))
    def length(m: int) -> int:
        return m

    assert found(registry_of(size, length)) == [("upgrade_broken", ("size", 1)), ("upgrade_broken", ("length", 1))]
    [problem] = check_registry(registry_of(size))
    assert problem.ctx["cause"] == ["invalid_value"]


# -- calling the body ------------------------------------------------------------------------------


def test_a_body_that_changes_a_value_it_was_given_is_changed_its_inputs():
    @node("appends", title="Appends", description="")
    def appends(items: Series[int]) -> int:
        items.append(1)
        return len(items)

    [problem] = check_registry(registry_of(appends), examples={"appends": [{"items": [1, 2]}]})
    assert (problem.type, problem.loc, problem.ctx["input"]) == ("changed_its_inputs", ("appends", 1), "items")


def test_an_example_that_does_not_run_to_the_end_or_a_pause_is_run_failed():
    @node("unsure", title="Unsure", description="")
    def unsure(text: str) -> bool:
        raise Ask("Sure?", default="maybe")  # type: ignore[arg-type]

    @node("offline", title="Offline", description="")
    def offline(text: str) -> str:
        raise ExternalError("The service is down.")

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


@node("confirmed", title="Confirmed", description="")
def confirmed(text: str) -> str:
    return _confirm(text)


def test_a_body_that_raises_ask_itself_or_through_a_helper_needs_an_example_that_pauses():
    assert found(registry_of(approve, confirmed, upper)) == [("pause_untested", ("approve", 1)), ("pause_untested", ("confirmed", 1))]
    assert found(registry_of(approve, confirmed), examples={"approve": [{"text": "a"}], "confirmed": [{"text": "b"}]}) == []


def test_an_older_versions_pause_does_not_test_the_newest():
    @node("gate", title="Gate v1", description="")
    def gate_v1(text: str) -> bool:
        raise Ask("Old?")

    @node("gate", title="Gate", description="", replaces=gate_v1)
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


def test_a_value_unequal_to_its_own_copy_is_never_called_changed():
    @node("reads", title="Reads", description="")
    def reads(values: Series[float], thing: Any) -> int:
        return len(values)

    examples = {"reads": [{"values": [1.0, float("nan")], "thing": float("nan")}]}
    assert found(registry_of(reads), examples=examples) == []


def test_an_example_with_no_json_form_is_run_failed_by_its_node_and_input_and_raises_nothing():
    class Opaque:
        pass

    @node("takes", title="Takes", description="")
    def takes(thing: Any) -> str:
        return repr(thing)

    (problem,) = check_registry(registry_of(takes), examples={"takes": [{"thing": Opaque()}]})
    assert (problem.type, problem.loc, problem.ctx["cause"]) == ("run_failed", ("takes", 1), ["invalid_value"])
    assert problem.ctx["problems"][0]["loc"] == ["takes", "thing"]


def test_a_body_returning_what_its_type_refuses_is_run_failed():
    @node("counts", title="Counts", description="")
    def counts(text: str) -> int:
        return "many"  # type: ignore[return-value]

    @node("tally", title="Tally", description="")
    def tally(labels: Series[str]) -> dict[str, int]:
        return {label: "one" for label in labels}  # type: ignore[misc]

    problems = check_registry(registry_of(counts, tally), examples={"counts": [{"text": "a"}], "tally": [{"labels": ["a"]}]})
    assert [(p.type, p.loc, p.ctx["cause"]) for p in problems] == [
        ("run_failed", ("counts", 1), ["invalid_output"]), ("run_failed", ("tally", 1), ["invalid_output"]),
    ]


def test_a_protocol_from_run_given_under_another_key_is_from_run_missing():
    from typing import Protocol

    class Authority(Protocol):
        def may_read(self, document: str) -> bool: ...

    @node("read", title="Read", description="")
    def read(x: int, authority: FromRun[Authority]) -> int:
        return x

    graph = registry_of(read).load(make_graph(read, values={"read.x": 1}))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(from_run={str: "a session"})
    assert [p.type for p in refused.value.errors()] == ["from_run_missing"]


def test_a_stale_waiver_names_a_node_whose_id_holds_a_colon():
    from conductor import NodeRegistry, node

    @node("ns:upper", title="Upper", description="")
    def upper(text: str) -> str:
        return text.upper()

    registry = NodeRegistry()
    registry.add(upper)
    stale = check_registry(registry, expected_failures={"ns:upper:run_failed": "fixed since"})
    assert [(p.type, p.loc) for p in stale] == [("waiver_not_needed", ("ns:upper",))]
