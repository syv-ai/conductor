"""Tests for ``conductor.testing.check_registry``, which checks every node of a registry in one go.

A host calls it from one test of its own. Each of its checks should pass for nodes that keep the rules, and name each
node that breaks one.
"""

import logging
from typing import Annotated, Any

import polars as pl
from conductor import (
    Ask,
    Context,
    ExternalError,
    NodeRegistry,
    Param,
    Problem,
    Series,
    Upgrade,
    make_graph,
    node,
    subgraph,
)
from conductor.testing import REGISTRY_CHECK_PROBLEMS, check_registry


class Session:
    pass


def registry_of(*definitions: Any) -> NodeRegistry[Session]:
    held = NodeRegistry(context_type=Session)
    held.add(*definitions)
    return held


def found(registry: NodeRegistry, **options: Any) -> list[tuple[str, tuple[str | int, ...]]]:
    return [(p.type, p.loc) for p in check_registry(registry, **options)]


def behind(problem: Problem) -> list[str]:
    """Returns the codes of the problems that one ``check_registry`` problem was made from."""
    return [one["type"] for one in problem.ctx["problems"]]


# -- nodes that keep the contract ------------------------------------------------------------------


@node("upper", title="Upper", description="For a test.")
def upper(text: str, times: Annotated[int, Param(ge=1)] = 1) -> str:
    return text.upper() * times


def _columns(names: str) -> dict[str, Any]:
    return {name: str for name in names.split(",")}


@node("columns", title="Columns", description="For a test.", compute_outputs=_columns)
def columns(text: str, names: str) -> dict[str, Any]:
    return dict.fromkeys(names.split(","), text)


@node("count", title="Count v1", description="For a test.")
def count_v1(n: int) -> int:
    return n


@node(
    "count",
    title="Count",
    description="For a test.",
    replaces=count_v1,
    upgrade=Upgrade(renamed_inputs={"n": "number"}),
)
def count(number: int) -> int:
    return number


@node("approve", title="Approve", description="For a test.", asks=True)
def approve(text: str) -> bool:
    raise Ask(f"Approve {text}?")


@node("who", title="Who", description="For a test.")
def who(ctx: Context[Session], text: str) -> str:
    return text


shout = subgraph("shout", title="Shout", description="For a test.")(make_graph(upper))

GOOD_EXAMPLES: dict[Any, list[dict[str, Any]]] = {
    "upper": [{"text": "a", "times": 2}],
    "approve": [{"text": "the plan"}],
    "who": [{"text": "a"}],
    ("count", 1): [{"n": 3}],
}


def test_nodes_that_keep_the_contract_give_no_problem():
    assert (
        found(
            registry_of(upper, columns, count, approve, who),
            examples=GOOD_EXAMPLES,
            context=Session(),
        )
        == []
    )


def test_a_node_built_from_a_graph_runs_its_example_as_any_node_does():
    assert found(registry_of(upper, shout), examples={"shout": [{"text": "hi"}]}) == []


def test_a_subgraph_whose_graph_does_not_compile_is_reported_with_its_graphs_problems():
    # the subgraph uses upper, which the registry doesn't have
    (problem,) = check_registry(registry_of(shout))

    assert (problem.type, problem.loc, behind(problem)) == (
        "subgraph_broken",
        ("shout", 1),
        ["unknown_node"],
    )
    assert found(registry_of(upper, shout)) == []


def test_a_broken_subgraph_with_an_example_is_reported_as_broken_and_its_example_is_not_run():
    # one fault gives one code, whether or not the subgraph has an example
    (problem,) = check_registry(registry_of(shout), examples={"shout": [{"text": "hi"}]})

    assert (problem.type, problem.loc, behind(problem)) == (
        "subgraph_broken",
        ("shout", 1),
        ["unknown_node"],
    )


def test_no_body_is_called_without_an_example_and_each_example_calls_it_once():
    calls: list[str] = []

    @node("noted", title="Noted", description="For a test.")
    def noted(text: str) -> str:
        calls.append(text)
        return text

    assert found(registry_of(noted)) == [] and calls == []
    assert found(
        registry_of(noted), examples={"noted": [{"text": "a"}, {"text": "b"}]}
    ) == [] and calls == ["a", "b"]


# -- hooks -----------------------------------------------------------------------------------------


def _breaks(names: str) -> dict[str, Any]:
    raise KeyError(names)


@node("breaks", title="Breaks", description="For a test.", compute_outputs=_breaks)
def breaks(names: str) -> dict[str, Any]:
    return {}


def test_each_code_check_registry_returns_is_in_the_catalogue_conductor_testing_exports():
    """``REGISTRY_CHECK_PROBLEMS`` is public, so a test can pin a developer's own handling to its codes."""
    returned = {
        problem.type for problem in check_registry(registry_of(breaks), examples={"nowhere": [{}]})
    }
    assert returned == {"example_for_no_node", "hook_failed"}
    assert returned <= REGISTRY_CHECK_PROBLEMS.codes


def test_a_hook_that_fails_on_made_up_values_or_an_example_is_hook_failed():
    assert found(registry_of(breaks)) == [("hook_failed", ("breaks", 1))]
    assert found(registry_of(breaks), examples={"breaks": [{"names": "a"}]}) == [
        ("hook_failed", ("breaks", 1))
    ]


def test_a_hook_that_gives_a_field_a_name_it_cannot_have_is_hook_failed():
    assert found(registry_of(columns), examples={"columns": [{"text": "a", "names": "a.b"}]}) == [
        ("hook_failed", ("columns", 1))
    ]


def test_a_hook_reading_an_input_no_value_can_be_made_up_for_needs_an_example():
    def _code(code: str) -> dict[str, Any]:
        return {"x": str}

    @node("coded", title="Coded", description="For a test.", compute_outputs=_code)
    def coded(code: Annotated[str, Param(pattern=r"^\d{4}$")]) -> dict[str, Any]:
        return {"x": code}

    assert found(registry_of(coded)) == [("needs_example", ("coded", 1))]
    assert found(registry_of(coded), examples={"coded": [{"code": "1234"}]}) == []


def test_a_hook_reading_a_number_between_two_limits_is_checked_on_a_value_between_them():
    def _cut(threshold: float) -> dict[str, Any]:
        return {"x": bool}

    @node(
        "cut",
        title="Cut",
        description="Tells whether a value passes a threshold.",
        compute_outputs=_cut,
    )
    def cut(threshold: Annotated[float, Param(gt=0, lt=1)]) -> dict[str, Any]:
        return {"x": True}

    assert found(registry_of(cut)) == []


# -- upgrades --------------------------------------------------------------------------------------


def test_an_upgrade_that_leaves_the_node_broken_or_raises_is_upgrade_broken():
    @node("size", title="Size v1", description="For a test.")
    def size_v1(n: int) -> int:
        return n

    @node(
        "size",
        title="Size",
        description="For a test.",
        replaces=size_v1,
        upgrade=Upgrade(rewrite=lambda typed: {**typed, "n": "many"}),
    )
    def size(n: int) -> int:
        return n

    @node("length", title="Length v1", description="For a test.")
    def length_v1(n: int) -> int:
        return n

    @node(
        "length",
        title="Length",
        description="For a test.",
        replaces=length_v1,
        upgrade=Upgrade(rewrite=lambda typed: {"m": typed["missing"]}),
    )
    def length(m: int) -> int:
        return m

    assert found(registry_of(size, length)) == [
        ("upgrade_broken", ("size", 1)),
        ("upgrade_broken", ("length", 1)),
    ]
    [problem] = check_registry(registry_of(size))
    assert behind(problem) == ["invalid_value"]


def test_an_older_version_whose_input_no_value_can_be_made_up_for_upgrades_without_an_example():
    """An input of type ``T`` takes no made-up value, so the upgraded graph gets it as a graph input."""

    @node("relay", title="Relay v1", description="For a test.")
    def relay_v1[T](value: T) -> T:
        return value

    @node("relay", title="Relay", description="For a test.", replaces=relay_v1)
    def relay[T](value: T, loud: bool = False) -> T:
        return value

    assert found(registry_of(relay)) == []


def test_an_older_subgraph_whose_input_gets_no_value_upgrades_without_an_example():
    """A subgraph's inputs take no made-up value, so the upgraded graph gets each as a graph input."""

    @node("pass-on", title="Pass on", description="For a test.")
    def pass_on[T](value: T) -> T:
        return value

    wrapped_v1 = subgraph("wrapped", title="Wrapped v1", description="For a test.")(
        make_graph(pass_on)
    )
    wrapped = subgraph("wrapped", title="Wrapped", description="For a test.", replaces=wrapped_v1)(
        make_graph(pass_on)
    )

    assert found(registry_of(pass_on, wrapped)) == []


def test_an_older_version_whose_constant_only_input_no_value_can_be_made_up_for_needs_an_example():
    """Its upgrade can't be checked without that value, so the version needs an example.

    A version whose hooks read the input gets the same one problem, not a second for its hooks.
    """
    pinned = Annotated[str, Param(constant_only=True, pattern=r"^\d{4}$")]

    @node("pin", title="Pin v1", description="For a test.")
    def pin_v1(code: pinned) -> str:
        return code

    @node("pin", title="Pin", description="For a test.", replaces=pin_v1)
    def pin(code: pinned, loud: bool = False) -> str:
        return code

    def _pin_fields(code: str) -> dict[str, Any]:
        return {"x": str}

    @node(
        "hooked-pin", title="Hooked pin v1", description="For a test.", compute_outputs=_pin_fields
    )
    def hooked_pin_v1(code: pinned) -> dict[str, Any]:
        return {"x": code}

    @node("hooked-pin", title="Hooked pin", description="For a test.", replaces=hooked_pin_v1)
    def hooked_pin(code: pinned) -> str:
        return code

    checked = registry_of(pin, hooked_pin)
    assert found(checked) == [("needs_example", ("pin", 1)), ("needs_example", ("hooked-pin", 1))]
    examples = {("pin", 1): [{"code": "1234"}], ("hooked-pin", 1): [{"code": "1234"}]}
    assert found(checked, examples=examples) == []


def test_an_upgrade_that_raises_is_upgrade_broken_with_the_refusal_of_registry_upgrade_behind_it(
    caplog,
):
    """``registry.upgrade`` refuses a rewrite that raises, and ``check_registry`` reports that refusal.

    The problem behind it has the exception's class, and its text goes to the log.
    """

    @node("leak", title="Leak v1", description="For a test.")
    def leak_v1(n: int) -> int:
        return n

    def leaking(values: dict[str, Any]) -> dict[str, Any]:
        raise KeyError("the secret is 42")

    @node(
        "leak",
        title="Leak",
        description="For a test.",
        replaces=leak_v1,
        upgrade=Upgrade(rewrite=leaking),
    )
    def leak(n: int) -> int:
        return n

    with caplog.at_level(logging.ERROR, logger="conductor"):
        [problem] = check_registry(registry_of(leak))
    assert (problem.type, problem.loc, behind(problem)) == (
        "upgrade_broken",
        ("leak", 1),
        ["cannot_move_to_version"],
    )
    [refusal] = problem.ctx["problems"]
    assert (refusal["loc"], refusal["ctx"]["variant"], refusal["ctx"]["error"]) == (
        ["leak"],
        "raised",
        "KeyError",
    )
    assert "secret" not in problem.model_dump_json() and "the secret is 42" in caplog.text


def test_an_upgrade_that_returns_none_is_upgrade_broken_and_the_other_nodes_are_still_checked():
    """A rewrite that forgets its ``return`` is reported for its node, and the check goes on to the next node."""

    @node("forgets", title="Forgets v1", description="For a test.")
    def forgets_v1(n: int) -> int:
        return n

    @node(
        "forgets",
        title="Forgets",
        description="For a test.",
        replaces=forgets_v1,
        upgrade=Upgrade(rewrite=lambda values: None),
    )
    def forgets(n: int) -> int:
        return n

    @node("fails", title="Fails v1", description="For a test.")
    def fails_v1(n: int) -> int:
        return n

    def failing(values: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("no")

    @node(
        "fails",
        title="Fails",
        description="For a test.",
        replaces=fails_v1,
        upgrade=Upgrade(rewrite=failing),
    )
    def fails(n: int) -> int:
        return n

    checked = check_registry(registry_of(forgets, fails))
    assert [(p.type, p.loc, behind(p)) for p in checked] == [
        ("upgrade_broken", ("forgets", 1), ["cannot_move_to_version"]),
        ("upgrade_broken", ("fails", 1), ["cannot_move_to_version"]),
    ]
    assert [p.ctx["problems"][0]["ctx"]["variant"] for p in checked] == ["rewrite_failed", "raised"]


def test_the_reason_of_every_kind_is_the_messages_of_the_problems_behind_it():
    """A ``check_registry`` problem's ``reason`` is the messages of the problems behind it.

    Those problems are in ``ctx.problems``. The ``reason`` is just their messages, joined by "; ". This holds for
    every kind alike.
    """

    @node("size", title="Size v1", description="For a test.")
    def size_v1(n: int) -> int:
        return n

    @node(
        "size",
        title="Size",
        description="For a test.",
        replaces=size_v1,
        upgrade=Upgrade(rewrite=lambda typed: {**typed, "n": "many"}),
    )
    def size(n: int) -> int:
        return n

    [problem] = check_registry(registry_of(size))
    [behind] = problem.ctx["problems"]
    assert problem.ctx["reason"] == behind["msg"]


# -- calling the body ------------------------------------------------------------------------------


def test_a_body_that_changes_a_value_it_was_given_is_changed_its_inputs():
    @node("appends", title="Appends", description="For a test.")
    def appends(items: Series[int]) -> int:
        items.append(1)
        return len(items)

    [problem] = check_registry(registry_of(appends), examples={"appends": [{"items": [1, 2]}]})
    assert (problem.type, problem.loc, problem.ctx["input"]) == (
        "changed_its_inputs",
        ("appends", 1),
        "items",
    )


def test_a_body_that_puts_a_frame_where_its_input_had_a_list_is_changed_its_inputs():
    """A frame is never the same value as a list of records with its rows, and comparing the two doesn't raise.

    ``==`` between a list and a frame raises a polars error, which was reported as the node's own failure.
    """

    @node("tabulates", title="Tabulates", description="For a test.")
    def tabulates(data: dict[str, Any]) -> int:
        data["rows"] = pl.DataFrame(data["rows"])
        return data["rows"].height

    examples = {"tabulates": [{"data": {"rows": [{"amount": 1}, {"amount": 2}]}}]}
    [problem] = check_registry(registry_of(tabulates), examples=examples)
    assert (problem.type, problem.loc, problem.ctx["input"]) == (
        "changed_its_inputs",
        ("tabulates", 1),
        "data",
    )


def test_an_example_that_does_not_run_to_the_end_or_a_pause_is_run_failed():
    @node("unsure", title="Unsure", description="For a test.", asks=True)
    def unsure(text: str) -> bool:
        raise Ask("Sure?", default="maybe")  # type: ignore[arg-type]

    @node("offline", title="Offline", description="For a test.")
    def offline(text: str) -> str:
        raise ExternalError("The service is down.")

    examples = {
        "unsure": [{"text": "a"}],
        "offline": [{"text": "a"}],
        "who": [{"text": "no session"}],
        "upper": [{"txt": "a"}],
    }
    problems = check_registry(registry_of(unsure, offline, who, upper), examples=examples)
    assert [(p.type, p.loc, behind(p)) for p in problems] == [
        ("run_failed", ("unsure", 1), ["invalid_ask"]),
        ("run_failed", ("offline", 1), ["external_failed"]),
        ("run_failed", ("who", 1), ["context_missing"]),
        ("run_failed", ("upper", 1), ["unknown_input", "input_has_no_value"]),
    ]


def test_a_run_failed_that_a_body_raised_names_the_error_class():
    @node("divides", title="Divides", description="For a test.")
    def divides(n: int) -> float:
        return 1 / n

    [problem] = check_registry(registry_of(divides), examples={"divides": [{"n": 0}]})
    assert problem.type == "run_failed"
    assert problem.msg.endswith("raised ZeroDivisionError")


# -- what check_registry is given --------------------------------------------------------------------------


def test_an_example_for_a_node_or_version_not_there_is_example_for_no_node():
    examples = {"nope": [{}], ("upper", 7): [{}]}
    assert found(registry_of(upper), examples=examples) == [
        ("example_for_no_node", ("nope",)),
        ("example_for_no_node", ("upper", 7)),
    ]


def test_a_value_unequal_to_its_own_copy_is_never_called_changed():
    @node("reads", title="Reads", description="For a test.")
    def reads(values: Series[float], thing: Any) -> int:
        return len(values)

    examples = {"reads": [{"values": [1.0, float("nan")], "thing": float("nan")}]}
    assert found(registry_of(reads), examples=examples) == []


def test_an_example_with_no_json_form_is_run_failed_by_its_node_and_input_and_raises_nothing():
    class Opaque:
        pass

    @node("takes", title="Takes", description="For a test.")
    def takes(thing: Any) -> str:
        return repr(thing)

    (problem,) = check_registry(registry_of(takes), examples={"takes": [{"thing": Opaque()}]})
    assert (problem.type, problem.loc, behind(problem)) == (
        "run_failed",
        ("takes", 1),
        ["invalid_value"],
    )
    assert problem.ctx["problems"][0]["loc"] == ["takes", "thing"]


def test_a_body_returning_what_its_type_refuses_is_run_failed():
    @node("counts", title="Counts", description="For a test.")
    def counts(text: str) -> int:
        return "many"  # type: ignore[return-value]

    @node("tally", title="Tally", description="For a test.")
    def tally(labels: Series[str]) -> dict[str, int]:
        return {label: "one" for label in labels}  # type: ignore[misc]

    problems = check_registry(
        registry_of(counts, tally),
        examples={"counts": [{"text": "a"}], "tally": [{"labels": ["a"]}]},
    )
    assert [(p.type, p.loc, behind(p)) for p in problems] == [
        ("run_failed", ("counts", 1), ["invalid_output"]),
        ("run_failed", ("tally", 1), ["invalid_output"]),
    ]


def test_what_a_check_says_is_made_from_its_ctx_alone():
    @node("halve", title="Halve", description="Halves a number.")
    def halve(n: int) -> float:
        return 1 / (n - n)

    registry = NodeRegistry()
    registry.add(halve)

    (found,) = check_registry(registry, examples={"halve": [{"n": 2}], "halv": [{"n": 2}]})[1:]
    (missing,) = check_registry(registry, examples={"halv": [{"n": 2}]})

    assert (found.type, found.loc, found.ctx["status"], found.ctx["reason"], behind(found)) == (
        "run_failed",
        ("halve", 1),
        "failed",
        "the node raised ZeroDivisionError",
        ["execution_failed"],
    )
    assert (missing.type, missing.loc, missing.ctx) == (
        "example_for_no_node",
        ("halv",),
        {"named": "'halv'"},
    )
    assert found.msg == "The example's run ended {status}: {reason}".format(**found.ctx)
    assert (
        missing.msg
        == "The example {named} is for a node the registry doesn't have. Correct its key, or remove the example.".format(
            **missing.ctx
        )
    )
