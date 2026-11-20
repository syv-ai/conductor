"""The problems ``check_registry`` returns.

This is the last section of the problems page, ``python -m conductor.about problems``. A developer meets these when a
test fails. A host never gets one from a graph or a run, so they aren't in ``PROBLEMS``. ``conductor.testing`` exports
the catalogue of them, ``REGISTRY_CHECK_PROBLEMS``.

Each kind of problem is a function, and calling it makes the ``Problem``::

    needs_example("coded", 1, inputs=["code"])
    # Problem(type="needs_example", loc=("coded", 1),
    #         msg="check_registry can't make up a value for code, and only a constant can give one. Give this version of the node an example.",
    #         ctx={"inputs": ["code"]})

Every problem here is about one version of a node, so its ``loc`` is ``(node_id, version)``. A check that a node
fails because of a problem conductor reports, such as an example that doesn't compile, gives one problem for each
problem behind it. Its ``ctx.reason`` is that problem's message, its ``ctx.problem`` its code (its ``type``) and its ``ctx.loc`` its
``loc``, as it is, so a test reads all three without unpacking anything. These kinds name their own ``loc``
``place``, since ``loc`` is their key for the one behind them.

Each function's docstring is its entry on the problems page. Its example runs after the setup at the top of that
page, and a test checks that the example makes this kind of problem.
"""

from __future__ import annotations

from conductor._internal.problem import ProblemCatalogue
from conductor._internal.utils import comma_separated

#: The kinds of problem ``check_registry`` returns, which make up this section of the problems page.
REGISTRY_CHECK_PROBLEMS = ProblemCatalogue()


@REGISTRY_CHECK_PROBLEMS.problem
def example_for_no_node(*loc: str | int, named: str) -> str:
    """An example is for a node, or a version of a node, that isn't in the registry.

    This is most often a typo in the node id.

    To fix it, correct the key in ``examples``, or remove the example.

    Args:
        loc: The node id the example is for, and the version if it names one.
        named: The example's key in ``examples``, as written.

    Example:
        checked = NodeRegistry()
        checked.add(upper)
        check_registry(checked, examples={"uper": [{"text": "a"}]})
    """
    return f"The example {named} is for a node the registry doesn't have. Correct its key, or remove the example."


@REGISTRY_CHECK_PROBLEMS.problem
def needs_example(*loc: str | int, inputs: list[str]) -> str:
    r"""A required input takes only a constant, and ``check_registry`` can't make up a value for it.

    Without an example, ``check_registry`` still compiles a version of a node that has hooks, to call them, and an
    older version, to upgrade it. Each required input that takes only a constant needs a value for that, and the
    check makes one up. Some values can't be made up, such as a text that must match a pattern. Then neither check
    is made for that version.

    To fix it, give that version an example, such as ``examples={"coded": [{"code": "1234"}]}``, or
    ``examples={("coded", 1): [...]}`` for an older version.

    Args:
        loc: The node and its version.
        inputs: The inputs that no value could be made up for.

    Example:
        def by_code(code: str) -> dict[str, type]:
            return {f"field_{code}": str}


        @node("coded", title="Coded", description="Returns one output per code.", compute_outputs=by_code)
        def coded(code: Annotated[str, Param(pattern=r"^\d{4}$")]) -> dict[str, str]:
            return {}


        checked = NodeRegistry()
        checked.add(coded)
        check_registry(checked)
    """
    return (
        f"check_registry can't make up a value for {comma_separated(inputs)}, and only a constant can give one. "
        "Give this version of the node an example."
    )


@REGISTRY_CHECK_PROBLEMS.problem
def hook_failed(*place: str | int, reason: str, problem: str, loc: list[str | int]) -> str:
    """A node's ``compute_...`` hook failed when it was called with the node's defaults and made-up values.

    The node has no example, so ``check_registry`` calls the hook with values it makes up itself.

    To fix it, make the hook work on those values, or give the node an example that it works on.

    Args:
        place: The node and its version.
        reason: The message of the problem that compiling the node reported.
        problem: That problem's code, such as ``compute_hook_raised``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        def no_names(names: str) -> dict[str, type]:
            raise KeyError(names)


        @node("unnamed", title="Unnamed", description="Returns one output per name.", compute_outputs=no_names)
        def unnamed(names: str) -> dict[str, str]:
            return {}


        checked = NodeRegistry()
        checked.add(unnamed)
        check_registry(checked)
    """
    return f"A hook failed on the node's defaults and the values made up for its inputs: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem
def example_hook_failed(*place: str | int, reason: str, problem: str, loc: list[str | int]) -> str:
    """A node's example doesn't compile, because a ``compute_...`` hook fails on it.

    In the example below, ``pick``'s hook raises when it is given an empty text.

    To fix it, fix the hook or the example.

    Args:
        place: The node and its version.
        reason: The message of the problem that compiling reported.
        problem: That problem's code, such as ``compute_hook_raised``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        checked = NodeRegistry()
        checked.add(pick)
        check_registry(checked, examples={"pick": [{"names": ""}]})
    """
    return f"A hook fails on the example, so it doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem
def upgrade_broken(
    *place: str | int, newest: int, reason: str, problem: str, loc: list[str | int]
) -> str:
    """A graph saved with an older version of a node doesn't upgrade cleanly to the newest.

    ``check_registry`` puts the node in a graph at the older version, upgrades it with ``registry.upgrade``, and
    compiles the result. The node's inputs get the first example's values, or made-up ones, and each required input
    left without a value is a graph input. Either step can fail. ``registry.upgrade`` refuses the graph when an
    ``Upgrade``'s rewrite raises or returns a value it can't write, and compiling reports the problems of the graph it
    returns. In the example below, the rewrite gives ``first`` a number, but the newest version takes a text.

    To fix it, fix the ``Upgrade`` the problem behind it names, so a graph saved with the older version compiles
    cleanly after the upgrade.

    Args:
        place: The node and the older version.
        newest: The version it was upgraded to.
        reason: The message of the problem that ``registry.upgrade`` raised or compiling reported.
        problem: That problem's code, such as ``invalid_value``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        @node("welcome", title="Welcome", description="Welcomes a person.")
        def welcome_v1(name: str) -> str:
            return f"Welcome {name}"


        @node("welcome", title="Welcome", description="Welcomes a person.", replaces=welcome_v1, upgrade=Upgrade(rewrite=lambda values: {"first": 3}))
        def welcome_v2(first: str) -> str:
            return first


        checked = NodeRegistry()
        checked.add(welcome_v2)
        check_registry(checked)
    """
    return (
        f"A graph saved with an older version doesn't upgrade cleanly to version {newest}: {reason}"
    )


@REGISTRY_CHECK_PROBLEMS.problem
def subgraph_broken(*place: str | int, reason: str, problem: str, loc: list[str | int]) -> str:
    """A subgraph's own graph doesn't compile on the registry's nodes.

    A graph that uses the subgraph would have the same problems, so it can't run, and ``registry.describe()`` leaves the
    subgraph out, since it has no inputs or outputs to offer. In the example below, the subgraph uses ``upper``, which
    the registry doesn't have.

    To fix it, add the nodes it uses to the registry, or fix its graph record.

    Args:
        place: The subgraph and its version.
        reason: The message of the problem that compiling its graph reported.
        problem: That problem's code, such as ``unknown_node``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        capitals = subgraph("capitals", title="Capitals", description="Puts a text in capitals.")(make_graph(upper))
        checked = NodeRegistry()
        checked.add(capitals)
        check_registry(checked)
    """
    return f"The subgraph's graph doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem
def changed_its_inputs(*loc: str | int, input: str) -> str:
    """A node changed a value it was given, in place.

    A node never changes a value it is given. Every node that reads an output gets the object its node returned, and
    the run keeps that same object as its result, so a change made in one call is seen by every other node that
    reads it, and by the run's result. ``check_registry`` copies each value it gives a node's example and compares
    the two after the call. It is the one place the rule is checked, since a run doesn't copy what it passes on.

    To fix it, make a new value instead of changing the one given: ``[*items, item]`` in place of
    ``items.append(item)``, ``dataclasses.replace(order, total=2)``, ``order.model_copy(update={"total": 2})``, or
    ``frame.with_columns(...)``.

    Args:
        loc: The node and its version.
        input: The input whose value was changed.

    Example:
        @node("add_tag", title="Add tag", description="Adds a tag to a list of tags.")
        def add_tag(items: list[str]) -> list[str]:
            items.append("new")
            return items


        checked = NodeRegistry()
        checked.add(add_tag)
        check_registry(checked, examples={"add_tag": [{"items": ["old"]}]})
    """
    return (
        f"The node changed the value of its input {input!r} in place, which every other node that reads it sees. "
        "Make a new value instead, such as [*items, item], dataclasses.replace(...), model_copy(update=...) or "
        "frame.with_columns(...)."
    )


@REGISTRY_CHECK_PROBLEMS.problem
def example_run_failed(
    *place: str | int, status: str, reason: str, problem: str, loc: list[str | int]
) -> str:
    """A node's example ran, and the run ended early.

    The run failed, was stopped, was cancelled or timed out. ``ctx.problem`` is the code of the problem the run ended
    on. When the node raised an exception of its own, the message names the exception's class.

    To fix it, fix the node or the example.

    If a node is known to fail, give it a test of its own and mark that test ``pytest.mark.xfail(strict=True)``. In
    that test, check only that node, with ``registry.filter(...)``. Once the node is fixed, the test fails, which
    tells you to remove the mark.

    Args:
        place: The node and its version.
        status: How the run ended.
        reason: The message of the problem the run ended on. For an exception the node raised, its class name.
        problem: That problem's code, such as ``execution_failed``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        @node("divide_by", title="Divide", description="Divides one by a number.")
        def divide_by(n: int) -> float:
            return 1 / n


        checked = NodeRegistry()
        checked.add(divide_by)
        check_registry(checked, examples={"divide_by": [{"n": 0}]})
    """
    return f"The example's run ended {status}: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem
def example_not_compiled(*place: str | int, reason: str, problem: str, loc: list[str | int]) -> str:
    """A node's example doesn't compile, for a reason other than a ``compute_...`` hook that fails.

    An example gives a value for some of the node's inputs, by name. Each name must be an input of the node, and each
    value must be of that input's type. In the example below, ``txt`` is a typo for ``text``.

    To fix it, fix the example.

    Args:
        place: The node and its version.
        reason: The message of the problem that compiling reported.
        problem: That problem's code, such as ``unknown_input``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        checked = NodeRegistry()
        checked.add(upper)
        check_registry(checked, examples={"upper": [{"txt": "a"}]})
    """
    return f"The example doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem
def example_refused(*place: str | int, reason: str, problem: str, loc: list[str | int]) -> str:
    """A node's example compiles, but running it is refused.

    The run is missing something it needs. In the example below, the node takes ``Context[HostContext]``, and no
    context is given.

    To fix it, give the run what it needs, such as
    ``check_registry(checked, context=HostContext(caller=Caller("Ida")))``.

    Args:
        place: The node and its version.
        reason: The message of the problem the run was refused for.
        problem: That problem's code, such as ``context_missing``.
        loc: That problem's ``loc``, as it is, such as ``["b", "n"]``.

    Example:
        served = NodeRegistry(context_type=HostContext)
        served.add(who)
        check_registry(served, examples={"who": [{}]})
    """
    return f"The example's run was refused: {reason}"
