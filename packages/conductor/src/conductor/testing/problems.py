"""The problems ``check_registry`` returns.

This is the last section of the problems page, ``python -m conductor.about problems``. A developer meets these when a
test fails. A host never gets one from a graph or a run, so they aren't in ``PROBLEMS``. ``conductor.testing`` exports
the catalogue of them, ``REGISTRY_CHECK_PROBLEMS``.

Each kind of problem is a function, and calling it makes the ``Problem``::

    needs_example("coded", 1, inputs=["code"])
    # Problem(type="needs_example", loc=("coded", 1),
    #         msg="The node's hooks read code, and check_registry can't make up a value for it. Give the node an example.",
    #         ctx={"inputs": ["code"]})

Every problem here is about one version of a node, so its ``loc`` is ``(node_id, version)``.

Some kinds share a code, such as the two kinds of ``hook_failed``. The second one names a variant, which the
problem's ``ctx.variant`` gives, and its message says which case it is.

Each function's docstring is its entry on the problems page. Its example runs after the setup at the top of that
page, and a test checks that the example makes this kind of problem.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.problem import ProblemCatalogue
from conductor._internal.utils import comma_separated

#: The kinds of problem ``check_registry`` returns, which make up this section of the problems page.
REGISTRY_CHECK_PROBLEMS = ProblemCatalogue()


@REGISTRY_CHECK_PROBLEMS.problem()
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


@REGISTRY_CHECK_PROBLEMS.problem()
def needs_example(*loc: str | int, inputs: list[str]) -> str:
    r"""A node's ``compute_...`` hook reads a required input, and ``check_registry`` can't make up a value for it.

    Without an example, a hook is called with the node's defaults, and with a made-up value for each required input
    it reads. Some values can't be made up, such as a text that must match a pattern.

    To fix it, give the node an example, such as ``examples={"coded": [{"code": "1234"}]}``.

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
        f"The node's hooks read {comma_separated(inputs)}, and check_registry can't make up a value for it. "
        "Give the node an example."
    )


@REGISTRY_CHECK_PROBLEMS.problem()
def hook_failed(*loc: str | int, reason: str, problems: list[dict[str, Any]]) -> str:
    """A node's ``compute_...`` hook failed when it was called with the node's defaults and made-up values.

    The node has no example, so ``check_registry`` calls the hook with values it makes up itself.

    To fix it, make the hook work on those values, or give the node an example that it works on.

    Args:
        loc: The node and its version.
        reason: The message of the problem that compiling the node reported.
        problems: That problem, as JSON.

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


@REGISTRY_CHECK_PROBLEMS.problem("hook_failed", variant="on_an_example")
def hook_failed_on_an_example(*loc: str | int, reason: str, problems: list[dict[str, Any]]) -> str:
    """A node's example doesn't compile, and every problem is in a ``compute_...`` hook.

    In the example below, ``pick``'s hook raises when it is given an empty text.

    To fix it, fix the hook or the example.

    Args:
        loc: The node and its version.
        reason: The message of each problem that compiling reported.
        problems: Those problems, as JSON.

    Example:
        checked = NodeRegistry()
        checked.add(pick)
        check_registry(checked, examples={"pick": [{"names": ""}]})
    """
    return f"The example doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem()
def upgrade_broken(
    *loc: str | int, newest: int, reason: str, problems: list[dict[str, Any]]
) -> str:
    """A graph saved with an older version of a node doesn't compile after it is upgraded to the newest.

    ``check_registry`` puts the node in a graph at the older version, upgrades it with ``registry.upgrade``, and
    compiles the result. In the example below, the rewrite gives ``first`` a number, but the newest version takes a
    text.

    To fix it, fix the newest version's ``Upgrade``, so a graph saved with the older version compiles cleanly after
    the upgrade.

    Args:
        loc: The node and the older version.
        newest: The version it was upgraded to.
        reason: The message of each problem that compiling reported.
        problems: Those problems, as JSON.

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
    return f"A graph upgraded to version {newest} doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("upgrade_broken", variant="raised")
def upgrade_raised(*loc: str | int, newest: int, error: str) -> str:
    """Upgrading a node from an older version to the newest raised an exception.

    The exception's text might quote a value, so the text and the traceback go to the ``conductor`` logger, and
    ``ctx.error`` only has the exception's class name.

    To fix it, fix the newest version's ``Upgrade``, so a graph saved with the older version is upgraded without an
    error.

    Args:
        loc: The node and the older version.
        newest: The version it was being upgraded to.
        error: The exception's class name.

    Example:
        @node("hi", title="Hi", description="Says hi to a person.")
        def hi_v1(name: str) -> str:
            return f"Hi {name}"


        def broken_rewrite(values: dict) -> dict:
            raise KeyError("first")


        @node("hi", title="Hi", description="Says hi to a person.", replaces=hi_v1, upgrade=Upgrade(rewrite=broken_rewrite))
        def hi_v2(first: str) -> str:
            return first


        checked = NodeRegistry()
        checked.add(hi_v2)
        check_registry(checked)
    """
    return f"Upgrading to version {newest} raised {error}. Its developer can find the details in the conductor log."


@REGISTRY_CHECK_PROBLEMS.problem()
def subgraph_broken(*loc: str | int, reason: str, problems: list[dict[str, Any]]) -> str:
    """A subgraph's own graph doesn't compile on the registry's nodes.

    A graph that uses the subgraph would have these problems, so it can't run, and ``registry.describe()`` leaves the
    subgraph out, since it has no inputs or outputs to offer. In the example below, the subgraph uses ``upper``, which
    the registry doesn't have.

    To fix it, add the nodes it uses to the registry, or fix its graph record.

    Args:
        loc: The subgraph and its version.
        reason: The message of each problem that compiling its graph reported.
        problems: Those problems, as JSON.

    Example:
        capitals = subgraph("capitals", title="Capitals", description="Puts a text in capitals.")(make_graph(upper))
        checked = NodeRegistry()
        checked.add(capitals)
        check_registry(checked)
    """
    return f"The subgraph's graph doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem()
def changed_its_inputs(*loc: str | int, input: str) -> str:  # noqa: A002  the ctx key a developer reads
    """A node changed a value it was given, in place.

    The same value can be given to several calls of a run. So a change made in one call is seen by the others.

    To fix it, copy the value before changing it, such as ``sorted(texts)`` in place of ``texts.sort()``.

    Args:
        loc: The node and its version.
        input: The input whose value was changed.

    Example:
        @node("sort_in_place", title="Sort", description="Sorts a list of texts.")
        def sort_in_place(texts: list[str]) -> list[str]:
            texts.sort()
            return texts


        checked = NodeRegistry()
        checked.add(sort_in_place)
        check_registry(checked, examples={"sort_in_place": [{"texts": ["b", "a"]}]})
    """
    return f"The node changed its input {input!r} in place. Several calls of a run can be given the same value, so copy it before you change it."


@REGISTRY_CHECK_PROBLEMS.problem()
def run_failed(*loc: str | int, status: str, reason: str, problems: list[dict[str, Any]]) -> str:
    """A node's example ran, and the run ended early.

    The run failed, was stopped, was cancelled or timed out. ``ctx.problems`` has the problem the run ended on. When
    the node raised an exception of its own, the message names the exception's class.

    To fix it, fix the node or the example.

    If a node is known to fail, give it a test of its own and mark that test ``pytest.mark.xfail(strict=True)``. In
    that test, check only that node, with ``registry.filter(...)``. Once the node is fixed, the test fails, which
    tells you to remove the mark.

    Args:
        loc: The node and its version.
        status: How the run ended.
        reason: The message of the problem the run ended on. For an exception the node raised, its class name.
        problems: That problem, as JSON.

    Example:
        @node("divide_by", title="Divide", description="Divides one by a number.")
        def divide_by(n: int) -> float:
            return 1 / n


        checked = NodeRegistry()
        checked.add(divide_by)
        check_registry(checked, examples={"divide_by": [{"n": 0}]})
    """
    return f"The example's run ended {status}: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("run_failed", variant="not_compiled")
def example_not_compiled(*loc: str | int, reason: str, problems: list[dict[str, Any]]) -> str:
    """A node's example doesn't compile, and the problem isn't in a ``compute_...`` hook.

    An example gives a value for some of the node's inputs, by name. Each name must be an input of the node, and each
    value must be of that input's type. In the example below, ``txt`` is a typo for ``text``.

    To fix it, fix the example.

    Args:
        loc: The node and its version.
        reason: The message of each problem that compiling reported.
        problems: Those problems, as JSON.

    Example:
        checked = NodeRegistry()
        checked.add(upper)
        check_registry(checked, examples={"upper": [{"txt": "a"}]})
    """
    return f"The example doesn't compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("run_failed", variant="refused")
def example_refused(*loc: str | int, reason: str, problems: list[dict[str, Any]]) -> str:
    """A node's example compiles, but running it is refused.

    The run is missing something it needs. In the example below, the node takes ``Context[HostContext]``, and no
    context is given.

    To fix it, give the run what it needs, such as
    ``check_registry(checked, context=HostContext(caller=Caller("Ida")))``.

    Args:
        loc: The node and its version.
        reason: The message of each problem the run was refused for.
        problems: Those problems, as JSON.

    Example:
        served = NodeRegistry(context_type=HostContext)
        served.add(who)
        check_registry(served, examples={"who": [{}]})
    """
    return f"The example's run was refused: {reason}"
