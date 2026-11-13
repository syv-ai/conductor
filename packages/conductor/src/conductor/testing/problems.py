"""The kinds of problem ``check_registry`` returns, one function each, in ``REGISTRY_CHECK_PROBLEMS``.

A kind is called to make its ``Problem``, as the kinds of ``PROBLEMS`` are (``ProblemCatalogue.problem`` gives the
rules)::

    needs_example("coded", 1, inputs=["code"])
    # Problem(type="needs_example", loc=("coded", 1),
    #         msg="its hooks read ['code'], and no value can be made up for them; give the node an example", ctx={"inputs": ["code"]})

Every problem here is about one version of a node, so its ``loc`` is ``(node_id, version)``. A developer reads them
when a test fails; a host never gets one from a graph or a run, so they are not in ``PROBLEMS``, the catalogue of the
problems a graph or a run reports. Some kinds share a code (``hook_failed``, ``upgrade_broken``, ``run_failed``): the
decorator names the code and a variant, which the problem's ``ctx.variant`` gives, and each kind's message says which
case it is. Each kind's docstring is its entry on the
problems page (``python -m conductor.about problems``); its example runs after the setup at the top of that page, and
a test checks it makes this kind of problem.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.problem import ProblemCatalogue

#: The kinds of problem ``check_registry`` returns.
REGISTRY_CHECK_PROBLEMS = ProblemCatalogue()


@REGISTRY_CHECK_PROBLEMS.problem()
def example_for_no_node(*loc: str | int, named: str) -> str:
    """An example is for a node id, or a node id and version, that the registry does not have.

    Correct the id in ``examples``, or remove the example.

    Args:
        loc: The node id the example is for, and the version when it names one.
        named: The example's key in ``examples``, as written.

    Example:
        checked = NodeRegistry()
        checked.add(upper)
        check_registry(checked, examples={"uper": [{"text": "a"}]})
    """
    return f"{named} names no node of the registry; check the id, or remove the example"


@REGISTRY_CHECK_PROBLEMS.problem()
def needs_example(*loc: str | int, inputs: list[str]) -> str:
    r"""A node's ``compute_...`` hook reads a required input, and ``check_registry`` cannot make up a value for it.

    Without an example, a hook is called with the node's defaults and a made-up value for each required input it
    reads. A value that must match a pattern cannot be made up: give the node an example,
    ``examples={"coded": [{"code": "1234"}]}``.

    Args:
        loc: The node and its version.
        inputs: The inputs no value could be made up for.

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
    return f"its hooks read {inputs}, and no value can be made up for them; give the node an example"


@REGISTRY_CHECK_PROBLEMS.problem()
def hook_failed(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node's ``compute_...`` hook failed when called with the node's defaults and made-up values.

    Make the hook work on those values, or give the node an example it works on.

    Args:
        loc: The node and its version.
        reason: The message of the problem compiling the node reported.
        cause: The code of that problem.
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
    return f"a hook failed on the node's defaults and made-up values: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("hook_failed", variant="on_an_example")
def hook_failed_on_an_example(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node's example cannot be compiled, and every problem is in a ``compute_...`` hook.

    Fix the hook, or the example. In the example, ``pick``'s hook raises for an empty text.

    Args:
        loc: The node and its version.
        reason: The message of each problem compiling reported.
        cause: The code of each.
        problems: Those problems, as JSON.

    Example:
        checked = NodeRegistry()
        checked.add(pick)
        check_registry(checked, examples={"pick": [{"names": ""}]})
    """
    return f"the example does not compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem()
def upgrade_broken(*loc: str | int, newest: int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node moved from an older version to the newest has problems when compiled.

    Fix the newest version's ``Upgrade``, so a graph saved with the older version compiles cleanly after the move. In
    the example, the rewrite gives ``first`` a number where the newest version takes a text.

    Args:
        loc: The node and the older version.
        newest: The version it was moved to.
        reason: The message of each problem compiling reported.
        cause: The code of each.
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
    return f"after moving to version {newest}: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("upgrade_broken", variant="raised")
def upgrade_raised(*loc: str | int, newest: int, error: str, reason: str) -> str:
    """Moving a node from an older version to the newest raised an exception.

    Fix the newest version's ``Upgrade``, so a graph saved with the older version is moved without an error.

    Args:
        loc: The node and the older version.
        newest: The version it was moved to.
        error: The exception's class name.
        reason: The exception's message.

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
    return f"moving to version {newest} raised {error}: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem()
def changed_its_inputs(*loc: str | int, input: str) -> str:  # noqa: A002  the ctx key a developer reads
    """A node changed a value it was given, in place.

    The same value can be given to several calls of a run, so a change in one call is seen by the others. Copy the
    value before changing it: ``sorted(texts)`` in place of ``texts.sort()``.

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
    return f"the node changed {input!r}, a value it was given; several calls of a run can be given the same value, so copy it before changing it"


@REGISTRY_CHECK_PROBLEMS.problem()
def run_failed(*loc: str | int, status: str, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node's example ran and ended early: it failed, stopped, was cancelled or timed out.

    Fix the node or the example. ``ctx.cause`` has the run's own problem; when the node raised an exception of its own,
    the message names its class. A node that is known to fail goes in a test of its own, marked
    ``pytest.mark.xfail(strict=True)``, that checks ``registry.filter(...)`` down to that node, so the test fails once
    the node is fixed.

    Args:
        loc: The node and its version.
        status: How the run ended.
        reason: The message of the problem it ended on; for an exception of the node's own, its class name.
        cause: The code of that problem.
        problems: That problem, as JSON.

    Example:
        @node("divide_by", title="Divide", description="Divides one by a number.")
        def divide_by(n: int) -> float:
            return 1 / n


        checked = NodeRegistry()
        checked.add(divide_by)
        check_registry(checked, examples={"divide_by": [{"n": 0}]})
    """
    return f"the example ended {status}: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("run_failed", variant="not_compiled")
def example_not_compiled(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node's example cannot be compiled, for a problem that is not in a ``compute_...`` hook.

    Fix the example: its values must be inputs of the node, of the inputs' types.

    Args:
        loc: The node and its version.
        reason: The message of each problem compiling reported.
        cause: The code of each.
        problems: Those problems, as JSON.

    Example:
        checked = NodeRegistry()
        checked.add(upper)
        check_registry(checked, examples={"upper": [{"txt": "a"}]})
    """
    return f"the example does not compile: {reason}"


@REGISTRY_CHECK_PROBLEMS.problem("run_failed", variant="refused")
def example_refused(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node's example compiles, and running it is refused.

    Give the run what it needs: in the example, the run's context for the node's ``Context[HostContext]``, with
    ``check_registry(checked, context=HostContext(caller=Caller("Ida")))``.

    Args:
        loc: The node and its version.
        reason: The message of each problem the run was refused for.
        cause: The code of each.
        problems: Those problems, as JSON.

    Example:
        served = NodeRegistry(context_type=HostContext)
        served.add(who)
        check_registry(served, examples={"who": [{}]})
    """
    return f"the example was refused: {reason}"
