"""The problems a run can stop on, in ``run.cause``.

This is the "Why a run stopped" section of the problems page. A run that stops early doesn't raise. ``run.resume()``
returns the run, and ``run.cause`` is one of these problems. Most are about one call of a node that failed. A few,
like a timeout or a cancel, are about the run as a whole.

Code that reports one imports this module and calls the kind, like ``run_problems.timeout(*loc, seconds=30.0)``.
"""

from __future__ import annotations

from typing import Any

from .type import ProblemCatalogue

#: This module's section of the problems page, with its kinds.
RUN_PROBLEMS = ProblemCatalogue(
    "Why a run stopped",
    "A run that ends early has one of these as its `run.cause`. It is also the `problem` of a `node_failed` event. "
    "For a call that failed, `loc` is the node's id and the call's item indexes, like `(\"shout\", 1)`. Inside a "
    "subgraph, the subgraph node's id comes first. For the run as a whole, `loc` is `()`. `resume()` goes on from "
    "where the run stopped.",
)


@RUN_PROBLEMS.problem()
def ask_not_declared(*loc: str | int) -> str:
    """A node raised ``Ask``, and its ``@node`` doesn't say ``asks=True``.

    A node that may wait for a person says so where it is declared. That way a host knows, before a run, which graphs
    can pause for a person.

    To fix it, add ``asks=True`` to the ``@node``.

    Args:
        loc: The call that raised it.

    Example:
        @registry.node("confirm_quietly", title="Confirm", description="Asks a person to confirm, without saying so.")
        def confirm_quietly(text: str) -> bool:
            raise Ask("Sure?")


        why_it_stopped(Graph.compile(make_graph(confirm_quietly), registry).run(text="a"))
    """
    return "The node asked a person, but its @node doesn't say it may. Add asks=True to its @node."


@RUN_PROBLEMS.problem()
def answer_declined(*loc: str | int, reason: str | None = None) -> str:
    """A person declined to answer a node's question, with ``run.decline``.

    The call fails as if the node had raised. It has no outcome, and nothing that reads its outputs is called.

    To fix it, call ``resume()``. The node is called again and asks again.

    Args:
        loc: The call that asked.
        reason: The reason the host gave to ``run.decline``. It is left out when the host gave none. The message
            doesn't quote it.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run(amount=5000)
        why_it_stopped(waiting.decline(waiting.waiting[0], reason="Over budget"))
    """
    return "The question was declined, so the call failed. Call resume() to ask it again."


@RUN_PROBLEMS.problem()
def execution_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception that isn't a ``NodeError``.

    The call isn't retried, whatever its ``Policy(retries=)`` says. Only an ``ExternalError``, or an exception class
    named in ``Policy(retry_on=)``, is: running the same computation again fails the same way.

    A problem may be shown to anyone, and the exception's text might quote a value. So the text and the traceback go to
    the ``conductor`` logger, and ``ctx.error`` only has the exception's class name.

    To fix it, fix the node, then call ``resume()``. If the exception comes from an outside service, name its class in
    ``Policy(retry_on=)``, or raise ``ExternalError``.

    Args:
        loc: The call that failed.
        error: The exception's class name.

    Example:
        @registry.node("divide", title="Divide", description="Divides one by a number.")
        def divide(n: int) -> float:
            return 1 / n


        why_it_stopped(Graph.compile(make_graph(divide), registry).run(n=0))
    """
    return (
        f"The node raised {error}, which isn't retried: only an ExternalError, or a class in Policy(retry_on=), is."
        " Its developer can find the details in the conductor log."
    )


@RUN_PROBLEMS.problem()
def external_failed(*loc: str | int, error: str | None = None) -> str:
    """A node kept raising an exception that its ``Policy(retry_on=)`` names, on every attempt.

    This means the outside world failed, with a network error, a rate limit or a server error. An ``ExternalError`` the
    node raises itself gets this code too, with the node's own message and ``ctx``.

    To fix it, call ``resume()`` to try again. For more attempts each time, increase ``Policy(retries=)``.

    Args:
        loc: The call that failed.
        error: The exception's class name. Missing for an ``ExternalError`` the node raised, whose ``ctx`` is its own.

    Example:
        @registry.node("fetch", title="Fetch", description="Fetches a page, and fails.", policy=Policy(retry_on=(ConnectionError,)))
        def fetch(url: str) -> str:
            raise ConnectionError(url)


        why_it_stopped(Graph.compile(make_graph(fetch), registry).run(url="https://example.com"))
    """
    return "A call to an outside service failed."


@RUN_PROBLEMS.problem()
def node_error(*loc: str | int) -> str:
    """A node raised a ``NodeError`` without a code of its own.

    The problem's message and ``ctx`` are the ones the node gave its ``NodeError``. The message below is only a fallback,
    for a host that has no message of its own for this code.

    A node gives a code of its own with ``NodeError(message, code=...)``. It gives the values a host can make its own
    message from with ``ctx=``, which takes JSON values only.

    Args:
        loc: The call that failed.

    Example:
        @registry.node("read_pdf", title="Read PDF", description="Reads the text from a PDF file, and fails.")
        def read_pdf(path: str) -> str:
            raise NodeError("The document could not be read.")


        why_it_stopped(Graph.compile(make_graph(read_pdf), registry).run(path="a.pdf"))
    """
    return "The node reported a failure."


@RUN_PROBLEMS.problem()
def invalid_ask(*loc: str | int, reason: str) -> str:
    """A node raised an ``Ask`` whose choices or default don't fit the node's outputs.

    The answer to an ``Ask`` becomes the node's outputs. So its choices and its default must be values the outputs'
    types take. In the example, ``confirm`` returns a ``bool``, and ``"maybe"`` isn't one.

    To fix it, fix the ``Ask``. ``ctx.reason`` says which part doesn't fit.

    Args:
        loc: The call that asked.
        reason: What is wrong with the question.

    Example:
        @registry.node("confirm", title="Confirm", description="Asks a person to confirm.", asks=True)
        def confirm(text: str) -> bool:
            raise Ask("Sure?", default="maybe")


        why_it_stopped(Graph.compile(make_graph(confirm), registry).run(text="a"))
    """
    return f"The node's Ask doesn't fit its outputs: {reason.rstrip('.')}. Fix the Ask."


@RUN_PROBLEMS.problem()
def invalid_edge_value(
    *loc: str | int,
    name: str,
    reason: str,
    record_field: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    column: str | None = None,
) -> str:
    """The value an edge gives an input is refused by the input's type, just before the call.

    Compiling checks that the types on each edge fit. Some things it can't check until there is a value, so they are
    checked on each call:

    - a limit or choices on the input,
    - a host type's own check,
    - a table whose columns are only known once it is made,
    - a field of a record whose type declares no fields.

    In the example, ``letters`` returns 12, and ``small`` takes at most 3.

    To fix it, make the node upstream return what the input takes. Or put a node in between that converts it.

    Args:
        loc: The call that was about to be made.
        name: The input.
        reason: Why the value is refused, without quoting it.
        record_field: The field of a record the edge names, if the record has no field by that name.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("letters", title="Letters", description="Counts the letters in a text.")
        def letters(text: str) -> int:
            return len(text)


        @registry.node("small", title="Small", description="Passes on a small number.")
        def small(n: Annotated[int, Param(le=3)]) -> int:
            return n


        why_it_stopped(Graph.compile(make_graph(letters, small), registry).run(text="many letters"))
    """
    return (
        f"The value an edge gave '{name}' isn't one it takes: {reason.rstrip('.')}. "
        "Fix the node upstream, or put a node in between that converts the value."
    )


@RUN_PROBLEMS.problem()
def invalid_output(
    *loc: str | int,
    reason: str,
    name: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    column: str | None = None,
) -> str:
    """A node returned a value that its return type doesn't allow.

    Return values are checked strictly, so ``"12"`` is never an ``int``. The node's branches are checked too: two
    branches with a value are refused, and so is ``SKIPPED`` on an output that isn't a branch. The reference has
    the rule under "Branches" (``python -m conductor.about the-node-contract``).

    To fix it, make the node return what it declares. Outputs that may each have a value on one call aren't branches:
    type them ``X | None``, and return ``None`` for one that has no value. To make an output a branch, give it the
    default ``SKIPPED``, written ``(type, SKIPPED)`` in a ``compute_outputs`` hook.

    Args:
        loc: The call that returned it.
        reason: Why the value is refused, without quoting it.
        name: The output, if the node has several.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("guess", title="Guess", description="Returns a text where a number is declared.")
        def guess(text: str) -> int:
            return text  # type: ignore[return-value]


        why_it_stopped(Graph.compile(make_graph(guess), registry).run(text="12"))
    """
    return f"The node returned something it doesn't declare: {reason.rstrip('.')}. Fix what the node returns."


@RUN_PROBLEMS.problem()
def item_limit(*loc: str | int, series: str, items: int, max_items: int) -> str:
    """A series has more items than the run's ``max_items``, so the run stopped before calling any node for them.

    Only a series that a node loops over counts. A series that is collected, or passed on whole, costs no calls.

    To fix it, call ``run.with_max_items(n).resume()``. It goes on with a higher limit, and calls nothing again.

    Args:
        loc: The call that returned the series.
        series: The series, written as its source, like ``"lines.result"``.
        items: How many items it has.
        max_items: The run's item limit.

    Example:
        why_it_stopped(Graph.compile(make_graph(words, upper), registry).run(text="a b c", max_items=2))
    """
    return f"{series} has {items} items, more than this run's max_items of {max_items}. Raise max_items and resume the run."


@RUN_PROBLEMS.problem()
def question_unwritable(*loc: str | int, error: str, input: str) -> str:  # noqa: A002  the ctx key a host reads
    """A node asked a person, and one of its arguments can't be written as JSON to show them.

    A question carries the arguments of the call that asked, so the person answering can see them. Those must be JSON.
    Why the argument can't be written might quote it, so the exception's text and its traceback go to the
    ``conductor`` logger, and ``ctx.error`` only has the exception's class name.

    To fix it, give the input a type that pydantic can write as JSON. Or ask in a node that takes such a type.

    Args:
        loc: The call that asked.
        error: The class name of the exception that writing the argument raised.
        input: The input.

    Example:
        @registry.node("approve-file", title="Approve file", description="Asks a person to approve sending a file.", asks=True)
        def approve_file(data: bytes) -> bool:
            raise Ask("Send this file?")


        why_it_stopped(Graph.compile(make_graph(approve_file), registry).run(data=b"\\xff"))
    """
    return (
        f"The node asked a person, but its input '{input}' can't be written as JSON to show them: writing it raised {error}. "
        "Give that input a type pydantic can write as JSON."
    )


@RUN_PROBLEMS.problem()
def run_cancelled() -> str:
    """The ``cancel`` event given to ``run.stream`` was set.

    Calls that returned before the cancel are kept. To go on, save the run and call ``resume()`` on it.

    Example:
        async def cancelled() -> Run:
            stop = asyncio.Event()
            stop.set()
            async for event in Graph.compile(make_graph(upper), registry).with_inputs(text="a").stream(cancel=stop):
                if event.type == "run_end":
                    return event.run
            raise AssertionError("the stream ended without a run_end event")


        why_it_stopped(asyncio.run(cancelled()))
    """
    return "The run was cancelled. Resume it to go on, or start a new run."


@RUN_PROBLEMS.problem()
def run_timed_out(*, timeout: float) -> str:
    """A call of ``run``, ``resume`` or ``stream`` took longer than the ``timeout`` it was given.

    To fix it, call ``resume()``. It goes on, and calls nothing again that already returned. Or give a longer
    ``timeout``, or none.

    Args:
        timeout: The ``timeout`` given, in seconds.

    Example:
        @registry.node("slow", title="Slow", description="Waits a moment, then returns the text.")
        def slow(text: str) -> str:
            time.sleep(0.2)
            return text


        why_it_stopped(Graph.compile(make_graph(slow, upper), registry).run(text="a", timeout=0.05))  # upper is never called
    """
    return f"The run took longer than its timeout of {timeout} seconds, so it stopped. Resume it to go on from where it stopped."


@RUN_PROBLEMS.problem()
def timeout(*loc: str | int, seconds: float) -> str:
    """One call of a node took longer than its ``Policy(attempt_timeout=)``.

    A call that timed out isn't retried, whatever its ``Policy(retries=)`` says.

    To fix it, increase the node's ``attempt_timeout``, or make the call faster.

    Args:
        loc: The call that failed.
        seconds: The node's ``attempt_timeout``.

    Example:
        @registry.node("stuck", title="Stuck", description="Waits too long.", policy=Policy(attempt_timeout=0.05))
        def stuck(text: str) -> str:
            time.sleep(0.2)
            return text


        why_it_stopped(Graph.compile(make_graph(stuck), registry).run(text="a"))
    """
    return (
        f"The node didn't return within {seconds} seconds, and a timed-out call isn't retried."
        " Its Policy(attempt_timeout=) sets how long it may take."
    )
