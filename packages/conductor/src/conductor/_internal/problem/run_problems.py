"""The kinds of problem a run ends early on, its ``run.cause``: the section "Why a run stopped" of ``PROBLEMS``.

A module that reports one imports this module and calls the kind: ``run_problems.timeout(*loc, seconds=30.0)``.
"""

from __future__ import annotations

from typing import Any

from .type import ProblemCatalogue

#: The kinds of this section of the problems page.
RUN_PROBLEMS = ProblemCatalogue(
    "Why a run stopped",
    "The `run.cause` of a run that ended early, and the `problem` of a `node_failed` event. For a call that failed, "
    '`loc` is the node\'s id and the call\'s item indexes: `("shout", 1)`; inside a subgraph, the subgraph node\'s id '
    "comes first. For the run as a whole, `loc` is `()`. `resume()` goes on from where the run stopped.",
)


@RUN_PROBLEMS.problem()
def ask_not_declared(*loc: str | int) -> str:
    """A node raised ``Ask``, and its ``@node`` does not say ``asks=True``.

    A node that may wait for a person says so where it is declared, so a host knows before a run which graphs can
    pause. Add ``asks=True`` to the ``@node``.

    Args:
        loc: The call that raised it.

    Example:
        @registry.node("confirm_quietly", title="Confirm", description="Asks a person to confirm, without saying so.")
        def confirm_quietly(text: str) -> bool:
            raise Ask("Sure?")


        why_it_stopped(Graph.compile(make_graph(confirm_quietly), registry).run(text="a"))
    """
    return "The node asked a person without saying so; add asks=True to its @node."


@RUN_PROBLEMS.problem()
def execution_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception that is not a ``NodeError``.

    The exception's text and traceback go to the ``conductor`` log and never into the problem, since a problem may be
    shown to anyone; ``ctx.error`` is its class name. Fix the node, then ``resume()``.

    Args:
        loc: The call that failed.
        error: The exception's class name.

    Example:
        @registry.node("divide", title="Divide", description="Divides one by a number.")
        def divide(n: int) -> float:
            return 1 / n


        why_it_stopped(Graph.compile(make_graph(divide), registry).run(n=0))
    """
    return "The node raised an exception; its developer finds it in the conductor log."


@RUN_PROBLEMS.problem()
def external_failed(*loc: str | int, error: str | None = None) -> str:
    """A node raised an exception its ``Policy(retry_on=)`` names, on every attempt.

    The outside world failed: a network error, a rate limit, a server error. ``resume()`` tries again; raise
    ``Policy(retries=)`` for more attempts. An ``ExternalError`` a node raises itself has this code too, with the
    node's own message and ``ctx``.

    Args:
        loc: The call that failed.
        error: The exception's class name; absent for an ``ExternalError`` the node raised, whose ``ctx`` is its own.

    Example:
        @registry.node("fetch", title="Fetch", description="Fetches a page, and fails.", policy=Policy(retry_on=(ConnectionError,)))
        def fetch(url: str) -> str:
            raise ConnectionError(url)


        why_it_stopped(Graph.compile(make_graph(fetch), registry).run(url="https://example.com"))
    """
    return "An outside service did not answer."


@RUN_PROBLEMS.problem()
def failed(*loc: str | int) -> str:
    """A node raised a ``NodeError`` that gives no code of its own.

    The problem is made from the node's ``NodeError``, not by calling this function: its message and ``ctx`` are the
    ones the node gave. This function declares the code, and its message is a fallback for a host that has no message
    of its own for the code. A node gives a code of its own with
    ``NodeError(message, code=)``, and the values a host fills its own message from with ``ctx=``, JSON values only.

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
    """A node raised an ``Ask`` whose choices or default are not values of the type its answer is read by.

    The answer to an ``Ask`` is the node's outputs, a value of its function's return type. So its choices and default
    must be of that type. Fix the ``Ask``; ``ctx.reason`` says which part does not fit.

    Args:
        loc: The call that asked.
        reason: What is wrong with the question.

    Example:
        @registry.node("confirm", title="Confirm", description="Asks a person to confirm.", asks=True)
        def confirm(text: str) -> bool:
            raise Ask("Sure?", default="maybe")


        why_it_stopped(Graph.compile(make_graph(confirm), registry).run(text="a"))
    """
    return f"The node asked a question with choices or a default the answer's type does not take: {reason.rstrip('.')}. Fix the node's Ask."


@RUN_PROBLEMS.problem()
def invalid_edge_value(
    *loc: str | int, name: str, reason: str, record_field: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None,
) -> str:
    """The value an edge gives an input is refused by the input's type, just before the call.

    Compiling checks that an edge's types fit; some things it cannot check, and those are checked on each call: a
    limit or choices on the input, a host type's own check, a table whose columns are known only once it is made, a
    field of a record whose type declares no fields. Make the upstream node return what the input takes, or put a
    node between that converts it. In the example, ``letters`` returns 12 and ``small`` takes at most 3.

    Args:
        loc: The call that was about to be made.
        name: The input.
        reason: Why the value is refused, without quoting it.
        record_field: The field of a record the edge names, when the record has no field by that name.
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
        f"The value an edge gave '{name}' is not one it takes: {reason.rstrip('.')}. "
        "Fix the node upstream, or wire an output of the type the input takes."
    )


@RUN_PROBLEMS.problem()
def invalid_output(*loc: str | int, reason: str, name: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A node returned a value its return type does not declare.

    Return what the node declares. Return values are checked strictly: ``"12"`` is never an ``int``.

    Args:
        loc: The call that returned it.
        reason: Why the value is refused, without quoting it.
        name: The output, when the node has several.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("guess", title="Guess", description="Returns a text where a number is declared.")
        def guess(text: str) -> int:
            return text  # type: ignore[return-value]


        why_it_stopped(Graph.compile(make_graph(guess), registry).run(text="12"))
    """
    return f"The node returned a value that is not what it declares: {reason.rstrip('.')}. Fix what the node returns."


@RUN_PROBLEMS.problem()
def item_limit(*loc: str | int, series: str, items: int, max_items: int) -> str:
    """A series has more items than the run's ``max_items``, so the run stopped before calling any node for them.

    Only a series that a node loops over counts: one that is collected or passed on whole costs no calls.
    ``run.with_max_items(n).resume()`` goes on with a higher limit, and calls nothing again.

    Args:
        loc: The call that returned the series.
        series: The series, as its source is written: ``"lines.result"``.
        items: How many items it has.
        max_items: The run's item limit.

    Example:
        why_it_stopped(Graph.compile(make_graph(words, upper), registry).run(text="a b c", max_items=2))
    """
    return f"{series} has {items} items, more than this run's max_items of {max_items}; raise max_items and resume the run."


@RUN_PROBLEMS.problem()
def question_unwritable(*loc: str | int, reason: str, input: str) -> str:  # noqa: A002  the ctx key a host reads
    """A node asked a person, and one of its arguments cannot be written as JSON to show them.

    A question carries the arguments of the call that asked, for the person answering it. Give the input a type
    pydantic writes as JSON, or ask in a node that takes one.

    Args:
        loc: The call that asked.
        reason: Why the argument cannot be written.
        input: The input.

    Example:
        @registry.node("approve-file", title="Approve file", description="Asks a person to approve sending a file.", asks=True)
        def approve_file(data: bytes) -> bool:
            raise Ask("Send this file?")


        why_it_stopped(Graph.compile(make_graph(approve_file), registry).run(data=b"\\xff"))
    """
    return (
        f"The node asked a person, but its argument '{input}' cannot be written as JSON to show them: {reason.rstrip('.')}. "
        "Give that input a type pydantic writes as JSON."
    )


@RUN_PROBLEMS.problem()
def run_cancelled() -> str:
    """The ``cancel`` event given to ``run.stream`` was set.

    Save the run and ``resume()`` it to go on. Calls that returned before the cancel are kept.

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
    return "The run was cancelled. Resume it to carry on, or start a new run."


@RUN_PROBLEMS.problem()
def run_timed_out(*, timeout: float) -> str:
    """A call of ``run``, ``resume`` or ``stream`` took longer than the ``timeout`` it was given.

    ``resume()`` goes on, and calls nothing again that returned. Give a longer ``timeout``, or none.

    Args:
        timeout: The ``timeout`` given, in seconds.

    Example:
        @registry.node("slow", title="Slow", description="Waits a moment, then returns the text.")
        def slow(text: str) -> str:
            time.sleep(0.2)
            return text


        why_it_stopped(Graph.compile(make_graph(slow, upper), registry).run(text="a", timeout=0.05))  # upper is never called
    """
    return "The run took too long and was stopped; run.resume() goes on from where it stopped."


@RUN_PROBLEMS.problem()
def timeout(*loc: str | int, seconds: float) -> str:
    """A node's call took longer than its ``Policy(attempt_timeout=)``.

    Raise the node's ``attempt_timeout``, or make the call faster. A call that timed out is not retried.

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
    return "The node did not return in time; its Policy(attempt_timeout=) sets how long it may take."
