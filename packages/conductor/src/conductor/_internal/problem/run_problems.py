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


#: How to fix a node that asks a person without saying so. ``@node`` refuses a function that raises ``Ask`` with the
#: same fix (``ask_not_declared_in_source``).
ADD_ASKS = "Add asks=True to its @node."


@RUN_PROBLEMS.problem
def ask_not_declared(*loc: str | int) -> str:
    """A node raised ``Ask``, and its ``@node`` doesn't say ``asks=True``.

    A node that may wait for a person says so where it is declared. That way a host knows, before a run, which graphs
    can pause for a person. ``@node`` already refuses a function whose own body has a ``raise Ask(...)``
    (``ask_not_declared_in_source``). The run refuses every other ``Ask``, such as one raised by a function the node
    calls or defines inside it.

    To fix it, add ``asks=True`` to the ``@node``.

    Args:
        loc: The call that raised it.

    Example:
        def ask_if_sure() -> bool:
            raise Ask("Sure?")


        @registry.node("confirm_quietly", title="Confirm", description="Asks a person to confirm, without saying so.")
        def confirm_quietly(text: str) -> bool:
            return ask_if_sure()


        why_it_stopped(Graph.compile(make_graph(confirm_quietly), registry).run({"text": "a"}))
    """
    return f"The node asked a person, but its @node doesn't say it may. {ADD_ASKS}"


@RUN_PROBLEMS.problem
def answer_declined(*loc: str | int, reason: str | None = None) -> str:
    """A person declined to answer a node's question, with ``run.with_decline``, and so ended the run for good.

    The run is ``declined``. Every call made before the decline keeps its outcome, so the run's results show what
    ran. No call is made any more, and nothing waits.

    To fix it, start a new run with ``run.again()``. It reuses what this run ran, and the node asks again.

    Args:
        loc: The call that asked.
        reason: The reason the host gave to ``run.with_decline``. It is left out when the host gave none. The message
            doesn't quote it.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        why_it_stopped(waiting.with_decline(waiting.waiting[0], reason="Over budget"))
    """
    return (
        "A person declined to answer, so the run ended here. The calls made before keep their results. "
        "Call run.again() to start a new run that asks again."
    )


@RUN_PROBLEMS.problem
def execution_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception that isn't a ``NodeError``.

    The call isn't retried. Which failures are is written in ``Policy``'s docstring and in the reference
    (``python -m conductor.about retries-and-timeouts``).

    A problem may be shown to anyone, and the exception's text might quote a value. So the text and the traceback go to
    the ``conductor`` logger, and ``ctx.error`` only has the exception's class name.

    To fix it, fix the node, then call ``resume()``.

    Args:
        loc: The call that failed.
        error: The exception's class name.

    Example:
        @registry.node("divide", title="Divide", description="Divides one by a number.")
        def divide(n: int) -> float:
            return 1 / n


        why_it_stopped(Graph.compile(make_graph(divide), registry).run({"n": 0}))
    """
    return (
        f"The node raised {error}, which isn't retried: only an ExternalError, or a class in Policy(retry_on=), is."
        " Its developer can find the details in the conductor log."
    )


@RUN_PROBLEMS.problem
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


        why_it_stopped(Graph.compile(make_graph(fetch), registry).run({"url": "https://example.com"}))
    """
    return "A call to an outside service failed."


@RUN_PROBLEMS.problem
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


        why_it_stopped(Graph.compile(make_graph(read_pdf), registry).run({"path": "a.pdf"}))
    """
    return "The node reported a failure."


@RUN_PROBLEMS.problem
def invalid_ask(*loc: str | int, reason: str, errors: list[dict[str, Any]]) -> str:
    """A node raised an ``Ask`` whose choices or default don't fit the node's outputs.

    The answer to an ``Ask`` becomes the node's outputs. So its choices and its default must be values the outputs'
    types take, and the default an answer a person could give. In the example, ``confirm`` returns a ``bool``, and
    ``"maybe"`` isn't one. Every fault is reported at once, in ``ctx.errors``, each ``loc`` starting with
    ``choices`` or ``default``, then the output's name, then a choice's index. A key of the mapping that is no output
    is ``"[key]"`` there, never quoted, since data may have built it.

    To fix it, fix the ``Ask``. ``ctx.reason`` says which part doesn't fit first. A node of several outputs takes
    ``choices=`` and ``default=`` as mappings by output name; a node of one output takes a list and a value.

    Args:
        loc: The call that asked.
        reason: Which part of the ``Ask`` is wrong first, and why.
        errors: Every fault, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting a value.

    Example:
        @registry.node("confirm", title="Confirm", description="Asks a person to confirm.", asks=True)
        def confirm(text: str) -> bool:
            raise Ask("Sure?", default="maybe")


        why_it_stopped(Graph.compile(make_graph(confirm), registry).run({"text": "a"}))
    """
    return f"The node's Ask doesn't fit its outputs: {reason.rstrip('.')}. Fix the Ask."


@RUN_PROBLEMS.problem
def invalid_edge_value(
    *loc: str | int,
    input: str,
    reason: str,
    field: str | None = None,
    errors: list[dict[str, Any]],
    column: str | None = None,
) -> str:
    """The value an edge gives an input is refused by the input's type, just before the call.

    Compiling checks that the types on each edge fit. Some things it can't check until there is a value, so they are
    checked on each call:

    - a limit or choices on the input,
    - a host type's own check,
    - a table whose columns are only known once it is made,
    - a field of a value whose type declares no fields, like a ``dict``. A value without the field the edge names
      has ``ctx.field``, and its one error in ``ctx.errors`` is ``no_such_field``.
    - a field that is ``SKIPPED``, such as the field of a model typed ``Any``. Only an output's whole value is a
      skipped branch, so the edge doesn't skip the node it feeds. It has ``ctx.field``, and its one error in
      ``ctx.errors`` is ``skipped_inside``.

    In the example, ``letters`` returns 12, and ``small`` takes at most 3.

    To fix it, make the node upstream return what the input takes. Or put a node in between that converts it.

    Args:
        loc: The call that was about to be made.
        input: The input.
        reason: Why the value is refused, without quoting it.
        field: The field the edge names, if the value has no field by that name.
        errors: What was refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("letters", title="Letters", description="Counts the letters in a text.")
        def letters(text: str) -> int:
            return len(text)


        @registry.node("small", title="Small", description="Passes on a small number.")
        def small(n: Annotated[int, Param(le=3)]) -> int:
            return n


        why_it_stopped(Graph.compile(make_graph(letters, small), registry).run({"text": "many letters"}))
    """
    return (
        f"The value an edge gave '{input}' isn't one it takes: {reason.rstrip('.')}. "
        "Fix the node upstream, or put a node in between that converts the value."
    )


@RUN_PROBLEMS.problem
def invalid_output(
    *loc: str | int,
    reason: str,
    output: str | None = None,
    errors: list[dict[str, Any]],
    column: str | None = None,
) -> str:
    """A node returned a value that its return type doesn't allow.

    The value may be of the wrong type or the wrong shape, or miss an output. For a value an output's type refuses,
    ``ctx.output`` names the output, and ``ctx.errors`` says where inside it the value is: a refused field of a returned
    dataclass is its own output, so its ``ctx.errors[0].loc`` is ``[]``. A fault in the shape of the return is the one
    error in ``ctx.errors``, and ``ctx.output`` names the output it is about, if any. An output missing from a
    returned mapping is pydantic's ``missing`` at the output's name, as an answer gives it, and a branch may be left
    out of the mapping. A key that is no output is pydantic's ``extra_forbidden``, or ``invalid_key`` for one that
    isn't text, at ``"[key]"``, with no ``ctx.output``: a key may be built from data, so it is never quoted. The others have
    ``loc`` ``[]`` and conductor's code, since pydantic has none: ``not_the_class`` (not the node's dataclass),
    ``not_a_mapping``, ``skipped_not_a_branch``, ``several_branches_given`` (as an answer's and an ``Ask``'s
    default's) and ``none_required`` (a node with no outputs returned something).

    Return values are checked strictly, so ``"12"`` is never an ``int``. The node's branches are checked too: two
    branches with a value are refused, and so is ``SKIPPED`` on an output that isn't a branch. ``SKIPPED`` is an
    output's whole value or one item of a ``Series``. The return isn't searched for it inside a value, such as an
    item of a list typed ``Any``: such a run is refused when it is saved (``run_unwritable``). The reference has the rule
    under "Branches" (``python -m conductor.about the-node-contract``).

    To fix it, make the node return what it declares. Outputs that may each have a value on one call aren't branches:
    type them ``X | None``, and return ``None`` for one that has no value. To make an output a branch, give it the
    default ``SKIPPED``, written ``(type, SKIPPED)`` in a ``compute_outputs`` hook.

    Args:
        loc: The call that returned it.
        reason: Why the value is refused, without quoting it.
        output: The output the fault is about, when there is one to name; ``result`` for a node of one output.
        errors: What was refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("guess", title="Guess", description="Returns a text where a number is declared.")
        def guess(text: str) -> int:
            return text  # type: ignore[return-value]


        why_it_stopped(Graph.compile(make_graph(guess), registry).run({"text": "12"}))
    """
    return f"The node's return doesn't fit its outputs: {reason.rstrip('.')}. Fix what the node returns."


@RUN_PROBLEMS.problem
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
        why_it_stopped(Graph.compile(make_graph(words, upper), registry).run({"text": "a b c"}, max_items=2))
    """
    return f"{series} has {items} items, more than this run's max_items of {max_items}. Raise max_items and resume the run."


@RUN_PROBLEMS.problem
def question_unwritable(*loc: str | int, error: str, input: str) -> str:
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
        @registry.node("half-emoji", title="Half emoji", description="Ends a text with half of an emoji's pair.")
        def half_emoji(name: str) -> str:
            return name + "\\ud83d"


        @registry.node("approve-text", title="Approve text", description="Asks a person to approve sending a text.", asks=True)
        def approve_text(text: str) -> bool:
            raise Ask("Send this text?")


        why_it_stopped(Graph.compile(make_graph(half_emoji, approve_text), registry).run({"name": "a"}))
    """
    return (
        f"The node asked a person, but its input '{input}' can't be written as JSON to show them: writing it raised {error}. "
        "Give that input a type pydantic can write as JSON."
    )


@RUN_PROBLEMS.problem
def run_cancelled() -> str:
    """The ``cancel`` event given to ``run.stream`` was set.

    Calls that returned before the cancel are kept. To go on, save the run and call ``resume()`` on it.

    Example:
        async def cancelled() -> Run:
            stop = asyncio.Event()
            stop.set()
            async for event in Graph.compile(make_graph(upper), registry).with_inputs({"text": "a"}).stream(cancel=stop):
                if event.type == "run_end":
                    return event.run
            raise AssertionError("the stream ended without a run_end event")


        why_it_stopped(asyncio.run(cancelled()))
    """
    return "The run was cancelled. Resume it to go on, or start a new run."


@RUN_PROBLEMS.problem
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


        why_it_stopped(Graph.compile(make_graph(slow, upper), registry).run({"text": "a"}, timeout=0.05))  # upper is never called
    """
    return f"The run took longer than its timeout of {timeout} seconds, so it stopped. Resume it to go on from where it stopped."


@RUN_PROBLEMS.problem
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


        why_it_stopped(Graph.compile(make_graph(stuck), registry).run({"text": "a"}))
    """
    return (
        f"The node didn't return within {seconds} seconds, and a timed-out call isn't retried."
        " Its Policy(attempt_timeout=) sets how long it may take."
    )
