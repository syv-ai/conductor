"""The nodes that the tests of the served app run.

The app runs in several worker processes, so these nodes can't share Python state. They share two files in the app's
directory instead:

- ``gate`` waits until a file named ``open`` exists. That lets a test pause a run in the middle of a resume call.
- Each call a test counts adds a line to the file ``calls``. That way the calls of every worker are counted together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from conductor import Ask, Context, NodeRegistry, Series

served = NodeRegistry()


def called(what: str) -> None:
    with Path("calls").open("a") as calls:
        calls.write(what + "\n")


def wait_for_the_gate() -> None:
    for _ in range(2000):
        if Path("open").exists():
            return
        time.sleep(0.01)
    raise AssertionError("the test never opened the gate")


@served.node("gate", title="Gate", description="For a test.")
def gate() -> float:
    called("gate")
    wait_for_the_gate()
    return 1.0


@served.node("ask", title="Ask", description="For a test.", asks=True)
def ask(value: float) -> bool:
    raise Ask("Approve?")


@served.node("greet", title="Greet", description="For a test.")
def greet(name: str) -> str:
    return f"hi {name}"


@served.node("numbers", title="Numbers", description="For a test.")
def numbers(n: int) -> Series[int]:
    return list(range(n))


@served.node("work", title="Work", description="For a test.")
def work(i: int) -> int:
    called(f"work {i}")
    time.sleep(0.02)
    return i


@served.node("approve_each", title="Approve each", description="For a test.", asks=True)
def approve_each(i: int) -> bool:
    raise Ask(f"Approve {i}?")


@served.node("after", title="After", description="For a test.")
def after(approved: bool) -> bool:
    called("after")
    wait_for_the_gate()
    return approved


@dataclass(frozen=True)
class Missing:
    """A context class that the app's registry doesn't use.

    ``needs_missing`` asks for a ``Context[Missing]``, but the app's run context is another class. So a graph that uses
    ``needs_missing`` has a problem when it is compiled.
    """


@served.node("needs_missing", title="Needs missing", description="For a test.")
def needs_missing(value: Context[Missing]) -> str:
    return ""
