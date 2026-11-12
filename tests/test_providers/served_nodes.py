"""The nodes the served app's tests run, shared by worker processes through two files in the app's directory.

``gate`` waits for the file ``open`` to exist, so a test can stop a run in the middle of a resume call; every call a test counts is a line
appended to ``calls``, so the calls of every worker are counted together.
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
    """A class the page's run context is not, so a graph with ``needs_missing`` has a problem when it is compiled."""


@served.node("needs_missing", title="Needs missing", description="For a test.")
def needs_missing(value: Context[Missing]) -> str:
    return ""
