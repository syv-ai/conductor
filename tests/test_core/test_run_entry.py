"""One way to run a graph: ``execute`` streams, ``await run`` returns the ending, ``run_sync`` is ``run`` outside a loop.

The three are root exports. ``run`` drains ``execute`` and returns the
event the leg ended on — complete, pending, error, cancelled or timed
out — so a pause is an ending like any other and nothing raises for it.
``run_sync`` is the same call for a script; under a running loop it
refuses, naming the call that works there. The quick start in the
reference is written for a notebook and runs both ways.
"""

import asyncio
import re
import textwrap
from importlib.resources import files
from typing import Annotated

import pytest
from conductor import CompiledGraph, DType, GraphRecord, NodeRecord, Param, Result, node
from conductor.widgets import Textarea
from helpers import registry_of


class Txt(DType, str):
    id = "run-entry-test-text"
    display_title = "Text"


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Out")]:
    return Txt(text.upper())


def _compiled() -> CompiledGraph:
    registry = registry_of(upper_node)
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="u", node="upper", values={"text": "hi"}),
        ]), registry
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


def test_run_sync_returns_the_ending():
    compiled = _compiled()
    ending = compiled.run_sync()

    assert ending.status == "complete"
    assert compiled.results(ending)["u"]["result"] == "HI"


def test_run_sync_inside_a_loop_refuses_naming_await_run():
    async def inside() -> None:
        with pytest.raises(RuntimeError, match=r"await compiled\.run\("):
            _compiled().run_sync()

    asyncio.run(inside())


def test_run_returns_the_same_ending_under_a_loop():
    compiled = _compiled()

    async def inside() -> dict:
        return await compiled.run()

    assert compiled.results(asyncio.run(inside()))["u"]["result"] == "HI"


def _quick_start() -> str:
    """The reference's quick start, as an ``async def`` body: it is written with ``await run``."""
    reference = (files("conductor.about") / "llms.txt").read_text()
    section = reference[reference.index("## Quick Start"):reference.index("## Core Concepts")]
    (block,) = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    return "async def main():\n" + textwrap.indent(block, "    ") + "    return results\n"


def test_the_quick_start_runs_under_asyncio_run():
    namespace: dict = {}
    exec(_quick_start(), namespace)  # noqa: S102

    results = asyncio.run(namespace["main"]())

    assert results["join"]["result"] == "A+B+C"


def test_the_quick_start_runs_under_a_running_loop():
    namespace: dict = {}
    exec(_quick_start(), namespace)  # noqa: S102

    async def notebook() -> dict:
        return await namespace["main"]()

    assert asyncio.run(notebook())["join"]["result"] == "A+B+C"
