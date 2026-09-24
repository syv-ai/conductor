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

import conductor
import pytest
from conductor import CompiledGraph, GraphNode, NodeRegistry, Param, Result, run, run_sync
from conductor.dtype import DType
from conductor.graph.binding import Static
from conductor.graph.model import Graph
from conductor.node import NodeDefinition
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "run-entry-test-text"
    title = "Text"


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Out")]:
        return Txt(text.upper())


def _compiled() -> CompiledGraph:
    registry = NodeRegistry()
    registry.register(Upper)
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="u", type="upper", version=1, bindings={"text": Static("hi")})]), registry
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


def test_run_sync_returns_the_ending():
    compiled = _compiled()
    ending = run_sync(compiled)

    assert ending.type == "graph_complete"
    assert compiled.results(ending.state)["u"]["result"] == "HI"


def test_run_sync_inside_a_loop_refuses_naming_await_run():
    async def inside() -> None:
        with pytest.raises(RuntimeError, match=r"await conductor\.run\("):
            run_sync(_compiled())

    asyncio.run(inside())


def test_run_returns_the_same_ending_under_a_loop():
    compiled = _compiled()

    async def inside() -> dict:
        return await run(compiled)

    assert compiled.results(asyncio.run(inside()).state)["u"]["result"] == "HI"


def test_the_three_are_root_exports_and_the_old_names_are_gone():
    import conductor.execution.engine as engine

    assert conductor.execute is engine.execute and conductor.run is engine.run and conductor.run_sync is engine.run_sync
    assert not any(name in dir(engine) for name in ("coll" + "ect", "execute_" + "sync"))


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
