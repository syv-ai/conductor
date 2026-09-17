"""Conductor stands up with no host installed and holds none of a host's words.

Conductor ships every mechanism — the type machinery, the node contract,
the widget controls, compile, the engine — and a host ships every word:
its types, its nodes, the choices inside a widget, its language. These
tests read the shipped source of all three packages, so a host's name, a
host's import or a concrete type slipping into the library fails here
rather than in the host that happens to install it.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.ref import Ref

PACKAGES = Path(__file__).resolve().parents[2] / "packages"
SOURCES = sorted(p for p in PACKAGES.glob("*/src/**/*") if p.is_file() and p.suffix in {".py", ".txt", ".md"})

#: What the library may import besides the standard library: itself, and
#: what its three packages declare as dependencies (PyYAML and FastAPI as extras).
DEPENDENCIES = {"conductor", "conductor_nodes", "conductor_providers", "pydantic", "pydantic_core", "yaml", "fastapi"}

#: A host's words: the product's own names, its access model and its
#: language. Danish letters and «» are how a host's titles are written, and
#: none of them is the library's to ship.
HOST_WORDS = {
    "the host's organisation": r"\b(AKA|Akademikerne)\b",
    "a host's product noun": r"\bApps?\b",
    "a host's access model": r"\b(owner_id|org_wide|ResourceShare|UserGroup|RunnerIdentity|RunnerAuthority|COMPLIANCE_OFFICER)\b",
    "a host's domain": r"(?i)\b(risikovurdering\w*|godkend\w*)\b",
    "Danish text": r"[æøåÆØÅ«»]",
    # The library's record is a graph; "flow" is a host's word for one.
    # "data flow" and "control flow" are English, not a host's.
    "a host's word for a graph": r"(?i)(?<!data )(?<!control )\bflows?\b",
}


def test_the_library_imports_nothing_but_itself_its_dependencies_and_the_standard_library():
    outside: list[str] = []
    for source in (p for p in SOURCES if p.suffix == ".py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in DEPENDENCIES and top not in sys.stdlib_module_names:
                    outside.append(f"{source.relative_to(PACKAGES)}: {name}")

    assert outside == []


@pytest.mark.parametrize("kind", HOST_WORDS)
def test_no_host_vocabulary_in_the_library(kind):
    pattern = re.compile(HOST_WORDS[kind])
    found = [
        f"{source.relative_to(PACKAGES)}:{number}: {line.strip()}"
        for source in SOURCES
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]

    assert found == []


def test_the_library_declares_no_type_of_its_own_beyond_series():
    """Which values exist is the host's decision; importing conductor alone registers only the series."""
    probe = "import conductor, conductor.graph.compiled, conductor.execution.engine; from conductor.dtype import registered_dtypes; print(sorted(t.id for t in registered_dtypes()))"

    printed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout

    assert printed.strip() == "['series']"


def test_an_iteration_and_a_reduction_run_on_the_standard_nodes_alone():
    from conductor.execution.engine import execute_sync
    from conductor_nodes import get_default_registry

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="split", type="text-split", version=1, bindings={"text": Static(value="a,b,c")}),
            GraphNode(id="upper", type="text-uppercase", version=1, bindings={"text": Edges(refs=(Ref("split", "result"),))}),
            GraphNode(id="join", type="text-join", version=1, bindings={"parts": Edges(refs=(Ref("upper", "result"),)), "separator": Static(value="+")}),
        ]),
        get_default_registry(),
    )
    assert compiled.is_runnable, compiled.problems

    results = execute_sync(compiled)

    assert compiled.node("upper").iterates_on is not None
    assert list(results["upper"]["result"]) == ["A", "B", "C"]
    assert results["join"]["result"] == "A+B+C"
