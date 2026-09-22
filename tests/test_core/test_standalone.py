"""Conductor stands up with no host installed and holds none of a host's words.

Conductor ships every mechanism — the type machinery, the node contract,
the widget controls, compile, the engine — and a host ships every word:
its types, its nodes, the choices inside a widget, its language. These
tests read the shipped source and `pyproject.toml` of all three packages, so a host's name, a
host's import or a concrete type slipping into the library fails here
rather than in the host that happens to install it.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.ref import Ref

PACKAGES = Path(__file__).resolve().parents[2] / "packages"
SOURCES = sorted(
    p for p in PACKAGES.glob("*/src/**/*") if p.is_file() and p.suffix in {".py", ".txt", ".md"}
) + sorted(PACKAGES.glob("*/pyproject.toml"))

#: What each package may import besides the standard library: itself, what
#: it builds on, and what its ``pyproject.toml`` declares (``pydantic_core``
#: comes with pydantic). The core imports no other package of the three.
IMPORTS = {
    "conductor": {"conductor", "pydantic", "pydantic_core", "yaml"},
    "conductor-nodes": {"conductor_nodes", "conductor", "pydantic", "pydantic_core"},
    "conductor-providers": {"conductor_providers", "conductor", "pydantic", "pydantic_core", "fastapi"},
}

#: What each ``pyproject.toml`` declares, extras included. A dependency added
#: there fails here until it is listed, so a host's package cannot arrive unseen.
DISTRIBUTIONS = {
    "conductor": {"pydantic", "pyyaml", "syv-conductor-nodes", "syv-conductor-providers"},
    "conductor-nodes": {"syv-conductor", "pydantic"},
    "conductor-providers": {"syv-conductor", "fastapi", "pydantic"},
}

#: A line as words: identifiers split at case and underscores, lowercased,
#: so ``FlowStore``, ``run_flow`` and ``flowId`` read "flow store", "run flow"
#: and "flow id". Letters outside ASCII are left out; "Danish text" reads the
#: line itself.
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _words(line: str) -> str:
    return " ".join(word.lower() for word in _WORD.findall(line))


#: A host's words, matched against a line's words: the product's own names,
#: its access model and its domain.
HOST_WORDS = {
    "the host's organisation": r"\b(aka|akademikerne)\b",
    # FastAPI calls its application object ``app``: ``app = FastAPI()``,
    # ``app.include_router(...)``. That is FastAPI's word, not a host's.
    "a host's product noun": r"\bapps?\b(?! fast api\b| include router\b)",
    "a host's access model": (
        r"\b(owner id|org wide|resource shares?|user groups?|runner identity|runner authority"
        r"|compliance officer|user role)\b"
    ),
    "a host's domain": r"\b(risikovurdering\w*|godkend\w*)\b",
    # The library's record is a graph; "flow" and "workflow" are a host's
    # words for one. "data flow" and "control flow" are English, and
    # ReactFlow is the name of the library the React provider speaks.
    "a host's word for a graph": r"(?<!data )(?<!control )(?<!react )\b(work)?flows?\b",
}

#: A host's words that only its spelling gives away, matched against the
#: line itself: its role constants, and the letters and quotes its titles
#: are written in.
HOST_TEXT = {
    "a host's role": r"\b(ADMIN|REVIEWER|DEVELOPER|COMPLIANCE_OFFICER)\b",
    "Danish text": r"[æøåÆØÅ«»]",
}


def _host_words_in(line: str) -> list[str]:
    words = _words(line)
    return [kind for kind, pattern in HOST_WORDS.items() if re.search(pattern, words)] + [
        kind for kind, pattern in HOST_TEXT.items() if re.search(pattern, line)
    ]


def _imports_outside(package: str, source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
    return [
        name for name in names
        if name.split(".")[0] not in IMPORTS[package] and name.split(".")[0] not in sys.stdlib_module_names
    ]


def _declared(pyproject: Path) -> set[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    requirements = project.get("dependencies", []) + [
        requirement for extra in project.get("optional-dependencies", {}).values() for requirement in extra
    ]
    return {re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower() for requirement in requirements}


@pytest.mark.parametrize("package", IMPORTS)
def test_each_package_imports_nothing_but_what_it_builds_on_and_the_standard_library(package):
    outside = [
        f"{source.relative_to(PACKAGES)}: {name}"
        for source in sorted((PACKAGES / package / "src").rglob("*.py"))
        for name in _imports_outside(package, source.read_text(encoding="utf-8"))
    ]

    assert outside == []


@pytest.mark.parametrize("package", DISTRIBUTIONS)
def test_each_package_declares_only_the_dependencies_listed_here(package):
    assert _declared(PACKAGES / package / "pyproject.toml") == DISTRIBUTIONS[package]


@pytest.mark.parametrize(
    ("package", "line"),
    [
        ("conductor", "import fastapi"),
        ("conductor", "from conductor_nodes.types import Text"),
        ("conductor-nodes", "import yaml"),
        ("conductor-nodes", "from conductor_providers.react import graph_to_react"),
        ("conductor-providers", "from app.models import Flow"),
    ],
)
def test_an_import_a_package_does_not_build_on_is_found(package, line):
    assert _imports_outside(package, line) != []


def test_no_host_vocabulary_in_the_library():
    found = [
        f"{source.relative_to(PACKAGES)}:{number}: {kinds}: {line.strip()}"
        for source in SOURCES
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1)
        if (kinds := _host_words_in(line))
    ]

    assert found == []


@pytest.mark.parametrize(
    "line",
    [
        "class FlowStore:", "def run_flow(x):", "workflow_id = 1", "flowId: str", "Flows run here",
        "app_id = 1", "an app runs", "AppVersion", "Apps are published",
        "AKA", "akademikerne", "owner_id", "org_wide", "resource_share", "ResourceShare",
        "user_group", "UserGroup", "RunnerIdentity", "UserRole.ADMIN", "COMPLIANCE_OFFICER",
        "godkendelse", "Risikovurdering", "«Kør herfra»",
    ],
)
def test_a_host_word_is_found_in_any_spelling(line):
    assert _host_words_in(line) != []


@pytest.mark.parametrize(
    "line",
    [
        "data flow and control flow", "ReactFlow positions", "app = FastAPI()",
        "app.include_router(conductor_router(registry))", "append the rows", "Mapping[str, Any]",
        "require_admin", "what a developer wrote", "overflow", "the first group of each",
    ],
)
def test_english_and_other_libraries_words_are_not_a_host_word(line):
    assert _host_words_in(line) == []


def _types_declared_by(*packages: str) -> list[str]:
    """Import every module of ``packages`` in a fresh interpreter and name the ``DType`` classes they define.

    ``__main__`` modules are left out: importing one runs it. Declaring a
    type records nothing anywhere, so the modules themselves are searched.
    """
    probe = (
        "import importlib, pkgutil\n"
        "from conductor.dtype import DType\n"
        "found = set()\n"
        f"for name in {packages!r}:\n"
        "    package = importlib.import_module(name)\n"
        "    for module in pkgutil.walk_packages(package.__path__, name + '.'):\n"
        "        if module.name.endswith('.__main__'):\n"
        "            continue\n"
        "        loaded = importlib.import_module(module.name)\n"
        "        for value in vars(loaded).values():\n"
        "            if isinstance(value, type) and issubclass(value, DType) and value is not DType and value.__module__ == module.name:\n"
        "                found.add(value.id)\n"
        "print(sorted(found))\n"
    )
    printed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    return ast.literal_eval(printed.strip())


def test_the_library_declares_no_type_of_its_own_beyond_series():
    """Which values exist is the host's decision; importing every module of the core and the providers registers only the series."""
    assert _types_declared_by("conductor", "conductor_providers") == ["series"]


def test_a_package_that_declares_types_is_found():
    """The standard nodes ship their own vocabulary, so the same probe over them finds it."""
    assert "text" in _types_declared_by("conductor_nodes")


def test_an_iteration_and_a_reduction_run_on_the_standard_nodes_alone():
    import conductor_nodes
    from conductor import run_sync

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="split", type="text-split", version=1, bindings={"text": Static("a,b,c")}),
            GraphNode(id="upper", type="text-uppercase", version=1, bindings={"text": From(Ref("split", "result"))}),
            GraphNode(id="join", type="text-join", version=1, bindings={"parts": From(Ref("upper", "result")), "separator": Static("+")}),
        ]),
        conductor_nodes.registry(),
    )
    assert compiled.is_runnable, compiled.problems

    results = run_sync(compiled)["results"]

    assert compiled.node("upper").iterates_on is not None
    assert list(results["upper"]["result"]) == ["A", "B", "C"]
    assert results["join"]["result"] == "A+B+C"
