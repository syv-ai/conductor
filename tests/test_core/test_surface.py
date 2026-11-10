"""What a caller sees of conductor without running anything in particular.

How the public objects print, what a problem and an event are as records,
which errors a caller catches, that the reference's quick start runs as
written, and that the library stands up with no host installed and holds
none of a host's words.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import re
import sys
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import conductor
import polars as pl
import pytest
from conductor import (
    PROBLEM_CODES,
    SKIPPED,
    Deprecation,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRegistry,
    Policy,
    Problem,
    Table,
    Waiting,
    node,
)
from conductor.events import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    RunEnd,
    RunStart,
    UnitWaiting,
)
from pydantic import TypeAdapter, ValidationError

PACKAGES = Path(__file__).resolve().parents[2] / "packages"
LIBRARY = Path(conductor.__file__).parent


def _registry() -> NodeRegistry:
    registry = NodeRegistry()

    @registry.node("hello")
    def hello(name: str) -> str:
        return f"hello {name}"

    @registry.node("scale")
    def scale(value: float, factor: int = 1) -> float:
        return value * factor

    return registry


# --- how things print -------------------------------------------------------


def test_a_node_prints_what_it_declares():
    @node("greet", title="Greeting", category="text")
    def greet(name: str) -> str:
        return name

    assert repr(greet) == "NodeDefinition(id='greet', version=1, title='Greeting', category='text')"


def test_a_node_prints_its_tags_only_when_it_has_them():
    @node("old", title="Old", tags=("legacy",))
    def old(name: str) -> str:
        return name

    @node("plain", title="Plain")
    def plain(name: str) -> str:
        return name

    assert repr(old) == "NodeDefinition(id='old', version=1, title='Old', tags=('legacy',))"
    assert "tags" not in repr(plain)


def test_a_registry_prints_its_nodes_each_as_its_own_repr():
    registry = NodeRegistry()
    assert repr(registry) == "<NodeRegistry of 0 nodes>"

    @registry.node("greet", title="Greeting")
    def greet(name: str) -> str:
        return name

    @registry.node("shout", title="Shout")
    def shout(name: str) -> str:
        return name

    assert repr(registry) == f"<NodeRegistry of 2 nodes:\n    {greet!r}\n    {shout!r}\n>"


def test_a_loaded_graph_prints_a_one_line_summary():
    registry = _registry()

    assert repr(registry.load({"nodes": [{"id": "s", "node": "scale"}]})) == "<Graph of 1 node, runnable>"
    assert repr(registry.load({"nodes": [{"id": "s", "node": "nope"}]})) == "<Graph of 1 node, 1 problem>"


def test_a_record_prints_only_what_is_not_at_its_default():
    assert repr(Policy()) == "Policy()"
    assert repr(Policy(retries=2)) == "Policy(retries=2)"
    assert repr(Deprecation()) == "Deprecation()"
    assert repr(Deprecation(header="Gone")) == "Deprecation(header='Gone')"
    assert repr(Problem(type="cycle", msg="m", loc=("a",))) == "Problem(type='cycle', loc=('a',), msg='m')"


def test_a_record_prints_a_field_set_back_to_its_default_value_as_absent():
    """The repr is the data, not the history: a value equal to the default is not news."""
    assert repr(Policy(retries=0, concurrency=1)) == "Policy()"
    assert repr(Deprecation(header=None)) == "Deprecation()"
    assert repr(Problem(type="cycle", msg="m", ctx={})) == "Problem(type='cycle', msg='m')"


def test_skipped_is_one_object_that_prints_as_its_name():
    assert type(SKIPPED)() is SKIPPED
    assert repr(SKIPPED) == "SKIPPED"


def test_skipped_is_falsy():
    assert not SKIPPED


# --- a problem as a record --------------------------------------------------


def test_a_problem_dumps_to_json_and_reads_back():
    graph = _registry().load({"nodes": [{"id": "s", "node": "scale", "values": {"value": 1.0, "facter": 3}}]})
    (told,) = graph.problems

    dumped = json.loads(told.model_dump_json())

    assert dumped == {"type": "unknown_input", "loc": ["s", "facter"], "msg": told.msg, "ctx": {"inputs": "value, factor"}}
    assert Problem.model_validate(dumped) == told


def test_a_problem_is_frozen():
    problem = Problem(type="c", msg="m", loc=("n",))

    with pytest.raises(ValidationError):
        problem.type = "other"  # type: ignore[misc]


def test_problems_compare_by_value():
    assert Problem(type="c", msg="m", loc=("n",)) == Problem(type="c", msg="m", loc=("n",))
    assert Problem(type="c", msg="m", loc=("n",)) != Problem(type="c", msg="m", loc=("o",))


def test_a_problems_message_is_its_sentence_filled_from_the_ctx_it_keeps():
    graph = _registry().load({
        "nodes": [{"id": "a", "node": "hello", "values": {"name": "x"}}, {"id": "b", "node": "scale", "values": {"facter": 2}}],
        "edges": [["a.reslt", "b.value"]],
    })

    assert {problem.type for problem in graph.problems} == {"unknown_output", "unknown_input"}
    for problem in graph.problems:
        assert "{" not in problem.msg
        assert all(value in problem.msg for value in problem.ctx.values() if isinstance(value, str))


def test_an_edge_to_a_missing_output_says_what_the_node_has():
    graph = _registry().load({
        "nodes": [{"id": "a", "node": "hello", "values": {"name": "x"}}, {"id": "b", "node": "scale"}],
        "edges": [["a.reslt", "b.value"]],
    })

    (found,) = graph.problems
    assert found.type == "unknown_output" and found.ctx["outputs"] == "result" and "it has result" in found.msg


def test_an_unknown_input_names_the_inputs_the_node_has():
    graph = _registry().load({"nodes": [{"id": "s", "node": "scale", "values": {"value": 1.0, "facter": 3}}]})

    (found,) = graph.problems
    assert found.msg == "'s' has no input 'facter'; it has value, factor."


def test_a_bad_typed_in_value_names_the_field_and_the_type_but_never_the_value():
    """A host streams problems to a browser, so what a person typed stays out of them."""
    graph = _registry().load({"nodes": [{"id": "s", "node": "scale", "values": {"value": 1.0, "factor": "seventy"}}]})

    (found,) = graph.problems
    assert (found.type, found.loc) == ("invalid_value", ("s", "factor"))
    assert "integer" in found.msg
    assert "seventy" not in found.msg and "seventy" not in json.dumps(found.ctx)


def test_a_refused_run_lists_each_problem_with_its_place_and_code():
    graph = _registry().load({"nodes": [{"id": "s", "node": "scale", "values": {"value": 1.0, "facter": 3}}]})

    with pytest.raises(InvalidCallError) as raised:
        graph.run()

    assert str(raised.value) == "1 problem(s):\n  s.facter — unknown_input: 's' has no input 'facter'; it has value, factor."


def test_the_outputs_of_a_failed_run_are_refused_and_its_results_say_what_it_did():
    registry = NodeRegistry()

    @registry.node("each")
    def each(n: int) -> int:
        if n == 1:
            raise ValueError("no")
        return n

    run = registry.load({"nodes": [{"id": "e", "node": "each", "many": ["n"]}]}).run(n=[0, 1, 2])

    assert run.status == "failed" and run.cause is not None and run.cause.loc == ("e", 1)
    with pytest.raises(InvalidCallError) as raised:
        run.outputs
    (refused,) = raised.value.errors()
    assert refused.type == "run_not_complete" and refused.ctx["status"] == "failed"
    assert run.results["e"]["result"][0] == 0


def _codes_written_by_the_library() -> set[str]:
    """Every problem code a site in the library writes: ``make_problem``'s or ``NotDeclared``'s first argument,
    ``Problem(type=...)``, a ``code=`` on any call but a declaration error's, and each error's code when none is given."""
    written: set[str] = set()

    def literals(expression: ast.expr) -> set[str]:
        return {found.value for found in ast.walk(expression) if isinstance(found, ast.Constant) and isinstance(found.value, str)}

    for path in LIBRARY.rglob("*.py"):
        for found in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(found, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_code_when_none_given" for t in found.targets):
                written |= literals(found.value)
            if not isinstance(found, ast.Call):
                continue
            called = found.func.id if isinstance(found.func, ast.Name) else None
            if called in ("make_problem", "NotDeclared") and found.args:
                written |= literals(found.args[0])
            if called != "NodeDeclarationError":
                written |= {code for keyword in found.keywords if keyword.arg == "code" for code in literals(keyword.value)}
            if called == "Problem":
                written |= {code for keyword in found.keywords if keyword.arg == "type" for code in literals(keyword.value)}
    return written


def test_every_code_the_library_writes_is_a_problem_code_and_every_problem_code_is_written():
    """``PROBLEM_CODES`` is what a host translating by code covers: no code goes out unlisted, and no row is dead."""
    assert _codes_written_by_the_library() == PROBLEM_CODES


# --- errors a caller catches ------------------------------------------------


def test_the_errors_a_caller_catches_share_one_base():
    (base,) = set(NodeDeclarationError.__mro__) & set(InvalidCallError.__mro__) & set(NodeError.__mro__) - {Exception, BaseException, object}

    assert base.__name__ == "ConductorError"
    assert issubclass(InvalidCallError, ValueError)
    assert issubclass(NodeDeclarationError, TypeError)


def test_an_external_failure_is_a_node_error_and_there_is_no_retry_flag():
    """The family is the class: an ``ExternalFailure`` is retried, nothing else is."""
    assert issubclass(ExternalFailure, NodeError)
    assert not hasattr(NodeError("x"), "retryable")


# --- events as records ------------------------------------------------------


#: Every event's fields, all required in the schema a generated client reads, by its ``type``.
SHAPES = {
    "run_start": ["type", "waiting"],
    "node_start": ["node_id", "type"],
    "node_progress": ["done", "node_id", "total", "type"],
    "node_complete": ["node_id", "not_called", "outputs", "reused_units", "type"],
    "node_skipped": ["node_id", "type"],
    "node_failed": ["cause", "node_id", "type"],
    "node_retry": ["attempt", "cause", "delay", "node_id", "retries", "row", "type"],
    "unit_waiting": ["type", "unit"],
    "run_end": ["status", "type"],
}


def test_every_event_names_its_fields_and_requires_them_in_the_schema():
    definitions = TypeAdapter(Event).json_schema(mode="serialization")["$defs"]
    shapes = {
        schema["properties"]["type"]["const"]: (sorted(schema["properties"]), sorted(schema.get("required", [])))
        for schema in definitions.values()
        if "const" in schema.get("properties", {}).get("type", {})
    }

    assert shapes == {event_type: (fields, fields) for event_type, fields in SHAPES.items()}


def test_an_event_is_frozen_and_refuses_a_key_it_does_not_have():
    event = NodeStart(node_id="a")

    with pytest.raises(ValidationError):
        event.node_id = "b"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        NodeStart(node_id="a", row=None)  # type: ignore[call-arg]


def test_the_union_reads_an_event_back_by_its_type():
    progress = TypeAdapter(Event).validate_python({"type": "node_progress", "node_id": "a", "done": 1, "total": None})

    assert progress == NodeProgress(node_id="a", done=1, total=None)


def _strict(document: str | bytes) -> Any:
    def refuse(token: str) -> None:
        raise AssertionError(f"{token} is not JSON")

    return json.loads(document, parse_constant=refuse)


def _streamed_events() -> list[Any]:
    """The events of a real run whose outputs are a frame holding NaN and a float that is not finite."""
    registry = NodeRegistry()

    @registry.node("frame")
    def frame(x: float) -> Table:
        return pl.DataFrame({"a": [x, 1.0], "b": ["x", None]})

    @registry.node("same")
    def same(x: float) -> float:
        return x

    graph = registry.load({"nodes": [{"id": "f", "node": "frame"}, {"id": "s", "node": "same"}]})

    async def stream() -> list[Any]:
        return [event async for event in graph.with_inputs({"f.x": math.nan, "s.x": math.inf}).stream()]

    return asyncio.run(stream())


def _every_event() -> list[Any]:
    streamed = _streamed_events()
    waiting = Waiting(id="n", node_id="n", row=(), prompt="Q?", questions=[{"name": "result", "default": 1.5}], inputs={"x": "a"})
    cause = Problem(type="external_failed", msg="e", loc=("n",), ctx={"error": "OSError"})
    built = [
        RunStart(waiting=(waiting,)),
        NodeProgress(node_id="n", done=1, total=None),
        NodeSkipped(node_id="n"),
        NodeFailed(node_id="n", cause=cause),
        NodeRetry(node_id="n", row=(), attempt=1, retries=2, delay=0.5, cause=cause),
        UnitWaiting(unit=waiting),
    ]
    return [*streamed, *built]


def test_a_run_writes_frames_and_floats_that_are_not_finite_as_strict_json():
    by_node = {event.node_id: event for event in _streamed_events() if isinstance(event, NodeComplete)}

    assert _strict(by_node["f"].model_dump_json())["outputs"] == {"result": {"a": ["NaN", 1.0], "b": ["x", None]}}
    assert _strict(by_node["s"].model_dump_json())["outputs"] == {"result": "Infinity"}


@pytest.mark.parametrize("event", _every_event(), ids=lambda event: event.type)
def test_every_event_writes_itself_as_json_and_a_strict_parser_reads_it(event):
    document = _strict(event.model_dump_json())

    assert document == _strict(TypeAdapter(Event).dump_json(event))
    assert document == _strict(json.dumps(event.model_dump(mode="json"), allow_nan=False))
    if not isinstance(event, RunEnd):
        assert TypeAdapter(Event).validate_json(event.model_dump_json()).type == event.type


def test_every_event_type_is_covered_above():
    assert {event.type for event in _every_event()} == set(SHAPES)


# --- the reference's quick start ---------------------------------------------


def _quick_start() -> str:
    reference = (files("conductor.about") / "llms.txt").read_text()
    section = reference[reference.index("## Quick Start"):reference.index("## Core Concepts")]
    (block,) = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    return block


def _run_quick_start() -> Any:
    namespace: dict[str, Any] = {name: getattr(conductor, name) for name in conductor.__all__}
    exec(_quick_start(), namespace)  # noqa: S102
    return namespace["run"].outputs


def test_the_quick_start_runs_as_written():
    assert _run_quick_start() == {"join.result": "A+B+C"}


def test_the_quick_start_runs_inside_a_running_loop_as_in_a_notebook():
    async def notebook() -> Any:
        return _run_quick_start()

    assert asyncio.run(notebook()) == {"join.result": "A+B+C"}


# --- the library stands alone ------------------------------------------------


#: What each package may import besides the standard library (``pydantic_core`` comes with pydantic).
IMPORTS = {"conductor": {"conductor", "pydantic", "pydantic_core", "polars"}}

#: What each ``pyproject.toml`` declares, extras included, so a host's package cannot arrive unseen.
DISTRIBUTIONS = {"conductor": {"pydantic", "polars"}}

SOURCES = sorted(p for p in PACKAGES.glob("*/src/**/*") if p.is_file() and p.suffix in {".py", ".pyi", ".txt", ".md"}) + sorted(
    PACKAGES.glob("*/pyproject.toml")
)

#: A line as words: identifiers split at case and underscores, lowercased, so
#: ``FlowStore``, ``run_flow`` and ``flowId`` read "flow store", "run flow" and "flow id".
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

#: A host's words, matched against a line's words: its own names, its access model and its domain.
HOST_WORDS = {
    "the host's organisation": r"\b(aka|akademikerne)\b",
    # FastAPI's application object is ``app``: ``app = FastAPI()``, ``app.include_router(...)``.
    "a host's product noun": r"\bapps?\b(?! fast api\b| include router\b)",
    "a host's access model": (
        r"\b(owner id|org wide|resource shares?|user groups?|runner identity|runner authority|compliance officer|user role)\b"
    ),
    "a host's domain": r"\b(risikovurdering\w*|godkend\w*)\b",
    # The library's record is a graph; "data flow" and "control flow" are English.
    "a host's word for a graph": r"(?<!data )(?<!control )\b(work)?flows?\b",
}

#: A host's words that only their spelling gives away: role constants and the letters its titles use.
HOST_TEXT = {
    "a host's role": r"\b(ADMIN|REVIEWER|DEVELOPER|COMPLIANCE_OFFICER)\b",
    "Danish text": r"[æøåÆØÅ«»]",
}


def _host_words_in(line: str) -> list[str]:
    words = " ".join(word.lower() for word in _WORD.findall(line))
    return [kind for kind, pattern in HOST_WORDS.items() if re.search(pattern, words)] + [
        kind for kind, pattern in HOST_TEXT.items() if re.search(pattern, line)
    ]


def _imports_outside(package: str, source: str) -> list[str]:
    names: list[str] = []
    for found in ast.walk(ast.parse(source)):
        if isinstance(found, ast.Import):
            names += [alias.name for alias in found.names]
        elif isinstance(found, ast.ImportFrom) and found.level == 0 and found.module is not None:
            names.append(found.module)
    return [name for name in names if name.split(".")[0] not in IMPORTS[package] | sys.stdlib_module_names]


def _declared(pyproject: Path) -> set[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    requirements = project.get("dependencies", []) + [r for extra in project.get("optional-dependencies", {}).values() for r in extra]
    return {re.split(r"[^A-Za-z0-9_.-]", requirement, maxsplit=1)[0].lower() for requirement in requirements}


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


@pytest.mark.parametrize("line", ["import fastapi", "import yaml", "from app.models import Flow"])
def test_an_import_a_package_does_not_build_on_is_found(line):
    assert _imports_outside("conductor", line) != []


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
        "data flow and control flow", "app = FastAPI()", "app.include_router(router)", "append the rows",
        "Mapping[str, Any]", "require_admin", "what a developer wrote", "overflow", "the first group of each",
    ],
)
def test_english_and_other_libraries_words_are_not_a_host_word(line):
    assert _host_words_in(line) == []
