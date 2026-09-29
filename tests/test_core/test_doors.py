"""The three public doors: ``conductor``, ``conductor.widgets`` and ``conductor.events``.

Everything a caller outside the library needs is exported from one of
these, and each lists its names in ``__all__``. The lists are pinned here
so that adding or dropping a public name is a visible, reviewed change; a
type checker accepts every name (``mypy --strict`` and pyright each refuse
an import a package does not export explicitly); and the packages beside
the core library, conductor-nodes and conductor-providers, import from
nothing else — neither a module past a door nor a name a door does not list.
"""

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mypy import api as mypy_api

ROOT = Path(__file__).resolve().parents[2]

DOORS: dict[str, list[str]] = {
    "conductor": [
        "ALWAYS", "Asks", "Atom", "Binding", "Broadcast", "CompilationError", "CompiledField", "CompiledGraph",
        "CompiledNode", "Condition", "ConductorError", "DType", "Deprecation", "DoneUnit",
        "ErrorCause", "ExternalFailure", "FieldContent", "From", "FromRun", "Gather", "Graph", "GraphNode",
        "GraphVersion", "Group", "Index", "Inlined", "Input", "InputNotOffered", "Interface", "Iterate", "NEVER",
        "NodeDefinition", "NodeDescription", "NodeError", "NodeExecutionError", "NodeRegistry", "NodeTimeoutError",
        "NodeValidationError", "NodeVersion", "NotDerived", "NotResolved", "Output", "Param", "Policy", "Problem",
        "Receive", "Ref", "Refuses", "RegistryDescription", "Result", "RunState", "SKIPPED", "Series", "Single",
        "StartRefused", "StateSkip", "StateValue", "Static", "TypeDescription", "Upgrade", "VersionDescription",
        "Whole", "dependencies_of", "deprecated", "execute", "from_wire", "is_asking", "is_input_node", "is_skipped", "model_of",
        "run", "run_sync", "to_wire", "upgrade", "version",
    ],
    "conductor.widgets": [
        "AnyWidget", "Choice", "CodeEditor", "DatePicker", "Dropdown", "EntityDropdown", "FileUpload",
        "IfElseBuilder", "ListWidget", "NumberWidget", "OperatorChoice", "Range", "SchemaBuilder", "Switch",
        "TableInput", "Tags", "TemplateTextarea", "TextWidget", "Textarea",
    ],
    "conductor.events": [
        "Ending", "EndingEvent", "ExecutionEvent", "GraphCancelledEvent", "GraphCompleteEvent", "GraphErrorEvent",
        "GraphPendingEvent", "GraphTimeoutEvent", "NodeCompleteEvent", "NodeErrorEvent", "NodeProgressEvent",
        "NodeRetryEvent", "NodeSkippedEvent", "NodeStartEvent", "PendingUnit",
    ],
}


@pytest.mark.parametrize("door", sorted(DOORS))
def test_a_door_exports_exactly_its_pinned_names(door):
    module = importlib.import_module(door)
    assert sorted(module.__all__) == DOORS[door]
    for name in module.__all__:
        getattr(module, name)


def test_a_type_checker_accepts_every_name_of_every_door(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("".join(f"from {door} import {', '.join(names)}\n" for door, names in DOORS.items()))
    stdout, _, status = mypy_api.run(["--strict", "--no-incremental", "--follow-imports=silent", str(probe)])
    assert "explicitly export" not in stdout, stdout
    assert status == 0, stdout


def test_pyright_accepts_every_name_of_every_door(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("".join(f"from {door} import {', '.join(names)}\n" for door, names in DOORS.items()))
    pyright = Path(sys.executable).with_name("pyright")
    found = subprocess.run(
        [str(pyright), "--outputjson", "--pythonpath", sys.executable, str(probe)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    report = json.loads(found.stdout)
    assert report["generalDiagnostics"] == [], report["generalDiagnostics"]


def _conductor_imports(source_root: Path) -> list[tuple[str, str]]:
    """Every name the package under ``source_root`` imports from ``conductor``: (file, dotted name).

    ``from conductor.events import Ending`` gives ``conductor.events.Ending``
    and ``import conductor.graph`` gives ``conductor.graph``, so a submodule
    reached through a door (``from conductor import graph``) is caught too.
    """
    found = []
    for path in sorted(source_root.rglob("*.py")):
        where = str(path.relative_to(ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "conductor":
                found += [(where, f"{node.module}.{alias.name}") for alias in node.names]
            if isinstance(node, ast.Import):
                found += [(where, alias.name) for alias in node.names if alias.name.split(".")[0] == "conductor"]
    return found


def _through_a_door(name: str) -> bool:
    """``name`` is a door itself, or a name one of the doors lists."""
    module, _, attribute = name.rpartition(".")
    return name in DOORS or attribute in DOORS.get(module, ())


@pytest.mark.parametrize("package", ["conductor-nodes", "conductor-providers"])
def test_the_packages_beside_the_library_import_only_the_doors(package):
    outside = [(f, n) for f, n in _conductor_imports(ROOT / "packages" / package / "src") if not _through_a_door(n)]
    assert outside == []
