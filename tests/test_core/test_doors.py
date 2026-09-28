"""The three public doors: ``conductor``, ``conductor.widgets`` and ``conductor.events``.

Everything a caller outside the library needs is exported from one of
these, and each lists its names in ``__all__``. The lists are pinned here
so that adding or dropping a public name is a visible, reviewed change; a
type checker accepts every name (``mypy --strict`` refuses an import a
package does not export explicitly); and the packages beside the core
library, conductor-nodes and conductor-providers, import from nothing else.
"""

import ast
import importlib
from pathlib import Path

import pytest
from mypy import api as mypy_api

ROOT = Path(__file__).resolve().parents[2]

DOORS: dict[str, list[str]] = {
    "conductor": [
        "ALWAYS", "AnyWidget", "Asks", "Atom", "Binding", "Broadcast", "CompilationError", "CompiledField",
        "CompiledGraph", "CompiledNode", "Condition", "ConductorError", "DType", "Deprecation", "ErrorCause",
        "ExternalFailure", "FieldContent", "From", "FromRun", "Gather", "Graph", "GraphNode", "GraphVersion",
        "Group", "Index", "Input", "InputNotOffered", "Interface", "Iterate", "NEVER", "NodeDefinition",
        "NodeDescription", "NodeError", "NodeExecutionError", "NodeRegistry", "NodeTimeoutError",
        "NodeValidationError", "NodeVersion", "Output", "Param", "Policy", "Problem", "Receive", "Ref", "Refuses",
        "RegistryDescription", "Result", "RunState", "SKIPPED", "Series", "Single", "StartRefused", "Static",
        "TypeDescription", "Whole", "dependencies_of", "deprecated", "execute", "from_wire", "is_asking",
        "is_input_node", "is_skipped", "model_of", "run", "run_sync", "to_wire", "upgrade", "version",
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


def _conductor_imports(source_root: Path) -> list[tuple[str, str]]:
    """Every ``from conductor… import`` in the package under ``source_root``: (file, module)."""
    found = []
    for path in sorted(source_root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "conductor":
                found.append((str(path.relative_to(ROOT)), node.module))
            if isinstance(node, ast.Import):
                found += [(str(path.relative_to(ROOT)), a.name) for a in node.names if a.name.split(".")[0] == "conductor"]
    return found


@pytest.mark.parametrize("package", ["conductor-nodes", "conductor-providers"])
def test_the_packages_beside_the_library_import_only_the_doors(package):
    outside = [(f, m) for f, m in _conductor_imports(ROOT / "packages" / package / "src") if m not in DOORS]
    assert outside == []
