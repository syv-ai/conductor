"""An import-linter contract the built-in ones cannot say: a package is imported only through its ``__init__``.

``pyproject.toml`` declares it, beside the layers contract, and ``tests/test_core/test_layers.py`` runs both.
``from conductor._internal.run import Run`` keeps it; ``from conductor._internal.run.type import Run``, written in
any module outside ``run/``, breaks it.
"""

from __future__ import annotations

import importlib.util
from ast import ImportFrom, parse
from pathlib import Path

from grimp import ImportGraph
from importlinter.application import output
from importlinter.domain.contract import Contract, ContractCheck
from importlinter.domain.fields import ListField, StringField


class ThroughInitContract(Contract):
    """Every module of the root package imports a package of ``container`` only through that package's ``__init__``.

    Attributes:
        container: The package whose packages are imported only through their ``__init__``.
        modules_imported_by_name: Modules a module elsewhere may import as they are, by the convention that writes
            their functions through the module (``catalogue as problem``).
    """

    container = StringField()
    modules_imported_by_name = ListField(subfield=StringField(), required=False)

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        container = str(self.container)
        allowed = {str(name) for name in self.modules_imported_by_name or []}
        depth = container.count(".") + 1
        broken = []
        for importer in sorted(graph.modules):
            importer_parts = importer.split(".")
            for imported in sorted(graph.find_modules_directly_imported_by(importer)):
                parts = imported.split(".")
                if not imported.startswith(container + ".") or len(parts) <= depth + 1 or imported in allowed:
                    continue
                if importer_parts[: depth + 1] == parts[: depth + 1]:  # a module of the same package
                    continue
                lines = [
                    detail["line_number"] for detail in graph.get_import_details(importer=importer, imported=imported)
                    if _names_the_module(detail["line_contents"], imported)
                ]
                if lines:
                    broken.append({"importer": importer, "imported": imported, "lines": lines})
        return ContractCheck(kept=not broken, metadata={"broken": broken})

    def render_broken_contract(self, check: ContractCheck) -> None:
        for found in check.metadata["broken"]:
            package = ".".join(found["imported"].split(".")[: str(self.container).count(".") + 2])
            lines = ", ".join(f"l.{line}" for line in found["lines"])
            output.print_error(f"{found['importer']} imports {found['imported']} ({lines}); import {package} instead.")
            output.new_line()


def _names_the_module(line: str, module: str) -> bool:
    """Whether the import on ``line`` takes ``module`` itself: ``from a.b.c import x`` or ``import a.b.c`` for ``a.b.c``,
    or ``from a.b import c`` when ``a/b/__init__.py`` does not give ``c`` another value. ``from
    conductor._internal.types import host_type`` takes the function ``types/__init__.py`` imports under that name,
    not the module ``host_type.py``. The ``__init__.py`` is read, not run, so a module that cannot be imported
    still gets its contract checked."""
    package, _, name = module.rpartition(".")
    if line.startswith((f"from {module} ", f"import {module}")):
        return True
    return line.startswith(f"from {package} import") and name not in _names_given_by_init(package)


def _names_given_by_init(package: str) -> set[str]:
    """The names a package's ``__init__.py`` imports from its modules: what ``from package import name`` takes."""
    root, *rest = package.split(".")
    spec = importlib.util.find_spec(root)  # a top-level package is found without being run
    assert spec is not None and spec.origin is not None
    init = Path(spec.origin).parent.joinpath(*rest, "__init__.py")
    tree = parse(init.read_text())
    return {alias.asname or alias.name for node in tree.body if isinstance(node, ImportFrom) for alias in node.names}
