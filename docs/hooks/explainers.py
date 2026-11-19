"""The docs build's step for the illustrated pages: the data two of them show, written from the code.

``docs/explainers/problems.html`` shows every kind of problem, and ``docs/explainers/standard-nodes.html`` every node
of ``conductor_nodes`` with one run of it. Neither page keeps that data: each has an empty
``<script type="application/json">`` with an id, and this hook writes the data into it as the site is built. So a
page opened straight from disk shows no catalogue or palette, and says where to find it.

``mkdocs.yml`` names this file under ``hooks``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import linecache
import logging
import os
import re
import sys
import tempfile
import typing
from collections.abc import Callable, Iterator
from inspect import formatannotation
from pathlib import Path
from types import ModuleType
from typing import Any

import conductor_nodes
import polars as pl
from conductor import DECLARATION_PROBLEMS, PROBLEMS, Graph, NodeDeclarationError, ProblemKind
from conductor.about import problems_page
from conductor.testing import REGISTRY_CHECK_PROBLEMS
from mkdocs.structure.files import File, Files

_ROOT = Path(__file__).parents[2]
#: The id a section of the problems page has in the catalogue's data, by the section's title.
_SECTIONS = {
    "Compiling a graph": "compile",
    "Refused calls": "call",
    "Why a run stopped": "run",
    "Declaring a node": "declare",
    "": "check",
}


def on_files(files: Files, config: Any) -> Files:
    """Replaces each illustrated page that has an empty data block by a copy with its data written in."""
    for page in [
        f for f in files if f.src_uri.startswith("explainers/") and f.src_uri.endswith(".html")
    ]:
        text = Path(page.abs_src_path).read_text()
        filled = _filled(text)
        if filled != text:
            files.remove(page)
            files.append(File.generated(config, page.src_uri, content=filled))
    return files


def _filled(text: str) -> str:
    for data_id, make in _DATA.items():
        empty = f'<script type="application/json" id="{data_id}">[]</script>'
        if empty in text:
            data = json.dumps(make(), ensure_ascii=False, separators=(",", ":")).replace(
                "</", "<\\/"
            )
            text = text.replace(
                empty, f'<script type="application/json" id="{data_id}">{data}</script>'
            )
    return text


def _catalogue() -> list[dict[str, Any]]:
    """Every kind of problem, as the problems page writes it, with what its example makes.

    The text, the example and the setup the examples build on are read from the problems page, and each example is
    run as ``tests/test_core/test_problems_page.py`` runs it.
    """
    page = problems_page()
    setup = re.findall(r"```python\n(.*?)```", page, re.S)[0]
    entries = {
        code: body
        for code, body in re.findall(r"^### `(\w+)`\n\n(.*?)(?=^#{2,3} |\Z)", page, re.S | re.M)
    }
    kinds = []
    with _in_an_empty_directory():
        for kind in [*PROBLEMS, *DECLARATION_PROBLEMS, *REGISTRY_CHECK_PROBLEMS]:
            body = entries[kind.code]
            example = re.search(r"```python\n(.*?)```", body, re.S)
            code = example[1].strip("\n") if example else ""
            made, raised = _made(kind, setup, code)
            entry = {
                "section": _SECTIONS[kind.section],
                "code": kind.code,
                "summary": body.split("\n\n")[0].replace("\n", " "),
                "sentence": kind.sentence,
                "ctx": _ctx_keys(kind),
                "example": code,
                "made": [_dumps(json.loads(p.model_dump_json())) for p in made],
            }
            if raised:
                entry["raised"] = raised[:1]
            kinds.append(entry)
    return kinds


def _ctx_keys(kind: ProblemKind[...]) -> list[dict[str, Any]]:
    hints = typing.get_type_hints(kind.ctx_type)
    return [
        {
            "key": key,
            "type": formatannotation(hint),
            "always": key in kind.ctx_type.__required_keys__,
        }
        for key, hint in hints.items()
    ]


def _made(kind: ProblemKind[...], setup: str, example: str) -> tuple[list[Any], list[str]]:
    """Runs a kind's example after the page's setup, and returns the problems it makes and any error it raised."""
    page = ModuleType("problems_page")
    sys.modules[page.__name__] = page
    exec(compile(setup, "problems page: setup", "exec"), page.__dict__)  # noqa: S102  the page's own setup
    raised: list[str] = []
    refused = page.__dict__["declaration_refused"]

    def declaration_refused(declare: Callable[[], object]) -> Any:
        try:
            declare()
        except NodeDeclarationError as error:
            raised.append(str(error))
        return refused(declare)

    page.__dict__["declaration_refused"] = declaration_refused
    where = f"problems page: {kind.code}"
    linecache.cache[where] = (len(example), None, example.splitlines(keepends=True), where)
    *body, last = ast.parse(example).body
    exec(compile(ast.Module(body=body, type_ignores=[]), where, "exec"), page.__dict__)  # noqa: S102
    assert isinstance(last, ast.Expr), kind.code
    made = eval(compile(ast.Expression(last.value), where, "eval"), page.__dict__)  # noqa: S307
    made = made if isinstance(made, list) else [made]
    assert made and all(p.type == kind.code for p in made), (kind.code, made)
    return made, raised


@contextlib.contextmanager
def _in_an_empty_directory() -> Iterator[None]:
    """Runs the examples where they can write files, with the tracebacks some of them log kept out of the build."""
    here = os.getcwd()
    logger = logging.getLogger("conductor")
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        logger.disabled = True
        try:
            yield
        finally:
            logger.disabled = False
            os.chdir(here)


def _dumps(value: Any, indent: int = 0) -> str:
    """JSON with two spaces of indent, and a list of plain values on one line."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{inner}{json.dumps(k, ensure_ascii=False)}: {_dumps(v, indent + 1)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in value) + "]"
        return "[\n" + ",\n".join(inner + _dumps(v, indent + 1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def _palette() -> list[dict[str, Any]]:
    """Every node of ``conductor_nodes``, as ``registry.describe()`` shows it, with one run of it.

    Each node runs alone in a graph, on its first example in ``tests/test_nodes/test_catalog.py`` (``EXAMPLES``),
    the examples ``check_registry`` runs it on there. ``control-switch`` runs on its second, which takes ``other``.
    """
    examples = _catalog_examples()
    defined = {
        value.id: (name, attribute)
        for name in conductor_nodes.__all__
        if name != "registry"
        for attribute, value in vars(getattr(conductor_nodes, name)).items()
        if not attribute.startswith("_")
        and getattr(value, "id", None) in {d.id for d in getattr(conductor_nodes, name).registry}
    }
    palette = []
    for node in json.loads(conductor_nodes.registry.describe().model_dump_json())["nodes"]:
        module, function = defined[node["id"]]
        example = examples[node["id"]][1 if node["id"] == "control-switch" else 0]
        # The node alone, each input a constant. Its outputs are read once it is compiled, since a hook may decide
        # them, and the graph returns each of them.
        alone = {
            "node": node["id"],
            "inputs": {name: {"constant": value} for name, value in example.items()},
        }
        first = Graph.compile(
            {"nodes": {node["id"]: alone}, "outputs": {}}, conductor_nodes.registry
        )
        outputs = {
            o.name: {"node": node["id"], "output": o.name}
            for o in first.describe().nodes[node["id"]].outputs
        }
        graph = Graph.compile(
            {"nodes": {node["id"]: alone}, "outputs": outputs}, conductor_nodes.registry
        )
        assert not graph.problems, (node["id"], graph.problems)
        compiled = graph.describe().nodes[node["id"]]
        run = graph.run({})
        palette.append(
            {
                "id": node["id"],
                "title": node["title"],
                "description": node["description"],
                "category": node["category"],
                "module": module,
                "function": function,
                "inputs": [_field(f) for f in node["inputs"]],
                "outputs": [_field(f) for f in node["outputs"]],
                "added_inputs": node["takes_added_inputs"],
                "computes_outputs": node["computes_outputs"],
                "example": example,
                "compiled_inputs": [
                    _field(i.field.model_dump(), full=False) for i in compiled.inputs
                ],
                "compiled_outputs": [_field(o.model_dump(), full=False) for o in compiled.outputs],
                "status": str(run.status),
                "results": {name: _shown(value) for name, value in run.results[node["id"]].items()},
            }
        )
    return palette


def _catalog_examples() -> dict[str, list[dict[str, Any]]]:
    path = _ROOT / "tests" / "test_nodes" / "test_catalog.py"
    spec = importlib.util.spec_from_file_location("test_catalog", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXAMPLES


def _field(f: dict[str, Any], *, full: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"name": f["name"], "type": f["type"], "series": f["is_series"]}
    if not full:
        return out
    out = {
        "name": f["name"],
        "title": f["title"],
        "type": f["type"],
        "series": f["is_series"],
        "required": f["required"],
    }
    if "default" in f:
        out["default"] = f["default"]
    if f["widget"]:
        out["widget"] = f["widget"]["type"]
    if f["constant_only"]:
        out["constant_only"] = True
    if f["type_variables"]:
        out["tv"] = f["type_variables"]
    return out


def _shown(value: Any) -> Any:
    if isinstance(value, pl.DataFrame):
        return {"table": value.to_dicts()}
    if isinstance(value, list):
        return [_shown(v) for v in value]
    return value


#: The data a page's empty ``<script type="application/json" id="...">`` gets, by its id.
_DATA: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "catalogue-data": _catalogue,
    "palette-data": _palette,
}
