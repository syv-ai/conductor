"""The docs build's step for the illustrated pages: the engine output some of them show, written from the code.

``problems.html`` shows every kind of problem, ``standard-nodes.html`` every node of ``conductor_nodes`` with one run
of it, ``graph-record-and-compiling.html`` what compiling says of each broken record on its bench, and
``series-and-tables.html`` each state of its loop, collecting and item-limit widgets. None of the pages keeps that
data: each has an empty ``<script type="application/json">`` with an id, and this hook runs the code and writes what
it gives into it as the site is built (``_DATA``). So a page opened straight from disk shows no such data, and says
where to find it.

``mkdocs.yml`` names this file under ``hooks``.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
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
from conductor import (
    DECLARATION_PROBLEMS,
    PROBLEMS,
    Graph,
    InvalidCallError,
    NodeRegistry,
    ProblemKind,
    Series,
    make_graph,
)
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
    def written(match: re.Match[str]) -> str:
        data = json.dumps(_DATA[match[1]](), ensure_ascii=False, separators=(",", ":"))
        return (
            f'<script type="application/json" id="{match[1]}">{data.replace("</", "<\\/")}</script>'
        )

    ids = "|".join(map(re.escape, _DATA))
    return re.sub(
        rf'<script type="application/json" id="({ids})">(?:\[\]|\{{\}})</script>', written, text
    )


def _catalogue() -> list[dict[str, Any]]:
    """Every kind of problem, as the problems page writes it, with what its example makes.

    Each example is run by ``run_problem_example`` in ``tests/test_core/doc_examples.py``, the runner
    ``tests/test_core/test_problems_page.py`` checks the examples with, so the page and the test run them alike.
    """
    examples = _load("tests/test_core/doc_examples.py")
    summaries = {
        code: body.split("\n\n")[0].replace("\n", " ")
        for code, body in re.findall(
            r"^### `(\w+)`\n\n(.*?)(?=^#{2,3} |\Z)", problems_page(), re.S | re.M
        )
    }
    kinds = []
    with _in_an_empty_directory():
        for kind in [*PROBLEMS, *DECLARATION_PROBLEMS, *REGISTRY_CHECK_PROBLEMS]:
            ran = examples.run_problem_example(kind.code)
            assert ran.made and all(p.type == kind.code for p in ran.made), (kind.code, ran.made)
            entry = {
                "section": _SECTIONS[kind.section],
                "code": kind.code,
                "summary": summaries[kind.code],
                "sentence": kind.sentence,
                "ctx": _ctx_keys(kind),
                "example": ran.example.strip("\n"),
                "made": [_dumps(json.loads(p.model_dump_json())) for p in ran.made],
            }
            if ran.raised:
                entry["raised"] = ran.raised[:1]
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
        assert run.status == "complete", (node["id"], run.status, run.cause)
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
                "status": run.status,
                "results": {name: _shown(value) for name, value in run.results[node["id"]].items()},
            }
        )
    return palette


def _catalog_examples() -> dict[str, list[dict[str, Any]]]:
    return _load("tests/test_nodes/test_catalog.py").EXAMPLES


def _load(path: str) -> ModuleType:
    """Imports a file of the repo by its path, since the docs build doesn't put ``tests`` on the import path."""
    name = "explainers_" + Path(path).stem
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass in the file looks up its module by name
    spec.loader.exec_module(module)
    return module


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


# The break-the-record bench of graph-record-and-compiling.html.


def _bench_registry() -> NodeRegistry:
    """Returns the registry the bench compiles against: lines, shout and join from the page, and count."""
    registry = NodeRegistry()

    @registry.node("lines", title="Lines", description="Splits a text into lines.")
    def lines(text: str) -> Series[str]:
        return text.split("\n")

    @registry.node("shout", title="Shout", description="Shouts one line.")
    def shout(line: str, mark: str = "!") -> str:
        return line.upper() + mark

    @registry.node("join", title="Join", description="Joins texts with a plus.")
    def join(texts: Series[str]) -> str:
        return "+".join(texts)

    @registry.node("count", title="Count", description="Counts the characters of a text.")
    def count(text: str) -> int:
        return len(text)

    return registry


def _bench_record() -> dict[str, Any]:
    """Returns the page's example record, unbroken."""
    return {
        "nodes": {
            "lines": {"node": "lines", "inputs": {"text": {"graph_input": "text"}}},
            "shout": {
                "node": "shout",
                "inputs": {
                    "line": {"sources": [{"node": "lines", "output": "result"}]},
                    "mark": {"constant": "?"},
                },
            },
            "join": {
                "node": "join",
                "inputs": {"texts": {"sources": [{"node": "shout", "output": "result"}]}},
            },
        },
        "outputs": {"result": {"node": "join", "output": "result"}},
    }


def _bench_set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    """Returns a change that sets the value at ``path`` in a record, keeping the place of a key already there."""

    def change(record: dict[str, Any]) -> None:
        *parents, last = path
        place = record
        for key in parents:
            place = place[key]
        place[last] = value

    return change


def _bench_two_edges(record: dict[str, Any]) -> None:
    """Adds a node ``more`` and a second edge from it into ``shout.line``."""
    record["nodes"]["shout"]["inputs"]["line"]["sources"].append(
        {"node": "more", "output": "result"}
    )
    record["nodes"]["more"] = {"node": "lines", "inputs": {"text": {"graph_input": "text"}}}


_BENCH_SHOUT = ("nodes", "shout")
_BENCH_CASES: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
    ("base", "Nothing broken", lambda record: None),
    ("unknown_node", "Misspell a node id", _bench_set((*_BENCH_SHOUT, "node"), "shuot")),
    (
        "unknown_version",
        "Ask for a version that doesn't exist",
        _bench_set((*_BENCH_SHOUT, "version"), 2),
    ),
    (
        "edge_names_missing_node",
        "Wire from a node that isn't there",
        _bench_set(
            (*_BENCH_SHOUT, "inputs", "line"), {"sources": [{"node": "liens", "output": "result"}]}
        ),
    ),
    (
        "unknown_output",
        "Wire from an output that isn't there",
        _bench_set(
            (*_BENCH_SHOUT, "inputs", "line"), {"sources": [{"node": "lines", "output": "results"}]}
        ),
    ),
    ("type_mismatch", "Wire an int into a str", _bench_set(("nodes", "lines", "node"), "count")),
    (
        "invalid_value",
        "Give a constant of the wrong type",
        _bench_set((*_BENCH_SHOUT, "inputs", "mark"), {"constant": 3}),
    ),
    (
        "unknown_input",
        "Give a constant to an input the node lacks",
        _bench_set((*_BENCH_SHOUT, "inputs", "colour"), {"constant": "red"}),
    ),
    (
        "input_has_no_value",
        "Leave a required input empty",
        _bench_set(("nodes", "lines"), {"node": "lines"}),
    ),
    ("several_edges_into_one_input", "Wire two edges into one input", _bench_two_edges),
    (
        "cycle",
        "Close a loop of edges",
        _bench_set(
            ("nodes", "lines", "inputs", "text"),
            {"sources": [{"node": "join", "output": "result"}]},
        ),
    ),
    (
        "graph_output_not_found",
        "Declare a graph output that isn't there",
        _bench_set(("outputs", "result"), {"node": "join", "output": "joined"}),
    ),
    (
        "one_of_constant_graph_input_sources",
        "Give an input two things at once",
        _bench_set((*_BENCH_SHOUT, "inputs", "mark"), {"constant": "?", "graph_input": "mark"}),
    ),
    (
        "not_a_name",
        "Put a dot in a graph input's name",
        _bench_set(("nodes", "lines", "inputs", "text"), {"graph_input": "my.text"}),
    ),
]


def _breaks() -> list[dict[str, Any]]:
    """Compiles each broken record of the page's bench and returns what the bench shows of it.

    Each case is the page's example record with one thing changed. A record ``Graph.compile`` takes gives the graph's
    repr, problems and, per node, what ``describe()`` says of it; a record it refuses gives the problems it raised.

    Returns:
        One dict per case, in the bench's order, with ``key``, ``label`` and ``record``, and either ``runnable``,
        ``repr``, ``problems`` and ``nodes``, or ``raised``.
    """
    registry = _bench_registry()
    found = []
    for key, label, change in _BENCH_CASES:
        record = _bench_record()
        change(record)
        entry: dict[str, Any] = {"key": key, "label": label, "record": copy.deepcopy(record)}
        try:
            graph = Graph.compile(record, registry)
        except InvalidCallError as error:
            entry["raised"] = [
                {
                    "type": p.type,
                    "loc": list(p.loc),
                    "msg": p.msg,
                    "error": p.ctx["errors"][0]["type"],
                }
                for p in error.problems
            ]
            assert [fault["error"] for fault in entry["raised"]] == [key], (key, entry["raised"])
        else:
            entry["runnable"] = not graph.problems
            entry["repr"] = repr(graph)
            entry["problems"] = [
                {"type": p.type, "loc": list(p.loc), "msg": p.msg} for p in graph.problems
            ]
            entry["nodes"] = {
                node_id: {
                    "found": described.found,
                    "per_item_of": described.per_item_of,
                    "inputs": {i.field.name: i.source.mode for i in described.inputs},
                }
                for node_id, described in graph.describe().nodes.items()
            }
            types = [p["type"] for p in entry["problems"]]
            assert (key == "base") == (not types) and (key == "base" or key in types), (key, types)
        found.append(entry)
    return found


# The loop, collecting and item-limit widgets of series-and-tables.html.


def _widget_data() -> dict[str, Any]:
    """Returns the states the three widgets of the page on series and tables show, from running each graph.

    Returns:
        A dict with ``loop`` (the loop widget, by ``"collect/<n>"`` or ``"per-item/<n>"``), ``collect`` (the
        collecting widget, by which of three names have no email, as ``"010"``) and ``limit`` (the item-limit
        widget, by ``"<text>/<max_items>"``).
    """
    return {"loop": _loop_states(), "collect": _collect_states(), "limit": _limit_states()}


def _loop_states() -> dict[str, Any]:
    registry = NodeRegistry()

    @registry.node("lines", title="Lines", description="Splits a text into lines.")
    def lines(text: str) -> Series[str]:
        return [] if text == "" else text.split("\n")

    @registry.node("shout", title="Shout", description="Writes a line in capitals.")
    def shout(line: str, mark: str = "!") -> str:
        return line.upper() + mark

    @registry.node("join", title="Join", description="Joins texts into one.")
    def join(texts: Series[str]) -> str:
        return " ".join(texts)

    @registry.node("join-one", title="Join one", description="Wraps one text.")
    def join_one(texts: str) -> str:
        return f"<{texts}>"

    states: dict[str, Any] = {}
    for kind, last in (("collect", join), ("per-item", join_one)):
        graph = Graph.compile(
            make_graph(
                lines,
                shout,
                last,
                edges=[("lines.result", "shout.line"), ("shout.result", f"{last.id}.texts")],
            ),
            registry,
        )
        assert not graph.problems, graph.problems
        per_item_of = {
            node_id: node.per_item_of for node_id, node in graph.describe().nodes.items()
        }
        for n in range(6):
            run = graph.run({"text": "\n".join("abcde"[:n])})
            assert run.status == "complete", run.cause
            calls = {
                node_id: run.progress.nodes[node_id].finished
                for node_id in ("lines", "shout", last.id)
            }
            assert calls["shout"] == n
            states[f"{kind}/{n}"] = {
                "per_item_of": per_item_of,
                "shout": run.results["shout"]["result"],
                "join": run.results[last.id]["result"],
                "calls": calls,
            }
    return states


def _collect_states() -> dict[str, Any]:
    registry = NodeRegistry()
    names_without_email: set[str] = set()

    @registry.node("names", title="Names", description="Splits a text into names.")
    def names(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("email", title="Email", description="Finds a name's email, if there is one.")
    def email(name: str) -> str | None:
        return None if name in names_without_email else f"{name.lower()}@a.dk"

    @registry.node("pair", title="Pair", description="Pairs names with emails.")
    def pair(names: Series[str], emails: Series[str]) -> list[str]:
        return [f"{n} <{e}>" for n, e in zip(names, emails, strict=True)]

    @registry.node("keep", title="Keep", description="Keeps the gaps.")
    def keep(emails: Series[str | None]) -> list[str | None]:
        return emails

    graph = Graph.compile(
        make_graph(
            names,
            email,
            pair,
            keep,
            edges=[
                ("names.result", "email.name"),
                ("names.result", "pair.names"),
                ("email.result", "pair.emails"),
                ("email.result", "keep.emails"),
            ],
            outputs={"pair": "pair.result", "keep": "keep.result"},
        ),
        registry,
    )
    assert not graph.problems, graph.problems
    every_name = ["Ida", "Bo", "Al"]
    states: dict[str, Any] = {}
    for mask in range(8):
        key = format(mask, "03b")
        names_without_email.clear()
        names_without_email.update(
            name for name, off in zip(every_name, key, strict=True) if off == "1"
        )
        run = graph.run({"text": ",".join(every_name)})
        assert run.status == "complete", run.cause
        assert len(run.outputs["pair"]) == key.count("0")
        states[key] = {
            "status": run.status,
            "keep": run.outputs["keep"],
            "pair": run.outputs["pair"],
            "none_input_calls": [list(call.loc) for call in run.none_input_calls],
        }
    return states


def _limit_states() -> dict[str, Any]:
    registry = NodeRegistry()
    shouted: list[str] = []

    @registry.node("docs", title="Documents", description="Splits a text into documents.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    @registry.node("lines", title="Lines", description="Splits a document into lines.")
    def lines(doc: str) -> Series[str]:
        return doc.split(",")

    @registry.node("shout", title="Shout", description="Writes a line in capitals.")
    def shout(line: str) -> str:
        shouted.append(line)
        return line.upper()

    graph = Graph.compile(
        make_graph(
            docs, lines, shout, edges=[("docs.result", "lines.doc"), ("lines.result", "shout.line")]
        ),
        registry,
    )
    assert not graph.problems, graph.problems
    states: dict[str, Any] = {}
    for text in ("a,b|c,d|e,f", "a,b|c,d,e,f|g"):
        for max_items in range(1, 6):
            shouted.clear()
            run = graph.run({"text": text}, max_items=max_items)
            cause = run.cause
            if run.status == "stopped":
                assert cause is not None and cause.type == "item_limit", cause
                assert not shouted
            else:
                assert run.status == "complete" and cause is None, cause
            states[f"{text}/{max_items}"] = {
                "status": run.status,
                "loc": list(cause.loc) if cause else None,
                "ctx": dict(cause.ctx) if cause else None,
                "shout_calls": len(shouted),
                "result": run.outputs["result"] if run.status == "complete" else None,
            }
    return states


#: The data a page's empty ``<script type="application/json" id="...">`` gets, by its id.
_DATA: dict[str, Callable[[], Any]] = {
    "catalogue-data": _catalogue,
    "palette-data": _palette,
    "breaks-data": _breaks,
    "widget-data": _widget_data,
}
