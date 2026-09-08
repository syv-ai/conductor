"""Reading and writing a ``Graph`` as a dict or a YAML file.

``TypeAdapter(Graph)`` already knows every record, binding variant and
``Ref``, so this module only wraps it; a host persisting flows its own
way can use the same adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from conductor.graph.model import Graph

_GRAPH = TypeAdapter(Graph)


def flow_to_dict(graph: Graph) -> dict[str, Any]:
    return _GRAPH.dump_python(graph, mode="json")


def load_flow(data: dict[str, Any]) -> Graph:
    return _GRAPH.validate_python(data)


def flow_to_yaml(graph: Graph) -> str:
    return yaml.safe_dump(flow_to_dict(graph), sort_keys=False, allow_unicode=True)


def yaml_to_flow(source: str) -> Graph:
    return load_flow(yaml.safe_load(source))


def load_flow_from_path(path: str | Path) -> Graph:
    return yaml_to_flow(Path(path).read_text(encoding="utf-8"))


def dump_flow(graph: Graph, path: str | Path) -> None:
    Path(path).write_text(flow_to_yaml(graph), encoding="utf-8")
