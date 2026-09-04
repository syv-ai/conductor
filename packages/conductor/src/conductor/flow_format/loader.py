"""Reading and writing a ``Flow`` as a dict or a YAML file.

``TypeAdapter(Flow)`` already knows every record, binding variant and
``Ref``, so this module only wraps it; a host persisting flows its own
way can use the same adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from conductor.graph.model import Flow

_FLOW = TypeAdapter(Flow)


def flow_to_dict(flow: Flow) -> dict[str, Any]:
    return _FLOW.dump_python(flow, mode="json")


def load_flow(data: dict[str, Any]) -> Flow:
    return _FLOW.validate_python(data)


def flow_to_yaml(flow: Flow) -> str:
    return yaml.safe_dump(flow_to_dict(flow), sort_keys=False, allow_unicode=True)


def yaml_to_flow(source: str) -> Flow:
    return load_flow(yaml.safe_load(source))


def load_flow_from_path(path: str | Path) -> Flow:
    return yaml_to_flow(Path(path).read_text(encoding="utf-8"))


def dump_flow(flow: Flow, path: str | Path) -> None:
    Path(path).write_text(flow_to_yaml(flow), encoding="utf-8")
