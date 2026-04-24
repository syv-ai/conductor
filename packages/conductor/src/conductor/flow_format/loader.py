"""Load and dump Flow objects from YAML / JSON / dict form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conductor.graph.model import (
    Flow,
    FlowDependency,
    FlowTrigger,
    GraphEdge,
    GraphNode,
)

# ---------------------------------------------------------------------------
# Dict ↔ Flow
# ---------------------------------------------------------------------------


def _parse_consumes(raw: Any) -> dict[str, tuple[str, str]] | None:
    """Normalize the ``consumes`` mapping.

    Accepts any of:
    * ``{input_handle: [producer_id, output_handle]}``
    * ``{input_handle: {"producer": id, "handle": output_handle}}``
    * ``{input_handle: (producer_id, output_handle)}``
    """
    if not raw:
        return None
    out: dict[str, tuple[str, str]] = {}
    for handle, ref in raw.items():
        if isinstance(ref, dict):
            out[handle] = (ref["producer"], ref.get("handle", "result"))
        elif isinstance(ref, (list, tuple)) and len(ref) == 2:
            out[handle] = (ref[0], ref[1])
        else:
            raise ValueError(
                f"Consume reference for {handle!r} must be a list, tuple, "
                f"or {{producer, handle}} dict, got {ref!r}"
            )
    return out


def load_flow(data: dict[str, Any]) -> Flow:
    """Build a :class:`Flow` from a canonical dict."""
    if not isinstance(data, dict):
        raise ValueError(f"Flow must be a mapping, got {type(data).__name__}")

    nodes_raw = data.get("nodes") or []
    edges_raw = data.get("edges") or []
    deps_raw = data.get("dependencies") or []
    triggers_raw = data.get("triggers") or []

    nodes: list[GraphNode] = []
    for n in nodes_raw:
        if "id" not in n:
            raise ValueError(f"Node missing `id`: {n}")
        if "type" not in n:
            raise ValueError(f"Node '{n.get('id')}' missing `type`")
        consumes = _parse_consumes(n.get("consumes"))
        nodes.append(GraphNode(
            id=n["id"],
            type=n["type"],
            data=n.get("data") or {},
            produces=n.get("produces") or None,
            consumes=consumes,
            compensation=n.get("compensation"),
            on_error=n.get("on_error"),
        ))

    edges: list[GraphEdge] = []
    for e in edges_raw:
        if "id" not in e:
            raise ValueError(f"Edge missing `id`: {e}")
        edges.append(GraphEdge(
            id=e["id"],
            source=e["source"],
            target=e["target"],
            source_handle=e.get("source_handle") or e.get("sourceHandle"),
            target_handle=e.get("target_handle") or e.get("targetHandle"),
            when=e.get("when"),
            priority=int(e.get("priority", 0)),
        ))

    deps = tuple(
        FlowDependency(
            id=d["id"], kind=d["kind"], config=d.get("config") or {},
        )
        for d in deps_raw
    )

    triggers = tuple(
        FlowTrigger(
            id=t["id"],
            kind=t["kind"],
            config=t.get("config") or {},
            input_map=t.get("input_map") or t.get("map"),
        )
        for t in triggers_raw
    )

    return Flow(
        nodes=nodes,
        edges=edges,
        id=data.get("id"),
        version=int(data.get("version", 1)),
        name=data.get("name"),
        description=data.get("description"),
        dependencies=deps,
        triggers=triggers,
        on_error_default=data.get("on_error_default", "fail"),
    )


def flow_to_dict(flow: Flow) -> dict[str, Any]:
    """Reverse of :func:`load_flow` — produce a round-trippable dict.

    Defaults are omitted to keep serialized files tidy:
    ``version=1``, ``on_error_default="fail"``, and any ``None`` /
    empty fields are dropped. ``load_flow`` re-applies the defaults.
    """
    out: dict[str, Any] = {}
    if flow.id is not None:
        out["id"] = flow.id
    if flow.version != 1:
        out["version"] = flow.version
    if flow.name is not None:
        out["name"] = flow.name
    if flow.description is not None:
        out["description"] = flow.description
    if flow.on_error_default != "fail":
        out["on_error_default"] = flow.on_error_default

    if flow.dependencies:
        out["dependencies"] = [
            {"id": d.id, "kind": d.kind, "config": dict(d.config)}
            for d in flow.dependencies
        ]
    if flow.triggers:
        out["triggers"] = [
            {
                "id": t.id,
                "kind": t.kind,
                "config": dict(t.config),
                **({"input_map": t.input_map} if t.input_map else {}),
            }
            for t in flow.triggers
        ]

    out["nodes"] = []
    for n in flow.nodes:
        nd: dict[str, Any] = {"id": n.id, "type": n.type}
        if n.data:
            nd["data"] = dict(n.data)
        if n.produces:
            nd["produces"] = dict(n.produces)
        if n.consumes:
            nd["consumes"] = {k: list(v) for k, v in n.consumes.items()}
        if n.compensation:
            nd["compensation"] = n.compensation
        if n.on_error:
            nd["on_error"] = n.on_error
        out["nodes"].append(nd)

    out["edges"] = []
    for e in flow.edges:
        ed: dict[str, Any] = {
            "id": e.id, "source": e.source, "target": e.target,
        }
        if e.source_handle is not None:
            ed["source_handle"] = e.source_handle
        if e.target_handle is not None:
            ed["target_handle"] = e.target_handle
        if e.when:
            ed["when"] = e.when
        if e.priority:
            ed["priority"] = e.priority
        out["edges"].append(ed)

    return out


# ---------------------------------------------------------------------------
# YAML helpers (require PyYAML at call time)
# ---------------------------------------------------------------------------


def yaml_to_flow(source: str) -> Flow:
    """Parse a YAML string and build a :class:`Flow`."""
    import yaml  # deferred so PyYAML is optional at import time

    data = yaml.safe_load(source)
    return load_flow(data or {})


def flow_to_yaml(flow: Flow) -> str:
    """Dump a :class:`Flow` to a YAML string."""
    import yaml

    return yaml.safe_dump(flow_to_dict(flow), sort_keys=False)


def load_flow_from_path(path: str | Path) -> Flow:
    """Load a flow from a YAML or JSON file, picking by extension."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".json",):
        return load_flow(json.loads(text))
    return yaml_to_flow(text)


def dump_flow(flow: Flow, path: str | Path) -> None:
    """Dump a flow to YAML or JSON, picking by extension."""
    p = Path(path)
    if p.suffix.lower() in (".json",):
        p.write_text(json.dumps(flow_to_dict(flow), indent=2), encoding="utf-8")
    else:
        p.write_text(flow_to_yaml(flow), encoding="utf-8")
