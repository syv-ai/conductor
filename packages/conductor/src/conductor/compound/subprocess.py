"""Subprocess compound node.

A subprocess node references another flow by ``id`` and ``version`` and
executes it synchronously as a single node in the outer flow. The target
flow's compile step is run at call time and its top-level results are
returned as the subprocess node's result.

The subprocess is a one-node "compound region" — the start and end ids
are the same. This keeps the dispatch path uniform with for-each and
while regions (they all flow through ``compiled.compound_nodes``) while
the engine still treats the subprocess as schedulable (see the
``end_id != start_id`` guard in ``graph/compiler.py``).

In v1:

* **Closed-world**: the target flow must be in the
  :class:`SubprocessRegistry` passed to ``compile(subprocess_registry=...)``.
* **Sync only** — the caller waits for the sub-flow to complete.
* Cycles are detected via a runtime depth cap (``_MAX_SUBPROCESS_DEPTH``).
* HITL/signal pauses inside a subprocess surface as
  :class:`~conductor.errors.SubprocessFailedError`.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.errors import (
    CompilationError,
    NodeExecutionError,
    SubprocessFailedError,
)
from conductor.expr import ExpressionError
from conductor.expr import parse as parse_expr
from conductor.graph.model import GraphEdge, GraphNode

_MAX_SUBPROCESS_DEPTH = 16


class SubprocessNode:
    """Executes a referenced flow as a single node in the caller's DAG."""

    def __init__(self, region: Region, execution_order: tuple[str, ...]) -> None:
        self.region = region
        self._subprocess_registry: Any = None

    def set_subprocess_registry(self, registry: Any) -> None:
        self._subprocess_registry = registry

    def execute(self, req: Any) -> Any:
        sub_id = req.inputs.get("flow_id") or req.data.get("flow_id")
        sub_version = int(req.inputs.get("flow_version", req.data.get("flow_version", 1)))
        input_map = req.inputs.get("inputs") or req.data.get("inputs") or {}

        if not sub_id:
            raise CompilationError(
                f"Subprocess node '{self.region.start_id}' has no `flow_id` — "
                f"set it as static data or wire it in."
            )

        sub_flow = None
        if self._subprocess_registry is not None:
            sub_flow = self._subprocess_registry.get(sub_id, sub_version)

        if sub_flow is None:
            raise NodeExecutionError(
                f"Subprocess '{sub_id}@{sub_version}' not found in subprocess registry",
                node_id=self.region.start_id,
                node_type="subprocess-call",
            )

        state = req.state
        depth = state.context.get("_subprocess_depth", 0) + 1
        if depth > _MAX_SUBPROCESS_DEPTH:
            raise NodeExecutionError(
                f"Subprocess recursion depth exceeded ({depth}>{_MAX_SUBPROCESS_DEPTH})",
                node_id=self.region.start_id,
                node_type="subprocess-call",
            )

        # Execute the sub-flow synchronously.
        try:
            from conductor.execution.engine import execute_sync
            from conductor.graph.compiler import compile as compile_flow
            from conductor.graph.model import Flow

            # Apply input mapping if it's a dict of CEL expressions or static values.
            resolved_inputs = _apply_input_map(input_map, req.inputs)

            # Rebuild sub-flow with overridden static data on matching handles.
            # This is cleaner than mutating frozen dataclasses in place.
            if resolved_inputs:
                patched_nodes = [
                    dc_replace(
                        n,
                        data={
                            **(n.data or {}),
                            **{k: v for k, v in resolved_inputs.items()
                               if n.data and k in n.data},
                        },
                    )
                    for n in sub_flow.nodes
                ]
                sub_flow = Flow(
                    nodes=patched_nodes,
                    edges=list(sub_flow.edges),
                    id=sub_flow.id,
                    version=sub_flow.version,
                    name=sub_flow.name,
                    description=sub_flow.description,
                    dependencies=sub_flow.dependencies,
                    triggers=sub_flow.triggers,
                    on_error_default=sub_flow.on_error_default,
                )

            compound_types = _collect_compound_types(state.compiled)
            sub_compiled = compile_flow(
                flow=sub_flow,
                registry=state.compiled.registry,
                compound_types=compound_types,
                extension_resolver=state.compiled.extension_resolver,
                subprocess_registry=self._subprocess_registry,
            )

            sub_context = {
                **state.context,
                "_subprocess_depth": depth,
                "_parent_node_id": self.region.start_id,
            }
            sub_results = execute_sync(
                sub_compiled,
                context=sub_context,
                store_data=state.store.to_dict(),
            )
        except SubprocessFailedError:
            raise
        except NodeExecutionError:
            raise
        except Exception as e:
            raise SubprocessFailedError(
                f"Subprocess '{sub_id}@{sub_version}' failed: {e}",
                node_id=self.region.start_id,
                node_type="subprocess-call",
                original=e,
            ) from e

        return sub_results


def _apply_input_map(input_map: Any, req_inputs: dict[str, Any]) -> dict[str, Any]:
    """Resolve the `inputs:` config. Values can be CEL or literal."""
    if not isinstance(input_map, dict):
        return {}
    resolved: dict[str, Any] = {}
    for key, val in input_map.items():
        if isinstance(val, str) and val.startswith("$"):
            try:
                expr = parse_expr(val)
                resolved[key] = expr.evaluate({"$": req_inputs, **req_inputs})
            except ExpressionError:
                resolved[key] = val
        else:
            resolved[key] = val
    return resolved


def _collect_compound_types(compiled: Any) -> list[Any]:
    """Best-effort: re-use the outer flow's compound types."""
    types: list[Any] = []
    seen: set[str] = set()
    for executor in compiled.compound_nodes.values():
        cls_name = executor.__class__.__name__
        if cls_name in seen:
            continue
        seen.add(cls_name)
        if cls_name == "ForEachNode":
            from conductor.compound.for_each import FOR_EACH
            types.append(FOR_EACH)
        elif cls_name == "WhileNode":
            from conductor.compound.while_loop import WHILE
            types.append(WHILE)
        elif cls_name == "SubprocessNode":
            types.append(SUBPROCESS)
    return types


def discover_subprocess_regions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[Region]:
    """Every ``subprocess-call`` node is its own single-node region."""
    regions: list[Region] = []
    for n in nodes:
        if n.type.startswith("subprocess-call"):
            regions.append(Region(
                start_id=n.id,
                end_id=n.id,
                body_ids=frozenset(),
            ))
    return regions


class SubprocessRegistry:
    """Lookup table for sub-flows referenced by id/version."""

    def __init__(self) -> None:
        self._flows: dict[tuple[str, int], Any] = {}

    def register(self, flow: Any) -> None:
        if flow.id is None:
            raise ValueError("Sub-flow must have an `id` to be registered")
        self._flows[(flow.id, flow.version)] = flow

    def get(self, flow_id: str, version: int) -> Any | None:
        return self._flows.get((flow_id, version))

    def all(self) -> list[Any]:
        return list(self._flows.values())


SUBPROCESS = CompoundNodeType(
    start_type_prefix="subprocess-call",
    end_type_prefix="subprocess-end",
    discover=discover_subprocess_regions,
    factory=lambda region, order: SubprocessNode(region, order),
)
