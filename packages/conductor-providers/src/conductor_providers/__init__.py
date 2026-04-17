"""conductor-providers — frontend/framework adapters for conductor.

Each provider is a subpackage translating between conductor's Python-side
data model (``NodeRegistry``, ``GraphNode``, ``GraphEdge``) and the
framework's wire format. The initial — and currently only — provider is
``conductor_providers.react`` which speaks ReactFlow.

    from conductor_providers import react

    palette = react.palette_from_registry(registry)
    flow_json = react.graph_to_react(nodes, edges)
    nodes2, edges2 = react.react_to_graph(flow_json)

New providers go in sibling subpackages (``conductor_providers.svelte``,
``conductor_providers.vue``, …). There is no shared abstract base — each
provider picks the shape that matches its framework idiom.
"""

from conductor_providers import react

PROVIDERS: list[str] = ["react"]

__all__ = ["PROVIDERS", "react"]
