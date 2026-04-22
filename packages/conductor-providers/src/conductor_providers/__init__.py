"""conductor-providers — adapters that sit between conductor and the
outside world.

Two flavors of subpackage:

* **Frontend wire-format adapters** — translate conductor's Python-side data
  model (``NodeRegistry``, ``GraphNode``, ``GraphEdge``) into a shape a
  specific frontend framework expects. ``conductor_providers.react``
  speaks ReactFlow. Svelte, Vue, etc. go in sibling subpackages.

* **Transport adapters** — mount conductor over a specific protocol or
  framework. ``conductor_providers.fastapi`` ships an ``APIRouter``
  factory so hosts don't have to hand-roll pydantic payloads + SSE
  framing + compile-result plumbing.

Transport subpackages declare their framework dep as an optional extra
(e.g. ``pip install conductor-providers[fastapi]``) so frontend-only
consumers don't pay for it.

    from conductor_providers import react
    palette = react.palette_from_registry(registry)

    from conductor_providers.fastapi import conductor_router
    app.include_router(conductor_router(registry, prefix="/flows"))
"""

from conductor_providers import react

PROVIDERS: list[str] = ["react", "fastapi"]

__all__ = ["PROVIDERS", "react"]
