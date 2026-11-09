"""conductor-providers — adapters that sit between conductor and the
outside world.

``conductor_providers.mermaid.flowchart(compiled)`` draws a compiled graph
as a Mermaid flowchart, for a person reading it in a notebook, a pull
request or a bug report.

``conductor_providers.fastapi`` mounts conductor on a FastAPI ``APIRouter``
so a host doesn't write the payloads, the SSE framing and the compile
answer itself. It needs an optional extra
(``uv add "syv-conductor-providers[fastapi]"``), so it is imported on its own::

    from conductor_providers.fastapi import conductor_router
    app.include_router(conductor_router(registry, prefix="/conductor"))
"""

from conductor_providers import mermaid

__all__ = ["mermaid"]
