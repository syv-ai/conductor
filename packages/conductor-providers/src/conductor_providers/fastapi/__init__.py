"""Conductor served over HTTP: ``run_router`` and the ``RunStore`` it keeps runs in.

Mount it in one line, with your store and your login::

    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=current_caller))

Any number of workers can serve it against one store: each takes over the
runs of a worker that died, and a worker that shuts down hands its runs back.
"""

from conductor_providers.fastapi.router import ROUTER_PROBLEM_CODES, run_router
from conductor_providers.fastapi.store import RunChanges, RunStore, StoredRun

__all__ = ["ROUTER_PROBLEM_CODES", "RunChanges", "RunStore", "StoredRun", "run_router"]
