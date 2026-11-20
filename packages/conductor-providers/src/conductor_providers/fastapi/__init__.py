"""Conductor served over HTTP: ``run_router``, the ``Worker`` under it, and the ``RunStore`` they keep runs in.

Make a worker on your store, and mount the router over it. ``metadata`` gives what a new run keeps about itself, such
as who started it, and ``run_filter`` says which runs a request may see::

    worker = Worker(store, nodes_for=lambda metadata: registry)
    app = FastAPI()
    app.include_router(run_router(worker, metadata=run_metadata, run_filter=own_runs))

Or call the ``Worker`` from routes of your own, on ``APIRouter(route_class=RefusalRoute)``, so that they refuse a
call as the router's routes do. Any number of workers can serve runs against one store: each takes over the runs of a
worker that died, and a worker that shuts down hands its runs back.
"""

from conductor_providers.fastapi.problems import SERVING_PROBLEMS
from conductor_providers.fastapi.router import RefusalRoute, run_router
from conductor_providers.fastapi.state import RunView
from conductor_providers.fastapi.store import RunChanges, RunStore, StoredRun
from conductor_providers.fastapi.worker import Worker

__all__ = [
    "run_router",
    "RefusalRoute",
    "Worker",
    "RunStore",
    "StoredRun",
    "RunChanges",
    "RunView",
    "SERVING_PROBLEMS",
]
