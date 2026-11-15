"""Conductor served over HTTP: ``run_router``, the ``Worker`` under it, and the ``RunStore`` they keep runs in.

Make a worker on your store, and mount the router over it with your login and who may see a run::

    worker = Worker(store=store, caller_type=Caller, nodes_for=lambda caller: registry)
    app = FastAPI()
    app.include_router(run_router(worker, caller=current_caller, may_see=same_caller))

Or call the ``Worker`` from routes of your own. Any number of workers can serve
runs against one store: each takes over the runs of a worker that died, and a
worker that shuts down hands its runs back.
"""

from conductor_providers.fastapi.problems import SERVING_PROBLEMS
from conductor_providers.fastapi.router import run_router
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import RunAsRead, RunChanges, RunView, StoredRun
from conductor_providers.fastapi.worker import Worker

__all__ = [
    "run_router",
    "Worker",
    "RunStore",
    "StoredRun",
    "RunChanges",
    "RunAsRead",
    "RunView",
    "SERVING_PROBLEMS",
]
