# syv-conductor-providers

Serve [conductor](https://github.com/syv-ai/conductor) runs over HTTP with FastAPI. Start a run, follow it, answer its questions, resume it or cancel it, on as many server processes as you like.

```bash
uv add syv-conductor-providers
```

You write a small store over your database, then mount the router:

```python
from collections.abc import Callable

from conductor import NodeRegistry
from conductor_providers.fastapi import RunStore, Worker, run_router
from fastapi import FastAPI


def run_metadata() -> dict[str, str]:
    return {"user": "anna"}  # yours: who is calling


def own_runs() -> Callable[[dict[str, str]], bool]:
    return lambda metadata: metadata["user"] == "anna"  # yours: which runs they may see


def mount(app: FastAPI, store: RunStore, registry: NodeRegistry) -> None:
    worker = Worker(store, nodes_for=lambda metadata: registry)
    app.include_router(run_router(worker, metadata=run_metadata, run_filter=own_runs))
```

Each run keeps its metadata, and the filter decides who sees it. The guide, [serving runs over HTTP](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md), has a store on SQLite, the routes, and what happens when a worker dies.
