# syv-conductor-providers

[Conductor](https://github.com/syv-ai/conductor) in the frameworks a host already uses. `conductor_providers.fastapi.run_router` serves runs over HTTP: start one, follow its events, answer it, resume it and cancel it, on any number of workers sharing one database.

```bash
uv add syv-conductor-providers
```

You write a `RunStore`, six async methods against your database, make a `Worker` on it, and mount the router over the worker. Each run keeps some metadata (here, who started it), and a filter says which runs a request may see:

```python
from collections.abc import Callable

from conductor import NodeRegistry
from conductor_providers.fastapi import RunStore, Worker, run_router
from fastapi import FastAPI


def current_user() -> str:
    return "anna"  # yours: your login


def run_metadata() -> dict[str, str]:
    return {"user": current_user()}


def own_runs() -> Callable[[dict[str, str]], bool]:
    user = current_user()
    return lambda metadata: metadata["user"] == user


def mount(app: FastAPI, store: RunStore, registry: NodeRegistry) -> None:
    worker = Worker(store, nodes_for=lambda metadata: registry)
    app.include_router(run_router(worker, metadata=run_metadata, run_filter=own_runs))
```

A host with routes of its own calls the same `Worker` from them.

[`docs/serving-runs.md`](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md) is the guide: a store on SQLite, the routes, a registry per tenant, and what happens when a worker dies or two people answer at once.

The documentation, the examples and the changelog live in the repository: https://github.com/syv-ai/conductor
