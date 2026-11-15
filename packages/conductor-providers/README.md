# syv-conductor-providers

[Conductor](https://github.com/syv-ai/conductor) in the frameworks a host already uses. `conductor_providers.fastapi.run_router` serves runs over HTTP: start one, follow its events, answer it, resume it and cancel it, on any number of workers sharing one database.

```bash
uv add syv-conductor-providers
```

You write a `RunStore`, six async methods against your database, make a `Worker` on it, and mount the router over the worker with your login:

```python
from conductor import NodeRegistry
from conductor_providers.fastapi import RunStore, Worker, run_router
from fastapi import FastAPI


def current_caller() -> str:
    return "anna"  # yours: your login


def mount(app: FastAPI, store: RunStore, registry: NodeRegistry) -> None:
    worker = Worker(store=store, caller_type=str, nodes_for=lambda caller: registry)
    app.include_router(run_router(
        worker, caller=current_caller, may_see=lambda caller, started_by: caller == started_by,
    ))
```

A host with routes of its own calls the same `Worker` from them.

[`docs/serving-runs.md`](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md) is the guide: a store on SQLite, the routes, a registry per tenant, and what happens when a worker dies or two people answer at once.

The documentation, the examples and the changelog live in the repository: https://github.com/syv-ai/conductor
