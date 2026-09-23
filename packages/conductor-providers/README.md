# syv-conductor-providers

Framework adapters for Conductor: `conductor_providers.react` translates a graph to ReactFlow JSON and back, and `conductor_providers.fastapi` mounts the engine's routes on a FastAPI `APIRouter` (the `fastapi` extra).

```python
import conductor_providers
```

The three distributions, `syv-conductor`, `syv-conductor-nodes` and `syv-conductor-providers`, are released together from one repository and pin each other to the same version. The documentation, the examples and the changelog live there: https://github.com/syv-ai/conductor
