# syv-conductor

The core of Conductor: nodes that are typed Python functions, graphs of them as data, and an engine that calls a node once per item wherever it is wired to a list another node returns as a `Series`, with retries and runs that pause for a person.

```python
import conductor
```

Two companion packages are released with it at the same version: `syv-conductor-nodes`, nodes that only compute, and `syv-conductor-providers`, a FastAPI router that serves runs. The documentation, the examples and the changelog live in the repository: https://github.com/syv-ai/conductor
