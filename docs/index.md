<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

A reusable, host-agnostic engine that compiles and runs graphs of typed nodes. Declare a node as a class whose typed `run` signature is its interface, place it in a graph, compile the graph into a record that says everything about it, and run it as a stream of events, once per row wherever a series arrives.

## Highlights

- **One node contract.** A node is a `NodeDefinition` subclass, and the signature of `run` is its interface. It is read once into `Input` and `Output` records, which drive validation, execution and the palette.
- **A type vocabulary the host owns.** Every value on an edge has a `DType`. Conductor ships the mechanism (`DType`, `accepts`, `Series[X]`) and no vocabulary.
- **Versions with a policy.** `@version(n, policy=Policy(retries=..., timeout=..., concurrency=...))`, `@upgrade(1, 2)`, `@deprecated`.
- **Compile, then execute.** `CompiledGraph.from_graph` never raises for a fault in the graph: everything wrong is a `Problem` with a stable `code`, anchored on a node.
- **Rows.** A series arriving on a scalar input runs the node once per row, concurrently under its policy; a `Series[X]` input receives the whole series.
- **A person in the loop.** A node returns `Asks`, the run ends pending with every question, and the next call to `execute` carries the answers.
- **Structured failures.** Every run-time failure carries an `ErrorCause` (`code`, `message`, `details`, `row`), on the exception and on the event.
- **Branching by value.** A node returns `SKIPPED` on the branch it did not take; exclusive outputs share a `choice`.

## Quick start

```python
from typing import Annotated

from conductor import CompiledGraph, Edges, Graph, GraphNode, NodeDefinition, NodeRegistry, Policy, Ref, Result, Static, version
from conductor.execution.engine import execute_sync
from conductor.widgets import Text as TextWidget
from conductor_nodes.types import Number, Text          # or DTypes of your own


class Fetch(NodeDefinition):
    id = "fetch"
    title = "Fetch"
    description = "Pretends to fetch a page"
    category = "http"

    @version(1, policy=Policy(retries=3, delay=0.5))
    def run(self, url: Annotated[Text, TextWidget(title="URL")]) -> Annotated[Text, Result(title="Body")]:
        return Text(f"<html>{url}</html>")


class Length(NodeDefinition):
    id = "length"
    title = "Length"
    description = "Counts characters"
    category = "text"

    def run(self, text: Annotated[Text, TextWidget(title="Text")]) -> Annotated[Number, Result(title="Length")]:
        return Number(len(text))


registry = NodeRegistry()
registry.register(Fetch)
registry.register(Length)

compiled = CompiledGraph.from_graph(
    Graph(nodes=[
        GraphNode(id="page", type="fetch", version=1, bindings={"url": Static(value=["https://a.example", "https://b.example"])}),
        GraphNode(id="size", type="length", version=1, bindings={"text": Edges(refs=(Ref("page", "result"),))}),
    ]),
    registry,
)
assert compiled.is_runnable, compiled.problems

results = execute_sync(compiled)
list(results["size"]["result"])     # [30.0, 30.0]: two URLs typed in, so both nodes ran once per URL
```

## Bindings: one input, one source

A graph is its nodes; there is no edge list. Each placed node says per input where the value comes from: `Edges` names other nodes' outputs, `Static` holds a typed-in value, and an input with no binding takes its declared default. Dependencies, cycles and what the graph takes and returns are all read off the bindings, so an edge has exactly one representation.

```python
graph = Graph(nodes=[
    GraphNode(id="mapper", type="build-map", version=1, bindings={"seed": Static(value="x")}),
    GraphNode(id="redactor", type="redact", version=1, bindings={
        "text": Static(value="Alice met Bob."),
        "mapping": Edges(refs=(Ref("mapper", "result"),)),
    }),
])
```

## Retry

Retries live on the version's `Policy` and nowhere else, and each row retries on its own.

- The delay before attempt `n` is `delay * 2 ** (n - 1)`.
- Retried: `NodeExecutionError`, `NodeConnectionError`, `NodeTimeoutError` and a plain exception from `run`. Never retried: `NodeValidationError`.
- Each retry emits `node_retry` with `row`, `attempt`, `retries`, `error` and `delay`.
- `Policy(timeout=...)` bounds one attempt; expiry is `NodeTimeoutError`. The thread the attempt ran on is not interrupted.

## Error hierarchy

```
ConductorError
├── CompilationError        execute was asked to run a graph compile found not runnable; carries problems
├── NodeError               one node failed; node_id, original, cause, retryable
│   ├── NodeValidationError     the inputs were wrong (never retried)
│   ├── NodeExecutionError      run raised
│   ├── NodeTimeoutError        the policy's timeout expired
│   └── NodeConnectionError     an external call failed (retried)
├── GraphExecutionError     execute_sync: the graph failed, was cancelled or timed out
└── GraphPendingError       execute_sync: the leg ended pending; carries pending and cells
```

Raise `NodeConnectionError` from `run` to mark a failure as transient.

## Further reading

- [`OVERVIEW.md`](./OVERVIEW.md): the architecture on one page.
- [`widgets.md`](./widgets.md): the controls, and how an input declares one.
- [`packages/conductor/src/conductor/about/llms.txt`](../packages/conductor/src/conductor/about/llms.txt): the packaged reference (also `python -m conductor.about`).
- The notebooks in [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples) cover nodes, graphs, class nodes, versions and discovery, a person in the loop, and widgets.
