# Architecture at a glance

Conductor compiles and runs graphs of typed nodes. It ships as three packages: `conductor` (the engine), `conductor_nodes` (standard nodes) and `conductor_providers` (ReactFlow, FastAPI and Mermaid adapters). The core has no vocabulary of its own: a host declares its types (`Text`, `Number`, `Document`, …) as `DType` classes.

## One declaration, three readers

A node's `run` signature is read once into `Input` and `Output` records. Execution calls `run` on a fresh instance. Validation builds a pydantic model from the inputs. An editor reads `describe()`, the palette entry with each field's type, widget and title. Nothing is declared twice.

## Declare, compile, execute

- **Declare.** A class is checked when it is defined: a missing `id`, `title`, `description` or `category`, a widget written bare instead of on a `Param`, or an `async def run` fails at import. `NodeRegistry.register` adds the catalogue's rules, such as versions numbered from 1 with no holes.
- **Compile.** `CompiledGraph.from_graph(graph, registry)` resolves each node's version, validates the bindings, types each field from its edges, decides which nodes run once per row, asks the field hooks and puts each embedded graph's nodes in its place. Everything wrong is a `Problem`; `is_runnable` says whether a run may start.
- **Execute.** `execute(compiled)` is an async generator of events. `await run(compiled)` returns the ending, and `run_sync(compiled)` does the same from a script. Every ending carries `state`, a `RunState`; `state.results(compiled)` reads every node's values.

## The row engine

The unit of work is a node on a row. A node fed a series on an input declared for one value runs once per row, and each unit starts as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being produced:

```
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A `Series[X]` input receives the whole series, or the rows under each parent row. A node's rows run at most `Policy.concurrency` at a time.

A node returns `SKIPPED` on a branch it did not take. A skip at a row leaves the series downstream sparse; a skip above a node's rows skips everything under it.

A node returns `Asks` when a person must answer. The rest of the graph runs on and the run ends `graph_pending`. The next call to `execute` takes `state` from that ending and the answers in `cache`, so nothing runs twice.

## Retries and timeouts

Retries live on the version's `Policy`, and each row retries on its own. Only the outside world's failure is retried: an `ExternalFailure` the node raises, or a foreign exception whose class `retry_on` names. The delay before attempt `n` is `delay * 2 ** (n - 1)`.

`Policy.timeout` is how long the engine waits on one attempt. It never interrupts the thread: a timed-out attempt is final, and the thread finishes on its own. The timeout worth retrying is the client's own, set inside `run`. A `run` that holds the GIL blocks the whole process, so where runs happen is the host's decision.

Every failure carries an `ErrorCause` (`code`, `message`, `details`, `row`) on the exception and on the event, so a host routes it by code.

## Bindings

A graph is its nodes; there is no edge list. Each input holds one binding at most: `From("node.output")` is an edge, `Static(value)` is what the author typed, and no binding means the declared default. Dependencies, cycles and what the graph takes and returns are derived from the bindings. A `Graph` saves itself with `to_path` / `from_path`.
