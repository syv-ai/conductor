# Architecture at a glance

Conductor compiles and runs graphs of typed nodes. It ships as three packages: `conductor` (the engine), `conductor_nodes` (standard nodes) and `conductor_providers` (ReactFlow, FastAPI and Mermaid adapters). The core has no vocabulary of its own: a host declares its types (`Text`, `Number`, `Document`, …) as `DType` classes.

## One declaration, three readers

A node's signature is read once into `Input` and `Output` records. Execution calls the function. Validation builds a pydantic model from the inputs. An editor reads `describe()`, the palette entry with each field's type, widget and title. Nothing is declared twice.

## Declare, compile, run

- **Declare.** `@node` checks a function when it is decorated: a widget written bare instead of on a `Param`, an `async def`, or an `Any` or `Mapping` return with no hook fails at import. `NodeRegistry` is built whole and adds the catalogue's rules, such as an alternative that exists.
- **Compile.** `CompiledGraph.from_graph(graph, registry)` resolves each node's version, validates the bindings, types each field from its edges, decides which nodes run once per row, asks the field hooks and puts each embedded graph's nodes in its place. Everything wrong is a `Problem`; `is_runnable` says whether a run may start.
- **Run.** `async with compiled.stream() as events:` yields the events of one leg; `await compiled.run()` returns the state it ended on, and `compiled.run_sync()` does the same from a script.

## The row engine

The unit of work is a node on a row. A node fed a series on an input declared for one value runs once per row, and each unit starts as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being produced:

```
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A `Series[X]` input receives the whole series, or all the rows under the node's row, however many levels down, flat. It chooses the node's row only when nothing else does; then the node runs once per row of the level right above the series. A node's rows run at most `Policy.concurrency` at a time.

A node returns `SKIPPED` on a branch it did not take. A skip at a row leaves the series downstream sparse; a skip above a node's rows skips everything under it.

A node returns `AskPerson` when a person must answer. The rest of the graph runs on and the state ends `waiting`. `compiled.answer(state, answers)` writes the answers in, and `compiled.run(state)` goes on from there, running no node twice.

## Retries and timeouts

Retries live on the version's `Policy`, and each row retries on its own. Only the outside world's failure is retried: an `ExternalFailure` the node raises, or a foreign exception whose class `retry_on` names. The delay before attempt `n` is `delay * 2 ** (n - 1)`.

`Policy.timeout` is how long the engine waits on one attempt. It never interrupts the thread: a timed-out attempt is final, and the thread finishes on its own. The timeout worth retrying is the client's own, set inside `run`. A `run` that holds the GIL blocks the whole process, so where runs happen is the host's decision.

Every failure is a `Problem` (`type`, `loc`, `msg`, `ctx`) on the event and on the state, the same record a compile reports, so a host routes it by its code.

## Bindings

A graph is its nodes; there is no edge list. Each input holds one binding at most: `From("node.output")` is an edge, `Static(value)` is what the author typed, and no binding means the declared default. Dependencies, cycles and what the graph takes and returns are derived from the bindings. A `Graph` saves itself with `to_path` / `from_path`.
