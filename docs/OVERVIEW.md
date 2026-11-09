# Architecture at a glance

Conductor compiles and runs graphs of typed nodes. It ships as three packages: `conductor` (the engine), `conductor_nodes` (standard nodes) and `conductor_providers` (the Mermaid drawing). Plain Python types travel on edges as they are. A host declares a `DType` class only for a type of its own.

## One declaration, three readers

A node's signature is read once into `Input` and `Output` records. The engine calls the function. A check built from the inputs validates each call. An editor reads `describe()`, the palette entry with each field's type, widget and title. Nothing is declared twice.

## Declare, compile, run

- **Declare.** `@node` checks a function when it is decorated: a widget written bare instead of on a `Param`, an `async def`, or an `Any` or `Mapping` return with no hook fails at import. `NodeRegistry` is built whole and adds the catalogue's rules, such as an alternative that exists.
- **Compile.** `CompiledGraph.from_graph(graph, registry)` resolves each node's version, validates the bindings, types each field from its edges, decides where each node runs, asks the field hooks and puts each embedded graph's nodes in its place. Everything wrong is a `Problem`; `is_runnable` says whether a run may start. `compiled.run_plan` says what compile decided.
- **Run.** `async with compiled.stream() as events:` yields the events of one leg. `await compiled.run()` returns the `RunState` it ended on, and `compiled.run_sync()` does the same from a script. Every result, failure and pause is on that state.

## Every level is a table

The unit of work is a node on a row. A node fed a column on an input declared for one value runs once per row, and its result is a column on the same rows. A unit starts as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being produced:

```
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A node that returns many values makes a new table under the row it ran on. Tables are flat and never hold a table.

A `Series[X]` input receives all the rows under the node's row, however many levels down, flat. It chooses the node's row only when nothing else does; then the node runs once per row of the level right above the series. A node's rows run at most `Policy.concurrency` at a time.

A `Table` input receives a `polars.DataFrame`. Each column of a table output is a field an edge can read (`csv.result.email`). A `__row` column every table carries holds each row's id (`contacts.result:0`). No node declares anything about rows: a returned row with no id, or with one an earlier row holds, gets a fresh one.

Tables and columns wired into one `Table` input are merged on those ids as the graph runs. A parent column beside its children is copied down, so the rows are those of the deepest source, and a parent with nothing under it has no row in that table. Sources on the same level keep all their rows, with blanks where one has nothing. Sources that can never match are refused when the graph is checked (`no_rows_in_common`), and alternatives (outputs of one `exclusive_group` or what hangs off them) are the branch that ran, not a merge.

A blank reaches a node only if its input says `| None`. Otherwise the node isn't called for that row and its outputs are blank there.

## Skips and pauses

A node returns `SKIPPED` on a branch it did not take. A skip at a row leaves the column downstream with a gap; a skip above a node's rows skips everything under it.

A node returns `AskPerson` when a person must answer. The rest of the graph runs on and the state ends `waiting`. `compiled.answer(state, answers)` writes the answers in, and `compiled.run(state)` goes on from there, calling no finished unit twice.

## Retries and timeouts

Retries live on the version's `Policy`, and each row retries on its own. Only the outside world's failure is retried: an `ExternalFailure` the node raises, or an exception whose class `retry_on` names. The delay before attempt `n` is `retry_delay * 2 ** (n - 1)`.

`Policy.attempt_timeout` is how long the engine waits on one attempt. It never interrupts the thread: a timed-out attempt is final, and the thread finishes on its own. A leg that fails waits for its other calls only that long, or for as long as the call takes where no timeout is set. The timeout worth retrying is the client's own, set inside the function. A function that holds the GIL blocks the whole process, so where runs happen is the host's decision.

Every failure is a `Problem` (`type`, `loc`, `msg`, `ctx`) on the event and on the state, the same record a compile reports, so a host routes it by its type.

## Bindings

A graph is its nodes; there is no edge list. Each input holds one binding at most: `From("node.output")` is an edge, `Static(value)` is what the author typed, and no binding means the declared default. Dependencies, cycles and what the graph takes and returns are worked out from the bindings. A `Graph` saves as JSON with pydantic's own `model_dump_json`.
