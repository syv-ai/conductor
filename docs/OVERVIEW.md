# Architecture at a glance

Conductor runs graphs of typed nodes. It ships as one package, `conductor`. Any type pydantic can check travels on edges as it is.

## One declaration, three readers

A node's signature is read once into a field per input and output. The engine calls the function, a check built from each field validates what is typed in and what a caller gives, and an editor reads `describe()`, the palette entry with each field's type, JSON schema, widget and title. Nothing is declared twice.

## Declare, load, run

- **Declare.** `@node` checks a function when it is decorated: an `async def`, a parameter without a type, a limit that doesn't fit its type, or a `Mapping` return with no hook fails at import. A `NodeRegistry` is what a host offers: `@registry.node` declares and adds, `add_subregistry` copies a module's registry in, and a second node under one id raises.
- **Load.** `registry.load(record)` resolves each node's version, puts each embedded graph's nodes in its place, checks every edge and value, asks the field hooks, and decides where each node runs. Everything wrong is a `Problem` in `graph.problems`; `graph.is_runnable` says whether a run may start, and `graph.plan` says where each node runs.
- **Run.** `graph.run(**inputs)` returns the `Run` its leg ended on. `run.stream()` yields the events of one leg. Every result, failure and pause is on the run, and `run.to_json()` is all of it.

## Rows

The unit of work is a node on a row. A node fed a column on an input declared for one value runs once per row, and its result is a column on the same rows. A unit starts as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being produced:

```
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A node that returns many values (`Series[X]` or `Table[R]`) makes a new table under the row it ran on, named by that output (`split.result`). A row of it is its parent's key plus a number, `(0, 3)`. Tables are flat and never hold a table.

A `Series[X]` input receives all the rows under the node's row, flat. It chooses the node's row only when nothing else does; then the node runs once per row of the level right above the series. A `Table` input receives a `polars.DataFrame`, its columns read along one path of the row tree, a parent's value copied down onto its children's rows. Each column of a table output is a field an edge can read (`csv.result.email`). Rows of two different tables never meet by themselves: the author joins them on a key in a node (`rows_not_related`).

A blank reaches a node only if its input says `| None`. Otherwise the node isn't called for that row, its outputs are blank there, and `run.not_called` says which row and input.

## Skips and pauses

A node returns `SKIPPED` on a branch it did not take. A skip at a row leaves the column downstream with a gap; a skip above a node's rows skips everything under it.

A node raises `Ask` when a person must answer. The rest of the graph runs on and the run ends `waiting`. `run.answer(unit, value)` writes an answer in, checked against the node's return type, and `run.resume()` goes on from there, calling no finished unit twice.

## Retries and timeouts

Retries live on the node's `Policy`, and each row retries on its own. Only the outside world's failure is retried: an `ExternalFailure` the node raises, or an exception whose class `retry_on` names. The delay before attempt `n` is `retry_delay * 2 ** (n - 1)`.

`Policy.attempt_timeout` is how long the engine waits on one attempt. It never interrupts the thread: a timed-out attempt is final, and the thread finishes on its own. A node with `concurrency` 1 and no timeout is called in order on the leg's own thread; any other runs on threads, at most `concurrency` rows at once.

Every failure is a `Problem` (`type`, `loc`, `msg`, `ctx`) on the event and on the run, the same record loading reports, so a host routes it by its type.

## The graph record

A graph is a `GraphRecord`, plain data: `{"nodes": [...], "edges": [["a.result", "b.text"], ...]}`. A node names the node it uses (`node`), its version, and what is typed into it: `values` are fixed, `defaults` stay open for a caller, and `many` marks an open input that takes a list. Edges are a set; several into one input gather in node order. An input takes an edge, a value or a default, never two, and with none of them its declared default. What the graph takes and returns is worked out from the edges. `make_graph(...)` writes a record from Python. A `GraphRecord` saves as JSON with pydantic's own `model_dump_json`.
