# How it works

Conductor is one package, `conductor`. This page is the picture you need to read a graph's behaviour off its drawing. The rules in full are in `python -m conductor.about`.

## One signature, three readers

A node's signature is read once, when `@node` decorates it, into one field per input and output. Three things read those fields. The engine calls the function. A check built from each field validates what is typed in, what a caller gives and what the node returns. An editor reads `describe()`: each field's type, JSON schema, title and widget. Nothing is declared twice.

`@node` checks the signature too. An `async def`, a parameter without a type, or a limit that doesn't fit its type fails at import with a `NodeDeclarationError`.

## Load, then run

`registry.load(record)` finds each node's definition at the version the record names, puts each embedded graph's nodes in its place, checks every edge and value, runs the field hooks, and works out where each node runs. It never raises for a fault in the graph. Everything wrong goes in `graph.problems`, and `graph.plan` says where each node runs.

`graph.run(**inputs)` runs one leg. A leg goes until the run is complete, waits for a person, fails or is stopped, and it returns a `Run`. `run.stream()` runs the same leg and yields events as it goes. `run.to_json()` is all of the run, so the next leg can start in another process.

## Rows

The engine's unit of work is a node on a row. A node typed for one value and fed a column runs once per row, and its result is a column on the same rows. A unit starts as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being made:

```text
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A node that returns many values, `Series[X]` or `Table[R]`, makes a table under the row it ran on, named by that output: `split.result`. A row of it is its parent row plus a number, `(0, 3)`. So rows form a tree, and tables never hold tables.

A `Series[X]` or `Table` input gathers every row under the node's row. If nothing else decides where the node runs, it runs one level above what it gathers. A value from a row above is copied down to the rows below it. Rows of two different tables never meet by themselves: the author joins them on a key in a node, and loading says `rows_not_related` until they do.

A blank reaches a node only if its input says `| None`. Otherwise the node isn't called for that row, its outputs are blank there, and `run.not_called` says which row and input.

## Branches and pauses

A node returns `SKIPPED` on a branch it didn't take. A skip on a row leaves a gap in the column below it, and a skip above a node's rows skips the node.

A node raises `Ask` when a person must answer. That row waits, and so does whatever reads it. The rest of the graph runs on, and the leg ends `waiting`. `run.answer(...)` writes answers in, each checked before anything runs, and `run.resume()` starts the next leg. No finished unit is called again.

## Running again

`run.again(**changed)` runs with changed inputs. A unit that receives exactly what a unit of its node received before takes that earlier result, or that earlier answer, and its node isn't called. So when one row is added to a CSV of 5,000, a node that runs per row is called for that one row. A node that reads the world rather than its inputs opts out with `Policy(reuse_when_inputs_match=False)`.

## Failures

Every failure is a `Problem`: `type`, `loc`, `msg` and `ctx`. Loading reports problems, a failed node puts one on its event and on `run.cause`, and a call that can't be acted on raises `InvalidCallError` with every problem listed. A host routes and translates by `type`.

Only the outside world's failures are retried, under the node's `Policy`. A failed unit stops anything new from starting. The calls already running finish and are kept, so resuming doesn't make them again.
