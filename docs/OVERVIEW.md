# How it works

Conductor is one package, `conductor`. This page is what you need to tell, from a graph as it is drawn, how it will run. The rules in full are in the reference that ships with the package: `python -m conductor.about`.

## One signature, three readers

A node's signature is read once, when `@node` decorates the function: each parameter becomes an input, and the return type becomes the outputs. Three things use what was read. The engine calls the function. A check made from each input's and output's type validates the graph's constants, what a caller gives and what the node returns. An editor reads `describe()`: each input's and output's type, JSON schema, title and widget. Nothing is declared twice.

`@node` checks the signature too. An `async def`, a parameter without a type, or a limit that doesn't fit its type fails at import with a `NodeDeclarationError`.

## Compile, then run

`Graph.compile(record, registry)` finds each node's function at the version the graph record names, compiles each [subgraph](subgraphs.md) once on its own, checks every edge and constant, gives a node written for any type (a type variable) the type that arrives on its edge, and calls the hooks that compute inputs and outputs. Compiling never raises for a wrong graph: everything wrong goes in `graph.problems` ([problem codes](problems.md)), and a graph with problems does not run. `graph.runs_on` says how often each node runs.

`graph.run(**inputs)` starts a run and calls every node that is ready, until the run is complete, waits for a person, fails or is stopped. It returns a `Run`, whose `status` says how it stopped: `complete`, `waiting` (for a person), `failed` (a node raised), `stopped` (a series was longer than the item limit, `max_items=`), `cancelled` or `timed_out` (`timeout=` seconds ran out); a checkpoint saved while nodes are still being called says `running`. `run.resume()` goes on in the same way, and `run.stream()` does the same while it yields an event for each node that starts, finishes or fails. `run.to_json()` is the whole run, so the next `resume()` can happen in another process.

## One or many

A node's types say how many values it deals with: `text: str` is one, `Series[str]` is many, and each value in a series is an item. Conductor calls a node once, or once per item.

```text
  split ──> clean (item 0, 1, 2 …) ──> summarise (item 0, 1, 2 …) ──> join (once)
```

- `split` returns a `Series` and `clean` takes one value, so `clean` is called once per item: a loop. So is `summarise`, which is wired to `clean`. A call starts as soon as its arguments exist, so item 0 can go through the whole chain while item 9 is still being cleaned.
- `join` takes a `Series`, so it collects the loop: it waits for every item and is called once with all of them as a list, in order. A `Table` input collects records into a table the same way.
- A value from outside the loop, returned by a node upstream of `split`, is the same on every call inside it.

A node inside a loop can return a `Series` too: each document has lines of its own. That starts a second loop inside the first, and a call for one line has two item indexes, `(0, 3)`: document 0, line 3. Collecting ends the inner loop only, so a node that collects the lines is called once per document.

A table is one value, and so is a `list`. Neither starts a loop by itself: a node that returns a `Series`, such as Split in `syv-conductor-nodes`, returns the records or items one each. Two series that are not one inside the other never pair up item by item: compiling reports `series_not_nested` until one is made inside the other.

A node isn't called for an item when an argument is `None` and its input does not take `None`; `run.not_called` lists those calls ([writing nodes](writing-nodes.md#types) says when).

## Branches and pauses

A node with branches returns `SKIPPED` for the ones it did not choose. The nodes wired only to a branch that was not chosen are skipped: inside a loop for that item alone, outside a loop altogether.

A node raises `Ask` when a person must answer. That call waits, and so does every call downstream of it. The rest of the graph runs on, and the run ends `waiting`. `run.answer(...)` gives answers, each checked before anything runs, and `run.resume()` goes on. No call that finished is made again.

## Running again

`run.again(**changed)` starts a run with changed inputs and reuses the previous run: where a call would get exactly the arguments a call of the same node got in the previous run, that call's output, or a person's answer, is reused, and the node isn't called. So when one line is added to a CSV of 5,000, a node that runs per item is called for that one line. A node that reads the world rather than its arguments opts out with `Policy(reuse_when_inputs_match=False)`. After an edit to the graph itself, `edited_graph.with_inputs(..., previous=old_run).resume()` reuses the old run the same way, for every node the edit left as it was (same id, version, code and inputs and outputs, inside the same loops). So changing the last node's prompt calls only that node.

## Failures

Every failure is a `Problem`: `type`, `loc`, `msg` and `ctx`. Compiling reports problems in `graph.problems`. A node that fails puts one on its `node_failed` event and on `run.state` (an `EndedEarly`, whose `cause` is the problem). A call that can't be acted on raises `InvalidCallError` with every problem listed. A host routes and translates by `type`.

Retries are the node's `Policy` ([writing nodes](writing-nodes.md#when-something-fails)). A failed call stops any new call from starting. The calls already running on threads finish and are kept, so resuming doesn't make them again.
