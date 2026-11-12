# How it works

Conductor is one package, `conductor`. This page is the picture you need to read a graph's behaviour off its drawing. The rules in full are in `python -m conductor.about`.

## One signature, three readers

A node's signature is read once, when `@node` decorates it, into one field per input and output. Three things read those fields. The engine calls the function. A check built from each field validates what is typed in, what a caller gives and what the node returns. An editor reads `describe()`: each field's type, JSON schema, title and widget. Nothing is declared twice.

`@node` checks the signature too. An `async def`, a parameter without a type, or a limit that doesn't fit its type fails at import with a `NodeDeclarationError`.

## Load, then run

`Graph.compile(record, registry)` finds each node's definition at the version the record names, loads each [subgraph](subgraphs.md) once on its own, checks every edge and value, gives a node written for any type the type that arrives on its edge, and runs the field hooks. Everything wrong goes in `graph.problems` ([problem codes](problems.md)), and `graph.runs_on` says how often each node runs.

`graph.run(**inputs)` runs one leg. A leg goes until the run is complete, waits for a person, fails or is stopped, and it returns a `Run`. `run.stream()` runs the same leg and yields events as it goes. `run.to_json()` is all of the run, so the next leg can start in another process.

## One or many

A node's types say how many values it deals with: `text: str` is one, `Series[str]` is many. The engine's unit of work is one call of a node.

```text
  split ──> clean (item 0, 1, 2 …) ──> summarise (item 0, 1, 2 …) ──> join (once)
```

- `split` returns a `Series` and `clean` takes one value, so `clean` runs once per item. So does `summarise`, which reads `clean`. A call starts as soon as what it reads exists, so item 0 can finish the whole chain while item 9 is still being made.
- `join` takes a `Series`, so it gathers: it waits for every item and runs once with all of them, in order. A `Table` input gathers records the same way.
- A value made outside the items, before `split`, is simply there on every call.

A node that runs per item can return a `Series` too: each document has lines of its own. Those lines are inside the document, and one of them is known by two numbers, `(0, 3)`. A gather goes up one level, so a node that gathers what was made per line runs once per document.

A table is one value, and so is a list. Neither makes items by itself: a node that returns a `Series`, such as Split in `syv-conductor-nodes`, gives the records or items one at a time. Items of two Series never pair up by themselves. One has to be made inside the other, and loading says `series_not_nested` until it is.

A blank can keep a node from being called for an item; `run.not_called` lists those ([writing nodes](writing-nodes.md#types) says when).

## Branches and pauses

A node returns `SKIPPED` on a branch it didn't take. A skip for one item leaves a gap there, and a skip outside the items skips the node.

A node raises `Ask` when a person must answer. That call waits, and so does whatever reads it. The rest of the graph runs on, and the leg ends `waiting`. `run.answer(...)` writes answers in, each checked before anything runs, and `run.resume()` starts the next leg. Nothing that finished is called again.

## Running again

`run.again(**changed)` runs with changed inputs. A call that would receive exactly what a call of its node received before takes that earlier result, or that earlier answer, and the node isn't called. So when one line is added to a CSV of 5,000, a node that runs per item is called for that one. A node that reads the world rather than its inputs opts out with `Policy(reuse_when_inputs_match=False)`. After an edit to the graph itself, `edited_graph.with_inputs(..., previous=old_run).resume()` lends the old run's work the same way: a node the edit left as it was (same id, version, code and fields, per item of the same Series) isn't called for an item that receives what it received, so changing the last node's prompt calls only that node.

## Failures

Every failure is a `Problem`: `type`, `loc`, `msg` and `ctx`. Loading reports problems, a failed node puts one on its event and on `run.state` (an `EndedEarly`'s `cause`), and a call that can't be acted on raises `InvalidCallError` with every problem listed. A host routes and translates by `type`.

Retries are the node's `Policy` ([writing nodes](writing-nodes.md#when-something-fails)). A failed call stops anything new from starting. The calls already running finish and are kept, so resuming doesn't make them again.
