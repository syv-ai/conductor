# How it works

This page explains how a graph runs, so you can look at a drawn graph and tell what it will do. It skips the fine print. The full rules are in the reference that ships with the package: `python -m conductor.about`.

## A node's signature is its interface

You write a node as a plain function with type hints. When `@node` decorates it, conductor reads the signature once. Each parameter becomes an input, and the return type becomes the outputs.

Three things use what was read:

- The engine calls the function.
- Checks built from the types validate the graph's constants, what a caller gives, and what the node returns.
- An editor reads `describe()` to show each input and output, with its type, JSON schema, title and widget.

So nothing is declared twice.

`@node` also checks the signature. An `async def`, a parameter without a type, or a limit that doesn't fit its type fails right away, at import, with a `NodeDeclarationError`.

## Compile, then run

`Graph.compile(record, registry)` turns a graph record into a `Graph`. On the way it:

- finds each node's function, at the version the record names;
- compiles each [subgraph](subgraphs.md) once, on its own;
- checks every edge and every constant;
- gives a node written for any type (a type variable) the type that arrives on its edge;
- calls the hooks that work out a node's inputs and outputs.

Compiling never raises for a broken graph. Everything wrong goes in `graph.problems` ([problem codes](problems.md)), and a graph with problems doesn't run. `graph.describe()` says how often each node will run.

`graph.run(**inputs)` starts a run. You pass the graph's inputs by the names its record gives them. Conductor then calls every node that is ready, until the run is complete, waits for a person, fails or is stopped. You get back a `Run`, and its `status` says how it stopped:

| `status` | What it means |
|---|---|
| `complete` | Every call is done. |
| `waiting` | A node is waiting for a person to answer. |
| `failed` | A node raised. |
| `stopped` | A series had more items than the limit you set with `max_items=`. |
| `cancelled` | The run was cancelled. |
| `timed_out` | The `timeout=` you gave ran out. |
| `ready` | There's a call left to make, as before the first call or once the last question is answered. |
| `running` | Only in a checkpoint taken while nodes were still being called, or in a run handed on to another process. |

`run.resume()` carries on the same way. `run.stream()` does the same, and also yields an event each time a node starts, finishes or fails. `run.to_json()` is the whole run, so the next `resume()` can happen in another process. How each status is worked out, and when it changes, is in `python -m conductor.about running`.

## One or many

A node's types say how many values it deals with. `text: str` is one value, and `Series[str]` is many. Each value in a series is an item. Conductor calls a node either once, or once per item.

```text
  split ──> clean (item 0, 1, 2 …) ──> summarise (item 0, 1, 2 …) ──> join (once)
```

- `split` returns a `Series`, and `clean` takes one value. So `clean` is called once per item. That's a loop.
- `summarise` is wired to `clean`, so it's in the loop too. A call starts as soon as its arguments exist, so item 0 can go all the way through while item 9 is still being cleaned.
- `join` takes a `Series`, so it collects the loop. It waits for every item, then it's called once with all of them as a list, in order. A `Table` input collects records into a table the same way.
- A value from outside the loop, returned by a node before `split`, is the same on every call inside it.

A node inside a loop can return a `Series` too. Say each document has lines of its own. That starts a second loop inside the first, and a call for one line has two item indexes: `(0, 3)` is document 0, line 3. Collecting ends only the inner loop, so a node that collects the lines is called once per document.

A table is one value, and so is a `list`. Neither starts a loop by itself. To loop over one, use a node that returns a `Series`, such as Split in `syv-conductor-nodes`.

Two series that aren't one inside the other never pair up item by item. Compiling reports `series_not_nested` until you put one inside the other.

When an argument is `None` and its input doesn't take `None`, the node isn't called for that item. `run.not_called` lists those calls ([writing nodes](writing-nodes.md#types) says more).

## Branches and questions

A node with branches returns `SKIPPED` for the ones it didn't choose. Nodes wired only to a branch that wasn't chosen are skipped. Inside a loop, that's for that one item. Outside a loop, they're skipped altogether.

A node raises `Ask` when a person has to answer. That call waits, and so does every call after it. The rest of the graph runs on, and the run ends `waiting`. `run.answer(...)` gives the answers, and each one is checked before anything runs. Then `run.resume()` carries on. No call that finished is made again.

## Running again

`run.again(**changed)` starts a new run with some inputs changed, and reuses what it can from the previous run. When a call would get exactly the same arguments as a call of the same node last time, conductor reuses that result, or that person's answer, and doesn't call the node. So if you add one line to a CSV of 5,000, a node that runs per line is called only for the new one.

A node whose result depends on the outside world rather than its arguments opts out with `Policy(reuse_when_inputs_match=False)`.

This works after you edit the graph too. `edited_graph.with_inputs(..., previous=old_run).resume()` reuses the old run for every node the edit left alone: the same id, version, code, inputs and outputs, inside the same loops. So changing the prompt of the last node calls only that node.

## When something fails

Every failure is a `Problem`, with a `type`, a `loc`, a `msg` and a `ctx`. A host routes and translates by `type`.

- Compiling puts its problems in `graph.problems`.
- A node that fails puts its problem on its `node_failed` event, and as the run's `cause`. If the run was cancelled by then, it ends `cancelled` instead, and the next resume makes the call again.
- A call conductor can't act on, such as a wrong answer, raises `InvalidCallError` with every problem listed.

Retries are set on the node's `Policy` ([writing nodes](writing-nodes.md#when-something-fails)). Once a call fails, no new call starts. The calls already running on threads finish and are kept, so resuming doesn't make them again.
