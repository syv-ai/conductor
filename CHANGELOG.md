# Changelog

All notable changes to `syv-conductor`, `syv-conductor-nodes` and `syv-conductor-providers`,
released together at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, loading and the engine are each
replaced, and no alias keeps a 1.x name alive. Conductor depends on `polars>=2.0` and
`pydantic>=2.11`. The README walks through the design, and `python -m conductor.about` has every
rule.

What changes in how you work, before the name-by-name tables:

- **A type means what it says, wherever it is read.** `Series[T]` and `Table[R]` are ordinary generic
  types: a table stands in a union, a `Series` and a record, and `Table[R]` read anywhere holds exactly
  `R`'s columns. A node may hold type variables (`def split[T](value: Table[T] | list[T]) -> Series[T]`),
  bound from its edges when a graph loads. Generic arguments are compared (`list[int]` lands on
  `list[float]`, never on `list[str]`) and shown in a type's id. `Annotated` metadata that is no `Param`
  is pydantic's. A host type is declared with `@host_type("money", title="Money", accepts=[...])` above the class, checked when the class statement ends;
  `accepts` widens what lands on it, and a subclass of a builtin needs no schema of its own. A table's one JSON form is
  `{"columns": [{"name", "type", "values"}]}`. `syv-conductor-nodes` gains `series-stack`.

- **A field's description is one flat record.** It has `many`, `type`, `schema` (always set) and
  `type_variables`; there is no `shape` and no `columns`, a table's columns are in its type and its
  schema. A placed input and a graph input hold the field as `field`. A node's description is a
  `FunctionNodeDescription` or a `SubgraphDescription`, told apart by `kind`, and a subgraph's
  fields are listed by `registry.describe()` only. `Policy` is a pydantic model, a policy out of bounds
  raises pydantic's validation error, and `PolicyDescription` is gone.

- **Five modules.** Import from `conductor`, `conductor.widgets`, `conductor.events`,
  `conductor.testing` and `conductor.about`, the reference. Everything else is `conductor._internal`.
  A type's `accepted_as` is read from `registry.describe().types`.
- **`conductor.about` has no names.** Its `__all__` is empty. `get_content`, `list_sections` and `get_section`
  are what `python -m conductor.about` calls, and are not part of the public API.
- **What a person sees for a type is declared.** Each entry of `registry.describe().types` has the
  `id` and `title` its owner declared: conductor's own types from one fixed table (`str` is `Text`,
  `int` `Whole number`), a host's class from its `@host_type` declaration;
  nothing falls back to the class's name, and a class with no declaration is `type_not_named` where its node is declared. The types a registry holds are
  every class a field's type id names: each member of a union, a `Literal`'s value types and a
  generic's class (`list[str]` names `list`); `Any`, `None`, a type alias and a table name none.
- **A node is a typed function.** `@node("id", title=, description=)` (registers nothing) or
  `@registry.node(...)` returns a frozen, callable `FunctionDefinition` (`subgraph(...)(record)` returns a `SubgraphDefinition`,
  with no function to call; `NodeDefinition` is either). Its title and description are
  required, and the docstring never reaches a person. The signature is the interface:
  `Annotated[str, Param(title=, widget=, ge=, ...)]` for what a person sees and the limits a value
  keeps (pydantic metadata, so its limits and choices hold inside your own `BaseModel` or dataclass
  too, and a field's `schema` carries the limits but not its title or description; `Param`s stacked in one `Annotated`, an alias wrapped again, merge, the later
  winning, and a `Param` on a `Series` item is the field's; a table's column
  typed `Annotated[...]` is refused, since its limits would go unread), `-> X` for one output named `result`, a frozen dataclass for several. A
  `bool` never lands on a number, a `Literal` of numbers or a union holding a number and no
  `bool` (`int | float`) included, as a field's value or in a table's column. `async def` and
  auto-discovery are gone. A node declared wrong raises `NodeDeclarationError` at import, with a
  `code`; [`docs/problems.md`](docs/problems.md) lists them, and the parameter is typed as those
  codes, so a type checker refuses one conductor doesn't have. A `Param` is checked against its type as pydantic
  builds the schema: a limit pydantic doesn't hold on the type (`ge` on a `bool` or an `IntEnum`), or
  a choice that isn't a value of it, is `bound_invalid` or `choices_invalid`, so a host's `BaseModel`
  holding one fails as its class is made. Limits are pydantic's own: bounds on numbers, a `Decimal`,
  dates, times and durations, lengths on texts, bytes and collections. `Param` is a frozen dataclass whose limits
  are its fields, so an argument it doesn't have is Python's own `TypeError`.
- **A graph record becomes a node, a subgraph, with `subgraph`.** `subgraph("id", title=, description=)(record)`
  and `registry.subgraph(...)` take a `GraphRecord` or its JSON text and return a
  `SubgraphDefinition`. `node(...)` and `@registry.node(...)` take only a function. `subgraph` has no
  `policy`, `compute_added_inputs` or `compute_outputs` argument, so passing one is Python's own `TypeError`,
  and the declaration code `not_for_a_graph` is gone.
- **A node's title and description are not empty.** `@node`, `@registry.node`, `subgraph` and
  `registry.subgraph` raise `NodeDeclarationError` with the code `empty_title_or_description` for a
  `title` or `description` that is empty or only spaces; the message names the node and which one.
- **The types decide one or many.** A `Series[X]` output wired to an input typed for one `X` runs
  that node once per item, and everything after it; the node never sees the difference. A
  `Series[X]` input gathers every item in one call, and a `Table[R]` input gathers records into a
  table. A value made outside the items is there on every call. A table (a whole
  `polars.DataFrame`) and a list are each one value: `series-split` in `syv-conductor-nodes`, or any node
  that returns a `Series`, makes the items, and an edge may name one field of a record
  (`series-split.result.email`). A list given to an input of one value is refused (`invalid_input`), and
  so is one typed in (`invalid_value`). Items of two Series meeting at one node are refused
  (`series_not_nested`) until one is made inside the other. `graph.runs_on` says how often each node
  runs, before any run. A blank keeps the node from being called for that item unless its input
  says `| None`. A node that gathers leaves out only what was decided per item: when no item
  could bring a value, for a branch not taken or a blank outside the items, it is skipped or not
  called, with two items as with none, and `Series[X | None]` receives a blank in each item's place. Each item
  of a list given there is held to the input's limits and choices, as one value is. A choice is
  read as its answer would be: strictly from its JSON form, held to the field's limits, so an enum's
  value or a date's ISO text is a choice. A value an edge brings is held to the input's limits and choices
  alike (`invalid_edge_value`).
- **Loading never raises for a fault in the graph.** `Graph.compile(record, registry)` returns a `Graph`;
  read `graph.problems`, each a `Problem(type, loc, msg, ctx)`. Only a record of the wrong shape raises,
  `InvalidCallError` with the code `invalid_record`. A value typed in or given that its field refuses
  (`invalid_value`, `invalid_input`) carries pydantic's value-free `errors` in `ctx`. `make_graph(*nodes, edges=, constants=, defaults=)`
  writes a record from Python and loads nothing: it raises one error listing what it can't write
  (a value for a node it wasn't given, a value with no JSON form, an edge that is no address
  (`invalid_record` at `("record", "edges", index, end)`, as loading the same record says it), a
  chain it can't draw), and an input a node doesn't have, a value its field refuses, a default no
  caller reaches or an edge whose types don't meet (`type_mismatch`) is in
  `Graph.compile(record, registry).problems`. A record's `constants`, `defaults` and
  `display` are JSON values; `make_graph` writes each value typed in as its JSON, a float that isn't
  finite as `"NaN"` or `"Infinity"`, so a record equals its JSON, and an `upgrade` rewrite is written
  the same way (`invalid_value` where it has no JSON form). A subgraph offers
  its graph's inputs by their names there and its outputs by their addresses there
  (`e.upper.result`), and `subgraph` takes no `policy` or hook. Its nodes run where it stands, each with a unit per item of the node that
  embeds it, so they are called, retried, timed out, cancelled and reused by `run.again` as the
  same nodes drawn in the graph itself, and say their events once, under `("e", "upper")`. A blank
  arriving at the embedding leaves the inner nodes that read it uncalled, and `run.not_called`
  lists them. A saved run holds an inner node's units under `e/upper`. A node outside reads an output returned inside loops as it would with the same nodes drawn in the graph, and an inner decision's group
  is named by its node, `d1.answer`. A problem inside has the embedding's node (and item) in front
  of its `loc`; there is no `embedded_graph_broken`. Every node event, a `Waiting` and a
  `NotCalled` say where they are with one `loc` in place of `node_id` and an item: a unit's is a
  problem's (`("e", 0, "ask")`), a node's its id behind the embedding's (`("e", "ask")`); a waiting id inside reads `e@0/ask~3fa2c0d1`.
- **`Graph.compile` and `registry.upgrade` take a `GraphRecord` or its JSON text.** A dict is not taken: a
  host that holds one writes `GraphRecord.model_validate(data)`. `InvalidCallError` with `invalid_record`
  problems is what JSON text of the wrong shape raises.
- **A run is one value.** `graph.run(**inputs)` returns a `Run` whose `status` says how the leg
  ended, and whose `state` is `Going` or, for a leg that failed, stopped, was cancelled or timed out,
  `EndedEarly` with its `cause` (`match run.state`). It pauses for a person (`raise Ask(...)`, then `run.answer(...)` and `run.resume()`), writes
  itself as JSON (`run.to_json()`, `Run.from_json(saved, graph)`, which reads a save from its JSON text; a save that
  doesn't fit is `saved_run_unreadable`, `ctx.where` its place in the save, such as `outcomes.upper.outputs.result.1`; a save holds each node's
  units as a column per output with a slot per instance, a `Series` of records returned as a table as that table, under `e/ask` for a node inside a subgraph, and how the run stands under `state`;
  a float that isn't finite is saved as its text and read back by its declared type, the float under `float` and the text under `Any`) and runs again on changed inputs,
  calling the nodes only for the items that changed (`run.again(...)`). A leg's `max_items` is a whole number of
  at least 1 and its `timeout` and `checkpoint_every` finite seconds above 0, a `bool` neither,
  refused at the call (`invalid_max_items`, `invalid_timeout`, `invalid_checkpoint_every`).
- **A problem's `ctx` is JSON.** `Problem.ctx` is typed `Mapping[str, JsonValue]`, and
  `NodeError(ctx=)` and `ExternalError(ctx=)` read theirs where the error is made: a value with no
  JSON form (an object, a set, a tuple, a date) raises pydantic's `ValidationError` at the node's
  own `raise`, so the node fails as `execution_failed` and the run still saves. The `loc` of each
  entry in a problem's `ctx["errors"]` is a list, so a problem equals what its JSON reads back.
- **Only an outside failure is retried.** `ExternalError`, or a class in `Policy(retry_on=)`;
  a `NodeError`, a wrong output or an `Ask` never is, whatever `retry_on` names.
  A node's own exception goes to the `conductor` logger, never to an event.
- **Two companion packages, rebuilt on this API.** `syv-conductor-nodes` holds 34 nodes that only
  compute, a registry per module (`conductor_nodes.text.registry`) and all of them in
  `conductor_nodes.registry`. `series-split` takes a table or a list apart into a `Series` (a node may return a table for `Series[R]`, and a run stores that table, not a copy of its rows), `series-stack` puts the tables made per item into one, `table-collect` puts records back into a
  table with new columns (its `records` input is a `Table`, so records split from a table arrive as that table and the new columns are added beside its own, with no copy and their column types kept), `text-list` gives typed-in texts, and `text-split` is titled "Split text" and splits on a newline. A table that two outputs share (a table read and then split) is written once in a saved run: the second output is `null` there, and the save's `tables_written_once` list names the output that holds the table.
  `math-round` rounds a half away from zero (2.5 to 3), and
  `regex-replace` fails a replacement that refers to a missing group as `invalid_replacement`. `syv-conductor-providers` holds `run_router`, a FastAPI router that
  starts, follows, answers, resumes and cancels runs on several workers over a `RunStore` you
  write ([`docs/serving-runs.md`](docs/serving-runs.md)). An answer sent while a leg runs waits for it to
  end, and one that then finds nothing waiting, or a leg that needs a resume, is kept in `refused`; `again` runs on the nodes of
  the caller who asks; a `caller` dependency's type is a class's own or a function's return
  annotation, which it must have; `resume` takes no body, and a ceiling it raises holds for the run's later legs; a leg whose event
  the store fails to take ends before its next unit, so no node is called once the run reads `failed`; `GET /runs/{id}` says the `cause` of a run that
  didn't complete; a run's `outputs` there are what `run.outputs_json()` writes, each by its output's type, so a host's type that only its own
  schema can write completes over HTTP as it does in the engine, a table reads as its columns and a float that isn't finite as `"Infinity"` or `"NaN"`;
  `syv-conductor-providers` no longer depends on polars.
  There is no canvas package.
- **`syv-conductor-nodes` has a `series` category.** It holds `series-gather` ("Gather"), which runs once
  and gives the value made for every item as one list. `split` and `stack` are `series-split` and
  `series-stack` and live in `conductor_nodes.series`. `decision`, `logic-if-empty` and `logic-if-equals`
  are `control-decide`, `control-if-empty` and `control-if-equals`. A graph record that names an old id
  needs the new one, and a run saved from a graph that used an old id is refused by `Run.from_json`
  (`saved_run_of_another_graph`), because the graph's fingerprint includes the ids of its nodes.
- **A declared choice's `id` is the field type's JSON form.** `Decimal` choices show `"1"` and a `datetime` its seconds, so the id a question shows is an answer it takes.
- **A table answer and a caller's records have one reader.** Extra columns are dropped for every table from outside, and a missing column is refused. An answer's cells are read strictly from their JSON form; a caller's records are read cell by cell, laxly, never from polars' dtype, so a bool among ints or a number among texts is `invalid_input`. A value that is neither a frame nor a list of records is refused as `invalid_answer` or `invalid_input` with the reason "a table is a frame or a list of records, one object per row".
- **A default no caller reaches is a problem at load.** `make_graph(..., defaults={"pad.width": 8})` on a node that is not an open input of the graph writes the record, and `Graph.compile(record, registry).problems` holds `default_on_later_node`.
- **A run's results are what a node reading them would be given.** For an output made per item, `run.results`, `run.outputs` and `outputs_json` hold `null` where the Series was never made for a blank outside the items, or a branch there wasn't taken (they held `[]`, the same as for a Series made with no items), and `null` where every item is skipped from outside (they held a list of `null`s). A blank outside still shows a blank in each place. The same nodes inside a subgraph show the same.
- **An output made inside two loops is shown as a list of lists.** `run.results`, `run.outputs` and `outputs_json` show the output of a node called per item of a list that is itself made per item as one list per outer item, `null` for an outer item whose inner list was never made (it was one flat list over every item): three documents split into lines, the second empty, show `[["A", "B"], null, ["C"]]`. The same nodes inside a subgraph show the same.
- **A node outside a subgraph reads its outputs as it would read the same nodes drawn in its own graph.** For the subgraph `documents -> lines -> shout`, a `Series[str]` input wired to `shout.result` is called once per document with that document's lines (it was called once, with one flat list over every document), and an input of one `str` once per line; a node outside and `run.results` see the same list per document. A loop outside is named by the list inside that is looped over (`runs_on` says `per item of e.documents.result`; it named the offered output), so outputs returned in the same loop inside line up: one item of each of two of them reaches one call (it was `series_not_nested`), and a `Series` input beside an input of one value wired to the same per-item output receives that item's one value, as drawn flat (it received every item). The output is described as `many` with one `Series` per loop (`type` `Series[str]` for a list per document) and lists its `loops`; every field description gains `loops`, empty but here. A `Series` returned once per document is `Series[Series[T]]` too (it was a `Series` of whole lists). `series_not_nested` for a list inside a subgraph has `ctx.variant` `inside_subgraph` and tells the author what they can do.
- **`Run.from_json` takes the JSON text `run.to_json()` wrote**, as `str` or `bytes`. It took a mapping too; a host that holds one writes `Run.from_json(json.dumps(saved), graph)`.
- **`run.results` reads a run in any state and never raises.** A run that failed because an edge named a field its record lacks (`invalid_edge_value`) shows a blank where that field was read; the failure is the run's cause.
- **A run's item ceiling is raised in one way: `run.with_max_items(n)`.** It gives the same run under the ceiling `n`, with nothing started, so `run.with_max_items(100).resume()` goes on with a run that stopped at its ceiling. `run.resume` and `run.stream` take no `max_items`; `graph.run`, `graph.with_inputs`, `run.with_inputs` and `run.again` set it where a run is made. The router's `POST /runs/{id}/resume` stores the run with the raised ceiling as it claims it, so a leg killed before its first checkpoint is gone on with under the raised ceiling.
- **A subgraph costs what its nodes cost.** Its node holds no result of its own: what reads one of its outputs reads the node inside that makes it, and `results`, the events and `not_called` say of it what the nodes inside came to. Two `Series` inputs fed by two outputs it makes per item of one Series inside are kept in step, as they are drawn flat. A save's `complete` lists function nodes only, the subgraph's node says no `node_progress`, and one with no nodes inside says no events.
- **A `Series` input fed one value made where its node runs receives that value, as a Series of one.** A node that runs per document, because another input gathers the document's lines, and whose `Series[str]` input is wired to the document's summary, receives its own document's summary. It received every document's summaries on each call, and the same nodes inside a subgraph received one.
- **`run.not_called` is a list of `NotCalled` records** (`loc`, `input`), exported from `conductor`, not a list of dicts. It is in item order on a live run, in a checkpoint and after a load.
- **A graph with problems is still drawn.** `graph.describe()` gives a node with a problem its fields and where it runs (`found` true), and the nodes that read it too; an input whose edge is refused is fed `unset`, a mode `Feed.mode` gains. `loaded` is false only where a node's fields aren't known.
- **Faults say more.** `invalid_edge_value` carries `ctx.record_field` when an edge names a field its record lacks. A hook that returns no mapping is `compute_hook_failed` with `error: not_a_mapping`; an `AttributeError` in a hook's hint is the host's own bug and raises out of `load`. `table-collect` fails with `columns_not_aligned` (`column`, `records`, `values`) when a column doesn't hold one value per record.
- **`SKIPPED` returned is a result.** A function that returns `SKIPPED` was called: `again` reuses it and a save writes its unit as `done`. `node_skipped` is said for a node whose every unit is a branch not taken, a decision whose every output is `SKIPPED` included; a node with no output that returns `SKIPPED` says `node_complete`.
- **A run handed back at shutdown has no `cause`.** The row's `cause` is null and its status `running`: the run did not end early, so it is `Going`. The run saved in the row says `running` too: the router hands a leg back with `run.stream(hand_back=)`, where it cancelled the leg and stored a run that said `cancelled`.
- **The runs router takes an answer whenever the run does.** `POST /runs/{id}/answers` on a run that `failed`, `stopped`, `timed_out` or was `cancelled` gives the answers to `run.answer` and starts the next leg, so one item that keeps failing doesn't make the questions beside it unanswerable; what the run refuses is a 422 with its problems (`not_waiting`, `invalid_answer`), and only a `complete` run is a 409. Answers queued during a leg are given to the run that leg ended with, whatever its status; a leg that raised keeps them queued. The code `run_not_waiting` is gone.
- **A leg's end is stored once, when the router has decided what it is.** A run handed back at shutdown stores no `run_end` (it read `cancelled`, for a run nobody cancelled that another worker then finished), so a follower's stream goes on into the next worker's events. A leg that raised outside a node reads `failed` with a cause, `leg_failed`, in `GET /runs/{id}` and on its `run_end` (the cause was null). The router's own codes are a catalogue, `conductor_providers.fastapi.ROUTER_PROBLEMS`, a function per code as conductor's `PROBLEMS` is.
- **Every refusal of the runs router is a list of problems.** A 404 and a 409 are `{"detail": [{type, loc, msg, ctx}]}` as a 422 is (they were an English sentence), with the router's own codes: `no_such_run`, `already_answered` (`ctx.waiting_id`), `run_complete`, `run_not_resumable` (`ctx.status`) and `run_changed`. The OpenAPI schema declares both, as `Refused`.
- **`run_router` has no default for who may see a run.** With a `caller`, `may_see(caller, started_by)` is required, and a router made without it raises `TypeError`. The default compared the whole caller as stored with the run, so a caller whose record changed (a `last_login`) got a 404 on their own run. Without a `caller` every request sees every run, as before.
- **A stored run's `lease_until` is null when no leg is wanted,** and a time for a run a worker holds or handed back, so a store's `expired(now)` is `lease_until < now` with no status clause, and a store that forgets the status can't re-run finished runs. `RunStore.add_event` returns whether it appended, not a number.
- **A stored run is ten columns.** `StoredRun` and the table a host writes for it keep `id`, `graph`, `caller`, `run`, `status`, `rev`, `owner` and `lease_until`, and two texts hold what the router alone reads: `state` (how the run stands: why it ended early, the answers queued and refused, a cancel that was asked; small whatever the run holds) and `outcome` (what the last leg left for a reader: what waits and the outputs). The columns `waiting`, `outputs`, `cancel`, `pending`, `refused` and `cause` are gone, and `RunChanges` is the six columns that change (`run`, `status`, `state`, `outcome`, `owner`, `lease_until`). `status` is written with `state`, from it, so a host's query on `status` never disagrees with what `GET /runs/{id}` says. The store on `docs/serving-runs.md` is the six methods with no per-column conversion.
- **A run whose leg raised lists what waits in the run as stored.** `GET /runs/{id}` on a run that reads `failed` because its worker raised lists as `waiting` what waits in its last checkpoint, the run an answer is given to (it listed what waited before the leg, some of it answered since); a run that could not be loaded lists nothing. A raise from the `RunStore` fails no run: the leg is given up and a worker goes on from the last checkpoint once the lease runs out.
- **A store that fails for one statement costs the runs router one try.** A sweep for runs whose lease ran out, a take-over of one of them and a renewal of a lease are each logged with their traceback when they raise, and the next one runs (a sweep that raised once took over nothing for the worker's life, and a renewal that raised once let another worker take a healthy leg and call its units again).
- **The worker behind the runs router is a class with no HTTP in it.** `conductor_providers.fastapi.worker.Worker` begins, answers, resumes and cancels runs and takes over those whose lease ran out; `run_router` makes one and its routes call it. A host calls `run_router` as before.
- **The runs router's schema reads as a generated client needs it.** Its operations are `list_nodes`, `load_graph`, `start_run`, `start_published`, `read_run`, `follow_run`, `answer_run`, `resume_run`, `run_again` and `cancel_run`, each with a description; the body of a resume is the component `Resume`; the three free-form bodies say what they hold. `conductor_providers.fastapi` exports `run_router`, `RunStore`, `StoredRun`, `RunChanges` and `ROUTER_PROBLEMS`; the wire models (`RunView`, `StartRun`, `Started`, `Answered`, `Cancelled`) are read in the schema and no longer exported.
- **A kind of problem is declared by one function.** `ProblemCatalogue().problem()` over a function makes a `ProblemKind`: the function's name is the code, a keyword-only parameter is a `ctx` key (one that defaults to `None` may be left out), a named positional parameter is a place in `loc` and a `ctx` key, `*loc` is a place the sentence doesn't read, and the function returns the English. Calling the kind makes the `Problem`, so a type checker refuses a misspelled code, a `ctx` key left out and a value of the wrong type where the problem is made; a signature outside those three rules is a `TypeError` where it is declared. A kind has its `code`, `hint`, `sentence` (the English with a slot per key) and `ctx_type` (a `TypedDict`). The reference (`python -m conductor.about`) prints each kind from its function: the sentence, and each `ctx` key with its type, whether it is always given and what it holds; `python -m conductor.about not-a-chain` prints one. `check_registry`'s problems are declared the same way and listed in the reference, and each holds in `ctx` what its sentence is made from: `reason` (the sentences of the problems behind it), and `newest`, `status` or `named` where the sentence says them. `PROBLEMS` is the catalogue of conductor's core, every code it writes; a host reads every code as `PROBLEMS.codes`, and `conductor.PROBLEM_CODES` is gone. Every kind's `sentence` can be filled with `sentence.format(**problem.ctx)`; `context_missing`'s ended `pass context={<class>: ...}.`, whose braces `str.format` refused, and now ends `pass context with a value for <class>.` A problem whose sentence names its node or field holds them in `ctx` too, as `node_id` and `field`, so a host fills its own sentence from `ctx` alone: the id is the node's in the graph that holds it, as the sentence says it, where `loc` of a problem inside a subgraph has the embedding node in front. A waiting id and a saved run's `code` digests differ from an earlier build's for a node that runs per item (the digest is made of the scope's addresses as text). Two older keys that used those words for something else are renamed: `edge_names_missing_node` names the node that isn't there as `ctx.missing`, and `unknown_output` and `invalid_edge_value` name the field of a record an edge reads as `ctx.record_field`.
- **A leg doesn't hang on Python 3.12 when a timer falls due as it starts to wait.** CPython 3.12's `SimpleQueue.get(timeout=)` never returns when its timeout runs out before its first look at an empty queue, so a run with a retry or an attempt deadline due within about a microsecond of the leg's wait stood still for ever (3.13 and 3.14 don't do it). The leg looks at its queue once without waiting before it waits with a timeout.
- **A call on a thread that raises past the engine ends the run with it.** A node with `concurrency` above 1 or an `attempt_timeout` that raised `SystemExit` or `KeyboardInterrupt`, or whose output type's own check raised what is no refusal, was kept by its thread: the leg waited for the call for ever, and a stream of it couldn't be cancelled. The exception is handed back and raised from the call that ran the leg, as it is for a node called on the leg's own thread.

A 1.x graph doesn't load as saved. Convert each graph record with the table under
"Converting a stored 1.x graph", then save the `GraphRecord`.

### From 1.12.2, name by name

Every public name of 1.12.2, by the module it was imported from. A name not listed under 2.0 is gone.

| 1.12.2 | 2.0 |
|---|---|
| **Nodes** | |
| `Actor` | gone: a node declares no actor; the host decides who runs what |
| `BaseNode` | a plain function under `@node("id", title=, description=)` or `@registry.node(...)` |
| `conductor.metadata.InputMetadata` | `FieldDescription`, an entry of `definition.describe().inputs`; the hint and its `Param(...)` say it |
| `conductor.metadata.OutputMetadata` | `FieldDescription`, an entry of `definition.describe().outputs` |
| `conductor.registry.dynamic_inputs.ComputeInputsContext` | gone: a `compute_added_inputs` hook is a plain function whose parameters name the inputs it reads |
| `conductor.registry.dynamic_inputs.ComputeInputsFn` | `compute_added_inputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.ComputeOutputsContext` | gone: a `compute_outputs` hook's parameters name what it reads; one annotated `type` receives the arriving type |
| `conductor.registry.dynamic_outputs.ComputeOutputsFn` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.IncomingBinding` | gone: a hook parameter annotated `type` receives the type arriving on that input |
| `conductor.registry.dynamic_outputs.strip_sub_output_prefix` | gone: an output is named by its field |
| `conductor.registry.NodeDefinition` | `NodeDefinition`, the union of `FunctionDefinition` (frozen and callable, returned by `@node`) and `SubgraphDefinition` (returned by `subgraph`) |
| `conductor.types.Base64Str` | gone: any type pydantic carries travels; `bytes` is one |
| `conductor.types.Date` | `datetime.date` |
| `conductor.types.MultiNamedFile` | gone: `Series[YourFile]` |
| `conductor.types.NamedFile` | gone: declare your own pydantic model or dataclass |
| `conductor.types.NodeResult` | gone: a node returns its value; a run's are `run.results` and `run.outputs` |
| `conductor.types.OUTPUT_PREFIX` | gone: outputs are named by their fields |
| `conductor.types.RESULT_KEY` | an output of `-> X` is named `result`; a frozen dataclass names several |
| `FlowStore` | a parameter typed `Context[T]`, filled from `context={T: value}` |
| `NodeCategory` | a plain string, `@node("id", title=, description=, category="text")` |
| `NodeRegistry` | `NodeRegistry(category=, tags=, extra=, types=)`, empty until nodes are added |
| `ResultFormat` | gone: a node returns its value, typed by its return hint |
| `RetryConfig` | `Policy(retries=, retry_delay=, attempt_timeout=, retry_on=)` on `@node(policy=)` |
| `SKIPPED` | `SKIPPED`, returned on a branch not taken |
| **Widgets** | |
| `conductor.widgets.Checkbox` | `Switch` |
| `conductor.widgets.CodeEditor` | gone: a host subclasses `Widget` |
| `conductor.widgets.ColumnSelect` | gone: an edge names one field of a record, `node.output.field`, after a node that gives a table's records one at a time (`series-split`) |
| `conductor.widgets.ConditionBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.ConnectionList` | gone: a `Series[X]` input takes every edge into it |
| `conductor.widgets.DatePicker` | `DatePicker` |
| `conductor.widgets.DependentDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.Dropdown` | `Dropdown`, which takes no choices: they live on the field, `Param(choices=[Choice(id, title)])` or a `Literal` |
| `conductor.widgets.EntityDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.FileUpload` | `FileUpload`, without `multiple`: a `Series` field takes several files |
| `conductor.widgets.HumanReview` | `raise Ask(...)` in the node; its return type says what is asked |
| `conductor.widgets.IfElseBuilder` | gone with the if-else node: outputs of one `Param(exclusive_group=)` are the branches |
| `conductor.widgets.List` | `ListInput` |
| `conductor.widgets.Multiselect` | an input typed `Series[Literal[...]]`, or `Dropdown` on a `Series` |
| `conductor.widgets.Number` | `NumberInput`, without `integer_only`: an `int` field takes whole numbers |
| `conductor.widgets.Output` | gone: an output has no widget |
| `conductor.widgets.Range` | `Slider`; its limits go on `Param(ge=, le=)` |
| `conductor.widgets.SchemaBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.Switch` | `Switch` |
| `conductor.widgets.TableInput` | an input typed `Table[R]`, the table typed in as records |
| `conductor.widgets.TableSource` | an input typed `Table[R]` |
| `conductor.widgets.Tags` | gone: an input typed `Series[str]` |
| `conductor.widgets.TemplateTextarea` | gone: a host subclasses `Widget` |
| `conductor.widgets.Text` | `TextInput` |
| `conductor.widgets.Textarea` | `Textarea` |
| `conductor.widgets.WIDGET_SCHEMA_KEYS` | gone: a widget is a pydantic model; its fields are its keys |
| `conductor.widgets.Widget` | `Widget`; a host subclasses it for a control of its own |
| `widgets` | `conductor.widgets`, imported as a module |
| `WidgetType` | a widget's `type` field |
| **Graphs and loading** | |
| `compile` | `Graph.compile(record, registry)`, which returns a `Graph` and never raises for a fault in the graph |
| `CompiledGraph` | `Graph` |
| `conductor.expr`: `Expression`, `ExpressionError`, `ExpressionParseError`, `ExpressionRuntimeError`, `ExpressionTypeError`, `evaluate`, `parse` | gone with CEL guards on edges |
| `conductor.flow_format.dump_flow` | `record.model_dump_json()` |
| `conductor.flow_format.flow_to_dict` | `record.model_dump(mode="json")` |
| `conductor.flow_format.flow_to_yaml` | gone: YAML isn't written |
| `conductor.flow_format.load_flow_from_path` | gone: read the file and `Graph.compile` its text |
| `conductor.flow_format.load_flow` | `Graph.compile(json_text, registry)` or `GraphRecord.model_validate_json` |
| `conductor.flow_format.yaml_to_flow` | gone: YAML isn't read |
| `conductor.graph.compiler.DecisionGuard` | gone with CEL guards: outputs of one `exclusive_group` are the branches |
| `conductor.graph.compiler.ExtensionResolver` | gone: a graph record is a node, `subgraph("id", title=, description=)(record)`, added to a registry |
| `conductor.graph.dynamic_inputs.resolve_node_inputs` | `Graph.compile(record, registry).describe()`, a `GraphDescription` with each node's fields as its hooks made them |
| `DefinitionSource` | gone with `RegistryView` |
| `expr` | gone with CEL guards on edges |
| `Flow` | `GraphRecord`; see "Converting a stored 1.x graph" |
| `FlowDependency` | gone: a graph embeds another as a node, `subgraph("id", title=, description=)(record)` |
| `FlowTrigger` | gone: the host decides when a graph runs |
| `GraphEdge` | a pair `("a.result", "b.text")` in `GraphRecord.edges` |
| `GraphNode` | `NodeRecord`; see "Converting a stored 1.x graph" |
| `RegistryView` | `registry.filter(predicate)` and `add_subregistry`; a request's own nodes go in a registry of their own |
| `resolve_graph_inputs` | `Graph.compile(record, registry).inputs`, each a `GraphInput` |
| `resolve_graph_outputs` | `Graph.compile(record, registry).outputs` |
| **Running** | |
| `compute_for_each_end_outputs` | gone: a `Series[X]` input gathers the items |
| `conductor.compound.CompoundNodeType` | gone with compound nodes |
| `conductor.compound.NodeExecutor` | gone with compound nodes |
| `conductor.compound.Region` | gone with regions |
| `conductor.execution.engine.collect` | `graph.run(...)`, then `run.results` |
| `conductor.execution.events.CompensationCompleteEvent` | gone with compensation |
| `conductor.execution.events.CompensationFailedEvent` | gone with compensation |
| `conductor.execution.events.CompensationStartEvent` | gone with compensation |
| `conductor.execution.events.EventSink` | `async for event in run.stream()` |
| `conductor.execution.events.ExecutionEvent` | `Event`, a union told apart by `type` |
| `conductor.execution.events.FlowCancelledEvent` | `RunEnd` with `status` `cancelled` |
| `conductor.execution.events.FlowCompleteEvent` | `RunEnd` with `status` `complete` |
| `conductor.execution.events.FlowErrorEvent` | `RunEnd` with `status` `failed` |
| `conductor.execution.events.FlowPausedEvent` | `NodeWaiting` per waiting call, then `RunEnd` with `status` `waiting` |
| `conductor.execution.events.FlowTimeoutEvent` | `RunEnd` with `status` `timed_out` |
| `conductor.execution.events.NodeCompleteEvent` | `NodeComplete` |
| `conductor.execution.events.NodeErrorEvent` | `NodeFailed`, its `cause` a `Problem` |
| `conductor.execution.events.NodeProgressEvent` | `NodeProgress` |
| `conductor.execution.events.NodeRetryEvent` | `NodeRetry` |
| `conductor.execution.events.NodeSkippedEvent` | `NodeSkipped` |
| `conductor.execution.events.NodeStartEvent` | `NodeStart` |
| `conductor.execution.events.RuntimeWarningEvent` | gone: a type fault is a load problem, `graph.problems` |
| `conductor.execution.events.SignalWaitingEvent` | `NodeWaiting` |
| `conductor.execution.resolver.finalize_connection_labels` | gone with connection labels: an input is named by its field |
| `conductor.execution.resolver.InputResolver` | gone: the graph reads its inputs, `graph.with_inputs(...)` |
| `conductor.execution.resolver.MULTI_VALUE_SEPARATOR` | gone: several edges into a `Series` input arrive as a list, never joined |
| `conductor.execution.results.extract_output` | `run.results[node_id][output]` |
| `conductor.execution.results.filter_all_skipped` | gone: a skipped item is a blank in `run.results` |
| `conductor.execution.results.filter_skipped` | gone: a skipped item is a blank in `run.results` |
| `conductor.execution.results.normalize_result` | gone: what a node returns is checked against its type |
| `conductor.execution.results.OutputRef` | an address, `node.output` |
| `conductor.execution.results.project_outputs` | `run.outputs`, by address |
| `conductor.execution.state.FlowRunState` | `Run`: `status`, `results`, `waiting`, `state` (`Going`, or `EndedEarly` with its `cause`) |
| `execute_sync` | `graph.run(**inputs)` |
| `execute` | `graph.run(**inputs)` returns a `Run`; `run.stream()` yields its events |
| `FlowCheckpoint` | a `Run`: `run.to_json()` and `Run.from_json(saved, graph)`; mid-leg, `run.stream(checkpoint_every=)` yields `RunCheckpoint` |
| `FOR_EACH` | gone with the for-each node |
| `ForEachNode` | gone: a `Series` output wired to an input of one value runs the node once per item |
| `resume_sync` | `run.resume()` |
| `resume` | `run.answer(...)`, then `run.resume()` |
| `SUBPROCESS` | gone with the subprocess node |
| `SubprocessNode` | a graph record is a node, `subgraph("id", title=, description=)(record)` |
| `SubprocessRegistry` | gone: a subgraph's node goes in a registry like any other |
| `WHILE` | gone with the while node |
| `WhileNode` | gone: a graph has no cycles |
| **Errors** | |
| `CompilationError` | gone: loading never raises; read `graph.problems` |
| `conductor.errors.CycleDetectionError` | the problem `cycle` in `graph.problems` |
| `conductor.errors.FlowExecutionException` | gone, as `FlowExecutionError` |
| `conductor.errors.FlowPausedException` | gone, as `FlowPausedError` |
| `conductor.errors.InputResolutionError` | `InvalidCallError` at `graph.run`, listing `missing_input`, `not_offered`, `invalid_input` |
| `conductor.errors.NodeExecutionException` | `NodeError` |
| `conductor.errors.NodeValidationException` | `NodeError` |
| `conductor.errors.TypeCheckError` | the problem `type_mismatch` in `graph.problems` |
| `ConductorError` | gone: catch `NodeDeclarationError`, `InvalidCallError` or `NodeError` |
| `errors` | gone as a module: the classes import from `conductor` |
| `FlowExecutionError` | gone: a failed leg returns its `Run`, its `state` an `EndedEarly` whose `status` is `failed` and `cause` a `Problem` |
| `FlowPausedError` | gone: a paused leg returns its `Run`, `status` `waiting` |
| `HumanInputRequired` | `raise Ask(prompt, choices=, default=)`; each has the answer's shape, so on several outputs each goes by output name; a question's widget is its output type's `Param` |
| `LoopRunawayError` | `graph.run(max_items=)`, the problem `item_limit` |
| `NodeConnectionError` | `ExternalError`, the one failure that is retried |
| `NodeError` | `NodeError(message, code=, ctx=)`, never retried |
| `NodeExecutionError` | `NodeError(message, code=, ctx=)` |
| `NodeTimeoutError` | `Policy(attempt_timeout=)`; the unit fails with the code `timeout` |
| `NodeValidationError` | `NodeError(message, code=, ctx=)` |
| `SignalRequired` | `raise Ask(...)` |
| `SubprocessFailedError` | the failing inner node's `node_failed`; its `cause.loc` starts with the embedding's node and item |
| **The palette** | |
| `conductor.registry.schema.serialize_input_model` | `FieldDescription` |
| `conductor.registry.schema.serialize_input` | an entry of `definition.describe().inputs`, a `FieldDescription` |
| `conductor.registry.schema.serialize_node_model` | `definition.describe()`, a `NodeDescription` |
| `conductor.registry.schema.serialize_node` | `definition.describe()`, a `NodeDescription`; `.model_dump(mode="json")` for a dict |
| `conductor.registry.schema.serialize_output_model` | `FieldDescription` |
| `conductor.registry.schema.serialize_output` | an entry of `definition.describe().outputs`, a `FieldDescription` |
| `conductor.registry.schema.serialize_registry` | `registry.describe()`, a `RegistryDescription` |
| `conductor.registry.serialized.SerializedInput` | `FieldDescription` |
| `conductor.registry.serialized.SerializedNode` | `NodeDescription` |
| `conductor.registry.serialized.SerializedOutput` | `FieldDescription` |
| **Packages** | |
| `conductor.about.get_content` | `get_content`, unchanged |
| `conductor.about.get_section` | `get_section`, unchanged |
| `conductor.about.list_sections` | `list_sections`, unchanged |
| `conductor_nodes.control_operators`: `OPERATORS`, `OPERATORS_BY_ID`, `OPERATORS_BY_LABEL`, `OPERATORS_BY_TYPE`, `OperatorEvaluationError`, `OperatorSpec`, `UNARY_OPERATORS` | gone: a condition is a node that returns a `bool` (`text-contains`, `regex-match`), wired into `control-decide.when` |
| `conductor_nodes`: `get_default_registry`, `register_all` | `conductor_nodes.registry`, added with `registry.add_subregistry(...)` |
| `conductor_nodes`: `CATEGORIES` | gone: each node has its `category`; `conductor_nodes.registry.filter(lambda d: d.category == "math")` |
| `conductor_nodes`: `decision`, `json_ops`, `logic`, `math`, `regex_ops`, `text` | a module per category, each with a `registry` of its own nodes: `control` (`control-decide`, `control-if-empty`, `control-if-equals`), `json_ops`, `logic` (`logic-not`), `math`, `regex_ops`, `text`; the ids of `decision`, `logic-if-empty` and `logic-if-equals` are the new ones, and the other node ids are unchanged |
| `conductor_nodes`: `loop`, `while_loop`, `signal`, `subprocess` | gone: a node typed for one value runs once per item of a `Series` wired to it, and a pause is `raise Ask(...)` |
| `conductor_providers.fastapi.conductor_router` | `conductor_providers.fastapi.run_router(store=, registry=, caller=, may_see=)`: runs by id, kept in a `RunStore`, on several workers; `/nodes` is `registry.describe()`, `/compile` is `POST /load` (`graph.describe()`), `/execute-stream` is `POST /runs` and `GET /runs/{id}/events` |
| `conductor_providers.fastapi`: `ExecuteRequest`, `NodeInput`, `EdgeInput` | `StartRun`, a `GraphRecord` and its inputs |
| `conductor_providers.fastapi`: `CompileResult`, `CompileWarning` | `GraphDescription`, whose problems name each node |
| `conductor_providers.fastapi.sse_frame` | gone: each frame is an event's own JSON |
| `conductor_providers.fastapi` (`context_factory=`, `entity_resolver=`) | `context=`, built from the caller who started the run; an entity list is a route of your own |
| `conductor_providers.react`: `graph_to_react`, `palette_from_registry`, `react_to_graph`, `topological_positions` | gone: a canvas stores the `GraphRecord` as it is, a node's position in `display` |
| `conductor_providers`: `PROVIDERS`, `react` | gone |

### `@registry.node(...)` arguments

| 1.12.2 | 2.0 |
|---|---|
| `base_id` | the id, first and required: `@node("upper", title=, description=)` |
| `version` | `version=`, one past the version replaced, or 1 |
| `name` | `title=`, required |
| `description` | `description=`, required; the docstring is never shown |
| `tags` | `tags=`; a registry's tags add to them |
| `category` | `category=`, a string |
| `max_retries` | `Policy(retries=)` |
| `retry_delay` | `Policy(retry_delay=)` |
| `width` | gone: how a node is drawn is the host's, `NodeRecord.display` |
| `docs` | `docs=` |
| `actor` | gone with `Actor` |
| `timeout` | `Policy(attempt_timeout=)` |
| `idempotency_key` | gone: `again` reuses a unit whose inputs match, `Policy(reuse_when_inputs_match=)` |
| `uses` | gone: a parameter typed `Context[T]` says what the node needs from the run |
| `is_decision` | `Param(exclusive_group=)` on the outputs that are branches |
| `is_signal` | gone: `raise Ask(...)` |
| `dynamic_handles` | gone: `compute_added_inputs=` and `compute_outputs=` |
| `compute_outputs` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `compute_inputs` | `compute_added_inputs=`, the inputs it adds, taken as `**kwargs` |

### Converting a stored 1.x graph

A 1.x `Flow` becomes a `GraphRecord`, `{"nodes": [...], "edges": [[from, to], ...]}`. Each `GraphNode` becomes a `NodeRecord`, `{id, node, version, values, defaults, display}`, empty fields left out. Each `GraphEdge` becomes a pair of addresses, `["source.source_handle", "target.target_handle"]`. Load the result and read `graph.problems` to see what didn't carry over. A 1.x for-each over a list is gone: give the list to the input that took one item, and the node runs once per item; a node after it that collected the items takes them through a `Series` input.

| 1.x field | 2.0 |
|---|---|
| `GraphNode.id` | `id`, without `.`, `/` or `@` |
| `GraphNode.type` | `node` (the id of the node it uses) and `version`, 1 when left out |
| `GraphNode.data` | `constants` for what is fixed, `defaults` for what a caller may change; an input wired by an edge has neither |
| `GraphNode.produces` | gone: the node's outputs are its return type's |
| `GraphNode.consumes` | gone: the node's inputs are its signature's |
| `GraphNode.compensation` | gone with compensation |
| `GraphNode.on_error` | gone: a failed unit ends the leg `failed` |
| `GraphNode.node_label` | `display`, the host's own |
| `GraphNode.output_labels` | `display`, or a `Param(title=)` on the output |
| `GraphEdge.id` | gone: an edge is its two addresses |
| `GraphEdge.source` | the first address's node: `source.output` |
| `GraphEdge.target` | the second address's node |
| `GraphEdge.source_handle` | the first address's field, `source.output`, or `source.output.field` for one field of a record of a `Series` (a table's column is no address: `series-split` gives its records one at a time) |
| `GraphEdge.target_handle` | the second address's field, `target.input` |
| `GraphEdge.when` | gone: a branch is an output of one `exclusive_group`; wire from that output |
| `GraphEdge.priority` | gone: edges are kept sorted; several into one input are alternatives, opposite branches of one decision |
| `Flow.nodes` | `nodes` |
| `Flow.edges` | `edges`, each `["a.result", "b.text"]` |
| `Flow.id` | not stored in the record: the host keys its rows |
| `Flow.version` | not stored: each node has its own `version` |
| `Flow.name` | not stored: the host's own, or the embedding node's `title` |
| `Flow.description` | not stored: the host's own, or the embedding node's `description` |
| `Flow.dependencies` | gone: a subgraph is a node added to the registry |
| `Flow.triggers` | gone: the host decides when a graph runs |
| `Flow.on_error_default` | gone: a failed unit ends the leg `failed` |

### Events

Events are frozen pydantic models in `conductor.events`, told apart by `type`. Two are new: `run_start`, always first, and `run_checkpoint`, the run mid-leg for a host to store, sent when `run.stream(checkpoint_every=)` asks. `run_end` is always last (but for a leg its host hands back with `run.stream(hand_back=event)`, which leaves the run `running` and ends with a `run_checkpoint` of it), its `status` one of `complete`, `waiting`, `failed`, `stopped`, `cancelled`, `timed_out`, and its `cause` the run's `Problem` for one that failed, stopped, was cancelled or timed out (`null` otherwise).

| 1.12.2 `type` | 2.0 |
|---|---|
| `node_start` | `node_start` |
| `node_complete` | `node_complete`: `outcomes`, `not_called`, `reused`, counts only; the values are `run.results` |
| `node_skipped` | `node_skipped` |
| `node_error` | `node_failed`, `cause` a `Problem` |
| `node_progress` | `node_progress`: `done`, `total` |
| `runtime_warning` | gone: a type fault is a load problem |
| `flow_complete` | `run_end`, `status` `complete` |
| `flow_error` | `run_end`, `status` `failed` |
| `flow_cancelled` | `run_end`, `status` `cancelled` |
| `flow_timeout` | `run_end`, `status` `timed_out` |
| `flow_paused` | `node_waiting` per waiting call, then `run_end`, `status` `waiting` |
| `node_retry` | `node_retry`: `loc`, `attempt`, `retries`, `delay`, `cause` |
| `compensation_start` | gone with compensation |
| `compensation_complete` | gone with compensation |
| `compensation_failed` | gone with compensation |
| `signal_waiting` | `node_waiting` |

## [1.12.2]

### Fixed

- The validation model no longer turns `**kwargs` into a required field, and
  a callable declaring it now accepts extra keys. `1.12.1` stopped the
  registry introspecting `**kwargs` as an input, but `create_validation_model`
  builds from the signature independently and still declared a required field
  named `kwargs` — so a `compute_inputs` node failed with
  `Invalid inputs — 'kwargs': Field required` before it ever ran. Worse, the
  model was built with pydantic's default `extra="ignore"` while the engine
  assigns `validated.model_dump()` back over the inputs, so once that field
  was satisfied every hook-declared handle was silently dropped and the node
  ran with nothing wired. `*args` is skipped for the same reason.

## [1.12.1]

### Fixed

- `**kwargs` / `*args` are no longer introspected as input handles. A node
  whose handles come from `compute_inputs` must declare `**kwargs` for the
  resolver's values to reach it; registering that as an input named
  `kwargs` gave it a Text widget that rendered as a stray field on the
  node, put a phantom handle in the palette payload, and tripped host-side
  checks that every declared handle be documented. Delivery is unaffected —
  `_filter_to_signature` reads the signature directly, not this metadata.

## [1.12.0]

### Added

- `compute_inputs` hook — a node can derive typed input handles from its
  instance `data`, mirroring `compute_outputs`. Accepted by both
  `@registry.node(...)` and `@category.node(...)`. New public types
  `ComputeInputsContext` / `ComputeInputsFn` in
  `conductor.registry.dynamic_inputs`, and `resolve_graph_inputs` exported
  from `conductor` as the ahead-of-compile counterpart to
  `resolve_graph_outputs`.
- `CompiledGraph.node_inputs` — the resolved input roster per node id,
  populated for every node.
- `has_dynamic_inputs` in the serialized palette payload, matching
  `has_dynamic_outputs`.

### Changed

- Edge type-checking, consume validation and validation-error labelling
  consult the resolved input roster in preference to the static schema.
  Nodes without a hook are unaffected. Note this makes an edge into a
  hook-declared handle *type-checked* where an unknown target handle was
  previously skipped silently.
- `InputResolver` accepts the resolved rosters and keys its param-info cache
  per node instance when a node's roster came from a hook; type-keyed for
  everything else.

## [1.11.0]

### Added

- `conductor.resolve_graph_outputs(nodes, edges, definitions)` — public
  graph-level output resolution: the exact topological `compute_outputs`
  pass `compile()` runs (step 8b), exposed ahead of compile so a host can
  obtain the authoritative per-node handle set for schema derivation or
  label seeding. Takes a required `definitions` mapping (node type →
  `NodeDefinition | None`) instead of a registry, so hosts can inject
  definitions for extension types (e.g. embedded sub-flows). Applies the
  same structural gates as compile (unknown type, dangling edge, cycle,
  hook-result validation), all raising `CompilationError`. Internally,
  `compile()`'s own dynamic-output pass (step 8b) now routes through the
  same shared engine, so the two entry points cannot diverge — no
  behavior change to `compile()`.

## [1.10.0]

### Added

- `conductor.execution.resolver.finalize_connection_labels(label_hints)` — the
  public form of the ConnectionList label algorithm (collision-aware
  finalization, then `_2`/`_3` dedup). A host that resolves source labels ahead
  of execution — e.g. rewriting stored templates to their live source labels —
  can now reproduce the aggregator's exact input keys without mirroring the
  resolver's private internals. Purely additive: `InputResolver` and its
  labelling behaviour are unchanged.

## [1.9.0]

### Added

- `conductor.registry.serialized.SerializedNode` and
  `conductor.registry.schema.serialize_node_model(nd, registry=None)` — the
  typed counterpart to `serialize_node`, mirroring the existing
  `serialize_input_model` / `serialize_output_model`. A host that serializes a
  node's catalog entry can validate it into `SerializedNode` instead of reading
  a loose `dict[str, Any]`.
- `serialize_node` (and `serialize_node_model`) now accept an optional
  `registry`. Omit it to serialize an ad-hoc definition that isn't registered:
  the node is then reported as the latest, non-deprecated version. Callers that
  pass a registry are unaffected.

### Changed

- `InputMetadata` derives `expects_list` from a `list[...]` `type_str` in
  `__post_init__` when it isn't set explicitly, so hand-built metadata matches
  what the registration decorator derives (a `list[...]` fan-in no longer
  silently string-joins). Metadata that already sets `expects_list` — including
  everything the registry builds — is unchanged.

## [1.8.0]

### Added

- `conductor.execution.results.project_outputs(results, outputs)` and the
  `OutputRef(name, node_id, handle)` value type — project a run's
  `{node_id: NodeResult}` into a flat `{name: value}` map through named
  `(node_id, handle)` refs. The graph-level counterpart to `extract_output`:
  handle-membership decides presence (a produced `None` survives; an absent
  node/handle or a `SKIPPED` value is omitted). `handle=None` selects the
  node's sole result (`RESULT_KEY`, with the `output_1` back-compat fallback).

## [1.7.0]

### Added

- **Typed serialized schema** (`conductor.registry.serialized`): `SerializedInput`
  and `SerializedOutput` — pydantic models the serializers are guaranteed to
  conform to — plus `serialize_input_model` / `serialize_output_model` typed
  accessors on `conductor.registry.schema`. Hosts that want types validate the
  wire payload into these models instead of reading loose dicts; extras are
  allowed so a host's custom `widget_config` keys pass through rather than being
  rejected or dropped.
- **`RegistryView`** (`conductor.registry.view`, re-exported as
  `conductor.RegistryView` with the `DefinitionSource` protocol): a read-only
  overlay that serves host-supplied `NodeDefinition`s for node types not
  statically registered. Point lookups (`get` / `contains`) resolve base-registry
  types first, then each source in order; everything else delegates to the base
  registry. A per-use overlay, not a mutable global hook — callers with different
  visibility never see each other's dynamic types. Generalizes the registry
  wrapper AKA-style hosts previously hand-rolled for extension node types.
- **Published palette TypeScript types**
  (`conductor_providers/react/palette.d.ts`): `PaletteNode`, `PaletteInput`,
  `PaletteOutput` — the TS counterpart of the serialized schema, shipped in the
  wheel. Frontends consuming the palette import these instead of hand-maintaining
  a mirror; an in-repo test pins them to the Python `SerializedInput` /
  `SerializedOutput` fields and the `serialize_node` key set.

### Changed

- `conductor.widgets.WIDGET_SCHEMA_KEYS` is now pinned equal by test to the
  widget-config fields of `SerializedInput`, so the constant, the typed model,
  and the widget `to_schema()` methods cannot drift from one another. The
  constant stays exported and unchanged for hosts that read it.

## [1.6.0]

### Added

- **Public typed registry serialization** (`feat(registry)`): `serialize_node`,
  `serialize_input`, and `serialize_output` are now public — hosts that project
  the serialized registry into their own typed models can call the exact
  functions the wire uses instead of re-implementing the flattening.
- **`conductor.widgets.WIDGET_SCHEMA_KEYS`** — the pinned vocabulary of
  `widget_config` keys stdlib widgets can emit, for hosts to validate their
  typed port models against (guarded by an in-repo bidirectional parity test).

### Changed

- **`FileUpload.accept` serializes as `list[str]`** (`feat(widgets)`): widget
  authors may pass a single string or a list; the wire shape is now always a
  list. Frontends that already handled both shapes are unaffected.

## [1.5.2]

### Fixed

- **Function-node dispatch filters resolved inputs to the node signature**
  (`fix(engine)`): stray `data` keys that aren't node parameters no longer
  crash a normal run with `TypeError: got an unexpected keyword argument`.
  The dispatch path now filters like the compensation path already did;
  functions declaring `**kwargs` are unaffected.

## [1.5.1]

### Fixed

- **For-each loop bodies now inherit the caller's `contextvars`**
  (`fix(engine)`): the for-each compound runs body nodes in its own
  `ThreadPoolExecutor` — for parallel iteration, and for a sequential body
  level with more than one independent node. `ThreadPoolExecutor` workers do
  not inherit `contextvars`, so request-scoped state a host attaches via a
  `ContextVar` (auth/user context, DB handles, tracing spans) was visible to
  every node **except** those inside a loop body, and only in the threaded
  paths. A body node reading that state would fail — but only in `Parallel`
  mode — which a host typically surfaces as a 500. The pools now run each task
  inside `contextvars.copy_context()`, restoring parity with the top-level
  engine (which dispatches via `asyncio.to_thread`, copying the context).

## [1.5.0]

### Added

- **`HumanReview` widget — declarative per-node human-in-the-loop approval**
  (`feat(engine)`): a new widget (`conductor.widgets.HumanReview`) that any
  node can attach to a boolean input. It renders as an ordinary `Switch`
  (`WidgetType.SWITCH`, so existing frontends need no new renderer) and stamps
  `human_review: true` into the input's `widget_config`. When the resolved
  toggle is truthy and the node produced a non-skipped value, the engine pauses
  **after** the node computes, emitting `flow_paused` with
  `schema={"kind": "approval", "value": <result>}`. `resume()` injects the
  human's response as the node's result, so the node is **not** re-run — no
  recomputation or re-billing. Resume semantics: the same value approves, a
  different value edits, and `SKIPPED` rejects (downstream is skipped). Unlike
  raising `HumanInputRequired`, this requires **no code in the node body** —
  opting in is a single annotated input. Additive and backwards-compatible:
  nodes without the widget are unaffected.

## [1.4.0]

### Added

- **`cache` on the FastAPI provider's `/execute` and `/execute-stream`**
  (`feat(fastapi)`): `ExecuteRequest` now accepts an optional
  `cache: dict[str, Any]` mapping node id → precomputed result, forwarded to
  the engine. Listed nodes are seeded as completed (the engine emits
  `node_complete` with `cached=True`) and skipped, so a host can reuse outputs
  from a previous run instead of recomputing the whole graph. Additive and
  backwards-compatible: omitting `cache` is unchanged behaviour, and the engine
  already supported the parameter — only the HTTP layer is new.

## [1.3.0]

### Added

- **`conductor.compute_for_each_end_outputs`** — the default `compute_outputs`
  hook for the `for-each-end` marker, now part of the public surface (also
  re-exported from `conductor.compound`). Hosts that re-register the loop
  markers (e.g. to localize labels) can pass it straight through instead of
  reimplementing the typing rule.

### Changed

- **Typed `for-each-end` collected outputs** (`feat(for-each)`): the stdlib
  `for-each-end` marker now ships `compute_for_each_end_outputs` by default.
  Each collected slot is typed `list[<inner>]` — where `<inner>` is the wired
  source's element type, with one `list[...]` level unwrapped when the source
  already produces a list — and labelled from the source (sub-output prefix
  stripped). Previously the collected outputs were untyped, which silently
  weakened compile-time type-checking of anything consuming a loop result.
  This is additive: slot names (`output_1`, `output_2`, …), ordering, and the
  dedup-by-`(source, handle)` rule are unchanged (the hook reuses
  `_is_end_input_edge` and mirrors `_discover_end_slots`), so the runtime
  contract and saved-flow wire targets are unaffected — including legacy
  `item`/`item_N` end handles.

## [1.2.0]

### Changed

- **DAG-scheduled for-each body** (`feat(for-each)`): the loop body no
  longer runs strictly sequentially. `_execute_subgraph` now groups body
  nodes into dependency levels (`_body_levels`) and runs each level
  concurrently — independent body nodes execute in parallel, dependent
  ones stay ordered, mirroring the top-level eager scheduler. A single
  `BoundedSemaphore(_BODY_CONCURRENCY=8)` shared by every iteration caps
  total body-node executions in flight across the whole loop, so a
  Parallel loop over a multi-node body can't multiply into a storm of
  concurrent calls. A linear body collapses to single-node levels and runs
  inline, identical to the previous path.
- Versions of `syv-conductor`, `syv-conductor-nodes`, and
  `syv-conductor-providers` bumped from `1.1.0` to `1.2.0`.
- Cross-package pins in `syv-conductor[nodes]`, `[providers]`, and `[all]`
  extras updated to `==1.2.0`.

## [1.1.0]

### Fixed

- **Live compound-node events** (`fix(engine)`): a running compound node
  (for-each / while) emits body-node `node_start` / `node_complete` and
  `node_progress` to its `_event_sink`, which the engine only flushed
  *after* the compound finished — so a host saw nothing until the loop
  ended, then a burst. `_execute_node_async` now runs the node's dispatch
  alongside a concurrent sink drainer (`_drain_sink_live`, polling every
  `_SINK_DRAIN_INTERVAL = 0.05s`) that forwards emitted events as they
  happen; the drainer is cancelled once dispatch settles and the existing
  tail-drain flushes any final stragglers. Sequential for-each progress and
  per-iteration body-node status now stream live.

### Changed

- **Parallel for-each progress** (`feat(for-each)`): the parallel branch
  switched from `pool.map` to `submit` + `as_completed`, emitting one
  `node_progress` event per item as it finishes (`1/N … N/N`) instead of a
  single terminal `N/N`. Results are slotted back by index so collected
  order still matches item order regardless of completion order. Combined
  with the live-drain fix above, parallel loops now show a live counter too.
- Versions of `syv-conductor`, `syv-conductor-nodes`, and
  `syv-conductor-providers` bumped from `1.0.1` to `1.1.0`.
- Cross-package pins in `syv-conductor[nodes]`, `[providers]`, and `[all]`
  extras updated to `==1.1.0`.

## [1.0.0]

First stable release. The public surface (everything listed in each module's
`__all__`) is now committed: it will not break without a major version bump.

### Added

- Stability and versioning policy in `README.md`. Semver applies from `1.0.0`
  onward; deprecations stay live for at least one minor release with a
  `DeprecationWarning`; only `__all__`-exported names are public.
- Explicit `__all__` lists on the conductor public modules: top-level
  `conductor.__init__`, `widgets`, `metadata`, `types`, `errors`,
  `registry/__init__`, `registry/dynamic_outputs`, `graph/compiler`,
  `compound/__init__`, `compound/protocol`, `execution/engine`,
  `execution/events`, `execution/results`, `execution/state`.
- `CHANGELOG.md` documenting the `0.1.0` → `1.0.0` history.
- Stress-test suite under `tests/test_stress/` (`pytest -m slow`) covering
  large-graph compilation and engine throughput; gated behind the new
  `slow` pytest marker so the default test run stays fast.

### Changed

- Versions of `syv-conductor`, `syv-conductor-nodes`, and
  `syv-conductor-providers` bumped from `0.1.7` to `1.0.0`.
- Cross-package pins in `syv-conductor[nodes]`, `[providers]`, and `[all]`
  extras updated to `==1.0.0`.

### Fixed

- Type-compatibility: `object` is now treated as a universal-accept type
  in `graph/type_check.py`, matching its Python-level semantics. Generic
  compound passthroughs (notably `for-each-end`'s `Item` input) declare
  `object`; before this fix a concrete source type like `namedfile`
  failed strict compile with a "Type mismatch" error.

## [0.1.7] — A9 widget primitives + A10 ergonomics

### Added

- **Tabular-data widget primitives** (`feat(widgets)`): `TableSource`,
  `ConditionBuilder`, `Tags`, `ColumnSelect`, `TableInput` widgets and the
  matching `WidgetType` enum values. These are skeleton widgets for the host
  to render; conductor declares them but assigns no execution semantics.
- **`compute_outputs` ergonomics** (`feat(registry)`):
  - `NodeCategory.node(...)` now forwards `compute_outputs=` like
    `@registry.node(...)` does, so categorized nodes can declare dynamic
    output shapes without dropping back to the raw registry decorator.
  - `strip_sub_output_prefix` promoted to a public helper on
    `conductor.registry.dynamic_outputs` for hooks that read sub-output
    handle names.
  - `ComputeOutputsContext.validated_data` exposes the node's `data` payload
    after running through the registered Pydantic validation model. Hooks
    for nodes with `SchemaBuilder` / `ConnectionList` widgets can now read
    coerced values without re-implementing the engine's coercion.

### Chore

- Ruff style pass over the touched files.

## [0.1.6] — A1 `compute_outputs` hook + A2–A7 fixes + D1-PR1 operator catalog

### Added

- **Dynamic outputs hook** (`feat(registry)`): `NodeDefinition.compute_outputs`
  callable runs at compile time, in topological order, to re-derive a node's
  output schema from its `data` and resolved upstream `OutputMetadata`.
  Companion module `conductor.registry.dynamic_outputs` defines
  `IncomingBinding`, `ComputeOutputsContext`, `ComputeOutputsFn`. Existing
  static-shape nodes are unaffected — a `None` hook means "use declared
  outputs verbatim".
- **`conductor-nodes` if-else operator catalog** (`feat(conductor-nodes)`,
  D1-PR1): full set of comparison / membership / regex operators wired into
  the if-else nodes for parity with the AKA host.

### Fixed

- **AKA migration parity** (`fix`, A2–A7): SKIPPED filter behavior, zip
  truncation runtime warning event, retry classification refinements,
  widget keys aligned with the frontend contract, `connection_input`
  promoted to the `Widget` base class, and a nested-loop depth cap to
  prevent runaway recursion in subprocess + for-each compositions.

## [0.1.5] — multi-output collection on for-each-end

### Added

- **For-each fan-out** (`feat(for-each)`): per-slot Collected lists on
  `for-each-end`, so a body that emits multiple outputs per iteration
  produces parallel collected lists at the end node.
- **Unlimited compound-node IO** (`feat`): removed the historical input/output
  count limits on loop nodes; for-each / while can now carry as many handles
  as the host wires.

### Fixed

- Lint pass over the touched modules.

## [0.1.4] — providers extras pin

### Fixed

- **Providers `[all]` extra pin** (`chore(release)`): correct
  `syv-conductor-providers` pin in the `syv-conductor[all]` extra so a
  bare `pip install syv-conductor[all]` resolves cleanly.

## [0.1.3] — parallel-zip multi-source for-each

### Added

- **Parallel-zip for-each** (`feat(for-each)`): `for-each-start` accepts
  multiple wired source lists and exposes per-source `Item` outputs. Each
  iteration receives one element from each list; the loop length is the
  shortest source (truncation is signalled via a runtime warning event in
  `0.1.6`).

## [0.1.2] — dependency extras

### Added

- **Optional dependency extras** (`feat`): `syv-conductor[yaml]` for the
  YAML/JSON flow format, `syv-conductor[nodes]`, `[providers]`, and `[all]`
  bundles — pinned `==` to the matching workspace version to prevent
  resolver skew.

## [0.1.1] — extension resolver wiring

### Added

- **Extension resolver forwarding** (`feat(providers)`):
  `conductor_providers.fastapi.conductor_router(...)` now accepts and
  forwards an `extension_resolver` so host-app-specific node types
  (sub-flows, etc.) participate in compile / execute on the HTTP surface.

## [0.1.0] — initial public release

First publish to PyPI as three workspace packages.

### Added

- **Core engine** (`syv-conductor`): decorator-based node registration,
  union-aware type checking, eager parallel scheduling, retry with
  exponential backoff, structured error hierarchy with `node_id` /
  `node_type` context, streaming execution events, shared references
  (produce / consume across edges), conditional branching via the
  `SKIPPED` sentinel, for-each / while / subprocess compounds,
  human-in-the-loop via `HumanInputRequired` + checkpoints, `BaseNode`
  ABC for stateful nodes, `FlowStore` side-channel cache, package
  auto-discovery, extension resolver protocol for host-app node types,
  decision nodes with CEL-guarded edges, sandboxed CEL expression engine
  (`conductor.expr`), actor metadata, top-level `Flow` metadata
  (`dependencies`, `triggers`, `on_error_default`), per-node `timeout=`
  + `idempotency_key=`, while-loop runaway protection, subprocess depth
  cap, compensation / saga cascade, signal nodes via `SignalRequired`,
  YAML / JSON flow format under `[yaml]`.
- **Standard node library** (`syv-conductor-nodes`): text, math, logic,
  json, regex, decision, while, subprocess, signal, and canonical
  for-each markers. Categories opt-in via
  `register_all(registry, categories=[...])`.
- **Framework adapters** (`syv-conductor-providers`):
  - `conductor_providers.react` — `graph_to_react` /
    `react_to_graph` / `palette_from_registry` for ReactFlow JSON.
  - `conductor_providers.fastapi` — `conductor_router(...)` factory
    exposing `/execute`, `/execute-stream`, `/compile`, `/nodes`,
    `/entities/{kind}` with optional `entity_resolver` and (from
    `0.1.1`) `extension_resolver` hooks.
- **Packaging**: PyPI distribution names prefixed with `syv-`; Python
  imports unchanged (`conductor`, `conductor_nodes`, `conductor_providers`).
  License: Apache-2.0. Each wheel ships `LICENSE`.

[Unreleased]: https://github.com/syv-ai/conductor/compare/v1.12.2...HEAD
[1.12.2]: https://github.com/syv-ai/conductor/compare/v1.12.1...v1.12.2
[1.12.1]: https://github.com/syv-ai/conductor/compare/v1.12.0...v1.12.1
[1.12.0]: https://github.com/syv-ai/conductor/compare/v1.11.0...v1.12.0
[1.11.0]: https://github.com/syv-ai/conductor/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/syv-ai/conductor/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/syv-ai/conductor/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/syv-ai/conductor/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/syv-ai/conductor/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/syv-ai/conductor/compare/v1.5.2...v1.6.0
[1.5.2]: https://github.com/syv-ai/conductor/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/syv-ai/conductor/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/syv-ai/conductor/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/syv-ai/conductor/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/syv-ai/conductor/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/syv-ai/conductor/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/syv-ai/conductor/compare/v1.0.1...v1.1.0
[1.0.0]: https://github.com/syv-ai/conductor/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/syv-ai/conductor/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/syv-ai/conductor/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/syv-ai/conductor/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/syv-ai/conductor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syv-ai/conductor/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syv-ai/conductor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syv-ai/conductor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syv-ai/conductor/releases/tag/v0.1.0
