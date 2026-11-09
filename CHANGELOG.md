# Changelog

All notable changes to the three workspace packages (`syv-conductor`, `syv-conductor-nodes`,
`syv-conductor-providers`), which release together at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, compile and the engine are each
replaced, and no alias keeps a 1.x name alive. A 1.x graph does not load as saved; a host converts
its stored graphs itself. What a 1.x user must change:

### Imports

- Import from `conductor`, `conductor.widgets` and `conductor.events`. `conductor_nodes` and
  `conductor_providers` are public too. Everything else is `conductor._internal`.

### Nodes

- `@registry.node("id", version=1, name=...)` and `BaseNode` are `@node("id")` above a plain
  function, which returns a frozen, callable `NodeDefinition` and registers nothing. Build the
  registry whole: `NodeRegistry([upper, join])`. `NodeCategory` and auto-discovery are gone.
- The type hints are the interface: `text: str`, or `Annotated[str, Param(title=..., widget=...)]`.
  `-> str` is one output named `result`, a frozen dataclass is one output per field, `Result(...)`
  titles one. `async def` isn't allowed: the function runs in a worker thread.
- A widget is the control alone, with a `kind`, and no type has a default one. `Output`, `Checkbox`,
  `Multiselect`, `DependentDropdown`, `HumanReview`, `TableSource`, `ConditionBuilder` and
  `ColumnSelect` are gone.
- `InputMetadata` / `OutputMetadata` are `Input` / `Output`. Fields that depend on configuration come
  from `compute_inputs=` and `compute_outputs=`, plain functions whose parameters name the inputs
  they read; `Arriving` gives the type on an edge; `ComputeHookError` reports what's wrong.
- Older versions are `older_versions=[OlderVersion(fn, upgrade=Upgrade(...))]`, oldest first, each
  with its own `Policy` and hooks. `upgrade_node(graph, node_id, registry)` moves a placed node on.
- `FlowStore` / `store_data` are a parameter typed `FromRun[T]`, filled from
  `compiled.run(from_run={T: value})`.
- `max_retries`, `retry_delay` and `RetryConfig` are `Policy(retries=, delay=, timeout=,
  concurrency=, retry_on=)`. Only an `ExternalFailure`, or an exception named in `retry_on`, is retried.

### Types, rows and tables

- Plain Python types and pydantic models travel on edges. Write `class X(DType, str)` with an `id`
  only for a type of your own; `target.accepts(source)` is the edge check.
- `Series[X]` is the one collection, and `Table[TypedDict]` a whole table, a `polars.DataFrame`.
  Conductor depends on `polars>=2.0` and `pydantic>=2.11`.
- The for-each, while, subprocess and signal nodes and regions are gone. A column arriving on an input
  declared for one value runs the node once per row; a `Series[X]` input gets every value under the
  node's row. `compiled.run_plan` says what compile decided.
- Tables and columns wired into one `Table` input merge on the hidden `__row` column. A parent is
  copied down onto its children, so the rows are those of the deepest source; sources on one level
  keep all their rows, with blanks beside.
- A node isn't called for a blank unless its input says `| None`.

### Graphs and compile

- `Flow` is `Graph(nodes=[...])`, and a node you declared is placed with
  `upper.graph_node("a", text="hello")`. An input is bound by `From("node.output")` or
  `Static(value)`; there is no edge list and no `"id@version"` string.
- An embedded graph is compiled in place, each of its nodes by the rule that compiles any node; a list
  or a hook inside reads the graph node's values with no second compile.
- `compiled.run_plan.nodes[id].inputs[name]` is a `ReceiveMode`, the record the run reads: a `kind` of
  `cell`, `rows_under`, `fixed`, `gather` or `merge`, and the author's `receive_mode` word as before.
  `InputPlan` is gone. A table wired alone into a `Table` input is received as that table, unmerged.
- `compile(...)` is `CompiledGraph.from_graph(graph, registry)`. It never raises for a fault in the
  graph: read `problems` and `is_runnable`. A `Problem` is `type`, `loc`, `msg`, `ctx`.
- `extension_resolver` is gone: a graph embedded as a node is
  `NodeDefinition.from_compiled_graphs(...)`, wired by the names it offers.
- `conductor.flow_format` is `graph.to_path(...)` / `Graph.from_path(...)`.
- CEL guards on edges, shared references, compensation and `on_error` are gone.

### Running

- `execute`, `execute_sync` and `collect` are `compiled.stream()`, `compiled.run()` and
  `compiled.run_sync()`. A run is its `RunState`: a pause, a failure or a timeout is a `status`, and
  `compiled.results(state)` or `compiled.outputs(state)` reads it.
- `HumanInputRequired` and `SignalRequired` are `AskPerson`, returned by a node typed
  `-> X | AskPerson`. `compiled.answer(state, {unit.id: {output: value}})` returns a state to run
  from, in place of `resume`.
- `flow_*` events are `run_start`, `node_failed`, `unit_waiting` and `run_end`, which carries the
  state. What goes wrong is always a `Problem`; a node's own exception goes to the `conductor` logger.
- A changed node reuses per row. A leg going on from a state hashes what each unit of a changed node
  is about to receive, and a unit whose inputs match one its earlier units received takes that result
  at its new row and isn't called, a waiting unit or an answer included: one row added to a CSV of
  5,000 sends one mail. The hash is of content, so a row keeps its work when another is inserted above
  it; a `Table` input is read with its `__row`s. Identical rows each get one result; a table a unit
  minted moved with it gets its ids re-spelled at the new row. `WaitingUnit.id` is `node@row#hash12`
  (`node#hash12` for a node that ran once), and a form opened before a row moved is refused
  (`not_waiting`). `Policy(reuse_when_inputs_match=False)` opts a node out. `NodeComplete.reused_units`
  and `RunState.reused_units` report it; `RunState` gains `input_hashes`, `answered_rows` and
  `earlier_work` (what a leg that stopped short hadn't reached yet). `RunState.without_nodes` forgets
  only the nodes it names. A state goes back only to the runner it came from.
- A dumped `RunState` holds each table once, under `tables` by digest, with `{"table": digest}` in
  `output_columns` values, `WaitingUnit.inputs` and question defaults. A plain `model_dump_json()`
  is one document, and a load holds one frame per digest. An event writes its tables whole.

- A node's output that makes rows is one value in a `RunState`: `OutputColumn.rows` holds the row it
  was written at and `values` the list of its cells, a cell skipped alone a `None` there and a row
  in `skipped`. A skip that left an index with no rows leaves nothing of its own in `finished_rows`
  or `skipped`; the nodes running on it complete through the state.

- `NodeValidationError`, `NodeExecutionError` and `NodeTimeoutError` are gone; a node raises `NodeError(message, code=..., ctx=...)`, or `ExternalFailure` for what the outside world caused. A problem's `type` is still the code it always was.

### Standard nodes and providers

- `get_default_registry` is `conductor_nodes.registry(categories=...)`, and
  `conductor_nodes.nodes()` lists the definitions.
- `/execute` and `/execute-stream` are `/run` and `/run-stream`, with the body
  `{graph, inputs, state, answers}`. `/run` answers `{state, results}`. `context_factory` is
  `from_run`, `extension_resolver` and `palette_from_registry` are gone (the palette is
  `registry.describe()`).

## [1.12.2]

- The validation model no longer turns `**kwargs` into a required field.

## [1.12.1]

- `**kwargs` / `*args` are no longer introspected as input handles.

## [1.12.0]

- `compute_inputs` hook, `CompiledGraph.node_inputs` and `has_dynamic_inputs` in the palette.

## [1.11.0]

- `conductor.resolve_graph_outputs(nodes, edges, definitions)`: compile's output resolution, ahead of compile.

## [1.10.0]

- `finalize_connection_labels(label_hints)`, the public connection-label algorithm.

## [1.9.0]

- `SerializedNode` and `serialize_node_model`; `serialize_node` takes an optional `registry`.

## [1.8.0]

- `project_outputs(results, outputs)` and `OutputRef`.

## [1.7.0]

- Typed serialized schema (`SerializedInput`, `SerializedOutput`), `RegistryView`, and published
  palette TypeScript types.

## [1.6.0]

- Public `serialize_node` / `serialize_input` / `serialize_output` and `WIDGET_SCHEMA_KEYS`.
- `FileUpload.accept` serializes as a list.

## [1.5.2]

- Function-node dispatch filters inputs to the node's signature.

## [1.5.1]

- For-each loop bodies inherit the caller's `contextvars`.

## [1.5.0]

- `HumanReview` widget: per-node approval that pauses after the node runs.

## [1.4.0]

- `cache` on the FastAPI provider's `/execute` and `/execute-stream`.

## [1.3.0]

- Typed `for-each-end` collected outputs, and `compute_for_each_end_outputs`.

## [1.2.0]

- For-each bodies run by dependency level, concurrently.

## [1.1.0]

- Compound-node events stream live; parallel for-each reports progress per item.

## [1.0.0]

- First stable release: semantic versioning, explicit `__all__` on public modules, stress tests.

## [0.1.7]

- Tabular widgets (`TableSource`, `ConditionBuilder`, `Tags`, `ColumnSelect`, `TableInput`) and
  `compute_outputs` ergonomics.

## [0.1.6]

- `compute_outputs` hook and the if-else operator catalog.

## [0.1.5]

- Multi-output collection on `for-each-end`; no limit on loop-node handles.

## [0.1.4]

- Correct providers pin in the `[all]` extra.

## [0.1.3]

- Parallel-zip for-each over several source lists.

## [0.1.2]

- Optional dependency extras: `[yaml]`, `[nodes]`, `[providers]`, `[all]`.

## [0.1.1]

- `conductor_router` forwards an `extension_resolver`.

## [0.1.0]

- First release on PyPI as three packages: the engine, the standard nodes and the framework adapters.

[Unreleased]: https://github.com/syvai/conductor/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/syvai/conductor/compare/v1.12.2...v2.0.0
[1.2.0]: https://github.com/syvai/conductor/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/syvai/conductor/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/syvai/conductor/compare/v0.1.7...v1.0.0
[0.1.7]: https://github.com/syvai/conductor/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/syvai/conductor/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/syvai/conductor/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/syvai/conductor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syvai/conductor/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syvai/conductor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syvai/conductor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syvai/conductor/releases/tag/v0.1.0
