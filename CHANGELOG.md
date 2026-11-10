# Changelog

All notable changes to `syv-conductor`; up to 1.x this also covered `syv-conductor-nodes` and
`syv-conductor-providers`, released with it at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, compile and the engine are each
replaced, and no alias keeps a 1.x name alive. A 1.x graph does not load as saved; a host converts
its stored graphs itself. What a 1.x user changes:

- **Imports.** Import from `conductor`, `conductor.widgets` and `conductor.events`; everything else is
  `conductor._internal`. Conductor depends on `polars>=2.0` and `pydantic>=2.11`. The records a
  public call returns but nobody names (`CompiledNode`, `RunPlan`, `WorkOfNode`, `Question` and the
  like) are not exported.
- **A node.** `BaseNode` is a plain function under `@registry.node("id")`, or under `@node("id")`,
  which takes the same arguments and registers nothing. Either returns a frozen, callable
  `NodeDefinition`; `version=` and `name=` are gone. `async def` isn't allowed: the function runs in a
  worker thread. `NodeCategory` and auto-discovery are gone. A node takes `extra=`, the host's own
  JSON about it, kept and described and never read.
- **A registry.** `NodeRegistry(category=, tags=, extra=, types=)` starts empty. `@registry.node`
  declares and adds; `add(*definitions)` adds plain `@node`s, and a second node under one id raises;
  `replace` and `remove` change one. `add_subregistry(other)` copies another registry's nodes in, so
  each module keeps a registry and the host joins them. A node entering takes the registry's
  defaults: tags add up, outermost first, and its own `category` and `extra` keys win.
  `filter(predicate)` is a new registry of the nodes kept. Looping yields the definitions. A
  deprecation's `alternative` need not be in the registry.
- **Inputs and outputs.** The type hints are the interface: `text: str`, or
  `Annotated[str, Param(title=..., widget=...)]`. `-> str` is one output named `result`, a frozen
  dataclass is one output per field, and `Result(...)` titles one. `InputMetadata` and
  `OutputMetadata` are `Input` and `Output`. A default is checked where it is written:
  `x: int = "abc"` is a `NodeDeclarationError` at import.
- **Fields that depend on configuration** come from `compute_added_inputs=` and `compute_outputs=`,
  plain functions whose parameters name the inputs they read. `ArrivingType` gives the type on an
  edge, and `ComputeHookError` reports what's wrong.
- **Versions.** A newer version names the one it replaces, `@node("id", replaces=older,
  upgrade=Upgrade(...))`, where `older` is a `@node` of the same id. Each version has its own
  `Policy`, hooks and `deprecation=`, and the newest version's notice is the node's.
  `upgrade_node(graph, node_id, registry)` moves a placed node on.
- **What the run supplies.** `FlowStore` and `store_data` are a parameter typed `FromRun[T]`, filled
  from `compiled.run(from_run={T: value})`.
- **Retries and failures.** `max_retries`, `retry_delay` and `RetryConfig` are
  `Policy(retries=, retry_delay=, attempt_timeout=, concurrency=, retry_on=)`. Only an
  `ExternalFailure`, or an exception named in `retry_on`, is retried. `NodeValidationError`,
  `NodeExecutionError` and `NodeTimeoutError` are `NodeError(message, code=, ctx=)`, or
  `ExternalFailure` for what the outside world caused.
- **Widgets.** A widget is the control alone, with a `type`, and no type has a default one.
  Conductor ships text, textarea, dropdown, number, range, switch, date, file and list, and a host
  subclasses `Widget` for its own. The limits of a value go on the `Param` (`ge`, `le`, `min_length`,
  `max_length`, `pattern`), where they are enforced at compile and on every call. `Checkbox`,
  `Multiselect`, `DependentDropdown`, `HumanReview`, `TableSource`, `ConditionBuilder` and
  `ColumnSelect` are gone, and so is the widget called `Output`.
- **Types.** Plain Python types and pydantic models travel on edges. Write `class X(DType, str)` with
  an `id` only for a type of your own; `target.accepts(source)` is the edge check.
- **Rows.** The for-each, while, subprocess and signal nodes and regions are gone. A column arriving
  on an input declared for one value runs the node once per row, and a node isn't called for a blank
  unless its input says `| None`. `Series[X]` marks a field as holding many values of `X`: a
  `Sequence[X]` to a type checker, a plain list to the node, every value under the node's row.
  `results()`, `outputs()` and `NodeComplete.outputs` give it as `{"rows": [...], "values": [...]}`.
  `compiled.run_plan` says what compile decided.
- **Tables.** `Table[TypedDict]` is a whole table, a `polars.DataFrame`. Tables and columns wired
  into one `Table` input merge on the hidden `__row` column: a parent is only copied down onto its
  children, so the rows are those of the deepest source; sources on one level keep all their rows,
  with blanks beside; and tables with no rows in common are refused when the graph is checked
  (`no_rows_in_common`), so join those on a key in a node of your own.
- **Graphs.** `Flow` is `Graph(nodes=[...])`, with a node placed by
  `upper.graph_node("a", text="hello")`. An input is bound by `From("node.output")` or
  `Static(value)`, and `text=OPEN` leaves it for the caller; there is no edge list and no
  `"id@version"` string. `conductor.flow_format` is `graph.model_dump_json()` and `Graph.model_validate_json(...)`, and YAML is gone with `pyyaml`.
  CEL guards on edges, shared references, compensation and `on_error` are gone.
- **Compile.** `compile(...)` is `CompiledGraph.from_graph(graph, registry)`. It never raises for a
  fault in the graph: read `problems` and `is_runnable`, or `compiled.describe()` for all of it as
  one JSON-ready record. A `Problem` is `type`, `loc`, `msg` and `ctx`.
- **Embedded graphs.** `extension_resolver` is gone. A graph embedded as a node is
  `NodeDefinition.from_compiled_graphs(...)`, wired by the names it offers, and it runs as its nodes
  drawn in its place.
- **Running.** `execute`, `execute_sync` and `collect` are `compiled.stream()`, `compiled.run()` and
  `compiled.run_sync()`. A run is its `RunState`: a pause, a failure or a timeout is a `status`, and
  `compiled.results(state)` or `compiled.outputs(state)` reads it. A failed unit stops anything new
  from starting; the calls already running finish and are kept.
- **Events.** `flow_*` events are `run_start`, `node_failed`, `unit_waiting` and `run_end`. What goes
  wrong is always a `Problem`, and a node's own exception goes to the `conductor` logger. An event
  writes itself with `event.model_dump_json()`; `to_jsonable(value)` does the same for results.
  `run_end` keeps its `state` for the host and writes only its `status`. `node_progress` arrives at
  most four times a second per node.
- **A person in the loop.** `HumanInputRequired` and `SignalRequired` are `AskPerson`, returned by a
  node typed `-> X | AskPerson`. In place of `resume`, `compiled.answer(state, {unit.id: {output:
  value}})` returns a state to run from. An answer stays with the row it was asked on: a row replaced
  or edited is asked again.
- **Running again.** Pass an earlier leg's state to `compiled.run`. A node nothing changed above is
  restored whole, and a changed node reuses every row that received the same as before, so one row
  added to a CSV of 5,000 sends one mail. `Policy(reuse_when_inputs_match=False)` opts a node out,
  `RunState.without_nodes(...)` forgets one, and a state goes back only to the runner it came from.
  A long leg saves as it goes with `compiled.stream(checkpoint_every=...)`.
- **Standard nodes and providers.** `syv-conductor-nodes` (`conductor_nodes`, with
  `get_default_registry`) and `syv-conductor-providers` (`conductor_providers`, with the Mermaid
  drawing) are no longer published: the docs declare the few nodes they need where they use them.
- **HTTP.** `conductor_providers.fastapi` and the `fastapi` extra are gone: `conductor_router`,
  `/execute`, `/execute-stream`, `/nodes`, `/compile`, `/entities/{kind}`, `context_factory`,
  `entity_resolver`, `palette_from_registry` and `sse_frame`. Copy `docs/serving-runs.md` onto your
  own application; it needs `fastapi>=0.135`. `/nodes` is `registry.describe()`, `/compile` is
  `CompiledGraph.from_graph(graph, registry).describe()`, and `context_factory` is a `Depends` on your
  route, passed as `from_run`.
- **Canvas.** `conductor_providers.react` is gone: `graph_to_react`, `react_to_graph` and
  `topological_positions`. A canvas stores the `Graph` as it is, with one edge per ref of each `From`
  binding and a node's position in `node.display["position"]`; for layout, use the canvas's own engine
  or `dependencies_of(graph.nodes)` for the order.

## [1.12.2]

- The validation model no longer turns `**kwargs` into a required field.

## [1.12.1]

- `**kwargs` / `*args` are no longer introspected as input handles.

## [1.12.0]

- `compute_added_inputs` hook, `CompiledGraph.node_inputs` and `has_dynamic_inputs` in the palette.

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

[Unreleased]: https://github.com/syv-ai/conductor/compare/v1.12.2...HEAD
[1.2.0]: https://github.com/syv-ai/conductor/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/syv-ai/conductor/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/syv-ai/conductor/compare/v0.1.7...v1.0.0
[0.1.7]: https://github.com/syv-ai/conductor/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/syv-ai/conductor/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/syv-ai/conductor/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/syv-ai/conductor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syv-ai/conductor/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syv-ai/conductor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syv-ai/conductor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syv-ai/conductor/releases/tag/v0.1.0
