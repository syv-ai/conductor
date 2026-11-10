# Changelog

All notable changes to `syv-conductor`; up to 1.x this also covered `syv-conductor-nodes` and
`syv-conductor-providers`, released with it at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, loading and the engine are each
replaced, and no alias keeps a 1.x name alive. A 1.x graph does not load as saved: a host converts
its stored graphs itself. The README walks through the new design, and `python -m conductor.about`
has every rule. What a 1.x user changes:

- **Imports.** Import from `conductor`, `conductor.widgets`, `conductor.events` and
  `conductor.testing`; everything else is `conductor._internal`. Conductor depends on `polars>=2.0`
  and `pydantic>=2.11`.
- **Nodes.** `BaseNode` is a plain function under `@registry.node("id")`, or under `@node("id")`,
  which registers nothing. Either returns a frozen, callable `NodeDefinition`. The type hints are
  the interface: `Annotated[str, Param(title=, widget=, ge=, ...)]` for what a person sees and the
  limits a value keeps, `-> X` for one output named `result`, a frozen dataclass for several.
  `async def`, `name=`, `NodeCategory` and auto-discovery are gone. A node takes `extra=`, the
  host's own JSON.
- **Registries.** `NodeRegistry(category=, tags=, extra=, types=)` starts empty. `add`, `replace`,
  `remove`, `add_subregistry` and `filter` change what it offers, and a second node under one id
  raises. `registry.describe()` is the palette.
- **Fields that depend on configuration** come from `compute_added_inputs=` and
  `compute_outputs=`, plain functions that return `{name: type hint}`. A hook that raises is the
  problem `compute_hook_failed`.
- **Versions.** A newer version names the one it replaces, `@node("id", replaces=older,
  upgrade=Upgrade(...))`. `registry.upgrade(record, node_id)` moves a saved node on;
  `Deprecation(...)` marks a version going away.
- **What the run supplies.** `FlowStore` and `store_data` are a parameter typed `FromRun[T]`,
  filled from `from_run={T: value}`.
- **Retries and failures.** `max_retries`, `retry_delay` and `RetryConfig` are `Policy(...)`, whose
  `concurrency` is 1 by default. Only an `ExternalFailure`, or a class in `retry_on`, is retried.
  `NodeValidationError`, `NodeExecutionError` and `NodeTimeoutError` are `NodeError(message,
  code=, ctx=)`. Every failure is a `Problem`, and a node's own exception goes to the `conductor`
  logger, not to an event. What a node returns is checked against its type (`invalid_output`).
- **Widgets.** A widget is the control alone, and no type has a default one. `Checkbox`,
  `Multiselect`, `DependentDropdown`, `HumanReview`, `TableSource`, `ConditionBuilder`,
  `ColumnSelect` and `Output` are gone; a host subclasses `Widget` for its own.
- **Types.** Any type pydantic can carry travels on edges, with no base class. `Series[X]` is many
  values, `Table[R]` a whole `polars.DataFrame`.
- **Rows.** The for-each, while, subprocess and signal nodes and regions are gone. A column on an
  input typed for one value runs the node once per row. A blank skips the node for that row unless
  its input says `| None`. `max_rows=` caps a run's rows (`row_ceiling`).
- **Graphs.** `Flow` is a `GraphRecord`, `{"nodes": [...], "edges": [[from, to], ...]}`, and a
  node's `type` is `node`. `make_graph(...)` writes one from Python. YAML, CEL guards on edges,
  shared references, compensation and `on_error` are gone, and so is `extension_resolver`: a stored
  graph is a node, `node("id")(record)`.
- **Loading.** `compile(...)` is `registry.load(record)`, which never raises for a fault in the
  graph: read `graph.problems`. An open input of a first node is an input of the graph.
- **Running.** `execute`, `execute_sync` and `collect` are `graph.run(**inputs)`, which returns a
  `Run`. A pause, a failure or a timeout is `run.status`. A run is JSON: `run.to_json()` and
  `graph.load_run(saved)`. `run.again(**changed)` reruns only the rows whose inputs changed, and
  `run.forget(*node_ids)` calls chosen nodes again.
- **Events.** `flow_*` events are `run_start`, `node_failed`, `unit_waiting` and `run_end`, pydantic
  models in `conductor.events`.
- **A person in the loop.** `HumanInputRequired` and `SignalRequired` are `raise Ask(prompt,
  choices=, default=, widget=)`, and the node's return type says what the answer is and how to ask.
  `run.answer(unit, value)`, or `run.answer({unit_id: value, ...})`, writes answers in, each checked
  before anything runs. `Choice` is imported from `conductor`.
- **Testing nodes.** `conductor.testing.check_registry(registry, examples=...)` is one test for a
  whole registry.
- **Standard nodes and providers.** `syv-conductor-nodes` and `syv-conductor-providers` are no
  longer published.
- **HTTP.** `conductor_providers.fastapi` and the `fastapi` extra are gone. Copy
  `docs/serving-runs.md` onto your own application. `/nodes` is `registry.describe()` and
  `/compile` is `registry.load(graph).describe()`.
- **Canvas.** `conductor_providers.react` is gone. A canvas stores the `GraphRecord` as it is, with
  a node's position in `node.display["position"]`.

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
