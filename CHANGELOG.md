# Changelog

All notable changes to the three workspace packages (`syv-conductor`, `syv-conductor-nodes`,
`syv-conductor-providers`), which release together at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Compile looks each placement up in the registry once, and a compiled node or placement holds the class it resolved to as `definition`. `compiled.node(id).runner` runs that class, so a module reloaded after compile runs once the graph is compiled again. `NodeRegistry.runner_for` is removed.
- `compiled.node(id)` answers for every node compile met, each with a `state` (`ready`, `not_derived`, `unresolved`) and its `problems`; a field has `problems` too. Reading what a state lacks raises `NotResolved` or `NotDerived`, carrying the `Problem` that explains it; `KeyError` is only for an id the graph lacks. A node whose version is a graph is a `CompiledPlacement`, with no `graph_node`, `statics`, `runner`, `fingerprint` or `validate`.
- `decisions` leaves out a node the walk did not derive, and a misaligned embedded graph is not derived.
- `conductor_providers.mermaid.flowchart` draws any graph, however broken.
- An input named with a leading underscore, from a `**inputs` edge or a `compute_inputs` hook, is `parameter_name_invalid` at compile. A call model is built on a node's first `validate`.
- Removed: `is_input_node` (the graph's inputs are `compiled.interface.inputs`) and `GraphNode.data` (read `node.bindings`).
- The public API is three doors: `conductor`, `conductor.widgets` and `conductor.events` (new), each with an `__all__` a type checker accepts. `NEVER`, the `Receive` records, `model_of`, `to_wire`, `from_wire`, `Upgrade`, `VersionDescription` and the run state's entries (`StateValue`, `StateSkip`, `DoneUnit`) join the root; `dtype_of` and `DTypeRef` leave it, and `AnyWidget` is imported from `conductor.widgets` only. Each `conductor_nodes` module lists its nodes in `NODES` instead of a `register()` function.

## [2.0.0]

A new major. The node contract, the graph record, compile and the engine are each replaced, and
no alias keeps a 1.x name alive. A 1.x graph does not load as saved. What a 1.x user must change:

### Nodes

- A node is a `NodeDefinition` subclass with `id`, `title`, `description` and `category`; its typed
  `run` signature is its interface. `@registry.node`, `BaseNode` and `NodeCategory` are gone.
- A parameter is `Annotated[DType, Param(title=..., widget=...)]`; one output is
  `-> Annotated[DType, Result(title=...)]`, several are a frozen dataclass of them.
- `run` is a plain function run in a worker thread; `async def run` is refused.
- Versions are `@version(n)` methods on one class, each with a `Policy`. `@upgrade(a, b)` rewrites
  saved values; `@deprecated` retires a node or a version.
- `InputMetadata` / `OutputMetadata` are `Input` / `Output`. The hooks are
  `compute_inputs(declared, values)` and `compute_outputs(declared, values, arriving)`, and a hook
  that cannot answer raises `Refuses`.
- `FlowStore` / `store_data` are `Annotated[T, FromRun()]`, filled from `execute(from_run={T: value})`.
- The widgets are the control alone, with a `kind`; there is no default widget for a type.
  `Output`, `Checkbox`, `Multiselect`, `DependentDropdown`, `HumanReview`, `TableSource`,
  `ConditionBuilder` and `ColumnSelect` are gone.

### Types

- A type is a `DType` class with its own `id`. A registry holds the types its nodes declare plus
  `registry.add_types(...)`; `target.accepts(source)` is the edge check.
- `Series[X]` is the one collection. The for-each, while, subprocess and signal marker nodes, and
  regions, are gone: a series arriving on a scalar input runs the node once per row.

### Graphs and compile

- `Flow` is `Graph(nodes=[GraphNode(id=, type=, version=, bindings=)])`. An input is bound by
  `From("node.output")` or `Static(value)`; there is no edge list and no `"id@version"` string.
- `compile(...)` is `CompiledGraph.from_graph(graph, registry)`. It never raises for a fault in the
  graph: read `problems` and `is_runnable`.
- `conductor.flow_format` is gone: `graph.to_path(...)` / `Graph.from_path(...)`.
- `registry.upgraded(graph, node_id)` moves a placed node to a newer version.
- CEL guards on edges, shared references, compensation and `on_error` are gone.

### Running

- `execute`, `run` and `run_sync` are the entry points; `execute_sync`, `collect`, `resume` and
  checkpoints are gone. A result is `ending.state.results(compiled)[node_id][output]`.
- A pause is a node returning `Asks`. The run ends `graph_pending`, and the next call is
  `execute(compiled, state=ending.state, cache={node_id: answers})`.
- Retries live on the version's `Policy`. Only an `ExternalFailure`, or an exception named in
  `Policy.retry_on`, is retried. `RetryConfig` and per-registration retry arguments are gone.
- Every failure carries an `ErrorCause` (`code`, `message`, `details`, `row`). `Flow*` errors and
  events are `Graph*`; `flow_paused` is `graph_pending`.

### Standard nodes and providers

- `get_default_registry` is `conductor_nodes.registry(categories=...)`. The standard nodes declare
  their own types in `conductor_nodes.types` (`Text`, `Number`, `Flag`, `Json`).
- `ExecuteRequest` is `{graph, state, cache}`, and `/execute` answers with the ending frame.
  `conductor_router`'s per-request hook is `from_run`; `context_factory`, `extension_resolver`
  and `palette_from_registry` are gone (the palette is `registry.describe()`).
- Requires pydantic 2.11 or later.

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
