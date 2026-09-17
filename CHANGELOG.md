# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and from `1.0.0` onward this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See the "Stability and versioning" section in [`README.md`](README.md) for the
public-API guarantees that take effect at `1.0.0`.

This file covers the three workspace packages — `syv-conductor`,
`syv-conductor-nodes`, and `syv-conductor-providers` — which are released in
lockstep from this monorepo.

## [Unreleased]

### Future deprecation candidates

- Cross-package `==` pin in `syv-conductor[all]` — could relax to
  `~=` once the providers/nodes packages stabilize independently.

## [2.0.0]

A new major, and not a small one: the node contract, the graph record, compile and the engine
are each replaced, and no alias keeps the 1.x names alive. Nothing here reads a 1.x graph as
it was saved. What stays is the idea — a registry of nodes, widgets declared on a node's
parameters, versions — and the three packages, released in lockstep. The deprecation policy in
the README (a name stays live for a minor release before it goes) is set aside for this release:
nothing is deprecated first, everything below is gone in 2.0.0.

### Upgrading, in brief

- **A node** is a `NodeDefinition` subclass with `id`, `title`, `description` and `category`,
  whose typed `run` signature is its interface; `@registry.node` and `BaseNode` are gone. Every
  parameter is `Annotated[DType, Widget(title=...)]`, and one output is
  `-> Annotated[DType, Result(title=...)]`. `run` is a plain function, called in a worker thread;
  an `async def run` is refused.
- **A graph** is `Graph(nodes=[GraphNode(id=, type=, version=, bindings=)])`. An input is bound
  by `Edges(refs=(Ref("node", "output"),))` or `Static(value=...)`; there is no edge list.
- **Compile** is `CompiledGraph.from_graph(graph, registry)`. It never raises for a fault in
  the graph: read `problems` and `is_runnable`.
- **Run** with `execute` / `execute_sync` from `conductor.execution.engine`; neither is exported
  from `conductor` any more. A result is `results[node_id][output_name]`; an iterating node's
  outputs are a `Series`. Events and errors say `graph`, not `flow`.
- **Retries** go on the version: `@version(1, policy=Policy(retries=3))`.
- **A pause** is a node returning `Asks`; the leg ends pending, and the next leg is
  `execute(compiled, cells=..., cache={node_id: answers})`. For an iterating node the answer is
  a `Series` naming the rows it answers. Checkpoints and `resume` are gone.
- **What a run supplies** (1.x's `FlowStore` and `store_data`) is `Annotated[T, FromRun()]` on
  `run`, filled from `execute(from_run={T: value})`.
- **Save a graph** with `graph.to_path("graph.yaml")` / `Graph.from_path(...)`; `conductor.flow_format`
  is gone.

### Added

- `Series[X]` and `Index`: the one collection, and the rows it is on.
- `Asks`, and legs: a run that ends pending and is answered by the next leg.
- `FromRun`, `Refuses`, `ErrorCause`, `Problem`.
- `CompiledGraph`, asked at the graph, the node and the field; `GraphVersion`, a version whose
  body is a graph.
- `registry.extended_with(...)`, `registry.definitions()`, `registry.upgrade_path(...)`,
  `NodeDescription`.
- `ConductorModel`: `to_yaml` / `from_yaml` / `to_path` / `from_path` on every saved record.
- The widgets' `Choice` and `OperatorChoice`.
- `conductor_nodes.types`: the standard library's own `Text`, `Number`, `Flag` and `Json`.
- `examples/06_human_in_the_loop.ipynb`, which teaches `Asks` and legs.

### Changed

- **One node contract.** A node is a `NodeDefinition` subclass whose typed
  `run` signature is its interface. Every `run` parameter is an `Input`: it
  declares a `DType` (or `Any`) and a widget. The return declaration declares
  the outputs: `Annotated[DType, Result(...)]` is one output named `result`, a
  record (a frozen dataclass) is one output per field, and `Mapping[str, Any]`
  is an interface computed by `compute_outputs`. Versions are `@version(n)`
  methods on the class, each with a `Policy` (`retries`, `delay`, `timeout`,
  `concurrency`); `@upgrade(a, b)` rewrites saved values and the class collects
  them into `cls.upgrades`; `@deprecated` retires a node or a version. A class
  without `id`, `title`, `description` or `category` is refused, and so is an
  `async def run`.
- **Fields are `Input` and `Output`** (1.x's `InputMetadata` / `OutputMetadata`).
  The hooks are `compute_inputs(declared, values)` and
  `compute_outputs(declared, values, arriving)`, with no context object; a hook
  that cannot answer raises `Refuses`, and compile reports it as the node's
  fatal `Problem`.
- **The registry holds classes.** `NodeRegistry.get(id)` returns the class,
  `registry.runner_for(id, version)` is what runs one version, and `describe()`
  is the one serialisation of a node.
- **The type vocabulary is the host's.** `DType` is a real class registered by
  `id`; `target.accepts(source)` is the one edge question. `Ref` is the address
  `"node.field"`. Conductor itself registers no type but `Series`.
- **A node pins `type` and `version`** as two fields on `GraphNode`; the
  `"id@version"` string is gone.
- **`Flow` is `Graph`, and bindings are its stored shape.** A graph is its
  nodes: `id`, `version`, `name`, `description`, `edges`, `dependencies`,
  `triggers` and `on_error_default` are gone. On `GraphNode`, `data`,
  `node_label` and `output_labels` are `bindings`, `title` and `fields`. Each
  input holds one `Edges` or `Static` binding, or none. What a node waits for
  (`dependencies_of`), which nodes are input nodes (`is_input_node`) and what a
  graph takes and returns (`CompiledGraph.interface`) are derived.
  `GraphNode.locked` closes an input to callers.
- **Compile is `CompiledGraph.from_graph`**, an immutable record asked at three
  scales: the graph (`problems`, `is_runnable`, `interface`, `execution_order`,
  `decisions`), one node (`node(node_id)`: `interface`, `iterates_on`,
  `statics`, `runner`, …) and one field (`field(ref)`: `type`, `index`,
  `binding`, `condition`). Everything wrong with the graph is a `Problem` with a
  stable `code`, a message and `details`; `accepts` is asked on every edge. A
  `GraphVersion` expands at compile under its placement's id (`outer/inner`)
  and runs as nodes of the one run.
- **The engine runs units**, a node at a row. `Series[X]` is the one
  collection. A series arriving on a scalar input makes the node iterate on its
  index, concurrently up to `Policy.concurrency`. A `Series[X]` input is a
  reduction: it receives the whole series, or the rows under each parent row on
  a child index. A row is a path on an `Index`.
- **A skip has a depth.** A node returns `SKIPPED` on the branch it did not
  take; outputs that are exclusive alternatives share a `choice`. `SKIPPED` at
  a row leaves the series downstream sparse, and above a node's rows skips
  everything under it.
- **A run has legs.** A node that needs a person returns `Asks(questions)`,
  annotated `-> X | Asks`. The leg runs on and ends `graph_pending` with every
  question; the next leg is `execute(compiled, cells=..., cache=...)`. `cache`
  records a node's outputs without running it; for an iterating node, only
  the rows its series names, so a row that ran keeps its value and a row left out
  asks again. A unit already done, or a row not yet produced, raises
  `ValueError`. Every ending carries `results` and `cells`.
- **`execute` takes `timeout_seconds`, `cells`, `cache`, `from_run` and
  `cancel`**; `context=`, `retry=` and `store_data=` are gone. It raises
  `CompilationError` for a graph that cannot run.
- **Failures carry an `ErrorCause`** (`code`, `message`, `details`, `row`) on the
  exception and on `node_error` / `graph_error`. `GraphExecutionError` and
  `GraphPendingError` are what `execute_sync` raises.
- **Events are `node_start`, `node_progress`, `node_complete`, `node_skipped`,
  `node_retry`, `node_error`** and the endings `graph_complete`,
  `graph_pending`, `graph_error`, `graph_cancelled`, `graph_timeout`.
  `flow_paused` is `graph_pending`, which carries a `pending` list; the other
  `flow_*` endings are `graph_*`. `execute(cancel=...)` stops a leg.
- **Saved and sent records are pydantic models** on `ConductorModel` (`Graph`,
  `GraphNode`, `Edges`, `Static`, `Problem`, `ErrorCause`, `Policy`,
  `NodeDescription`, `Input`, `Output`, the widgets, `Index`, …), with
  `to_yaml` / `from_yaml` / `to_path` / `from_path` beside pydantic's JSON.
- **Widgets** have a `kind` discriminator and a required `title`; `description`
  and `show_handle` on the widget belong to the field. There is no default
  widget for any type.
- **The standard library declares its own vocabulary**: `conductor_nodes.types`
  ships `Text`, `Number`, `Flag`, `Json`, and `StdlibNode` pins each node's
  `category`. Its categories are `text`, `math`, `logic`, `json`, `regex` and
  `decision`; asking `register_all` for `loop`, `while`, `subprocess` or
  `signal` raises `KeyError`.
- **Providers.** `ExecuteRequest` is `{graph, cache}`. `conductor_router`'s
  per-request hook is `from_run`; `/compile` returns the problems themselves; a
  server-sent frame dumps records, series and what they hold through pydantic,
  a float that is not a number is `null`, and a value with no JSON form raises.
  A run refused before anything runs fails `/execute-stream` the way it fails
  `/execute`, rather than streaming nothing. `graph_to_react` puts the node record under `data`,
  `react_to_graph` returns a `Graph`, and `palette_from_registry` returns
  `NodeDescription`s. The fastapi handlers are `execute_graph` /
  `execute_graph_stream`.
- **Messages say graph.** Problem, error and engine messages no longer say
  "flow".
- **Requires pydantic 2.11** or later.

### Removed

- The root exports of the engine and compiler: `from conductor import execute,
  execute_sync, compile, RetryConfig, …` no longer works.
- `@registry.node()`, `BaseNode`, `registry.register_class()`, `NodeCategory`,
  `registry.include()`, `registry.merge()`, `registry.discover()`,
  `get_latest()`, `all()`, `all_current()`, `is_deprecated()`,
  `serialize_registry`, the flattened node record, `type_str`, `label=`,
  `to_schema()` and the type-to-default-widget table.
- `InputMetadata`, `OutputMetadata`, `ComputeInputsContext`,
  `ComputeOutputsContext`, `conductor.validation` (`create_validation_model`),
  the `serialize_*_model` functions and `SerializedNode` / `SerializedInput` /
  `SerializedOutput`, `strip_sub_output_prefix`, `OUTPUT_PREFIX`,
  `finalize_connection_labels`, `OutputRef`, `WIDGET_SCHEMA_KEYS`,
  `WidgetType`, `ResultFormat`, `TypeCheckError`.
- The widgets `Output`, `Checkbox`, `Multiselect`, `DependentDropdown`,
  `HumanReview`, `TableSource`, `ConditionBuilder` and `ColumnSelect`.
- `dynamic_handles`, `is_decision`, `is_signal`, `actor`, `uses`,
  `idempotency_key`, `max_retries=` / `retry_delay=` / `timeout=` on a
  registration, and `RetryConfig`: a `Policy` on the version is the only retry.
- The marker and signal nodes: `for-each-start`, `for-each-end`,
  `while-start`, `while-end`, `subprocess-call`, `signal-wait`,
  `signal-timer`; the `compound` package and regions.
- `GraphEdge`, `compile(nodes=, edges=)`, the old `CompiledGraph` fields
  (`edge_map`, `incoming_map`, `consume_map`, …), `topological_sort`,
  `resolve_graph_inputs` / `resolve_graph_outputs`, `TypeWarning` and
  `type_check.py`.
- Shared references (`produces` / `consumes`), compensation and `on_error`,
  `FlowDependency`, `FlowTrigger`, `RegistryView`, `ExtensionResolver`.
- `conductor.expr` (CEL) and guards on edges.
- Checkpoints and resuming: `FlowCheckpoint`, `resume`, `resume_sync`,
  `HumanInputRequired`, `SignalRequired`, `FlowPausedError`.
- The events `compensation_start`, `compensation_complete`,
  `compensation_failed` and `signal_waiting`.
- The old engine's modules: `resolver`, `skip`, `state`, `request`
  (`NodeExecRequest`), `retry`, `results` (`normalize_result`,
  `project_outputs`), `checkpoint`, `store` (`FlowStore`, `store_data`),
  `conductor.types`, `EventSink` and `runtime_warning`.
- The `Flow*` error and wire names (now `Graph*`); `node_type` on errors; the
  exception aliases `NodeValidationException`, `NodeExecutionException`,
  `FlowExecutionException`, `FlowPausedException`.
- `CycleDetectionError`, `InputResolutionError`, `LoopRunawayError`,
  `SubprocessFailedError`; `fastapi.compile` (`CompileResult`), `NodeInput` and
  `EdgeInput`, and the router's `context_factory`, `strict_types`,
  `extension_resolver` and `compound_types`.
- `conductor.flow_format` (`load_flow`, `flow_to_yaml`, `dump_flow`, …): a
  graph saves itself.
- `control_operators`, its JSON mirror and `dump_operator_catalog`.
- The notebooks on control flow, shared references and the old
  human-in-the-loop; the extraction design spec, the process-standard spec and
  progress log, and the `demo/` app.

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
