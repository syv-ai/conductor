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

These shapes are part of the `1.0.0` public surface and are not deprecated, but
they are likely targets for a future major bump:

- `conductor.errors` legacy exception aliases — `NodeValidationException`,
  `NodeExecutionException`, `FlowExecutionException`, `FlowPausedException`.
  These are kept as aliases of the `*Error` names for back-compat.
- `result` key duplication in `normalize_result` for dict returns
  (`{result: dict, **dict}`) — surface area that exists for back-compat with
  early node authors.
- Cross-package `==` pin in `syv-conductor[all]` — could relax to
  `~=` once the providers/nodes packages stabilize independently.

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

[Unreleased]: https://github.com/syvai/conductor/compare/v1.2.0...HEAD
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
