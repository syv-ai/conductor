# Progress — Conductor Process-Standard Changes

All ten features from `spec.md` plus the cross-cutting YAML file format are
**implemented and tested**. The existing 289 tests still pass, and **82 new
tests** cover the new surface area (total: **371 tests**, all green).

A second review pass tightened the implementation:

* **Inverted edge map** (`CompiledGraph.incoming_map`) — `should_skip_node`
  and `InputResolver.resolve` no longer scan the whole edge map per node.
* **Param metadata cache** on `InputResolver` — per-node-type lookup is O(1)
  instead of a linear scan over inputs.
* **`FlowStore.to_dict()`** — public API replaces `._data` accesses across
  the engine, compound nodes, and subprocess.
* **`dataclasses.replace()`** — subprocess input mapping rebuilds nodes
  cleanly instead of using `object.__setattr__` on frozen dataclasses.
* **`_parse_consumes` helper** — flow-format consume parsing is no longer a
  tangled dict comprehension; it validates shape and raises clear errors.
* **Node-vs-flow timeout detection** — the engine now correctly
  distinguishes per-node timeouts (retryable, `is_timeout: true` in
  `node_error`) from flow-level timeouts (`flow_timeout`).
* **`Actor._KINDS` as `ClassVar`** — explicit marker so the dataclass
  machinery never treats it as a field.
* **5 new integration tests** that combine decision + compensation,
  while + decision, subprocess with full flow metadata, YAML round-trip of
  every new field, and actor discovery.

## Status at a glance

| # | Feature | Status | Where |
| - | ------- | ------ | ----- |
| 1 | Decision node + edge guards | ✅ Done | `conductor/graph/compiler.py`, `conductor_nodes/decision.py`, tests in `test_decision.py` |
| 2 | CEL expression language | ✅ Done | `conductor/expr/` module, tests in `test_expr.py` |
| 3 | Actor metadata on nodes | ✅ Done | `conductor/registry/definition.py::Actor`, tests in `test_actor_deps_triggers.py` |
| 4 | Top-level dependencies | ✅ Done | `FlowDependency` + compile validation, tests in `test_actor_deps_triggers.py` |
| 5 | Top-level triggers | ✅ Done | `FlowTrigger` on `Flow`, tests in `test_actor_deps_triggers.py` |
| 6 | Per-node timeout + idempotency_key | ✅ Done | `conductor/registry/__init__.py`, `conductor/execution/engine.py`, tests in `test_timeout_idempotency.py` |
| 7 | While / until compound region | ✅ Done | `conductor/compound/while_loop.py`, `conductor_nodes/while_loop.py`, tests in `test_while_loop.py` |
| 8 | Subprocess as first-class compound | ✅ Done | `conductor/compound/subprocess.py`, `conductor_nodes/subprocess.py`, `SubprocessRegistry`, tests in `test_subprocess.py` |
| 9 | Compensation / saga support | ✅ Done | `conductor/execution/engine.py::_run_compensation`, tests in `test_compensation.py` |
| 10 | Signal / event node | ✅ Done | `conductor/errors.py::SignalRequired`, `conductor_nodes/signal.py`, tests in `test_signal.py` |
| — | YAML/JSON flow file format | ✅ Done | `conductor/flow_format/`, tests in `test_flow_format.py` |

## What changed

### New top-level exports (`conductor`)

* `Flow`, `FlowDependency`, `FlowTrigger` — declarative top-level metadata.
* `Actor` — structured actor metadata (`{kind, role}`).
* `WHILE`, `SUBPROCESS` — compound types alongside the existing `FOR_EACH`.
* `SubprocessRegistry` — lookup for sub-flow references.
* `SignalRequired`, `LoopRunawayError`, `SubprocessFailedError` — new errors.
* `execute`/`resume` — `resume` is now a public async entrypoint.
* `expr` submodule — CEL parser/evaluator.

### New modules

* **`conductor/expr/`** — self-contained CEL evaluator (no external dep). Handles
  literals, arithmetic/comparison/logical ops, ternary, in-operator, identifier
  resolution (dotted + bracket), method calls, and a library of built-ins
  (`size`, `has`, `contains`, `startsWith`, `endsWith`, `matches`, `string`,
  `int`, `double`, `bool`, `min`, `max`, `abs`, `lower`, `upper`, `exists`).
* **`conductor/compound/while_loop.py`** — while/until compound with
  `max_iterations` cap, `negate` flag, and `LoopRunawayError` when the cap is
  exceeded. Sequential only; cancellation-aware between iterations.
* **`conductor/compound/subprocess.py`** — first-class subprocess calls. A
  sub-flow is pre-registered in a `SubprocessRegistry`; the caller's
  `subprocess-call` node looks it up, compiles it on the outer registry,
  and runs it synchronously. Runtime depth cap prevents runaway recursion.
* **`conductor/flow_format/`** — YAML/JSON round-trip for `Flow` objects.
  Supports a stable, author-friendly shape:
  ```yaml
  id: my-flow
  version: 1
  dependencies: [{id: stripe, kind: api}]
  triggers:   [{id: nightly, kind: schedule, config: {cron: "0 9 * * 1"}}]
  nodes: [...]
  edges: [...]
  ```
* **`conductor-nodes/decision.py`**, **`while_loop.py`**, **`subprocess.py`**,
  **`signal.py`** — marker nodes for the new compound types and the signal
  runtime primitive.

### Engine changes

* **Eager scheduler** gained:
  * `skipped_edges: set[str]` on `FlowRunState`, used by the resolver and skip
    checker to honor decision-node branching.
  * `completed_order: list[str]` for reverse-topological compensation walks.
  * `idempotency_keys: dict[str, str]` surfaced on `node_start` events and
    injected into node functions that declare an `idempotency_key` param.
  * Per-node timeout wrapping `asyncio.wait_for`.
  * Compensation cascade on flow failure, with `compensation_start`,
    `compensation_complete`, and `compensation_failed` events.
  * `on_error` per-node policy: `fail` (default), `continue`, `compensate`.
  * Signal wait / resume parity with HITL (checkpoint carries
    `signal_name`, `correlation`, `signal_timeout_seconds`).

* **Compiler** gained:
  * Decision-node validation: exactly one else-edge, at least one guarded
    edge, `when` only on decision outgoing edges, CEL parse errors fatal.
  * Dependency usage validation (`uses=` must reference declared
    `dependencies:`).
  * Compensation reference validation (existence, no self-loops, valid
    `on_error` policy).
  * Idempotency-key CEL pre-parsing (compile-time error on bad syntax).
  * Subprocess region support (start==end single-node compound).

### Graph model

* `GraphEdge` gained `when: str | None` (CEL) and `priority: int`.
* `GraphNode` gained `compensation: str | None` and `on_error: str | None`.
* `Flow`, `FlowDependency`, `FlowTrigger` dataclasses added.

### Registry

* `@registry.node()` and `@category.node()` gained six new kwargs:
  `actor`, `timeout` (seconds or ISO 8601), `idempotency_key` (CEL expression),
  `uses` (list of dependency ids), `is_decision`, `is_signal`.
* ISO 8601 duration parsing (`PT30S`, `PT5M`, `PT1H`) and shorthand (`30s`,
  `250ms`, `5m`, `2h`) for `timeout`.

### Events

* New events: `compensation_start`, `compensation_complete`,
  `compensation_failed`, `signal_waiting`.
* `node_start` carries an optional `idempotency_key`.
* `node_error` carries an optional `is_timeout`.

### Errors

* Added `SignalRequired`, `LoopRunawayError`, `SubprocessFailedError`.

### Checkpoints

* `FlowCheckpoint` now carries `skipped_edges`, `signal_name`, `correlation`,
  `signal_timeout_seconds` so decision-node branch state and signal metadata
  survive pause/resume. `from_dict` tolerates older checkpoints that lack
  these fields.

## Tests

New files under `tests/test_core/`:

* `test_expr.py` — 29 tests covering CEL parse, eval, identifiers,
  methods, error paths, and introspection.
* `test_decision.py` — 8 tests: compile-time validation (missing else, missing
  guards, else count, when on non-decision, invalid CEL), runtime branching
  (guard taken, else taken, priority ordering).
* `test_while_loop.py` — 4 tests: counting, zero iterations, runaway, `until`
  semantics.
* `test_timeout_idempotency.py` — 8 tests: per-node timeout (numeric, ISO 8601,
  shorthand), invalid timeout, idempotency evaluation, stability across
  retries, compile-time CEL validation, event surfacing.
* `test_actor_deps_triggers.py` — 9 tests: Actor coercion and validation,
  `uses:` dependency validation, triggers-are-metadata-only.
* `test_compensation.py` — 7 tests: validation (missing target, self-comp,
  invalid on_error), cascade ordering, compensation-only nodes not running
  on the happy path, on_error=continue behavior, events emitted.
* `test_subprocess.py` — 3 tests: sub-flow runs, missing sub-flow errors,
  registry requires id.
* `test_signal.py` — 2 tests: pause-and-resume round trip, timeout
  surfacing.
* `test_flow_format.py` — 7 tests: dict round-trip, YAML round-trip, JSON
  file round-trip, YAML file round-trip, minimal YAML, compensation in YAML,
  invalid input.

## Linting

`uvx ruff check .` — **All checks passed**.

## How to run

```bash
uv sync
uv run pytest tests/ -v
uvx ruff check .
```

**Final status:** 371 tests pass, ruff clean, all features implemented.
