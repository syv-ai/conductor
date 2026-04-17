# Conductor — architecture at a glance
## What it is

Conductor is a Python library for building DAG-based workflow and agent execution systems. Any tool where users wire nodes together — visually in a flow builder, programmatically in a script — can sit on top of it. The core is host-agnostic: no FastAPI, no database, no auth. Only hard dependency is pydantic.

Three uv-workspace packages ship today:

| Package | What it is |
|---------|------------|
| **`conductor`** | The engine: registry, compile, execute, widgets, errors, compound regions, shared references. |
| **`conductor-nodes`** | Standard-library nodes — `text`, `math`, `logic`, `loop` markers, `json`, `regex`. |
| **`conductor-providers`** | Framework adapters. `conductor_providers.react` round-trips graphs through ReactFlow JSON today; more providers live in sibling subpackages. |

271 tests, ruff-clean, CI on every PR.

## Design principle — one annotation, three consumers

A node is a plain Python function with `Annotated[T, Widget]` parameters:

```python
@registry.node("uppercase", version=1, name="Uppercase", description="Capitalizes text")
def uppercase(
    text: Annotated[str, Text(label="Input")],
) -> Annotated[str, Output(label="Output")]:
    return text.upper()
```

That single annotation drives three things: **backend execution** (the function runs as-is), **input validation** (a Pydantic model is generated from the signature), and **frontend rendering** (widget type + label + choices serialize to JSON for the UI). No parallel schemas, no framework coupling, no sync points to forget.

Nodes are versioned as `base_id@version`. Registering `echo@2` never overwrites `echo@1`, so existing flows keep working across library evolution. Class-based nodes (`BaseNode` subclasses) exist for the rare cases needing direct run-state access; 95% are plain functions.

## Widgets — one class per UI control, inferred for common types

Every `WidgetType` enum value has a concrete Python class (`Text`, `Textarea`, `Number`, `Range`, `Dropdown`, `Multiselect`, `Checkbox`, `Switch`, `DatePicker`, `FileUpload`, `List`, `SchemaBuilder`, `CodeEditor`, `TemplateTextarea`, `DependentDropdown`, `EntityDropdown`, `IfElseBuilder`, `ConnectionList`, `Output`), so a generic frontend can render any registered node by reading the registry — no bespoke backend code per widget.

When a parameter has no widget on its annotation, the registry infers a sensible default from the Python type: `str → Text`, `int → Number(integer_only=True)`, `float → Number`, `bool → Checkbox`, `Date → DatePicker`, `list[T] → List(item_widget=default(T))`, `dict → SchemaBuilder`. Explicit `Annotated[T, Widget(...)]` always wins. So `def f(x: int)` is enough for the common case; annotate when you want constraints, a different widget, or a prettier label.

Full catalog + recipe for adding new widgets: [`widgets.md`](widgets.md). Hands-on tour: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

## Three phases: register → compile → execute

Each phase fails fast on problems the next can't handle.

- **`compile(nodes, edges, registry)`** validates node types, edge targets, cycles, edge/consume type compatibility, and shared-reference bindings. Discovers compound regions (for-each). Returns an immutable `CompiledGraph`. Nothing runs yet.
- **`execute(compiled)`** is an async generator yielding events: `node_start`, `node_complete`, `node_retry`, `flow_paused`, `flow_complete`, … . `execute_sync(compiled)` is a blocking wrapper; `collect(execute(...))` is the notebook idiom.

## Execution — eager parallel with retry

Nodes dispatch the moment their dependencies complete. Independent branches overlap:

```
  A ──> C ──┐
            ├──> E        sequential: 5 × 0.3s = 1.5s
  B ──> D ──┘             eager:            ≈  0.9s
```

Sync node functions are offloaded to `asyncio.to_thread` so they don't block the event loop. There is no sequential-execute flag; eager is the only mode.

Retries are first-class. Per-node `max_retries` / `retry_delay` (exponential backoff), or a global `RetryConfig`; node-level wins when both are set. Validation failures are never retried (bad input won't fix itself); `NodeExecutionError` and `NodeConnectionError` are. Every attempt emits a `node_retry` event.

Errors carry structured context (`node_id`, `node_type`, original exception) so host apps can log, display, or route them without re-parsing message strings:

```
ConductorError
├── CompilationError (CycleDetectionError, TypeCheckError)
├── NodeError (Validation, Execution, Timeout, Connection)
├── InputResolutionError
├── FlowExecutionError
├── FlowPausedError / HumanInputRequired
```

**Human-in-the-loop** uses the same pause mechanism. A node raises `HumanInputRequired(prompt, schema)`, the engine checkpoints state to a JSON-serializable dict, you persist it anywhere (hours or days), and later call `resume()` with the human's response. Shared-reference values and `FlowStore` data both survive the cycle.

## Shared references — fan-out and cross-region wiring

Edges work for 1:1 data flow but are awkward for N:1 fan-out (one value feeding many consumers) and impossible across for-each boundaries (feeding a constant into a loop body). Shared references fill both gaps — without adding a new node type. The library author doesn't decide what's shareable; the flow builder does, per instance:

```python
GraphNode("mapper",   "build-map@1", ..., produces={"result": "pseudonym map"})
GraphNode("redactor", "redact@1",    ..., consumes={"mapping": ("mapper", "result")})
# no edge between them — the consume binding is the dependency
```

Reference identity is `(producer_id, output_handle)`; the label is UI-only so renames never break subscribers. Validated at compile time (cycles, type compatibility, no collision with explicit edges, single producer per reference). Consumers inside a for-each body see the same producer value on every iteration — broadcast, not per-iteration.

v1 constraint: producers must be top-level (not inside a compound region). Consumers can be anywhere. Full spec: [`shared-references.md`](shared-references.md).

## Standard nodes + frontend providers

**`conductor-nodes`** ships the usual suspects so downstream flows don't re-author them. Each category exposes a `register(registry)` function; `register_all(registry)` pulls in everything:

```python
from conductor_nodes import register_all
register_all(reg)   # or register_all(reg, categories=["text", "math"])
```

Node IDs are category-prefixed (`text-uppercase`, `math-add`, …) so they don't collide with application-level IDs. The `for-each-start` / `for-each-end` markers keep their canonical names because the `FOR_EACH` compound discovers them by prefix.

**`conductor-providers`** is the adapter layer between conductor's Python objects and specific frontend frameworks. Today it ships `conductor_providers.react`:

```python
from conductor_providers import react

palette = react.palette_from_registry(registry)  # sidebar JSON
flow    = react.graph_to_react(nodes, edges)     # conductor → ReactFlow
nodes2, edges2 = react.react_to_graph(flow)      # ReactFlow → conductor
```

Round-trip preserves `produces`/`consumes` (tuples ↔ lists across JSON). Unknown wire keys are ignored so hosts can decorate without breaking compatibility. New providers (Svelte, Vue, Gradio, …) are sibling subpackages — no abstract base class to satisfy.

## Runnable library reference

`python -m conductor.about` prints the full reference text from inside the installed wheel. Useful when an agent in a downstream project needs context without repo access:

```bash
python -m conductor.about                 # full reference
python -m conductor.about sections        # list topic slugs
python -m conductor.about retry           # just that section (prefix match)
```

Same text programmatically: `from conductor.about import get_content, get_section`.

## Working agreements

- **Tests first for non-trivial features.** The shared-references feature landed as a design doc + skipped spec tests first, then implementation removed the skip marker. This is the pattern for any future v2 change.
- **CI runs ruff + pytest on every PR** (`.github/workflows/ci.yml`). Locally: `uvx ruff check .` and `uv run pytest tests/`.
- **Docs drift is actively audited.** `/docs-audit` Claude Code slash command runs on-demand after feature sessions; a weekly CI audit opens a PR as a safety net. `CLAUDE.md` and `llms.txt` should always match the shipped surface.
- **Notebook outputs are stripped on commit** by the `nbstripout` pre-commit hook; readers run cells locally to see values.
- **Duplicate node registration is a clear error**, not a silent overwrite — the message tells you to bump `version`, create a fresh registry (for notebooks), or pick a different `base_id`.

## Further reading

- [`README.md`](README.md) — install, quickstart, usage recipes
- [`CLAUDE.md`](CLAUDE.md) — architecture + conventions (primary context for agent sessions)
- [`llms.txt`](llms.txt) — full API reference (also shipped inside the wheel)
- [`shared-references.md`](shared-references.md) — v1 design spec for produce/consume
- [`conductor-design.md`](conductor-design.md) — original library design document
- [`examples/*.ipynb`](examples/) — 7 tutorial notebooks covering the whole surface
