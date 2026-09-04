# Conductor — architecture at a glance
## What it is

Conductor is a Python library for building DAG-based workflow and agent execution systems. Any tool where users wire nodes together — visually in a flow builder, programmatically in a script — can sit on top of it. The core is host-agnostic: no FastAPI, no database, no auth. The only hard dependency is pydantic.

Three uv-workspace packages ship today:

| Package | What it is |
|---------|------------|
| **`conductor`** | The engine: the node contract, the type mechanism, compile, execute, widgets, errors. |
| **`conductor-nodes`** | Standard-library nodes — `text`, `math`, `logic`, `json_ops`, `regex_ops`, `decision` — in a four-word vocabulary of their own. |
| **`conductor-providers`** | Framework adapters. `conductor_providers.react` round-trips graphs through ReactFlow JSON and builds the palette; `conductor_providers.fastapi` mounts a router. |

Ruff-clean, CI on every PR.

## Design principle — one declaration, three consumers

A node is a class whose typed `run` signature is its interface:

```python
class Uppercase(NodeDefinition):
    id = "uppercase"
    title = "Uppercase"
    description = "Capitalizes text"
    category = "text"

    def run(self, text: Annotated[Text, Textarea(title="Input")]) -> Annotated[Text, Result(title="Output")]:
        return Text(text.upper())
```

That one declaration drives three things: **execution** (the method runs as-is, a fresh instance per call), **validation** (a pydantic model is built from the `Input` records) and **rendering** (`describe()` is the palette entry — dtype, widget, title, choices — dumped through pydantic for the UI). No parallel schemas, no framework coupling, no sync points to forget.

Every value on a wire has a `DType`. Conductor declares none: a host says what `Text`, `Number` or `Document` is, and `target.accepts(source)` is the one wiring question. `Series[X]` is the one collection.

Several versions live in one class (`@version(2)` on the method named `run`, `@version(1)` on an older one), each with its own signature and `Policy`; `@upgrade(1, 2)` rewrites saved values; `@deprecated` retires a node or a version. A placement pins `type` and `version`, so existing flows keep working across library evolution.

## Widgets — one class per UI control

Every control is a frozen record with a `kind` discriminator: `Text`, `Textarea`, `Number`, `Range`, `Dropdown`, `EntityDropdown`, `Switch`, `DatePicker`, `FileUpload`, `List`, `Tags`, `TableInput`, `SchemaBuilder`, `CodeEditor`, `TemplateTextarea`, `IfElseBuilder`, `ConnectionList`. `AnyWidget` is their discriminated union, so a generic frontend can render any registered node by reading the palette.

Conductor ships no default widget for any type: the same `Text` may be a textarea, a single line or a dropdown, so every input declares its own. Vocabulary inside a control — a dropdown's `choices`, a condition builder's `operators` — is the host's, carried as data.

Full catalog: [`widgets.md`](widgets.md). Hands-on tour: [`examples/08_widgets.ipynb`](../examples/08_widgets.ipynb).

## Three phases: declare → compile → execute

Each phase fails fast on problems the next can't handle.

- **Declaring** a node checks it at import: a missing `id`, `title`, `description` or `category`, a parameter without a widget or a `DType`, a return without a `Result` fail with the traceback at the class. `NodeRegistry.register(cls)` adds the catalogue rules (versions from 1 with no holes, a deprecated version pointing somewhere).
- **`compile(flow, registry)`** validates node types, that every wire names an existing node, and cycles, and asks each placement's roster hooks. Returns an immutable `CompiledGraph`. Nothing runs yet.
- **`execute(compiled)`** is an async generator yielding events: `node_start`, `node_complete`, `node_retry`, `node_skipped`, `flow_complete`, … . `execute_sync(compiled)` is a blocking wrapper; `collect(execute(...))` is the notebook idiom.

## Execution — eager parallel with retry

Nodes dispatch the moment their dependencies complete. Independent branches overlap:

```
  A ──> C ──┐
            ├──> E        sequential: 5 × 0.3s = 1.5s
  B ──> D ──┘             eager:            ≈  0.9s
```

Sync `run` methods are offloaded to `asyncio.to_thread` so they don't block the event loop. There is no sequential-execute flag; eager is the only mode.

Retries are first-class. A version's `Policy` carries `retries`, `delay`, `timeout` and `concurrency`; a run-level `RetryConfig` covers nodes whose policy sets none. Validation failures are never retried (bad input won't fix itself); `NodeExecutionError` and `NodeConnectionError` are. Every attempt emits a `node_retry` event.

Errors carry structured context (`node_id`, `node_type`, original exception) so host apps can log, display, or route them without re-parsing message strings:

```
ConductorError
├── CompilationError (CycleDetectionError)
├── NodeError (Validation, Execution, Timeout, Connection)
├── InputResolutionError
└── FlowExecutionError
```

**Branching** is a value. A node returns `SKIPPED` on the branch it did not take, downstream nodes fed only `SKIPPED` are skipped in turn, and outputs that are exclusive alternatives share a `choice` so an editor knows exactly one arrives. There is no role or flag on the class telling the engine what to do.

## Bindings — one input, one source

A flow is its nodes and nothing else; there is no edge list. Each placement says per input where the value comes from:

```python
GraphNode("mapper",   "build-map", 1, bindings={"seed": Static(value="x")})
GraphNode("redactor", "redact",    1, bindings={"mapping": Sources(refs=(Ref("mapper", "result"),))})
# the Sources binding is the wire and the dependency
```

A `Sources` holds refs in operand order; a `Static` is what the author typed; an absent binding means the declared default. Dependencies, cycle detection and what the flow itself takes and returns are derived from the bindings.

## Standard nodes + frontend providers

**`conductor-nodes`** ships the usual suspects so downstream flows don't re-author them. Each category module exposes `NODES` and `register(registry)`; `register_all(registry)` pulls in everything:

```python
from conductor_nodes import register_all
register_all(reg)   # or register_all(reg, categories=["text", "math"])
```

Its nodes are declared in `conductor_nodes.types` — `Text`, `Number`, `Flag`, `Json` — because a node library has to say what its nodes take. Node ids are category-prefixed (`text-uppercase`, `math-add`, …) so they don't collide with application-level ids.

**`conductor-providers`** is the adapter layer between conductor's Python objects and specific frontend frameworks:

```python
from conductor_providers import react

palette = react.palette_from_registry(registry)  # [cls.describe() ...]
flow_json = react.graph_to_react(flow)           # conductor → ReactFlow
flow2 = react.react_to_graph(flow_json)          # ReactFlow → conductor
```

`conductor_providers.fastapi.conductor_router(registry)` mounts `/nodes`, `/compile`, `/execute`, `/execute-stream` and `/entities/{kind}`. New providers (Svelte, Vue, Gradio, …) are sibling subpackages — no abstract base class to satisfy.

## Runnable library reference

`python -m conductor.about` prints the packaged reference text from inside the installed wheel:

```bash
python -m conductor.about                 # full reference
python -m conductor.about sections        # list topic slugs
python -m conductor.about retry           # just that section (prefix match)
```

Same text programmatically: `from conductor.about import get_content, get_section`.

## Working agreements

- **CI runs ruff + pytest on every PR** (`.github/workflows/ci.yml`). Locally: `uvx ruff check .` and `uv run pytest tests/`.
- **Docs drift is actively audited.** `/docs-audit` Claude Code slash command runs on-demand after feature sessions; a weekly CI audit opens a PR as a safety net. `CLAUDE.md` and `llms.txt` should always match the shipped surface.
- **Notebook outputs are stripped on commit** by the `nbstripout` pre-commit hook; readers run cells locally to see values.
- **Duplicate node registration is a clear error**, not a silent overwrite — two classes under one id raise.

## Further reading

- [`README.md`](../README.md) — install, quickstart, usage recipes
- [`CLAUDE.md`](../CLAUDE.md) — architecture + conventions (primary context for agent sessions)
- [`packages/conductor/src/conductor/about/llms.txt`](../packages/conductor/src/conductor/about/llms.txt) — the packaged reference, accessible via `python -m conductor.about`
- [`examples/*.ipynb`](../examples/) — tutorial notebooks
