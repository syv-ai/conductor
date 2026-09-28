# Conductor

A general-purpose Python library that compiles and runs graphs of typed nodes. It serves any host; nothing in it is fitted to one application. The user docs are `README.md`, `docs/` and the packaged reference `packages/conductor/src/conductor/about/llms.txt` (`python -m conductor.about`).

## Layout

- `packages/conductor/` (`syv-conductor`, import `conductor`): the engine. `node.py` is the node contract, `interface.py` reads a `run` signature, `dtype.py` / `series.py` / `ref.py` are the type mechanism, `widgets.py` the controls, `errors.py` the exceptions, `registry/`, `graph/` (the record, compile, `CompiledGraph`) and `execution/` (the engine, events, `RunState`).
- `packages/conductor-nodes/` (`conductor_nodes`): the standard nodes and their four types.
- `packages/conductor-providers/` (`conductor_providers`): `react`, `fastapi` and `mermaid` adapters, one subpackage each, with no shared base class.
- `tests/`: `test_core`, `test_nodes`, `test_providers`, `test_stress`.
- `examples/`: Jupyter notebooks. `skills/`: agent skills for adding a node and building a graph.

The three packages release together at one version, and the sibling packages pin `syv-conductor` with `==`. Pushing a `v*` tag publishes all three to PyPI.

## Commands

```bash
uv sync                      # install everything
uv run pre-commit install    # strip notebook outputs on commit
uv run pytest tests/ -q      # the whole suite
uvx ruff check .             # lint, as CI runs it
```

CI runs ruff and pytest on every PR. Tests marked `slow` time the wall clock and are skipped when `CI` is set.

## Rules

- **Conductor stands alone.** No host's words ("flow", "app", an access model), no host's imports, and no concrete type declared by the core. `tests/test_core/test_standalone.py` enforces this. The library's word is graph.
- **One node contract.** A node is a `NodeDefinition` subclass; its typed `run` signature is its interface. Every parameter an edge can reach declares a `DType` (or `Any`) inside `Annotated[..., Param(title=..., widget=...)]`. The return annotation declares the outputs. Don't add a second place to declare anything a signature already says.
- **Behaviour is a value, not a flag.** A branch not taken returns `SKIPPED`; a node that needs a person returns `Asks`. There is no role or marker on a class that tells the engine what to do.
- **Placement shape lives in the hooks.** `compute_inputs(declared, values)` and `compute_outputs(declared, values, arriving)` are the only home for inputs or outputs that depend on configuration. A hook that cannot answer raises `Refuses`.
- **Compile reports, it does not raise.** A fault in a graph is a `Problem` with a stable `code`, anchored on a node. Only `execute` on a graph that is not runnable raises (`CompilationError`).
- **Retries belong to the version's `Policy`.** Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried. Every run-time failure carries an `ErrorCause`; a foreign exception's text stays on `original` and never reaches an event.
- **Streaming is the only execution path.** `execute` is an async generator; `run` and `run_sync` drain it.
- **Records are pydantic models** on `ConductorModel` and save themselves (`to_yaml`, `to_path`). What compile and the engine build per call stays a frozen dataclass.
- **The widget set is closed.** `AnyWidget` is built from the subclasses in `widgets.py`. A new control is a class there, a test, and a component the host's frontend owes.
- **A `run` returns its declared types** (`Text(...)`, never a bare `str`).
- **Registering two classes under one id raises**; so does one type id from two classes on one registry.
- **Fail loud.** No defensive `None` checks or silent defaults where the state means a bug.
- **Imports.** Code outside the core library (conductor-nodes, providers, examples, docs, hosts) imports only from the three doors, `conductor`, `conductor.widgets` and `conductor.events`; each door lists its names in `__all__`, pinned by `tests/test_core/test_doors.py`. Code inside the core library imports a name from the module that defines it, and no other module declares `__all__`.
- **Identifiers are English.** A host's language lives in titles, descriptions and messages.
- **Notebook outputs are stripped on commit.** Run the cells to see values.
- **Keep the docs true.** A change to public API updates `README.md`, `docs/`, `llms.txt` and the skills in the same change, and adds a `CHANGELOG.md` line under `[Unreleased]`. The README's Python blocks run as a test (`tests/test_core/test_vocabulary.py`), so keep them runnable top to bottom. Docs describe what the code does; where a commit message and the code disagree, trust the code.

## Public API

The public API is what each package's `__init__` exports plus `conductor.widgets`, `conductor.errors`, `conductor.metadata`, `conductor.model` and `conductor.execution.events`. Anything else, including `_`-prefixed names, is internal. A breaking change to the public API needs a major version.
