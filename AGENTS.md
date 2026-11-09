# Conductor

A general-purpose Python library that compiles and runs graphs of typed nodes. It serves any host; nothing in it is fitted to one application. The user docs are `README.md`, `docs/` and the packaged reference `packages/conductor/src/conductor/about/llms.txt` (`python -m conductor.about`).

## Layout

- `packages/conductor/` (`syv-conductor`, import `conductor`): the engine. Three public modules, `__init__.py`, `widgets.py` (the controls) and `events.py`, plus `about/` and `_internal/`, which holds everything else: `node_definition.py` is the node contract, `node_signature.py` reads a signature, `field_records.py` the records of a field, `dtype.py` / `table.py` / `ref.py` the types, `errors.py` the exceptions, `registry.py`, `graph/` (the record, compile, `CompiledGraph`) and `execution/` (the engine, events, `RunState`).
- `packages/conductor-nodes/` (`conductor_nodes`): the standard nodes and the one type of their own, `Json`.
- `packages/conductor-providers/` (`conductor_providers`): the `mermaid` adapter.
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

- **Conductor stands alone.** No host's words ("flow", "app", an access model), no host's imports, and no concrete type declared by the core. The library's word is graph.
- **One node contract.** A node is a function under `@node`, its typed signature its interface, and `@node` returns a frozen `NodeDefinition` that registers nothing. A plain Python type travels on edges as it is; a type no edge can carry (`list[X]`, a `Literal`) closes its input by itself.
- **Behaviour is a value, not a flag.** A branch not taken returns `SKIPPED`; a node that needs a person returns `AskPerson`. There is no role or marker on a class that tells the engine what to do.
- **Placement shape lives in the hooks.** `compute_added_inputs` and `compute_outputs` are plain functions, given per version, whose parameters name the inputs they read; `ArrivingType` gives the type on an edge. They are the only home for fields that depend on configuration.
- **Compile reports, it does not raise.** A fault in a graph is a `Problem` with a stable `type`, anchored on a node. Only running a graph that is not runnable raises (`InvalidCallError`).
- **Retries belong to the version's `Policy`.** Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried. Every run-time failure carries a `Problem`; a node's own exception goes to the `conductor` logger, and only its class name reaches an event.
- **Streaming is the only execution path.** `compiled.stream` is the leg; `run` and `run_sync` drain it.
- **Records are pydantic models** on `ConductorModel` and save themselves (`to_yaml`, `to_path`). What compile and the engine build per call stays a frozen dataclass.
- **The widget set is open.** Conductor ships the common controls; a host subclasses `Widget` for its own. Compile and the engine never read a widget. What a value must satisfy goes on `Param` (`ge`, `le`, `max_length`, ...), where it is enforced.
- **Two definitions under one id raise** when a registry is built; so does one type id from two classes.
- **Fail loud.** No defensive `None` checks or silent defaults where the state means a bug.
- **Imports.** Code outside the core library imports only from the three public modules: `conductor`, `conductor.widgets` and `conductor.events`. Inside the core, import a name from the module that defines it. Only those three and `conductor.about` declare `__all__`.
- **Identifiers are English.**
- **Notebook outputs are stripped on commit.** Run the cells to see values.
- **Keep the docs true.** A change to public API updates `README.md`, `docs/`, `llms.txt` and the skills in the same change, and adds a `CHANGELOG.md` line under `[Unreleased]`. Keep every Python block in the docs runnable. Docs describe what the code does; where a commit message and the code disagree, trust the code.

## Public API

The public API is what the three public modules list in `__all__`, plus `conductor_nodes` and `conductor_providers`. Everything in `conductor._internal`, including anything with a leading underscore, is internal. A breaking change to the public API needs a major version.
