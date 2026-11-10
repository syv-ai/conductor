# Conductor

A general-purpose Python library that runs graphs of typed nodes. It serves any host; nothing in it is fitted to one application. The user docs are `README.md` (the first page; `docs/index.md` includes it whole), `docs/writing-nodes.md` (for node authors), `docs/OVERVIEW.md`, `docs/serving-runs.md`, and the packaged reference `packages/conductor/src/conductor/about/llms.txt` (`python -m conductor.about`), the one complete reference. A rule is written out once, in the reference; the pages teach by example.

## Layout

- `packages/conductor/` (`syv-conductor`, import `conductor`): the engine. Four public modules, `__init__.py`, `widgets.py` (the controls), `events.py` and `testing.py` (`check_registry`, one test for a whole registry), plus `about/` and `_internal/`, which holds everything else: `fields.py` reads a signature's annotations into fields and says which types an edge may join, `node_definition.py` is the node contract (`@node`, `Policy`, `Upgrade`), `errors.py` the problems and exceptions, `registry.py`, `record.py` (the stored graph, `make_graph`), `graph.py` (loading: `Graph`, where each node runs), `run.py` (`Run`: what each node came to on each row, and its JSON) and `leg.py` (one leg: the scheduler, retries, events).
- `packages/conductor-nodes/` (`syv-conductor-nodes`, import `conductor_nodes`): nodes that only compute, a registry per module.
- `packages/conductor-providers/` (`syv-conductor-providers`, import `conductor_providers`): `fastapi`, the router that serves runs (`run_router`) and the `RunStore` a host writes for it.
- `tests/`: `test_core`, `test_nodes` (the node package), `test_providers` (`docs/serving-runs.md` on two uvicorn workers, and event JSON).
- `examples/`: Jupyter notebooks. `skills/`: agent skills for adding a node and building a graph.

Pushing a `v*` tag publishes the three packages to PyPI, at one version. The two companions import only conductor's public modules.

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
- **One node contract.** A node is a function under `@node`, its typed signature its interface, and `@node` returns a frozen `NodeDefinition` that registers nothing. Any type pydantic can carry travels on edges as it is; a host's own type needs no base class.
- **Behaviour is a value, not a flag.** A branch not taken returns `SKIPPED`; a node that needs a person raises `Ask`. There is no role or marker on a class that tells the engine what to do.
- **Placement shape lives in the hooks.** `compute_added_inputs` and `compute_outputs` are plain functions, given per version, whose parameters name the inputs they read and that return `{name: type hint}`; a parameter annotated `type` gets the type on an edge. They are the only home for fields that depend on configuration, and they run when a graph is loaded.
- **Loading reports, it does not raise.** A fault in a graph is a `Problem` with a stable `type`, anchored on a node, in `graph.problems`; a failing hook is one too. Only running a graph that is not runnable raises (`InvalidCallError`).
- **Retries belong to the node's `Policy`.** Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried. Every run-time failure carries a `Problem`; a node's own exception other than a `NodeError` goes to the `conductor` logger, and only its class name reaches an event.
- **A leg is one walk, with or without a listener.** `run.resume()` and `run.stream()` run the same leg; streaming only adds the events. A node with `concurrency` 1 and no `attempt_timeout` is called on the leg's own thread; any other runs on threads.
- **Records are pydantic models** on `ConductorModel` (the stored graph, a problem, an event, a waiting unit) and cross JSON with pydantic's own `model_dump_json` and `model_validate_json`. A `Run` writes itself with `to_json()`. What loading and the engine build per call stays a dataclass.
- **The widget set is open.** Conductor ships the common controls; a host subclasses `Widget` for its own. Loading and the engine never read a widget. What a value must satisfy goes on `Param` (`ge`, `le`, `max_length`, ...), where it is enforced.
- **Two definitions under one id raise** when the second is added to a registry; so does one type id from two classes.
- **Fail loud.** No defensive `None` checks or silent defaults where the state means a bug.
- **Imports.** Code outside the core library imports only from the four public modules: `conductor`, `conductor.widgets`, `conductor.events` and `conductor.testing`. Inside the core, import a name from the module that defines it. Only those four and `conductor.about` declare `__all__`.
- **Identifiers are English.**
- **Notebook outputs are stripped on commit.** Run the cells to see values.
- **Keep the docs true.** A change to public API updates `README.md`, `docs/`, `llms.txt` and the skills in the same change, and adds a `CHANGELOG.md` line under `[Unreleased]`. Keep every Python block in the docs runnable. Docs describe what the code does; where a commit message and the code disagree, trust the code.

## Public API

The public API is what the four public modules list in `__all__`. Everything in `conductor._internal`, including anything with a leading underscore, is internal. A breaking change to the public API needs a major version.
