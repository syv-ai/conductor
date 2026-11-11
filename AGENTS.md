# Conductor

A general-purpose Python library that runs graphs of typed nodes. It serves any host; nothing in it is fitted to one application. The user docs are `README.md` (the first page; `docs/index.md` includes it whole), `docs/writing-nodes.md` (for node authors), `docs/OVERVIEW.md`, `docs/serving-runs.md`, and the packaged reference `packages/conductor/src/conductor/about/llms.txt` (`python -m conductor.about`), the one complete reference. A rule is written out once, in the reference; the pages teach by example. The reference holds no list of public names: a line `<!-- names: conductor.events -->` stands where a module's names go, and `conductor.about` writes them there from the module's `__all__`, each with the first paragraph of its docstring, so a public name is described in its docstring and ordered in `__all__`.

## Layout

- `packages/conductor/` (`syv-conductor`, import `conductor`): the engine. Five public modules, `__init__.py`, `widgets.py` (the controls), `events.py`, `testing.py` (`check_registry`, one test for a whole registry) and `about/` (the reference), plus `_internal/`, which holds everything else. A concept there is a package whose `__init__.py` is its surface. Inside, the same file name means the same job in every package, and a file exists only when the concept has that job: `type.py` is the class the package is named after, `read.py` its schema for untyped data, `write.py` its JSON form, `held.py` how a run holds one, `describe.py` what a host is shown. Any other file is named for the noun it holds. A concept with one job is one file, not a package, and a class whose methods share its state stays one file. The files: `fields.py` reads a signature's annotations into fields and says which types an edge may join, `node_definition.py` is the node contract (`@node`, `Policy`, `Upgrade`), `problem/` a problem, the kinds of it conductor writes and the exceptions (`type.py` is `Problem` and how a package declares its kinds, `catalogue.py` the core's kinds, a function per code, `errors.py` the exceptions), `kit_problem/` the kinds `check_registry` returns (`catalogue.py`, a function per code), `registry.py`, `canonical_json.py` (a value as JSON to hash, its keys sorted), `address.py` (where something is in a graph and in a run, as the tuples the engine holds and as the text a host and a save hold), `graph_record.py` (the stored graph), `make_graph.py` (`make_graph`, a record written from Python), `loaded_graph/` (a stored graph checked against the nodes it uses: `loader.py` loads one, `type.py` is `CheckedGraph` with its placed nodes, `input_plans.py` what an edge brings and how each input gets its value: the plan kinds and the cells they read, `tree.py` the nodes where they stand, which a run walks, `describe.py` what a host is shown), `graph.py` (`Graph`, a loaded graph as a host holds it: the verbs that start a run), `run_status.py` (how a run stands: going on, or ended early with a cause), `units.py` (what a node came to for one item: the five kinds of unit, and which items a node has), `earlier_work.py` (the work a run may take from the run it went on from: what is kept of it, the fingerprint of what a unit received, which earlier unit a unit takes), `saved_run/` (a run as JSON: `type.py` is `SavedRun` and the records inside it, `form.py` how the runs of one graph are read and written, `read.py` units from their columns, `write.py` units as columns), `run/` (`type.py` is `Run`: what each node came to on each call; `call_inputs.py` what a call receives: the node rule, read for one item or for an item nobody knows) and `leg/` (one leg, which is given what it changes and reads and never a `Run`: `walk.py` is `Leg`, the scheduler and retries, `call.py` one call of a node with what it receives and returns checked, `events.py` the records a stream yields).
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
- **Retries belong to the node's `Policy`.** Only an `ExternalError`, or an exception class named in `retry_on`, is retried; a node's own `NodeError`, an output its type refuses and an `Ask` never are. Every run-time failure carries a `Problem`; a node's own exception other than a `NodeError` goes to the `conductor` logger, and only its class name reaches an event.
- **A leg is one walk, with or without a listener.** `run.resume()` and `run.stream()` run the same leg; streaming only adds the events. A node with `concurrency` 1 and no `attempt_timeout` is called on the leg's own thread; any other runs on threads.
- **Records are pydantic models** on `ConductorModel` (the stored graph, a problem, an event, a waiting unit) and cross JSON with pydantic's own `model_dump_json` and `model_validate_json`. A `Run` writes itself with `to_json()`. What loading and the engine build per call stays a dataclass.
- **The widget set is open.** Conductor ships the common controls; a host subclasses `Widget` for its own. Loading and the engine never read a widget. What a value must satisfy goes on `Param` (`ge`, `le`, `max_length`, ...), where it is enforced.
- **Two definitions under one id raise** when the second is added to a registry; so does one type id from two classes.
- **Fail loud.** No defensive `None` checks or silent defaults where the state means a bug.
- **Imports.** Code outside the core library imports only from the five public modules: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and `conductor.about`. Inside the core, import a name from the package that holds it, never from a file inside another package; a package's `__init__.py` is its surface and declares `__all__`, as the five public modules do.
- **Identifiers are English.**
- **Notebook outputs are stripped on commit.** Run the cells to see values.
- **Keep the docs true.** A change to public API updates `README.md`, `docs/`, `llms.txt` and the skills in the same change, and adds a `CHANGELOG.md` line under `[Unreleased]`. Keep every Python block in the docs runnable. Docs describe what the code does; where a commit message and the code disagree, trust the code.

## Rules for the 2.0 rethink (temporary)

This section holds while the 2.0 rethink is under review, and goes when it's merged. It adds to the rules above.

- **Trace types from the doorway.** For each function, method and class, start where outside code or data comes in
  and work inward. At each layer, ask what has to change earlier so the types here are certain, and change it there.
  Then delete the check that's no longer needed, instead of keeping it as a safety net.
- **A Python argument is typed, not checked.** A `NodeDefinition` given to a registry, `extra=`, `Param(...)`, a
  policy, a widget, or `make_graph`'s nodes is enforced by its declared type and the type checker. It is never
  re-checked at runtime: no `isinstance` guard, no `_raise_unless_...`, no `checked_...` helper.
- **Validate only where untyped data arrives:**
  - JSON being loaded, such as a graph record or a saved run;
  - values a caller gives to a run;
  - what a node returns.

  Each is validated once, by pydantic, where it arrives. Inside that boundary, trust the types.
- **Make invalid states unrepresentable.** A record whose fields hide an either/or becomes two types, and the code
  dispatches on the type. A field that must hold JSON is typed as JSON. A union in a field usually means a split
  further up, so trace it and fix it there.
- **No container kept only to be passed through.** Use typed fields, not a dict, on an object that is built once.
- **What a person sees is declared, never guessed.**
  - A node's `title` and `description` are required on `@node`, and a docstring never reaches a user.
  - A field's label is its name, made readable, unless `Param(title=...)` says otherwise.
  - A type shown to people declares `type_id` and `type_title`. Nothing falls back, inspects a class or derives text
    from code.
- **No helper without a job.** A helper with one caller and no real work of its own is folded into its caller.
  Every helper that stays has a purpose a reader can see, and a test.
- **Errors end in `Error`.** A name says what went wrong. `Ask` is the one exception that isn't an error.
- **Google-style docstrings.**
  - Every public and internal function, method and class says what it does.
  - Use `Args`, `Returns`, `Raises` and `Attributes`, and say what each field holds.
  - No plan references, no wording about time, no coined metaphors.
- **No host decides conductor's shape.** A finding that comes from one host changes conductor only if it holds as a
  general library need.

## Public API

The public API is what the five public modules list in `__all__`. Everything in `conductor._internal`, including anything with a leading underscore, is internal. A breaking change to the public API needs a major version.
