# Conductor

A general-purpose Python library that runs graphs of typed nodes. It serves any host; nothing in it is fitted to one application. The user docs are `README.md` (the first page; `docs/index.md` includes it whole), `docs/writing-nodes.md` (for node authors), `docs/OVERVIEW.md`, `docs/subgraphs.md`, `docs/serving-runs.md`, `docs/design.md` (a reviewer's guide to the design), and the packaged reference `packages/conductor/src/conductor/about/llms.txt` (`python -m conductor.about`), the one complete reference. A rule is written out once, in the reference; the pages teach by example. The reference has no list of public names: a line `<!-- names: conductor.events -->` marks where a module's names go, and `conductor.about` writes them there from the module's `__all__`, each with the first paragraph of its docstring, so a public name is described in its docstring and ordered in `__all__`.

## Layout

- `packages/conductor/` (`syv-conductor`, import `conductor`): the engine. Five public modules, `__init__.py`, `widgets.py` (the controls), `events.py`, `testing/` (`check_registry`, one test for a whole registry, with the kinds of problem it returns in `problems.py`) and `about/` (the reference: `llms.txt`, the rules, and `problems.md`, the problems page, whose every kind of problem is written in from the function that declares it, with its example), plus `_internal/`, which has everything else. A concept there is a package whose `__init__.py` is its surface. Inside, the same file name means the same job in every package, and a file exists only when the concept has that job: `type.py` is the class the package is named after, `read.py` its schema for untyped data, `write.py` its JSON form, `stored.py` how a run stores one, `describe.py` what a host is shown. Any other file is named for the noun it is about. A concept with one job is one file, not a package, and a class whose methods share its state stays one file. The files: `markers.py` (`REQUIRED` and `SKIPPED`), `context.py` (`Context`), `value_type/` (one type hint as conductor reads it: `type.py` is `Type`, `fit.py` says whether an edge may join two types, `param.py` is `Param` and how a hint is split into its type and its `Param`, `choice.py` is `Choice`, `hint.py` small helpers on hints, `not_a_bool.py` the rule that `True` is not a number, applied in one walk of a type's pydantic schema, `host_type.py` is `@host_type`, how a host declares a class of its own, and the names of conductor's own classes), `gate/` (every place a value from outside enters the engine, called a door, where it is checked once by pydantic and nothing inside checks it again: `refused.py` is `RefusedError`, the one exception a door raises, with the problems it found, `returned.py` the return door, what a node's function returns read into its outputs, `edge.py` the edge door, a value an edge gives an input whose type demands more than the edge's source declares; `__init__.py` lists every door, with those still checked by hand where the value arrives and the saved-run door, which stays in `run/json/`), `series/` (`Series[T]`: `type.py` is the hint and `SeriesType`, `read.py` its pydantic schema and what a Series is read from besides a list, `stored.py` is `StoredSeries`, how a run stores a Series output and the one interface the engine reads it by, with `SeriesItems`, the list form), `table/` (`Table[R]`, the only package that imports polars: `type.py` is the hint and `TableType`, `read.py` its pydantic schema, `write.py` its JSON form, `stored.py` is `Records`, a Series of records stored as the table a node returned), `record_field.py` (`field_of`, one named field of a record, as an edge like `split.result.name` reads it), `field/` (`type.py` is `Field`, one input or output of a node read from its type hint, `describe.py` is `FieldDescription`, the same as JSON for an editor), `node_definition.py` is the node contract (`@node`, `Policy`, `Upgrade`), `problem/` a problem, the kinds of it conductor writes and the exceptions (`type.py` is `Problem` and how a package declares its kinds, `catalogue.py` the core's kinds, a function per code, imported by the modules that call them as `catalogue as problem`, `errors.py` the exceptions), `registry.py`, `hashing.py` (`short_hash`, and `canonical_json`: a value as JSON to hash, its keys sorted), `text.py` (the English that messages are written with: `comma_separated`, `plural`, `qualified_name`), `plain_json.py` (`plain_json`, a value as the plain JSON data its JSON text reads back as), `address.py` (where something is in a graph and in a run, as tuples and as text; `split_address` splits `node.name` at its first dot), `graph_record.py` (the graph record), `make_graph.py` (`make_graph`, a record written from Python), `compiled/` (a graph record compiled against the nodes it uses: `compiler.py` compiles one, `type.py` is `CompiledGraph`, `nodes.py` its nodes as the compiler makes them, `input_plans.py` how each input of a node gets its value: one input plan per input, and what each edge reads, `flat_nodes.py` the flat graph, which a run works on: every function node of the graph and of its subgraphs, `describe.py` what a host is shown), `graph.py` (`Graph`, the compiled graph a host uses: the calls that start a run, and how its runs are saved), `run/state.py` (where a run is: going on, or ended early with a cause), `outcomes.py` (the outcome of one call of a node: the five kinds of outcome, and the item indexes of a node's calls), `run/reuse.py` (the outcomes a run may reuse from the previous run: what is kept of it, the fingerprint of a call's arguments, which previous outcome a call reuses), `run/json/` (a run as JSON: `type.py` is `SavedRun` and the records inside it, `codec.py` how the runs of one graph are read and written, `read.py` outcomes from their saved lists, `write.py` outcomes as one list per output), `run/` (`type.py` is `Run`: the outcome of each call of each node; `arguments.py` the arguments of one call, read from the run's outcomes: one rule, which also says whether any item of a loop can have a value when it is asked with item indexes one index short) and `scheduler/` (one call of `run.resume()` or `run.stream()`, which is given the run's outcomes to write into and never a `Run`: `type.py` is `Scheduler`, which calls every node that is ready, on threads, with retries and timeouts where the node's `Policy` says so, `call.py` one call of a node's function with its arguments and its outputs checked, `node_events.py` is `NodeEvents`, which works out the events about each node from what the scheduler tells it, `events.py` the event classes a stream yields).
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
- **Behaviour is a value, not a flag.** A branch not chosen returns `SKIPPED`; a node that needs a person raises `Ask`. There is no role or marker on a class that tells the engine what to do.
- **Inputs and outputs a graph decides live in the hooks.** `compute_added_inputs` and `compute_outputs` are plain functions, given per version, whose parameters name the inputs they read and that return `{name: type hint}`; a parameter annotated `type` gets the type on an edge. They are the only home for inputs and outputs that depend on a graph's constants, and they are called when a graph is compiled.
- **Compiling reports, it does not raise.** A fault in a graph is a `Problem` with a stable `type`, anchored on a node, in `graph.problems`; a failing hook is one too. Only running a graph that is not runnable raises (`InvalidCallError`).
- **Retries belong to the node's `Policy`.** Only an `ExternalError`, or an exception class named in `retry_on`, is retried; a node's own `NodeError`, an output its type refuses and an `Ask` never are. Every run-time failure carries a `Problem`; a node's own exception other than a `NodeError` goes to the `conductor` logger, and only its class name reaches an event.
- **One scheduler, with or without a listener.** `run.resume()` and `run.stream()` call the nodes the same way; streaming only adds the events. A node with `concurrency` 1 and no `attempt_timeout` is called on the run's own thread; any other on threads.
- **Records are pydantic models** on `ConductorModel` (the graph record, a problem, an event, a waiting call) and cross JSON with pydantic's own `model_dump_json` and `model_validate_json`. A `Run` writes itself with `to_json()`. What compiling and running build per call stays a dataclass.
- **The widget set is open.** Conductor ships the common controls; a host subclasses `Widget` for its own. Compiling and running never read a widget. What a value must satisfy goes on `Param` (`ge`, `le`, `max_length`, ...), where it is enforced.
- **Two definitions under one id raise** when the second is added to a registry; so does one type id from two classes.
- **Fail loud.** No defensive `None` checks or silent defaults where the state means a bug.
- **Imports.** Code outside the core library imports only from the five public modules: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and `conductor.about`. Inside the core, import a name from the package that defines it, never from a file inside another package; a package's `__init__.py` is its surface and declares `__all__`, as the five public modules do.
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
  - A type shown to people is declared with `@host_type("id", title="Title")`. Nothing falls back, inspects a class or derives text
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
