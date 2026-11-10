# Changelog

All notable changes to `syv-conductor`; up to 1.x this also covered `syv-conductor-nodes` and
`syv-conductor-providers`, released with it at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, loading and the engine are each
replaced, and no alias keeps a 1.x name alive. A 1.x graph does not load as saved; a host converts
its stored graphs itself. What a 1.x user changes:

- **Imports.** Import from `conductor`, `conductor.widgets` and `conductor.events`; everything else is
  `conductor._internal`. Conductor depends on `polars>=2.0` and `pydantic>=2.11`.
- **A node.** `BaseNode` is a plain function under `@registry.node("id")`, or under `@node("id")`,
  which takes the same arguments and registers nothing. Either returns a frozen, callable
  `NodeDefinition`; `name=` is gone. `async def` isn't allowed. `NodeCategory` and auto-discovery are
  gone. A node takes `extra=`, the host's own JSON about it, kept and described and never read.
- **A registry.** `NodeRegistry(category=, tags=, extra=, types=)` starts empty. `@registry.node`
  declares and adds; `add(*definitions)` adds plain `@node`s, and a second node under one id raises;
  `replace` and `remove` change one. `add_subregistry(other)` copies another registry's nodes in, so
  each module keeps a registry and the host joins them. A node entering takes the registry's
  defaults: tags add up, outermost first, and its own `category` and `extra` keys win.
  `filter(predicate)` is a new registry of the nodes kept. Looping yields the definitions.
  `registry.describe()` is the palette as JSON.
- **Inputs and outputs.** The type hints are the interface: `text: str`, or
  `Annotated[str, Param(title=..., widget=..., ge=..., max_length=...)]`. `-> str` is one output
  named `result`, a frozen dataclass is one output per field, and `Param(title=...)` titles one. A
  default is checked where it is written, and so is a limit against its type: `x: int = "abc"` and
  `Param(ge=0)` on a `str` are each a `NodeDeclarationError` at import.
- **Fields that depend on configuration** come from `compute_added_inputs=` and `compute_outputs=`,
  plain functions whose parameters name the inputs they read and that return `{name: type hint}`. A
  parameter annotated `type` gets the type on an edge. They run when a graph is loaded, and one that
  raises is the problem `compute_hook_failed` on its node.
- **Versions.** A newer version names the one it replaces, `@node("id", replaces=older,
  upgrade=Upgrade(...))`, where `older` is a `@node` of the same id; a definition is one version,
  `version`, the older ones on `replaces`. `registry.upgrade(record, node_id)` moves one node of a
  record on: an `Upgrade`'s renames move its values, defaults and edges, and its `rewrite` takes and
  returns what is typed into the node, values and defaults as one map (a default stays one; anything
  new is a value). `Deprecation(...)` marks a version going away.
- **What the run supplies.** `FlowStore` and `store_data` are a parameter typed `FromRun[T]`, filled
  from `run.resume(from_run={T: value})`.
- **Retries and failures.** `max_retries`, `retry_delay` and `RetryConfig` are
  `Policy(concurrency=, retries=, retry_delay=, attempt_timeout=, retry_on=)`. A node with
  `concurrency` 1 (the default) and no `attempt_timeout` is called in order on the leg's own thread.
  Only an `ExternalFailure`, or an exception named in `retry_on`, is retried. `NodeValidationError`,
  `NodeExecutionError` and `NodeTimeoutError` are `NodeError(message, code=, ctx=)`, or
  `ExternalFailure` for what the outside world caused.
- **Widgets.** A widget is the control alone, with a `type`, and no type has a default one.
  Conductor ships text, textarea, dropdown, number, range, switch, date, file and list, and a host
  subclasses `Widget` for its own. The limits of a value go on the `Param` (`ge`, `le`, `min_length`,
  `max_length`, `pattern`), where they are enforced on every value typed in or given. `Checkbox`,
  `Multiselect`, `DependentDropdown`, `HumanReview`, `TableSource`, `ConditionBuilder` and
  `ColumnSelect` are gone, and so is the widget called `Output`.
- **Types.** Any type pydantic can carry travels on edges, a host's own class included, with no base
  class. A class may declare `id`, `display_title` and a classmethod `accepts(source)` that lets other
  types land on it.
- **Rows.** The for-each, while, subprocess and signal nodes and regions are gone. A list or a column
  arriving on an input declared for one value runs the node once per row, and a node isn't called for
  a blank unless its input says `| None`; `run.not_called` records the row and input. `Series[X]`
  marks a field as holding many values of `X`, a plain list to the node, every value under the
  node's row. An open input named in its node's `many` takes a list from the caller. Beside a value, `many` makes it a list typed in: the node runs once per item under each row it
  runs on (questions typed into a node that runs per document), and two lists typed into one node
  are `two_typed_in_lists`. Rows of two
  different tables never meet by themselves (`rows_not_related`): join them on a key in a node.
  `max_rows=` on a run is a ceiling checked before any node is called on the rows (`row_ceiling`).
- **Tables.** `Table[TypedDict]` is a whole table, a `polars.DataFrame`. Columns wired into one
  `Table` input are read along one path of the row tree, a parent's value copied down onto its
  children's rows.
- **Graphs.** `Flow` is a `GraphRecord`, plain data: `{"nodes": [...], "edges": [["a.result",
  "b.text"], ...]}`. A `NodeRecord` is `id`, `node` (the node it uses; was `type`), `version` (1 when
  left out), `values` (typed in, fixed), `defaults` (typed in, open to a caller), `many` and
  `display` (the host's own); empty fields are left out when saved. Edges are a set: several into one
  input gather in the order their nodes appear in `nodes`, then the order a node declares its
  outputs. An edge drawn twice, an edge naming a node the graph doesn't have and an input set two
  ways are each a problem. `make_graph(upper, exclaim, values={"exclaim.times": 2})` writes a record
  from Python: a chain with no `edges`, or exactly the `edges` given. YAML is gone with `pyyaml`.
  CEL guards on edges, shared references, compensation and `on_error` are gone.
- **Loading.** `compile(...)` is `registry.load(record)`, which takes a record, JSON text or a
  mapping and gives a `Graph` with its own copy of the definitions it resolved. It never raises for a
  fault in the graph: read `graph.problems` and `graph.is_runnable`, or `graph.describe()` for all of
  it as JSON. A `Problem` is `type`, `loc`, `msg` and `ctx`. An open input of a first node is an input
  of the graph, not a problem; `unbound_on_later_node` is a later node's.
- **Embedded graphs.** `extension_resolver` is gone. A stored graph is a node, `node("id")(record)`,
  wired by the names it offers, and it runs as its nodes drawn in its place.
- **Running.** `execute`, `execute_sync` and `collect` are `graph.run(**inputs)`, which returns a
  `Run`; `graph.with_inputs(...)` checks the inputs and returns a run not yet started, and
  `run.resume()` and `run.stream()` run the next leg. A pause, a failure or a timeout is
  `run.status`, with `run.cause`; `run.results` and `run.outputs` read it. A run is one JSON value:
  `run.to_json()` and `graph.load_run(saved)`. A failed unit stops anything new from starting; the
  calls already running finish and are kept.
- **Checked values.** What a node returns is checked against what it declares: a value that isn't
  its type fails that node as `invalid_output`, naming the output (and a table's `column`) and
  quoting nothing of the value; a value of another class is read strictly by the type, so a model
  may come back as a dict but `"no"` is never a `bool`, `"12"` never an `int` and `12.0` never an
  `int` column. Two outputs of one `exclusive_group` both holding a value are `invalid_output`. A value an edge brings that its
  source's type doesn't settle (a limit on the input, an `int` into a `float`, a host type's own
  check, a table's declared columns) is read by the input's type before the call, and one it refuses
  fails the node it reached as `invalid_input`.
- **Events.** `flow_*` events are `run_start`, `node_failed`, `unit_waiting` and `run_end`. What goes
  wrong is always a `Problem`, and a node's own exception goes to the `conductor` logger. An event
  writes itself with `event.model_dump_json()`. `run_end` keeps its `run` for the host and writes
  only its `status`. `node_progress` arrives at most four times a second per node.
- **A person in the loop.** `HumanInputRequired` and `SignalRequired` are `raise Ask(prompt,
  choices=, default=, widget=)`, and the node's return type says what the answer is and how to ask:
  `bool` a yes or no, a `Literal` one of its values, `str`, `int`, `float` or `date` a value, a
  frozen dataclass a form (a field's default is proposed), `Table[R]` records, and `Series[X]`
  several, so `Series[Literal[...]]` is a pick of several. A node of several outputs asks with
  `defaults=`, `widgets=` and `choices=` by output name; `Choice(id, title)` titles a choice (it is
  imported from `conductor`). Each `Waiting` has `questions`, one per output, in the shape
  `describe()` gives a field, with `choices`, `default` and `widget`. An `Ask` its outputs can't be
  answered with fails the node as `invalid_ask`. In place of `resume`, `run.answer(unit, value)`, or
  `run.answer({unit_id: value, ...})` for several, returns a run to resume; an answer is read strictly
  from its JSON form (a yes or no is `true`, a number is no text); every wrong answer is an
  `invalid_answer` at once, naming the `question`, and nothing is answered. A `Param` widget with no
  JSON form raises `widget_not_json`.
- **Running again.** `run.again(**changed)` (`run.with_inputs(**changed)` to resume or stream it yourself) runs with changed inputs, the rest as before: a row that
  receives what it received before is not called again, so one row added to a CSV of 5,000 sends one
  mail. Identical rows take the earlier ones k-th to k-th, an answer only under the same rows above
  it, a row still waiting goes on waiting, and work from every earlier run is carried. A node with
  other code or another signature is new work. `run.forget(*node_ids)` calls chosen nodes again.
  `Policy(reuse_when_inputs_match=False)` opts a node out.
- **Standard nodes and providers.** `syv-conductor-nodes` (`conductor_nodes`, with
  `get_default_registry`) and `syv-conductor-providers` (`conductor_providers`, with the Mermaid
  drawing) are no longer published: the docs declare the few nodes they need where they use them.
- **HTTP.** `conductor_providers.fastapi` and the `fastapi` extra are gone: `conductor_router`,
  `/execute`, `/execute-stream`, `/nodes`, `/compile`, `/entities/{kind}`, `context_factory`,
  `entity_resolver`, `palette_from_registry` and `sse_frame`. Copy `docs/serving-runs.md` onto your
  own application; it needs `fastapi>=0.135`. `/nodes` is `registry.describe()`, `/load` is
  `registry.load(graph).describe()`, and `context_factory` is a `Depends` on your route, passed as
  `from_run`.
- **Canvas.** `conductor_providers.react` is gone: `graph_to_react`, `react_to_graph` and
  `topological_positions`. A canvas stores the `GraphRecord` as it is, its edges as drawn and a
  node's position in `node.display["position"]`; for layout, use the canvas's own engine.

## [1.12.2]

- The validation model no longer turns `**kwargs` into a required field.

## [1.12.1]

- `**kwargs` / `*args` are no longer introspected as input handles.

## [1.12.0]

- `compute_added_inputs` hook, `CompiledGraph.node_inputs` and `has_dynamic_inputs` in the palette.

## [1.11.0]

- `conductor.resolve_graph_outputs(nodes, edges, definitions)`: compile's output resolution, ahead of compile.

## [1.10.0]

- `finalize_connection_labels(label_hints)`, the public connection-label algorithm.

## [1.9.0]

- `SerializedNode` and `serialize_node_model`; `serialize_node` takes an optional `registry`.

## [1.8.0]

- `project_outputs(results, outputs)` and `OutputRef`.

## [1.7.0]

- Typed serialized schema (`SerializedInput`, `SerializedOutput`), `RegistryView`, and published
  palette TypeScript types.

## [1.6.0]

- Public `serialize_node` / `serialize_input` / `serialize_output` and `WIDGET_SCHEMA_KEYS`.
- `FileUpload.accept` serializes as a list.

## [1.5.2]

- Function-node dispatch filters inputs to the node's signature.

## [1.5.1]

- For-each loop bodies inherit the caller's `contextvars`.

## [1.5.0]

- `HumanReview` widget: per-node approval that pauses after the node runs.

## [1.4.0]

- `cache` on the FastAPI provider's `/execute` and `/execute-stream`.

## [1.3.0]

- Typed `for-each-end` collected outputs, and `compute_for_each_end_outputs`.

## [1.2.0]

- For-each bodies run by dependency level, concurrently.

## [1.1.0]

- Compound-node events stream live; parallel for-each reports progress per item.

## [1.0.0]

- First stable release: semantic versioning, explicit `__all__` on public modules, stress tests.

## [0.1.7]

- Tabular widgets (`TableSource`, `ConditionBuilder`, `Tags`, `ColumnSelect`, `TableInput`) and
  `compute_outputs` ergonomics.

## [0.1.6]

- `compute_outputs` hook and the if-else operator catalog.

## [0.1.5]

- Multi-output collection on `for-each-end`; no limit on loop-node handles.

## [0.1.4]

- Correct providers pin in the `[all]` extra.

## [0.1.3]

- Parallel-zip for-each over several source lists.

## [0.1.2]

- Optional dependency extras: `[yaml]`, `[nodes]`, `[providers]`, `[all]`.

## [0.1.1]

- `conductor_router` forwards an `extension_resolver`.

## [0.1.0]

- First release on PyPI as three packages: the engine, the standard nodes and the framework adapters.

[Unreleased]: https://github.com/syv-ai/conductor/compare/v1.12.2...HEAD
[1.2.0]: https://github.com/syv-ai/conductor/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/syv-ai/conductor/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/syv-ai/conductor/compare/v0.1.7...v1.0.0
[0.1.7]: https://github.com/syv-ai/conductor/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/syv-ai/conductor/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/syv-ai/conductor/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/syv-ai/conductor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syv-ai/conductor/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syv-ai/conductor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syv-ai/conductor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syv-ai/conductor/releases/tag/v0.1.0
