# Changelog

All notable changes to `syv-conductor`, `syv-conductor-nodes` and `syv-conductor-providers`,
released together at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the stored graph, loading and the engine are each
replaced, and no alias keeps a 1.x name alive. Conductor depends on `polars>=2.0` and
`pydantic>=2.11`. The README walks through the design, and `python -m conductor.about` has every
rule.

What changes in how you work, before the name-by-name tables:

- **Four modules.** Import from `conductor`, `conductor.widgets`, `conductor.events` and
  `conductor.testing`. Everything else is `conductor._internal`.
- **A node is a typed function.** `@node("id")` (registers nothing) or `@registry.node("id")`
  returns a frozen, callable `NodeDefinition`. The signature is the interface:
  `Annotated[str, Param(title=, widget=, ge=, ...)]` for what a person sees and the limits a value
  keeps (pydantic metadata, so its limits and choices hold inside your own `BaseModel` or dataclass
  too, and a field's `schema` carries the limits but not its title or description; `Param`s stacked in one `Annotated`, an alias wrapped again, merge, the later
  winning, and a `Param` on a `Series` item is the field's; a table's column
  typed `Annotated[...]` is refused, since its limits would go unread), `-> X` for one output named `result`, a frozen dataclass for several. A
  `bool` never lands on a number, a `Literal` of numbers included. `async def` and
  auto-discovery are gone. A node declared wrong raises `NodeDeclarationError` at import, with a
  `code`; [`docs/problems.md`](docs/problems.md) lists them.
- **Loops are rows.** A list or a column on an input typed for one value runs the node once per
  row, and the node never sees the difference: the value a caller gives decides, with no mark on
  the graph (`graph.inputs[name].per_item` says where a list is rows). `Series[X]` takes many values
  and `Table[R]` a whole `polars.DataFrame`. Two lists meeting at one node are refused
  (`lists_not_collected`) until a node collects one. `graph.calls_for(file=11)` says each node's
  calls before a run, refusing an input given by name and address as `with_inputs` does. A blank skips the node for that row unless its input says `| None`. Each item
  of a list given there is held to the input's limits and choices, as one value is. A choice is
  read as its answer would be: strictly from its JSON form, held to the field's limits, so an enum's
  value or a date's ISO text is a choice.
- **Loading never raises for a fault in the graph.** `registry.load(record)` returns a `Graph`;
  read `graph.problems`, each a `Problem(type, loc, msg, ctx)`. Only a record of the wrong shape raises,
  `InvalidCallError` with the code `invalid_record`. A value typed in or given that its field refuses
  (`invalid_value`, `invalid_input`) carries pydantic's value-free `errors` in `ctx`. `make_graph(*nodes, edges=, values=, defaults=)`
  writes a record from Python, loading each node against the definition it was given, and raises one
  error at the call listing every fault of what it was told: an address, a
  value, or an edge whose types don't meet (`type_mismatch`); two nodes of one id that aren't
  versions of one node are `duplicate_node_id`. A record's `values`, `defaults` and
  `display` are JSON values; `make_graph` writes each value typed in as its JSON, a float that isn't
  finite as `"NaN"` or `"Infinity"`, so a record equals its JSON, and an `upgrade` rewrite is written
  the same way (`invalid_value` where it has no JSON form). A stored graph used as a node offers
  the inputs and outputs it offers loaded alone, the fields its hooks add included, and `@node` on
  a stored graph refuses a `policy` or a hook as `not_for_a_graph`.
- **A run is one value.** `graph.run(**inputs)` returns a `Run` whose `status` says how the leg
  ended. It pauses for a person (`raise Ask(...)`, then `run.answer(...)` and `run.resume()`), writes
  itself as JSON (`run.to_json()`, `graph.load_run(saved)`, which reads a save from its JSON text; a save that
  doesn't fit is `state_unreadable`, `ctx.at` its place in the save; a node with no work done is
  left out of it) and runs again on changed inputs,
  calling only the rows that changed (`run.again(...)`). A leg's `max_rows` is a whole number of
  at least 1 and its `timeout` and `checkpoint_every` finite seconds above 0, a `bool` neither,
  refused at the call (`invalid_max_rows`, `invalid_timeout`, `invalid_checkpoint_every`).
- **Only an outside failure is retried.** `ExternalFailure`, or a class in `Policy(retry_on=)`;
  a `NodeError`, a wrong output or an `Ask` never is, whatever `retry_on` names.
  A node's own exception goes to the `conductor` logger, never to an event.
- **Two companion packages, rebuilt on this API.** `syv-conductor-nodes` holds 29 nodes that only
  compute, a registry per module (`conductor_nodes.text.registry`) and all of them in
  `conductor_nodes.registry`. `math-round` rounds a half away from zero (2.5 to 3), and
  `regex-replace` fails a replacement that refers to a missing group as `invalid_replacement`. `syv-conductor-providers` holds `run_router`, a FastAPI router that
  starts, follows, answers, resumes and cancels runs on several workers over a `RunStore` you
  write ([`docs/serving-runs.md`](docs/serving-runs.md)). An answer sent while a leg runs waits for it to
  end, and one that then finds nothing waiting, or a leg that needs a resume, is kept in `refused`; `again` runs on the nodes of
  the caller who asks; a `caller` dependency's type is a class's own or a function's return
  annotation, which it must have; `resume` takes no body, and a ceiling it raises holds for the run's later legs; a leg whose event
  the store fails to take ends before its next unit, so no node is called once the run reads `failed`; `GET /runs/{id}` says the `cause` of a run that
  didn't complete; a float that isn't finite in a run's outputs reads as `"Infinity"` or `"NaN"`.
  There is no canvas package.

A 1.x graph doesn't load as saved. Convert each stored graph with the table under
"Converting a stored 1.x graph", then save the `GraphRecord`.

### From 1.12.2, name by name

Every public name of 1.12.2, by the module it was imported from. A name not listed under 2.0 is gone.

| 1.12.2 | 2.0 |
|---|---|
| **Nodes** | |
| `Actor` | gone: a node declares no actor; the host decides who runs what |
| `BaseNode` | a plain function under `@node("id")` or `@registry.node("id")` |
| `conductor.metadata.InputMetadata` | `FieldDescription`, an entry of `definition.describe().inputs`; the hint and its `Param(...)` say it |
| `conductor.metadata.OutputMetadata` | `FieldDescription`, an entry of `definition.describe().outputs` |
| `conductor.registry.dynamic_inputs.ComputeInputsContext` | gone: a `compute_added_inputs` hook is a plain function whose parameters name the inputs it reads |
| `conductor.registry.dynamic_inputs.ComputeInputsFn` | `compute_added_inputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.ComputeOutputsContext` | gone: a `compute_outputs` hook's parameters name what it reads; one annotated `type` receives the arriving type |
| `conductor.registry.dynamic_outputs.ComputeOutputsFn` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.IncomingBinding` | gone: a hook parameter annotated `type` receives the type arriving on that input |
| `conductor.registry.dynamic_outputs.strip_sub_output_prefix` | gone: an output is named by its field |
| `conductor.registry.NodeDefinition` | `NodeDefinition`, frozen and callable, returned by `@node` |
| `conductor.types.Base64Str` | gone: any type pydantic carries travels; `bytes` is one |
| `conductor.types.Date` | `datetime.date` |
| `conductor.types.MultiNamedFile` | gone: `Series[YourFile]` |
| `conductor.types.NamedFile` | gone: declare your own pydantic model or dataclass |
| `conductor.types.NodeResult` | gone: a node returns its value; a run's are `run.results` and `run.outputs` |
| `conductor.types.OUTPUT_PREFIX` | gone: outputs are named by their fields |
| `conductor.types.RESULT_KEY` | an output of `-> X` is named `result`; a frozen dataclass names several |
| `FlowStore` | a parameter typed `FromRun[T]`, filled from `from_run={T: value}` |
| `NodeCategory` | a plain string, `@node("id", category="text")` |
| `NodeRegistry` | `NodeRegistry(category=, tags=, extra=, types=)`, empty until nodes are added |
| `ResultFormat` | gone: a node returns its value, typed by its return hint |
| `RetryConfig` | `Policy(retries=, retry_delay=, attempt_timeout=, retry_on=)` on `@node(policy=)` |
| `SKIPPED` | `SKIPPED`, returned on a branch not taken |
| **Widgets** | |
| `conductor.widgets.Checkbox` | `Switch` |
| `conductor.widgets.CodeEditor` | gone: a host subclasses `Widget` |
| `conductor.widgets.ColumnSelect` | gone: an edge names a column, `node.output.column` |
| `conductor.widgets.ConditionBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.ConnectionList` | gone: a `Series[X]` input takes every edge into it |
| `conductor.widgets.DatePicker` | `DatePicker` |
| `conductor.widgets.DependentDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.Dropdown` | `Dropdown`, which takes no choices: they live on the field, `Param(choices=[Choice(id, title)])` or a `Literal` |
| `conductor.widgets.EntityDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.FileUpload` | `FileUpload` |
| `conductor.widgets.HumanReview` | `raise Ask(...)` in the node; its return type says what is asked |
| `conductor.widgets.IfElseBuilder` | gone with the if-else node: outputs of one `Param(exclusive_group=)` are the branches |
| `conductor.widgets.List` | `ListInput` |
| `conductor.widgets.Multiselect` | an input typed `Series[Literal[...]]`, or `Dropdown` on a `Series` |
| `conductor.widgets.Number` | `NumberInput` |
| `conductor.widgets.Output` | gone: an output has no widget |
| `conductor.widgets.Range` | `Slider`; its limits go on `Param(ge=, le=)` |
| `conductor.widgets.SchemaBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.Switch` | `Switch` |
| `conductor.widgets.TableInput` | an input typed `Table[R]`, the table typed in as records |
| `conductor.widgets.TableSource` | an input typed `Table[R]` |
| `conductor.widgets.Tags` | gone: an input typed `Series[str]` |
| `conductor.widgets.TemplateTextarea` | gone: a host subclasses `Widget` |
| `conductor.widgets.Text` | `TextInput` |
| `conductor.widgets.Textarea` | `Textarea` |
| `conductor.widgets.WIDGET_SCHEMA_KEYS` | gone: a widget is a pydantic model; its fields are its keys |
| `conductor.widgets.Widget` | `Widget`; a host subclasses it for a control of its own |
| `widgets` | `conductor.widgets`, imported as a module |
| `WidgetType` | a widget's `type` field |
| **Graphs and loading** | |
| `compile` | `registry.load(record)`, which returns a `Graph` and never raises for a fault in the graph |
| `CompiledGraph` | `Graph` |
| `conductor.expr`: `Expression`, `ExpressionError`, `ExpressionParseError`, `ExpressionRuntimeError`, `ExpressionTypeError`, `evaluate`, `parse` | gone with CEL guards on edges |
| `conductor.flow_format.dump_flow` | `record.model_dump_json()` |
| `conductor.flow_format.flow_to_dict` | `record.model_dump(mode="json")` |
| `conductor.flow_format.flow_to_yaml` | gone: YAML isn't written |
| `conductor.flow_format.load_flow_from_path` | gone: read the file and `registry.load` its text |
| `conductor.flow_format.load_flow` | `registry.load(json_text)` or `GraphRecord.model_validate_json` |
| `conductor.flow_format.yaml_to_flow` | gone: YAML isn't read |
| `conductor.graph.compiler.DecisionGuard` | gone with CEL guards: outputs of one `exclusive_group` are the branches |
| `conductor.graph.compiler.ExtensionResolver` | gone: a stored graph is a node, `node("id")(record)`, added to a registry |
| `conductor.graph.dynamic_inputs.resolve_node_inputs` | `registry.load(record).describe()`, a `GraphDescription` with each node's fields as its hooks made them |
| `DefinitionSource` | gone with `RegistryView` |
| `expr` | gone with CEL guards on edges |
| `Flow` | `GraphRecord`; see "Converting a stored 1.x graph" |
| `FlowDependency` | gone: a graph embeds another as a node, `node("id")(record)` |
| `FlowTrigger` | gone: the host decides when a graph runs |
| `GraphEdge` | a pair `("a.result", "b.text")` in `GraphRecord.edges` |
| `GraphNode` | `NodeRecord`; see "Converting a stored 1.x graph" |
| `RegistryView` | `registry.filter(predicate)` and `add_subregistry`; a request's own nodes go in a registry of their own |
| `resolve_graph_inputs` | `registry.load(record).inputs`, each a `GraphInput` |
| `resolve_graph_outputs` | `registry.load(record).outputs` |
| **Running** | |
| `compute_for_each_end_outputs` | gone: a `Series[X]` input gathers the rows |
| `conductor.compound.CompoundNodeType` | gone with compound nodes |
| `conductor.compound.NodeExecutor` | gone with compound nodes |
| `conductor.compound.Region` | gone with regions |
| `conductor.execution.engine.collect` | `graph.run(...)`, then `run.results` |
| `conductor.execution.events.CompensationCompleteEvent` | gone with compensation |
| `conductor.execution.events.CompensationFailedEvent` | gone with compensation |
| `conductor.execution.events.CompensationStartEvent` | gone with compensation |
| `conductor.execution.events.EventSink` | `async for event in run.stream()` |
| `conductor.execution.events.ExecutionEvent` | `Event`, a union told apart by `type` |
| `conductor.execution.events.FlowCancelledEvent` | `RunEnd` with `status` `cancelled` |
| `conductor.execution.events.FlowCompleteEvent` | `RunEnd` with `status` `complete` |
| `conductor.execution.events.FlowErrorEvent` | `RunEnd` with `status` `failed` |
| `conductor.execution.events.FlowPausedEvent` | `UnitWaiting` per waiting row, then `RunEnd` with `status` `waiting` |
| `conductor.execution.events.FlowTimeoutEvent` | `RunEnd` with `status` `timed_out` |
| `conductor.execution.events.NodeCompleteEvent` | `NodeComplete` |
| `conductor.execution.events.NodeErrorEvent` | `NodeFailed`, its `cause` a `Problem` |
| `conductor.execution.events.NodeProgressEvent` | `NodeProgress` |
| `conductor.execution.events.NodeRetryEvent` | `NodeRetry` |
| `conductor.execution.events.NodeSkippedEvent` | `NodeSkipped` |
| `conductor.execution.events.NodeStartEvent` | `NodeStart` |
| `conductor.execution.events.RuntimeWarningEvent` | gone: a type fault is a load problem, `graph.problems` |
| `conductor.execution.events.SignalWaitingEvent` | `UnitWaiting` |
| `conductor.execution.resolver.finalize_connection_labels` | gone with connection labels: an input is named by its field |
| `conductor.execution.resolver.InputResolver` | gone: the graph reads its inputs, `graph.with_inputs(...)` |
| `conductor.execution.resolver.MULTI_VALUE_SEPARATOR` | gone: several edges into a `Series` input arrive as a list, never joined |
| `conductor.execution.results.extract_output` | `run.results[node_id][output]` |
| `conductor.execution.results.filter_all_skipped` | gone: a skipped row is a blank in `run.results` |
| `conductor.execution.results.filter_skipped` | gone: a skipped row is a blank in `run.results` |
| `conductor.execution.results.normalize_result` | gone: what a node returns is checked against its type |
| `conductor.execution.results.OutputRef` | an address, `node.output` |
| `conductor.execution.results.project_outputs` | `run.outputs`, by address |
| `conductor.execution.state.FlowRunState` | `Run`: `status`, `results`, `waiting`, `cause` |
| `execute_sync` | `graph.run(**inputs)` |
| `execute` | `graph.run(**inputs)` returns a `Run`; `run.stream()` yields its events |
| `FlowCheckpoint` | a `Run`: `run.to_json()` and `graph.load_run(saved)`; mid-leg, `run.stream(checkpoint_every=)` yields `RunCheckpoint` |
| `FOR_EACH` | gone with the for-each node |
| `ForEachNode` | gone: a list or column on a single-value input runs the node once per row |
| `resume_sync` | `run.resume()` |
| `resume` | `run.answer(...)`, then `run.resume()` |
| `SUBPROCESS` | gone with the subprocess node |
| `SubprocessNode` | a stored graph is a node, `node("id")(record)` |
| `SubprocessRegistry` | gone: an embedded graph's node goes in a registry like any other |
| `WHILE` | gone with the while node |
| `WhileNode` | gone: a graph has no cycles |
| **Errors** | |
| `CompilationError` | gone: loading never raises; read `graph.problems` |
| `conductor.errors.CycleDetectionError` | the problem `cycle` in `graph.problems` |
| `conductor.errors.FlowExecutionException` | gone, as `FlowExecutionError` |
| `conductor.errors.FlowPausedException` | gone, as `FlowPausedError` |
| `conductor.errors.InputResolutionError` | `InvalidCallError` at `graph.run`, listing `missing_input`, `not_offered`, `invalid_input` |
| `conductor.errors.NodeExecutionException` | `NodeError` |
| `conductor.errors.NodeValidationException` | `NodeError` |
| `conductor.errors.TypeCheckError` | the problem `type_mismatch` in `graph.problems` |
| `ConductorError` | gone: catch `NodeDeclarationError`, `InvalidCallError` or `NodeError` |
| `errors` | gone as a module: the classes import from `conductor` |
| `FlowExecutionError` | gone: a failed leg returns its `Run`, `status` `failed` and `cause` a `Problem` |
| `FlowPausedError` | gone: a paused leg returns its `Run`, `status` `waiting` |
| `HumanInputRequired` | `raise Ask(prompt, choices=, default=, widget=)`; each has the answer's shape, so on several outputs each goes by output name |
| `LoopRunawayError` | `graph.run(max_rows=)`, the problem `row_ceiling` |
| `NodeConnectionError` | `ExternalFailure`, the one failure that is retried |
| `NodeError` | `NodeError(message, code=, ctx=)`, never retried |
| `NodeExecutionError` | `NodeError(message, code=, ctx=)` |
| `NodeTimeoutError` | `Policy(attempt_timeout=)`; the unit fails with the code `timeout` |
| `NodeValidationError` | `NodeError(message, code=, ctx=)` |
| `SignalRequired` | `raise Ask(...)` |
| `SubprocessFailedError` | the embedding node's `node_failed`, naming the embedding |
| **The palette** | |
| `conductor.registry.schema.serialize_input_model` | `FieldDescription` |
| `conductor.registry.schema.serialize_input` | an entry of `definition.describe().inputs`, a `FieldDescription` |
| `conductor.registry.schema.serialize_node_model` | `definition.describe()`, a `NodeDescription` |
| `conductor.registry.schema.serialize_node` | `definition.describe()`, a `NodeDescription`; `.model_dump(mode="json")` for a dict |
| `conductor.registry.schema.serialize_output_model` | `FieldDescription` |
| `conductor.registry.schema.serialize_output` | an entry of `definition.describe().outputs`, a `FieldDescription` |
| `conductor.registry.schema.serialize_registry` | `registry.describe()`, a `RegistryDescription` |
| `conductor.registry.serialized.SerializedInput` | `FieldDescription` |
| `conductor.registry.serialized.SerializedNode` | `NodeDescription` |
| `conductor.registry.serialized.SerializedOutput` | `FieldDescription` |
| **Packages** | |
| `conductor.about.get_content` | `get_content`, unchanged |
| `conductor.about.get_section` | `get_section`, unchanged |
| `conductor.about.list_sections` | `list_sections`, unchanged |
| `conductor_nodes.control_operators`: `OPERATORS`, `OPERATORS_BY_ID`, `OPERATORS_BY_LABEL`, `OPERATORS_BY_TYPE`, `OperatorEvaluationError`, `OperatorSpec`, `UNARY_OPERATORS` | gone: a condition is a node that returns a `bool` (`text-contains`, `regex-match`), wired into `decision.when` |
| `conductor_nodes`: `get_default_registry`, `register_all` | `conductor_nodes.registry`, added with `registry.add_subregistry(...)` |
| `conductor_nodes`: `CATEGORIES` | gone: each node has its `category`; `conductor_nodes.registry.filter(lambda d: d.category == "math")` |
| `conductor_nodes`: `decision`, `json_ops`, `logic`, `math`, `regex_ops`, `text` | a module per category, each with a `registry` of its own nodes: `control` (`decision`, `logic-if-empty`, `logic-if-equals`), `json_ops`, `logic` (`logic-not`), `math`, `regex_ops`, `text`; node ids unchanged |
| `conductor_nodes`: `loop`, `while_loop`, `signal`, `subprocess` | gone: a node typed for one value runs once per row by itself, and a pause is `raise Ask(...)` |
| `conductor_providers.fastapi.conductor_router` | `conductor_providers.fastapi.run_router(store=, registry=, caller=)`: runs by id, kept in a `RunStore`, on several workers; `/nodes` is `registry.describe()`, `/compile` is `POST /load` (`graph.describe()`), `/execute-stream` is `POST /runs` and `GET /runs/{id}/events` |
| `conductor_providers.fastapi`: `ExecuteRequest`, `NodeInput`, `EdgeInput` | `StartRun`, a `GraphRecord` and its inputs |
| `conductor_providers.fastapi`: `CompileResult`, `CompileWarning` | `GraphDescription`, whose problems name each node |
| `conductor_providers.fastapi.sse_frame` | gone: each frame is an event's own JSON |
| `conductor_providers.fastapi` (`context_factory=`, `entity_resolver=`) | `from_run=`, built from the caller who started the run; an entity list is a route of your own |
| `conductor_providers.react`: `graph_to_react`, `palette_from_registry`, `react_to_graph`, `topological_positions` | gone: a canvas stores the `GraphRecord` as it is, a node's position in `display` |
| `conductor_providers`: `PROVIDERS`, `react` | gone |

### `@registry.node(...)` arguments

| 1.12.2 | 2.0 |
|---|---|
| `base_id` | the id, first and required: `@node("upper")` |
| `version` | `version=`, one past the version replaced, or 1 |
| `name` | `title=`, optional: the function's name in everyday words |
| `description` | `description=`, optional: the docstring's first paragraph |
| `tags` | `tags=`; a registry's tags add to them |
| `category` | `category=`, a string |
| `max_retries` | `Policy(retries=)` |
| `retry_delay` | `Policy(retry_delay=)` |
| `width` | gone: how a node is drawn is the host's, `NodeRecord.display` |
| `docs` | `docs=` |
| `actor` | gone with `Actor` |
| `timeout` | `Policy(attempt_timeout=)` |
| `idempotency_key` | gone: `again` reuses a unit whose inputs match, `Policy(reuse_when_inputs_match=)` |
| `uses` | gone: a parameter typed `FromRun[T]` says what the node needs from the run |
| `is_decision` | `Param(exclusive_group=)` on the outputs that are branches |
| `is_signal` | gone: `raise Ask(...)` |
| `dynamic_handles` | gone: `compute_added_inputs=` and `compute_outputs=` |
| `compute_outputs` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `compute_inputs` | `compute_added_inputs=`, the inputs it adds, taken as `**kwargs` |

### Converting a stored 1.x graph

A 1.x `Flow` becomes a `GraphRecord`, `{"nodes": [...], "edges": [[from, to], ...]}`. Each `GraphNode` becomes a `NodeRecord`, `{id, node, version, values, defaults, display}`, empty fields left out. Each `GraphEdge` becomes a pair of addresses, `["source.source_handle", "target.target_handle"]`. Load the result and read `graph.problems` to see what didn't carry over. A 1.x for-each over a list is gone: give the list to the input that took one item, and the node runs once per item; a node after it that collected the items takes them through a `Series` input.

| 1.x field | 2.0 |
|---|---|
| `GraphNode.id` | `id`, without `.`, `/` or `@` |
| `GraphNode.type` | `node` (the id of the node it uses) and `version`, 1 when left out |
| `GraphNode.data` | `values` for what is fixed, `defaults` for what a caller may change; an input wired by an edge has neither |
| `GraphNode.produces` | gone: the node's outputs are its return type's |
| `GraphNode.consumes` | gone: the node's inputs are its signature's |
| `GraphNode.compensation` | gone with compensation |
| `GraphNode.on_error` | gone: a failed unit ends the leg `failed` |
| `GraphNode.node_label` | `display`, the host's own |
| `GraphNode.output_labels` | `display`, or a `Param(title=)` on the output |
| `GraphEdge.id` | gone: an edge is its two addresses |
| `GraphEdge.source` | the first address's node: `source.output` |
| `GraphEdge.target` | the second address's node |
| `GraphEdge.source_handle` | the first address's field, `source.output`, or `source.output.column` for a table's column |
| `GraphEdge.target_handle` | the second address's field, `target.input` |
| `GraphEdge.when` | gone: a branch is an output of one `exclusive_group`; wire from that output |
| `GraphEdge.priority` | gone: edges are kept sorted; several into one input are read in row order |
| `Flow.nodes` | `nodes` |
| `Flow.edges` | `edges`, each `["a.result", "b.text"]` |
| `Flow.id` | not stored in the record: the host keys its rows |
| `Flow.version` | not stored: each node has its own `version` |
| `Flow.name` | not stored: the host's own, or the embedding node's `title` |
| `Flow.description` | not stored: the host's own, or the embedding node's `description` |
| `Flow.dependencies` | gone: an embedded graph is a node added to the registry |
| `Flow.triggers` | gone: the host decides when a graph runs |
| `Flow.on_error_default` | gone: a failed unit ends the leg `failed` |

### Events

Events are frozen pydantic models in `conductor.events`, told apart by `type`. Two are new: `run_start`, always first, and `run_checkpoint`, the run mid-leg for a host to store, sent when `run.stream(checkpoint_every=)` asks. `run_end` is always last, its `status` one of `complete`, `waiting`, `failed`, `stopped`, `cancelled`, `timed_out`, and its `cause` the run's `Problem` for one that failed, stopped, was cancelled or timed out (`null` otherwise).

| 1.12.2 `type` | 2.0 |
|---|---|
| `node_start` | `node_start` |
| `node_complete` | `node_complete`: `rows`, `not_called`, `reused_units`, counts only; the values are `run.results` |
| `node_skipped` | `node_skipped` |
| `node_error` | `node_failed`, `cause` a `Problem` |
| `node_progress` | `node_progress`: `done`, `total` |
| `runtime_warning` | gone: a type fault is a load problem |
| `flow_complete` | `run_end`, `status` `complete` |
| `flow_error` | `run_end`, `status` `failed` |
| `flow_cancelled` | `run_end`, `status` `cancelled` |
| `flow_timeout` | `run_end`, `status` `timed_out` |
| `flow_paused` | `unit_waiting` per row, then `run_end`, `status` `waiting` |
| `node_retry` | `node_retry`: `row`, `attempt`, `retries`, `delay`, `cause` |
| `compensation_start` | gone with compensation |
| `compensation_complete` | gone with compensation |
| `compensation_failed` | gone with compensation |
| `signal_waiting` | `unit_waiting` |

## [1.12.2]

### Fixed

- The validation model no longer turns `**kwargs` into a required field, and
  a callable declaring it now accepts extra keys. `1.12.1` stopped the
  registry introspecting `**kwargs` as an input, but `create_validation_model`
  builds from the signature independently and still declared a required field
  named `kwargs` — so a `compute_inputs` node failed with
  `Invalid inputs — 'kwargs': Field required` before it ever ran. Worse, the
  model was built with pydantic's default `extra="ignore"` while the engine
  assigns `validated.model_dump()` back over the inputs, so once that field
  was satisfied every hook-declared handle was silently dropped and the node
  ran with nothing wired. `*args` is skipped for the same reason.

## [1.12.1]

### Fixed

- `**kwargs` / `*args` are no longer introspected as input handles. A node
  whose handles come from `compute_inputs` must declare `**kwargs` for the
  resolver's values to reach it; registering that as an input named
  `kwargs` gave it a Text widget that rendered as a stray field on the
  node, put a phantom handle in the palette payload, and tripped host-side
  checks that every declared handle be documented. Delivery is unaffected —
  `_filter_to_signature` reads the signature directly, not this metadata.

## [1.12.0]

### Added

- `compute_inputs` hook — a node can derive typed input handles from its
  instance `data`, mirroring `compute_outputs`. Accepted by both
  `@registry.node(...)` and `@category.node(...)`. New public types
  `ComputeInputsContext` / `ComputeInputsFn` in
  `conductor.registry.dynamic_inputs`, and `resolve_graph_inputs` exported
  from `conductor` as the ahead-of-compile counterpart to
  `resolve_graph_outputs`.
- `CompiledGraph.node_inputs` — the resolved input roster per node id,
  populated for every node.
- `has_dynamic_inputs` in the serialized palette payload, matching
  `has_dynamic_outputs`.

### Changed

- Edge type-checking, consume validation and validation-error labelling
  consult the resolved input roster in preference to the static schema.
  Nodes without a hook are unaffected. Note this makes an edge into a
  hook-declared handle *type-checked* where an unknown target handle was
  previously skipped silently.
- `InputResolver` accepts the resolved rosters and keys its param-info cache
  per node instance when a node's roster came from a hook; type-keyed for
  everything else.

## [1.11.0]

### Added

- `conductor.resolve_graph_outputs(nodes, edges, definitions)` — public
  graph-level output resolution: the exact topological `compute_outputs`
  pass `compile()` runs (step 8b), exposed ahead of compile so a host can
  obtain the authoritative per-node handle set for schema derivation or
  label seeding. Takes a required `definitions` mapping (node type →
  `NodeDefinition | None`) instead of a registry, so hosts can inject
  definitions for extension types (e.g. embedded sub-flows). Applies the
  same structural gates as compile (unknown type, dangling edge, cycle,
  hook-result validation), all raising `CompilationError`. Internally,
  `compile()`'s own dynamic-output pass (step 8b) now routes through the
  same shared engine, so the two entry points cannot diverge — no
  behavior change to `compile()`.

## [1.10.0]

### Added

- `conductor.execution.resolver.finalize_connection_labels(label_hints)` — the
  public form of the ConnectionList label algorithm (collision-aware
  finalization, then `_2`/`_3` dedup). A host that resolves source labels ahead
  of execution — e.g. rewriting stored templates to their live source labels —
  can now reproduce the aggregator's exact input keys without mirroring the
  resolver's private internals. Purely additive: `InputResolver` and its
  labelling behaviour are unchanged.

## [1.9.0]

### Added

- `conductor.registry.serialized.SerializedNode` and
  `conductor.registry.schema.serialize_node_model(nd, registry=None)` — the
  typed counterpart to `serialize_node`, mirroring the existing
  `serialize_input_model` / `serialize_output_model`. A host that serializes a
  node's catalog entry can validate it into `SerializedNode` instead of reading
  a loose `dict[str, Any]`.
- `serialize_node` (and `serialize_node_model`) now accept an optional
  `registry`. Omit it to serialize an ad-hoc definition that isn't registered:
  the node is then reported as the latest, non-deprecated version. Callers that
  pass a registry are unaffected.

### Changed

- `InputMetadata` derives `expects_list` from a `list[...]` `type_str` in
  `__post_init__` when it isn't set explicitly, so hand-built metadata matches
  what the registration decorator derives (a `list[...]` fan-in no longer
  silently string-joins). Metadata that already sets `expects_list` — including
  everything the registry builds — is unchanged.

## [1.8.0]

### Added

- `conductor.execution.results.project_outputs(results, outputs)` and the
  `OutputRef(name, node_id, handle)` value type — project a run's
  `{node_id: NodeResult}` into a flat `{name: value}` map through named
  `(node_id, handle)` refs. The graph-level counterpart to `extract_output`:
  handle-membership decides presence (a produced `None` survives; an absent
  node/handle or a `SKIPPED` value is omitted). `handle=None` selects the
  node's sole result (`RESULT_KEY`, with the `output_1` back-compat fallback).

## [1.7.0]

### Added

- **Typed serialized schema** (`conductor.registry.serialized`): `SerializedInput`
  and `SerializedOutput` — pydantic models the serializers are guaranteed to
  conform to — plus `serialize_input_model` / `serialize_output_model` typed
  accessors on `conductor.registry.schema`. Hosts that want types validate the
  wire payload into these models instead of reading loose dicts; extras are
  allowed so a host's custom `widget_config` keys pass through rather than being
  rejected or dropped.
- **`RegistryView`** (`conductor.registry.view`, re-exported as
  `conductor.RegistryView` with the `DefinitionSource` protocol): a read-only
  overlay that serves host-supplied `NodeDefinition`s for node types not
  statically registered. Point lookups (`get` / `contains`) resolve base-registry
  types first, then each source in order; everything else delegates to the base
  registry. A per-use overlay, not a mutable global hook — callers with different
  visibility never see each other's dynamic types. Generalizes the registry
  wrapper AKA-style hosts previously hand-rolled for extension node types.
- **Published palette TypeScript types**
  (`conductor_providers/react/palette.d.ts`): `PaletteNode`, `PaletteInput`,
  `PaletteOutput` — the TS counterpart of the serialized schema, shipped in the
  wheel. Frontends consuming the palette import these instead of hand-maintaining
  a mirror; an in-repo test pins them to the Python `SerializedInput` /
  `SerializedOutput` fields and the `serialize_node` key set.

### Changed

- `conductor.widgets.WIDGET_SCHEMA_KEYS` is now pinned equal by test to the
  widget-config fields of `SerializedInput`, so the constant, the typed model,
  and the widget `to_schema()` methods cannot drift from one another. The
  constant stays exported and unchanged for hosts that read it.

## [1.6.0]

### Added

- **Public typed registry serialization** (`feat(registry)`): `serialize_node`,
  `serialize_input`, and `serialize_output` are now public — hosts that project
  the serialized registry into their own typed models can call the exact
  functions the wire uses instead of re-implementing the flattening.
- **`conductor.widgets.WIDGET_SCHEMA_KEYS`** — the pinned vocabulary of
  `widget_config` keys stdlib widgets can emit, for hosts to validate their
  typed port models against (guarded by an in-repo bidirectional parity test).

### Changed

- **`FileUpload.accept` serializes as `list[str]`** (`feat(widgets)`): widget
  authors may pass a single string or a list; the wire shape is now always a
  list. Frontends that already handled both shapes are unaffected.

## [1.5.2]

### Fixed

- **Function-node dispatch filters resolved inputs to the node signature**
  (`fix(engine)`): stray `data` keys that aren't node parameters no longer
  crash a normal run with `TypeError: got an unexpected keyword argument`.
  The dispatch path now filters like the compensation path already did;
  functions declaring `**kwargs` are unaffected.

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

[Unreleased]: https://github.com/syv-ai/conductor/compare/v1.12.2...HEAD
[1.12.2]: https://github.com/syv-ai/conductor/compare/v1.12.1...v1.12.2
[1.12.1]: https://github.com/syv-ai/conductor/compare/v1.12.0...v1.12.1
[1.12.0]: https://github.com/syv-ai/conductor/compare/v1.11.0...v1.12.0
[1.11.0]: https://github.com/syv-ai/conductor/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/syv-ai/conductor/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/syv-ai/conductor/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/syv-ai/conductor/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/syv-ai/conductor/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/syv-ai/conductor/compare/v1.5.2...v1.6.0
[1.5.2]: https://github.com/syv-ai/conductor/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/syv-ai/conductor/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/syv-ai/conductor/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/syv-ai/conductor/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/syv-ai/conductor/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/syv-ai/conductor/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/syv-ai/conductor/compare/v1.0.1...v1.1.0
[1.0.0]: https://github.com/syv-ai/conductor/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/syv-ai/conductor/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/syv-ai/conductor/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/syv-ai/conductor/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/syv-ai/conductor/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syv-ai/conductor/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syv-ai/conductor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syv-ai/conductor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syv-ai/conductor/releases/tag/v0.1.0
