# Changelog

All notable changes to `syv-conductor`, `syv-conductor-nodes` and `syv-conductor-providers`,
released together at one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); from `1.0.0` the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Ships as 2.0.0, a new major. The node contract, the graph record, compiling and the engine are each
replaced, and no alias keeps a 1.x name alive. Conductor depends on `polars>=2.0`,
`pydantic>=2.14` (the first to export `MISSING`, which a record without a value is written with) and `typing-extensions>=4.16`. The README walks through the design, and `python -m conductor.about` has every
rule.

What changes in how you work, before the name-by-name tables:

- **A graph record is its nodes by id, and declares what the graph takes and returns.** `{"nodes": {"shout":
  {"node": "shout", "inputs": {...}}}, "outputs": {"result": {"node": "join", "output": "result"}}}`. Each input a
  node lists gets exactly one of a `constant`, a `graph_input` (with an optional `default`) or the `sources` of the
  edges into it, each `{"node", "output", "field"?}`. A graph input sits on any node's input, and several inputs
  may name one when they read the same type from their hints (type, limits, choices and validators) and are shown
  alike (title, description, widget), so `int` and `Annotated[int, Param()]` agree, and so do a `Ge(0)` and a `Param(ge=0)`, else
  `graph_input_used_two_ways`; a graph output is named in `outputs`. A subgraph's inputs and
  outputs are those names, so an edge reads `loud.result`, never a node inside. The record keeps no order of its
  nodes: where order matters, nodes go by id. A key written twice in the JSON (two nodes with one id), a name with
  `.`, `/`, `@` or `~`, a source listed twice, an empty list of sources, an input given no kind or two, a `default`
  without a `graph_input` and a field on a graph output are refused when the record is read (`invalid_record`, with
  the fault in `ctx.error`). A graph input's default is the record's: an entry with no `default` is required.
  `graph.inputs`, `run.inputs`, `graph.outputs` and `run.outputs` are keyed by the declared names. `graph.inputs` and
  `graph.outputs` are read-only mappings: writing into `graph.outputs` changed what `describe()`, a subgraph node
  that uses the graph and the registry's palette showed, and writing into one is now a `TypeError`. `make_graph`
  declares the graph's inputs and outputs itself: each required input that gets nothing, on any node, and each key
  of `defaults` is a graph input of the input's name, and each output of a node no edge leaves a graph output of the
  output's name; `inputs=` and `outputs=` name some of them otherwise, on top of that, and two of one name are
  refused (`graph_inputs_share_a_name`, `graph_outputs_share_a_name`). A node given as `(id, definition)` gets that
  id. `Graph.compile` takes a `GraphRecord`, its JSON text or that JSON as data, and `read_json(text)` reads JSON
  text with every key seen. New public names: `InputEntry`, `Source`, `read_json`. New codes: `input_has_no_value`,
  `graph_input_used_two_ways`, `graph_output_not_found`, `graph_inputs_share_a_name`, `graph_outputs_share_a_name`,
  `invalid_place`.
- **A node that asks says so.** `@node(..., asks=True)` on a node whose function raises `Ask`. Without it,
  `@node` refuses a function whose own body has a `raise Ask(...)`, `Ask` named as the module imports it, with a
  `NodeDeclarationError` (`ask_not_declared_in_source`) [provisional], and any other `Ask`, such as one raised by
  a function the node calls, fails the call as `ask_not_declared`. Both say to add `asks=True`. So
  `describe().asks` tells a host which graphs can pause. A subgraph's palette entry in `registry.describe()` has `asks` too, true when a node inside it asks, at
  any depth. The answer is the node's outputs, and a node that needs it reads it from an edge: "approve, then
  pay" is an asking `approval(invoice) -> bool` wired into `pay(invoice, ok: bool)`, and a host that offers
  the two as one palette entry makes them a `subgraph`. New codes: `ask_not_declared`, `ask_not_declared_in_source`.
- **A type means what it says, wherever it is read.** `Series[T]` and `Table[R]` are ordinary generic
  types: a table stands in a union, a `Series` and a record, and `Table[R]` read anywhere has exactly
  `R`'s columns. A node may have type variables (`def split[T](value: Table[T] | list[T]) -> Series[T]`),
  bound from its edges when a graph is compiled. Generic arguments are compared (`list[int]` lands on
  `list[float]`, never on `list[str]`) and shown in a type's id. `Annotated` metadata that is no `Param`
  is pydantic's, except pydantic's `Field(...)`, which is refused (`pydantic_field_in_hint`): `Param` is the one way to
  write a title, a description and limits on an input or an output. A function in a hint (`AfterValidator(strip)`) is
  written into a node's code fingerprint by its module and name, so a run saved in one process is reused in another. A host type is declared with `@host_type("money", title="Money", accepts=[...])` above the class, checked when the class statement ends;
  `accepts` widens what lands on it, and a subclass of a builtin needs no schema of its own. A table's one JSON form is
  `{"columns": [{"name", "type", "values"}]}`. `syv-conductor-nodes` gains `series-stack`.

- **A `Run` can't be changed after it is made.** `with_answers`, `with_decline`, `resume`, `again`, `with_inputs`, `with_forgotten` and `with_max_items` each return a new run, as before; the run they are called on is left as it was, so `run.with_answers(w, True); run.resume()` leaves `run` waiting; keep what they return, `run = run.with_answers(w, True).resume()`. A run is now a frozen dataclass, so setting an attribute (`run.max_items = 5`) raises `dataclasses.FrozenInstanceError` (it was taken unchecked), and `run.inputs` is a read-only mapping (changing one run's inputs changed every run made from it).
- **`run.answer`, `run.decline` and `run.forget` are `run.with_answers`, `run.with_decline` and `run.with_forgotten`** [provisional]. Each returns a new run and calls no node, as `run.with_inputs` and `run.with_max_items` do, so every method of that kind is named `with_...`, and the name says a new run comes back: `run.answer(w, True)` without `run = ...` read as if it changed `run`. The arguments are the same. `resume` and `again` keep their names, since they call nodes. The worker's `answer` and `decline` and the routes `POST /runs/{id}/answers` and `/declines` keep theirs: they act on the stored run rather than return one.
- **A field's JSON schema is `json_schema` in Python and in JSON** [provisional]. `FieldDescription` had the attribute `json_schema` and wrote it as `"schema"`, an alias kept because a pydantic field named `schema` shadows `BaseModel.schema`. It is now `"json_schema"` in the JSON too, in `describe()`, a waiting call's `questions`, the events and the runs router, and a description is read back by that key only.
- **A field's description is one flat record.** It has `is_series`, `type`, `json_schema` (always set),
  `type_variables` and `loops`; there is no `shape` and no `columns`, a table's columns are in its type and its
  schema. An input of a node in `graph.describe()` and a graph input have the field as `field`. A node's description is
  one `NodeDescription` for a function and a subgraph alike, with no `kind`: a subgraph's `takes_added_inputs` and
  `computes_outputs` are false, and its fields are listed by `registry.describe()` only. A palette carries no policy:
  `PolicyDescription` is gone, and a host reads a function's `Policy` from its definition. `Policy` is a frozen
  pydantic dataclass, never written as JSON, and a policy out of bounds raises pydantic's validation error.
- **A field's `title` is what its `Param(title=)` says, and `null` without one.** Nothing is guessed from the
  field's name: `case_sensitive` with no title has `"title": null` in `describe()`, `graph.describe()`, a graph
  input and a waiting call's question (it was "Case sensitive", and a lone output's "Result"). A host that shows a
  label for an untitled field writes its own from `name`.
- **A widget with no JSON form is a declaration problem.** A widget given a value its `Any` field cannot write
  as JSON raises `NodeDeclarationError` where the node is declared, with the code `widget_without_json_form`,
  `ctx.widget` the widget's class and the field's name in front of the message (it raised pydantic's
  `PydanticSerializationError`, which named no field).
- **An input added by `compute_added_inputs` is held to what `@node` holds a parameter to.** The default `SKIPPED`
  on it is the hook's failure, `compute_hook_returned_invalid` with `ctx.error` `default_skipped_on_input` (it was accepted, and the
  sentinel reached the function). A parameter whose default is `SKIPPED` raises `NodeDeclarationError` with
  `default_skipped_on_input`: `SKIPPED` is only ever the value of a skipped output.
- **Breaking: a branch is an output whose default is `SKIPPED`, and a call gives a value to at most one branch of
  a node.** `Param(exclusive_group=)` is gone, and with it `FieldDescription.exclusive_group`, the declaration code
  `exclusive_group_on_input`, the `default_invalid` variant `on_exclusive_group` and the record errors
  `required_in_exclusive_group` and `default_in_exclusive_group`. Write `large: int = SKIPPED` on a returned
  dataclass, or `(int, SKIPPED)` from a `compute_outputs` hook. Branches of one node may be wired into one input, as
  outputs of one group could.
  - A return that gives two branches a value fails the call with `invalid_output`. That holds for every output
    that defaults to `SKIPPED`: outputs that defaulted to `SKIPPED` with no group, and could both have a value, are
    now refused when both get one. A branch returned as `None` counts as given. A return that gives no branch a value
    skips them all, so the `neither` output a node needed for "none of them" is gone.
  - `SKIPPED` on an output that isn't a branch fails the call with `invalid_output` too. `SKIPPED` means a way not
    taken; an output that may be missing beside other outputs is typed `X | None` and returned as `None`. A node has
    one set of branches, so a node with two independent ways to go is two nodes.
  - An answer gives a value to at most one branch, and an answer that leaves every branch out skips them all.
  - A subgraph node's outputs are never branches, even one read from a branch inside, so no two of them may share
    an input (provisional). Join branches inside the subgraph. An output that needs a branch inside may still be
    skipped, and `describe()` shows it so: `"required": false`, with no `default`. Two lists collected from its
    outputs on different branches of one node inside are still `series_from_different_branches`.
  - The compile problem `alternatives_from_different_loops` is `branches_from_different_loops` (provisional).
    `several_edges_into_one_input`'s message gives one fix, a node with an input for each edge; its entry on the
    problems page says when the default `SKIPPED` fixes it instead.
  - Where its node is not called (an argument `None` its input does not take), a branch is `SKIPPED`, and the nodes
    wired to it are skipped.
- **Breaking: a call not made because an input got `None` has one name, none input (provisional).** It was
  "not called", which also describes a skipped call. `run.not_called` is `run.none_input_calls`, a list of
  `NoneInputCall` (was `NotCalled`); a `CallCount`'s and a `RunProgress`'s `not_called` is `none_input`, so the
  `count` of `node_progress` and `node_complete` events has `none_input`; a saved run's `not_called_because` is
  `none_inputs`. A run saved before can't be read.
- **Breaking: a kind of problem has one code, its function's name (provisional).** A problem's `ctx` has no
  `variant`: each case that was a variant has a code of its own, so a host translates by `type` alone, and a code's
  suffix names the cause, never the fix. Old to new:
  - compile: `compute_hook_failed` / `raised` is `compute_hook_raised`; `context_does_not_fit` / `lacks` and
    `no_context_type` are `context_does_not_fit_missing_member` and `context_does_not_fit_no_context_type`;
    `invalid_value` / `bound_default` is `invalid_value_bound_default` for an input and the new
    `output_default_bound_invalid` for an output, whose message says to change the default;
    `series_not_nested` / `inside_subgraph` is `series_not_nested_inside_subgraph`; `type_mismatch` /
    `make_a_series`, `add_column` and `one_value` are `type_mismatch_whole_table_or_list`,
    `type_mismatch_missing_column` and `type_mismatch_series_given_one_value`; `input_has_no_value` /
    `constant_only` is `input_has_no_value_constant_only`; `unknown_output` / `make_a_series` is
    `unknown_output_field_of_whole_table_or_list`; `lists_from_different_loops`, which is about two branches of
    one node, is `series_from_different_branches`.
  - call: `cannot_move_to_version` / `rewrite_failed` and `raised` are `cannot_move_to_version_rewrite_failed` and
    `cannot_move_to_version_raised`; `invalid_answer` / `reason` is `invalid_decline_reason`; `invalid_context` /
    `no_context_type` is `invalid_context_no_context_type`; `invalid_input` / `make_a_series` is
    `invalid_input_list_for_one_value`; `invalid_limit` / `seconds` is `invalid_limit_seconds`; `not_a_chain` /
    `constant_only` is `not_a_chain_constant_only`.
  - declaration: `limit_not_supported` is `param_limit_not_supported`, and its `limit_value` is
    `param_limit_value_invalid`, so a `Param`'s limits don't share a stem with a run's (`invalid_limit`);
    `choices_invalid` / `not_a_value` and `no_json_form`, `named_value_invalid` / `enum` and
    `alike`, `default_invalid` / `none_not_taken` and `skipped_on_input`, `hook_parameter_unknown` /
    `not_by_name` and `var_keyword`, `pydantic_field_in_hint` / `discriminator` and `type_declaration_invalid` /
    `not_on_a_class` and `not_writable` are `choices_invalid_not_a_value`, `choices_invalid_no_json_form`, `named_value_invalid_enum`, `named_value_invalid_alike`,
    `default_none_not_taken`, `default_skipped_on_input`, `hook_parameter_not_by_name`,
    `hook_parameter_var_keyword`, `pydantic_discriminator_in_hint`, `type_declaration_not_on_a_class` and
    `type_declaration_not_writable`; `context_type_differs` / `on_subregistry_only` is
    `context_type_on_subregistry_only`; `type_not_supported` / `host_type_not_supported` is
    `host_type_not_supported`.
  - `check_registry`: what an example meets shares the prefix `example_`: `run_failed` is `example_run_failed`,
    its `not_compiled` and `refused` are `example_not_compiled` and `example_refused`, and `hook_failed` /
    `on_an_example` is `example_hook_failed`, with a message of its own ("A hook fails on the example, ...").

  The input a problem is on is `ctx.input` on fourteen kinds, where it was `ctx.field` (`branches_from_different_loops`,
  `invalid_value`, `invalid_value_bound_default`, `several_edges_into_one_input`, the four `type_mismatch` codes,
  the two `input_has_no_value` codes, `input_takes_only_a_constant`, `graph_input_used_two_ways` and
  `unknown_input`) or `ctx.name` (`invalid_edge_value`); `output_default_bound_invalid` has `ctx.output`.
  `ProblemCatalogue.problem` is a plain decorator, `@catalogue.problem`, with no `code` or `variant`, and
  `ProblemKind.variant` is gone. `graph_inputs_share_a_name` and `graph_outputs_share_a_name` say "The inputs ..."
  and "The outputs ..." where they said "Two inputs" and "Two outputs" whatever the count. An edge is
  `type_mismatch_whole_table_or_list` only into an input that takes no list, by the rule a call uses for
  `invalid_input_list_for_one_value` (every option of the input's type refuses a list as a whole): a list into
  `list[str]`, `tuple[int, ...]`, `set[int]` or `int | list[str]`, and a table into a list input, are plain
  `type_mismatch`, where they had the make-a-series message, which says the input takes one value.
- **Breaking: `invalid_output` names its output `ctx.output` (provisional).** It was `ctx.name`, so two kinds named a
  node's output with two keys; `output_default_bound_invalid` already has `ctx.output`, and the reference's rule says a
  node's output is `ctx.output`. `ctx.name` is now only a graph input a call names.
- **A problem's `ctx` keys follow the kind's declared order (provisional).** `ProblemKind.__call__` writes `ctx` in the order of
  the kind's function's parameters, not the order the caller gave its keywords. No key or value changes.
- **Breaking: one value is never a list of one (provisional).** An edge that gives each call of its node one value,
  wired to a `Series[T]` input or to a `Table[R]` input that collects records, is the compile problem
  `type_mismatch_series_given_one_value`, at that input, with the edge's source in `ctx.source` and the item type in
  `ctx.item`. An edge gives each call one value when its source is in no loop, or in a loop the node is inside too.
  Such a graph compiled and called the node with a list of one; it now has a problem and doesn't run, and a run saved
  from it can't be read back (`Run.from_json` raises `InvalidCallError`). Declare the input as one `T`, collect the
  loop first in a node outside it, such as `series-collect`, or use the node's own option for one value, such as
  `table-collect`'s `constant_columns`. A subgraph is held to it when its own record is compiled, and is refused
  wherever it is used. `describe()` no longer gives the source `{"mode": "collected", "series": null}`: `collected`
  always names its series. An output declared `Any` into a `Table` input is a whole table wherever it gives each call
  one value, inside a loop too. `table-collect`'s `columns_not_aligned` no longer suggests `constant_columns` for a
  column of one value, since such a column is now refused when the graph is compiled.
- **A graph's inputs and outputs are read from the compiled graph (provisional).** `graph.inputs` lists the graph
  inputs in the order the nodes run in: a node's after those of the nodes it reads from, by node id where that
  leaves a choice, and a node's own in the order it declares them. The record keeps no order, so
  `GraphRecord.input_names` and `required_inputs`, and a `SubgraphDefinition`'s `input_names`, `required_inputs`
  and `output_names`, are sorted; they followed the order the record's JSON listed its nodes in, which
  `graph.inputs` and the palette didn't. `graph.outputs`, and `graph.describe().outputs`, are the graph outputs by
  name, sorted, each the `FieldDescription` of what a complete run gives, where they were a list of names: as a
  subgraph node's output is described, one that needs a branch is `"required": false` with no `default`, and one
  whose node is inside loops is a list with its `loops`. `graph.asks` says whether a node of the graph, or of a
  subgraph in it, may stop to ask a person. A subgraph's palette entry in `registry.describe()` has its own graph's
  `inputs`, `outputs` and `asks`. A subgraph whose graph has problems on the registry's nodes is left out of
  `registry.describe()`, where it was an entry with no inputs or outputs and no word why, with one warning on the
  `conductor` logger naming it and its problems' codes, and `check_registry` reports it as `subgraph_broken`, one for
  each of its graph's problems, whether or not it has examples, which then aren't run.
- **A field's `default` is what leaving it out gives, and `null` is only `None`.** `default` is no key in the JSON
  (pydantic's `MISSING` in Python, as on `InputEntry`) when leaving the field out gives no value: with
  `"required": true` it is refused (`text: str` has no `default` key, where it had `"default": null`), and with
  `"required": false` it is a branch, an output whose default is `SKIPPED`, skipped when left out.
  `"default": null` is a default of `None`. A `default` with `"required": true` is the value an
  editor starts with: the one question of a node with one output is required, a branch too, since it is answered
  with a value, and keeps the `Ask`'s default. `loops` without `is_series` is refused
  (`loops_without_series`). A host that writes these records with `exclude_none` turns `"default": null` into a
  skipped branch. An output in `describe()`, `registry.describe()` and
  `graph.describe()` is the question its type sets, the same record a waiting call shows without an `Ask` on a
  node of several outputs: `-> str | None` is `"required": false, "default": null` (it was `"required": true` in
  the palette and `false` in the question). A subgraph's output returned inside loops keeps `is_series` true, and its `type`, `json_schema` and
  `choices` are the value the node inside returns (`"type": "str"`, where it was `"Series[str]"` for an output
  inside two loops); its `loops` give one level of list per entry, and a `Series` it returns outside every loop
  has `loops` `[]` and is one list. A `Waiting`, and so a stored outcome of a call that waits (the providers'
  `Outcome`), written before this change no longer loads: a required question there has `"default": null`.
  There is no migration.
  An `Ask`'s defaults fill an answer in, so they give a value to at most one branch: defaults for two branches of
  one node fail the node (`invalid_ask`). A branch with a default that an answer leaves out is skipped when the
  answer gives another branch a value, and gets its default when it gives none.

- **The run's context is one object of the host's class.** Name the class once,
  `NodeRegistry(context_type=AppContext)`, and pass an instance to each run,
  `graph.run(context=AppContext(...))`. A node that needs it takes it as its first parameter,
  `def sign(ctx: Context[AppContext], text: str)`, and a test calls `sign(app_context, "Done")`. A node may
  name the class, a base of it, or a `Protocol` the class satisfies. Compiling checks each node against the
  class, a `Protocol`'s members by name only (`context_does_not_fit`, at the node and parameter); starting a run checks the object
  (`invalid_context`, `context_missing`). `NodeRegistry`, `Graph` and `Run` are generic in the class, so a
  type checker refuses a `context=` of another class. The context is never saved with a run.

- **Five modules.** Import from `conductor`, `conductor.widgets`, `conductor.events`,
  `conductor.testing` and `conductor.about`, the reference. Everything else is `conductor._internal`.
  A type's `accepted_as` is read from `registry.describe().types`.
- **`conductor.about` has no names.** Its `__all__` is empty. `get_content`, `list_sections` and `get_section`
  are what `python -m conductor.about` calls, and are not part of the public API.
- **The problems page is printed from the code.** `python -m conductor.about problems` prints every
  problem code with an example, its message, what its `ctx` keys mean and how to fix it, each kind from the
  docstring of the function that declares it, and a test runs every example. The docs build writes the same
  page; there is no `docs/problems.md`. `ProblemCatalogue(title, text)` is one section of the page,
  `ProblemCatalogue.joined(...)` one catalogue of several, and `ProblemKind.section` says which a kind is in.
  `PROBLEMS` has three: compiling a graph (in `graph.problems`), refused calls (raised as `InvalidCallError`)
  and why a run stopped.
- **Problem messages are short and plain.** Each says what is wrong, then what to do, in a sentence or two
  (`'upper' has no input 'txet'. Its inputs are text.`), and so do the reasons inside them. Codes and `ctx`
  are unchanged, so a host that translates by code needs nothing new. The messages of `execution_failed`,
  `timeout` and `run_timed_out` name the exception's class and the seconds from their `ctx`.
- **Seven names in problems are plainer.** `bound_invalid` is `param_limit_not_supported`, with `ctx.unknown_limits` for
  `unheld`; `type_not_carried` is `type_not_supported`, and `host_type_not_supported` for a class under `@host_type`;
  `not_offered` is `unknown_graph_input`; `not_a_place` is `invalid_place`; a `NodeError` without a code of its
  own is `node_error`, where it was `failed` (a run's status `failed` stays); the `ctx.whole` of
  `type_mismatch_whole_table_or_list` and `unknown_output_field_of_whole_table_or_list` is `ctx.kind`; and `compute_hook_failed`'s `ctx.error` `added_input_taken` is `compute_hook_returned_invalid`'s `added_input_exists`.
- **A refused call lists its problems in `problems`.** `InvalidCallError.problems` is every `Problem` the call
  was refused for, as `graph.problems` is for a graph, and `read_json` raises it too.
- **`type_not_named` gives a declaration to paste.** For a class with no `@host_type`, the message suggests an id and a title made from the class's name, `@host_type("invoice-line", title="Invoice line")` for `InvoiceLine`, where it wrote `@host_type("...", title="...")`; `ctx` gains `suggested_id` and `suggested_title`. An id that would be one of conductor's own gets `-type` after it.
- **What a person sees for a type is declared.** Each entry of `registry.describe().types` has the
  `id` and `title` its owner declared: conductor's own types from one fixed table (`str` is `Text`,
  `int` `Whole number`), a host's class from its `@host_type` declaration;
  nothing falls back to the class's name, and a class with no declaration is `type_not_named` where its node is declared. The types a registry holds are
  every class a field's type id names: each member of a union, a `Literal`'s value types and a
  generic's class (`list[str]` names `list`); `Any`, `None`, a type alias and a table name none.
- **A node is a typed function.** `@node("id", title=, description=)` (registers nothing) or
  `@registry.node(...)` returns a frozen, callable `FunctionDefinition` (`subgraph(...)(record)` returns a `SubgraphDefinition`,
  with no function to call; `NodeDefinition` is either). Its title and description are
  required, and the docstring never reaches a person. The signature is the interface:
  `Annotated[str, Param(title=, widget=, ge=, ...)]` for what a person sees and the limits a value
  keeps (pydantic metadata, so its limits and choices apply inside your own `BaseModel` or dataclass
  too, and a field's `json_schema` carries the limits but not its title or description; `Param`s stacked in one `Annotated`, an alias wrapped again, merge, the later
  winning, and a `Param` on a `Series` item is the field's; a table's column
  typed `Annotated[...]` is refused, since its limits would go unread), `-> X` for one output named `result`, a frozen dataclass for several (kept as the definition's `outputs_dataclass`). A
  `bool` never lands on a number, a `Literal` of numbers or a union holding a number and no
  `bool` (`int | float`) included, as a field's value or in a table's column. `async def` and
  auto-discovery are gone. A node declared wrong raises `NodeDeclarationError` at import, whose
  `problems` is the one problem behind it, a kind of `DECLARATION_PROBLEMS`; `python -m conductor.about problems`
  lists each kind with an example. A `Param` is checked against its type as pydantic
  builds the schema: a limit pydantic doesn't hold on the type (`ge` on a `bool` or an `IntEnum`), or
  a choice that isn't a value of it, is `param_limit_not_supported` or `choices_invalid`, so a host's `BaseModel`
  with one fails as its class is made. Limits are pydantic's own: bounds on numbers, a `Decimal`,
  dates, times and durations, lengths on texts, bytes and collections. `Param` is a frozen dataclass whose limits
  are its fields, so an argument it doesn't have is Python's own `TypeError`.
- **A graph record becomes a node, a subgraph, with `subgraph`.** `subgraph("id", title=, description=)(record)`
  and `registry.subgraph(...)` take a `GraphRecord` or its JSON text and return a
  `SubgraphDefinition`. `node(...)` and `@registry.node(...)` take only a function. `subgraph` has no
  `policy`, `compute_added_inputs` or `compute_outputs` argument, so passing one is Python's own `TypeError`,
  and the declaration code `not_for_a_graph` is gone.
- **A node's title and description are not empty.** `@node`, `@registry.node`, `subgraph` and
  `registry.subgraph` raise `NodeDeclarationError` with the code `empty_title_or_description` for a
  `title` or `description` that is empty, only spaces or not text [provisional]; the message names the node and
  which one. An id that isn't text is `invalid_node_id`, and a `version=` that isn't a whole number of 1 or more
  (`0`, `-3`, `True`, `"2"`, `1.5`) is the new code `invalid_version` [provisional], where each was taken or
  raised Python's own `AttributeError` or `TypeError`.
- **The types decide one or many.** A `Series[X]` output wired to an input typed for one `X` runs
  that node once per item, and everything after it; the node never sees the difference. A
  `Series[X]` input collects every item in one call, and a `Table[R]` input collects records into a
  table. An outer value, from a node outside the loop, is the same on every call. A table (a whole
  `polars.DataFrame`) and a list are each one value: `series-split` in `syv-conductor-nodes`, or any node
  that returns a `Series`, makes the items, and an edge may name one field of a record
  (`series-split.result.email`). A list given to an input of one value is refused (`invalid_input`), and
  so is one written as a constant (`invalid_value`). Items of two Series meeting at one node are refused
  (`series_not_nested`) until one is returned by a node inside the other's loop. `graph.describe()` says how often each node
  runs, before any run. A `None` keeps the node from being called for that item unless its input
  says `| None`. A node that collects a loop leaves out only what was decided per item: when no item
  could have a value, for a skipped branch or a `None` outside the loop, it is skipped or not
  called, with two items as with none, and `Series[X | None]` receives a `None` in each item's place. Each item
  of a list given there is held to the input's limits and choices, as one value is. A choice is
  read as its answer would be: strictly from its JSON form, held to the field's limits, so an enum's
  value or a date's ISO text is a choice. A value an edge gives is held to the input's limits and choices
  alike (`invalid_edge_value`).
- **Compiling never raises for a fault in the graph.** `Graph.compile(record, registry)` returns a `Graph`;
  read `graph.problems`, each a `Problem(type, loc, msg, ctx)`. Only a record of the wrong shape raises,
  `InvalidCallError` with the code `invalid_record`. A constant or a graph input that its field refuses
  (`invalid_value`, `invalid_input`) carries pydantic's value-free `errors` in `ctx`. `make_graph(*nodes, edges=, constants=, defaults=, inputs=, outputs=)`
  writes a record from Python and compiles nothing: it raises one error listing what it can't write
  (a value for a node it wasn't given, a value with no JSON form, a place that is not `node.name`
  (`invalid_place`), two graph inputs or outputs of one name (`graph_inputs_share_a_name`,
  `graph_outputs_share_a_name`), what the record refuses (`invalid_record`), a node with a `compute_outputs` hook
  whose outputs no edge reads and `outputs=` doesn't name, which would give the graph nothing from it
  (`hook_outputs_not_named`), a chain it can't draw; a chain goes through a subgraph by the graph inputs and
  outputs its record declares, and never wires
  an input that takes only a constant, and when only such inputs are left it is `not_a_chain_constant_only`, naming them in
  `ctx.inputs`), and an input a
  node doesn't have, a value its field refuses or an edge whose types don't meet (`type_mismatch`) is in
  `Graph.compile(record, registry).problems`. A record's constants, defaults and
  `display` are JSON values; `make_graph` writes each constant and default as its JSON, a float that isn't
  finite as `"NaN"` or `"Infinity"`, so a record equals its JSON, and an `upgrade` rewrite is written
  the same way (`invalid_value` where it has no JSON form). A subgraph offers
  the graph inputs and outputs its record declares, by name (`e.result`), and `subgraph` takes no `policy` or hook. Its inner nodes are laid out in the graph that uses it, each with a call per item of every loop the subgraph
  node is inside, so they are called, retried, timed out, cancelled and reused by `run.again` as the
  same nodes drawn in the graph itself, and say their events once, under `("e", "upper")`. A `None`
  arriving at the subgraph node leaves the inner nodes that read it uncalled, and `run.none_input_calls`
  lists them. A saved run has an inner node's outcomes under `e/upper`. A node outside reads an output returned inside loops as it would with the same nodes drawn in the graph, and an inner decision's group
  is named by its node's path, `d1.answer` (for a node `d1` in the subgraph), `g/d1.answer` for one a subgraph deeper. A problem inside has the subgraph node (and item) in front
  of its `loc`; there is no `embedded_graph_broken`. Every node event, a `Waiting` and a
  `NoneInputCall` say where they are with one `loc` in place of `node_id` and an item: a call's is a
  problem's (`("e", 0, "ask")`), a node's its id behind the subgraph node's (`("e", "ask")`); a waiting id inside reads `e@0/ask~3fa2c0d1`.
  Every node event's `loc` names the node, `node_waiting` and `node_retry` included; the call is in the `loc` of
  its `waiting` or its `cause`.
  An id with an index that is not a number is `not_waiting` at the place read up to it: `e@1/ask@x` at `("e", 1, "ask")`.
  A waiting id has one spelling, the one `run.waiting` gives (or that text without `~` and its fingerprint); any other names nothing and is `not_waiting`: `e/ask@0` for `e@0/ask`, `ask@01`, `ask@0~`, `ask@١` (an Arabic digit), and an index longer than Python converts to an `int` (it raised `ValueError`, a 500 from the worker's routes).
- **`Graph.compile`, `registry.upgrade` and `subgraph(...)(record)` take a `GraphRecord` or its JSON text** (`str` or `bytes`),
  as their type says. `InvalidCallError` with `invalid_record` problems is what JSON text of the wrong shape raises.
- **`registry.upgrade` drops an input the newer version doesn't have, for a subgraph as for a function.** What it gets (a constant, a graph input
  or the edges into it) goes once a `rewrite` has read its constant or default, and a subgraph's inputs are the graph inputs its record names, so
  a subgraph node upgraded to a version that takes fewer graph inputs compiles clean (it kept them, and compiling said `unknown_input`). A subgraph's
  newer version that drops a graph input or output of the one it replaces needs an `Upgrade` saying where it went, as a function's does
  (`upgrade_missing` where it is declared).
- **An `Upgrade` renames only declared fields.** `@node` and `subgraph` raise `NodeDeclarationError` with the new
  declaration problem `upgrade_renames_unknown` (`ctx.old`, `ctx.new`) when a name in `renamed_inputs` or
  `renamed_outputs` isn't a field the replaced version declares, or its new name isn't one the new version declares;
  such a rename moved nothing, and the upgraded graph lost what the field got. A version with a `compute_added_inputs`
  or `compute_outputs` hook has its renames of that kind of field left unchecked.
- **`Run.from_json` holds a node to its current `Policy`.** It drops the outcomes a saved run could still reuse for a
  node whose `Policy` says `reuse_when_inputs_match=False` now; such a node is called again for every item, where a
  run saved before the policy changed reused them. The next save writes no `reusable` entry for it.
- **Breaking: a graph's inputs are given as one mapping, never as keywords.** `graph.run({"text": "a b"})`,
  and the same for `graph.with_inputs`, `run.with_inputs` and `run.again`. Every keyword is a setting of the call, and
  each is in the signature: `graph.run(inputs={}, *, max_items=, previous=, context=, timeout=)`, `inputs=` included.
  So a graph input named `context`, `timeout`, `previous` or `max_items` reaches its node, where a keyword of that name
  was taken for the setting, and an input given as a keyword is a `TypeError`.
- **A run is one value.** `graph.run(inputs)` returns a `Run` whose `status` says whether it is
  ready, running, waiting, complete or ended early, and whose `cause` is the `Problem` that ended a run that failed,
  stopped, was cancelled or timed out (`None` otherwise). One rule reads it from the outcomes and from how the last
  `resume()` left the run: a checkpoint is `running` until it is saved, and read back is what its outcomes say; a run whose every call returned is `complete` however its resume
  ended; a failed call keeps the run `failed` until it is tried again; and an answer to a cancelled or timed-out run
  reads it from its outcomes, so answering the last question makes a run `ready`, or `complete` when no call is left,
  before any `resume()`. It pauses for a person (`raise Ask(...)`, then `run.with_answers(...)` and `run.resume()`), writes
  itself as JSON (`run.to_json()`, `Run.from_json(saved, graph)`, which reads a save from its JSON text; a save that
  doesn't fit (it may have been changed, other code may have written it, or a value may not read back from its JSON; a new run is the remedy) is `saved_run_unreadable`, `ctx.where` its place in the save, such as `outcomes.upper.outputs.result.1`; a save has each node's
  outcomes as a list per output with one entry per call, a `Series` of records returned as a table as that table, under `e/ask` for a node inside a subgraph, and how the last resume ended early under `ended_early`, `null` for one that did not;
  a float that isn't finite is saved as its text and read back by its declared type, the float under `float` and the text under `Any`) and runs again on changed inputs,
  calling the nodes only for the items that changed (`run.again(...)`). A run's `max_items` is a whole number of
  at least 1 and its `timeout` and `checkpoint_every` finite seconds above 0, a `bool` neither,
  refused at the call as `invalid_limit` (`invalid_limit_seconds` for `timeout` and `checkpoint_every`), whose `loc` and `ctx.limit` name the limit and whose `ctx.value` is the value given. A number of seconds of another type, such as `Decimal("1.5")`, is read as the `float` `1.5`.
- **Every exception conductor raises says what is wrong as problems.** `NodeDeclarationError`, `InvalidCallError` and
  `NodeError` each have `problems`, a list of `Problem`. A `NodeError` makes its one problem where it is raised, from its
  message, `code` and `ctx`, with an empty `loc` that the run fills with the call's; it has no `code`, `ctx` or `message`
  attribute, and `NodeDeclarationError` no `code` or `problem`.
- **A problem never quotes an exception's text.** A `compute_...` hook that raises anything but a `NodeError` is
  `compute_hook_raised` with `ctx.error` the exception's class, and no `ctx.reason`; the
  text and traceback go to the `conductor` logger. A hook's `NodeError` keeps its code and message. A field a hook returns that `@node` would refuse keeps the
  declaration's code in `ctx.error`, and the message names the field but not the declaration's text. `question_unwritable`
  has `ctx.error`, the class of the exception writing the input raised, in place of `ctx.reason`, and logs the text.
  An `Upgrade`'s rewrite that raises makes `registry.upgrade` raise `InvalidCallError` with
  `cannot_move_to_version_raised`, at the node: `ctx.version` and `ctx.target` the step whose rewrite raised, `ctx.error` the
  exception's class, and the text in the log. A rewrite's `NodeError` keeps its message as `ctx.reason`, its code as
  `ctx.error` and its `ctx` as `ctx.error_ctx`, under `cannot_move_to_version_rewrite_failed`; so does a rewrite that returns
  no dict whose keys are text, such as `None`, with `ctx.error` `not_a_mapping`. The author's exception, or a bare
  `AttributeError` for a `None` return, no longer escapes. `check_registry` reports
  it, like any refused upgrade, as `upgrade_broken` with the refusal's code in `ctx.problem`.
  `saved_run_of_another_graph` has `ctx.saved` only when the save's `graph` has a fingerprint's form, and its message
  quotes only this graph's fingerprint.
- **A problem never quotes a value a graph carries, a key inside one included.** A record's fault inside a
  constant, a graph input's `default` or `display` has a `loc` that stops there
  (`("record", "nodes", "a", "inputs", "x", "constant")`), and a key written twice in the record's JSON
  (`repeated_key`) isn't quoted. A problem may name what a developer or host wrote: ids, places, field names (a
  `compute_...` hook's too, even one built from a constant), waiting ids and a type's limits.
- **A host's own pydantic error quotes nothing of the value.** In `reason` and `ctx.errors`, an error keeps its
  message only when pydantic or conductor wrote it. A `PydanticCustomError` a host type's validator raises, with
  the value in its `ctx` or in its text, keeps its `type`, and its message is "The value did not pass its type's
  own check", with no final period, so a problem that quotes it as its reason reads as one sentence, as a `ValueError`'s is. That holds when it gives one of pydantic's types, like `value_error`: a
  message counts as pydantic's only when pydantic writes the same text for that type and `ctx`. So do pydantic's `bytes_invalid_encoding` and `timezone_offset`, whose
  messages quote a byte or the offset of the value.
- **A hint that is no type at all is `type_not_supported`.** `def f() -> 5` or `-> [int]` raised a bare `TypeError`
  at `@node`, and a `compute_outputs` hook that returned `{"out": 5}`, a list, a dict or a set made `Graph.compile`
  raise one. Now `@node` raises
  `NodeDeclarationError` with `type_not_supported`, and the hook is `compute_hook_returned_invalid` with `ctx.error`
  `type_not_supported`, as a hook's undeclared class is with `type_not_named`.
- **`conductor.testing` exports `REGISTRY_CHECK_PROBLEMS`**, the catalogue of `check_registry`'s codes, beside
  `check_registry`. The reference says what it, `PROBLEMS` and `DECLARATION_PROBLEMS` each hold, and its translation
  test names all three.
- **A problem's `ctx` is JSON.** `Problem.ctx` is typed `Mapping[str, JsonValue]`, and
  `NodeError(ctx=)` and `ExternalError(ctx=)` read theirs where the error is made: a value with no
  JSON form (an object, a set, a tuple, a date) raises pydantic's `ValidationError` at the node's
  own `raise`, so the node fails as `execution_failed` and the run still saves. The `loc` of each
  entry in a problem's `ctx["errors"]` is a list, so a problem equals what its JSON reads back.
- **Only an outside failure is retried.** `ExternalError`, or a class in `Policy(retry_on=)`;
  a `NodeError`, a wrong output or an `Ask` never is, whatever `retry_on` names.
  A node's own exception goes to the `conductor` logger; only its class name reaches the problem, as `ctx.error`.
- **Two companion packages, rebuilt on this API.** `syv-conductor-nodes` holds 36 nodes that only
  compute, a registry per module (`conductor_nodes.text.registry`) and all of them in
  `conductor_nodes.registry`. `series-split` takes a table or a list apart into a `Series` (a node may return a table for `Series[R]`, and a run stores that table, not a copy of its rows), `series-stack` puts the table of each item into one, `table-collect` puts records back into a
  table with new columns (its `records` input is a `Table`, so records split from a table arrive as that table and the new columns are added beside its own, with no copy and their column types kept), `text-list` gives a list of texts a person enters, and `text-split` is titled "Split text" and splits on a newline. A saved run writes each output's value under that output, a table that two outputs share (a table read and then split) under both.
  `math-round` rounds a half away from zero (2.5 to 3), and
  `regex-replace` fails a replacement that refers to a missing group as `invalid_replacement`. `syv-conductor-providers` holds `run_router`, a FastAPI router that
  starts, follows, answers, resumes and cancels runs on several workers over a `RunStore` you
  write ([`docs/serving-runs.md`](docs/serving-runs.md)). An answer sent while the run is running waits for it to
  stop, and one that then finds nothing waiting, or a run that needs a resume, is kept in `refused`; `again` runs on the nodes of
  the request that asks; `resume` takes no body, and a ceiling it raises applies to the run from then on; a run whose event
  the store fails to take stops before its next call, so no node is called once the run reads `failed`; `GET /runs/{id}` says the `cause` of a run that
  didn't complete; a run's `outputs` there are what `run.outputs_to_json()` writes, each by its output's type, so a host's type that only its own
  schema can write completes over HTTP as it does in the engine, a table reads as its columns and a float that isn't finite as `"Infinity"` or `"NaN"`;
  `syv-conductor-providers` no longer depends on polars.
  There is no canvas package.
- **`syv-conductor-nodes` has a `series` category.** It holds `series-collect` ("Collect"), which runs once
  and gives the value of every item as one list. `split` and `stack` are `series-split` and
  `series-stack` and live in `conductor_nodes.series`. `decision` is `control-decide`. `logic-if-empty` and
  `logic-if-equals`, which sent a text one way or the other, are gone: `logic-is-empty` and `logic-equals` give a
  yes or no on a value of any type, to wire into `control-decide.when`. `control-switch` sends its `value` down the
  branch its `case` names, one output per name in the constant `cases`, or down `other` (a case named `other` or
  given twice is `compute_hook_failed`, `ctx.error` `case_named_other` or `case_given_twice`), and `control-default`
  gives its `value`, or its `fallback` when the value is `None`. A graph record that names an old id
  needs the new one, and a run saved from a graph that used an old id is refused by `Run.from_json`
  (`saved_run_of_another_graph`), because the graph's fingerprint includes the ids of its nodes.
- **`syv-conductor-nodes` gives a widget only where it says more than the type.** A text to work on
  keeps its `Textarea`, and so does `json-parse`'s JSON text. Every `TextInput` on a `str`, `NumberInput` on
  a number, `Switch` on a `bool` and `ListInput` on a list is gone, so those inputs' `widget` is `null` in
  `describe()`.
- **Every node in `syv-conductor-nodes` costs nothing.** Each one declares `Policy(costs_nothing=True)`, every
  version included, so `run.resume(only_costs_nothing=True)` calls it, and an estimate counts the calls after a
  `series-split` or a `text-list` before any call that costs something.
- **`control-decide`'s `when` has no default** [provisional, breaking]. A `when` with nothing wired to it was true,
  so every value went to `if_true` and nothing said so. A saved graph that leaves `when` unwired is now
  `input_has_no_value` when it is compiled, and `make_graph` makes `when` a graph input. `describe()` shows `when`
  required, with no default.
- **`json-parse` and `json-stringify` refuse a number that isn't finite, with `not_json`.** JSON has no number that
  isn't finite, but Python's `json` reads `NaN`, `Infinity` and `-Infinity`, reads a number too large for a float
  (`1e999`) as `inf`, and writes `inf` and `nan` back. So `json-parse` gave `[nan, 1]` for `[NaN, 1]`, which a saved
  run read back as `["NaN", 1]`, and `json-stringify` wrote text that isn't JSON. `json-parse` now refuses the three
  words and a number too large for a float, and `json-stringify` a value holding `inf` or `nan`.
- **`json-get` reads each part of its path as a key exactly as written** [breaking]. It stripped the spaces from
  each part, so the key `" a"` couldn't be reached, and `user. name` found `user.name`. Now `" a"` finds `" a"`, and
  `user. name` finds the key `" name"`. A saved graph whose path has spaces around a part, such as `user. name` or
  `items.1 `, gets `None` where it got the value, unless its JSON has a key with those spaces. A key with a dot in it
  still can't be reached, since the dot always splits the path.
- **`text-replace` and `text-contains` refuse an empty `needle`** [breaking], with `Param(min_length=1)` as
  `text-split` refuses an empty separator. An empty `needle` put the replacement between every letter, and was in
  every text. A saved graph with the constant needle `""` is now `invalid_value` when it is compiled, and an empty
  needle given as a graph input is `invalid_input`. `describe()` shows `min_length` 1 on `needle`.
- **A declared choice's `id` is the field type's JSON form.** `Decimal` choices show `"1"` and a `datetime` its seconds, so the id a question shows is an answer it takes.
- **A choice is a `Choice` wherever conductor gives one.** `FieldDescription.choices`, on a field `describe()` writes and on a waiting call's question, is a tuple of `Choice`, each with every field a host gave it; its JSON is unchanged. `Choice("ida", "Ida B.", team="sales")` carries `team` without a subclass, and a description or a saved run read back keeps it. A `Param`'s choices and an `Ask`'s are read by one rule, `read_choices`: two ids are one choice when their JSON is, with the keys of a dict in any order, so `Param(choices=[float("inf"), float("nan")])` is taken and `[{"a": 1, "b": 2}, {"b": 2, "a": 1}]` is one choice given twice, refused by `Param` (`choices_invalid`) and by an `Ask` (`invalid_ask`) alike. An `Ask`'s choices are read by the validator of the output, as an answer is, so a yes or no is never a number there. An `Ask` refused for its choices says so as `invalid_ask` with `ctx.reason` "the choices for 'result' are refused: one is given twice" (or "none is given", "one has no JSON form", or the reason its type refuses one).
- **An answer is read as a return is.** A person's answer is read from its JSON form by the schema that reads what the node returns, so a host type whose own check gives another class refuses an answer as it refuses a return (`invalid_answer`, "its type's own check gave a dict, not a Square"). An `Ask` choice with no JSON form fails the call with `invalid_ask` ("the value has no JSON form"), where it raised out of `resume()`.
- **A value matches a choice by its JSON text, wherever it is checked.** An input, a return and an answer are held to choices by one rule: a value and a choice id are the same when their JSON is, with a dict's keys in any order and a float that is not finite written as its text (`"Infinity"`). On an `int | float` or `Any` field with the choice `1`, the values `1.0` and `True` are not one of the choices, as an input or a return too (`Param(choices=)` took them by Python's `==`). Two fields whose choices differ only as Python compares them hold a value each to their own: `Annotated[Any, Param(choices=[True])]` declared after `Param(choices=[1])` refused `True` and took `1`, and `Param(choices=[True, 2])` on an `int` declared after `[1, 2]` was not refused. Only an answer picks each choice once (`picked_twice`): a node may return, and an input may take, a `Series` that holds a value twice. `None` is never one of the choices on `Annotated[str | None, Param(choices=["a"])]`, which takes `"a"`, and `None` beside it. A value holding a float that is not finite is hashed by that text too, where some hashes held the bare word `Infinity`, which is not JSON: for a call that received one, the fingerprint a later run reuses its outcome by (a table's digest inside it included) differs from an earlier build's, and so does `graph.fingerprint` of a record read from JSON text with a bare `Infinity`, now the same as the record written with `"Infinity"`, so a run of that graph saved by an earlier build is refused when loaded. A waiting id is unchanged. `None` is no pick: a type that takes `None` takes it beside its choices, so `run.with_answers` takes `None` for `Series[str] | None` with choices (it raised `TypeError`), and an `Ask(choices=[...], default=None)` on an output that takes `None` waits with that default (it failed with `invalid_ask`, or raised `TypeError` on a `Series`). An answer outside the choices reads "the value is not one of the choices", as an input does.
- **Fixed choices go on the type, and `Ask(choices=)` offers choices from the run's data.** The docs, the `Ask` docstring, the add-node skill and notebook 06 write choices a node author knows as `Param(choices=[Choice(id, title)])` on the return type or a dataclass field (`escalate_to: Annotated[str, Param(choices=["ida", "bo"])] = SKIPPED` is a branch with choices), so `describe()` shows them before any run. `Ask(choices=)` is shown for choices that come in on an input or the context.
- **A JSON schema lists choices as its `enum`, at any depth.** `Param(choices=)` writes the `enum` of its type's JSON schema: the choices' ids, with `null` when the type takes `None`. That holds on a field, a list's item, a record's field, a table's column and a host model's field, in `describe()` and on a waiting call's question; `str` with `Param(choices=)` was `{"type": "string"}`. A question whose `Ask` narrows its choices has the schema of the type narrowed to them, so a narrowed `Literal` no longer lists every value. A bare id an `Ask` offers keeps the title the output's type gives it (`"ida"` stays `Ida B.`).
- **An edge into an input with choices or limits follows one rule, at any depth.** A value of the right class wired into an input whose type has a limit, choices (a `Literal`'s values or `Param(choices=)`) or another check, on the input or on a list's item, a record's field or a table's column, compiles, and a value the check refuses fails its call as `invalid_edge_value`. A `str` or a wider `Literal` into a `Literal` input was `type_mismatch`, and `list[str]` into `list[Annotated[str, Param(choices=)]]`, a record field or a table column with choices or `Param(le=)` ran with the value unchecked. A value the check converts reaches the node converted, at any depth as on the input: `list[str]` giving `["ABC"]` into `list[Annotated[str, AfterValidator(str.lower)]]` gives the node `["abc"]`, where it got `["ABC"]`, so such a graph may give a different result. A `Literal` source is refused when the input's own check takes none of its values, the check each value gets on its call: `Literal["x"]` into `Param(choices=["ida", "bo"])` is `type_mismatch` (it compiled and failed every call), while `Literal[1]` into float choices `[1.0]` and a `Literal` that may give `None` into choices on a type that takes `None` compile. A union of `Literal`s is read as one `Literal`, and a union lands when each member does, so `int | float` lands on `float` (it was `type_mismatch`). `type_mismatch`, `type_mismatch_series_given_one_value` too, names a type with choices by its values (`takes "ida" or "bo"`), where a `Literal` read as its values' class ("gives str, but takes str"); two types that read alike are named in full.
- **Breaking: a named value must read back from its own JSON** [provisional]. A `Literal`'s value and an enum's member, a type's named values, are its choices, and a saved run, an answer and a description hold each as its JSON. `@node` refuses one that doesn't read back from that JSON as itself, at any depth (on the field, a list's or a dict's or a `Series`'s item, a record's field, a table's column), but not inside a host's own model, whose fields are its own to check, with the new declaration problem `named_value_invalid`, whose message says what to write instead. Refused: a `Literal` of a plain enum's member, `bytes`, a tuple, a date or `nan` (take the enum, or the value's type, with `Param(choices=[...])`), which compiled and ran but couldn't be saved, described or asked; and an enum whose members don't read back, like one of tuples or `bytes` (`named_value_invalid_enum`), which ran but couldn't be saved. Still taken: text, numbers, yes or no, `None`, and the members of an `IntEnum`, a `StrEnum` or a `(str, Enum)`, which read back as themselves. `named_value_invalid_alike` refuses two of a `Literal`'s values, or two of an enum's members, written as the same JSON, like `inf` and `"Infinity"`; `Literal[1, True]` and `Literal["1", 1]` are two values each and read back as themselves.
- **An enum's members are its choices.** A field typed by an enum lists its members as `choices` in `describe()` and on a waiting call's question, as a `Literal` lists its values; `Param(choices=)` on it titles or limits them, and an `Ask` narrows them. A bare member's title is its value's text, `"red"` or `"1"` [provisional], where it was `str(member)`, like `Color.RED`. With `Param(choices=)` on an enum field, the JSON schema is `{"allOf": [{"$ref": "#/$defs/Color"}], "enum": [...]}`, the enum's schema in `$defs`, where it was the inline enum: a `$ref` that `Param` adds an `enum`, a `title` or a `description` beside goes in an `allOf`, since JSON Schema draft-07 ignores every key beside a `$ref` and a form reading it so would offer every member. A table column of an enum is a `$ref` into `$defs` too. `type_mismatch` names an enum input by its id, as before, a `Literal` or `Param(choices=)` input by its values, at most five and then how many more, and two types that read alike as "… with other choices or limits".
- **Compiling asks an input's argument check of a `Literal` source's values, the check the run gives each value.** `Type` owns the argument check (`argument_validator`, the type's schema with conductor's rules); compiling, the run, a saved run and a graph input all read through it. A validator that looks at a value's class gets the value itself, never its JSON. A `Param` choice `True` on a number is refused with "a yes or no is not a number", as an answer is. A `Literal` source lands only when the input's check takes one of its values and their classes fit the input, for every input: `Literal[1]` into `Literal[True]` and `Literal[1.0]` into `Literal[1, 2]` are `type_mismatch`, as they were before this release, though pydantic takes `1` as `True`.
- **Metadata on a union's member checks only that member.** In `Annotated[str, AfterValidator(f)] | None`, `f` checks the `str` and never sees `None`, which reaches the node; `f` was applied to the whole union, so a validator that refuses `None` refused it. A `Param` on a member is still the whole field's.
- **A yes or no is not a number in JSON either.** `Run.from_json` refuses a saved run with `true` where an `int` or `float` is saved (a graph input, an output, an item of a `Series`, a field of a record, a cell of a table) as `saved_run_unreadable`, with the reason "a yes or no is not a number; give a number"; it read `true` as `1`. A person's answer with `true` for a number, and an `Ask` choice of `True` for one, are refused with that reason too, where pydantic's "Input should be a valid integer" was given.
- **A list refused by an input whose type takes a list says why.** `Annotated[list[str], Param(min_length=1)]` given `[]` is `invalid_input` with the `min_length` reason ("List should have at least 1 item"), not `invalid_input_list_for_one_value`, which is kept for a list given to an input whose type takes no list; so is `int | list[int]` given `["x"]`, refused for its item.
- **A table answer and a caller's records have one reader.** Extra columns are dropped for every table from outside, and a missing column is refused. Their cells are read strictly from their JSON form, never from polars' dtype, so a bool among ints or a number among texts is `invalid_input`. A value that is neither a frame nor a list of records is refused as `invalid_answer` or `invalid_input` with the reason "a table is a frame or a list of records, one object per row".
- **Breaking: a caller's graph inputs are read strictly, as a person's answers are** [provisional]. `graph.run`, `graph.with_inputs` and `run.with_inputs` read each value by its input's type as its JSON text would be read, so a value given from Python, the same value sent over HTTP and the value a saved run reads back read alike. Refused as `invalid_input`, where they were converted: `"5"` and `5.0` for an `int`, `"5.5"` for a `float`, `"yes"` and `1` for a `bool`, a `datetime` for a `date`, a number for a `datetime`, and the same inside a list, a record, a model, a dataclass or a table's records (`{"amount": "1"}`). Still taken: a date or datetime as its ISO text, a model or dataclass as itself or as a mapping of its fields, a host's `@host_type` class as itself (written by its own schema and read back) or as its JSON, `None` for an optional, a tuple for a list, a list for a `Series`, and a frame for a table, read as it is, so a column the table doesn't name is dropped as before. What a type doesn't name is its JSON data, as a saved run reads it back: an `Any` input, at any depth, gets a tuple as a list, a `Decimal` and `bytes` as text, a date or datetime as its ISO text, and a dataclass, a model or a host's object as its mapping (each was passed on as it was). Newly refused, as `invalid_input` with the refusal's code in `ctx.errors`: a value JSON can't write, such as an object held in memory, `object()` for `Any` or bytes that aren't text (`no_json_form`; it ran, and the run failed to save as `run_unwritable`), which goes in the run's context instead; a frame anywhere but where the type is a table, for `Any` and inside a list or a dict too (`no_json_form`; it ran, and failed to save); and a set anywhere in a value, a dataclass's field included (`set_has_no_order`; a `list[...]`, `set[...]` or `Any` input took one, in whatever order it had; a set-typed input takes a list). Answers: each question's value is read by the same reader, so a frame answers a table question as a caller's frame is read, keeping only its columns (a column a saved run can't keep was refused even where the table didn't name it); a set in an answer is `invalid_answer` (a `Series` answer took one); an answer of a host's class with its own schema, given as itself, is written by that schema (it was refused). `Run.queued_as_json` still writes a frame as its columns. A frame for a `Series` of records stays refused as an input and as an answer, as before; only a node's return takes one. A pydantic error's message worded for JSON input, such as "Input should be a valid array", is kept in `ctx.reason` and `ctx.errors` (it was replaced by "The value did not pass its type's own check."). No code and no `ctx` key changes.
- **Breaking: a graph record's constants and graph input defaults are read strictly, by the reader a caller's graph inputs are read by** [provisional]. When a graph is compiled, each constant and default is read by its input's type as its JSON text would be, so a value in a record, the same value given from Python and the same value sent over HTTP read alike. Refused as `invalid_value`, where they were converted: `"5"` and `5.0` for an `int`, `"5.5"` for a `float`, `"yes"` and `1` for a `bool`, a number for a `datetime`, and the same inside a list, a record or a model (`{"amount": "1"}`); `make_graph` turned a `datetime` given for a `date` into its ISO text, which is refused too. A saved graph record whose constant or default passed only the lax reading now has a compile problem. Still taken: a date or datetime as its ISO text, a model or dataclass as a mapping of its fields, `None` for an optional, a table as a list of records. A host's own check that raises something other than pydantic's error, such as `float(None)`'s `TypeError`, is a refusal at every door (`value_error` in `ctx.errors`): a constant is `invalid_value` where `Graph.compile` raised, a caller's input `invalid_input` where `graph.run` raised, and a person's answer `invalid_answer` ("The value did not pass its type's own check"), which the worker reported as `raised_outside_a_node`. `make_graph` and an upgrade's rewrite write a value as a caller's value for an `Any` input is written: a host's object with its own schema is written by that schema (it was refused), and a value JSON can't write is `invalid_value` with `no_json_form` in `ctx.errors` and the reason "the value can't be written as JSON" (it had no `ctx.errors`, and the reason "it can't be written as JSON, and a graph record is saved as JSON"); a frame's reason says to give its rows as a list of records. An object held in memory whose class writes it as JSON, but can't read it back, is written by `make_graph` and refused when the graph is compiled. A node's own default in its signature is code, not JSON, and is read as before. `invalid_value` always has `ctx.errors`, and its `ctx.reason` keeps pydantic's wording for JSON input ("Input should be a valid array").
- **Breaking: a number that enters a graph is finite** [provisional]. `inf`, `-inf` and `nan` given by a caller, in an answer, as a graph record's constant or default, or written by `make_graph` or an upgrade's rewrite are refused, at any depth and whatever the type (`Any`, a `TypedDict`, a host's model or dataclass, a table's records), with `finite_number` (`ctx.errors` of `invalid_input` and `invalid_value`; the message of `invalid_answer`), where each was taken. A frame given for a table with a `NaN` or an infinite cell in a float column is refused the same way. The texts `"Infinity"`, `"-Infinity"` and `"NaN"` are text, so a `float` refuses them in a record (they were read as the float), and `make_graph(constants={...: inf})` is refused at the call (it wrote `"Infinity"`). An `Ask`'s choices are read as an answer is, so `Ask(choices=[1.5, inf])` is `invalid_ask`, as `Ask(default=inf)` already was (it offered a choice no one could pick). What a node computes is not held to it: a return and an edge's value may hold `inf` or `nan`, and a saved run writes them as text and reads them back as before.
- **A dict's number keys are read from their text** [provisional]. A `dict[int, str]` input, constant or answer took no value: `{1: "a"}` and `{"1": "a"}` were both `invalid_input` `int_type`, since conductor's guard that a yes or no is not a number wrapped the key's schema, and pydantic then no longer read the key from its JSON text. The guard leaves a dict's key alone, so both are `{1: "a"}`; `{True: "a"}` is still refused (`int_parsing`). A key that isn't finite (`{"Infinity": "a"}`, `{"NaN": "a"}`, `{"inf": "a"}`, `{inf: "a"}` for `dict[float, str]`) is refused with `finite_number` where the value enters.
- **Breaking: a node's default and a `Param`'s choice are finite** [provisional]. `def f(limit: float = math.inf)`, a common way to say "no cap", ran wherever the input was left unwired, and is now refused at `@node`: write `limit: float | None = None` and read `None` as no cap. Such a default wrote `"default": "Infinity"` into a record that `make_graph(f, inputs=...)` made, which then failed to compile, and `Param(choices=[1.0, inf])` offered a choice no one could give (it left the schema with no `enum`). `@node` refuses a default holding `inf` or `nan`, at any depth, as `default_invalid` with pydantic's `finite_number` reason, and such a choice, or a `Literal`'s value, as `choices_invalid_not_a_value`.
- **A dict's keys are text** [provisional]. A dict or mapping whose key type isn't `str`, a `str` enum, a `Literal` of texts or `Any`, such as `dict[int, str]`, on a field at any depth conductor reads, is refused at `@node` with the new declaration code `dict_key_not_text`. Every value that enters is read from its JSON form, where keys are text, so `{1: "a"}` and `{"1": "a"}` were both `invalid_input` and no value could be given.
- **`annotated_types.Timezone` is refused** [provisional]. Pydantic builds `Annotated[datetime, Timezone(...)]` and never checks it, so a naive datetime passed. `@node` refuses it as `param_limit_not_supported`, on a field and at any depth conductor reads; take `pydantic.AwareDatetime` or `NaiveDatetime`.
- **A dataclass answers a node with branches** [provisional]. A dataclass of outputs given as an answer, to `run.with_answers` or `Run.queued_as_json`, was refused as `invalid_answer` ("the value can't be written as JSON") whenever the node had a branch, since the branch not chosen is `SKIPPED` in it. Such a field is now left out, as a branch left out of a mapping is, so the branch is skipped.
- **A `Table`'s row type is a `TypedDict` or a bare row** [provisional]. `Table[R]` with a dataclass, a model, `int` or `dict[str, int]` as `R` named no columns, so it read as a bare table: a frame kept every column, one `R` doesn't name included, and a saved run wrote it. A node that declares one, on a field, in a union, a list or a `TypedDict`'s field, is now refused with the new declaration code `table_row_not_a_typed_dict`. `Table`, `Table[Any]`, `Table[dict]`, `Table[dict[str, Any]]` and a type variable are taken as before.
- **Each error in `ctx.errors` keeps its `ctx`** [provisional]. An error whose message is pydantic's or conductor's own, such as `greater_than_equal`, keeps the keys of its `ctx` that its message reads (`{"ge": 3}`; a host's extra key is dropped), so a host can write its own message by the code; one whose message is replaced, such as a host validator's, has none. Its `loc` is, as before, the list indexes and record fields inside the value, stopping at a dict's key.
- **The runs routes' 422 quotes nothing the request sent** [provisional]. A body or a header FastAPI refuses, and a `POST /runs` or `POST /estimate` body of the wrong shape, was answered with pydantic's `input` in each error: the whole body for a missing `graph`, the refused `max_items` of `/resume`. Every route of `run_router` now leaves `input` out, and keeps `type`, `loc`, `msg` and `ctx`.
- **An annotated-types limit is a `Param` limit** [provisional]. `Gt`, `Ge`, `Lt`, `Le`, `MultipleOf`, `MinLen` and `MaxLen`, and the groups `Interval` and `Len`, are read as the `Param` limit of the same name, so `Annotated[int, Ge(0)]` and `Annotated[int, Param(ge=0)]` are one type (one graph input may feed both), and a limit pydantic has no rule for on its type, such as `MaxLen(3)` on an `int`, is `param_limit_not_supported` at `@node`, where the graph compiled and every value was refused. A length limit on a `Table`, `Param(min_length=, max_length=)` as well as `MinLen` and `MaxLen`, counts its rows (`too_short`, `too_long`); `Param`'s was refused. On a `Series` everything in its `Annotated` is its items', a validator too: `Annotated[Series[int], AfterValidator(f)]` checks each item, where it was dropped, and `Annotated[Series[int], MinLen(1)]` is refused, where pydantic ignored it and `[]` ran. A limit on one member of a union stays that member's: `Annotated[int, Ge(0)] | str` limits only the `int`, and so does `Annotated[int, Param(ge=0)] | str`, which was refused. That code's message ends "Remove it from the hint" (was "from the Param").
- **A host's pydantic model writes `inf` and `nan` as conductor does** [provisional]. A model writes its own floats by its own config, which writes them as `null` by default, so `HostModel(x=inf)` given as an input was refused as `float_type`, a `float | None` field became `None` and the run went on, `make_graph` wrote `{"x": null}` into the record, and a waiting call's `inputs` showed `null`. Now such a value is refused with `finite_number` where it enters, as `{"x": inf}` is, and a waiting call shows `"Infinity"`.
- **An input can take only a constant: `Param(constant_only=True)`**, a setting fixed when the graph is drawn (1.x's `disable_handle=True` on its widget). Every input a `compute_...` hook reads is one too, and the one rule holds for both: its `FieldDescription` has `constant_only: true`, `make_graph` by itself never wires it or makes it a graph input (`not_a_chain_constant_only`, `ctx.inputs`), a record that gives it `sources` or a `graph_input` is `input_takes_only_a_constant` (`ctx.given`) when compiled, and one with no constant and no default is `input_has_no_value_constant_only`. An edge into an input a hook reads, which was taken while the hook read the input's default, is that problem too. On an output it is `constant_only_on_output`.
- **A run's results are what a node reading them would be given.** For an output made per item, `run.results`, `run.outputs` and `outputs_to_json` are `null` where the Series was never returned because of a `None` outside the loop, or a skipped branch there (they were `[]`, the same as for a Series returned with no items), and `null` where every item is skipped from outside (they were a list of `null`s). A `None` outside still shows a `None` in each place. The same nodes inside a subgraph show the same.
- **An output made inside two loops is shown as a list of lists.** `run.results`, `run.outputs` and `outputs_to_json` show the output of a node called per item of a list that is itself returned per item as one list per outer item, `null` for an outer item whose inner list was never returned (it was one flat list over every item): three documents split into lines, the second empty, show `[["A", "B"], null, ["C"]]`. The same nodes inside a subgraph show the same.
- **A node outside a subgraph reads its outputs as it would read the same nodes drawn in its own graph.** For the subgraph `documents -> lines -> shout`, a `Series[str]` input wired to `shout.result` is called once per document with that document's lines (it was called once, with one flat list over every document), and an input of one `str` once per line; a node outside and `run.results` see the same list per document. A loop outside is named by the list inside that is looped over (`describe()` gives `per_item_of` as `e/documents.result`; it named the offered output), so outputs returned in the same loop inside are paired item by item: one item of each of two of them reaches one call (it was `series_not_nested`), also when one of them is the list looped over itself (`e.parts`, and `e.loud` returned once per item of it), and a `Series` input beside an input of one value wired to the same per-item output is `type_mismatch_series_given_one_value`, as drawn flat (it received every item). The output is described with `is_series` and one `Series` per loop (`type` `Series[str]` for a list per document) and lists its `loops`; every field description gains `loops`, empty but here. A `Series` returned once per document is `Series[Series[T]]` too (it was a `Series` of whole lists). `series_not_nested_inside_subgraph`, for a list inside a subgraph, tells the author what they can do.
- **`json-parse` fails with `not_json` on a number of more than 4,300 digits.** Python refuses it with a `ValueError`, which escaped the node as it was.
- **A saved run keeps what a waiting call's `Ask` said, and nothing of a question once it is answered.** A waiting call's entry under `questions` is its `prompt` and, each left out when the `Ask` set none, `"choices": {"result": [...]}` and `"defaults": {"result": "bo"}` by output name. An answered call is `"done"` in `states`, as a call whose function returned is, with its outputs and no question. What a waiting call received is not saved: `Waiting.inputs` is read again from the outcomes the call reads.
- **`Run.from_json` takes the JSON text `run.to_json()` wrote**, as `str` or `bytes`. It took a mapping too; a host that holds one writes `Run.from_json(json.dumps(saved), graph)`.
- **`run.results` reads a run in any state and never raises.** A run that failed because an edge named a field its record lacks (`invalid_edge_value`) shows `None` where that field was read; the failure is the run's cause.
- **A run's item ceiling is raised in one way: `run.with_max_items(n)`.** It gives the same run under the ceiling `n`, with nothing started, so `run.with_max_items(100).resume()` goes on with a run that stopped at its ceiling. `run.resume` and `run.stream` take no `max_items`; `graph.run`, `graph.with_inputs`, `run.with_inputs` and `run.again` set it where a run is made. The router's `POST /runs/{id}/resume` stores the run with the raised ceiling as it claims it, so a run whose worker is killed before its first checkpoint goes on under the raised ceiling. Its body's `max_items` is an integer (`true` and `2.5` are a 422 `int_type`), and conductor decides that it is at least 1: `0` is a 422 with conductor's `invalid_limit`.
- **A node's code fingerprint is read from its source, so a new Python version keeps it.** The `code` a saved run records, and the fingerprint in a waiting id, hash the function's source as Python parses it, without its name, decorators, comments or docstrings, rather than its bytecode, which every Python version changes. A function whose source `inspect.getsource` can't read is hashed by its bytecode: one made by `exec` with no file, one typed into `python -c` or a REPL that keeps no source, or one from a package installed without its `.py` files. So is a lambda, and a function whose lines at its place aren't Python, hold no `def`, or whose first `def` has another name than the one its code was compiled with, as when `exec` compiles a text under a real file's name; a factory that sets `__name__` after the `def` keeps its source. `@node` works the code out when it declares the function, so a run saved later can't fail on it.
- **A node inside a subgraph is called again when its subgraph node moves into a loop.** A node's code fingerprint covers the loops it runs in, and for a node inside a subgraph these include the loops around the subgraph node, as they would if the node were drawn in the graph itself. The same subgraph used once and then inside a loop reused the call made once, even though it was then called per item. So the `code` a saved run records, and the fingerprint in a waiting id, change for every node inside a subgraph whose subgraph node is inside a loop, at any level: a run saved by an earlier build reuses none of that node's outcomes, and an answer to a waiting id it gave is `answer_stale`. The loops around the subgraph node and those inside its graph are hashed apart, so a loop moved from around a subgraph node to inside it calls the node again too. A node inside a subgraph that is in no loop keeps its fingerprint.
- **The `item_limit` problem's `loc` names the call that returned the series**, as a failure of that call would: `("lines", 1)` for the second call of `lines`, and `("inner", 1, "chars", 2)` inside a subgraph.
- **A subgraph costs what its nodes cost.** Its node has no outputs of its own: what reads one of its outputs reads the inner node that returns it, and `results`, the events and `none_input_calls` say of it what they say of the inner nodes. Two `Series` inputs wired to two of its outputs returned per item of one Series inside are paired item by item, as they are drawn flat. A save's `complete` lists function nodes only, the subgraph's node says no `node_progress`, and one with no nodes inside says no events.
- **A `Series` input wired to one value is refused, flat and inside a subgraph alike.** A node that runs per document, because another input collects the document's lines, and whose `Series[str]` input is wired to the document's summary, is `type_mismatch_series_given_one_value`; declare the input as one `str`. It received every document's summaries on each call, and the same nodes inside a subgraph received one.
- **`run.none_input_calls` is a list of `NoneInputCall` records** (`loc`, `input`), exported from `conductor`, not a list of dicts. It is in item order on a live run, in a checkpoint and after `Run.from_json`.
- **A graph with problems is still drawn.** `graph.describe()` gives a node with a problem its fields and where it runs (`found` true), and the nodes that read it too; an input whose edge is refused has the mode `missing`, which `InputSource.mode` gains. `found` is false only where the node's definition isn't found, and such a node has no fields.
- **A default a generic node declares is read by the type its edges bind.** `first(one: T, other: T = 0)` wired from a `str` gives the graph the problem `invalid_value_bound_default` at `("first", "other")`; `Graph.compile` raised `NodeDeclarationError` for it. A default of `None` is read the same way, so `first(one: T, other: T = None)` wired from a `str` has the same problem, and so has a generic dataclass of outputs whose field defaults to `None`. An input an edge or a constant gives its value never uses its default, so its default is not read.
- **A whole table or list wired where one value is taken says to make a series.** `type_mismatch_whole_table_or_list` and `unknown_output_field_of_whole_table_or_list` say it, as `invalid_input_list_for_one_value` does for a list given to an input of one value: put a node between that returns the rows or items as a `Series`.
- **A row type's fields are all required.** A `Table[R]` or a `Series[R]` of records whose `R` has a `NotRequired` field, or is declared `total=False`, is refused when the node is declared (`row_field_optional`, a new declaration code); a node returning such a `Series` failed its run with pydantic's own schema error. Declare the field `X | None`.
- **Faults say more.** `invalid_edge_value` carries `ctx.record_field` when an edge names a field its record lacks. A hook that returns no mapping is `compute_hook_returned_invalid` with `error: not_a_mapping`, and one that gives a field a name a graph cannot have is `compute_hook_returned_invalid` at that name with `error: not_a_name`; an `AttributeError` in a hook's hint is the host's own bug and raises out of `Graph.compile`. `table-collect` fails with `columns_not_aligned` (`column`, `records`, `values`) when a column doesn't hold one value per record.
- **`SKIPPED` returned is a result.** A function that returns `SKIPPED` was called: `again` reuses it and a save writes its call as `done`. A node's `CallCount` counts its skipped calls in `skipped`: a call downstream of a skipped branch, or one whose every output is `SKIPPED`; a node with no output that returns `SKIPPED` is not counted. A node was skipped when `finished > 0` and `skipped` equals `finished`.
- **A run handed back at shutdown has no `cause`.** The row's `cause` is null and its status `running`: the run did not end early. The run saved in the row is its last checkpoint, which reads back as its outcomes say: the router hands the run back with `run.stream(hand_back=)`, where it cancelled the run and stored a run that said `cancelled`.
- **A person can decline a question, and that ends the run for good** (provisional). `run.with_decline(waiting,
  reason=None)` takes a waiting call as `run.with_answers` does and returns a new run, `declined`, a status of its own,
  with the new problem `answer_declined` as its `cause`, whose `ctx.reason` holds the host's reason. Every call
  made before keeps its outcome, so the run's results show what ran; nothing waits any more, `resume()` and
  `stream()` call no node, and a later `with_answers` or `with_decline` is refused as `not_waiting` (`ctx.status`
  `declined`), as on a complete run. `run.again()` starts a new run that reuses what ran and asks again. A saved
  run reads back `declined`. An id is refused as `with_answers` refuses it (`not_waiting`, `answer_stale`). A reason is
  checked as its JSON text would be: `bytes` and a `str` enum member are taken as their text, and one that isn't
  text JSON can write, such as `1.5` or a `str` with a lone surrogate, is refused as `invalid_decline_reason`
  (`ctx.got`, its class).
- **The runs router takes a decline** (provisional). `POST /runs/{id}/declines` with `{"waiting_id", "reason"}`, and
  `Worker.decline(run_id, waiting_id, *, reason=None, allows=)`, decline one question. The body's `reason` is a
  `str` or left out, and `Worker.decline` takes a reason as `run.with_decline` does. A run that isn't running is given
  the decline at once and written `declined` with no resume call, so a worker that never served takes it; its
  events get one `run_end` that says so. A
  decline it refuses is a 422 (`not_waiting`, `answer_stale`), and nothing is written. On a running run the
  decline is queued as an answer is, as one entry of the row's `queued`, and given in order when the resume call
  stops: the first of two replies to one question wins, and a run that takes a decline ends `declined`, so that
  call's last `run_end` says `declined`, though its stream ended `waiting`, or `cancelled` with a cancel asked too.
  A declined run is never resumed: `POST /runs/{id}/resume` is `run_not_resumable` (409), and an answer or a
  decline is `not_waiting` (422). A takeover and a run handed back give a queued decline when the resume call
  stops, as any queued reply; to stop the calls at once too, cancel as well. `run_not_resumable`'s message lists the
  statuses a resume goes on from. The body is `{"waiting_id", "reason"}` alone: a body with `"kind": "decline"`,
  which the pre-releases took, is a 422 `extra_forbidden`.
- **A resume call that raises gives the run its queue at once** (provisional). The run as the row has it, its last
  checkpoint, takes the replies queued while the call ran, as a run whose call stops does, so a row that isn't
  running never holds a queue: a decline queued then ends the run `declined` at once, and no later answer or resume
  calls a node. An answer to a question the run asked after its last checkpoint is `not_waiting` in `refused`, and
  the question waits again on the next resume (it used to stay queued and be taken after that resume). When the
  row's run can't be loaded, each queued reply is refused with the call's cause. A held run that stops on a fault
  it can't fail the run with, such as a row the worker can't read, is logged with the run's id.
- **The runs router takes an answer whenever the run does.** `POST /runs/{id}/answers` on a run that `failed`, `stopped`, `timed_out` or was `cancelled` gives the answers to `run.with_answers` and resumes the run, so one item that keeps failing doesn't make the questions beside it unanswerable; what the run refuses is a 422 with its problems (`invalid_answer`, and `not_waiting`, also for an answer to a `complete` run). Answers queued while the run is running are given to the run it stopped as, whatever its status; a resume call that raised gives them to the run's last checkpoint as it fails. The code `run_not_waiting` is gone.
- **The runs router cancels a waiting run.** `POST /runs/{id}/cancel` on a run that is `waiting` resumes it with the cancel already set, so it ends `cancelled` before any node call, as `run.stream(cancel=...)` does, and answers with the run, `running`: its questions still wait, and an answer or a resume carries it on. A host's own scheduler sends it to end a run whose question nobody answers. Like any resume call, it can end the run `failed` when its context or a node can't be had. A cancel that meets a run another call changed first goes by the run as it now is, and a run that is complete or already ended early is answered as it is.
- **A cancel sent to any worker ends the run `cancelled`, even when the run completes before its worker hears it.** The worker that holds a run hears a cancel sent to another worker at its next checkpoint or lease renewal, and a run that completed before then ended `complete`, though `POST /runs/{id}/cancel` had answered `running`. Now a run that completes with a cancel asked ends `cancelled`, with conductor's `run_cancelled` as its cause and its `run_end`, whichever worker took the cancel, so a cancel that came during the run's last call is no longer dropped either. Every call's outcome is kept and its outputs aren't shown, as no cancelled run's are; `POST /runs/{id}/resume` makes no call and completes it.
- **How a run stopped is stored once, when the router has decided what it is.** A run handed back at shutdown stores no `run_end` (it read `cancelled`, for a run nobody cancelled that another worker then finished), so a follower's stream goes on into the next worker's events. A resume that raised outside a node reads `failed` with a cause, `raised_outside_a_node` (`ctx.error` the exception's class name), in `GET /runs/{id}` and on its `run_end` (the cause was null). The worker's own codes are a catalogue, `conductor_providers.fastapi.SERVING_PROBLEMS`, a function per code as conductor's `PROBLEMS` is: `no_such_run`, `run_not_resumable` and `raised_outside_a_node`. The metadata for a new run is checked against the worker's `metadata_type` before anything is stored, and written by alias, so it always reads back; a value of another type is the host's own bug and raises pydantic's `ValidationError`, a 500.
- **Every refusal of the runs router is a list of problems.** A 404 and a 409 are `{"detail": [{type, loc, msg, ctx}]}` as a 422 is (they were an English sentence), with the worker's own codes: `no_such_run` (404) and `run_not_resumable` (409, `ctx.status`). An answer to a complete run is conductor's `not_waiting` (422). Answers sent while a run is running are all queued, in the order they came, and a second answer to a question the first answered is refused as `not_waiting` when the run stops; of two calls that act on one run at once, the one whose write loses reads the run again and acts on it as it is then. The worker raises every refusal as conductor's `InvalidCallError`, and the router picks the status from the first problem's code. The OpenAPI schema declares both as FastAPI's own `HTTPValidationError`, which a problem fits.
- **The runs router is mounted over a `Worker`.** `run_router(worker, metadata=, run_filter=)`: the worker holds the store and the settings (`Worker(store, nodes_for=, context=, metadata_type=, max_items=, lease=, checkpoint_every=, resume_timeout=, poll=)`, each setting with its default there), and the router only the routes. The worker is generic in `M`, the type of a run's metadata (`dict[str, str]` unless `metadata_type` names a pydantic model), and `X`, the registry's `context_type`, with one constructor and no overloads: `nodes_for=` is a function of the metadata that returns the registry its graphs compile on (`lambda metadata: registry` for one registry), and `context=` a function of the metadata and a stop event; the type checker holds both to `M` and `X`. A registry that needs a context with no `context=` given passes the type checker, and conductor refuses the run (`context_missing`).
- **`run_router` requires `metadata` and `run_filter`.** `metadata` is a FastAPI dependency that gives the metadata for a request: a run the request starts keeps it, written once as JSON, and every resume builds the run's registry and context from it; on `/nodes`, `/compile`, `/estimate` and `again` it picks the request's own nodes. `run_filter` is a FastAPI dependency that gives, for the request, a test of a run's metadata, `Callable[[M], bool]`, asked on every route under `/runs/{run_id}` on the row the route reads anyway, once, passed to the worker method as `allows=`: a run it turns away is a 404 `no_such_run`, as an unknown id is. There is no default for either, since no rule is safe for every host; a host with no logins gives `metadata` a function that returns `{}` and `run_filter` one that returns `lambda metadata: True`.
- **[provisional] `RunStore.update` takes an optional `event`.** `update(run_id, rev, changes, event=None)` appends the event in the same statement or transaction, numbered as `add_event` numbers one, only when the compare-and-set lands: both land or neither does, whoever the row's `owner` is. The worker writes every `run_end` this way, with the row that says how the run stopped, so a follower that reads the row stopped has the event, and a failed write leaves neither. A host's store must add the parameter, and one whose `update` takes no `event` is refused with a `TypeError` that names it, when a `Worker` or `check_run_store` is given it (every write of a run's end would raise, and the worker would read each as an outage); the SQLite store on `docs/serving-runs.md` writes both in one transaction, and `check_run_store` reports a store that drops the event or appends it for a write that didn't land.
- **A stored run's `lease_until` is null when no worker needs to resume it,** and a time for a run a worker holds or handed back, so a store's `expired(now)` is `lease_until < now` with no status clause, and a store that forgets the status can't re-run finished runs. `RunStore.add_event` returns whether it appended, not a number.
- **A stored run is ten columns.** `StoredRun` and the table a host writes for it keep `id`, `graph`, `metadata`, `run`, `status`, `rev`, `owner` and `lease_until`, and two texts hold what the router alone reads: `state` (why the run ended early, the replies queued and refused, a cancel that was asked; small whatever the size of the run) and `outcome` (what the last resume left for a reader: what waits and the outputs). The columns `waiting`, `outputs`, `cancel`, `pending`, `refused` and `cause` are gone, and `RunChanges` is the six columns that change (`run`, `status`, `state`, `outcome`, `owner`, `lease_until`). `status` is written with `state`, from it, so a host's query on `status` never disagrees with what `GET /runs/{id}` says. The store on `docs/serving-runs.md` is the six methods with no per-column conversion. The `state` keeps each reply sent while the run runs as one entry, an answer or a decline to one waiting id, tagged by its `kind`: `{"kind": "answer", "waiting_id", "answer"}` or `{"kind": "decline", "waiting_id", "reason"}`. A row written by a 2.0 pre-release, whose `queued` held a call's answers together and the declines body, or an entry with a `declined` flag, can't be read.
- **A run whose resume raised lists what waits in the run as stored.** `GET /runs/{id}` on a run that reads `failed` because its worker raised lists as `waiting` what waits in its last checkpoint, the run an answer is given to (it listed what waited before the resume, some of it answered since); a run that could not be loaded lists nothing. A raise from the `RunStore` fails no run: the resume is given up and a worker goes on from the last checkpoint once the lease runs out.
- **A store that fails for one statement costs the runs router one try.** A sweep for runs whose lease ran out, a take-over of one of them and a renewal of a lease are each logged with their traceback when they raise, and the next one runs (a sweep that raised once took over nothing for the worker's life, and a renewal that raised once let another worker take a healthy run and make its calls again).
- **A worker that never served says how to start it.** The `RuntimeError` from a route of an app that never started (FastAPI's `TestClient` used without `with`) says to enter `worker.serving()`, and that `with TestClient(app) as client:` does it, where it spoke of a sweep.
- **A run answered while its last resume call still writes its end is handed back at shutdown.** A worker forgot the run it took for an answer when the store returned the write that ended the run's previous resume call after that answer had begun the next one. A cancel on that worker was then heard only at the next renewal of the lease, and at shutdown the run was left running and owned until its lease ran out, so another worker made its calls since the last checkpoint again.
- **An answer or a decline's reason JSON can't write is the same 422 whether the run runs or not.** On a running run, `POST /runs/{id}/answers` and `/declines` (and `Worker.answer` and `Worker.decline`) refused a `str` with a lone surrogate with a 500, where a run that isn't running refuses it as `invalid_answer`; now both give `run.with_answers`'s and `run.with_decline`'s problem. Answers queued while a run runs are kept as their JSON data, so an answer pydantic can't write itself, such as a `polars.DataFrame` for a `Table` question, is queued and taken when the run stops (it was a 500). The step is the new, provisional `Run.queued_as_json(answers, declines=)`, for any host that queues answers or declines to give a run later: it returns `(answers, reasons)` as the JSON data the run will be given, refusing what JSON can't write. `run.with_answers` and `run.with_decline` take every value through the same step, so the problems are the same. That makes `run.with_answers` write the whole answer as JSON before it reads it: a dataclass or model instance now answers a node of several questions as the mapping of its fields (it was refused as not a mapping). A value JSON can't write is refused as before: for a mapping answer, at each question whose value it is (`ctx.question`), and for a bare value, for the whole answer.
- **The worker behind the runs router is public, for a host with routes of its own.** `conductor_providers.fastapi.Worker` begins, reads, answers, resumes, cancels and follows runs and takes over those whose lease ran out, with no HTTP in it; `run_router` is a thin layer of routes over one, and a host that admits runs its own way calls it the same way, inside `worker.serving()`. A run's metadata is stored once with it, and every resume builds the run's registry and context from it; a run's view gives it back as its `metadata`. Every method that acts on a run takes its id and the request's filter (`allows=`), reads the run once itself, and returns the run as it wrote it, a `RunView`. `begin(graph, inputs, metadata, previous=run_id)` starts a graph from the work of another run, as `graph.with_inputs(previous=)` does, and, once the graph and the inputs are checked, refuses with conductor's own problems, as `Run.from_json` gives them for that run's saved form (its graph's problems, `saved_run_of_another_graph` or `saved_run_unreadable`, each with its `loc` and `ctx`), when that run can't be loaded on the new run's nodes; a worker that has never entered `serving()` raises `RuntimeError` and writes nothing from a call that would make a run running, and one that has served and left hands such a run back at once for another worker; `again(run_id, inputs, metadata)` is `run.with_inputs`. Exported beside it: `RunView`, the run as a reader sees it, its `metadata` the JSON object the run keeps (`id`, `metadata`, `status`, `cause`, `waiting`, `outputs`, `progress`, `refused`), which `worker.read` gives and every route under `/runs` answers with: `POST /runs`, `/again`, `/answers`, `/resume` and `/cancel` answer with the run as written (where they answered `{"id"}`, `{"answers": "started" | "queued"}` and `{"cancelled": bool}`). The answers a running run has queued and a cancel asked of it are not shown. `RunView` has no type parameter, so it is one OpenAPI component, `RunView`, whatever the metadata type, and a host's own route that answers with it shares it. The worker refuses with conductor's `InvalidCallError`, as conductor does, with the router's problems.
- **`Worker.events` is awaited, then iterated.** `async for number, body in await worker.events(run_id, allows=...)`: awaiting it reads the run and checks it against the filter, so a run that isn't there, or that the request may not see, is refused as `no_such_run` before any event; it was an async generator iterated directly, which raised only once iteration started.
- **[provisional] `Worker.metadata_of` is gone.** It was the one way to read a run by id with no run filter, against 2.0's pre-releases. A run's metadata is the `metadata` of the view `worker.read(run_id, allows=...)` gives, which a host validates by its own type.
- **A node can stop a slow call when its run is cancelled under the worker.** The worker's `context` function is called with the run's metadata and a `threading.Event`, `context(metadata, stop)`, new for each resume call and set when that call stops early: the run was cancelled (on this worker, or on another and heard at the next renewal of the lease), another worker took it over, or the store raised and the worker gave the run up. It is not set when the worker shuts down. A host puts it in its context, and a node that streams checks `ctx.stop.is_set()` between chunks and raises; it never sets it. `run.stream(cancel=)` takes a `threading.Event` as well as an `asyncio.Event`; the worker's stop event is the one it passes there. A call that fails once its run was cancelled, by `run.stream(cancel=)` or the worker, ends the run `cancelled`, not `failed`, and a resume makes the call again; a call that fails after the `timeout` passed or while the run is handed back still ends it `failed`.
- **A host's run store is tested with one call.** `await conductor_providers.fastapi.testing.check_run_store(store)` writes runs and events through a `RunStore`'s six methods and returns every way the store differs from what the worker relies on, as plain messages that each start with the method at fault, such as `"update: wrote at rev 1 while the run was at rev 2; write only while the run is at the rev given"`: a run not kept as written (a JSON column as the same JSON value, so a `jsonb` column passes), a run found for an id that has none, an update written at an old `rev` or two written at once, `expired` giving the wrong runs, `add_event` refusing the owner or appending for another worker, `events` out of order, and a method that raises where it should answer. `RunStore`'s docstrings say that `get` reads what another worker just wrote, that a run's first event is numbered `0`, and that a method raises only when the database fails.
- **The runs router's schema reads as a generated client needs it.** Its operations are `list_nodes`, `compile_graph`, `start_run`, `estimate_run`, `read_run`, `follow_run`, `answer_run`, `resume_run`, `run_again` and `cancel_run`, each with a description; the body of a resume is the component `Resume`; the two free-form bodies say what they contain. The bodies that carry a graph (`/compile`, `/runs`, `/estimate`) are read as text, once, by conductor's `read_json`, and their schema is declared through `openapi_extra` with every `$ref` written in, so every reference in the document resolves; a fault in a graph record is conductor's `invalid_record`, and a fault in the rest of the body (inputs that aren't an object) is FastAPI's own (`dict_type` at `["body", "inputs"]`). `POST /runs` and `/estimate` take `"previous": run_id`, which the run filter checks as it checks a route under `/runs/{run_id}`. `published=` and its routes `POST /graphs/{name}/runs` and `/graphs/{name}/estimate` are gone: a host offers a saved graph with a route of its own that calls `worker.begin`. `RunView` is exported with the worker.
- **A kind of problem is declared by one function.** `@ProblemCatalogue().problem` over a function makes a `ProblemKind`: the function's name is the code, one function per code, a keyword-only parameter is a `ctx` key (one that defaults to `None` may be left out), a positional-only parameter (before the `/`) is a part of `loc` and a `ctx` key, so it cannot be given by name and the two always agree, `*loc` is the parts of `loc` the message does not use, and the function returns the message. Calling the kind makes the `Problem`, so a type checker refuses a misspelled code, a `ctx` key left out and a value of the wrong type where the problem is made; a signature outside those three rules is a `TypeError` where it is declared. A kind has its `code`, `section`, `sentence` (the English with a slot per key) and `ctx_type` (a `TypedDict`). A second kind with a code the catalogue has is a `TypeError` where it is declared. The reference (`python -m conductor.about`) prints each kind from its function: the sentence, and each `ctx` key with its type, whether it is always given and what it holds; `python -m conductor.about not-a-chain` prints one. `check_registry`'s problems are declared the same way and listed in the reference, and each has in `ctx` what its sentence is made from: `reason` (the sentences of the problems behind it), and `newest`, `status` or `named` where the sentence says them. `PROBLEMS` is the catalogue of conductor's core, every code it writes; a host reads every code as `PROBLEMS.codes`, and `conductor.PROBLEM_CODES` is gone. Every kind's `sentence` can be filled with `sentence.format(**problem.ctx)`. A problem whose sentence names its node or input has them in `ctx` too, as `node_id` and `input`, so a host fills its own sentence from `ctx` alone: the id is the node's in the graph it is in, as the sentence says it, where `loc` of a problem inside a subgraph has the subgraph node in front. A waiting id and a saved run's `code` digests differ from an earlier build's for a node that runs per item (the digest is made of the text of each loop the node is inside, its node's path and output: `e/split.result` for a loop over a list returned inside the subgraph node `e`, so a node in such a loop, or in a subgraph inside another, has a digest of its own once more, and an earlier save of it reuses nothing). Two older keys that used those words for something else are renamed: `edge_names_missing_node` names the node that isn't there as `ctx.missing`, and `unknown_output` and `invalid_edge_value` name the field of a record an edge reads as `ctx.record_field`.
- **A failure that is retried is logged by `resume()` too.** The exception a retry follows went to the `conductor` logger only when the run was streamed; `run.resume()` retried it without a word.
- **A retry's delay never overflows.** `Policy` refuses a `retry_delay` or an `attempt_timeout` above `threading.TIMEOUT_MAX` (about 292 years), where `resume()` raised `OverflowError` waiting for it, and a retry's delay stops doubling there: more than 1024 retries raised `OverflowError` even with `retry_delay=0`.
- **A timeout ends the run `failed` even after a cancel.** A call that ran past its `attempt_timeout` once a cancel was asked left the run `cancelled`, as if the cancel had stopped it; its thread runs on, so the run now ends `failed` with `timeout`, as it does without a cancel. The reference and `Policy` say that a timed-out attempt is abandoned, not killed, and that a later `resume()` may make the call again beside it.
- **A call that returned is let go at once.** The timer for a call's `attempt_timeout` stayed until the timeout ran out, holding the call and its arguments, so a run of many items with a long `attempt_timeout` kept every item's arguments alive; it is dropped when the call returns.
- **An `attempt_timeout` is kept while the run makes calls on its own thread.** A call on a thread that ran past its `attempt_timeout` while the scheduler was busy with a call on its own thread was recorded as a success, and the run could end `complete`; it now fails with `timeout`, judged by when its function returned. Between any two calls the scheduler now takes in the calls back from their threads and the retries and deadlines that are due (calls on its own thread that waited for a node's `concurrency` are made one per turn of its loop, where they ran back to back), so a node on threads gets its next call, and a retry is made when it is due, while nodes on the run's own thread are called one after another (they waited until nothing else was ready). A failure on a thread is so seen sooner, and the calls not started by then aren't started.
- **A reused series over the item limit starts its node.** A run that reused, from the previous run, a series with more items than its `max_items` streamed `node_complete` for its node with no `node_start` (and no `node_progress`), though only a node never called (skipped by a skipped branch, or looped over an empty series) completes unstarted, as the reference now says of both. The scheduler now tells the stream of every outcome it writes before it checks the item limit.
- **`check_registry` notices a frame swapped for a list.** A node that put a `polars.DataFrame` where its input had a list of records or another value is `changed_its_inputs`. Comparing the two raised a polars error, which the check reported as `example_run_failed`, as if the node had raised it. A frame is the same value only as a frame with equal cells, never as a list with its rows.
- **A run doesn't hang on Python 3.12 when a timer falls due as it starts to wait.** CPython 3.12's `SimpleQueue.get(timeout=)` never returns when its timeout runs out before its first look at an empty queue, so a run with a retry or an attempt deadline due within about a microsecond of the scheduler's wait stood still for ever (3.13 and 3.14 don't do it). The scheduler looks at its queue once without waiting before it waits with a timeout.
- **`Policy(concurrency=n)` caps a node's calls in the whole process.** At most `n` calls of one `@node` definition
  run at once, over every place a graph uses it and every run going on, so a node used twice with `concurrency=2`
  makes two calls at once, not four, and two runs of it at once make two as well. A run that waits for another run's
  call calls its other nodes meanwhile, and a run that keeps the node busy can't keep the others out. Setting
  `concurrency` is how a node keeps an outside service's quota; set on a node whose function runs a graph that
  uses the node again, it is a deadlock. The
  default is `concurrency=None`: each run calls the node one item at a time and takes no notice of other runs.
- **A call on a thread that raises past the engine ends the run with it.** A node with `concurrency` above 1 or an `attempt_timeout` that raised `SystemExit` or `KeyboardInterrupt`, or whose output type's own check raised what is no refusal, was kept by its thread: the run waited for the call for ever, and a stream of it couldn't be cancelled. The exception is handed back and raised from `resume()` or `stream()`, as it is for a node called on the scheduler's own thread.
- **A host reads how far a run is, and can know it before any costly call.** `run.progress` (a `RunProgress`) is read from the run's outcomes each time, for the run and for each node by node path: `finished` (calls with an outcome that is not waiting), `waiting`, `skipped` and `none_input` (how many of the finished calls were skipped, or not made for a `None`), `known` (a floor that only grows) and `total` (`None` while a series that decides it does not exist yet), each a `CallCount`. A subgraph node is never called itself, so its count is the sum of the nodes inside it. A stream's `node_progress` and `node_complete` carry the same `CallCount` of their node, as `count`, so the events and `run.progress` never give two numbers for one node; `node_progress` comes when any of its numbers changes. Nothing of it is saved, so a run read back in another process has the same. A node author marks a node whose call costs nothing with `Policy(costs_nothing=True)` (a node that takes `Context[...]` is refused it, `costs_nothing_takes_context`), and `run.resume(only_costs_nothing=True)` (or `run.stream(only_costs_nothing=True)`) calls only those and leaves the run `ready` (`failed` or `waiting` when one of those calls fails or asks, as in `resume()`), so the totals the inputs decide are known first; the next `resume()` repeats none of those calls or their events. The runs router has `POST /estimate`, which answers that progress and stores nothing, and `GET /runs/{id}` has the run's `progress`, written with its `outcome`, also at each checkpoint.
- **A subgraph node's `node_complete` counts what its inner nodes did.** Its `count` is the sum of its inner nodes' counts, as in `run.progress`, and its `reused` is the sum of their `reused` (it was always 0).
- **A run redone keeps each person's answer with the item it belongs to, whatever waits upstream.** Of the items that get the same arguments, the first takes the first previous answer even when a question upstream of it waits: an item after it takes no answer until that question is answered, and the run waits (a later identical item took the first answer, and the earlier one the second). With two loops, one inside the other, an item takes no answer until the inner lists of every item before it are known, so two equal pages keep their own answers whichever page's lines come back from a thread first (they could swap them). A node that may ask (`asks=True`) has every outcome reused this way, a call that returned without asking too: an identical item beyond the outcomes there were calls the function again, where it reused the first return. Such a call also waits its turn: while a question upstream of an earlier item waits, it reuses nothing, and nothing downstream of it runs (with `check -> stamp`, both `asks=True` and `stamp` never asking, `again(items=["zz", "bo"])` leaves `stamp` for `"bo"` waiting until `check`'s question about `"zz"` is answered, where it reused `stamp`'s return at once). Whether a node may ask is part of its code fingerprint, so a node that stops or starts asking is called again, and its waiting ids change.
- **A waiting id and an answer follow the call's items.** A node that asks has one fingerprint per call, the one reuse matches calls by, and its waiting id ends with its first eight hex digits. It covers the call's item in every loop it is inside, an item the node doesn't take included. An answer meant for the line `a` of one page was taken by the line `a` of another page that moved to its place, and `answer_stale`'s `ctx.moved_to` could name the other page's call; the answer is now refused, and `moved_to` names its own page's call. With `pages -> size -> approve`, where `approve` gets a page's length, swapping two pages of one length moved the answer to the other page; it stays with its page. A call whose inputs or items JSON can't write has no fingerprint: it is never reused, and its waiting id is the call alone (`a@0@0`). A saved run's `reusable` has no `outer_item_fingerprints`.
- **`graph.describe()` says how often a node runs in one field, `per_item_of`.** It is the list of the innermost loop the node is inside, as its node path and output (`lines.result`, or `per-word/split.result` for a list returned inside the subgraph node `per-word`), and `None` for a node called once and for one that was not compiled. `RunsOn`, the record that said it with `mode`, `series` and an English `msg`, is gone, and so is `InputSource.msg`: an input's `source` is its `mode` and `series`.
- **Every cell of a table a run keeps reads back from its JSON as it is.** A table a node returns, a caller gives, a person answers or an edge brings is converted where it arrives to the column types a saved run reads it back as, when no cell changes: every whole-number type as `Int64` (a `UInt64` when every cell fits), `Float32` as `Float64`, `Categorical` and `Enum` as `String`, a `Datetime` or `Duration` in milliseconds, or in nanoseconds when every cell is a whole microsecond, in microseconds, a `Decimal` as precision 38, an `Array` as a `List`, and a list or struct of these inside; the run, the nodes downstream and a loaded run all have the converted table. A column that would read back changed is refused, naming the column and the type to cast it to (`invalid_output`, `invalid_input`, `invalid_answer` or `invalid_edge_value`): a `Datetime` in another time zone than UTC, bytes, a list or struct holding a date, a time, a decimal or a float that is not finite, a `UInt64` past the largest `Int64`, a time finer than a microsecond, a millisecond time too far out for microseconds. A column with no value at all (empty, or every cell null) may still read back with another type. Such columns were saved and read back changed with no error (a `UInt64` cell past `Int64` as null, an `Int32` column as `Int64`), or refused when loaded. A saved run's table is read strictly: a cell that does not fit its column's type is refused, not read as null.
- **`@node(policy=)` is a `Policy`, `Policy()` when left out.** `None` is not a policy: mypy and pyright refuse `@node(policy=None)`.
- **A node's versions are one read-only mapping.** `definition.versions` is the node at each of its versions, by version number, newest first: `definition.versions[1]` is version 1, `list(definition.versions)` the numbers, and `1 in definition.versions` asks. A version it doesn't have is a plain `KeyError`, and writing into it a `TypeError`.
- **`run.with_answers({})` changes nothing.** An empty mapping answers no call, so the run is returned as it is. It
  made a `cancelled` or `timed_out` run `waiting` or `ready`, as an answer to a call does.
- **A saved run is held to each output's type as a node's return is.** `Run.from_json` refuses a `null` item in a
  `Series[str]` that `skipped_items` doesn't list (`saved_run_unreadable` at `outcomes.<node>`). It read it back as a
  `None` item, though a node that returns such a list fails with `invalid_output`.
- **Whether a type takes `None` is its own check's answer, everywhere.** `object` and `Literal["a", None]` take
  `None`, and `Annotated[str | None, AfterValidator(f)]` with an `f` that refuses `None` does not, as when a node
  returns one; a rule written beside the check said otherwise. So a saved `null` for such an output or item reads
  back (`object`, `Literal["a", None]`) or is refused (the validator), a `Series[Literal["a", None]]` input keeps
  its `None` items, a `None` argument to such an input calls the node, and such an output's description has the
  default `null`.
- **`run.results` shows a loop's items while a question outside it waits.** A node called per item of a returned
  series, with an input from a call outside the loop that waits for a person or hasn't returned, shows one `None`
  per item, as the reference says. An output of an `exclusive_group` there, and every node reading one, showed
  `None` in place of the list, as if the branch were not chosen.
- **`run.with_max_items(None)` removes a run's item limit.** Its signature says `int | None`; it said `int`, so a
  type checker refused the one call that removes the limit, while `max_items=None` on `run.with_inputs` and
  `run.again` keeps the run's own.
- **A checkpoint remembers which nodes its stream announced complete.** A run read back from a `run_checkpoint`
  streams `node_complete` only for the nodes not announced before that checkpoint. It announced again every node
  that completed during the stream, since a checkpoint kept the list the stream started with. The checkpoint's
  saved `complete` lists those nodes.
- **An outcome carried on keeps the call it came from.** A run made from a run that was itself made and not run to
  the end (`run.with_inputs(...).with_inputs(...)`, or `with_forgotten` then `again`) reuses for each call the outcome of
  the call with the same item indexes, as one `again` does. Two identical items each got the first one's outcome.
- **`Policy` refuses a keyword it doesn't have.** `Policy(retries=3, timeout=0.2)` raises pydantic's `ValidationError` (`unexpected_keyword_argument`), as a value of the wrong type does; it made a policy with no timeout, and a node that ran without one.
- **A failure says why it wasn't retried.** `execution_failed`'s message says the exception isn't retried, since only an `ExternalError` or a class in `Policy(retry_on=)` is, and `timeout`'s that a timed-out call isn't retried, so a node with `Policy(retries=3)` that failed once says why at the failure. An attempt that is retried logs its exception to the `conductor` logger as a warning that counts the attempts and says when the next is due (`on attempt 1 of 3; the next attempt is due in 2.0 seconds`) (it was an error, so a call that then succeeded read as two failures); a call's last failure is still an error.
- **A `Series` input given one value per call says how to collect.** `type_mismatch_series_given_one_value` said "Wire a Series of str to it", which is what the author thought they had done; it says to collect the loop's items first in a node outside the loop that takes a Series, or else to make it an input that takes one value. The problems page names `series-collect` and `table-collect`'s `constant_columns`, and the README's "One or many" names `series-collect`.
- **`make_graph` reports a chain's gap alone.** When it can't chain the nodes, it makes no automatic graph inputs and only the last node's automatic outputs, so `make_graph(parse, greet)` no longer also lists `graph_outputs_share_a_name` (or `graph_inputs_share_a_name`) for the outputs and inputs that would only clash because of the gap. What `inputs=`, `outputs=` and `defaults=` name is still checked and listed beside it. An edge it can't write, such as `("shuot.result", "count.text")` naming a node it wasn't given, leaves out every automatic graph input and output the same way, so it raises `node_not_in_graph` alone, without the two clashes that followed from the edge left out.
- **`invalid_output` opens with what is true of every bad return.** "The node's return doesn't fit its outputs: …", where it said "The node returned something it doesn't declare", which was wrong for an output it declared and didn't return, or a value of a field it does declare. A hook's node that returns something other than a mapping is told "the node must return a mapping with one key per output (a, b), not a str".
- **A hook's node declares a return a mapping fits.** `@node(..., compute_outputs=...)` on a function declared `-> str` (or any return type no mapping fits) raises `NodeDeclarationError` with the new code `outputs_not_a_mapping` (`ctx.returns`), where every call failed as `invalid_output`. Whether a mapping fits is asked by the rule an edge is checked by, so `dict[str, Any]`, `Mapping[str, Any]`, `object`, `Any`, `JsonValue`, an alias of one and a union with one are taken, and a `TypedDict` is refused.
- **A `previous` the runs router can't find says so in its `loc`.** `POST /runs` and `POST /estimate` with a `previous` no run has, or one the filter hides, answer 404 `no_such_run` with `loc` `["previous"]` (it was `[]`, which read as a route that doesn't exist); `Worker.begin` and `Worker.estimate` raise it so. A route's own run that isn't there keeps `loc` `[]`.
- **A field missing inside a table's cell is not a missing column.** A `Table` whose column is a record, given a
  row whose cell lacks one of that record's fields, was refused with "the column 'price' is missing, and the row
  type needs it". It now reads "column 'price': the field 'currency' is required", and the error's type stays
  `missing`.
- **A missing column is `column_missing` in every form of a table (provisional).** Records that lack a column of a
  `Table`'s row type gave the error type `missing` in a problem's `ctx.errors`, while a frame without the column
  gave `column_missing`. Both now give `column_missing`, with the same message. That changes the `ctx.errors` of a
  node's return given as records (`invalid_output`), a graph input (`invalid_input`), a constant (`invalid_value`)
  and gathered rows collected into a `Table` input (`invalid_edge_value`). An answer's refusal has no `ctx.errors`,
  so it doesn't change.
- **A `Series[dict]` and a `Series[dict[str, Any]]` take a table, as `Series[Any]` does.** A node declared
  `-> Series[dict]` that returned a `polars.DataFrame` failed with `invalid_output` ("Input should be a valid list"),
  while `-> Series[Any]` took the same frame as its rows. A `dict` and a `dict[str, Any]` are the row of a bare
  `Table`, so all three now read the frame as one and give every column. A `Series[dict[str, int]]` still refuses a
  frame, since a bare table checks no cell.
- **A place whose id or name isn't a valid name is `invalid_place` in `make_graph`.** An edge source or a place in
  `outputs=` such as `"greet.result~"` or `"greet.result.."` raised a bare pydantic `ValidationError`. A target, a
  key of `constants=` or `defaults=` and a place in `inputs=` such as `"greet.na~me"` was `invalid_record` from the
  record, and `"gr/eet.result"` was `node_not_in_graph`. Each is now `invalid_place` at the argument it is given in,
  inside the one `InvalidCallError`. Its message says why: "Write it as 'node.name', where neither part is empty or
  contains any of '.', '/', '@', '~'." (it ended "Write it as 'node.name'.").
- **A waiting id is read only up to a node id that isn't a name.** `not_waiting`'s `loc` for an id with a node id
  that is empty or has a `.` in it named that id: `""` gave `("",)`, `"e@0//q"` gave `("e", 0, "", "q")` and
  `"e@0/q.x"` gave `("e", 0, "q.x")`. It now stops before it, as it stops before an index that isn't a number:
  `""` and `"~3fa2c0d1"` give `()`, `"e@0//q"` gives `("e", 0)` and `"e@0/q.x"` gives `("e", 0)`.
- **`check_registry` upgrades an older version whose required input gets no value.** An older version, function or
  subgraph, with no example and a required input no value can be made up for (such as `value: T`, or any input of a
  subgraph) was `upgrade_broken`: "'value' gets no value". The check left the input empty in the graph it upgraded.
  Each required input that gets no value, from an example or made up, is now a graph input there, as `make_graph`
  makes it. So an older version's example that leaves out an input is reported once, as `example_run_failed`, where it was
  also `upgrade_broken`. A required input that takes only a constant can't be a graph input: when no value can be
  made up for one, the older version is `needs_example`, as a version with hooks already was, where it was
  `upgrade_broken` (and both, for a version with hooks). `needs_example`'s message no longer speaks of hooks:
  "check_registry can't make up a value for code, and only a constant can give one. Give this version of the node an
  example."
- **`answer_stale` says the inputs or the code changed.** "The call 'ok~8aba8744' names now waits under
  'ok~7eb8816a', since its inputs or its node's code changed after that id was written. Use that id instead." It
  said the question "was replaced by a newer one", which was wrong after a deploy that changed only the node's code.
- **`make_graph`'s `inputs=` keeps an optional input optional.** `make_graph(shout, inputs={"m": "shout.mark"})`
  for `shout(text, mark: str = "!")` writes `{"graph_input": "m", "default": "!"}`, so a run may leave `m` out. It
  wrote no default, so the graph input was required and the function's default was lost. A graph input that feeds
  several inputs takes their default when each declares the same one, and is required otherwise.
- **A `Series` wired into a `list[T]` input says to declare it `Series[T]` (provisional).** The new compile
  code `type_mismatch_series_into_list` (`ctx` `source`, `got`, `expected`, `item`) reads "'words.result' gives
  Series[str], but 'bundle.parts' takes list[str]. To take every item as one list, declare the input Series[str] in
  place of list[str]." It was plain `type_mismatch`, which said to wire an output that gives `list[str]`. A `Series`
  whose items a `list[T]` doesn't take stays `type_mismatch`.
- **A value outside the choices is refused naming them.** An answer outside an `Ask`'s choices, and a value outside
  a `Param(choices=)`, is refused with "the value is not "ida" or "bo"" (`not_a_choice` in `ctx.errors`, and in the
  `reason` of `invalid_answer`), past five counted as `type_mismatch` counts them: `"a", "b", "c", "d", "e" or 4
  more`. It said "the value is not one of the choices", while a `Literal`'s refusal named its values.
- **`check_registry` gives one problem for each problem behind it (provisional).** `hook_failed`,
  `example_hook_failed`, `subgraph_broken`, `upgrade_broken`, `example_run_failed`, `example_not_compiled` and
  `example_refused` each carry one problem behind them: `ctx.reason` is its message, `ctx.problem` its code, such
  as `"unknown_input"`, and `ctx.loc` its `loc`, as it is, such as `["b", "n"]` for a node of a subgraph's graph. They carried every problem behind them as JSON in `ctx.problems`, with `ctx.reason` their
  messages joined. So an example with a misspelt input gives two `example_not_compiled`, one `unknown_input` and one
  `input_has_no_value`, and a test reads `p.ctx["problem"]` with nothing to unpack. An example that doesn't compile
  is `example_hook_failed` for each hook's problem and `example_not_compiled` for each other one, where a mix was one
  `example_not_compiled`.
- **The docs teach one form for fixed choices.** `docs/`, the reference, the skills, the notebooks and the
  docstrings write fixed choices as `Param(choices=[...])` on a plain type, each value once:
  `Annotated[str, Param(choices=[Choice("s", "Small"), Choice("m", "Medium")])]`. A `Literal` gives the same
  choices, but it can't carry titles, so the docs write none for choices, and never one beside `Param(choices=)`
  that repeats its values. `named_value_invalid_enum`'s message says to "take the values' type, such as str, with
  Param(choices=[...])", where it said to take a `Literal` titled with `Param(choices=)`.
- **Breaking: `@node` refuses `Param(choices=)` on a `Literal` (provisional).** A `Literal`'s values are its
  choices already, so titling them wrote each value twice, and `choices_invalid_on_a_literal` refused the two lists
  when they disagreed. That code is gone. Choices on a `Literal`, or on a union of `Literal`s and `None` such as
  `Literal["a"] | None`, are the new declaration code `choices_on_a_literal`, `ctx` `type`, the plain type of the
  `Literal`'s values (`str`, `str | None` for `Literal["a", None]`), whose message says to write it in place of the
  `Literal` and keep the `Param`. Choices on a `Literal` beside another type, such as `Literal["auto"] | int`, are the
  new code `choices_on_a_literal_beside_another_type`, no `ctx`, whose message says to remove them: choices there
  would limit the field to them. Such a field could title its `Literal`'s values and leave the other type open, and
  now can't title them. A plain `Literal` keeps its values as untitled choices, and `Param(choices=)` on an enum
  still limits or titles its members.
- **A type's own check that a host's error replaces reads "The value did not pass its type's own check"** (provisional), with no final period, where it had one: a problem that quotes it as its reason, such as `choices_invalid_not_a_value`, read "check.. Give". In `ctx.errors[].msg` and `ctx.reason` the text has no period.
- **A kind of problem writes its message one way (provisional).** Two codes that each covered two mistakes are
  split. A hook that returns something compiling can't use is the new compile code `compute_hook_returned_invalid`
  (`ctx` `reason`, `error`: `not_a_mapping`, `not_a_name`, `added_input_exists` or the declaration's code), and
  `compute_hook_failed` is only a hook's `NodeError`, whose message is the hook's own, with no fix added. A graph
  output that names a node the record doesn't have is the new compile code `graph_output_names_missing_node` (`ctx`
  `output`, `source`, `missing`), as an edge's is `edge_names_missing_node`; `graph_output_not_found` is only an
  output its node doesn't have. `unknown_node` writes one message, "The registry has no node 'read'. Add it to the
  registry, or use one the registry has: upper, count." (`none` on an empty registry), where it wrote "Its nodes are
  ..." or, on an empty registry, "It has no nodes at all."
- **A refused default of a table names its column.** `invalid_value_bound_default` and
  `output_default_bound_invalid` carry `ctx.column` for a table's default that a type refuses, as `invalid_value`,
  `invalid_input`, `invalid_edge_value` and `invalid_output` do. They left it out.
- **A declaration problem says where it is in its `loc` (provisional).** A `NodeDeclarationError`'s problem has the
  `loc` a graph problem has: `("fetch",)` for a fault in the node, such as an `async def`, an empty title, a missing
  `Upgrade` or a node id used twice, and `("fetch", "url")` for one in a parameter or an output. A fault about no one
  node keeps `()`: an id that isn't a valid name, two registries whose `context_type` differ, a `@host_type`
  declared wrong, two classes under one type id. `NodeDeclarationError.where` is gone: it said the same as text,
  like `("@node('fetch') on fetch", "url")`, for some faults only. The error's text is the `loc` written as a place,
  with the node's version after its id, then the message: `fetch (version 2).url: the input has no type hint. ...`
  (was `@node('fetch') on fetch: url: ...`). The version is in the text only, for every fault at a node; the text no
  longer names the function, which the traceback's decorator line shows. A declaration problem's `loc[0]` is the node definition's id, such as `"upper"`, where a graph
  problem's is a node's id in the graph record, such as `"u"`.

A 1.x graph doesn't compile as saved. Convert each graph record with the table under
"Converting a stored 1.x graph", then save the `GraphRecord`.

### From 1.12.2, name by name

Every public name of 1.12.2, by the module it was imported from. A name not listed under 2.0 is gone.

| 1.12.2 | 2.0 |
|---|---|
| **Nodes** | |
| `Actor` | gone: a node declares no actor; the host decides who runs what |
| `BaseNode` | a plain function under `@node("id", title=, description=)` or `@registry.node(...)` |
| `conductor.metadata.InputMetadata` | `FieldDescription`, an entry of `definition.describe().inputs`; the hint and its `Param(...)` say it |
| `conductor.metadata.OutputMetadata` | `FieldDescription`, an entry of `definition.describe().outputs` |
| `conductor.registry.dynamic_inputs.ComputeInputsContext` | gone: a `compute_added_inputs` hook is a plain function whose parameters name the inputs it reads |
| `conductor.registry.dynamic_inputs.ComputeInputsFn` | `compute_added_inputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.ComputeOutputsContext` | gone: a `compute_outputs` hook's parameters name what it reads; one annotated `type` receives the arriving type |
| `conductor.registry.dynamic_outputs.ComputeOutputsFn` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `conductor.registry.dynamic_outputs.IncomingBinding` | gone: a hook parameter annotated `type` receives the type arriving on that input |
| `conductor.registry.dynamic_outputs.strip_sub_output_prefix` | gone: an output is named by its field |
| `conductor.registry.NodeDefinition` | `NodeDefinition`, the union of `FunctionDefinition` (frozen and callable, returned by `@node`) and `SubgraphDefinition` (returned by `subgraph`) |
| `conductor.types.Base64Str` | gone: any type pydantic carries travels; `bytes` is one |
| `conductor.types.Date` | `datetime.date` |
| `conductor.types.MultiNamedFile` | gone: `Series[YourFile]` |
| `conductor.types.NamedFile` | gone: declare your own pydantic model or dataclass |
| `conductor.types.NodeResult` | gone: a node returns its value; a run's are `run.results` and `run.outputs` |
| `conductor.types.OUTPUT_PREFIX` | gone: outputs are named by their fields |
| `conductor.types.RESULT_KEY` | an output of `-> X` is named `result`; a frozen dataclass names several |
| `FlowStore` | the run's context: a node's first parameter `Context[AppContext]`, the class named by `NodeRegistry(context_type=)`, the object passed as `context=` |
| `NodeCategory` | a plain string, `@node("id", title=, description=, category="text")` |
| `NodeRegistry` | `NodeRegistry(category=, tags=, extra=, types=)`, empty until nodes are added |
| `ResultFormat` | gone: a node returns its value, typed by its return hint |
| `RetryConfig` | `Policy(retries=, retry_delay=, attempt_timeout=, retry_on=)` on `@node(policy=)` |
| `SKIPPED` | `SKIPPED`, returned on a branch not taken |
| **Widgets** | |
| `conductor.widgets.Checkbox` | `Switch` |
| `conductor.widgets.CodeEditor` | gone: a host subclasses `Widget` |
| `conductor.widgets.ColumnSelect` | gone: an edge names one field of a record, `node.output.field`, after a node that gives a table's records one at a time (`series-split`) |
| `conductor.widgets.ConditionBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.ConnectionList` | gone: a `Series[X]` input takes every edge into it |
| `conductor.widgets.DatePicker` | `DatePicker` |
| `conductor.widgets.DependentDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.Dropdown` | `Dropdown`, which takes no choices: they live on the field, `Param(choices=[Choice(id, title)])` or a `Literal` |
| `conductor.widgets.EntityDropdown` | gone: a host subclasses `Widget` |
| `conductor.widgets.FileUpload` | `FileUpload`, without `multiple`: a `Series` field takes several files |
| `conductor.widgets.HumanReview` | `raise Ask(...)` in the node; its return type says what is asked |
| `conductor.widgets.IfElseBuilder` | gone with the if-else node: outputs whose default is `SKIPPED` are the branches |
| `conductor.widgets.List` | `ListInput` |
| `conductor.widgets.Multiselect` | an input typed `Series[Literal[...]]`, or `Dropdown` on a `Series` |
| `conductor.widgets.Number` | `NumberInput`, without `integer_only`: an `int` field takes whole numbers |
| `conductor.widgets.Output` | gone: an output has no widget |
| `conductor.widgets.Range` | `Slider`; its limits go on `Param(ge=, le=)` |
| `conductor.widgets.SchemaBuilder` | gone: a host subclasses `Widget` |
| `conductor.widgets.Switch` | `Switch` |
| `conductor.widgets.TableInput` | an input typed `Table[R]`, the table given as records |
| `conductor.widgets.TableSource` | an input typed `Table[R]` |
| `conductor.widgets.Tags` | gone: an input typed `Series[str]` |
| `conductor.widgets.TemplateTextarea` | gone: a host subclasses `Widget` |
| `conductor.widgets.Text` | `TextInput` |
| `conductor.widgets.Textarea` | `Textarea` |
| `conductor.widgets.WIDGET_SCHEMA_KEYS` | gone: a widget is a pydantic model; its fields are its keys |
| `conductor.widgets.Widget` | `Widget`; a host subclasses it for a control of its own, and a subclass with a field that has no JSON form (a class pydantic checks only with `isinstance`, a `polars.DataFrame`, a `type[X]`) raises pydantic's `PydanticInvalidForJsonSchema` where its class statement ends |
| `widgets` | `conductor.widgets`, imported as a module |
| `WidgetType` | a widget's `type` field |
| **Graphs and compiling** | |
| `compile` | `Graph.compile(record, registry)`, which returns a `Graph` and never raises for a fault in the graph |
| `CompiledGraph` | `Graph` |
| `conductor.expr`: `Expression`, `ExpressionError`, `ExpressionParseError`, `ExpressionRuntimeError`, `ExpressionTypeError`, `evaluate`, `parse` | gone with CEL guards on edges |
| `conductor.flow_format.dump_flow` | `record.model_dump_json()` |
| `conductor.flow_format.flow_to_dict` | `record.model_dump(mode="json")` |
| `conductor.flow_format.flow_to_yaml` | gone: YAML isn't written |
| `conductor.flow_format.load_flow_from_path` | gone: read the file and `Graph.compile` its text |
| `conductor.flow_format.load_flow` | `Graph.compile(json_text, registry)` or `GraphRecord.model_validate_json` |
| `conductor.flow_format.yaml_to_flow` | gone: YAML isn't read |
| `conductor.graph.compiler.DecisionGuard` | gone with CEL guards: outputs whose default is `SKIPPED` are the branches |
| `conductor.graph.compiler.ExtensionResolver` | gone: a graph record is a node, `subgraph("id", title=, description=)(record)`, added to a registry |
| `conductor.graph.dynamic_inputs.resolve_node_inputs` | `Graph.compile(record, registry).describe()`, a `GraphDescription` with each node's fields as its hooks made them |
| `DefinitionSource` | gone with `RegistryView` |
| `expr` | gone with CEL guards on edges |
| `Flow` | `GraphRecord`; see "Converting a stored 1.x graph" |
| `FlowDependency` | gone: a graph uses another as a node, `subgraph("id", title=, description=)(record)` |
| `FlowTrigger` | gone: the host decides when a graph runs |
| `GraphEdge` | a source, `{"node": "a", "output": "result"}`, on the input it leads into |
| `GraphNode` | `NodeRecord`; see "Converting a stored 1.x graph" |
| `RegistryView` | `registry.filter(predicate)` and `add_subregistry`; a request's own nodes go in a registry of their own |
| `resolve_graph_inputs` | `Graph.compile(record, registry).inputs`, each a `GraphInputDescription` |
| `resolve_graph_outputs` | `Graph.compile(record, registry).outputs` |
| **Running** | |
| `compute_for_each_end_outputs` | gone: a `Series[X]` input collects the items |
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
| `conductor.execution.events.FlowPausedEvent` | `NodeWaiting` per waiting call, then `RunEnd` with `status` `waiting` |
| `conductor.execution.events.FlowTimeoutEvent` | `RunEnd` with `status` `timed_out` |
| `conductor.execution.events.NodeCompleteEvent` | `NodeComplete` |
| `conductor.execution.events.NodeErrorEvent` | `NodeFailed`, its `cause` a `Problem` |
| `conductor.execution.events.NodeProgressEvent` | `NodeProgress` |
| `conductor.execution.events.NodeRetryEvent` | `NodeRetry` |
| `conductor.execution.events.NodeSkippedEvent` | `NodeComplete` with `count.finished > 0` and `count.skipped` equal to it |
| `conductor.execution.events.NodeStartEvent` | `NodeStart` |
| `conductor.execution.events.RuntimeWarningEvent` | gone: a type fault is a problem in `graph.problems` |
| `conductor.execution.events.SignalWaitingEvent` | `NodeWaiting` |
| `conductor.execution.resolver.finalize_connection_labels` | gone with connection labels: an input is named by its field |
| `conductor.execution.resolver.InputResolver` | gone: the graph reads its inputs, `graph.with_inputs(...)` |
| `conductor.execution.resolver.MULTI_VALUE_SEPARATOR` | gone: several edges into a `Series` input arrive as a list, never joined |
| `conductor.execution.results.extract_output` | `run.results[node_id][output]` |
| `conductor.execution.results.filter_all_skipped` | gone: a skipped item is `None` in `run.results` |
| `conductor.execution.results.filter_skipped` | gone: a skipped item is `None` in `run.results` |
| `conductor.execution.results.normalize_result` | gone: what a node returns is checked against its type |
| `conductor.execution.results.OutputRef` | a `Source`, written `node.output` |
| `conductor.execution.results.project_outputs` | `run.outputs`, by the names of the graph's outputs |
| `conductor.execution.state.FlowRunState` | `Run`: `status`, `results`, `waiting`, `cause` |
| `execute_sync` | `graph.run(inputs)` |
| `execute` | `graph.run(inputs)` returns a `Run`; `run.stream()` yields its events |
| `FlowCheckpoint` | a `Run`: `run.to_json()` and `Run.from_json(saved, graph)`; while it runs, `run.stream(checkpoint_every=)` yields `RunCheckpoint` |
| `FOR_EACH` | gone with the for-each node |
| `ForEachNode` | gone: a `Series` output wired to an input of one value runs the node once per item |
| `resume_sync` | `run.resume()` |
| `resume` | `run.with_answers(...)`, then `run.resume()` |
| `SUBPROCESS` | gone with the subprocess node |
| `SubprocessNode` | a graph record is a node, `subgraph("id", title=, description=)(record)` |
| `SubprocessRegistry` | gone: a subgraph's node goes in a registry like any other |
| `WHILE` | gone with the while node |
| `WhileNode` | gone: a graph has no cycles |
| **Errors** | |
| `CompilationError` | gone: compiling never raises; read `graph.problems` |
| `conductor.errors.CycleDetectionError` | the problem `cycle` in `graph.problems` |
| `conductor.errors.FlowExecutionException` | gone, as `FlowExecutionError` |
| `conductor.errors.FlowPausedException` | gone, as `FlowPausedError` |
| `conductor.errors.InputResolutionError` | `InvalidCallError` at `graph.run`, listing `missing_input`, `unknown_graph_input`, `invalid_input` |
| `conductor.errors.NodeExecutionException` | `NodeError` |
| `conductor.errors.NodeValidationException` | `NodeError` |
| `conductor.errors.TypeCheckError` | the problem `type_mismatch` in `graph.problems` |
| `ConductorError` | gone: catch `NodeDeclarationError`, `InvalidCallError` or `NodeError` |
| `errors` | gone as a module: the classes import from `conductor` |
| `FlowExecutionError` | gone: a run that failed is returned as a `Run` whose `status` is `failed` and `cause` a `Problem` |
| `FlowPausedError` | gone: a run waiting for a person is returned as a `Run`, `status` `waiting` |
| `HumanInputRequired` | `raise Ask(prompt, choices=, default=)`; each has the answer's shape, so on several outputs each goes by output name; a question's widget is its output type's `Param` |
| `LoopRunawayError` | `graph.run(max_items=)`, the problem `item_limit` |
| `NodeConnectionError` | `ExternalError`, the one failure that is retried |
| `NodeError` | `NodeError(message, code=, ctx=)`, never retried; its `problems` is its one problem |
| `NodeExecutionError` | `NodeError(message, code=, ctx=)` |
| `NodeTimeoutError` | `Policy(attempt_timeout=)`; the call fails with the code `timeout` |
| `NodeValidationError` | `NodeError(message, code=, ctx=)` |
| `SignalRequired` | `raise Ask(...)` |
| `SubprocessFailedError` | the failing inner node's `node_failed`; its `cause.loc` starts with the subgraph node and its item |
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
| `conductor_nodes.control_operators`: `OPERATORS`, `OPERATORS_BY_ID`, `OPERATORS_BY_LABEL`, `OPERATORS_BY_TYPE`, `OperatorEvaluationError`, `OperatorSpec`, `UNARY_OPERATORS` | gone: a condition is a node that returns a `bool` (`logic-equals`, `logic-is-empty`, `text-contains`, `regex-match`), wired into `control-decide.when` |
| `conductor_nodes`: `get_default_registry`, `register_all` | `conductor_nodes.registry`, added with `registry.add_subregistry(...)` |
| `conductor_nodes`: `CATEGORIES` | gone: each node has its `category`; `conductor_nodes.registry.filter(lambda d: d.category == "math")` |
| `conductor_nodes`: `decision`, `json_ops`, `logic`, `math`, `regex_ops`, `text` | a module per category, each with a `registry` of its own nodes: `control` (`control-decide`, `control-switch`, `control-default`), `json_ops`, `logic` (`logic-not`, `logic-equals`, `logic-is-empty`), `math`, `regex_ops`, `text`; `decision` is `control-decide`, `logic-if-empty` and `logic-if-equals` are gone (`logic-is-empty` and `logic-equals` wired into `control-decide.when`), and the other node ids are unchanged |
| `conductor_nodes`: `loop`, `while_loop`, `signal`, `subprocess` | gone: a node typed for one value runs once per item of a `Series` wired to it, and a pause is `raise Ask(...)` |
| `conductor_providers.fastapi.conductor_router` | `conductor_providers.fastapi.run_router(worker, metadata=, run_filter=)` over a `Worker(store, nodes_for=)`, `nodes_for=` a function of a run's metadata: runs by id, kept in a `RunStore`, on several workers; `/nodes` is the `describe()` of the request's registry, `/compile` is `POST /compile` (`graph.describe()`), `/execute-stream` is `POST /runs` and `GET /runs/{id}/events` |
| `conductor_providers.fastapi`: `ExecuteRequest`, `NodeInput`, `EdgeInput` | `StartRun`, a `GraphRecord` and its inputs |
| `conductor_providers.fastapi`: `CompileResult`, `CompileWarning` | `GraphDescription`, whose problems name each node |
| `conductor_providers.fastapi.sse_frame` | gone: each frame is an event's own JSON |
| `conductor_providers.fastapi` (`context_factory=`, `entity_resolver=`) | `context=`, a function of the run's metadata that returns the run's context; an entity list is a route of your own |
| `conductor_providers.react`: `graph_to_react`, `palette_from_registry`, `react_to_graph`, `topological_positions` | gone: a canvas stores the `GraphRecord` as it is, a node's position in `display` |
| `conductor_providers`: `PROVIDERS`, `react` | gone |

### `@registry.node(...)` arguments

| 1.12.2 | 2.0 |
|---|---|
| `base_id` | the id, first and required: `@node("upper", title=, description=)` |
| `version` | `version=`, one past the version replaced, or 1 |
| `name` | `title=`, required |
| `description` | `description=`, required; the docstring is never shown |
| `tags` | `tags=`; a registry's tags add to them |
| `category` | `category=`, a string |
| `max_retries` | `Policy(retries=)` |
| `retry_delay` | `Policy(retry_delay=)` |
| `width` | gone: how a node is drawn is the host's, `NodeRecord.display` |
| `docs` | `docs=` |
| `actor` | gone with `Actor` |
| `timeout` | `Policy(attempt_timeout=)` |
| `idempotency_key` | gone: `again` reuses a call whose arguments match, `Policy(reuse_when_inputs_match=)` |
| `uses` | gone: a first parameter typed `Context[...]` says what the node needs from the run |
| `is_decision` | the default `SKIPPED` on the outputs that are branches |
| `is_signal` | gone: `raise Ask(...)` |
| `dynamic_handles` | gone: `compute_added_inputs=` and `compute_outputs=` |
| `compute_outputs` | `compute_outputs=`, a plain function returning `{name: type hint}` |
| `compute_inputs` | `compute_added_inputs=`, the inputs it adds, taken as `**kwargs` |

### Converting a stored 1.x graph

A 1.x `Flow` becomes a `GraphRecord`, `{"nodes": {id: node, ...}, "outputs": {...}}`. Each `GraphNode` becomes a `NodeRecord` under its id, `{node, version, inputs, display}`, empty fields left out. Each `GraphEdge` becomes a source on the input it leads into, `{"sources": [{"node": source, "output": source_handle}]}`. Declare the graph's inputs (`{"graph_input": name}` on each input a caller fills) and outputs (`outputs`). Compile the result and read `graph.problems` to see what didn't carry over. A 1.x for-each over a list is gone: return the list from a node as a `Series[X]` and wire it to the input that took one item, and the node runs once per item; a node after it that collected the items takes them through a `Series` input.

| 1.x field | 2.0 |
|---|---|
| `GraphNode.id` | its key in `nodes`, without `.`, `/`, `@` or `~` |
| `GraphNode.type` | `node` (the id of the node it uses) and `version`, 1 when left out |
| `GraphNode.data` | an input's entry: `{"constant": v}` for what is fixed, `{"graph_input": name, "default": v}` for what a caller may change; an input wired by an edge has `{"sources": [...]}` |
| `GraphNode.produces` | gone: the node's outputs are its return type's |
| `GraphNode.consumes` | gone: the node's inputs are its signature's |
| `GraphNode.compensation` | gone with compensation |
| `GraphNode.on_error` | gone: a failed call ends the run `failed` |
| `GraphNode.node_label` | `display`, the host's own |
| `GraphNode.output_labels` | `display`, or a `Param(title=)` on the output |
| `GraphEdge.id` | gone: an edge is a source on the input it leads into |
| `GraphEdge.source` | the source's `node` |
| `GraphEdge.target` | the node whose input lists the source |
| `GraphEdge.source_handle` | the source's `output`, with `field` for one field of a record of a `Series` (a table's column is no field: `series-split` gives its records one at a time) |
| `GraphEdge.target_handle` | the input the source is listed on |
| `GraphEdge.when` | gone: a branch is an output whose default is `SKIPPED`; wire from that output |
| `GraphEdge.priority` | gone: several sources of one input come from different branches of one node |
| `Flow.nodes` | `nodes`, by id |
| `Flow.edges` | the `sources` of each input |
| `Flow.id` | not stored in the record: the host keys its rows |
| `Flow.version` | not stored: each node has its own `version` |
| `Flow.name` | not stored: the host's own, or the subgraph's `title` |
| `Flow.description` | not stored: the host's own, or the subgraph's `description` |
| `Flow.dependencies` | gone: a subgraph is a node added to the registry |
| `Flow.triggers` | gone: the host decides when a graph runs |
| `Flow.on_error_default` | gone: a failed call ends the run `failed` |

### Events

Events are frozen pydantic models in `conductor.events`, told apart by `type`. Two are new: `run_start`, always first, and `run_checkpoint`, the run while it runs, for a host to store, sent when `run.stream(checkpoint_every=)` asks. `run_end` is always last (but for a run its host hands back with `run.stream(hand_back=event)`, which leaves the run with calls to make and ends with a `run_checkpoint` of it), its `status` one of `complete`, `waiting`, `failed`, `stopped`, `cancelled`, `timed_out`, and its `cause` the run's `Problem` for one that failed, stopped, was cancelled or timed out (`null` otherwise). Every event reads back from its JSON as an equal event, with `TypeAdapter(Event).validate_json(text)`. The `run_end` and `run_checkpoint` a stream yields give the host the run as `event.run`, which is not part of the event: it is not in their JSON and equality leaves it out, so one read back from JSON, or made as `RunEnd(status="failed", cause=...)`, equals the streamed one and has no `run`. `Event` is declared with a `type` statement, so a JSON schema that uses it names it: a model with a field `event: Event` has `{"$ref": "#/$defs/Event"}`, and FastAPI lists it as the component `Event`.

| 1.12.2 `type` | 2.0 |
|---|---|
| `node_start` | `node_start` |
| `node_complete` | `node_complete`: `count`, the node's `CallCount` as `run.progress` has it, and `reused`, counts only; the values are `run.results` |
| `node_skipped` | `node_complete` with `count.finished > 0` and `count.skipped` equal to it |
| `node_error` | `node_failed`, `cause` a `Problem` |
| `node_progress` | `node_progress`: `count`, the node's `CallCount` (`finished`, `waiting`, `skipped`, `none_input`, `known`, `total`), as `run.progress` has it |
| `runtime_warning` | gone: a type fault is a problem in `graph.problems` |
| `flow_complete` | `run_end`, `status` `complete` |
| `flow_error` | `run_end`, `status` `failed` |
| `flow_cancelled` | `run_end`, `status` `cancelled` |
| `flow_timeout` | `run_end`, `status` `timed_out` |
| `flow_paused` | `node_waiting` per waiting call, then `run_end`, `status` `waiting` |
| `node_retry` | `node_retry`: `loc` (the node), `attempt`, `retries`, `delay`, `cause` (its `loc` the call) |
| `compensation_start` | gone with compensation |
| `compensation_complete` | gone with compensation |
| `compensation_failed` | gone with compensation |
| `signal_waiting` | `node_waiting`: `loc` (the node), `waiting` (its `loc` the call) |

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
