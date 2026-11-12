"""Compiling a graph record: check every node and edge against the node definitions, and work out once, for
every run, where each input gets its value, which loops each node is inside, and the order of the nodes.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``compile_graph`` puts the nodes in the order ``lines``, ``shout``, ``join`` and gives every input an input plan
(``input_plans.py``): ``lines.text`` is a graph input, ``shout.mark`` is the constant ``"!"``, ``shout.line`` reads
one item of ``lines.result``, and ``join.texts`` collects what ``shout`` returned for every line. ``shout`` is
inside the loop over ``lines.result``; ``lines`` and ``join`` are inside no loop.

Compiling never raises for a wrong graph. What is wrong becomes a ``Problem`` in the graph's ``problems``, naming
the node and the input it is on, and a graph with problems does not run.

The declared types alone decide the loops:

- An input that takes one ``T``, wired to an output declared ``Series[T]``, reads one item per call: its node is
  inside the loop over that output. A ``Table`` input wired to a ``Series`` of tables reads one table per call.
- An input wired to an output of a node outside the loop gets the same value on every call inside it: an outer
  value.
- An input declared ``Series[T]``, or ``Table[R]`` wired to records, collects. Wired to an output inside a loop, it
  gets one value per item of the innermost of those loops, as a list, and its node is outside that loop. Wired to
  an output inside no loop, it gets that one value as a list of one.
- A node is inside the loops of its input with the most loops. Every other input's loops must be the outermost of
  those, because the items of two different lists are never paired; an input whose loops are not has the problem
  ``series_not_nested``.
- A loop is named by the output it loops over, so every node that reads items of ``lines.result`` is inside the
  same loop.
"""

from __future__ import annotations

import heapq
import logging
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

from pydantic import StringConstraints, TypeAdapter, ValidationError

from conductor._internal.address import split_address
from conductor._internal.compiled.describe import GraphInput
from conductor._internal.compiled.input_plans import (
    EdgeSource,
    InputCollectsLoopResults,
    InputConstant,
    InputFromEdges,
    InputFromGraphInput,
    InputMissing,
    InputPlan,
    InputReadsOneValue,
    InputWrapsOneValue,
    OutputAddress,
    OutputInLoops,
)
from conductor._internal.compiled.nodes import (
    CompiledFunctionNode,
    CompiledNode,
    CompiledSubgraphNode,
    RequiredBranch,
)
from conductor._internal.compiled.type import CompiledGraph, InputField
from conductor._internal.field import Field
from conductor._internal.graph_record import GraphRecord, NodeRecord
from conductor._internal.markers import REQUIRED
from conductor._internal.node_definition import (
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    is_node_id,
    read_by_hooks,
)
from conductor._internal.problem import NodeDeclarationError, NodeError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import FieldValueError, Param, Type

_logger = logging.getLogger("conductor")


def compile_graph[GraphT: CompiledGraph](record: GraphRecord, definitions: Mapping[str, NodeDefinition], graph_type: type[GraphT]) -> GraphT:
    """Compiles ``record`` and returns it as a ``graph_type``. Every compiled graph a host gets is made here.

    Args:
        record: The graph record.
        definitions: The newest version of each node definition, by id. An older version a node asks for is found
            through it.
        graph_type: The class of the graph to return: ``Graph``, the class a host uses.

    Never raises for a wrong graph: the returned graph's ``problems`` say what is wrong.
    """
    return _Compiler(definitions, (), {}).compile(record, graph_type)



#: Stands in for a constant or default of the graph record that its input's type rejected, while a node is
#: compiled. The input gets ``InputMissing``, and the problem ``invalid_value`` says why.
_INVALID: Any = object()

#: Validates what a hook returns: a type for each input or output, by name. A name is not empty and has none of
#: ``.``, ``/`` and ``@``, which addresses are written with.
_HOOK_FIELDS = TypeAdapter(dict[Annotated[str, StringConstraints(min_length=1, pattern=r"^[^./@]+$")], Any])


@dataclass(frozen=True, slots=True)
class _InputPlans:
    """The input plans of one node, and what follows from them.

    Attributes:
        inputs: The input plan of each input, by name.
        loops: The loops the node is inside, outermost first. In the example, ``shout`` is inside the loop over
            ``lines.result``.
        required_branches: The branches that must be chosen upstream for the node to be called.
        bound: For a node whose input types have type variables (``T``), the type each variable gets from the
            edges into the node. ``None`` for every other node, which is nearly all of them.
    """

    inputs: dict[str, InputPlan]
    loops: tuple[OutputAddress, ...]
    required_branches: frozenset[RequiredBranch]
    bound: Mapping[TypeVar, Any] | None


class _Compiler:
    """Compiles one graph record. It compiles the nodes one at a time, each after the nodes upstream of it, and
    keeps what it has found so far. A subgraph used as a node is compiled by a ``_Compiler`` of its own.

    Attributes:
        definitions: The newest version of each node definition, by id.
        inside_subgraphs: The ids of the subgraph definitions whose records contain this record, outermost first.
            A subgraph found among them contains itself.
        compiled_subgraphs: Each subgraph compiled so far, by its id, its version and the subgraphs it is inside.
            Every compiler of one ``compile_graph`` call shares it, so a subgraph used twice is compiled once.
        problems: The problems found so far.
        nodes: Each node compiled so far, by id.
        graph_inputs: Each graph input found so far, in the order found, with the field the caller gives a
            value for.
        position: Each node's index in the record, which breaks ties in the order of the nodes.
    """

    def __init__(self, definitions: Mapping[str, NodeDefinition], inside_subgraphs: tuple[str, ...], compiled_subgraphs: dict[tuple[str, int, tuple[str, ...]], CompiledGraph]) -> None:
        self.definitions, self.inside_subgraphs, self.compiled_subgraphs = definitions, inside_subgraphs, compiled_subgraphs
        self.problems: list[Problem] = []
        self.nodes: dict[str, CompiledNode] = {}
        self.graph_inputs: list[tuple[GraphInput, Field]] = []
        self.position: dict[str, int] = {}

    def compile[GraphT: CompiledGraph](self, record: GraphRecord, graph_type: type[GraphT]) -> GraphT:
        """Returns ``record`` compiled, as a ``graph_type``.

        It never raises for a wrong record. The graph has the problems, and every node whose inputs and outputs
        could be worked out. A node whose definition is not found, or whose hook failed, is left out.
        """
        invalid_ids = {n.id for n in record.nodes if not is_node_id(n.id)}
        self.problems += [problem.invalid_node_id(node_id) for node_id in dict.fromkeys(n.id for n in record.nodes) if node_id in invalid_ids]
        known: dict[str, NodeRecord] = {}
        twice: set[str] = set()
        for node_record in record.nodes:
            if node_record.id in known:
                twice.add(node_record.id)
            elif node_record.id not in invalid_ids:
                known[node_record.id] = node_record
        self.problems += [problem.duplicate_node_id(node_id) for node_id in sorted(twice)]
        self.position = {node_id: i for i, node_id in enumerate(known)}
        edges_into: dict[str, dict[str, list[str]]] = {node_id: {} for node_id in known}
        seen: set[tuple[str, str]] = set()
        for source, target in record.edges:
            node_id, name = split_address(target)
            source_id = split_address(source)[0]
            if source_id in invalid_ids or node_id in invalid_ids:
                continue  # the problem invalid_node_id covers this edge
            if (source, target) in seen:
                self.problems.append(problem.repeated_edge(node_id, name, source=source))
                continue
            seen.add((source, target))
            missing = [n for n in (source_id, node_id) if n not in known]
            if missing:
                self.problems.append(problem.edge_names_missing_node(node_id, name, missing=missing[0], source=source, target=target))
            if node_id in known:  # wired even when the source does not exist, so the input is no graph input
                edges_into[node_id].setdefault(name, []).append(source)
        ordered = self._order(list(known), edges_into)
        for node_id in ordered:
            match self._resolve(known[node_id]):
                case FunctionDefinition() as function:
                    self._compile_function_node(known[node_id], function, edges_into[node_id])
                case SubgraphDefinition() as graph:
                    self._compile_subgraph_node(known[node_id], graph, edges_into[node_id])
                case None:
                    pass
        names = [field.name for _, field in self.graph_inputs]
        inputs: dict[str, GraphInput] = {}
        input_fields: dict[str, InputField] = {}
        for graph_input, field in self.graph_inputs:
            name = field.name if names.count(field.name) == 1 else graph_input.address  # two graph inputs with one name go by address
            inputs[name] = graph_input
            input_fields[graph_input.address] = InputField(name, graph_input.address, field)
        downstream: dict[str, list[CompiledNode]] = {node_id: [] for node_id in self.nodes}
        needs_context: list[tuple[str, FunctionDefinition[..., Any]]] = []
        for compiled_node in self.nodes.values():
            needs_context += compiled_node.functions_with_context
            for source_id in compiled_node.upstream_ids:
                downstream[source_id].append(compiled_node)
        return graph_type(record, self.problems, inputs, self.nodes, input_fields, downstream, needs_context)

    def _resolve(self, record: NodeRecord) -> NodeDefinition | None:
        """Returns the node definition a node of the record names, at the record's version; ``None``, with a
        problem, when there is none."""
        newest = self.definitions.get(record.node)
        if newest is None:
            self.problems.append(problem.unknown_node(record.id, node=record.node, choices=sorted(self.definitions)))
            return None
        if record.version not in newest.versions:
            self.problems.append(problem.unknown_version(record.id, node=record.node, version=record.version, newest=newest.version))
            return None
        return newest.version_of(record.version)

    def _order(self, order: list[str], edges_into: dict[str, dict[str, list[str]]]) -> list[str]:
        """Returns the node ids in the order they are compiled: each after the nodes its edges come from, and
        in the record's order where that leaves a choice.

        Every node on a cycle has the problem ``cycle``. To go on, the cycle's first node in the record is put
        first. The edge into it from inside the cycle then comes from a node that is not compiled yet, so the
        input it leads to gets no value. The rest of the cycle follows in order.
        """
        position = self.position
        reads = {n: frozenset({split_address(s)[0] for sources in edges_into[n].values() for s in sources} & position.keys()) for n in order}
        downstream: dict[str, list[str]] = {n: [] for n in order}
        for n, sources in reads.items():
            for source in sources:
                downstream[source].append(n)
        waiting = {n: len(sources) for n, sources in reads.items()}
        ready = [position[n] for n in order if not waiting[n]]
        heapq.heapify(ready)
        done: list[str] = []
        on_cycles: list[str] | None = None
        while len(done) < len(order):
            if not ready:  # every node not yet ordered is on a cycle or downstream of one
                if on_cycles is None:
                    stuck = {n for n in order if waiting[n]}
                    on_cycles = [n for n in order if n in stuck and _reaches(n, n, downstream, stuck)]  # on a cycle, not only downstream of one
                    self.problems += [problem.cycle(n) for n in on_cycles]
                first = next(n for n in on_cycles if waiting[n] > 0)
                waiting[first] = 0
                ready.append(position[first])
            n = order[heapq.heappop(ready)]
            done.append(n)
            for reader in downstream[n]:
                waiting[reader] -= 1
                if not waiting[reader]:
                    heapq.heappush(ready, position[reader])
        return done

    # -- compiling a node -------------------------------------------------------

    def _compile_function_node(self, record: NodeRecord, definition: FunctionDefinition[..., Any], edges: dict[str, list[str]]) -> None:
        """Compiles a node whose definition is a function, and adds it to ``nodes``.

        Its inputs are those its definition declares and those its ``compute_added_inputs`` hook adds; each gets
        an input plan. Its outputs are those its definition declares, or what its ``compute_outputs`` hook
        returns, given the types its edges read. When a hook fails, the node is left out and the graph has the
        problem. Every other node is added, even when some of its inputs have problems.
        """
        fields = dict(definition.inputs)
        constants = self._constants_of(record, fields)
        added: dict[str, Field] = {}
        if definition.compute_added_inputs is not None:
            made = self._hook(record, definition, "compute_added_inputs", definition.compute_added_inputs, constants, {}, ())
            if made is None:
                return
            added = made
            taken = sorted(added.keys() & (fields.keys() | definition.context.keys()))
            if taken:
                self.problems.append(problem.compute_hook_failed(record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            fields.update(added)
            constants.update(self._constants_of(record, added))
        planned = self._plan_inputs(record, fields, constants, read_by_hooks(definition), edges, generic=definition.generic)
        bound = planned.bound
        outputs: dict[str, Field] | None = dict(definition.outputs) if bound is None else {name: field.bind(bound) for name, field in definition.outputs.items()}
        if definition.compute_outputs is not None:
            arriving = {name: plan.sources[0].type.bare for name, plan in planned.inputs.items() if plan.sources}
            outputs = self._hook(record, definition, "compute_outputs", definition.compute_outputs, constants, arriving, tuple(added))
        if outputs is not None:
            self.nodes[record.id] = CompiledFunctionNode(
                id=record.id, loops=planned.loops, inputs=planned.inputs, outputs=outputs, required_branches=planned.required_branches,
                definition=definition,
            )

    def _compile_subgraph_node(self, record: NodeRecord, definition: SubgraphDefinition, edges: dict[str, list[str]]) -> None:
        """Compiles a node whose definition is a subgraph, and adds it to ``nodes``.

        Example: the subgraph ``strip -> shout`` used as the node ``loud``. The subgraph's graph record is
        compiled first, on its own. The node's inputs are that graph's inputs, by their names there (``line``).
        Its outputs are that graph's outputs, by their addresses there, so an edge reads ``loud.shout.result``.
        An output keeps the field the node inside declares. The loops and the branches an output depends on are
        asked of the compiled node (``loops_around``, ``branches_required_for``), which answers as it would if the
        subgraph's nodes were drawn in this graph.

        A subgraph that contains itself is a problem. So is every problem of the subgraph's own graph, with this
        node's id put in front of its ``loc``. In both cases the node is left out, because the inputs and outputs
        of a graph with problems are not known.
        """
        if definition.id in self.inside_subgraphs:
            self.problems.append(problem.subgraph_contains_itself(record.id, node=definition.id))
            return
        stack = (*self.inside_subgraphs, definition.id)
        key = (definition.id, definition.version, stack)
        if key not in self.compiled_subgraphs:
            self.compiled_subgraphs[key] = _Compiler(self.definitions, stack, self.compiled_subgraphs).compile(definition.graph, CompiledGraph)
        inner = self.compiled_subgraphs[key]
        if inner.problems:
            self.problems += [inner_problem.model_copy(update={"loc": (record.id, *inner_problem.loc)}) for inner_problem in inner.problems]
            return
        fields = {graph_input.name: Field(graph_input.name, graph_input.field.hint, graph_input.field.default) for graph_input in inner._input_fields.values()}
        outputs: dict[str, Field] = {}
        for address in inner.outputs:
            node_id, output = split_address(address)
            inner_node = inner._nodes[node_id]
            field = inner_node.outputs[output]
            group = field.param.exclusive_group
            if group:  # the group is named by the node inside (``route.length``), so groups of two nodes stay apart
                field = Field(field.name, Annotated[field.hint, Param(exclusive_group=f"{node_id}.{group}")], field.default)
            outputs[address] = Field(address, field.hint, field.default)
        planned = self._plan_inputs(record, fields, self._constants_of(record, fields), frozenset(), edges)
        self.nodes[record.id] = CompiledSubgraphNode(
            id=record.id, loops=planned.loops, inputs=planned.inputs, outputs=outputs, required_branches=planned.required_branches,
            definition=definition, inner=inner,
        )

    def _plan_inputs(
        self, record: NodeRecord, fields: Mapping[str, Field], constants: Mapping[str, Any], constant_at_compile: frozenset[str],
        edges: dict[str, list[str]], *, generic: bool = False,
    ) -> _InputPlans:
        """Makes the input plan of every input of a node, and works out which loops the node is inside.

        In the example, ``shout.line`` reads one item of ``lines.result``, so ``shout`` is inside the loop over
        ``lines.result``. A node is inside the loops of its input with the most loops, and the loops of every
        other input must be the outermost of those (else ``series_not_nested``).

        An input that cannot get a value gets ``InputMissing``, and the graph has a problem that says why: an
        edge the input cannot read, loops that do not nest, or several edges that are not alternatives. The node
        is kept so an editor can still draw it; the graph does not run.

        Args:
            record: The node in the graph record.
            fields: The node's inputs, by name.
            constants: The constant or default the record gives each input, already read by the input's type, by
                name; ``_INVALID`` where the type rejected it.
            constant_at_compile: The inputs a hook reads when the graph is compiled. Their value must be known
                then, so they never become graph inputs.
            edges: For each input that edges lead to, by name, the addresses of the outputs they come from.
            generic: Whether an input's type has a type variable, which the edges then bind.
        """
        for name in sorted({*record.constants, *record.defaults, *edges} - fields.keys()):
            self.problems.append(problem.unknown_input(record.id, name, inputs=list(fields)))
        sources = {name: self._sources(record.id, name, edges[name]) for name in fields if name in edges}
        bound = None
        if generic:
            bound = _bound(fields, sources)
            fields = {name: field.bind(bound) for name, field in fields.items()}  # the compiled node's fields have no type variables left
        first = not any(name in fields for name in edges)  # no edge leads into the node, so its open inputs are graph inputs
        inputs: dict[str, InputPlan] = {}
        for name, field in fields.items():
            if name in record.constants and name in record.defaults:
                self.problems.append(problem.constant_and_default(record.id, name))
            if name in edges:
                if name in record.constants or name in record.defaults:
                    self.problems.append(problem.constant_and_edge(record.id, name))
                edge_sources = sources[name]
                inputs[name] = InputMissing(field) if edge_sources is None else self._plan_from_edges(record.id, field, edge_sources)
            elif name in record.constants:
                inputs[name] = _constant_plan(field, constants[name])
            elif first and name not in constant_at_compile:
                default = constants.get(name, field.default)
                if default is _INVALID:
                    inputs[name] = InputMissing(field)
                    continue
                address = f"{record.id}.{name}"
                # a default in the record replaces the node's own
                caller_field = field if default is field.default else Field(name, field.hint, default)
                self.graph_inputs.append((GraphInput(address=address, node_id=record.id, field=caller_field.description), caller_field))
                inputs[name] = InputFromGraphInput(field, address)
            elif name in record.defaults:
                self.problems.append(problem.default_on_later_node(record.id, name))
                inputs[name] = _constant_plan(field, constants[name])
            elif field.default is REQUIRED and name not in constant_at_compile:  # an input a hook reads gets hook_needs_value instead
                self.problems.append(problem.unbound_on_later_node(record.id, name))
                inputs[name] = InputMissing(field)
            else:
                inputs[name] = InputConstant(field, field.default)
        for name in self._inputs_with_conflicting_edges(record.id, inputs):
            inputs[name] = InputMissing(fields[name])
        loops = max((plan.loops for plan in inputs.values()), key=len, default=())
        for name, plan in inputs.items():
            if isinstance(plan, InputCollectsLoopResults) and (source_loops := plan.sources[0].loops_of_value) == loops[: len(source_loops)] and not any(s.item for s in plan.sources):
                # Another input already puts the node inside the loop this input would collect, so each call has
                # one value here, its own item's: the input gets it as a list of one.
                inputs[name] = InputWrapsOneValue(plan.field, plan.sources)
        for name, plan in inputs.items():
            if loops[: len(plan.loops)] != plan.loops:  # not the node's outermost loops: items of two different lists are never paired
                series = [loops[-1], plan.loops[-1]]
                # the fix suggested: wire the list compiled first into the other node, which makes no cycle
                outer, inner = sorted(series, key=lambda port: list(self.nodes).index(port.node_id))
                reading = next(n for n, p in inputs.items() if p.loops == loops)
                subgraphs = list(dict.fromkeys(port.node_id for port in series if isinstance(self.nodes[port.node_id], CompiledSubgraphNode)))
                if subgraphs:  # an edge of this graph cannot change the loops inside a subgraph: no wire to suggest
                    self.problems.append(problem.loops_not_nested_inside_subgraph(
                        record.id, series=[str(s) for s in series], inputs=[reading, name], subgraphs=subgraphs,
                    ))
                else:
                    self.problems.append(problem.series_not_nested(
                        record.id, series=[str(s) for s in series], inputs=[reading, name],
                        wire_from=str(outer), wire_into=inner.node_id,
                    ))
                inputs[name] = InputMissing(plan.field)
        return _InputPlans(inputs, loops, self._required_branches(inputs), bound)

    def _constants_of(self, record: NodeRecord, fields: Mapping[str, Field]) -> dict[str, Any]:
        """Returns the constant or default the record gives each of ``fields``, read by the input's type, by
        name. An input the record gives neither is left out. Where the type rejects the value, the graph gets the
        problem ``invalid_value`` and the input gets ``_INVALID``."""
        constants: dict[str, Any] = {}
        for name, field in fields.items():
            if name not in record.constants and name not in record.defaults:
                continue
            try:
                constants[name] = field.whole.read(record.constants[name] if name in record.constants else record.defaults[name])
            except FieldValueError as refused:
                self.problems.append(problem.invalid_value(record.id, name, reason=refused.reason, errors=refused.errors, column=refused.column))
                constants[name] = _INVALID
        return constants

    def _hook(
        self, record: NodeRecord, definition: FunctionDefinition[..., Any], kind: HookName, hook: Callable[..., Mapping[Any, Any]],
        constants: Mapping[str, Any], arriving: Mapping[str, Any], added: tuple[str, ...],
    ) -> dict[str, Field] | None:
        """Calls one of a node's hooks and returns what it returned, as fields by name; ``None``, with a
        problem, when the hook cannot be called, fails, or returns something that is not fields.

        ``@node`` read from the hook's signature what each parameter gets: the value of the input of the same
        name (its constant or default), the type the edges into that input read (``Any`` when no edge leads to
        it), or, for ``**added: type``, those types for every input ``compute_added_inputs`` added.

        Args:
            record: The node in the graph record.
            definition: The node's definition.
            kind: Which hook it is: ``compute_added_inputs`` or ``compute_outputs``.
            hook: The hook.
            constants: The constant or default the record gives each input, read by the input's type, by name.
            arriving: The type the edges into each input read, by input name.
            added: The inputs ``compute_added_inputs`` added.
        """
        kwargs: dict[str, Any] = {}
        for name, receives in definition.hook_parameters[kind].items():
            match receives:
                case "arriving_type":
                    kwargs[name] = arriving.get(name, Any)
                case "added_arriving_types":
                    kwargs.update({input_name: arriving.get(input_name, Any) for input_name in added})
                case "value":
                    value = constants.get(name, definition.inputs[name].default)  # @node made sure the parameter names a declared input
                    if value is REQUIRED:
                        self.problems.append(problem.hook_needs_value(record.id, name))
                        return None
                    if value is _INVALID:
                        return None  # the problem invalid_value says why
                    kwargs[name] = value
        try:
            returned = hook(**kwargs)
        except NodeError as error:  # the hook raised a NodeError: keep its message, code and ctx
            self.problems.append(problem.compute_hook_failed(record.id, reason=error.message, error=error.code, error_ctx=dict(error.ctx) or None))
            return None
        except Exception as error:  # noqa: BLE001  any other exception of the hook is a problem of this node
            _logger.error("the %s hook of node %r failed", kind, record.id, exc_info=error)
            self.problems.append(problem.compute_hook_failed(record.id, reason=str(error), error=type(error).__name__))
            return None
        try:
            hints = _HOOK_FIELDS.validate_python(returned)
        except ValidationError as invalid:
            for fault in invalid.errors():
                if fault["loc"]:  # an invalid name: the error's loc is that name
                    self.problems.append(problem.parameter_name_invalid(record.id, str(fault["loc"][0])))
                else:  # the returned value is not a mapping
                    reason = f"its {kind} hook returned a {type(returned).__name__}; a hook returns its fields by name, as a mapping of each name to its type"
                    self.problems.append(problem.compute_hook_failed(record.id, reason=reason, error="not_a_mapping"))
            return None
        try:
            return {name: Field(name, *_hint_and_default(hint)) for name, hint in hints.items()}
        except NodeDeclarationError as error:  # a returned type conductor does not support is the hook's failure
            self.problems.append(problem.compute_hook_failed(record.id, reason=error.message, error=error.code))
            return None

    # -- edges ------------------------------------------------------------------

    def _sources(self, node_id: str, name: str, addresses: list[str]) -> list[EdgeSource] | None:
        """Returns what the edges from ``addresses`` into one input read, in the record's order of their nodes
        and then of the outputs of each node.

        Returns:
            The sources. ``None`` when an edge cannot be read: its address names no output of its node (the graph
            gets the problem), or its node was not compiled (a problem of its own says why, or it is on a cycle
            and comes later).
        """
        sources: list[EdgeSource] = []
        for address in addresses:
            source = self._source(address, node_id, name)
            if source is None:
                return None
            sources.append(source)
        if len(sources) > 1:
            sources.sort(key=self._edge_order)
        return sources

    def _plan_from_edges(self, node_id: str, field: Field, sources: list[EdgeSource]) -> InputFromEdges | InputMissing:
        """Returns the plan of an input whose edges read ``sources``.

        An input that takes one value reads one value per call, whatever its type; when a source is a
        ``Series``, that is one item of it. A ``Series`` input collects the innermost loop its sources are inside,
        or gets the one value of a source inside no loop as a list of one. A ``Table`` input reads a whole table
        when every edge reads one (from a ``Series`` of tables, one table per call), and else collects records.

        Returns:
            The plan. ``InputMissing``, with the problem, when an edge reads a type the input does not accept.
        """
        collects, expected = _collects_and_expected_type(field, sources)
        for source in sources:
            if expected is not source.type and expected.fits(source.type) is None:  # the identity test first: most edges read exactly the input's type
                self.problems.append(_mismatch(node_id, field, source))
                return InputMissing(field)
        deepest = max((s.loops_of_value for s in sources), key=len)
        if not collects:
            return InputReadsOneValue(field, tuple(sources), deepest)
        if not deepest:
            return InputWrapsOneValue(field, tuple(sources))
        source_node, output = deepest[-1]
        return InputCollectsLoopResults(field, OutputInLoops(source_node, output, deepest[:-1]), tuple(sources))

    def _source(self, address: str, node_id: str, name: str) -> EdgeSource | None:
        """Returns what an edge from ``address`` reads: ``node.output``, or ``node.output.field`` for one field
        of a record.

        The output's name is what follows the node's id. The outputs of a subgraph node are themselves addresses
        (``shout.result``), so in ``loud.shout.result`` the output is ``shout.result``.

        Returns:
            The edge's source, which reads one item per call when the output is a ``Series``. ``None``, with a
            problem, when the address names no output of its node; ``None`` when its node was not compiled.
        """
        source_id, rest = split_address(address)
        source = self.nodes.get(source_id)
        if source is None:  # not compiled, on a cycle and later, or no such node: a problem of its own says why
            return None
        output, _, record_field = rest.partition(".")
        if output not in source.outputs:  # a subgraph node's output: the output name the address goes on with
            output = next((declared for declared in source.outputs if rest == declared or rest.startswith(f"{declared}.")), "")
            record_field = rest[len(output) + 1 :]
        field = source.outputs.get(output)
        if field is None:
            self.problems.append(problem.unknown_output(node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs]))
            return None
        if record_field and (split := _split_gives(field.type)):  # only a record has fields to wire; a whole table or list needs a Split node first
            self.problems.append(problem.unknown_output_split_first(
                node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs], record_field=record_field,
                whole=split[0], fields=split[1],
            ))
            return None
        read_type = _field_type(field.type, record_field) if record_field else field.type
        if read_type is None:
            fields = [f"{source_id}.{output}.{f}" for f in field.type.fields]
            self.problems.append(problem.unknown_output(node_id, name, source=address, outputs=fields, record_field=record_field or None))
            return None
        return EdgeSource(source_id, output, source.loops_around(output), record_field or None, read_type, item=field.is_series)

    def _edge_order(self, source: EdgeSource) -> tuple[int, int, str]:
        """Returns where an edge falls among several into one input: by its node's index in the record, then by
        its output's index in that node."""
        return self.position[source.node_id], list(self.nodes[source.node_id].outputs).index(source.output), source.address

    # -- branches ---------------------------------------------------------------

    def _branches_required_for(self, source: EdgeSource) -> frozenset[RequiredBranch]:
        """The branches that must be chosen for the output ``source`` reads to have a value."""
        return self.nodes[source.node_id].branches_required_for(source.output)

    def _inputs_with_conflicting_edges(self, node_id: str, inputs: Mapping[str, InputPlan]) -> set[str]:
        """Finds the inputs whose edges cannot be read together, and adds a problem for each.

        Several edges into one input must be alternatives: each pair must need different outputs of one exclusive
        group, so a call gets a value from at most one of them (else ``several_edges_into_one_input``). They must
        also be inside the same loops, so the one that has the value is found by the same item indexes (else
        ``alternatives_from_different_loops``).

        Two collecting inputs that do not accept ``None`` items, and whose edges are all on opposite branches, are
        a problem too (``lists_from_different_loops``). An item one list has no value for is left out of both
        lists, so both would always be empty.

        Returns:
            The names of the inputs with a problem.
        """
        refused: set[str] = set()
        for name, plan in inputs.items():
            branches = [self._branches_required_for(s) for s in plan.sources]
            loops_of_inputs = list(dict.fromkeys(s.loops_of_value for s in plan.sources))
            if not all(_never_together(a, b) for i, a in enumerate(branches) for b in branches[i + 1:]):
                self.problems.append(problem.several_edges_into_one_input(node_id, name, sources=[s.address for s in plan.sources]))
                refused.add(name)
            elif len(loops_of_inputs) > 1:
                series = [str(loops[-1]) if loops else None for loops in loops_of_inputs[:2]]
                self.problems.append(problem.alternatives_from_different_loops(node_id, name, series=series))
                refused.add(name)
        aligned = [
            (name, [self._branches_required_for(s) for s in plan.sources]) for name, plan in inputs.items()
            if isinstance(plan, InputCollectsLoopResults) and not plan.field.accepts_none_items and name not in refused
        ]
        for i, (a, branches_a) in enumerate(aligned):
            for b, branches_b in aligned[i + 1:]:
                if all(_never_together(x, y) for x in branches_a for y in branches_b):
                    self.problems.append(problem.lists_from_different_loops(node_id, inputs=[a, b]))
                    refused.update((a, b))
        return refused

    def _required_branches(self, inputs: Mapping[str, InputPlan]) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for a node with these ``inputs`` to be called: for each
        input, the branches that every edge into it needs.

        A collecting input needs only the branches chosen outside the loop it collects. A branch chosen per
        item inside the loop only leaves items out of the list; the node is still called.
        """
        required_branches: set[RequiredBranch] = set()
        for plan in inputs.values():
            if not plan.sources:
                continue
            branches = [self._branches_required_for(s) for s in plan.sources]
            shared = branches[0].intersection(*branches[1:])
            if isinstance(plan, InputCollectsLoopResults):
                shared = frozenset(branch for branch in shared if branch.depth <= len(plan.loops))
            required_branches |= shared
        return frozenset(required_branches)


def _collects_and_expected_type(field: Field, sources: list[EdgeSource]) -> tuple[bool, Type]:
    """Returns whether the input ``field`` collects what ``sources`` read, and the type each source must fit:
    the input's whole type when it reads one value; when it collects, the type of one item of the ``Series`` or of
    one record of the table.

    A ``Table`` input reads a whole table when every edge reads one, as one value or as each item of a ``Series``
    of tables. An output declared ``Any`` could be a table or a record: it counts as a table when it has one value
    in the run, and as a record when it has one value per item of a loop.
    """
    record = field.whole.record
    collects = field.is_series or (record is not None and not all(
        field.whole.fits(s.type) is not None and (s.type.bare is not Any or not s.loops_of_value) for s in sources
    ))
    return collects, field.whole if not collects else field.type if record is None else record


def _bound(fields: Mapping[str, Field], sources: Mapping[str, list[EdgeSource] | None]) -> dict[TypeVar, Any]:
    """Returns the type each type variable of a node's inputs gets from the edges into them. When edges into
    two inputs that share a variable read ``int`` and ``str``, the variable gets ``int | str``.

    Only an edge binds a variable: a constant, a default and a graph input bind nothing. A variable no edge binds
    becomes its bound, else ``Any``, when the fields are bound.
    """
    bound: dict[TypeVar, Any] = {}
    for name, field in fields.items():
        edge_sources = sources.get(name)
        if not field.whole.variables or not edge_sources:
            continue
        expected = _collects_and_expected_type(field, edge_sources)[1]
        for source in edge_sources:
            for variable, hint in (expected.fits(source.type) or {}).items():
                bound[variable] = bound[variable] | hint if variable in bound else hint
    return bound


def _split_gives(whole: Type) -> tuple[Literal["table", "list"], list[str]] | None:
    """Returns what the Split node would make of a value of the type ``whole``, for a problem to suggest it.

    Returns:
        ``"table"`` or ``"list"``, and the fields of each record or item it gives: none for a table whose columns
        are not known, and none for a list whose items are not records. ``None`` for a type that is neither a
        table nor a list.
    """
    if whole.record is not None:
        return "table", list(whole.fields)
    one = whole.options[0] if len(whole.options) == 1 else None
    if one is list or typing.get_origin(one) is list:
        return "list", list(Type(next(iter(typing.get_args(one)), Any)).fields)
    return None


def _mismatch(node_id: str, field: Field, source: EdgeSource) -> Problem:
    """Returns the ``type_mismatch`` problem of an edge from ``source`` into the input ``field``.

    Where one mistake explains it, the problem says the fix: a whole table or list wired to an input that takes
    one value needs a Split node first, and a table input wired to records that lack one of its columns needs
    that column added.
    """
    address, expected = source.address, field.whole.id
    got = f"Series[{source.type.text}]" if source.item else source.type.text
    split = _split_gives(source.type)
    if field.whole.record is not None:
        has = (source.type.record or source.type).fields
        missing = [column for column in field.whole.fields if column not in has]
        if has and missing:
            column_type = Type(field.whole.fields[missing[0]]).id
            return problem.type_mismatch_add_column(node_id, field.name, source=address, got=got, expected=expected, column=missing[0], column_type=column_type, has=list(has))
    elif split and not field.is_series:
        return problem.type_mismatch_split_first(node_id, field.name, source=address, got=got, expected=expected, whole=split[0], fields=split[1])
    return problem.type_mismatch(node_id, field.name, source=address, got=got, expected=expected)


def _constant_plan(field: Field, value: Any) -> InputConstant | InputMissing:
    """Returns the plan of an input with a constant: ``InputConstant``, or ``InputMissing`` when the input's type
    rejected the constant."""
    return InputMissing(field) if value is _INVALID else InputConstant(field, value)


def _field_type(record: Type, name: str) -> Type | None:
    """Returns the type of the field ``name`` of a record of the type ``record``.

    Returns:
        A ``TypedDict``'s, dataclass's or pydantic model's declared type for it; a mapping's value type, ``Any``
        for a bare ``dict`` and for a value of any type; ``None`` for a record without the field, and for a type
        that is not a record.
    """
    one = record.options[0] if len(record.options) == 1 else None
    if one is Any or one is dict:
        return Type(Any)
    if typing.get_origin(one) in (dict, Mapping):
        return Type(typing.get_args(one)[1])
    return Type(record.fields[name]) if name in record.fields else None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """Reads what a hook returns for one field: a type, or ``(type, default)`` as pydantic's ``create_model``
    takes it."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


def _never_together(a: frozenset[RequiredBranch], b: frozenset[RequiredBranch]) -> bool:
    """Whether two sets of required branches can never both be chosen: one exclusive group must choose one output
    for ``a`` and another for ``b``."""
    return any(x[:2] == y[:2] and x.output != y.output for x in a for y in b)


def _reaches(start: str, goal: str, downstream: Mapping[str, list[str]], within: set[str]) -> bool:
    """Whether ``goal`` is downstream of ``start``, following only edges through the nodes in ``within``."""
    seen: set[str] = set()
    stack = list(downstream[start])
    while stack:
        n = stack.pop()
        if n == goal:
            return True
        if n in within and n not in seen:
            seen.add(n)
            stack.extend(downstream[n])
    return False
