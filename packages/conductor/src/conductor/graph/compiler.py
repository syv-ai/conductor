"""How a ``CompiledGraph`` is built: one graph compiled, pass by pass.

``CompiledGraph.from_graph(graph, registry)`` is the door; it builds a
``_Compilation`` and asks it to ``build``. Pure: no session, no I/O, no
loading. Every definition the graph names must already be in the
registry; a host that had to load one built it and called
``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` on the
result rather than raising, so an editor can show a half-finished graph
with its faults marked. What raises is a caller asking something no
graph can produce — an input a node does not have, a node compile could
not resolve.

The passes each read the one before, through the state ``_Compilation``
holds. A node that fails a pass carries a fatal ``Problem`` and drops out
of the passes after it, so nothing is guessed about a node downstream of
a fault.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor.dtype_ref import name_of
from conductor.errors import Refuses
from conductor.graph.binding import Edges, static_values
from conductor.graph.compiled import CompiledGraph
from conductor.graph.conditions import Condition, conditions_of
from conductor.graph.expand import SEPARATOR, Expansion, authored_ref, expand, surfaced
from conductor.graph.iteration import Iteration, derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, field_problems, lock_problems
from conductor.interface import Interface, model_of
from conductor.metadata import Input
from conductor.node import GraphVersion, NodeDefinition, NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Series


class _Compilation:
    """One graph being compiled: the state every pass reads and writes.

    ``build`` runs the passes in order and returns the ``CompiledGraph``.
    Each pass is a method that reads what the passes before it left on
    ``self`` and appends what it finds wrong to ``problems``. Nothing here
    outlives ``build``; the result is the immutable record.
    """

    def __init__(self, graph: Graph, registry: NodeRegistry) -> None:
        self.graph = graph
        self.registry = registry
        self.problems: list[Problem] = []
        #: Every id the author wrote, refused ones included: a problem on one
        #: of these is the author's as it stands.
        self.graph_ids: frozenset[str] = frozenset(node.id for node in graph.nodes)
        #: The nodes the author placed, by id (pass 1).
        self.authored: dict[str, GraphNode] = {}
        #: The version each authored node uses (pass 2).
        self.pinned: dict[str, NodeVersion | GraphVersion] = {}
        #: Who waits for whom, and an execution order, over the authored graph (pass 3).
        self.authored_dependencies: dict[str, frozenset[str]] = {}
        self.authored_order: tuple[str, ...] = ()
        #: The expanded graph: every embedded graph inlined (pass 4).
        self.expansion: Expansion
        #: Each node's inputs and outputs, and the values the author typed, by
        #: expanded id (pass 5; the walk over the edges completes them).
        self.interfaces: dict[str, Interface] = {}
        self.statics: dict[str, dict[str, Any]] = {}
        #: Per node, the scalar inputs where the author typed many values —
        #: three files, a list of texts — which the node runs once per value of.
        self.listed: dict[str, frozenset[str]] = {}
        #: Nodes whose stored edges are wrong; the edge walk leaves them out (pass 6).
        self.broken: frozenset[str] = frozenset()
        #: What the walk over the edges decided (pass 7).
        self.iteration: Iteration
        #: The condition under which each output appears (pass 9).
        self.output_conditions: dict[Ref, Condition] = {}
        #: What the graph takes and returns (pass 10).
        self.graph_interface: Interface

    def build(self) -> CompiledGraph:
        """The passes, in order. Each reads the one before:

        1. ``place`` — the nodes by id; two with one id is a problem.
        2. ``pin`` — the version each node uses; an unknown type or version
           is a problem on the node.
        3. ``order`` — who waits for whom and an execution order, over the
           graph as authored; a cycle is a problem.
        4. ``expand`` — every node whose version is a graph is inlined as
           its inner nodes.
        5. ``ask_inputs`` — each node's inputs as its hook answers, with the
           values the author typed read through their declared types.
        6. ``check_bindings`` — the stored bindings against those inputs, and
           the inputs themselves against the two field rules.
        7. ``walk_edges`` — one walk over the edges that types every field,
           decides which nodes run once per row and completes the outputs,
           checking each node's outputs against the field rules as it does.
        8. ``conditions`` — the condition under which each output appears.
        9. ``interface`` — what each embedded graph, and then the graph,
           takes and returns.

        A problem found inside an embedded graph is reported on the node
        the author placed.
        """
        self.place()
        self.pin()
        self.order()
        self.expand()
        self.ask_inputs()
        self.check_bindings()
        self.walk_edges()
        self.conditions()
        self.interface()
        expansion, iteration = self.expansion, self.iteration
        return CompiledGraph(
            _graph=self.graph,
            _registry=self.registry,
            _nodes=expansion.nodes,
            _versions={**expansion.versions, **expansion.placement_versions},
            _interfaces=self.interfaces,
            _statics=self.statics,
            _call_models={
                node_id: model_of(self.interfaces[node_id].inputs)
                for node_id in expansion.nodes if node_id in iteration.iterated
            },
            _order=expansion.order,
            _iterated=iteration.iterated,
            _indexes=iteration.indexes,
            _types=iteration.types,
            _receives=iteration.receives,
            _conditions=self.output_conditions,
            _placements=frozenset(expansion.placement_versions),
            _placement_of=expansion.placement_of,
            interface=self.graph_interface,
            problems=tuple(surfaced(found, expansion.nodes, self.graph_ids) for found in self.problems),
        )

    # -- the passes -------------------------------------------------------------

    def place(self) -> None:
        """Every node the author placed, by id. Two nodes with one id is a
        fatal problem: the second would silently shadow the first everywhere
        else. An id holding ``/`` is one too: compile names an embedded
        graph's nodes with it, and an authored ``e/holder`` beside a
        placement ``e`` would be read in the inner node's place."""
        for node in self.graph.nodes:
            if SEPARATOR in node.id:
                self.problems.append(problem("invalid_node_id", node.id))
                continue
            if node.id in self.authored:
                self.problems.append(problem("duplicate_node_id", node.id))
                continue
            self.authored[node.id] = node

    def pin(self) -> None:
        """The version each node uses, looked up once, here.

        A node stores a ``type`` and a ``version`` number; ``registry[type]``
        gives the definition and ``definition.versions[version]`` the version
        record. A stored graph can name a type the catalog has since lost or a
        version the class has since dropped, so either miss is a problem on
        the node rather than an error. A version may be a ``GraphVersion`` —
        an embedded graph — which ``expand`` inlines.
        """
        for node in self.authored.values():
            if node.type not in self.registry:
                self.problems.append(problem("unknown_node_type", node.id, node_type=node.type))
                continue
            version = self.registry[node.type].versions.get(node.version)
            if version is None:
                self.problems.append(
                    problem("unknown_node_version", node.id, node_type=node.type, version=node.version)
                )
                continue
            self.pinned[node.id] = version

    def order(self) -> None:
        """Who waits for whom, read off the authored edges, and an execution
        order over it. A node on a cycle is left out of the order and gets a
        fatal problem."""
        self.authored_dependencies = dependencies_of(self.authored.values())
        self.authored_order, cyclic = order_of(self.authored_dependencies)
        self.problems.extend(problem("cycle", node_id) for node_id in sorted(cyclic))

    def expand(self) -> None:
        """Every node whose version is a graph, inlined as its inner nodes
        under its name (``expand``). From here on the passes see the
        expanded graph."""
        self.expansion = expand(self.authored, self.authored_order, self.pinned, self.registry)
        self.problems.extend(self.expansion.problems)

    def ask_inputs(self) -> None:
        """Ask each node which inputs it has, once, on a fresh instance.

        A node's inputs may depend on the values the author typed into it (a
        mode dropdown that adds fields), so ``compute_inputs`` is called with
        those values, each read through its declared type by ``_typed_statics``
        and laid over the declaration's defaults, so a hook that reads a table's
        columns or a schema's fields gets the typed value and never parses JSON
        itself. That reading is the hook's argument and nothing more: once the
        hook has answered, every value is read once, against the inputs the
        hook returned — a hook may retype an input, and the value the author
        typed has to be read as that type — and that is what ``statics`` and
        ``listed`` keep, and where ``invalid_static`` is reported. Both hold
        only what the author typed: a default is the declaration's and applies
        wherever nothing is bound. Only the inputs are asked here. The outputs
        stay as declared until the walk over the edges, which can tell
        ``compute_outputs`` what type arrives on each connected input.

        A hook that cannot answer for these values raises ``Refuses``; its
        code and message are the node's one fatal ``Problem``, and the node is
        left out of every later pass, as a node whose type is unknown is.

        On a version that takes ``**inputs``, every connected name that is not
        a declared parameter becomes an ``Input`` of its own: typed ``Any`` or
        ``Series[Any]`` until the edge walk types it, titled by its name.

        Nothing else is asked of the node. Two inputs sharing a name, or a
        handle without a type an edge can carry, is ``check_bindings``'s to
        report.
        """
        for node_id, version in self.expansion.versions.items():
            node = self.expansion.nodes[node_id]
            instance = self.registry[node.type]()
            defaults = {i.name: i.default for i in version.interface.inputs if i.optional}
            for_hook, _, _ = self._typed_statics(version.interface.inputs, node)
            try:
                inputs = instance.compute_inputs(version.interface.inputs, {**defaults, **for_hook})
            except Refuses as refusal:
                self.problems.append(Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node_id))
                continue
            typed, listed, invalid = self._typed_statics(inputs, node)
            self.problems.extend(invalid)
            if version.interface.open is not None:
                # One Input per edge: `Series[Any]` when the node reduces each
                # ("series"), `Any` when it receives each whole ("single"). The
                # edge walk types it from what arrives.
                shape = Series[Any] if version.interface.open == "series" else Any
                named = {i.name for i in inputs}
                inputs = (*inputs, *(
                    Input(name=name, dtype=shape, title=name)
                    for name, binding in node.bindings.items()
                    if name not in named and isinstance(binding, Edges)
                ))
            self.interfaces[node_id] = replace(version.interface, inputs=inputs)
            self.statics[node_id] = typed
            self.listed[node_id] = listed

    def check_bindings(self) -> None:
        """Check the stored bindings against the inputs each node actually has, and those inputs against the field rules.

        Reports a lock on a field the node does not have (``unknown_locked_field``,
        not fatal), a binding on a field the node does not have (``stale_binding``:
        fatal, a typo that would otherwise run with the default, unless the
        node's ``compute_inputs`` makes its fields come and go), an edge into an input that cannot be connected
        (``show_handle=False``), an edge from a node that does not exist, an
        edge into a field an embedded graph does not have, and a required
        input nothing binds (``unbound_required``). A parameter typed ``Any``
        gets its type from its edge and nothing else, so with no edge it is
        ``unbound_required`` as well. Two inputs sharing a name, or a handle
        without a type an edge can carry (``field_problems``), are checked
        here on the inputs the hook answered, before the walk reads them.

        An edge from an id the author wrote but compile could not resolve —
        an unknown type, a cycle, a refused id — is left alone: that node
        carries its own fatal problem, and "not in the graph" would be
        untrue. Leaves in ``broken``
        the ids of the nodes whose edges or inputs are wrong; the edge walk
        leaves them out, since nothing can be derived from a broken edge.
        """
        nodes = self.expansion.nodes
        broken: set[str] = set()
        self.problems.extend(lock_problems(nodes, self.interfaces))
        for node_id, interface in self.interfaces.items():
            node = nodes[node_id]
            declared = {i.name: i for i in interface.inputs}
            faults = field_problems(node_id, interface.inputs, untyped_ok=True)
            if faults:
                broken.add(node_id)
                self.problems.extend(faults)

            for name, binding in node.bindings.items():
                if name not in declared:
                    dead = problem("stale_binding", node_id, name, inputs=", ".join(sorted(declared)) or "none")
                    if self.registry[node.type].compute_inputs is not NodeDefinition.compute_inputs:
                        dead = dead.model_copy(update={"fatal": False})
                    self.problems.append(dead)
                    continue
                if not isinstance(binding, Edges):
                    continue
                target = declared[name]
                if not target.show_handle:
                    broken.add(node_id)
                    self.problems.append(problem("edge_into_closed_handle", node_id, name))
                for ref in binding.refs:
                    if ref.node_id in nodes:
                        continue  # a node of the run; the walk checks that it has the output
                    broken.add(node_id)
                    source = authored_ref(ref)
                    if source.node_id in self.expansion.placement_versions:
                        # An embedded graph the author placed, but no inner node of that name.
                        self.problems.append(problem("unknown_ref_output", node_id, name, source=str(source)))
                    elif ref.node_id not in self.graph_ids:
                        self.problems.append(problem("unknown_ref_node", node_id, name, source_node=ref.node_id))
                    # else: an id the author wrote that compile could not resolve; it carries its own fatal problem

            for inp in interface.inputs:
                if inp.dtype is Any:
                    if not isinstance(node.bindings.get(inp.name), Edges):
                        broken.add(node_id)
                        self.problems.append(problem("unbound_required", node_id, inp.name))
                elif not inp.optional and inp.name not in node.bindings:
                    broken.add(node_id)
                    self.problems.append(problem("unbound_required", node_id, inp.name))
        self.broken = frozenset(broken)

    def walk_edges(self) -> None:
        """One walk over the edges in expanded order (``iteration.derive``):
        what type every field carries, which nodes run once per row and on
        which index, and each node's completed outputs. Nodes whose edges
        are broken are left out."""
        expansion = self.expansion
        self.iteration = derive(
            [
                expansion.nodes[node_id]
                for node_id in expansion.order
                if node_id in self.interfaces and node_id not in self.broken
            ],
            self.interfaces, expansion.versions, self.registry, self.statics, self.listed,
            placement_of=expansion.placement_of, members=expansion.members,
        )
        self.problems.extend(self.iteration.problems)
        self.interfaces = {**self.interfaces, **self.iteration.interfaces}

    def conditions(self) -> None:
        """The condition under which each output appears, from the ``choice``
        groups of the nodes that run once (``conditions_of``)."""
        expansion, iteration = self.expansion, self.iteration
        self.output_conditions = conditions_of(
            [expansion.nodes[node_id] for node_id in expansion.order if node_id in iteration.iterated],
            self.interfaces, iteration.iterated,
        )

    def interface(self) -> None:
        """What each embedded graph takes and returns, and then what the graph does.

        Both are read off the nodes (``derive_interface``): an embedded
        graph's off its inner nodes as the walk completed them, innermost
        first so a graph embedding another reads the derived interface of
        the one inside. From here on a node whose version is a graph answers
        ``CompiledGraph.node(...).interface`` with that derived interface,
        not the one the host declared on its ``GraphVersion``. The
        declaration is a host's copy of the same fact, so a field it names
        that the graph lacks, or names with another type, is reported on the
        placement as ``graph_interface_mismatch`` — not fatal, since what
        runs is the graph's.
        """
        for placement, version in reversed(self.expansion.placement_versions.items()):
            inner_ids = {inner.id: f"{placement}{SEPARATOR}{inner.id}" for inner in version.graph}
            derived = derive_interface(
                Graph(nodes=version.graph),
                {inner: self.interfaces[expanded] for inner, expanded in inner_ids.items() if expanded in self.interfaces},
                dependencies_of(version.graph),
            )
            self.problems.extend(self._interface_mismatches(placement, version.interface, derived))
            self.interfaces[placement] = derived
        self.graph_interface = derive_interface(
            self.graph,
            {n: self.interfaces[n] for n in self.authored if n in self.interfaces},
            self.authored_dependencies,
        )

    @staticmethod
    def _interface_mismatches(placement: str, declared: Interface, derived: Interface) -> list[Problem]:
        """Where the host's declaration of an embedded graph's interface
        disagrees with the graph: a field the graph lacks, or one it types
        differently. Fields the graph has and the declaration omits are not
        reported; the derived interface simply has them."""
        actual = {str(f.name): f.dtype for f in (*derived.inputs, *derived.outputs)}
        found: list[Problem] = []
        for field in (*declared.inputs, *declared.outputs):
            name = str(field.name)
            if name not in actual:
                found.append(problem("graph_interface_mismatch", placement, name, declared=name_of(field.dtype), actual="nothing"))
            elif actual[name] is not field.dtype:
                found.append(problem(
                    "graph_interface_mismatch", placement, name, declared=name_of(field.dtype), actual=name_of(actual[name]),
                ))
        return found

    # -- the values the author typed --------------------------------------------

    def _typed_statics(self, inputs: tuple[Any, ...], node: GraphNode) -> tuple[dict[str, Any], frozenset[str], list[Problem]]:
        """Every value the author typed into ``node``, read through its field's
        type as ``inputs`` states it; which of the scalar inputs hold many
        values; and the values no type could read.

        A stored ``Static`` holds JSON — a file comes back as a dict, an
        authored schema as a list — and every reader downstream wants the
        value, not its JSON form, so each is converted here. A sequence the
        type cannot read as one value is read as a sequence of values, which
        keeps a list-shaped scalar (a schema, a list of tags) one value while
        three uploaded files are three rows. Which of the two happened is
        decided by which reading succeeded, and returned as the set of
        inputs holding many — never again from the shape of the value. A
        value the type cannot read at all is a fatal ``invalid_static``,
        carrying whatever the type's constructor said; the caller decides
        whether to report it (``ask_inputs`` reads once for the hook's
        argument and once, reported, against the hook's answer). A value on
        a parameter typed ``Any`` is skipped: only an edge can give that
        parameter a type, and ``check_bindings`` reports it as unbound.
        """
        declared = {i.name: i for i in inputs}
        typed: dict[str, Any] = {}
        listed: set[str] = set()
        invalid: list[Problem] = []
        for name, value in static_values(node.bindings).items():
            inp = declared.get(name)
            if inp is None or inp.dtype is Any:
                continue  # a field the node lacks, or one only an edge can type; check_bindings reports both
            try:
                typed[name], many = self._typed_static(inp, value)
            except (ValidationError, TypeError, ValueError) as unreadable:
                said = _what_the_type_said(unreadable)
                invalid.append(problem("invalid_static", node.id, name, **({"reason": said} if said else {})))
                continue
            if many:
                listed.add(name)
        return typed, frozenset(listed), invalid

    @staticmethod
    def _typed_static(inp: Any, value: Any) -> tuple[Any, bool]:
        """``value`` read through ``inp``'s declared type, and whether it is
        many values: a sequence element by element for a ``Series[X]`` input
        (one series, not many); one value, or — when the type refuses the
        whole — a sequence of values, for anything else."""
        element = getattr(inp.dtype, "element", None)
        if element is not None:
            # A Series[X] input takes the whole sequence, each element typed.
            if not _sequence(value):
                raise TypeError("a Series input takes a sequence")
            return [_adapter(element).validate_python(v) for v in value], False
        try:
            return _adapter(inp.dtype).validate_python(value), False
        except ValidationError:
            if _sequence(value):
                return [_adapter(inp.dtype).validate_python(v) for v in value], True
            raise


def _sequence(value: Any) -> bool:
    """Is ``value`` a sequence of values, as a list widget or a multi-upload holds one? Text and bytes are one value each."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


@cache
def _adapter(dtype: type) -> TypeAdapter[Any]:
    return TypeAdapter(dtype)


def _what_the_type_said(invalid: Exception) -> str:
    """The sentence the type's own constructor raised, if it raised one.

    A type's constraints live in its constructor, and a constructor
    written for people says something specific — "the field must not be
    empty", "every row needs three cells". That sentence is the one thing
    that tells an author what to change, so it is appended to the generic
    ``invalid_static`` message. pydantic keeps a validator's ``ValueError``
    under ``ctx["error"]``; anything pydantic says on its own is about
    JSON shape and is not shown.
    """
    if isinstance(invalid, ValidationError):
        errors = invalid.errors()
        cause = errors[0].get("ctx", {}).get("error") if errors else None
        return str(cause) if isinstance(cause, ValueError) else ""
    return str(invalid) if isinstance(invalid, ValueError) else ""
