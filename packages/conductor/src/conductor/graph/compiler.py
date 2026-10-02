"""How a graph is compiled: a fixed list of steps, one after another, over one graph.

``CompiledGraph.from_graph(graph, registry)`` is the door; it runs a
``Compilation`` and builds the ``CompiledGraph`` from what the steps
recorded. Pure: no session, no I/O, no loading. Every definition the graph
names must already be in the registry; a host that had to load one built
it and called ``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` rather than
raising, so an editor can show a half-finished graph with its faults
marked.

The steps each read the one before, through the state ``Compilation``
holds. A node that fails a step carries a fatal ``Problem`` and drops out
of the steps after it, so nothing is guessed about a node downstream of
a fault. This module knows nothing of ``CompiledGraph``: the result is
built on the other side, in ``conductor.graph.compiled``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor.dtype_ref import name_of
from conductor.errors import Refuses
from conductor.graph.binding import From, static_values
from conductor.graph.conditions import Condition, conditions_of
from conductor.graph.expand import (
    ADDRESS_KEYS,
    SEPARATOR,
    Expansion,
    authored_address,
    authored_ref,
    expand,
    resolved,
)
from conductor.graph.iteration import Iteration, derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, field_problems, lock_problems
from conductor.interface import Interface
from conductor.metadata import Input
from conductor.node import GraphVersion, NodeDefinition, NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Series


class Compilation:
    """One graph being compiled: the state every step reads and writes.

    ``run`` runs the steps in order. Each step is a method that reads what
    the steps before it left on ``self`` and records what it finds wrong in
    ``problems``, as the author sees it (``record_as_the_author_sees_it``).
    ``CompiledGraph.from_graph`` makes one, runs it and reads what it left
    into the stored ``CompiledNode`` and ``CompiledField`` values; nothing
    here outlives that.
    """

    def __init__(self, graph: Graph, registry: NodeRegistry) -> None:
        self.graph = graph
        self.registry = registry
        self.problems: list[Problem] = []
        #: Every id the author wrote, refused ones included: an edge from one
        #: of these reads a node the author wrote, not a missing one.
        self.graph_ids: frozenset[str] = frozenset(node.id for node in graph.nodes)
        #: The nodes the author placed, by id (set by ``place``).
        self.authored: dict[str, GraphNode] = {}
        #: The class each authored node resolved to and the version it uses (set by ``pin``).
        self.pinned: dict[str, tuple[type[NodeDefinition], NodeVersion | GraphVersion]] = {}
        #: Who waits for whom, and an execution order, over the authored graph (set by ``order``).
        self.authored_dependencies: dict[str, frozenset[str]] = {}
        self.authored_order: tuple[str, ...] = ()
        #: The expanded graph: every embedded graph inlined (set by ``expand``).
        self.expansion: Expansion
        #: Each node's inputs and outputs, and the values the author typed, by
        #: expanded id (set by ``ask_inputs``; ``walk_edges`` completes them).
        self.interfaces: dict[str, Interface] = {}
        self.statics: dict[str, dict[str, Any]] = {}
        #: Per node, the scalar inputs where the author typed many values —
        #: three files, a list of texts — which the node runs once per value of.
        self.listed: dict[str, frozenset[str]] = {}
        #: Nodes whose stored edges are wrong; the edge walk leaves them out (set by ``check_bindings``).
        self.broken: frozenset[str] = frozenset()
        #: What the walk over the edges decided (set by ``walk_edges``).
        self.iteration: Iteration
        #: The condition under which each output appears (set by ``conditions``).
        self.output_conditions: dict[Ref, Condition] = {}
        #: What the graph takes and returns (set by ``interface``).
        self.graph_interface: Interface

    def run(self) -> None:
        """The steps, in order. Each reads the one before:

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

        Every step records its problems as the author sees them, so a
        problem found inside an embedded graph is already on the node the
        author placed when it is recorded, and nothing is rewritten later.
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

    def record_as_the_author_sees_it(self, *found: Problem) -> None:
        """Record problems on the graph as the author built it, not as compile expanded it.

        Compile works on the expanded graph, where an embedded graph's inner
        nodes stand in its place under ids like ``approve/check``. The author
        never sees those ids: they see the node they placed, ``approve``. So
        a problem found on an inner node is recorded on that placed node,
        with the inner node's address as its field (``check.amount``, or
        ``check`` for a problem on the whole inner node), and its message
        starts with the inner node's title. Its code stays the inner
        problem's, and so do its ``details``, under the same keys: every
        address in them is rewritten the same way (``approve/check.amount``
        becomes ``approve.check.amount``), and two keys are added beside
        them, ``placement`` (the inner node's title; hosts read the key by
        that name) and ``inner_message``.
        A host that translates problems by code finds the same keys inside
        an embedded graph as outside one. A problem on a node the author
        placed is recorded as it is.

        Every step records through here except ``place``. Its problems are
        about ids exactly as the author wrote them, and an id the author
        wrote with a ``/`` in it — refused there — may look like an inner
        node's; it must stay the author's own node.
        """
        for found_problem in found:
            if SEPARATOR not in found_problem.node_id:
                self.problems.append(found_problem)
                continue
            outer_node_id, inner_id = found_problem.node_id.split(SEPARATOR, 1)
            inner_address = inner_id.replace(SEPARATOR, ".")
            inner_node = self.expansion.nodes.get(found_problem.node_id)
            title = (inner_node.title if inner_node is not None else None) or inner_address
            self.problems.append(found_problem.model_copy(update={
                "node_id": outer_node_id,
                "field": (
                    authored_ref(Ref(found_problem.node_id, found_problem.field)).field
                    if found_problem.field else inner_address
                ),
                "message": f"In '{title}': {found_problem.message}",
                "details": {
                    **{
                        key: authored_address(value) if key in ADDRESS_KEYS else value
                        for key, value in found_problem.details.items()
                    },
                    "placement": title,
                    "inner_message": found_problem.message,
                },
            }))

    # -- the steps --------------------------------------------------------------

    def place(self) -> None:
        """Every node the author placed, by id. Two nodes with one id is a
        fatal problem: the second would silently shadow the first everywhere
        else. An id holding ``/`` is one too: compile names an embedded
        graph's nodes with it, and an authored ``e/holder`` beside a graph
        node ``e`` would be read in the inner node's place."""
        for node in self.graph.nodes:
            if SEPARATOR in node.id:
                self.problems.append(problem("invalid_node_id", node.id))
                continue
            if node.id in self.authored:
                self.problems.append(problem("duplicate_node_id", node.id))
                continue
            self.authored[node.id] = node

    def pin(self) -> None:
        """The class and version each authored node uses, resolved once, here (``resolved``).

        A node stores a ``type`` and a ``version`` number; the registry gives
        the class and the class its version record. A stored graph can name
        a type the registry has since lost or a version the class has since
        dropped, so either miss is a problem on the node rather than an
        error. A version may be a ``GraphVersion`` — an embedded graph —
        which ``expand`` inlines, resolving its inner nodes the same way.
        """
        for node in self.authored.values():
            found = resolved(node, self.registry)
            if isinstance(found, Problem):
                self.record_as_the_author_sees_it(found)
            else:
                self.pinned[node.id] = found

    def order(self) -> None:
        """Who waits for whom, read off the authored edges, and an execution
        order over it. A node on a cycle is left out of the order and gets a
        fatal problem."""
        self.authored_dependencies = dependencies_of(self.authored.values())
        self.authored_order, cyclic = order_of(self.authored_dependencies)
        self.record_as_the_author_sees_it(*(problem("cycle", node_id) for node_id in sorted(cyclic)))

    def expand(self) -> None:
        """Every node whose version is a graph, inlined as its inner nodes
        under its name (``expand``). From here on the steps see the
        expanded graph."""
        self.expansion = expand(self.authored, self.authored_order, self.pinned, self.registry)
        self.record_as_the_author_sees_it(*self.expansion.problems)

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
        for node_id, node in self.expansion.nodes.items():
            version = self.expansion.versions[node_id]
            instance = self.expansion.definitions[node_id]()
            defaults = {i.name: i.default for i in version.interface.inputs if i.optional}
            for_hook, _, _ = self._typed_statics(version.interface.inputs, node)
            try:
                inputs = instance.compute_inputs(version.interface.inputs, {**defaults, **for_hook})
            except Refuses as refusal:
                self.record_as_the_author_sees_it(Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node_id))
                continue
            typed, listed, invalid = self._typed_statics(inputs, node)
            self.record_as_the_author_sees_it(*invalid)
            if version.interface.open is not None:
                # One Input per edge: `Series[Any]` when the node reduces each
                # ("series"), `Any` when it receives each whole ("single"). The
                # edge walk types it from what arrives.
                shape = Series[Any] if version.interface.open == "series" else Any
                named = {i.name for i in inputs}
                inputs = (*inputs, *(
                    Input(name=name, dtype=shape, title=name)
                    for name, binding in node.bindings.items()
                    if name not in named and isinstance(binding, From)
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
        input nothing binds (``unbound_required``). A parameter typed ``Any``,
        or ``Series[Any]``, gets its type from its edge and nothing else, so
        with no edge it is ``unbound_required`` as well, and its node can't be
        worked out. Two inputs sharing a name, or a handle
        without a type an edge can carry (``field_problems``), and an input
        named with a leading underscore (``parameter_name_invalid``), are
        checked here on the inputs the hook answered, before the walk reads
        them.

        An edge from an id the author wrote but compile could not resolve —
        an unknown type, a cycle, a refused id — is left alone: that node
        carries its own fatal problem, and "not in the graph" would be
        untrue. Leaves in ``broken``
        the ids of the nodes whose edges or inputs are wrong; the edge walk
        leaves them out, since nothing can be derived from a broken edge.
        A typed required input left empty is fatal but does not break its
        node: the walk types it from its declaration, so the node and
        everything after it are still worked out.
        """
        nodes = self.expansion.nodes
        broken: set[str] = set()
        self.record_as_the_author_sees_it(*lock_problems(nodes, self.interfaces))
        for node_id, interface in self.interfaces.items():
            node = nodes[node_id]
            declared = {i.name: i for i in interface.inputs}
            faults = field_problems(node_id, interface.inputs, untyped_ok=True)
            # A leading underscore is private by convention and never a field
            # a call can carry: a declared parameter is refused at class
            # definition, and a ``**inputs`` edge or a hook's input here.
            faults += [problem("parameter_name_invalid", node_id, i.name) for i in interface.inputs if i.name.startswith("_")]
            if faults:
                broken.add(node_id)
                self.record_as_the_author_sees_it(*faults)

            for name, binding in node.bindings.items():
                if name not in declared:
                    dead = problem("stale_binding", node_id, name, inputs=", ".join(sorted(declared)) or "none")
                    if self.expansion.definitions[node_id].compute_inputs is not NodeDefinition.compute_inputs:
                        dead = dead.model_copy(update={"fatal": False})
                    self.record_as_the_author_sees_it(dead)
                    continue
                if not isinstance(binding, From):
                    continue
                target = declared[name]
                if not target.show_handle:
                    broken.add(node_id)
                    self.record_as_the_author_sees_it(problem("edge_into_closed_handle", node_id, name))
                for ref in binding.refs:
                    if ref.node_id in nodes:
                        continue  # a node of the run; the walk checks that it has the output
                    broken.add(node_id)
                    source = authored_ref(ref)
                    if source.node_id in self.expansion.graphs:
                        # An embedded graph the author placed, but no inner node of that name.
                        self.record_as_the_author_sees_it(problem("unknown_ref_output", node_id, name, source=str(source)))
                    elif ref.node_id not in self.graph_ids:
                        self.record_as_the_author_sees_it(problem("unknown_ref_node", node_id, name, source_node=ref.node_id))
                    # else: an id the author wrote that compile could not resolve; it carries its own fatal problem

            for inp in interface.inputs:
                if inp.dtype is Any or inp.dtype is Series[Any]:
                    if not isinstance(node.bindings.get(inp.name), From):
                        broken.add(node_id)
                        self.record_as_the_author_sees_it(problem("unbound_required", node_id, inp.name))
                elif not inp.optional and inp.name not in node.bindings:
                    # Not broken: its type is declared, so the walk still works the node out, as a value that arrives at run time.
                    self.record_as_the_author_sees_it(problem("unbound_required", node_id, inp.name))
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
            self.interfaces, expansion.versions, expansion.definitions, self.statics, self.listed,
            members=expansion.members,
        )
        self.record_as_the_author_sees_it(*self.iteration.problems)
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
        graph node as ``graph_interface_mismatch`` — not fatal, since what
        runs is the graph's. A graph node comes before its inner ones in
        ``graphs``, so the reverse is innermost first.
        """
        for outer, version in reversed(self.expansion.graphs.items()):
            inner_ids = {inner.id: f"{outer}{SEPARATOR}{inner.id}" for inner in version.graph}
            derived = derive_interface(
                Graph(nodes=version.graph),
                {inner: self.interfaces[expanded] for inner, expanded in inner_ids.items() if expanded in self.interfaces},
                dependencies_of(version.graph),
            )
            self.record_as_the_author_sees_it(*self._interface_mismatches(outer, version.interface, derived))
            self.interfaces[outer] = derived
        self.graph_interface = derive_interface(
            self.graph,
            {n: self.interfaces[n] for n in self.authored if n in self.interfaces},
            self.authored_dependencies,
        )

    @staticmethod
    def _interface_mismatches(outer: str, declared: Interface, derived: Interface) -> list[Problem]:
        """Where the host's declaration of an embedded graph's interface
        disagrees with the graph: a field the graph lacks, or one it types
        differently. Fields the graph has and the declaration omits are not
        reported; the derived interface simply has them."""
        actual = {str(f.name): f.dtype for f in (*derived.inputs, *derived.outputs)}
        found: list[Problem] = []
        for field in (*declared.inputs, *declared.outputs):
            name = str(field.name)
            if name not in actual:
                found.append(problem("graph_interface_mismatch", outer, name, declared=name_of(field.dtype), actual="nothing"))
            elif actual[name] is not field.dtype:
                found.append(problem(
                    "graph_interface_mismatch", outer, name, declared=name_of(field.dtype), actual=name_of(actual[name]),
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
