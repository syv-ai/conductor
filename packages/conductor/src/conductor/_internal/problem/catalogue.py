"""Every kind of problem conductor's core reports, one function each, in ``PROBLEMS``.

A kind is a function under ``@PROBLEMS.problem()``. Its name is the problem's code. Its positional-only parameters
(before the ``/``) and its ``*loc`` are the parts of ``loc``, and a named one is a key of ``ctx`` too; its keyword-only
parameters are the other keys of ``ctx``.
It returns the message. Calling it makes the problem::

    unknown_input("upper", "txet", inputs=["text"])
    # Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'; it has text.",
    #         ctx={"node_id": "upper", "field": "txet", "inputs": ["text"]})

A type checker checks every call against the function's signature; ``ProblemCatalogue.problem`` in ``type.py`` gives
the rules for the signature in full. A few kinds name their code in the decorator,
``@PROBLEMS.problem("type_mismatch", variant="make_a_series")``: a second message under that code, for one particular
mistake, and the problem's ``ctx.variant`` says which. ``of_a_hook=True`` marks a problem in a node's ``compute_...``
hook, which the node's developer fixes rather than the graph's author; ``check_registry`` reports those as its own.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``): what is wrong and
how to fix it, a line for each ``loc`` and ``ctx`` parameter, and an example. The example runs after the setup at
the top of that page (``conductor/about/problems.md``), which makes a ``registry`` of small nodes (``upper``,
``greet``, ``count``, ``words``, ``join``, ``route``, ``approve``, ``who``, ``pick``, ``contacts``) and three
helpers: ``problems(record)`` returns the problems of a compiled graph record, ``refused(call)`` returns the
problems a refused call raises, and ``why_it_stopped(run)`` returns the problem a run ended on. Its last line gives
the problems, and a test checks they are of this kind and no other.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor._internal.base import comma_separated
from conductor._internal.problem.type import ProblemCatalogue

#: Every kind of problem conductor's core reports.
PROBLEMS = ProblemCatalogue()


PROBLEMS.section(
    "Compiling a graph",
    "In `graph.problems`; compiling never raises for them, and a graph with problems does not run. `loc` is "
    "`(node_id,)` or `(node_id, input)`. For a node inside a subgraph, the subgraph node's id comes first: "
    '`("loud", "shout", "line")`.',
)


@PROBLEMS.problem()
def alternatives_from_different_loops(node_id: str, field: str, /, *, series: list[str | None]) -> str:
    """Two edges into one input come from branches whose values are inside different loops.

    In the example, ``u.text`` has an edge from each branch of ``r``. The long branch gives one value per item of
    ``w.result``. The short branch goes through ``letters``, which returns a series of its own, so it gives one value
    per item of ``letters.result``. A call of ``u`` is for an item of one loop and can have no value from the other.
    Collect the inner loop first, in a node with a ``Series`` input, or wire the input from one branch only.

    Args:
        node_id: The node.
        field: The input.
        series: The series each edge's values are items of, by address; ``None`` for an edge from a node inside no
            loop.

    Example:
        problems({
            "nodes": [
                {"id": "w", "node": "words"}, {"id": "r", "node": "route"},
                {"id": "letters", "node": "words"}, {"id": "u", "node": "upper"},
            ],
            "edges": [
                ["w.result", "r.text"], ["r.short", "letters.text"],
                ["letters.result", "u.text"], ["r.long", "u.text"],
            ],
        })
    """
    series_text = comma_separated(["none (not in a loop)" if one is None else one for one in series])
    return (
        f"'{field}' has edges from branches inside different loops ({series_text}), so a call never has the other branch's value. "
        f"Collect the inner loop first, or wire '{field}' from one branch."
    )


@PROBLEMS.problem(of_a_hook=True)
def compute_hook_failed(*loc: str | int, reason: str, error: str, error_ctx: dict[str, Any] | None = None) -> str:
    """A node's ``compute_...`` hook raised, or returned something compiling cannot use.

    The hook's exception and its traceback are in the ``conductor`` log. A hook that raises ``NodeError`` gives its
    ``code`` as ``ctx.error`` (``failed`` when it gives none, ``external_failed`` for an ``ExternalError``) and its
    ``ctx`` as ``ctx.error_ctx``; any other exception gives its class name. Change the constant the hook reads, or
    fix the hook. In the example, ``pick``'s hook raises ``ValueError`` for an empty text.

    Args:
        loc: The node.
        reason: The hook's message, or why compiling cannot use what it returned.
        error: The ``code`` of the hook's ``NodeError``, or the class name of the exception it raised.
        error_ctx: The ``ctx`` of the hook's ``NodeError``, when it has one.

    Example:
        problems({"nodes": [{"id": "pick", "node": "pick", "constants": {"names": ""}}]})
    """
    return f"The node could not work out its inputs and outputs: {reason.rstrip('.')}. Check the node's constants and what is wired to it."


@PROBLEMS.problem()
def context_does_not_fit(node_id: str, parameter: str, /, *, needs: str, context_type: str) -> str:
    """A node takes the run's context as a class that the registry's ``context_type`` is not.

    A node's ``Context[X]`` fits when ``X`` is the registry's class, a base of it, or a ``Protocol`` whose members the
    class has. Ask for one of those, or name another class on the registry. In the example, ``count`` asks for an
    ``int``.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, its first.
        needs: The name of the class inside ``Context[...]``.
        context_type: The name of the registry's class.

    Example:
        @registry.node("count_it", title="Count it", description="Returns the number the run is given.")
        def count_it(ctx: Context[int]) -> int:
            return ctx

        problems({"nodes": [{"id": "count", "node": "count_it"}]})
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry's context class {context_type} is not a {needs}; ask for {context_type}, a base of it or a Protocol it satisfies."


@PROBLEMS.problem("context_does_not_fit", variant="lacks")
def context_does_not_fit_lacks(node_id: str, parameter: str, /, *, needs: str, context_type: str, lacks: list[str]) -> str:
    """A node takes the run's context as a ``Protocol`` whose members the registry's ``context_type`` does not all have.

    Give the registry's class what the node reads, with the classes the Protocol declares, or change what the node asks
    for. In the example, ``HostContext`` has no ``database``.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, its first.
        needs: The name of the ``Protocol`` inside ``Context[...]``.
        context_type: The name of the registry's class.
        lacks: Each member the class does not have, as its name, or as ``"name as a X"`` when the class declares it
            as another class.

    Example:
        class HasDatabase(Protocol):
            @property
            def database(self) -> str: ...

        @registry.node("tables", title="Tables", description="Names the database it reads.")
        def tables(ctx: Context[HasDatabase]) -> str:
            return ctx.database

        problems({"nodes": [{"id": "tables", "node": "tables"}]})
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry's context class {context_type} lacks {comma_separated(lacks)}; give {context_type} what the node reads, or change what it asks for."


@PROBLEMS.problem("context_does_not_fit", variant="no_context_type")
def context_does_not_fit_no_context_type(node_id: str, parameter: str, /, *, needs: str) -> str:
    """A node takes the run's context, and the registry names no ``context_type``.

    Name the class of the object your runs get on the registry, ``NodeRegistry(context_type=HostContext)``. A
    registry of nodes that do not know the host, such as a package of nodes, names none; add it to yours with
    ``add_subregistry``, and its nodes are checked against your class.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, its first.
        needs: The name of the class inside ``Context[...]``.

    Example:
        plain = NodeRegistry()
        plain.add(who)
        Graph.compile(make_graph(who), plain).problems
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry names no context class; name one with NodeRegistry(context_type=...)."


@PROBLEMS.problem()
def cycle(*loc: str | int) -> str:
    """A node is on a cycle of edges.

    A graph calls each node after the nodes upstream of it, so a graph with a cycle cannot run. Remove one of the
    edges that close the cycle. Every node on it has this problem.

    Args:
        loc: The node.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
            "edges": [["a.result", "b.text"], ["b.result", "a.text"]],
        })
    """
    return "The node is part of a cycle, so the graph cannot run; remove one of the edges that close it."


@PROBLEMS.problem()
def default_not_a_graph_input(node_id: str, field: str, /) -> str:
    """The graph record gives a default to an input that is not a graph input.

    Only a graph input gets its value from the caller, so only there can a default be replaced. Write the value as a
    constant (``constants``) instead. In the example, ``g.greeting`` is not a graph input because ``g`` has an edge
    into it, so it is not one of the graph's first nodes. An input that a node's ``compute_added_inputs`` or
    ``compute_outputs`` reads is never a graph input either, on any node: the hook reads it when the graph is
    compiled, before any caller.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet", "defaults": {"greeting": "hi"}}],
            "edges": [["a.result", "g.name"]],
        })
    """
    return f"'{field}' is not a graph input, so no caller can replace its default; make it a constant."


@PROBLEMS.problem()
def duplicate_node_id(node_id: str, /) -> str:
    """Two nodes of the graph record have the same id.

    Give one of them another id. ``make_graph`` picks free ids itself: ``upper-1``, ``upper-2``.

    Args:
        node_id: The id.

    Example:
        problems({"nodes": [{"id": "a", "node": "upper"}, {"id": "a", "node": "count"}]})
    """
    return f"Two nodes have the id '{node_id}'; rename one."


@PROBLEMS.problem()
def edge_names_missing_node(*loc: str | int, missing: str, source: str, target: str) -> str:
    """An edge names a node the graph record does not have.

    Add the node, or remove the edge.

    Args:
        loc: The node at the edge's target, and the input.
        missing: The id of the node that is not in the graph record.
        source: The edge's source, as an address.
        target: The edge's target, as an address.

    Example:
        problems({"nodes": [{"id": "a", "node": "upper"}], "edges": [["a.result", "ghost.text"]]})
    """
    return f"The edge from '{source}' to '{target}' names '{missing}', which is not a node of the graph; add that node or remove the edge."


@PROBLEMS.problem()
def subgraph_contains_itself(*loc: str | int, node: str) -> str:
    """A subgraph contains itself, directly or through another subgraph.

    Remove the node that leads back. The problem is on that node, inside the subgraph, so its ``loc`` starts with the
    subgraph node's id: in the example, ``("i", "again")``.

    Args:
        loc: The node that leads back.
        node: The id of the subgraph it uses.

    Example:
        itself = subgraph("itself", title="Itself", description="A graph that contains itself.")(
            GraphRecord.model_validate({"nodes": [{"id": "again", "node": "itself"}]})
        )
        looping = NodeRegistry()
        looping.add(itself)
        Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "i", "node": "itself"}]}), looping).problems
    """
    return f"The subgraph '{node}' contains itself; remove the node that leads back to it."


@PROBLEMS.problem()
def hook_needs_value(node_id: str, field: str, /) -> str:
    """A node's ``compute_...`` hook reads an input that has no constant in the graph record.

    The node's inputs or outputs depend on that value, so it must be known when the graph is compiled. Give the input
    a constant. An input a hook reads is never a graph input.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": [{"id": "pick", "node": "pick"}]})
    """
    return f"'{field}' needs a constant: the node works out its inputs and outputs from it when the graph is compiled."


@PROBLEMS.problem()
def invalid_node_id(node_id: str, /) -> str:
    """A node's id contains ``.``, ``/``, ``@`` or ``~``.

    Those characters build addresses (``a.result``), node paths (``loud/shout``) and waiting ids (``a@1~3fa2``). Pick
    an id without them.

    Args:
        node_id: The id.

    Example:
        problems({"nodes": [{"id": "a.b", "node": "upper"}]})
    """
    return f"'{node_id}' cannot be a node id: '.', '/', '@' and '~' are used in addresses and waiting ids; choose an id without them."


@PROBLEMS.problem()
def invalid_value(node_id: str, field: str, /, *, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A constant in the graph record is not a value of its input's type.

    ``reason`` says what the type refused, without quoting the value: a problem may be shown to anyone. ``errors`` is
    pydantic's list of what it refused, each a ``type``, ``loc`` and ``msg`` without the value, so a host can key on
    ``greater_than_equal`` or ``bool_is_not_a_number``. The ``loc`` of each has only list indexes and the fields the
    type declares, never a key of a dict, since a key is part of the value.

    Args:
        node_id: The node.
        field: The input.
        reason: Why the value is refused, without quoting it.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        problems({"nodes": [{"id": "c", "node": "count", "constants": {"n": "many"}}]})
    """
    return f"The constant of '{field}' is not valid: {reason.rstrip('.')}. Give a value of the input's type."


@PROBLEMS.problem("invalid_value", variant="bound_default")
def invalid_value_bound_default(node_id: str, field: str, /, *, type: str, reason: str, errors: list[dict[str, Any]]) -> str:
    """An input's or output's type is a type variable, the edges into the node bind it, and the default the node
    declares is not a value of the type it is bound to.

    ``first(one: T, other: T = 0)`` wired from an output of ``str`` binds ``T`` to ``str``, and ``0`` is no ``str``.
    Give the input a default of every type its edges may bind, give it a constant or an edge in the graph, or declare
    it with no default. An input an edge gives a value to never uses its default, so its default is not read.

    Args:
        node_id: The node.
        field: The input or output.
        type: The type the edges bind it to.
        reason: Why that type refuses the default, without quoting it.
        errors: pydantic's list of what it refused, without the value.

    Example:
        @registry.node("first", title="First", description="Returns the first of two values.")
        def first[T](one: T, other: T = 0) -> T:  # type: ignore[assignment]
            return one


        problems({"nodes": [{"id": "u", "node": "upper"}, {"id": "f", "node": "first"}], "edges": [["u.result", "f.one"]]})
    """
    return f"The default of '{field}' is not a value of {type}, the type the edges into '{node_id}' give it: {reason.rstrip('.')}. Give a default of that type, or none."


@PROBLEMS.problem(of_a_hook=True)
def parameter_name_invalid(node_id: str, field: str, /) -> str:
    """A node's ``compute_...`` hook returns an input or output name that cannot be one.

    A name may not be empty or contain ``.``, ``/``, ``@`` or ``~``, the characters addresses (``a.result``), node
    paths (``loud/shout``) and waiting ids (``a@1~3fa2``) are written with. A node id is held to the same rule
    (``invalid_node_id``). Change the hook.

    Args:
        node_id: The node.
        field: The name.

    Example:
        def dotted(names: str) -> dict[str, type]:
            return {name: str for name in names.split(",")}


        @registry.node("fields", title="Fields", description="Gives one answer per name.", compute_outputs=dotted)
        def fields(names: str) -> dict[str, str]:
            return {}


        problems({"nodes": [{"id": "f", "node": "fields", "constants": {"names": "a.b"}}]})
    """
    return f"'{field}' cannot be an input or output name: a name is not empty and has no '.', '/', '@' or '~'; change the node's hook."


@PROBLEMS.problem()
def repeated_edge(node_id: str, field: str, /, *, source: str) -> str:
    """The same edge is in the graph record twice.

    Remove one.

    Args:
        node_id: The node at the edge's target.
        field: The input.
        source: The edge's source, as an address.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
            "edges": [["a.result", "b.text"], ["a.result", "b.text"]],
        })
    """
    return f"The edge from '{source}' to '{field}' is in the graph twice; remove one."


@PROBLEMS.problem()
def lists_from_different_loops(*loc: str | int, inputs: list[str]) -> str:
    """Two ``Series`` inputs of a node read branches that never both have a value for the same item.

    A node with two ``Series`` inputs leaves out an item when one of them has no value for it. In the example each
    word is short or long, never both, so every item would be left out. Write ``Series[X | None]`` on both inputs to
    keep every item, with ``None`` for the branch not chosen, or wire both branches into one input.

    Args:
        loc: The node.
        inputs: The two inputs.

    Example:
        @registry.node("pair", title="Pair", description="Puts two lists of words side by side.")
        def pair(a: Series[str], b: Series[str]) -> str:
            return f"{a} {b}"


        problems({
            "nodes": [{"id": "w", "node": "words"}, {"id": "r", "node": "route"}, {"id": "p", "node": "pair"}],
            "edges": [["w.result", "r.text"], ["r.short", "p.a"], ["r.long", "p.b"]],
        })
    """
    return (
        f"{comma_separated(inputs)} read branches that never both have a value for the same item, so the node would get no items. "
        "Write Series[X | None] on them to keep every item, or wire both branches into one input."
    )


@PROBLEMS.problem()
def series_not_nested(node_id: str, /, *, series: list[str], inputs: list[str], wire_from: str, wire_into: str) -> str:
    """A node reads one item of each of two series, and neither series is returned inside the loop over the other.

    In the example ``g`` would be called once per name and once per greeting, and nothing says which name goes with
    which greeting. ``ctx.wire_from`` and ``ctx.wire_into`` name an edge that fixes it: wire an item of the first
    series into the node that returns the second, so that node is called once per item of the first. Or give one of
    the series whole to a ``Series`` input, so it arrives as one list.

    Args:
        node_id: The node.
        series: The two series, by address.
        inputs: The two inputs that read them.
        wire_from: The address of the output to wire into ``wire_into``.
        wire_into: The id of the node that returns the other series.

    Example:
        problems({
            "nodes": [{"id": "names", "node": "words"}, {"id": "greetings", "node": "words"}, {"id": "g", "node": "greet"}],
            "edges": [["names.result", "g.name"], ["greetings.result", "g.greeting"]],
        })
    """
    return (
        f"'{node_id}' reads items of {comma_separated(series)}, and neither is inside the loop over the other, "
        "so nothing says which items go together. "
        f"Wire '{wire_from}' into '{wire_into}' to put one inside the other, or give one of them whole to a Series input."
    )


@PROBLEMS.problem("series_not_nested", variant="inside_subgraph")
def loops_not_nested_inside_subgraph(node_id: str, /, *, series: list[str], inputs: list[str], subgraphs: list[str]) -> str:
    """As ``series_not_nested``, where the node that returns one of the two series is inside a subgraph.

    No edge of the outer graph can change how a subgraph loops. Give one of the series whole to a ``Series`` input,
    or change the subgraph so it returns both outputs from the same loop.

    Args:
        node_id: The node.
        series: The two series, by address; a series inside a subgraph is named by the subgraph node's id and its
            address inside: ``e.words.result``.
        inputs: The two inputs that read them.
        subgraphs: The ids of the subgraph nodes the series are inside.

    Example:
        registry.subgraph("split", title="Split", description="Returns the words of a text.")(make_graph(words))
        problems({
            "nodes": [{"id": "e", "node": "split"}, {"id": "w", "node": "words"}, {"id": "g", "node": "greet"}],
            "edges": [["w.result", "g.name"], ["e.words.result", "g.greeting"]],
        })
    """
    return (
        f"'{node_id}' reads items of {comma_separated(series)}, and neither is inside the loop over the other, "
        "so nothing says which items go together. "
        f"One is inside a subgraph ({comma_separated(subgraphs)}), so no edge here can change that. "
        "Give one of them whole to a Series input, or change the subgraph so it returns both from the same loop."
    )


@PROBLEMS.problem()
def several_edges_into_one_input(node_id: str, field: str, /, *, sources: list[str]) -> str:
    """More than one edge into an input can give it a value for the same call.

    An input of one value takes several edges only when at most one of them has a value on each call: when they come
    from different branches of one node, outputs of one ``exclusive_group``. Otherwise write a node with an input for
    each, and wire them there. A ``Series`` input takes several edges on the same terms.

    Args:
        node_id: The node.
        field: The input.
        sources: The edges' sources, as addresses.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "upper"}],
            "edges": [["a.result", "c.text"], ["b.result", "c.text"]],
        })
    """
    return (
        f"More than one edge into '{field}' (from {comma_separated(sources)}) can give it a value on the same call. "
        "Wire it from different branches of one node, or add a node with an input for each."
    )


@PROBLEMS.problem()
def type_mismatch(node_id: str, field: str, /, *, source: str, got: str, expected: str) -> str:
    """An edge's source gives a type its target does not take.

    Wire an output of the type the input takes, or put a node between that converts one into the other. Two
    mismatches have a message of their own, below: ``ctx.variant`` says which.

    Args:
        node_id: The node at the edge's target.
        field: The input.
        source: The edge's source, as an address.
        got: The type the source gives.
        expected: The type the input takes.

    Example:
        problems({
            "nodes": [{"id": "c", "node": "count"}, {"id": "u", "node": "upper"}],
            "edges": [["c.result", "u.text"]],
        })
    """
    return (
        f"'{source}' gives {got}, but '{node_id}.{field}' takes {expected}; "
        f"connect an output that gives {expected}, or convert it in a node between."
    )


@PROBLEMS.problem("type_mismatch", variant="make_a_series")
def type_mismatch_make_a_series(node_id: str, field: str, /, *, source: str, got: str, expected: str, whole: Literal["table", "list"], fields: list[str]) -> str:
    """A whole table or list is wired into an input that takes one value.

    A table or a list is one value. To call the node once per row or item, put a node between them that returns the
    rows or items as a ``Series`` (a node declared ``-> Series[R]``), and wire that node's result, or one field of
    each record of it (``each.result.email``).

    Args:
        node_id: The node at the edge's target.
        field: The input.
        source: The edge's source, as an address.
        got: The type the source gives.
        expected: The type the input takes.
        whole: Whether the source is a table or a list.
        fields: The fields each row or item has, so an editor can offer them; empty when they are not known.

    Example:
        problems({
            "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
            "edges": [["c.result", "u.text"]],
        })
    """
    return (
        f"'{source}' is a whole {whole}, and '{node_id}.{field}' takes one {expected}. To call '{node_id}' once per row or item, "
        "put a node between them that returns its rows or items as a Series, and wire that node's result, or one field of it."
    )


@PROBLEMS.problem("type_mismatch", variant="add_column")
def type_mismatch_add_column(node_id: str, field: str, /, *, source: str, got: str, expected: str, column: str, column_type: str, has: list[str]) -> str:
    """A table is wired into a ``Table[R]`` input that needs a column the table does not have.

    Wire a table that has every column the input needs, add the column in a node between, or type the input as a bare
    ``Table``, which takes any columns.

    Args:
        node_id: The node at the edge's target.
        field: The input.
        source: The edge's source, as an address.
        got: The type the source gives.
        expected: The type the input takes.
        column: The first column the input needs that the table does not have.
        column_type: That column's type.
        has: The columns the table has.

    Example:
        class Invoice(TypedDict):
            name: str
            amount: int


        @registry.node("total", title="Total", description="Adds up the amounts on all invoices.")
        def total(invoices: Table[Invoice]) -> int:
            return int(invoices["amount"].sum())


        problems({
            "nodes": [{"id": "c", "node": "contacts"}, {"id": "t", "node": "total"}],
            "edges": [["c.result", "t.invoices"]],
        })
    """
    return (
        f"'{node_id}.{field}' needs the column '{column}' ({column_type}); '{source}' has {comma_separated(has)}. "
        "Add the column in a node between, or connect a table that has it."
    )


@PROBLEMS.problem()
def unbound_on_later_node(node_id: str, field: str, /) -> str:
    """An input with no edge and no constant, on a node that is not one of the graph's first nodes.

    Only the inputs of a graph's first nodes (the nodes with no edge into them) are graph inputs, which the caller
    gives. Draw an edge into the input, or give it a constant.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet"}],
            "edges": [["a.result", "g.name"]],
        })
    """
    return f"'{field}' has no edge and no constant, and only a first node's inputs come from the caller; wire an edge into it or give it a constant."


@PROBLEMS.problem()
def unknown_input(node_id: str, field: str, /, *, inputs: list[str]) -> str:
    """A constant or an edge is for an input the node does not have.

    Use one of the inputs ``ctx.inputs`` lists. When a newer version of the node renamed the input,
    ``registry.upgrade`` moves the constant to the new name.

    Args:
        node_id: The node.
        field: The name that is not an input of the node.
        inputs: The node's inputs.

    Example:
        problems({"nodes": [{"id": "u", "node": "upper", "constants": {"txt": "a"}}]})
    """
    return f"'{node_id}' has no input '{field}'; it has {comma_separated(inputs)}."


@PROBLEMS.problem()
def unknown_node(*loc: str | int, node: str, choices: list[str]) -> str:
    """A node of the graph record names a node definition the registry does not have.

    Use one the registry has (``ctx.choices``), or add the missing one to the registry the graph is compiled with.

    Args:
        loc: The node.
        node: The id of the node definition it names.
        choices: The ids of the registry's node definitions.

    Example:
        problems({"nodes": [{"id": "m", "node": "send-mail"}]})
    """
    return f"There is no node '{node}'; the nodes are {comma_separated(choices)}."


@PROBLEMS.problem()
def unknown_output(*loc: str | int, source: str, outputs: list[str], record_field: str | None = None) -> str:
    """An edge's source is an output, or a field of one, that its node does not have.

    Wire from one of the addresses ``ctx.outputs`` lists. An edge may name one field of a record an output gives,
    ``node.output.field``; ``ctx.outputs`` lists those too.

    Args:
        loc: The node at the edge's target, and the input.
        source: The edge's source, as an address.
        outputs: The addresses the source's node has.
        record_field: The field of a record the edge names, when the record has no field by that name.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
            "edges": [["a.text", "b.text"]],
        })
    """
    return f"'{source}' is not an output of that node; it has {comma_separated(outputs)}."


@PROBLEMS.problem("unknown_output", variant="make_a_series")
def unknown_output_make_a_series(*loc: str | int, source: str, outputs: list[str], record_field: str, whole: Literal["table", "list"], fields: list[str]) -> str:
    """An edge's source names a field of a whole table or list.

    A table or a list is one value, so it has no field to wire. Put a node after it that returns its rows or items as
    a ``Series`` (a node declared ``-> Series[R]``), and wire the field from that node's result: ``each.result.email``.

    Args:
        loc: The node at the edge's target, and the input.
        source: The edge's source, as an address.
        outputs: The addresses the source's node has.
        record_field: The field the edge names.
        whole: Whether the source is a table or a list.
        fields: The fields each row or item has, so an editor can offer them; empty when they are not known.

    Example:
        problems({
            "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
            "edges": [["c.result.email", "u.text"]],
        })
    """
    return (
        f"'{source}' names a field of a whole {whole}, which has no fields to wire. Put a node after it that returns its rows "
        "or items as a Series, and wire the field from that node's result."
    )


@PROBLEMS.problem()
def unknown_version(*loc: str | int, node: str, version: int, newest: int) -> str:
    """A node of the graph record names a version its node definition does not have.

    Use a version the node definition has. ``registry.upgrade(record, node_id)`` moves a saved node to the newest.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the graph record names.
        newest: The newest version of the node definition.

    Example:
        problems({"nodes": [{"id": "u", "node": "upper", "version": 9}]})
    """
    return f"'{node}' has no version {version}; the newest is {newest}."


@PROBLEMS.problem()
def constant_and_default(node_id: str, field: str, /) -> str:
    """An input has both a constant and a default in the graph record.

    Keep one: a constant fixes the input's value, a default makes it a graph input the caller may give.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": [{"id": "u", "node": "upper", "constants": {"text": "a"}, "defaults": {"text": "b"}}]})
    """
    return f"'{field}' has both a constant and a default; keep one."


@PROBLEMS.problem()
def constant_and_edge(node_id: str, field: str, /) -> str:
    """An input has both an edge into it and a constant in the graph record.

    Keep one: remove the edge or the constant.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({
            "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper", "constants": {"text": "x"}}],
            "edges": [["a.result", "b.text"]],
        })
    """
    return f"'{field}' has both an edge and a constant; keep one."


PROBLEMS.section(
    "A refused call",
    "Raised as `InvalidCallError` by the call that caused it, before anything ran; `error.errors()` lists every "
    "problem at once, and the graph and the run are unchanged. `loc` says what the call got wrong: "
    '`("inputs", name)` for an input, `(node_id, *item_indexes)` for an answer, `("saved_run", ...)` for a saved run.',
)


@PROBLEMS.problem()
def answer_stale(*loc: str | int, waiting_id: str, current: str, moved_to: str | None = None) -> str:
    """An answer is for a question that another question has replaced.

    A run made with other inputs (``run.again``, ``run.with_inputs``) asks again where the inputs of the call changed,
    under a new waiting id. Answer the question that waits now: ``ctx.current`` is its id, and ``run.waiting`` has it.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        current: The waiting id of the question that waits in its place.
        moved_to: The waiting id the same question has now, when it waits for another call of the node: a change of
            inputs gave its item another index.

    Example:
        asked = Graph.compile(make_graph(approve), registry).run(amount=5000)
        asked_again = asked.again(amount=5500)
        refused(lambda: asked_again.answer(asked.waiting[0].id, True))
    """
    return f"The answer is for {waiting_id!r}, and another question, {current!r}, has replaced it; answer that one."


@PROBLEMS.problem()
def cannot_move_to_version(*loc: str | int, node: str, version: int, target: int, versions: list[int]) -> str:
    """``registry.upgrade`` is asked to move a node to a version it cannot reach from the version it is at.

    Ask for a version ``ctx.versions`` lists, or leave ``to`` out for the newest.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the node is at.
        target: The version asked for.
        versions: The versions of the node definition.

    Example:
        refused(lambda: registry.upgrade(make_graph(upper), "upper", to=3))
    """
    return f"'{node}' cannot move from version {version} to {target}; it has {comma_separated(versions)}."


@PROBLEMS.problem()
def context_missing(*loc: str | int, parameter: str, needs: str) -> str:
    """A node takes the run's context, and the call that starts or resumes the run gives no ``context``.

    Pass the object on every call that goes on with the run: ``graph.run(context=HostContext(...))``,
    ``run.resume(context=...)``, ``run.stream(context=...)``. A run never saves its context. Every node that takes it
    has this problem.

    Args:
        loc: The node's path, then its ``Context`` parameter.
        parameter: The node's ``Context`` parameter, its first.
        needs: The name of the class inside ``Context[...]``.

    Example:
        refused(lambda: Graph.compile(make_graph(who), registry).run())
    """
    return f"The node takes the run's context, a {needs}, as '{parameter}', and the call gave none; pass context= on every call that starts or resumes the run."


@PROBLEMS.problem()
def input_given_twice(*loc: str | int, name: str, address: str) -> str:
    """A call gives a graph input both by its name and by its address.

    Give it once.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        address: The input's address.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run({"upper.text": "a"}, text="b"))
    """
    return f"'{name}' is given twice, by its name and by its address '{address}'; give it once."


@PROBLEMS.problem()
def invalid_answer(*loc: str | int, waiting_id: str, reason: str, question: str | None = None) -> str:
    """An answer is not a value of the type the question takes.

    Answer with a value of the question's type, as JSON reads it: ``true``, not ``"yes"``.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        reason: What is wrong with the answer.
        question: For a node that asks several questions (one per output), the name of the one the fault is in.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run(amount=5000)
        refused(lambda: waiting.answer(waiting.waiting[0].id, "yes"))
    """
    return f"That is not a valid answer: {reason.rstrip('.')}"


@PROBLEMS.problem()
def invalid_checkpoint_every(*loc: str | int, checkpoint_every: float) -> str:
    """A call gives ``checkpoint_every`` a value that is not a number of seconds above 0: ``0``, ``inf`` or ``True``.

    Give a number above 0, or leave it out for no checkpoints.

    Args:
        loc: ``"checkpoint_every"``.
        checkpoint_every: The value given.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).with_inputs(text="a").stream(checkpoint_every=0))
    """
    return f"checkpoint_every must be a number of seconds above 0; it was {checkpoint_every}."


@PROBLEMS.problem()
def invalid_context(*loc: str | int, expected: str, got: str) -> str:
    """A call gives ``context`` an object that is not an instance of the class the registry names.

    Pass an instance of the registry's ``context_type``, or of a subclass of it.

    Args:
        loc: ``"context"``.
        expected: The name of the registry's class.
        got: The name of the class of the object given.

    Example:
        refused(lambda: Graph.compile(make_graph(who), registry).run(context=Caller("Ida")))  # type: ignore[arg-type]
    """
    return f"context was given a {got}; pass a {expected}, the class the registry names."


@PROBLEMS.problem("invalid_context", variant="no_context_type")
def invalid_context_no_context_type(*loc: str | int, got: str) -> str:
    """A call gives ``context`` to a run of a graph whose registry names no ``context_type``.

    Pass no context, or name the class on the registry, ``NodeRegistry(context_type=HostContext)``.

    Args:
        loc: ``"context"``.
        got: The name of the class of the object given.

    Example:
        plain = NodeRegistry()
        plain.add(upper)
        refused(lambda: Graph.compile(make_graph(upper), plain).run(text="a", context=Caller("Ida")))  # type: ignore[arg-type]
    """
    return f"context was given a {got}, and the graph's registry names no context class; pass no context, or name one with NodeRegistry(context_type=...)."


@PROBLEMS.problem()
def invalid_input(*loc: str | int, name: str, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A call gives a graph input a value its type does not take.

    Give a value of the input's type; ``graph.inputs[name].json_schema`` is its schema, limits included. ``reason``
    and ``errors`` are as ``invalid_value`` has them.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        reason: Why the value is refused, without quoting it.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        refused(lambda: Graph.compile(make_graph(count), registry).run(n="many"))
    """
    return f"The value given for '{name}' is not valid: {reason.rstrip('.')}. Give a value of the input's type."


@PROBLEMS.problem("invalid_input", variant="make_a_series")
def invalid_input_make_a_series(*loc: str | int, name: str, expected: str, length: int) -> str:
    """A call gives a list to a graph input that takes one value.

    A graph loops over a list only when a node returns it as a series, so a list given to an input of one value is
    refused. Give the list to a node that returns it as a series, or give one value. A ``Series`` graph input takes a
    list, and a single value as a list of one.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        expected: The type the input takes.
        length: How many items the list has.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text=["a", "b"]))
    """
    return (
        f"'{name}' takes one {expected} and was given a list of {length}. A graph loops over a list only when a node returns it "
        "as a Series: give the list to such a node, or give one value."
    )


@PROBLEMS.problem()
def invalid_max_items(*loc: str | int, max_items: int) -> str:
    """A call gives ``max_items`` a value that is not a whole number of at least 1: ``0``, ``2.5`` or ``True``.

    Give a whole number of 1 or more, or leave it out for no item limit.

    Args:
        loc: ``"max_items"``.
        max_items: The value given.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text="a", max_items=0))
    """
    return f"max_items must be a whole number of at least 1; it was {max_items}."


@PROBLEMS.problem()
def invalid_record(*loc: str | int, where: str, reason: str, error: str) -> str:
    """What was given as a graph record is not one: its JSON does not have a graph record's shape.

    Fix the JSON: ``ctx.where`` says where it stops being a graph record. A graph record of the right shape is always
    compiled, however unfinished; what is wrong with it is in ``graph.problems``. In the example, the node says
    ``type`` where a graph record says ``node``.

    Args:
        loc: ``"record"`` and the steps to the fault inside the graph record.
        where: The same steps as one text, joined by dots; ``the record`` for the graph record as a whole.
        reason: What is wrong there.
        error: The type of pydantic's error, such as ``extra_forbidden``.

    Example:
        refused(lambda: Graph.compile('{"nodes": [{"id": "u", "type": "upper"}]}', registry))
    """
    return f"The record is not a graph record at '{where}': {reason.rstrip('.')}. Fix the record's JSON."


@PROBLEMS.problem()
def invalid_timeout(*loc: str | int, timeout: float) -> str:
    """A call gives ``timeout`` a value that is not a number of seconds above 0: ``0``, ``nan`` or ``True``.

    Give a number above 0, or leave it out for no time limit.

    Args:
        loc: ``"timeout"``.
        timeout: The value given.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).with_inputs(text="a").resume(timeout=-1))
    """
    return f"timeout must be a number of seconds above 0; it was {timeout}."


@PROBLEMS.problem()
def missing_input(*loc: str | int, name: str) -> str:
    """A call leaves out a graph input that has no default.

    Give it in the call. ``graph.inputs`` lists the graph inputs, and which of them are required.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run())
    """
    return f"'{name}' is required; give it in the call."


@PROBLEMS.problem()
def node_not_in_graph(node_id: str, /, *, nodes: list[str]) -> str:
    """A call names a node the graph does not have.

    Name one of the nodes ``ctx.nodes`` lists.

    Args:
        node_id: The id given.
        nodes: The ids of the graph's nodes.

    Example:
        done = Graph.compile(make_graph(upper), registry).run(text="a")
        refused(lambda: done.forget("lower"))
    """
    return f"'{node_id}' is not a node of this graph; it has {comma_separated(nodes)}."


@PROBLEMS.problem()
def not_a_chain(node_id: str, /, *, before: str, inputs: list[str], outputs: list[str]) -> str:
    """``make_graph`` is given no edges, and cannot wire one of its nodes to the node before it.

    Without ``edges=``, ``make_graph`` wires each node's one required input that has no constant to the one output of
    the node before it. Pass the edges instead: ``make_graph(upper, greet, edges=[("upper.result", "greet.name")])``.

    Args:
        node_id: The node that cannot be wired.
        before: The node before it.
        inputs: The required inputs of ``node_id`` that have no constant.
        outputs: The outputs of ``before``.

    Example:
        refused(lambda: make_graph(upper, greet))
    """
    return (
        f"'{node_id}' can't follow '{before}' without edges: make_graph wires a node's one required input (it has {comma_separated(inputs)}) "
        f"to the one output of the node before it (that has {comma_separated(outputs)}); pass edges= to make_graph."
    )


@PROBLEMS.problem()
def not_offered(*loc: str | int, name: str, inputs: list[str]) -> str:
    """A call gives a value under a name that is not a graph input.

    Use one of the names ``ctx.inputs`` lists.

    Args:
        loc: ``"inputs"`` and the name.
        name: The name the value was given under.
        inputs: The names of the graph inputs.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text="a", txt="a"))
    """
    return f"The graph has no input '{name}'; it has {comma_separated(inputs)}."


@PROBLEMS.problem()
def not_waiting(*loc: str | int, waiting_id: str, status: str, moved_to: str | None = None) -> str:
    """An answer is for a waiting id that no call of the run is waiting under.

    Answer one of the ids in ``run.waiting``.

    Args:
        loc: The node and the item indexes the waiting id names; the node alone when the id names no call.
        waiting_id: The waiting id the answer was given for.
        status: The run's status.
        moved_to: The waiting id the same question has now, when it waits for another call of the node: a change of
            inputs gave its item another index.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run(amount=5000)
        refused(lambda: waiting.answer("upper", True))
    """
    return f"Nothing in the run waits for an answer under {waiting_id!r} (the run is {status}); run.waiting lists what does."


@PROBLEMS.problem()
def run_not_complete(*loc: str | int, status: str, reason: str) -> str:
    """``run.outputs`` is read on a run that is not complete.

    Answer what waits and ``resume()``, or read ``run.results``, which has every node's outputs so far.

    Args:
        loc: ``"outputs"``.
        status: The run's status.
        reason: The message of the problem the run ended on; empty when the run did not end early.

    Example:
        stopped = Graph.compile(make_graph(words, upper), registry).run(text="a b c", max_items=2)
        refused(lambda: stopped.outputs)
    """
    return f"The run is {status}, so it has no outputs yet; run.results has the outputs so far. {reason.rstrip('.')}".rstrip()


@PROBLEMS.problem()
def saved_run_of_another_graph(*loc: str | int, saved: str, graph: str) -> str:
    """A saved run is loaded into another graph than the one it was saved from.

    Load it into the graph it was saved from. A changed graph starts a new run.

    Args:
        loc: ``"saved_run"``.
        saved: The fingerprint of the graph the run was saved from.
        graph: The fingerprint of the graph it is loaded into.

    Example:
        saved = Graph.compile(make_graph(upper), registry).run(text="a").to_json()
        refused(lambda: Run.from_json(saved, Graph.compile(make_graph(count), registry)))
    """
    return f"The saved run is of another graph ({saved}) than this one ({graph}); load it into the graph it was saved from."


@PROBLEMS.problem()
def saved_run_unreadable(*loc: str | int, where: str, reason: str) -> str:
    """A saved run is not what ``run.to_json()`` writes for this graph.

    Load what ``run.to_json()`` wrote, unchanged. ``ctx.where`` says where it stops fitting.

    Args:
        loc: ``"saved_run"`` and the steps to the fault inside the saved run.
        where: The same steps as one text, joined by dots; ``the run`` for the saved run as a whole.
        reason: What is wrong there.

    Example:
        uppering = Graph.compile(make_graph(upper), registry)
        changed = json.loads(uppering.run(text="a").to_json())
        changed["outcomes"] = {"upper": "not outcomes"}
        refused(lambda: Run.from_json(json.dumps(changed), uppering))
    """
    return f"The saved run does not fit this graph at '{where}': {reason.rstrip('.')}. Load it into the graph it was saved from."


@PROBLEMS.problem()
def run_unwritable(*loc: str | int, where: str, reason: str) -> str:
    """A run has a value that cannot be written as JSON.

    Give the output a type pydantic writes as JSON. ``ctx.where`` names the output. In the example, ``raw`` returns
    bytes that are not text.

    Args:
        loc: ``"saved_run"`` and the steps to the value, or ``"outputs"`` and the output's address.
        where: The same steps as one text.
        reason: Why the value cannot be written.

    Example:
        @registry.node("raw", title="Raw", description="Turns a text into raw bytes.")
        def raw(text: str) -> bytes:
            return b"\\xff" + text.encode()


        unwritable = Graph.compile(make_graph(raw), registry).run(text="a")
        refused(lambda: unwritable.to_json())
    """
    return f"The run cannot be written as JSON at '{where}': {reason.rstrip('.')}. Give that output a type pydantic writes as JSON."


PROBLEMS.section(
    "Why a run stopped",
    "The `cause` of an `EndedEarly` `run.state`, and the `problem` of a `node_failed` event. For a call that failed, "
    '`loc` is the node\'s id and the call\'s item indexes: `("shout", 1)`; inside a subgraph, the subgraph node\'s id '
    "comes first. For the run as a whole, `loc` is `()`. `resume()` goes on from where the run stopped.",
)


@PROBLEMS.problem()
def ask_not_declared(*loc: str | int) -> str:
    """A node raised ``Ask`` in its own function, and its ``@node`` does not say ``asks=True``.

    A node that may wait for a person says so where it is declared, so a host knows before a run which graphs can
    pause. Add ``asks=True`` to the ``@node``, or ask through an ``Asked`` parameter.

    Args:
        loc: The call that raised it.

    Example:
        @registry.node("confirm_quietly", title="Confirm", description="Asks a person to confirm, without saying so.")
        def confirm_quietly(text: str) -> bool:
            raise Ask("Sure?")


        why_it_stopped(Graph.compile(make_graph(confirm_quietly), registry).run(text="a"))
    """
    return "The node asked a person, but its @node does not say asks=True. Add asks=True, or ask through an Asked parameter."


@PROBLEMS.problem()
def execution_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception that is not a ``NodeError``.

    The exception's text and traceback go to the ``conductor`` log and never into the problem, since a problem may be
    shown to anyone; ``ctx.error`` is its class name. Fix the node, then ``resume()``.

    Args:
        loc: The call that failed.
        error: The exception's class name.

    Example:
        @registry.node("divide", title="Divide", description="Divides one by a number.")
        def divide(n: int) -> float:
            return 1 / n


        why_it_stopped(Graph.compile(make_graph(divide), registry).run(n=0))
    """
    return "The node raised an exception; its developer finds it in the conductor log."


@PROBLEMS.problem()
def external_failed(*loc: str | int, error: str | None = None) -> str:
    """A node raised an exception its ``Policy(retry_on=)`` names, on every attempt.

    The outside world failed: a network error, a rate limit, a server error. ``resume()`` tries again; raise
    ``Policy(retries=)`` for more attempts. An ``ExternalError`` a node raises itself has this code too, with the
    node's own message and ``ctx``.

    Args:
        loc: The call that failed.
        error: The exception's class name; absent for an ``ExternalError`` the node raised, whose ``ctx`` is its own.

    Example:
        @registry.node("fetch", title="Fetch", description="Fetches a page, and fails.", policy=Policy(retry_on=(ConnectionError,)))
        def fetch(url: str) -> str:
            raise ConnectionError(url)


        why_it_stopped(Graph.compile(make_graph(fetch), registry).run(url="https://example.com"))
    """
    return "An outside service did not answer."


@PROBLEMS.problem()
def failed(*loc: str | int) -> str:
    """A node raised a ``NodeError`` that gives no code of its own.

    The problem is made from the node's ``NodeError``, not by calling this function: its message and ``ctx`` are the
    ones the node gave. This function declares the code, and its message is a fallback for a host that has no message
    of its own for the code. A node gives a code of its own with
    ``NodeError(message, code=)``, and the values a host fills its own message from with ``ctx=``, JSON values only.

    Args:
        loc: The call that failed.

    Example:
        @registry.node("read_pdf", title="Read PDF", description="Reads the text from a PDF file, and fails.")
        def read_pdf(path: str) -> str:
            raise NodeError("The document could not be read.")


        why_it_stopped(Graph.compile(make_graph(read_pdf), registry).run(path="a.pdf"))
    """
    return "The node reported a failure."


@PROBLEMS.problem()
def invalid_ask(*loc: str | int, reason: str) -> str:
    """A node raised an ``Ask`` whose choices or default are not values of the type its answer is read by.

    The answer to an ``Ask`` is a value of the return type of the function that raised it: the node's outputs, or an
    ``Asked`` parameter's value. So its choices and default must be of that type. Fix the ``Ask``; ``ctx.reason`` says
    which part does not fit.

    Args:
        loc: The call that asked.
        reason: What is wrong with the question.

    Example:
        @registry.node("confirm", title="Confirm", description="Asks a person to confirm.", asks=True)
        def confirm(text: str) -> bool:
            raise Ask("Sure?", default="maybe")


        why_it_stopped(Graph.compile(make_graph(confirm), registry).run(text="a"))
    """
    return f"The node asked a question with choices or a default the answer's type does not take: {reason.rstrip('.')}. Fix the node's Ask."


@PROBLEMS.problem()
def invalid_asked_value(*loc: str | int, name: str, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """The function of an ``Asked`` parameter returned a value the parameter's type refuses.

    Its return value is the parameter's value, checked as a node's return is: strictly, so ``"yes"`` is never a
    ``bool``. Return what the parameter declares.

    Args:
        loc: The call the value was for.
        name: The parameter.
        reason: Why the value is refused, without quoting it.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        def says_yes(amount: int) -> bool:
            return "yes"  # type: ignore[return-value]


        @registry.node("pay_on_yes", title="Pay", description="Pays an amount a person approves.")
        def pay_on_yes(amount: int, ok: Annotated[bool, Asked(says_yes)]) -> str:
            return "paid"


        why_it_stopped(Graph.compile(make_graph(pay_on_yes), registry).run(amount=5))
    """
    return f"The node's Asked function for '{name}' returned a value that is not what the parameter declares: {reason.rstrip('.')}. Fix what it returns."


@PROBLEMS.problem()
def invalid_edge_value(
    *loc: str | int, name: str, reason: str, record_field: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None,
) -> str:
    """The value an edge gives an input is refused by the input's type, just before the call.

    Compiling checks that an edge's types fit; some things it cannot check, and those are checked on each call: a
    limit or choices on the input, a host type's own check, a table whose columns are known only once it is made, a
    field of a record whose type declares no fields. Make the upstream node return what the input takes, or put a
    node between that converts it. In the example, ``letters`` returns 12 and ``small`` takes at most 3.

    Args:
        loc: The call that was about to be made.
        name: The input.
        reason: Why the value is refused, without quoting it.
        record_field: The field of a record the edge names, when the record has no field by that name.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("letters", title="Letters", description="Counts the letters in a text.")
        def letters(text: str) -> int:
            return len(text)


        @registry.node("small", title="Small", description="Passes on a small number.")
        def small(n: Annotated[int, Param(le=3)]) -> int:
            return n


        why_it_stopped(Graph.compile(make_graph(letters, small), registry).run(text="many letters"))
    """
    return (
        f"The value an edge gave '{name}' is not one it takes: {reason.rstrip('.')}. "
        "Fix the node upstream, or wire an output of the type the input takes."
    )


@PROBLEMS.problem()
def invalid_output(*loc: str | int, reason: str, name: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A node returned a value its return type does not declare.

    Return what the node declares. Return values are checked strictly: ``"12"`` is never an ``int``.

    Args:
        loc: The call that returned it.
        reason: Why the value is refused, without quoting it.
        name: The output, when the node has several.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        @registry.node("guess", title="Guess", description="Returns a text where a number is declared.")
        def guess(text: str) -> int:
            return text  # type: ignore[return-value]


        why_it_stopped(Graph.compile(make_graph(guess), registry).run(text="12"))
    """
    return f"The node returned a value that is not what it declares: {reason.rstrip('.')}. Fix what the node returns."


@PROBLEMS.problem()
def item_limit(*loc: str | int, series: str, items: int, max_items: int) -> str:
    """A series has more items than the run's ``max_items``, so the run stopped before calling any node for them.

    Only a series that a node loops over counts: one that is collected or passed on whole costs no calls.
    ``run.with_max_items(n).resume()`` goes on with a higher limit, and calls nothing again.

    Args:
        loc: The call that returned the series.
        series: The series, by address.
        items: How many items it has.
        max_items: The run's item limit.

    Example:
        why_it_stopped(Graph.compile(make_graph(words, upper), registry).run(text="a b c", max_items=2))
    """
    return f"{series} has {items} items, more than this run's max_items of {max_items}; raise max_items and resume the run."


@PROBLEMS.problem()
def question_unwritable(*loc: str | int, reason: str, input: str) -> str:  # noqa: A002  the ctx key a host reads
    """A node asked a person, and one of its arguments cannot be written as JSON to show them.

    A question carries the arguments of the call that asked, for the person answering it. Give the input a type
    pydantic writes as JSON, or ask in a node that takes one.

    Args:
        loc: The call that asked.
        reason: Why the argument cannot be written.
        input: The input.

    Example:
        @registry.node("approve-file", title="Approve file", description="Asks a person to approve sending a file.", asks=True)
        def approve_file(data: bytes) -> bool:
            raise Ask("Send this file?")


        why_it_stopped(Graph.compile(make_graph(approve_file), registry).run(data=b"\\xff"))
    """
    return (
        f"The node asked a person, but its argument '{input}' cannot be written as JSON to show them: {reason.rstrip('.')}. "
        "Give that input a type pydantic writes as JSON."
    )


@PROBLEMS.problem()
def run_cancelled() -> str:
    """The ``cancel`` event given to ``run.stream`` was set.

    Save the run and ``resume()`` it to go on. Calls that returned before the cancel are kept.

    Example:
        async def cancelled() -> Run:
            stop = asyncio.Event()
            stop.set()
            async for event in Graph.compile(make_graph(upper), registry).with_inputs(text="a").stream(cancel=stop):
                if event.type == "run_end":
                    return event.run
            raise AssertionError("the stream ended without a run_end event")


        why_it_stopped(asyncio.run(cancelled()))
    """
    return "The run was cancelled. Resume it to carry on, or start a new run."


@PROBLEMS.problem()
def run_timed_out(*, timeout: float) -> str:
    """A call of ``run``, ``resume`` or ``stream`` took longer than the ``timeout`` it was given.

    ``resume()`` goes on, and calls nothing again that returned. Give a longer ``timeout``, or none.

    Args:
        timeout: The ``timeout`` given, in seconds.

    Example:
        @registry.node("slow", title="Slow", description="Waits a moment, then returns the text.")
        def slow(text: str) -> str:
            time.sleep(0.2)
            return text


        why_it_stopped(Graph.compile(make_graph(slow), registry).run(text="a", timeout=0.05))
    """
    return "The run took too long and was stopped; run.resume() goes on from where it stopped."


@PROBLEMS.problem()
def timeout(*loc: str | int, seconds: float) -> str:
    """A node's call took longer than its ``Policy(attempt_timeout=)``.

    Raise the node's ``attempt_timeout``, or make the call faster. A call that timed out is not retried.

    Args:
        loc: The call that failed.
        seconds: The node's ``attempt_timeout``.

    Example:
        @registry.node("stuck", title="Stuck", description="Waits too long.", policy=Policy(attempt_timeout=0.05))
        def stuck(text: str) -> str:
            time.sleep(0.2)
            return text


        why_it_stopped(Graph.compile(make_graph(stuck), registry).run(text="a"))
    """
    return "The node did not return in time; its Policy(attempt_timeout=) sets how long it may take."
