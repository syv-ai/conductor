"""The kinds of problem compiling a graph reports, in ``graph.problems``: the section "Compiling a graph" of ``PROBLEMS``.

A module that reports one imports this module and calls the kind: ``compile_problems.cycle(node_id)``.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor._internal.base import comma_separated

from .type import ProblemCatalogue

#: The kinds of this section of the problems page.
COMPILE_PROBLEMS = ProblemCatalogue(
    "Compiling a graph",
    "In `graph.problems`; compiling never raises for them, and a graph with problems does not run. `loc` is "
    "`(node_id,)` or `(node_id, input)`, and `()` for a problem of the graph's declared outputs. For a node inside a "
    'subgraph, the subgraph node\'s id comes first: `("loud", "shout", "line")`.',
)


@COMPILE_PROBLEMS.problem()
def alternatives_from_different_loops(node_id: str, field: str, /, *, series: list[str | None]) -> str:
    """Two edges into one input come from branches whose values are inside different loops.

    In the example, ``u.text`` has an edge from each branch of ``r``. The long branch gives one value per item of
    ``w.result``. The short branch goes through ``letters``, which returns a series of its own, so it gives one value
    per item of ``letters.result``. A call of ``u`` is for an item of one loop and can have no value from the other.
    Collect the inner loop first, in a node with a ``Series`` input, or wire the input from one branch only.

    Args:
        node_id: The node.
        field: The input.
        series: The series each edge's values are items of, as its node's path and output (``lines.result``,
            ``e/words.result`` for a node inside the subgraph node ``e``); ``None`` for an edge from a node inside no
            loop.

    Example:
        problems({"nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "r": {"node": "route", "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}}},
            "letters": {"node": "words", "inputs": {"text": {"sources": [{"node": "r", "output": "short"}]}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "letters", "output": "result"}, {"node": "r", "output": "long"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    series_text = comma_separated(["none (not in a loop)" if one is None else one for one in series])
    return (
        f"'{field}' has edges from branches inside different loops ({series_text}), so a call never has the other branch's value. "
        f"Collect the inner loop first, or wire '{field}' from one branch."
    )


@COMPILE_PROBLEMS.problem(of_a_hook=True)
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
        problems({"nodes": {
            "pick": {"node": "pick", "inputs": {"names": {"constant": ""}}}
        }})
    """
    return f"The node could not work out its inputs and outputs: {reason.rstrip('.')}. Check the node's constants and what is wired to it."


@COMPILE_PROBLEMS.problem()
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

        problems({"nodes": {
            "count": {"node": "count_it"}
        }})
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry's context class {context_type} is not a {needs}; ask for {context_type}, a base of it or a Protocol it satisfies."


@COMPILE_PROBLEMS.problem("context_does_not_fit", variant="lacks")
def context_does_not_fit_lacks(node_id: str, parameter: str, /, *, needs: str, context_type: str, lacks: list[str]) -> str:
    """A node takes the run's context as a ``Protocol`` whose members the registry's ``context_type`` does not all have.

    Compiling compares the members by name only: whether their types fit is the host's type checker's job. Give the
    registry's class a member of each name the node reads, or change what the node asks for. In the example,
    ``HostContext`` has no ``database``.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, its first.
        needs: The name of the ``Protocol`` inside ``Context[...]``.
        context_type: The name of the registry's class.
        lacks: The name of each member the class does not have.

    Example:
        class HasDatabase(Protocol):
            @property
            def database(self) -> str: ...

        @registry.node("tables", title="Tables", description="Names the database it reads.")
        def tables(ctx: Context[HasDatabase]) -> str:
            return ctx.database

        problems({"nodes": {
            "tables": {"node": "tables"}
        }})
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry's context class {context_type} lacks {comma_separated(lacks)}; give {context_type} what the node reads, or change what it asks for."


@COMPILE_PROBLEMS.problem("context_does_not_fit", variant="no_context_type")
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
        problems(make_graph(who), plain)
    """
    return f"'{node_id}' takes the run's context as a {needs}, and the registry names no context class; name one with NodeRegistry(context_type=...)."


@COMPILE_PROBLEMS.problem()
def cycle(*loc: str | int) -> str:
    """A node is on a cycle of edges.

    A graph calls each node after the nodes upstream of it, so a graph with a cycle cannot run. Remove one of the
    edges that close the cycle. Every node on it has this problem.

    Args:
        loc: The node.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}},
            "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}
        }})
    """
    return "The node is part of a cycle, so the graph cannot run; remove one of the edges that close it."


@COMPILE_PROBLEMS.problem()
def graph_output_not_found(*, output: str, source: str, outputs: list[str]) -> str:
    """A graph output names a node the record does not have, or an output its node does not have.

    The problem is the graph's, at no node: ``loc`` is empty and ``ctx.output`` names the graph output. Name an
    output of a node the graph has.

    Args:
        output: The graph output's name.
        source: The output it names, ``node.output``.
        outputs: The outputs its node has, each ``node.output``; empty when the record has no such node.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}
        }, "outputs": {"result": {"node": "u", "output": "shout"}}})
    """
    if outputs:
        return f"The graph output '{output}' is '{source}', which its node does not have; it has {comma_separated(outputs)}."
    return f"The graph output '{output}' is '{source}', whose node is not in the graph; name an output of a node it has."


@COMPILE_PROBLEMS.problem()
def edge_names_missing_node(*loc: str | int, missing: str, source: str, target: str) -> str:
    """An edge comes from a node the graph record does not have.

    Add the node, or remove the edge.

    Args:
        loc: The node the edge leads into, and the input.
        missing: The id of the node that is not in the graph record.
        source: The edge's source, as an address.
        target: The input the edge leads into, as ``node.input``.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "ghost", "output": "result"}]}}}
        }})
    """
    return f"The edge from '{source}' to '{target}' names '{missing}', which is not a node of the graph; add that node or remove the edge."


@COMPILE_PROBLEMS.problem()
def subgraph_contains_itself(*loc: str | int, node: str) -> str:
    """A subgraph contains itself, directly or through another subgraph.

    Remove the node that leads back. The problem is on that node, inside the subgraph, so its ``loc`` starts with the
    subgraph node's id: in the example, ``("i", "again")``.

    Args:
        loc: The node that leads back.
        node: The id of the subgraph it uses.

    Example:
        itself = subgraph("itself", title="Itself", description="A graph that contains itself.")(
            GraphRecord.model_validate({"nodes": {"again": {"node": "itself"}}})
        )
        looping = NodeRegistry()
        looping.add(itself)
        problems({"nodes": {"i": {"node": "itself"}}}, looping)
    """
    return f"The subgraph '{node}' contains itself; remove the node that leads back to it."


@COMPILE_PROBLEMS.problem()
def hook_needs_value(node_id: str, field: str, /) -> str:
    """A node's ``compute_...`` hook reads an input that has no constant in the graph record.

    The node's inputs or outputs depend on that value, so it must be known when the graph is compiled. Give the input
    a constant. An input a hook reads is never a graph input.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": {
            "pick": {"node": "pick"}
        }})
    """
    return f"'{field}' needs a constant: the node works out its inputs and outputs from it when the graph is compiled."


@COMPILE_PROBLEMS.problem()
def invalid_value(node_id: str, field: str, /, *, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A constant in the graph record, or a graph input's default, is not a value of its input's type.

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
        problems({"nodes": {
            "c": {"node": "count", "inputs": {"n": {"constant": "many"}}}
        }, "outputs": {"result": {"node": "c", "output": "result"}}})
    """
    return f"The value the graph record gives '{field}' is not valid: {reason.rstrip('.')}. Give a value of the input's type."


@COMPILE_PROBLEMS.problem("invalid_value", variant="bound_default")
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


        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "f": {"node": "first", "inputs": {"one": {"sources": [{"node": "u", "output": "result"}]}}}
        }})
    """
    return f"The default of '{field}' is not a value of {type}, the type the edges into '{node_id}' give it: {reason.rstrip('.')}. Give a default of that type, or none."


@COMPILE_PROBLEMS.problem(of_a_hook=True)
def parameter_name_invalid(node_id: str, field: str, /) -> str:
    """A node's ``compute_...`` hook returns an input or output name that cannot be one.

    A name may not be empty or contain ``.``, ``/``, ``@`` or ``~``, the characters addresses (``a.result``), node
    paths (``loud/shout``) and waiting ids (``a@1~3fa2``) are written with. Every name in a graph record is held to
    the same rule. Change the hook.

    Args:
        node_id: The node.
        field: The name.

    Example:
        def dotted(names: str) -> dict[str, type]:
            return {name: str for name in names.split(",")}


        @registry.node("fields", title="Fields", description="Gives one answer per name.", compute_outputs=dotted)
        def fields(names: str) -> dict[str, str]:
            return {}


        problems({"nodes": {
            "f": {"node": "fields", "inputs": {"names": {"constant": "a.b"}}}
        }})
    """
    return f"'{field}' cannot be an input or output name: a name is not empty and has no '.', '/', '@' or '~'; change the node's hook."


@COMPILE_PROBLEMS.problem()
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


        problems({"nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "r": {"node": "route", "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}}},
            "p": {"node": "pair", "inputs": {"a": {"sources": [{"node": "r", "output": "short"}]}, "b": {"sources": [{"node": "r", "output": "long"}]}}}
        }})
    """
    return (
        f"{comma_separated(inputs)} read branches that never both have a value for the same item, so the node would get no items. "
        "Write Series[X | None] on them to keep every item, or wire both branches into one input."
    )


@COMPILE_PROBLEMS.problem()
def series_not_nested(node_id: str, /, *, series: list[str], inputs: list[str]) -> str:
    """A node reads one item of each of two series, and neither series is returned inside the loop over the other.

    In the example ``g`` would be called once per name and once per greeting, and nothing says which name goes with
    which greeting. Wire an item of one series into the node that returns the other, so that node is called once per
    item of the first and returns its series inside that loop. Or give one of the series whole to a ``Series`` input,
    so it arrives as one list.

    Args:
        node_id: The node.
        series: The two series, each as its node's path and output: ``names.result``.
        inputs: The two inputs that read them.

    Example:
        problems({"nodes": {
            "names": {"node": "words", "inputs": {"text": {"graph_input": "names_text"}}},
            "greetings": {"node": "words", "inputs": {"text": {"graph_input": "greetings_text"}}},
            "g": {"node": "greet", "inputs": {"name": {"sources": [{"node": "names", "output": "result"}]}, "greeting": {"sources": [{"node": "greetings", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "g", "output": "result"}}})
    """
    return (
        f"'{node_id}' reads items of {comma_separated(series)}, and neither is inside the loop over the other, "
        "so nothing says which items go together. "
        "Wire an item of one into the node that returns the other, or give one of them whole to a Series input."
    )


@COMPILE_PROBLEMS.problem("series_not_nested", variant="inside_subgraph")
def loops_not_nested_inside_subgraph(node_id: str, /, *, series: list[str], inputs: list[str], subgraphs: list[str]) -> str:
    """As ``series_not_nested``, where the node that returns one of the two series is inside a subgraph.

    No edge of the outer graph can change how a subgraph loops. Give one of the series whole to a ``Series`` input,
    or change the subgraph so it returns both outputs from the same loop.

    Args:
        node_id: The node.
        series: The two series, each as its node's path and output; a series inside a subgraph is named by the path
            of the node inside, from this graph: ``e/words.result``.
        inputs: The two inputs that read them.
        subgraphs: The ids of the subgraph nodes the series are inside.

    Example:
        registry.subgraph("split", title="Split", description="Returns the words of a text.")(make_graph(words))
        problems({"nodes": {
            "e": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "g": {"node": "greet", "inputs": {"name": {"sources": [{"node": "w", "output": "result"}]}, "greeting": {"sources": [{"node": "e", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "g", "output": "result"}}})
    """
    return (
        f"'{node_id}' reads items of {comma_separated(series)}, and neither is inside the loop over the other, "
        "so nothing says which items go together. "
        f"One is inside a subgraph ({comma_separated(subgraphs)}), so no edge here can change that. "
        "Give one of them whole to a Series input, or change the subgraph so it returns both from the same loop."
    )


@COMPILE_PROBLEMS.problem()
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
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"graph_input": "a_text"}}},
            "b": {"node": "upper", "inputs": {"text": {"graph_input": "b_text"}}},
            "c": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}, {"node": "b", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "c", "output": "result"}}})
    """
    return (
        f"More than one edge into '{field}' (from {comma_separated(sources)}) can give it a value on the same call. "
        "Wire it from different branches of one node, or add a node with an input for each."
    )


@COMPILE_PROBLEMS.problem()
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
        problems({"nodes": {
            "c": {"node": "count", "inputs": {"n": {"graph_input": "n"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' gives {got}, but '{node_id}.{field}' takes {expected}; "
        f"connect an output that gives {expected}, or convert it in a node between."
    )


@COMPILE_PROBLEMS.problem("type_mismatch", variant="make_a_series")
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
        problems({"nodes": {
            "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' is a whole {whole}, and '{node_id}.{field}' takes one {expected}. To call '{node_id}' once per row or item, "
        "put a node between them that returns its rows or items as a Series, and wire that node's result, or one field of it."
    )


@COMPILE_PROBLEMS.problem("type_mismatch", variant="add_column")
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


        problems({"nodes": {
            "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
            "t": {"node": "total", "inputs": {"invoices": {"sources": [{"node": "c", "output": "result"}]}}}
        }})
    """
    return (
        f"'{node_id}.{field}' needs the column '{column}' ({column_type}); '{source}' has {comma_separated(has)}. "
        "Add the column in a node between, or connect a table that has it."
    )


@COMPILE_PROBLEMS.problem()
def input_has_no_value(node_id: str, field: str, /) -> str:
    """An input with no default gets nothing in the graph record: no edge, no constant and no graph input.

    Wire an edge into it, give it a constant, or make it a graph input, ``{"graph_input": "name"}``, so the caller
    gives its value.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": {"g": {"node": "greet", "inputs": {"name": {"constant": "Ida"}}}}})
    """
    return f"'{field}' gets no value: wire an edge into it, give it a constant, or make it a graph input."


@COMPILE_PROBLEMS.problem()
def graph_input_used_two_ways(node_id: str, field: str, /, *, graph_input: str) -> str:
    """Two inputs that name one graph input take different types, or the record gives them different defaults.

    A caller gives a graph input one value, read by one type, so every input it feeds must take that type. Its default
    is the record's, written on any of them (and the same wherever it is written twice); an input's own default plays
    no part. Give the inputs graph inputs of their own, or make them agree.

    Args:
        node_id: The node of the second input.
        field: The second input.
        graph_input: The graph input's name.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
            "c": {"node": "count", "inputs": {"n": {"graph_input": "x"}}},
        }})
    """
    return f"'{field}' and another input both take the graph input '{graph_input}' with different types or defaults; give each its own graph input."


@COMPILE_PROBLEMS.problem()
def unknown_input(node_id: str, field: str, /, *, inputs: list[str]) -> str:
    """A constant or an edge is for an input the node does not have.

    Use one of the inputs ``ctx.inputs`` lists. When a newer version of the node renamed the input,
    ``registry.upgrade`` moves the constant to the new name.

    Args:
        node_id: The node.
        field: The name that is not an input of the node.
        inputs: The node's inputs.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"txt": {"constant": "a"}, "text": {"graph_input": "text"}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return f"'{node_id}' has no input '{field}'; it has {comma_separated(inputs)}."


@COMPILE_PROBLEMS.problem()
def unknown_node(*loc: str | int, node: str, choices: list[str]) -> str:
    """A node of the graph record names a node definition the registry does not have.

    Use one the registry has (``ctx.choices``), or add the missing one to the registry the graph is compiled with.

    Args:
        loc: The node.
        node: The id of the node definition it names.
        choices: The ids of the registry's node definitions.

    Example:
        problems({"nodes": {
            "m": {"node": "send-mail"}
        }})
    """
    return f"There is no node '{node}'; the nodes are {comma_separated(choices)}."


@COMPILE_PROBLEMS.problem()
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
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "text"}]}}}
        }, "outputs": {"result": {"node": "b", "output": "result"}}})
    """
    return f"'{source}' is not an output of that node; it has {comma_separated(outputs)}."


@COMPILE_PROBLEMS.problem("unknown_output", variant="make_a_series")
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
        problems({"nodes": {
            "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result", "field": "email"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' names a field of a whole {whole}, which has no fields to wire. Put a node after it that returns its rows "
        "or items as a Series, and wire the field from that node's result."
    )


@COMPILE_PROBLEMS.problem()
def unknown_version(*loc: str | int, node: str, version: int, newest: int) -> str:
    """A node of the graph record names a version its node definition does not have.

    Use a version the node definition has. ``registry.upgrade(record, node_id)`` moves a saved node to the newest.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the graph record names.
        newest: The newest version of the node definition.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "version": 9, "inputs": {"text": {"graph_input": "text"}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return f"'{node}' has no version {version}; the newest is {newest}."
