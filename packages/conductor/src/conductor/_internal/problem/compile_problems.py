"""The problems compiling a graph can find, listed in ``graph.problems``.

This is the "Compiling a graph" section of the problems page. Compiling never raises for these. It lists them, so an
editor can show them while the graph is being drawn, and a graph that has any of them doesn't run.

Code that reports one imports this module and calls the kind, like ``compile_problems.cycle(node_id)``.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor._internal.utils import comma_separated

from .type import ProblemCatalogue

#: This module's section of the problems page, with its kinds.
COMPILE_PROBLEMS = ProblemCatalogue(
    "Compiling a graph",
    "These are listed in `graph.problems`. Compiling never raises for them, but a graph with any of them doesn't "
    "run. `loc` is `(node_id,)` or `(node_id, input)`, or `()` for a problem with the graph's declared outputs. For a "
    'node inside a subgraph, the subgraph node\'s id comes first: `("loud", "shout", "line")`.',
)


@COMPILE_PROBLEMS.problem()
def alternatives_from_different_loops(
    node_id: str, field: str, /, *, series: list[str | None]
) -> str:
    """Two edges into one input come from branches whose values are inside different loops.

    A call of a node inside a loop is for one item of that loop. So it can only take values from the same loop. In the
    example, ``u.text`` has an edge from each branch of ``r``. The long branch gives one value for each item of
    ``w.result``. The short branch goes through ``letters``, which returns a series of its own, so it gives one value
    for each item of ``letters.result``. A call of ``u`` can never have a value from both.

    To fix it, collect the inner loop first, in a node with a ``Series`` input. Or wire the input from one branch only.

    Args:
        node_id: The node.
        field: The input.
        series: For each edge, the series its values are items of, as its node's path and output, like
            ``lines.result``. A node inside the subgraph node ``e`` is written ``e/words.result``. ``None`` for an
            edge from a node that is in no loop.

    Example:
        problems({"nodes": {
            "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
            "r": {"node": "route", "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}}},
            "letters": {"node": "words", "inputs": {"text": {"sources": [{"node": "r", "output": "short"}]}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "letters", "output": "result"}, {"node": "r", "output": "long"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    series_text = comma_separated(
        ["none (not in a loop)" if one is None else one for one in series]
    )
    return (
        f"'{field}' has edges from branches inside different loops: {series_text}. One call can never get a value from both. "
        f"Collect the inner loop first, or wire '{field}' from one branch only."
    )


@COMPILE_PROBLEMS.problem()
def compute_hook_failed(
    *loc: str | int, reason: str, error: str, error_ctx: dict[str, Any] | None = None
) -> str:
    """A node's ``compute_...`` hook raised a ``NodeError``, or returned something compiling can't use.

    A ``NodeError`` keeps its message, since the hook's developer wrote it for people. ``ctx.error`` is its
    ``code`` and ``ctx.error_ctx`` its ``ctx``. A hook that raises any other exception is the variant ``raised``.

    A hook can also return a name a field can't have. Then the problem is at that name, and ``ctx.error`` is
    ``not_a_name``. A name must not be empty, and must not contain ``.``, ``/``, ``@`` or ``~``, since those are used to
    write sources, node paths and waiting ids.

    To fix it, change the constant the hook reads, or fix the hook. In the example, the hook of ``pick_some`` raises
    a ``NodeError`` for an empty text.

    Args:
        loc: The node. If the hook gave a field a name it can't have, that name comes next.
        reason: The ``NodeError``'s message, or why compiling can't use what the hook returned.
        error: The ``code`` of the hook's ``NodeError``. If the hook returned something compiling can't use, it says
            why: ``not_a_mapping``, ``not_a_name``, ``added_input_exists``, or the code of a type conductor doesn't
            support.
        error_ctx: The ``ctx`` of the hook's ``NodeError``, if it has one.

    Example:
        def some_names(names: str) -> dict[str, type]:
            if not names:
                raise NodeError("No output is named", code="no_names")
            return {name: str for name in names.split(",")}


        @registry.node("pick_some", title="Pick some", description="Returns one output per name.", compute_outputs=some_names)
        def pick_some(names: str) -> dict[str, str]:
            return {name: name for name in names.split(",")}


        problems({"nodes": {
            "pick": {"node": "pick_some", "inputs": {"names": {"constant": ""}}}
        }})
    """
    # A reason that ends in a period is a NodeError's whole message, with its own fix.
    if reason.endswith("."):
        return f"The node couldn't work out its inputs and outputs: {reason}"
    return f"The node couldn't work out its inputs and outputs: {reason.rstrip('.')}. Check the node's constants and what is wired into it."


@COMPILE_PROBLEMS.problem("compute_hook_failed", variant="raised")
def compute_hook_raised(*loc: str | int, error: str) -> str:
    """A node's ``compute_...`` hook raised an exception that isn't a ``NodeError``.

    A problem may be shown to anyone, and the exception's text might quote a value. So the text and the traceback go
    to the ``conductor`` logger, and ``ctx.error`` only has the exception's class name.

    To fix it, change the constant the hook reads, or fix the hook. In the example, ``pick``'s hook raises
    ``ValueError`` for an empty text.

    Args:
        loc: The node.
        error: The exception's class name.

    Example:
        problems({"nodes": {
            "pick": {"node": "pick", "inputs": {"names": {"constant": ""}}}
        }})
    """
    return (
        f"The node couldn't work out its inputs and outputs: its hook raised {error}. Check the node's constants and "
        "what is wired into it; its developer can find the details in the conductor log."
    )


@COMPILE_PROBLEMS.problem()
def context_does_not_fit(node_id: str, parameter: str, /, *, needs: str, context_type: str) -> str:
    """A node takes the run's context as a class that the registry's ``context_type`` isn't.

    The registry names the class of the context a host passes to each run, with ``NodeRegistry(context_type=...)``. A
    node asks for it as ``Context[X]``. That fits when ``X`` is the registry's class, a base class of it, or a
    ``Protocol`` the class satisfies.

    To fix it, ask for one of those, or name another class on the registry. In the example, ``count`` asks for an
    ``int``.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, which is its first.
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
    return f"'{node_id}' asks for the run's context as {needs}, but the registry's context class is {context_type}. Ask for {context_type}, a base class of it, or a Protocol it satisfies."


@COMPILE_PROBLEMS.problem("context_does_not_fit", variant="lacks")
def context_does_not_fit_lacks(
    node_id: str, parameter: str, /, *, needs: str, context_type: str, lacks: list[str]
) -> str:
    """A node takes the run's context as a ``Protocol``, and the registry's ``context_type`` lacks some of its members.

    Compiling only checks that a member of each name is there. Whether its type fits is for the host's type checker to
    say.

    To fix it, give the registry's class each member the node reads, or change what the node asks for. In the example,
    ``HostContext`` has no ``database``.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, which is its first.
        needs: The name of the ``Protocol`` inside ``Context[...]``.
        context_type: The name of the registry's class.
        lacks: The name of each member the class doesn't have.

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
    return f"'{node_id}' asks for the run's context as {needs}, but the registry's context class, {context_type}, has no {comma_separated(lacks)}. Add what the node reads to {context_type}, or change what the node asks for."


@COMPILE_PROBLEMS.problem("context_does_not_fit", variant="no_context_type")
def context_does_not_fit_no_context_type(node_id: str, parameter: str, /, *, needs: str) -> str:
    """A node takes the run's context, and the registry doesn't name a ``context_type``.

    To fix it, name the class of the context your runs get, like ``NodeRegistry(context_type=HostContext)``.

    A registry of nodes that don't know about any host, such as a package of nodes, names no class. Add it to your own
    registry with ``add_subregistry``. Its nodes are then checked against your class.

    Args:
        node_id: The node.
        parameter: The node's ``Context`` parameter, which is its first.
        needs: The name of the class inside ``Context[...]``.

    Example:
        plain = NodeRegistry()
        plain.add(who)
        problems(make_graph(who), plain)
    """
    return f"'{node_id}' asks for the run's context as {needs}, but the registry doesn't say what the context is. Name its class with NodeRegistry(context_type=...)."


@COMPILE_PROBLEMS.problem()
def cycle(*loc: str | int) -> str:
    """A node is on a cycle of edges.

    A graph calls each node after the nodes upstream of it, so a graph with a cycle can't run. Every node on the cycle
    gets this problem.

    To fix it, remove one of the edges that close the cycle.

    Args:
        loc: The node.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}},
            "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}
        }})
    """
    return "The node is part of a cycle, so the graph can't run. Remove one of the edges that close the cycle."


@COMPILE_PROBLEMS.problem()
def graph_output_not_found(*, output: str, source: str, outputs: list[str]) -> str:
    """A graph output names a node the record doesn't have, or an output its node doesn't have.

    The problem belongs to the graph, not to a node. So its ``loc`` is empty, and ``ctx.output`` names the graph output.

    To fix it, name an output of a node the graph has.

    Args:
        output: The graph output's name.
        source: The output it names, as ``node.output``.
        outputs: The outputs its node has, each as ``node.output``. Empty if the record has no such node.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}
        }, "outputs": {"result": {"node": "u", "output": "shout"}}})
    """
    if outputs:
        return f"The graph output '{output}' reads '{source}', but that node has no such output. It has {comma_separated(outputs)}."
    return f"The graph output '{output}' reads '{source}', but the graph has no such node. Name an output of a node in the graph."


@COMPILE_PROBLEMS.problem()
def edge_names_missing_node(*loc: str | int, missing: str, source: str, target: str) -> str:
    """An edge comes from a node the graph record doesn't have.

    To fix it, add the node, or remove the edge.

    Args:
        loc: The node the edge leads into, and the input.
        missing: The id of the node that isn't in the graph record.
        source: The edge's source, as text, like ``"a.result"``.
        target: The input the edge leads into, as ``node.input``.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "ghost", "output": "result"}]}}}
        }})
    """
    return f"The edge from '{source}' to '{target}' comes from '{missing}', which isn't a node in the graph. Add that node, or remove the edge."


@COMPILE_PROBLEMS.problem()
def subgraph_contains_itself(*loc: str | int, node: str) -> str:
    """A subgraph contains itself, directly or through another subgraph.

    The problem is on the node inside the subgraph that leads back. So its ``loc`` starts with the subgraph node's id. In
    the example, it is ``("i", "again")``.

    To fix it, remove the node that leads back.

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
    return f"The subgraph '{node}' contains itself. Remove the node that leads back to it."


@COMPILE_PROBLEMS.problem()
def invalid_value(
    node_id: str,
    field: str,
    /,
    *,
    reason: str,
    errors: list[dict[str, Any]] | None = None,
    column: str | None = None,
) -> str:
    """A constant in the graph record, or a graph input's default, isn't a value of its input's type.

    A problem may be shown to anyone, so it never quotes the value. ``ctx.reason`` says what the type refused.
    ``ctx.errors`` is pydantic's list of what it refused, each with a ``type``, a ``loc`` and a ``msg``. A host can key
    on a ``type`` such as ``greater_than_equal`` or ``bool_is_not_a_number``. Each ``loc`` only has list indexes and the
    fields the type declares. It never has a key of a dict, since a key is part of the value.

    To fix it, give a value of the input's type.

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
    return f"The graph record gives '{field}' a value that isn't valid: {reason.rstrip('.')}. Give it a value of the input's type."


@COMPILE_PROBLEMS.problem("invalid_value", variant="bound_default")
def invalid_value_bound_default(
    node_id: str, field: str, /, *, type: str, reason: str, errors: list[dict[str, Any]]
) -> str:
    """A default the node declares doesn't fit the type its edges give a type variable.

    Take ``first(one: T, other: T = 0)``. Wired from a ``str`` output, ``T`` becomes ``str``, and ``0`` isn't a ``str``.
    This only matters for an input that uses its default. An input that gets a value from an edge never uses its
    default, so its default isn't checked.

    To fix it, give the input a default that fits every type its edges may give ``T``. Or give it a constant or an edge
    in the graph, or declare it with no default.

    Args:
        node_id: The node.
        field: The input or output.
        type: The type its edges give the type variable.
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
    return f"The edges into '{node_id}' make '{field}' of type {type}, and its default doesn't fit: {reason.rstrip('.')}. Give it a constant or an edge, or change its default."


@COMPILE_PROBLEMS.problem()
def lists_from_different_loops(*loc: str | int, inputs: list[str]) -> str:
    """A node has two ``Series`` inputs, and they read branches that never both have a value for the same item.

    A node with two ``Series`` inputs leaves out an item when one of them has no value for it. In the example, each word
    is either short or long, never both. So every item would be left out.

    To fix it, write ``Series[X | None]`` on both inputs. That keeps every item, with ``None`` for the skipped
    branch. Or wire both branches into one input.

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
        f"{comma_separated(inputs)} read branches that never have a value for the same item, so the node would get no items at all. "
        "Write Series[X | None] on both to keep every item, or wire both branches into one input."
    )


@COMPILE_PROBLEMS.problem()
def series_not_nested(node_id: str, /, *, series: list[str], inputs: list[str]) -> str:
    """A node reads one item from each of two series, and nothing says which items go together.

    A node that reads an item of a series is called once for each item. With two series, that only works when one is
    returned inside the loop over the other. In the example, ``g`` would be called once per name and once per greeting,
    and nothing says which name goes with which greeting.

    There are two ways to fix it. Wire an item of one series into the node that returns the other. That node is then
    called once for each item of the first, and returns its series inside that loop. Or give one of the series whole to
    a ``Series`` input, so it arrives as one list.

    Args:
        node_id: The node.
        series: The two series, each as its node's path and output, like ``names.result``.
        inputs: The two inputs that read them.

    Example:
        problems({"nodes": {
            "names": {"node": "words", "inputs": {"text": {"graph_input": "names_text"}}},
            "greetings": {"node": "words", "inputs": {"text": {"graph_input": "greetings_text"}}},
            "g": {"node": "greet", "inputs": {"name": {"sources": [{"node": "names", "output": "result"}]}, "greeting": {"sources": [{"node": "greetings", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "g", "output": "result"}}})
    """
    return (
        f"'{node_id}' reads items from {comma_separated(series)}, but nothing says which items go together. "
        "Wire an item of one into the node that returns the other, or give one of them whole to a Series input."
    )


@COMPILE_PROBLEMS.problem("series_not_nested", variant="inside_subgraph")
def loops_not_nested_inside_subgraph(
    node_id: str, /, *, series: list[str], inputs: list[str], subgraphs: list[str]
) -> str:
    """The same as ``series_not_nested``, but one of the two series is returned inside a subgraph.

    No edge in the outer graph can change how a subgraph loops. So the first fix of ``series_not_nested`` doesn't work
    here.

    To fix it, give one of the series whole to a ``Series`` input. Or change the subgraph so it returns both outputs
    from the same loop.

    Args:
        node_id: The node.
        series: The two series, each as its node's path and output. A series inside a subgraph is named by the path of
            the node inside, from this graph, like ``e/words.result``.
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
        f"'{node_id}' reads items from {comma_separated(series)}, but nothing says which items go together. "
        f"One of them is made inside a subgraph ({comma_separated(subgraphs)}), so no edge here can fix that. "
        "Give one of them whole to a Series input, or change the subgraph so it returns both from the same loop."
    )


@COMPILE_PROBLEMS.problem()
def several_edges_into_one_input(node_id: str, field: str, /, *, sources: list[str]) -> str:
    """More than one edge into an input can give it a value on the same call.

    An input can take several edges only if at most one of them has a value on each call. That is the case when they
    come from different branches of one node, or from nodes downstream of them. A branch is an output whose default is
    ``SKIPPED``, and a call gives a value to at most one branch of a node. A ``Series`` input follows the same rule.

    To fix it, give the outputs the default ``SKIPPED`` to make them branches of one node, if a call gives a value to
    only one of them. Or add a node with an input for each edge, and wire them there.

    Args:
        node_id: The node.
        field: The input.
        sources: The edges' sources, as text.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"graph_input": "a_text"}}},
            "b": {"node": "upper", "inputs": {"text": {"graph_input": "b_text"}}},
            "c": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}, {"node": "b", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "c", "output": "result"}}})
    """
    return (
        f"'{field}' has edges from {comma_separated(sources)}, and more than one of them can give it a value on the same call. "
        "Only different branches of one node may share an input: give those outputs the default SKIPPED to make them "
        "branches, if a call gives a value to only one of them. Or add a node with an input for each edge."
    )


@COMPILE_PROBLEMS.problem()
def type_mismatch(node_id: str, field: str, /, *, source: str, got: str, expected: str) -> str:
    """An edge's source gives a type that its input doesn't take.

    To fix it, wire an output of the type the input takes. Or put a node in between that converts one type into the
    other.

    Two common mismatches have a message of their own, below. ``ctx.variant`` says which one it is.

    Args:
        node_id: The node the edge leads into.
        field: The input.
        source: The edge's source, as text, like ``"a.result"``.
        got: The type the source gives.
        expected: The type the input takes.

    Example:
        problems({"nodes": {
            "c": {"node": "count", "inputs": {"n": {"graph_input": "n"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' gives {got}, but '{node_id}.{field}' takes {expected}. "
        f"Wire an output that gives {expected}, or convert the value in a node in between."
    )


@COMPILE_PROBLEMS.problem("type_mismatch", variant="make_a_series")
def type_mismatch_make_a_series(
    node_id: str,
    field: str,
    /,
    *,
    source: str,
    got: str,
    expected: str,
    kind: Literal["table", "list"],
    fields: list[str],
) -> str:
    """A whole table or list is wired into an input that takes one value.

    A table or a list is one value. To call the node once for each row or item, put a node in between that returns the
    rows or items as a ``Series``, that is, a node declared ``-> Series[R]``. Then wire that node's result into the
    input, or one field of each record, like ``each.result.email``.

    Args:
        node_id: The node the edge leads into.
        field: The input.
        source: The edge's source, as text, like ``"a.result"``.
        got: The type the source gives.
        expected: The type the input takes.
        kind: Whether the source is a table or a list.
        fields: The fields each row or item has, so an editor can offer them. Empty if they aren't known.

    Example:
        problems({"nodes": {
            "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' is a whole {kind}, but '{node_id}.{field}' takes one {expected}. To call '{node_id}' once per row or item, "
        "put a node in between that returns them as a Series. Then wire its result, or one field of it."
    )


@COMPILE_PROBLEMS.problem("type_mismatch", variant="add_column")
def type_mismatch_add_column(
    node_id: str,
    field: str,
    /,
    *,
    source: str,
    got: str,
    expected: str,
    column: str,
    column_type: str,
    has: list[str],
) -> str:
    """A table is wired into a ``Table[R]`` input, and the table lacks a column the input needs.

    To fix it, wire a table that has every column the input needs, or add the column in a node in between. Or type the
    input as a bare ``Table``, which takes any columns.

    Args:
        node_id: The node the edge leads into.
        field: The input.
        source: The edge's source, as text, like ``"a.result"``.
        got: The type the source gives.
        expected: The type the input takes.
        column: The first column the input needs that the table doesn't have.
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
        f"'{node_id}.{field}' needs the column '{column}' ({column_type}), and '{source}' doesn't have it. It has {comma_separated(has)}. "
        "Add the column in a node in between, or wire a table that has it."
    )


@COMPILE_PROBLEMS.problem()
def input_has_no_value(node_id: str, field: str, /) -> str:
    """An input with no default gets no value from the graph record: no edge, no constant and no graph input.

    To fix it, wire an edge into it, or give it a constant. Or make it a graph input, like
    ``{"graph_input": "name"}``, so the caller gives its value.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": {"g": {"node": "greet", "inputs": {"name": {"constant": "Ida"}}}}})
    """
    return f"'{field}' gets no value. Wire an edge into it, give it a constant, or make it a graph input."


@COMPILE_PROBLEMS.problem("input_has_no_value", variant="constant_only")
def input_has_no_value_constant_only(node_id: str, field: str, /) -> str:
    """An input that only takes a constant has no default, and the graph record gives it no constant.

    An input whose description says ``constant_only`` gets its value from the graph record when the graph is compiled.
    No edge or caller can give it one.

    To fix it, give it a constant.

    Args:
        node_id: The node.
        field: The input.

    Example:
        problems({"nodes": {
            "pick": {"node": "pick"}
        }})
    """
    return f"'{field}' gets no value, and it only takes a constant. Give it one."


@COMPILE_PROBLEMS.problem()
def input_takes_only_a_constant(
    node_id: str, field: str, /, *, given: Literal["sources", "graph_input"]
) -> str:
    """The graph record wires an edge or a graph input into an input that only takes a constant.

    An input whose description says ``constant_only`` gets its value from the record's constant, or from its default.
    It never gets one from an edge or a caller. ``ctx.given`` says what the record gives it instead.

    To fix it, give it a constant, like ``{"constant": ...}``. Or leave it out, so it takes its default.

    Args:
        node_id: The node.
        field: The input.
        given: The record's key for what it gives the input: ``sources`` for edges, or ``graph_input``.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "pick": {"node": "pick", "inputs": {"names": {"sources": [{"node": "u", "output": "result"}]}}}
        }})
    """
    return f"'{field}' only takes a constant, not an edge or a graph input. Give it a constant."


@COMPILE_PROBLEMS.problem()
def graph_input_used_two_ways(node_id: str, field: str, /, *, graph_input: str) -> str:
    """Two inputs take the same graph input, and they don't agree on it.

    A caller gives a graph input one value, and a host shows it once. So every input it feeds must agree on three
    things:

    - The type, as written in its hint, with its limits, choices and validators. ``int`` and
      ``Annotated[int, Param()]`` agree. But one rule written two ways, such as ``annotated_types.Ge(0)`` and
      ``Param(ge=0)``, counts as two types, even though both take the same values.
    - How a host shows it: the same title, description and widget.
    - The default. A graph input's default comes from the record, and may be written on any of the inputs. Where it is
      written twice, it must be the same. An input's own default plays no part.

    To fix it, give each input a graph input of its own, or make them agree.

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
    return f"'{field}' and another input both take the graph input '{graph_input}', but they don't agree on its type, how it's shown or its default. Give each input a graph input of its own."


@COMPILE_PROBLEMS.problem()
def unknown_input(node_id: str, field: str, /, *, inputs: list[str]) -> str:
    """A constant or an edge is for an input the node doesn't have.

    To fix it, use one of the inputs in ``ctx.inputs``. If a newer version of the node renamed the input,
    ``registry.upgrade`` moves the constant to the new name.

    Args:
        node_id: The node.
        field: The name that isn't an input of the node.
        inputs: The node's inputs.

    Example:
        problems({"nodes": {
            "u": {"node": "upper", "inputs": {"txt": {"constant": "a"}, "text": {"graph_input": "text"}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return f"'{node_id}' has no input '{field}'. Its inputs are {comma_separated(inputs)}."


@COMPILE_PROBLEMS.problem()
def unknown_node(*loc: str | int, node: str, choices: list[str]) -> str:
    """A node in the graph record names a node definition the registry doesn't have.

    To fix it, use one the registry has, listed in ``ctx.choices``. Or add the missing one to the registry the graph is
    compiled with.

    Args:
        loc: The node.
        node: The id of the node definition it names.
        choices: The ids of the registry's node definitions.

    Example:
        problems({"nodes": {
            "m": {"node": "send-mail"}
        }})
    """
    if not choices:
        return f"The registry has no node '{node}'. It has no nodes at all."
    return f"The registry has no node '{node}'. Its nodes are {comma_separated(choices)}."


@COMPILE_PROBLEMS.problem()
def unknown_output(
    *loc: str | int, source: str, outputs: list[str], record_field: str | None = None
) -> str:
    """An edge's source is an output, or a field of one, that its node doesn't have.

    An edge may name one field of a record that an output gives, like ``node.output.field``.

    To fix it, wire from one of the outputs in ``ctx.outputs``. The list has those fields too.

    Args:
        loc: The node the edge leads into, and the input.
        source: The edge's source, as text, like ``"a.result"``.
        outputs: The outputs the source's node has, as text.
        record_field: The field of a record the edge names, if the record has no field by that name.

    Example:
        problems({"nodes": {
            "a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "text"}]}}}
        }, "outputs": {"result": {"node": "b", "output": "result"}}})
    """
    return f"'{source}' isn't an output. That node has {comma_separated(outputs)}."


@COMPILE_PROBLEMS.problem("unknown_output", variant="make_a_series")
def unknown_output_make_a_series(
    *loc: str | int,
    source: str,
    outputs: list[str],
    record_field: str,
    kind: Literal["table", "list"],
    fields: list[str],
) -> str:
    """An edge's source names a field of a whole table or list.

    A table or a list is one value, so it has no field to wire. To fix it, put a node after it that returns its rows or
    items as a ``Series``, that is, a node declared ``-> Series[R]``. Then wire the field from that node's result, like
    ``each.result.email``.

    Args:
        loc: The node the edge leads into, and the input.
        source: The edge's source, as text, like ``"a.result"``.
        outputs: The outputs the source's node has, as text.
        record_field: The field the edge names.
        kind: Whether the source is a table or a list.
        fields: The fields each row or item has, so an editor can offer them. Empty if they aren't known.

    Example:
        problems({"nodes": {
            "c": {"node": "contacts", "inputs": {"text": {"graph_input": "text"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "c", "output": "result", "field": "email"}]}}}
        }, "outputs": {"result": {"node": "u", "output": "result"}}})
    """
    return (
        f"'{source}' names a field of a whole {kind}, and a whole {kind} has no fields to wire. Put a node after it that returns its rows "
        "or items as a Series. Then wire the field from that node's result."
    )


@COMPILE_PROBLEMS.problem()
def unknown_version(*loc: str | int, node: str, version: int, newest: int) -> str:
    """A node in the graph record names a version its node definition doesn't have.

    To fix it, use a version the node definition has. ``registry.upgrade(record, node_id)`` moves a saved node to the
    newest version.

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
    return f"'{node}' has no version {version}. The newest is {newest}."
