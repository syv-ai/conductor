"""The problems a refused call raises, in its ``InvalidCallError``.

This is the "Refused calls" section of the problems page. A call such as ``graph.run`` or ``run.answer`` checks what
it is given before it does anything. If something is wrong, it raises ``InvalidCallError`` with these problems, and
nothing changes.

Code that reports one imports this module and calls the kind, like
``call_problems.missing_input("inputs", name, name=name)``.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.utils import comma_separated

from .type import ProblemCatalogue

#: This module's section of the problems page, with its kinds.
CALL_PROBLEMS = ProblemCatalogue(
    "Refused calls",
    "A call raises these as `InvalidCallError` when it is given something it can't act on. That is `graph.run`, "
    "`run.answer`, `Run.from_json`, `make_graph` and the like, and `error.problems` lists them. `loc` says what in the "
    'call is wrong. It can be an argument (`("timeout",)`), a name inside one (`("inputs", "text")`), the steps to the '
    'fault in a graph record or a saved run (`("record", "nodes", "u", "node")`), or the node the problem is about, '
    'with the parameter that is wrong (`("approve",)`, `("who", "ctx")`).',
)


@CALL_PROBLEMS.problem()
def answer_stale(
    *loc: str | int, waiting_id: str, current: str, moved_to: str | None = None
) -> str:
    """An answer is for a question that a newer question has replaced.

    A run made with other inputs, by ``run.again`` or ``run.with_inputs``, asks again wherever a call's inputs changed.
    The new question has a new waiting id.

    To fix it, answer the question that is waiting now. ``ctx.current`` is its id, and ``run.waiting`` lists it.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        current: The waiting id of the question that waits in its place.
        moved_to: The waiting id the same question has now, if it waits for another call of the node. That happens
            when a change of inputs gave its item another index.

    Example:
        asked = Graph.compile(make_graph(approve), registry).run(amount=5000)
        asked_again = asked.again(amount=5500)
        refused(lambda: asked_again.answer(asked.waiting[0].id, True))
    """
    return f"This answer is for {waiting_id!r}, but a newer question, {current!r}, has replaced it. Answer that one instead."


@CALL_PROBLEMS.problem()
def cannot_move_to_version(
    *loc: str | int, node: str, version: int, target: int, versions: list[int]
) -> str:
    """``registry.upgrade`` is asked to move a node to a version it can't reach from the version it is at.

    To fix it, ask for a version in ``ctx.versions``, or leave ``to`` out to get the newest.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the node is at.
        target: The version asked for.
        versions: The versions of the node definition.

    Example:
        refused(lambda: registry.upgrade(make_graph(upper), "upper", to=3))
    """
    return f"'{node}' can't move from version {version} to version {target}. Its versions are {comma_separated(versions)}."


@CALL_PROBLEMS.problem()
def context_missing(*loc: str | int, parameter: str, needs: str) -> str:
    """A node takes the run's context, and the call that starts or resumes the run gives no ``context``.

    A run never saves its context, so every call that goes on with the run must pass it again. Every node that takes
    the context gets this problem.

    To fix it, pass the object on each such call: ``graph.run(context=HostContext(...))``, ``run.resume(context=...)``
    and ``run.stream(context=...)``.

    Args:
        loc: The node's path, then its ``Context`` parameter.
        parameter: The node's ``Context`` parameter, which is its first.
        needs: The name of the class inside ``Context[...]``.

    Example:
        refused(lambda: Graph.compile(make_graph(who), registry).run())
    """
    return f"The node takes the run's context as '{parameter}', of class {needs}, but the call gave no context. Pass context= on every call that starts or resumes the run."


@CALL_PROBLEMS.problem()
def invalid_answer(
    *loc: str | int, waiting_id: str, reason: str, question: str | None = None
) -> str:
    """An answer isn't a value of the type the question takes.

    Answers are checked the way their JSON would be, so a yes or no is ``true``, not ``"yes"``.

    To fix it, answer with a value of the question's type.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        reason: What is wrong with the answer.
        question: For a node that asks several questions, one per output, the name of the question that is wrong.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run(amount=5000)
        refused(lambda: waiting.answer(waiting.waiting[0].id, "yes"))
    """
    return f"That isn't a valid answer: {reason.rstrip('.')}."


@CALL_PROBLEMS.problem()
def invalid_context(*loc: str | int, expected: str, got: str) -> str:
    """A call gives ``context`` an object that isn't an instance of the class the registry names.

    To fix it, pass an instance of the registry's ``context_type``, or of a subclass of it.

    Args:
        loc: ``"context"``.
        expected: The name of the registry's class.
        got: The name of the class of the object given.

    Example:
        refused(lambda: Graph.compile(make_graph(who), registry).run(context=Caller("Ida")))  # type: ignore[arg-type]
    """
    return f"The context given is of class {got}, but the registry's context class is {expected}. Pass an instance of {expected}."


@CALL_PROBLEMS.problem("invalid_context", variant="no_context_type")
def invalid_context_no_context_type(*loc: str | int, got: str) -> str:
    """A call gives ``context`` to a run of a graph whose registry names no ``context_type``.

    To fix it, pass no context. Or name the class on the registry, like ``NodeRegistry(context_type=HostContext)``.

    Args:
        loc: ``"context"``.
        got: The name of the class of the object given.

    Example:
        plain = NodeRegistry()
        plain.add(upper)
        refused(lambda: Graph.compile(make_graph(upper), plain).run(text="a", context=Caller("Ida")))  # type: ignore[arg-type]
    """
    return f"The context given is of class {got}, but this graph's registry takes no context. Pass no context, or name its class with NodeRegistry(context_type=...)."


@CALL_PROBLEMS.problem()
def invalid_input(
    *loc: str | int,
    name: str,
    reason: str,
    errors: list[dict[str, Any]] | None = None,
    column: str | None = None,
) -> str:
    """A call gives a graph input a value its type doesn't take.

    ``ctx.reason`` and ``ctx.errors`` say what was refused, just as they do for ``invalid_value``.

    To fix it, give a value of the input's type. ``graph.inputs[name].json_schema`` is the input's schema, with its
    limits.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        reason: Why the value is refused, without quoting it.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        refused(lambda: Graph.compile(make_graph(count), registry).run(n="many"))
    """
    return f"The value given for '{name}' isn't valid: {reason.rstrip('.')}. Give a value of the input's type."


@CALL_PROBLEMS.problem("invalid_input", variant="make_a_series")
def invalid_input_make_a_series(*loc: str | int, name: str, expected: str, length: int) -> str:
    """A call gives a list to a graph input that doesn't take a list.

    A graph only loops over a list when a node returns it as a ``Series``. So a list given to an input of one value is
    refused.

    To fix it, give the list to a node that returns it as a ``Series``, or give one value. A ``Series`` graph input
    takes a list, and takes a single value as a list of one.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        expected: The type the input takes.
        length: How many items the list has.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text=["a", "b"]))
    """
    return (
        f"'{name}' takes one {expected}, but it was given a list of {length}. A graph only loops over a list that a node returns "
        "as a Series. Give the list to such a node, or give one value."
    )


@CALL_PROBLEMS.problem()
def invalid_limit(limit: str, /, *, value: Any) -> str:
    """A call gives ``max_items`` a value that isn't a whole number of at least 1, such as ``0``, ``2.5`` or ``True``.

    A yes or no isn't a number, so ``True`` is refused too.

    To fix it, give a whole number of 1 or more. Or leave it out, for no limit on items.

    Args:
        limit: ``max_items``.
        value: The value given, of any type. ``True`` and ``"5"`` are refused as they are.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text="a", max_items=0))
    """
    return f"{limit} must be a whole number of 1 or more, not {value}. Leave it out for no limit on items."


@CALL_PROBLEMS.problem("invalid_limit", variant="seconds")
def invalid_limit_seconds(limit: str, /, *, value: Any) -> str:
    """A call gives ``timeout`` or ``checkpoint_every`` a value that isn't a finite number of seconds above 0.

    So ``0``, ``inf``, ``nan`` and ``True`` are all refused. A yes or no isn't a number.

    To fix it, give a number of seconds above 0. Or leave the limit out, for no time limit or no checkpoints.

    Args:
        limit: ``timeout`` or ``checkpoint_every``.
        value: The value given, of any type. ``True`` and ``"5"`` are refused as they are.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).with_inputs(text="a").resume(timeout=0))
    """
    return f"{limit} must be a finite number of seconds above 0, not {value}. Leave it out for no limit."


@CALL_PROBLEMS.problem()
def invalid_record(
    *loc: str | int, where: str, reason: str, error: str, written_by: str | None = None
) -> str:
    """What was given as a graph record isn't one, because its JSON doesn't have a graph record's shape.

    A graph record of the right shape is always compiled, however unfinished it is. What is wrong with it then goes in
    ``graph.problems``. This problem is only for JSON that isn't shaped like a record at all. In the example, the node
    says ``type`` where a graph record says ``node``.

    To fix it, fix the JSON. ``ctx.where`` says where it stops being a graph record.

    Args:
        loc: ``"record"`` and the steps to the fault inside the graph record.
        where: The same steps as one text, joined by dots. ``the record`` means the graph record as a whole.
        reason: What is wrong there.
        error: The type of pydantic's error, such as ``extra_forbidden``.
        written_by: What wrote the record, when it wasn't given as JSON: ``make_graph``'s arguments, or a node's
            ``Upgrade``.

    Example:
        refused(lambda: Graph.compile('{"nodes": {"u": {"type": "upper"}}}', registry))
    """
    return f"This isn't a valid graph record. At '{where}': {reason.rstrip('.')}. Fix {written_by or "the record's JSON"}."


@CALL_PROBLEMS.problem()
def missing_input(*loc: str | int, name: str) -> str:
    """A call leaves out a graph input that has no default.

    To fix it, give the input in the call. ``graph.inputs`` lists the graph inputs, and says which are required.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run())
    """
    return f"The graph input '{name}' is required. Give it in the call."


@CALL_PROBLEMS.problem()
def node_not_in_graph(node_id: str, /, *, nodes: list[str]) -> str:
    """A call names a node the graph doesn't have.

    To fix it, name one of the nodes in ``ctx.nodes``.

    Args:
        node_id: The id given.
        nodes: The ids of the graph's nodes.

    Example:
        done = Graph.compile(make_graph(upper), registry).run(text="a")
        refused(lambda: done.forget("lower"))
    """
    return f"'{node_id}' isn't a node of this graph. Its nodes are {comma_separated(nodes)}."


@CALL_PROBLEMS.problem()
def not_a_chain(node_id: str, /, *, before: str, inputs: list[str], outputs: list[str]) -> str:
    """``make_graph`` is given no edges, and it can't wire one of its nodes to the node before it.

    Without ``edges=``, ``make_graph`` wires the nodes in a chain. It takes the one output of the node before, and wires
    it to the node's one required input that has no constant and that an edge can feed. When a node doesn't have
    exactly one such input, or the node before it doesn't have exactly one output, it can't.

    To fix it, pass the edges, like ``make_graph(upper, greet, edges=[("upper.result", "greet.name")])``.

    Args:
        node_id: The node that can't be wired.
        before: The node before it.
        inputs: The required inputs of ``node_id`` that have no constant and that an edge can feed.
        outputs: The outputs of ``before``.

    Example:
        refused(lambda: make_graph(upper, greet, outputs={"shouted": "upper.result"}))
    """
    return (
        f"make_graph can't wire '{node_id}' after '{before}' by itself. That needs exactly one required input on '{node_id}' "
        f"(it has {comma_separated(inputs)}) and one output on '{before}' (it has {comma_separated(outputs)}). Pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem("not_a_chain", variant="constant_only")
def not_a_chain_constant_only(node_id: str, /, *, before: str, inputs: list[str]) -> str:
    """``make_graph`` is given no edges, and the node's required inputs only take a constant.

    An input whose description says ``constant_only`` takes a constant, never an edge. So ``make_graph`` has nothing to
    wire the node before into.

    To fix it, give the input a constant and pass the edges, like
    ``make_graph(upper, pick, edges=[], constants={"pick.names": "a,b"})``.

    Args:
        node_id: The node that can't be wired.
        before: The node before it.
        inputs: The required inputs of ``node_id`` that only take a constant and have none.

    Example:
        refused(lambda: make_graph(upper, pick))
    """
    return (
        f"make_graph can't wire '{node_id}' after '{before}' by itself, because only a constant can give {comma_separated(inputs)} a value. "
        f"Give {comma_separated(inputs)} a constant, and pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem()
def graph_inputs_share_a_name(*loc: str | int, name: str, inputs: list[str]) -> str:
    """``make_graph`` would make two inputs into graph inputs with the same name, and nothing says they take one value.

    ``make_graph`` turns each required input that gets nothing, no edge and no constant, into a graph input with the
    input's name. Each key of ``defaults`` becomes one too. Two inputs with the same name would then share the caller's
    one value. That may not be what you meant, so ``make_graph`` asks you to say.

    To fix it, give them names, like ``inputs={"name": "greet.name", "who": "greet-2.name"}``. Or say they share one
    value, like ``inputs={"text": ["a.text", "b.text"]}``.

    Args:
        loc: ``"inputs"`` and the name.
        name: The name.
        inputs: The inputs, each as ``node.input``.

    Example:
        refused(lambda: make_graph(upper, upper, edges=[], outputs={"a": "upper-1.result", "b": "upper-2.result"}))
    """
    return f"Two inputs, {comma_separated(inputs)}, would both become the graph input '{name}'. Name them, or say they share one value, with make_graph's inputs=."


@CALL_PROBLEMS.problem()
def graph_outputs_share_a_name(*loc: str | int, name: str, outputs: list[str]) -> str:
    """``make_graph`` would make two outputs into graph outputs with the same name.

    ``make_graph`` turns each output that no edge leaves into a graph output with the output's name. Two outputs with
    the same name are still two values, so they can't share it.

    To fix it, name them, like ``outputs={"long": "a.result", "short": "b.result"}``.

    Args:
        loc: ``"outputs"`` and the name.
        name: The name.
        outputs: The outputs, each as ``node.output``.

    Example:
        refused(lambda: make_graph(upper, words, edges=[], inputs={"a": "upper.text", "b": "words.text"}))
    """
    return f"Two outputs, {comma_separated(outputs)}, would both become the graph output '{name}'. Name them with make_graph's outputs=."


@CALL_PROBLEMS.problem()
def invalid_place(*loc: str | int, text: str) -> str:
    """``make_graph`` is given a place in the graph that isn't written ``"node.name"``.

    Each end of an edge, each key of ``constants`` or ``defaults``, and each place in ``inputs`` or ``outputs`` is a
    node's id and a name, joined by a dot. An edge's source may add one field of the record its output returns, like
    ``"read.result.email"``.

    Args:
        loc: The argument, and where in it. ``("edges", 1, 0)`` is the source of the second edge, and
            ``("constants", key)`` is a key of ``constants``.
        text: The text given.

    Example:
        refused(lambda: make_graph(upper, constants={"upper": "a"}))
    """
    return f"{text!r} isn't a place in the graph. Write it as 'node.name'."


@CALL_PROBLEMS.problem()
def unknown_graph_input(*loc: str | int, name: str, inputs: list[str]) -> str:
    """A call gives a value under a name that isn't a graph input.

    To fix it, use one of the names in ``ctx.inputs``.

    Args:
        loc: ``"inputs"`` and the name.
        name: The name the value was given under.
        inputs: The names of the graph inputs.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run(text="a", txt="a"))
    """
    return f"The graph has no input '{name}'. Its inputs are {comma_separated(inputs)}."


@CALL_PROBLEMS.problem()
def not_waiting(*loc: str | int, waiting_id: str, status: str, moved_to: str | None = None) -> str:
    """An answer is for a waiting id that no call of the run is waiting under.

    To fix it, answer one of the ids in ``run.waiting``.

    Args:
        loc: The node and the item indexes the waiting id names. If one of the indexes isn't a number, only the
            indexes before it.
        waiting_id: The waiting id the answer was given for.
        status: The run's status.
        moved_to: The waiting id the same question has now, if it waits for another call of the node. That happens
            when a change of inputs gave its item another index.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run(amount=5000)
        refused(lambda: waiting.answer("upper", True))
    """
    return f"Nothing in the run is waiting for an answer under {waiting_id!r}. The run is {status}. run.waiting lists the questions that are waiting."


@CALL_PROBLEMS.problem()
def run_not_complete(*loc: str | int, status: str, reason: str) -> str:
    """``run.outputs`` is read on a run that isn't complete.

    To fix it, answer what is waiting and call ``resume()``. Or read ``run.results``, which has every node's outputs so
    far.

    Args:
        loc: ``"outputs"``.
        status: The run's status.
        reason: The message of the problem the run stopped on. Empty if the run didn't stop early.

    Example:
        stopped = Graph.compile(make_graph(words, upper), registry).run(text="a b c", max_items=2)
        refused(lambda: stopped.outputs)
    """
    return f"The run is {status}, so it has no outputs yet. run.results has the outputs so far. {reason}".rstrip()


@CALL_PROBLEMS.problem()
def saved_run_of_another_graph(*loc: str | int, graph: str, saved: str | None = None) -> str:
    """A saved run is loaded into a different graph from the one it was saved from.

    The saved run's ``graph`` is quoted only when it has a fingerprint's form, as in every run conductor saves. A
    hand-edited save may hold any value there, and a problem never quotes a value.

    To fix it, load it into the graph it was saved from. A graph that has changed starts a new run.

    Args:
        loc: ``"saved_run"``.
        graph: The fingerprint of the graph it is loaded into.
        saved: The fingerprint of the graph the run was saved from. Left out when the saved run's ``graph`` isn't
            one.

    Example:
        saved = Graph.compile(make_graph(upper), registry).run(text="a").to_json()
        refused(lambda: Run.from_json(saved, Graph.compile(make_graph(count), registry)))
    """
    return f"This saved run is of another graph, not this one ({graph}). Load it into the graph it was saved from."


@CALL_PROBLEMS.problem()
def saved_run_unreadable(*loc: str | int, where: str, reason: str) -> str:
    """A saved run of this graph can't be read back.

    There are a few ways this happens. The text may have been changed after it was written. Other code may have written
    it, such as another version of conductor, or of a node or a type in the run. Or a value in the run may have a JSON
    form that doesn't read back. ``ctx.where`` says where the saved run stops fitting.

    To fix it, start a new run.

    Args:
        loc: ``"saved_run"`` and the steps to the fault inside the saved run.
        where: The same steps as one text, joined by dots. ``the run`` means the saved run as a whole.
        reason: What is wrong there.

    Example:
        uppering = Graph.compile(make_graph(upper), registry)
        changed = json.loads(uppering.run(text="a").to_json())
        changed["outcomes"] = {"upper": "not outcomes"}
        refused(lambda: Run.from_json(json.dumps(changed), uppering))
    """
    return (
        f"The saved run doesn't fit this graph at '{where}': {reason.rstrip('.')}. Start a new run."
    )


@CALL_PROBLEMS.problem()
def run_unwritable(*loc: str | int, where: str, reason: str) -> str:
    """A run has a value that can't be written as JSON.

    ``ctx.where`` names the output. In the example, ``raw`` returns bytes that aren't text.

    To fix it, give the output a type that pydantic can write as JSON.

    Args:
        loc: ``"saved_run"`` and the steps to the value, or ``"outputs"`` and the graph output's name.
        where: The same steps as one text.
        reason: Why the value can't be written.

    Example:
        @registry.node("raw", title="Raw", description="Turns a text into raw bytes.")
        def raw(text: str) -> bytes:
            return b"\\xff" + text.encode()


        unwritable = Graph.compile(make_graph(raw), registry).run(text="a")
        refused(lambda: unwritable.to_json())
    """
    return f"The run can't be written as JSON at '{where}': {reason.rstrip('.')}. Give that output a type pydantic can write as JSON."
