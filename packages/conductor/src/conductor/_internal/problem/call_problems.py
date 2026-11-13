"""The kinds of problem a refused call raises, in its ``InvalidCallError``: the section "Refused calls" of ``PROBLEMS``.

A module that reports one imports this module and calls the kind: ``call_problems.missing_input("inputs", name, name=name)``.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.base import comma_separated

from .type import ProblemCatalogue

#: The kinds of this section of the problems page.
CALL_PROBLEMS = ProblemCatalogue(
    "Refused calls",
    "Raised as `InvalidCallError` by a call given what it cannot act on: `graph.run`, `run.answer`, `Run.from_json`, "
    "`make_graph` and the like; `error.errors()` lists them. `loc` is what in the call is wrong: an argument "
    '(`("timeout",)`), a name inside one (`("inputs", "text")`), the steps to the fault in a graph record or a saved '
    'run (`("record", "nodes", "u", "node")`), or the node it is about, with the parameter that is wrong '
    '(`("approve",)`, `("who", "ctx")`).',
)


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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
    return f"The node takes the run's context, of class {needs}, as '{parameter}', and the call gave none; pass context= on every call that starts or resumes the run."


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem("invalid_context", variant="no_context_type")
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem("invalid_input", variant="make_a_series")
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
def invalid_record(*loc: str | int, where: str, reason: str, error: str, written_by: str | None = None) -> str:
    """What was given as a graph record is not one: its JSON does not have a graph record's shape.

    Fix the JSON: ``ctx.where`` says where it stops being a graph record. A graph record of the right shape is always
    compiled, however unfinished; what is wrong with it is in ``graph.problems``. In the example, the node says
    ``type`` where a graph record says ``node``.

    Args:
        loc: ``"record"`` and the steps to the fault inside the graph record.
        where: The same steps as one text, joined by dots; ``the record`` for the graph record as a whole.
        reason: What is wrong there.
        error: The type of pydantic's error, such as ``extra_forbidden``.
        written_by: What wrote the record when it was not its JSON: ``make_graph``'s arguments, or a node's
            ``Upgrade``.

    Example:
        refused(lambda: Graph.compile('{"nodes": {"u": {"type": "upper"}}}', registry))
    """
    return f"The record is not a graph record at '{where}': {reason.rstrip('.')}. Fix {written_by or "the record's JSON"}."


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
def not_a_chain(node_id: str, /, *, before: str, inputs: list[str], outputs: list[str]) -> str:
    """``make_graph`` is given no edges, and cannot wire one of its nodes to the node before it.

    Without ``edges=``, ``make_graph`` wires each node's one required input that has no constant and that no hook
    reads to the one output of the node before it. Pass the edges instead: ``make_graph(upper, greet,
    edges=[("upper.result", "greet.name")])``.

    Args:
        node_id: The node that cannot be wired.
        before: The node before it.
        inputs: The required inputs of ``node_id`` that have no constant and that no hook reads.
        outputs: The outputs of ``before``.

    Example:
        refused(lambda: make_graph(upper, greet, outputs={"shouted": "upper.result"}))
    """
    return (
        f"'{node_id}' can't follow '{before}' without edges: make_graph wires a node's one required input (it has {comma_separated(inputs)}) "
        f"to the one output of the node before it (that has {comma_separated(outputs)}); pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem("not_a_chain", variant="read_by_a_hook")
def not_a_chain_read_by_a_hook(node_id: str, /, *, before: str, read_by_hooks: list[str]) -> str:
    """``make_graph`` is given no edges, and every required input of a node that has no constant is read by a hook.

    A hook reads its inputs when the graph is compiled, so an input it reads takes a constant, and no edge can feed
    it. Give it a constant, and pass the edges: ``make_graph(upper, pick, edges=[], constants={"pick.names": "a,b"})``.

    Args:
        node_id: The node that cannot be wired.
        before: The node before it.
        read_by_hooks: The required inputs of ``node_id`` that a hook reads and that have no constant.

    Example:
        refused(lambda: make_graph(upper, pick))
    """
    return (
        f"'{node_id}' can't follow '{before}' without edges: a hook reads {comma_separated(read_by_hooks)} when the graph is compiled, "
        f"and an input a hook reads takes a constant, which no edge can give; give {comma_separated(read_by_hooks)} a constant and pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem()
def graph_inputs_share_a_name(*loc: str | int, name: str, inputs: list[str]) -> str:
    """``make_graph`` would make two inputs graph inputs of one name, though nothing says they take one value.

    ``make_graph`` makes each required input that gets nothing (no edge, no constant) a graph input of the input's
    name, and each key of ``defaults`` too. Two inputs of one name would then share the caller's one value, so it
    asks: give them names with ``inputs={"name": "greet.name", "who": "greet-2.name"}``, or say they share one with
    ``inputs={"text": ["a.text", "b.text"]}``.

    Args:
        loc: ``"inputs"`` and the name.
        name: The name.
        inputs: The inputs, each ``node.input``.

    Example:
        refused(lambda: make_graph(upper, upper, edges=[], outputs={"a": "upper-1.result", "b": "upper-2.result"}))
    """
    return f"Two inputs would both be the graph input '{name}' ({comma_separated(inputs)}); name them, or join them, with make_graph's inputs=."


@CALL_PROBLEMS.problem()
def graph_outputs_share_a_name(*loc: str | int, name: str, outputs: list[str]) -> str:
    """``make_graph`` would make two outputs graph outputs of one name.

    ``make_graph`` makes each output of a node no edge leaves a graph output of the output's name. Two of one name are
    two values, so name them with ``outputs={"long": "a.result", "short": "b.result"}``.

    Args:
        loc: ``"outputs"`` and the name.
        name: The name.
        outputs: The outputs, each ``node.output``.

    Example:
        refused(lambda: make_graph(upper, words, edges=[], inputs={"a": "upper.text", "b": "words.text"}))
    """
    return f"Two outputs would both be the graph output '{name}' ({comma_separated(outputs)}); name them with make_graph's outputs=."


@CALL_PROBLEMS.problem()
def not_a_place(*loc: str | int, text: str) -> str:
    """``make_graph`` is given a place in the graph that is not ``"node.name"``.

    An edge's two ends, a key of ``constants`` or ``defaults``, and a place in ``inputs`` or ``outputs`` are each a
    node's id and a name, joined by a dot; an edge's source may add a field of the record its output returns
    (``"read.result.email"``).

    Args:
        loc: The argument and where in it: ``("edges", 1, 0)`` for the source of the second edge, ``("constants",
            key)``.
        text: The text given.

    Example:
        refused(lambda: make_graph(upper, constants={"upper": "a"}))
    """
    return f"{text!r} is not a place in the graph; write 'node.name' in make_graph's arguments."


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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


@CALL_PROBLEMS.problem()
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
