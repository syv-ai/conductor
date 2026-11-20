"""The problems a refused call raises, in its ``InvalidCallError``.

This is the "Refused calls" section of the problems page. A call such as ``graph.run`` or ``run.with_answers`` checks
what it is given before it does anything. If something is wrong, it raises ``InvalidCallError`` with these problems,
and nothing changes.

Code that reports one imports this module and calls the kind, like
``call_problems.missing_input("inputs", name, name=name)``.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.base import SEPARATORS
from conductor._internal.utils import comma_separated

from .type import ProblemCatalogue

#: This module's section of the problems page, with its kinds.
CALL_PROBLEMS = ProblemCatalogue(
    "Refused calls",
    "A call raises these as `InvalidCallError` when it is given something it can't act on. That is `graph.run`, "
    "`run.with_answers`, `Run.from_json`, `make_graph` and the like, and `error.problems` lists them. `loc` says what "
    'in the call is wrong. It can be an argument (`("timeout",)`), a name inside one (`("inputs", "text")`), the '
    'steps to the fault in a graph record or a saved run (`("record", "nodes", "u", "node")`), or the node the '
    'problem is about, with the parameter that is wrong (`("approve",)`, `("who", "ctx")`).',
)


@CALL_PROBLEMS.problem
def answer_given_twice(*loc: str | int, waiting_id: str) -> str:
    """One ``run.with_answers`` answers the same call twice, under two waiting ids that name it.

    A call can be named by its whole waiting id or by the id without its ``~`` part, so a mapping of answers can
    name one call twice. Neither answer is taken, since it isn't clear which one is meant.

    To fix it, give the call one answer.

    Args:
        loc: The node and the item indexes of the call.
        waiting_id: The second waiting id that names the call.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        refused(lambda: waiting.with_answers({waiting.waiting[0].id: True, "approve": False}))
    """
    return f"The call {waiting_id!r} names is answered twice in one run.with_answers. Give it one answer."


@CALL_PROBLEMS.problem
def answer_stale(
    *loc: str | int, waiting_id: str, current: str, moved_to: str | None = None
) -> str:
    """An answer, or a decline with ``run.with_decline``, gives a waiting id that is out of date.

    A waiting id ends in a fingerprint of the call's inputs and of the node's code. So the call waits under a new id
    when a run made by ``run.again`` or ``run.with_inputs`` gives it other inputs, and when the node's code changed
    since the id was written, as after a deploy, even when the question reads the same.

    To fix it, answer or decline the question that is waiting now. ``ctx.current`` is its id, and ``run.waiting``
    lists it.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        current: The waiting id of the question that waits in its place.
        moved_to: The waiting id the same question has now, if it waits for another call of the node. That happens
            when a change of inputs gave its item another index.

    Example:
        asked = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        asked_again = asked.again({"amount": 5500})
        refused(lambda: asked_again.with_answers(asked.waiting[0].id, True))
    """
    return (
        f"The call {waiting_id!r} names now waits under {current!r}, since its inputs or its node's code changed "
        "after that id was written. Use that id instead."
    )


@CALL_PROBLEMS.problem
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


@CALL_PROBLEMS.problem
def cannot_move_to_version_rewrite_failed(
    *loc: str | int,
    node: str,
    version: int,
    target: int,
    reason: str,
    error: str,
    error_ctx: dict[str, Any] | None = None,
) -> str:
    """``registry.upgrade`` ran the ``rewrite`` of a version's ``Upgrade``, and it raised a ``NodeError``.

    A ``NodeError`` keeps its message, since the node's developer wrote it for people. ``ctx.error`` is its ``code``
    and ``ctx.error_ctx`` its ``ctx``. A rewrite that raises any other exception is
    ``cannot_move_to_version_raised``, and one that returns no dict of input names is
    ``cannot_move_to_version_rewrite_returned_invalid``.

    ``ctx.version`` and ``ctx.target`` are the step whose rewrite failed. On a move over several versions, that may
    be a later step than the first.

    To fix it, follow the ``NodeError``'s message, or fix that version's ``Upgrade``. In the example, the rewrite
    refuses a full name.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the step moves from.
        target: The version the step moves to.
        reason: The ``NodeError``'s message.
        error: The ``code`` of the rewrite's ``NodeError``.
        error_ctx: The ``ctx`` of the rewrite's ``NodeError``, if it has one.

    Example:
        @node("wave", title="Wave", description="Waves at a person.")
        def wave_v1(name: str) -> str:
            return f"Hello {name}"


        def first_name(values: dict) -> dict:
            if " " in values["name"]:
                raise NodeError("A full name can't be moved to a first name.", code="full_name")
            return {"first": values["name"]}


        @node("wave", title="Wave", description="Waves at a person.", replaces=wave_v1, upgrade=Upgrade(rewrite=first_name))
        def wave(first: str) -> str:
            return f"Hello {first}"


        waving = NodeRegistry()
        waving.add(wave)
        refused(lambda: waving.upgrade(make_graph(wave_v1, constants={"wave.name": "Ada Lovelace"}), "wave"))
    """
    return (
        f"'{node}' can't move from version {version} to version {target}."
        f" The rewrite of its Upgrade failed: {reason.rstrip('.')}."
    )


@CALL_PROBLEMS.problem
def cannot_move_to_version_rewrite_returned_invalid(
    *loc: str | int,
    node: str,
    version: int,
    target: int,
    reason: str,
    errors: list[dict[str, Any]],
) -> str:
    """``registry.upgrade`` ran the ``rewrite`` of a version's ``Upgrade``, and it returned no dict of input names.

    A rewrite returns the node's constants as a dict whose keys are input names. Anything
    else, such as ``None`` when it forgets its ``return``, is refused. ``ctx.errors`` has pydantic's error, with
    ``loc`` ``[]``: ``dict_type`` for no dict, ``string_type`` for a key that isn't text. A rewrite that raises is
    ``cannot_move_to_version_rewrite_failed`` or ``cannot_move_to_version_raised``.

    ``ctx.version`` and ``ctx.target`` are the step whose rewrite returned it. On a move over several versions, that
    may be a later step than the first.

    To fix it, make that version's ``Upgrade`` return a dict of input names to values. In the example, the rewrite
    returns a list of pairs.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the step moves from.
        target: The version the step moves to.
        reason: What is wrong with what the rewrite returned.
        errors: pydantic's error, with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.

    Example:
        @node("nod", title="Nod", description="Nods at a person.")
        def nod_v1(name: str) -> str:
            return f"Nod {name}"


        def as_pairs(values: dict) -> dict:
            return [("first", values["name"])]  # type: ignore[return-value]


        @node("nod", title="Nod", description="Nods at a person.", replaces=nod_v1, upgrade=Upgrade(rewrite=as_pairs))
        def nod(first: str) -> str:
            return f"Nod {first}"


        nodding = NodeRegistry()
        nodding.add(nod)
        refused(lambda: nodding.upgrade(make_graph(nod_v1, constants={"nod.name": "Ada"}), "nod"))
    """
    return (
        f"'{node}' can't move from version {version} to version {target}."
        f" The rewrite of its Upgrade returned {reason.rstrip('.')}. Return a dict of input names to values."
    )


@CALL_PROBLEMS.problem
def cannot_move_to_version_raised(
    *loc: str | int, node: str, version: int, target: int, error: str
) -> str:
    """``registry.upgrade`` ran the ``rewrite`` of a version's ``Upgrade``, and it raised an exception that isn't a ``NodeError``.

    The rewrite is the node author's code. The exception's text might quote a value, so the text and the traceback
    go to the ``conductor`` logger, and ``ctx.error`` only has the exception's class name. ``ctx.version`` and
    ``ctx.target`` are the step whose rewrite raised. On a move over several versions, that may be a later step than
    the first.

    To fix it, fix that version's ``Upgrade``, so its rewrite takes every value a graph saved with the older version
    can have.

    Args:
        loc: The node.
        node: The id of its node definition.
        version: The version the step moves from.
        target: The version the step moves to.
        error: The exception's class name.

    Example:
        @node("farewell", title="Farewell", description="Says goodbye to a person.")
        def farewell_v1(name: str) -> str:
            return f"Bye {name}"


        def split_rewrite(values: dict) -> dict:
            raise KeyError("first")


        @node("farewell", title="Farewell", description="Says goodbye to a person.", replaces=farewell_v1, upgrade=Upgrade(rewrite=split_rewrite))
        def farewell(first: str) -> str:
            return f"Bye {first}"


        upgrading = NodeRegistry()
        upgrading.add(farewell)
        refused(lambda: upgrading.upgrade(make_graph(farewell_v1, constants={"farewell.name": "Ada"}), "farewell"))
    """
    return (
        f"'{node}' can't move from version {version} to version {target}. The rewrite of its Upgrade raised {error},"
        " and the conductor log has the details."
    )


@CALL_PROBLEMS.problem
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


@CALL_PROBLEMS.problem
def invalid_answer(
    *loc: str | int,
    waiting_id: str,
    reason: str,
    errors: list[dict[str, Any]],
    question: str | None = None,
    column: str | None = None,
) -> str:
    """An answer isn't a value of the type the question takes, or doesn't fit the questions the call asked.

    Answers are checked the way their JSON would be, so a yes or no is ``true``, not ``"yes"``. Every fault is
    reported at once: one problem per question that is wrong, and one for the answer as a whole, such as no mapping
    where there are several questions, or two branches answered. ``ctx.errors`` has each fault with pydantic's code,
    such as ``bool_type``, ``missing`` for a question left out, ``extra_forbidden`` for a name that isn't a question,
    ``not_a_choice``, ``skipped_not_a_branch`` for ``SKIPPED`` on an output that isn't a branch, or
    ``several_branches_given``, and a ``loc`` inside the question's value.

    To fix it, answer with a value of the question's type.

    Args:
        loc: The node and the item indexes of the call the answer is for.
        waiting_id: The waiting id the answer was given for.
        reason: What is wrong with the answer: the first error's message.
        errors: Every fault, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the answer.
        question: For a node that asks several questions, one per output, the name of the question that is wrong.
        column: For a table, the column of the first value refused.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        refused(lambda: waiting.with_answers(waiting.waiting[0].id, "yes"))
    """
    to = f" to '{question}'" if question is not None else ""
    return f"That isn't a valid answer{to}: {reason.rstrip('.')}."


@CALL_PROBLEMS.problem
def invalid_decline_reason(*loc: str | int, waiting_id: str, errors: list[dict[str, Any]]) -> str:
    """The reason given to ``run.with_decline`` isn't text that JSON can write.

    The reason goes in the ``ctx`` of the run's ``answer_declined`` problem, for a host to show, and a saved run
    writes it as JSON. So it is a ``str`` JSON can write, or ``None`` for no reason. ``ctx.errors`` says why, without
    the reason: ``string_type`` for a value that isn't text, and ``no_json_form`` for a ``str`` with a lone surrogate
    or an object JSON can't write.

    To fix it, give the reason as a ``str``, or leave it out.

    Args:
        loc: The node and the item indexes of the call that was declined.
        waiting_id: The waiting id given to ``run.with_decline``.
        errors: What was refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the reason.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        refused(lambda: waiting.with_decline(waiting.waiting[0], reason=123))  # type: ignore[arg-type]
    """
    return "The reason for declining must be text that can be written as JSON. Give a str, or leave the reason out."


@CALL_PROBLEMS.problem
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


@CALL_PROBLEMS.problem
def invalid_context_no_context_type(*loc: str | int, got: str) -> str:
    """A call gives ``context`` to a run of a graph whose registry names no ``context_type``.

    To fix it, pass no context. Or name the class on the registry, like ``NodeRegistry(context_type=HostContext)``.

    Args:
        loc: ``"context"``.
        got: The name of the class of the object given.

    Example:
        plain = NodeRegistry()
        plain.add(upper)
        refused(lambda: Graph.compile(make_graph(upper), plain).run({"text": "a"}, context=Caller("Ida")))  # type: ignore[arg-type]
    """
    return f"The context given is of class {got}, but this graph's registry takes no context. Pass no context, or name its class with NodeRegistry(context_type=...)."


@CALL_PROBLEMS.problem
def invalid_input(
    *loc: str | int,
    name: str,
    reason: str,
    errors: list[dict[str, Any]],
    column: str | None = None,
) -> str:
    """A call gives a graph input a value its type doesn't take.

    Inputs are checked the way their JSON would be, as answers are, so a number is ``3``, not ``"3"``. A value JSON
    can't write, or with a set in it, is refused too. ``ctx.reason`` and ``ctx.errors`` say what was refused, just as
    they do for ``invalid_value``.

    To fix it, give a value of the input's type. ``graph.inputs[name].json_schema`` is the input's schema, with its
    limits.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        reason: Why the value is refused, without quoting it.
        errors: pydantic's list of what it refused, without the value.
        column: For a table, the column of the first value refused.

    Example:
        refused(lambda: Graph.compile(make_graph(count), registry).run({"n": "many"}))
    """
    return f"The value given for '{name}' isn't valid: {reason.rstrip('.')}. Give a value of the input's type."


@CALL_PROBLEMS.problem
def invalid_input_list_for_one_value(*loc: str | int, name: str, expected: str, length: int) -> str:
    """A call gives a list to a graph input that doesn't take a list.

    A graph only loops over a list when a node returns it as a ``Series``. So a list given to an input of one value is
    refused.

    To fix it, give the list to a node that returns it as a ``Series``, or give one value. A ``Series`` graph input
    takes a list, and refuses a single value.

    Args:
        loc: ``"inputs"`` and the input's name.
        name: The input's name.
        expected: The type the input takes.
        length: How many items the list has.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run({"text": ["a", "b"]}))
    """
    return (
        f"'{name}' takes one {expected}, but it was given a list of {length}. A graph only loops over a list that a node returns "
        "as a Series. Give the list to such a node, or give one value."
    )


@CALL_PROBLEMS.problem
def invalid_limit(limit: str, /, *, errors: list[dict[str, Any]]) -> str:
    """A call gives ``max_items`` a value that isn't a whole number of at least 1, such as ``0``, ``2.5`` or ``True``.

    A yes or no isn't a number, so ``True`` is refused too. ``ctx.errors`` has pydantic's error, such as
    ``greater_than_equal`` for ``0`` or ``int_type`` for ``True``, without the value.

    To fix it, give a whole number of 1 or more. Or leave it out, for no limit on items.

    Args:
        limit: ``max_items``.
        errors: What pydantic refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the
            value.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run({"text": "a"}, max_items=0))
    """
    return f"{limit} must be a whole number of 1 or more. Leave it out for no limit on items."


@CALL_PROBLEMS.problem
def invalid_limit_seconds(limit: str, /, *, errors: list[dict[str, Any]]) -> str:
    """A call gives ``timeout`` or ``checkpoint_every`` a value that isn't a finite number of seconds above 0.

    So ``0``, ``inf``, ``nan`` and ``True`` are all refused. A yes or no isn't a number. ``ctx.errors`` has pydantic's
    error, such as ``greater_than`` for ``0`` or ``finite_number`` for ``inf``, without the value.

    To fix it, give a number of seconds above 0. Or leave the limit out, for no time limit or no checkpoints.

    Args:
        limit: ``timeout`` or ``checkpoint_every``.
        errors: What pydantic refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``, none quoting the
            value.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).with_inputs({"text": "a"}).resume(timeout=0))
    """
    return f"{limit} must be a finite number of seconds above 0. Leave it out for no limit."


@CALL_PROBLEMS.problem
def invalid_record(
    *loc: str | int,
    where: str,
    reason: str,
    errors: list[dict[str, Any]],
    written_by: str | None = None,
) -> str:
    """What was given as a graph record isn't one, because its JSON doesn't have a graph record's shape.

    A graph record of the right shape is always compiled, however unfinished it is. What is wrong with it then goes in
    ``graph.problems``. This problem is only for JSON that isn't shaped like a record at all. In the example, the node
    says ``type`` where a graph record says ``node``.

    To fix it, fix the JSON. ``ctx.where`` says where it stops being a graph record. There is one problem per fault,
    and its ``ctx.errors`` has the fault's error, with ``loc`` ``[]``: pydantic's, such as ``extra_forbidden``, or
    conductor's own, such as ``json_invalid`` for text that isn't JSON or ``repeated_key`` for a key written twice.

    Args:
        loc: ``"record"`` and the steps to the fault inside the graph record.
        where: The same steps as one text, joined by dots. ``the record`` means the graph record as a whole.
        reason: What is wrong there.
        errors: The fault's error, with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.
        written_by: What wrote the record, when it wasn't given as JSON: ``make_graph``'s arguments, or a node's
            ``Upgrade``.

    Example:
        refused(lambda: Graph.compile('{"nodes": {"u": {"type": "upper"}}}', registry))
    """
    return f"This isn't a valid graph record. At '{where}': {reason.rstrip('.')}. Fix {written_by or "the record's JSON"}."


@CALL_PROBLEMS.problem
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


@CALL_PROBLEMS.problem
def node_not_in_graph(node_id: str, /, *, nodes: list[str]) -> str:
    """A call names a node the graph doesn't have.

    To fix it, name one of the nodes in ``ctx.nodes``.

    Args:
        node_id: The id given.
        nodes: The ids of the graph's nodes.

    Example:
        done = Graph.compile(make_graph(upper), registry).run({"text": "a"})
        refused(lambda: done.with_forgotten("lower"))
    """
    return f"'{node_id}' isn't a node of this graph. Its nodes are {comma_separated(nodes)}."


@CALL_PROBLEMS.problem
def hook_outputs_not_named(node_id: str, /) -> str:
    """``make_graph`` is given a node whose ``compute_outputs`` hook gives its outputs, and nothing reads them.

    ``make_graph`` makes each output that no edge reads a graph output. A hook's outputs aren't known until the
    graph is compiled, so ``make_graph`` can't make them graph outputs by itself. Without this problem, the graph
    would return nothing at all.

    To fix it, name the outputs the graph returns with ``outputs=``, like ``outputs={"a": "pick.a"}``. Or wire them
    into another node with ``edges=``.

    Args:
        node_id: The node with the hook.

    Example:
        refused(lambda: make_graph(pick, constants={"pick.names": "a,b"}))
    """
    return (
        f"'{node_id}' gets its outputs from its compute_outputs hook when the graph is compiled, so make_graph can't make them graph outputs by itself. "
        "Name the ones the graph returns with make_graph's outputs=."
    )


@CALL_PROBLEMS.problem
def not_a_chain(node_id: str, /, *, before: str, inputs: list[str], outputs: list[str]) -> str:
    """``make_graph`` is given no edges, and it can't wire one of its nodes to the node before it.

    Without ``edges=``, ``make_graph`` wires the nodes in a chain. It takes the one output of the node before, and wires
    it to the node's one required input that an edge can feed and that ``constants=``, ``defaults=`` and ``inputs=``
    don't name. A subgraph's inputs and outputs
    are the graph inputs and outputs its record declares. When a node doesn't have exactly one such input, or the node
    before it doesn't have exactly one output, it can't.

    To fix it, pass the edges, like ``make_graph(upper, greet, edges=[("upper.result", "greet.name")])``.

    Args:
        node_id: The node that can't be wired.
        before: The node before it.
        inputs: The required inputs of ``node_id`` that an edge can feed and that no other argument names.
        outputs: The outputs of ``before``.

    Example:
        refused(lambda: make_graph(upper, greet, outputs={"shouted": "upper.result"}))
    """
    return (
        f"make_graph can't wire '{node_id}' after '{before}' by itself. That needs exactly one required input on '{node_id}' "
        f"that constants=, defaults= and inputs= don't name (it has {comma_separated(inputs)}) and one output on '{before}' "
        f"(it has {comma_separated(outputs)}). Pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem
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
        refused(lambda: make_graph(upper, pick, outputs={"a": "pick.a"}))
    """
    return (
        f"make_graph can't wire '{node_id}' after '{before}' by itself, because only a constant can give {comma_separated(inputs)} a value. "
        f"Give {comma_separated(inputs)} a constant, and pass edges= to make_graph."
    )


@CALL_PROBLEMS.problem
def graph_inputs_share_a_name(*loc: str | int, name: str, inputs: list[str]) -> str:
    """``make_graph`` would make two or more inputs into graph inputs with the same name, and nothing says they take one value.

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
    return f"The inputs {comma_separated(inputs)} would each become the graph input '{name}'. Name them, or say they share one value, with make_graph's inputs=."


@CALL_PROBLEMS.problem
def graph_input_given_two_defaults(*loc: str | int, name: str, inputs: list[str]) -> str:
    """``make_graph``'s ``defaults=`` gives two inputs of one graph input different defaults.

    A graph record declares each graph input once, with one default. So when ``inputs=`` makes two inputs one graph
    input, ``defaults=`` can give it a default on either of them, but not two different ones.

    To fix it, give the default on one of the inputs, or give them both the same one.

    Args:
        loc: ``"inputs"`` and the graph input's name.
        name: The graph input's name.
        inputs: The inputs ``defaults=`` gives a default, each as ``node.input``.

    Example:
        refused(lambda: make_graph(
            ("a", upper), ("b", upper), edges=[], inputs={"text": ["a.text", "b.text"]},
            defaults={"a.text": "x", "b.text": "y"}, outputs={"a": "a.result", "b": "b.result"},
        ))
    """
    return f"The inputs {comma_separated(inputs)} both take the graph input '{name}', and defaults= gives them different defaults. Give the graph input one default."


@CALL_PROBLEMS.problem
def input_given_two_ways(*loc: str | int, input: str, ways: list[str]) -> str:
    """``make_graph`` is given one input in two ways, such as an edge in ``edges=`` and a graph input in ``inputs=``.

    An input in a graph record gets one thing: a constant, a graph input, or the edges into it. ``constants=`` gives
    it a constant, ``inputs=`` and ``defaults=`` make it a graph input, and ``edges=`` gives it edges. So an input can
    be named by only one of those three, though ``inputs=`` and ``defaults=`` may both name it.

    To fix it, keep the input in the one argument that gives it what you mean, and take it out of the others.
    ``ctx.ways`` lists the arguments that name it.

    Args:
        loc: The node and the input.
        input: The input, as ``node.input``.
        ways: The arguments that name it, in this order: ``constants``, ``defaults``, ``inputs`` and ``edges``.

    Example:
        refused(lambda: make_graph(upper, constants={"upper.text": "a"}, inputs={"text": "upper.text"}))
    """
    return (
        f"More than one of make_graph's arguments gives the input '{input}' a value: {comma_separated(ways)}. "
        "An input takes one of a constant, a graph input or edges, so name it in only one of them."
    )


@CALL_PROBLEMS.problem
def graph_outputs_share_a_name(*loc: str | int, name: str, outputs: list[str]) -> str:
    """``make_graph`` would make two or more outputs into graph outputs with the same name.

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
    return f"The outputs {comma_separated(outputs)} would each become the graph output '{name}'. Name them with make_graph's outputs=."


@CALL_PROBLEMS.problem
def invalid_argument(argument: str, /, *, reason: str, errors: list[dict[str, Any]]) -> str:
    """``make_graph`` is given an argument that isn't of the type its signature says, such as ``outputs=None``.

    The nodes are definitions, or tuples of an id and a definition such as ``("a", upper)``. ``edges`` is a sequence of
    pairs of text, such as a list of tuples. ``constants``, ``defaults`` and ``outputs`` are mappings keyed by text,
    and ``outputs`` maps to text. ``inputs`` maps text to one place, or to a sequence of them. A sequence is any but
    text or bytes. A set is never taken, since it has no order. A type checker finds these faults too.

    To fix it, give the argument the type ``make_graph``'s signature says. ``ctx.errors`` has every fault in it, each
    with pydantic's code, or ``not_a_node`` for a node, and its ``loc`` inside the argument: an index, or a key of a
    mapping. A key that isn't text is not quoted: the ``loc`` stops before it, at ``"[key]"``, and a fault in the value
    under it is left out.

    Args:
        argument: The argument: ``nodes``, ``edges``, ``constants``, ``defaults``, ``inputs`` or ``outputs``.
        reason: What is wrong with it: the message of the first error in ``errors``.
        errors: Every fault in it, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.

    Example:
        refused(lambda: make_graph(upper, inputs={"text": {"upper.text"}}))  # type: ignore[dict-item]
    """
    return f"make_graph's {argument} isn't of the type it takes: {reason}. Give it the type make_graph's signature says."


@CALL_PROBLEMS.problem
def invalid_place(*loc: str | int, text: str) -> str:
    """``make_graph`` is given a place in the graph that isn't written ``"node.name"``.

    Each end of an edge, each key of ``constants`` or ``defaults``, and each place in ``inputs`` or ``outputs`` is a
    node's id and a name, joined by a dot. An edge's source may add one field of the value its output returns, like
    ``"read.result.email"``. Neither the id nor the name may be empty or contain a character places are written
    with, so ``"upper.result~"`` isn't a place either. The message lists those characters.

    Args:
        loc: The argument, and where in it. ``("edges", 1, 0)`` is the source of the second edge, and
            ``("constants", key)`` is a key of ``constants``.
        text: The text given.

    Example:
        refused(lambda: make_graph(upper, constants={"upper": "a"}))
    """
    separators = comma_separated(f"'{separator}'" for separator in SEPARATORS)
    return (
        f"{text!r} isn't a place in the graph. Write it as 'node.name', where neither part is empty or contains "
        f"any of {separators}."
    )


@CALL_PROBLEMS.problem
def unknown_graph_input(*loc: str | int, name: str, inputs: list[str]) -> str:
    """A call gives a value under a name that isn't a graph input.

    To fix it, use one of the names in ``ctx.inputs``.

    Args:
        loc: ``"inputs"`` and the name.
        name: The name the value was given under.
        inputs: The names of the graph inputs.

    Example:
        refused(lambda: Graph.compile(make_graph(upper), registry).run({"text": "a", "txt": "a"}))
    """
    return f"The graph has no input '{name}'. Its inputs are {comma_separated(inputs)}."


@CALL_PROBLEMS.problem
def not_waiting(*loc: str | int, waiting_id: str, status: str, moved_to: str | None = None) -> str:
    """An answer, or a decline with ``run.with_decline``, is for a waiting id that no call of the run is waiting under.

    To fix it, use one of the ids in ``run.waiting``.

    Args:
        loc: The node and the item indexes the waiting id names. If one of the indexes isn't a number, or a node id
            isn't a name (it is empty, or has a ``.`` in it), only what comes before it, so ``""`` is ``()``.
        waiting_id: The waiting id the answer was given for.
        status: The run's status.
        moved_to: The waiting id the same question has now, if it waits for another call of the node. That happens
            when a change of inputs gave its item another index.

    Example:
        waiting = Graph.compile(make_graph(approve), registry).run({"amount": 5000})
        refused(lambda: waiting.with_answers("upper", True))
    """
    return f"The id {waiting_id!r} waits on nothing. The run is {status}. run.waiting lists the questions that are waiting."


@CALL_PROBLEMS.problem
def run_not_complete(*loc: str | int, status: str, reason: str) -> str:
    """``run.outputs`` is read on a run that isn't complete.

    To fix it, answer what is waiting and call ``resume()``. Or read ``run.results``, which has every node's outputs so
    far.

    Args:
        loc: ``"outputs"``.
        status: The run's status.
        reason: The message of the problem the run stopped on. Empty if the run didn't stop early.

    Example:
        stopped = Graph.compile(make_graph(words, upper), registry).run({"text": "a b c"}, max_items=2)
        refused(lambda: stopped.outputs)
    """
    return f"The run is {status}, so it has no outputs yet. run.results has the outputs so far. {reason}".rstrip()


@CALL_PROBLEMS.problem
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
        saved = Graph.compile(make_graph(upper), registry).run({"text": "a"}).to_json()
        refused(lambda: Run.from_json(saved, Graph.compile(make_graph(count), registry)))
    """
    return f"This saved run is of another graph, not this one ({graph}). Load it into the graph it was saved from."


@CALL_PROBLEMS.problem
def saved_run_unreadable(
    *loc: str | int, where: str, reason: str, errors: list[dict[str, Any]]
) -> str:
    """A saved run of this graph can't be read back.

    There are a few ways this happens. The text may have been changed after it was written. Other code may have written
    it, such as another version of conductor, or of a node or a type in the run. Or a value in the run may have a JSON
    form that doesn't read back. ``ctx.where`` says where the saved run stops fitting. ``ctx.errors`` has the fault's
    one error, with ``loc`` ``[]``: pydantic's, such as ``int_parsing`` or ``bytes_invalid_encoding``, without the
    value, or ``saved_lists_disagree`` where the saved lists of a node don't agree with each other or with the graph,
    such as an output's list longer than its calls.

    To fix it, start a new run.

    Args:
        loc: ``"saved_run"`` and the steps to the fault inside the saved run.
        where: The same steps as one text, joined by dots. ``the run`` means the saved run as a whole.
        reason: What is wrong there.
        errors: The fault's error, with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.

    Example:
        uppering = Graph.compile(make_graph(upper), registry)
        changed = json.loads(uppering.run({"text": "a"}).to_json())
        changed["outcomes"] = {"upper": "not outcomes"}
        refused(lambda: Run.from_json(json.dumps(changed), uppering))
    """
    return (
        f"The saved run doesn't fit this graph at '{where}': {reason.rstrip('.')}. Start a new run."
    )


@CALL_PROBLEMS.problem
def run_unwritable(*loc: str | int, where: str, reason: str) -> str:
    """A run has a value that can't be written as JSON.

    ``ctx.where`` names the output. In the example, ``cut_emoji`` returns a text with half of an emoji's pair, a lone
    surrogate, which JSON can't write. A value may also hold an object pydantic can't write, or ``SKIPPED`` inside
    it, such as an item of a list typed ``Any``. A node's return isn't searched for ``SKIPPED``, so such a run
    finishes in memory and is refused here, when it is saved. Until then, a node typed ``Any`` that passes on that
    ``SKIPPED``, whole or as an item of a ``Series``, skips what it feeds, and the run reads ``complete`` with its
    results shown. A host that reads the results without saving never sees the fault; the providers' worker saves at
    every checkpoint, so it fails the run there.

    To fix it, give the output a type that pydantic can write as JSON, and return ``SKIPPED`` only as a branch's
    whole value or an item of a ``Series``.

    Args:
        loc: ``"saved_run"`` and the steps to the value, or ``"outputs"`` and the graph output's name.
        where: The same steps as one text.
        reason: Why the value can't be written.

    Example:
        @registry.node("cut-emoji", title="Cut emoji", description="Ends a text with half of an emoji's pair.")
        def cut_emoji(text: str) -> str:
            return text + "\\ud83d"


        unwritable = Graph.compile(make_graph(cut_emoji), registry).run({"text": "a"})
        refused(lambda: unwritable.to_json())
    """
    return (
        f"The run can't be written as JSON at '{where}': {reason.rstrip('.')}, such as an object pydantic can't write "
        "or SKIPPED inside a value. Give that output a type pydantic can write as JSON, and keep SKIPPED out of its "
        "value."
    )
