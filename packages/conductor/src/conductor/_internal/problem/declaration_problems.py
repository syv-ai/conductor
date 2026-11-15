"""The faults a node's or a type's declaration can have, in ``DECLARATION_PROBLEMS``.

These are the mistakes a node's developer can make in code: an ``async def`` node, an input with no type hint, a
limit that doesn't fit its type. They are found as soon as the module that declares the node or the type is
imported. ``@node``, ``subgraph``, ``@host_type``, ``Param`` and ``NodeRegistry`` raise them as a
``NodeDeclarationError``, so only the developer meets them::

    @node("fetch", title="Fetch", description="Fetches a page.")
    async def fetch(url: str) -> str: ...
    # NodeDeclarationError: @node('fetch') on fetch: a node is a plain def; ...
    # error.problems == [async_run()], whose type is "async_run"

A kind is declared the same way as the other kinds of problem (see ``ProblemCatalogue.problem``). The difference is
that a declaration problem has no ``loc``. The error's ``where`` says where the fault is instead, like
``@node('fetch') on fetch`` and then the input's name, and the error's text puts that in front of the message.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``). Its example runs
after the setup at the top of that page. The example's last line, ``declaration_refused(lambda: ...)``, returns the
problem the declaration raises, and a test checks that it is of this kind.
"""

from __future__ import annotations

from typing import Any, Literal

from .type import ProblemCatalogue

#: Every kind of problem a declaration can have.
DECLARATION_PROBLEMS = ProblemCatalogue(
    "Declaring a node",
    "Raised as `NodeDeclarationError` when the module that declares the node or the type is imported. `error.problems` "
    "is the one problem, its `type` the code and its message written for the node's developer; the error's text puts "
    "where the fault is in front of it: `@node('adds') on adds: value: ...`.",
)


@DECLARATION_PROBLEMS.problem()
def arriving_on_compute_added_inputs(*, parameter: str) -> str:
    """A ``compute_added_inputs`` hook asks for the type arriving on an input.

    A hook parameter annotated ``type`` gets the type on the edge into that input. But the inputs a node adds are worked
    out before any edge is read, so nothing has arrived yet.

    To fix it, read the arriving type in ``compute_outputs`` instead.

    Args:
        parameter: The hook's parameter.

    Example:
        def added(value: type) -> dict[str, type]:
            return {"extra": value}


        def adds(value: str, **extra: str) -> str:
            return value


        declaration_refused(lambda: node("adds", title="Adds", description="Passes the text on as it is.", compute_added_inputs=added)(adds))
    """
    return (
        f"compute_added_inputs parameter {parameter!r} reads an arriving type, but inputs are decided before anything arrives; "
        "read it in compute_outputs"
    )


@DECLARATION_PROBLEMS.problem()
def async_run() -> str:
    """A node's function is an ``async def``.

    To fix it, write a plain ``def``. For a slow call, ``Policy(concurrency=8)`` runs several calls at once, on threads.

    Example:
        async def fetch_async(url: str) -> str:
            return url


        declaration_refused(lambda: node("fetch_async", title="Fetch async", description="Fetches a page from an address.")(fetch_async))
    """
    return "a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)"


@DECLARATION_PROBLEMS.problem()
def bound_invalid(*, unheld: list[str], type: str) -> str:
    """A ``Param`` sets a limit that pydantic has no rule for on the input's type.

    pydantic knows these limits:

    - ``ge``, ``gt``, ``le``, ``lt`` and ``multiple_of`` on numbers (``Decimal`` too), dates, times and durations,
    - ``min_length`` and ``max_length`` on texts, bytes, lists, tuples, sets and dicts,
    - ``pattern`` on texts.

    A table takes no limits. The same goes for a ``Param`` on a field of a model that a node takes.

    To fix it, use only limits pydantic knows for the type.

    Args:
        unheld: The limits pydantic has no rule for.
        type: The type, as its developer wrote it.

    Example:
        def short(text: Annotated[str, Param(ge=0)]) -> str:
            return text


        declaration_refused(lambda: node("short", title="Short", description="Passes the text on as it is.")(short))
    """
    return f"{unheld} can't apply to {type}, as pydantic holds no such limit on it; take them off, or check the value in the node"


@DECLARATION_PROBLEMS.problem("bound_invalid", variant="limit_value")
def bound_invalid_value(*, limits: str, type: str, error: str) -> str:
    """A ``Param`` sets a limit to the wrong kind of value for the type, such as ``ge="a"`` on an ``int``.

    To fix it, give each limit a value of the input's type, such as a number for ``ge`` on an ``int``.

    Args:
        limits: The limits, as written.
        type: The type's id.
        error: pydantic's reason why it can't build the type with those limits.

    Example:
        def narrow(number: Annotated[int, Param(ge="a")]) -> int:  # type: ignore[arg-type]
            return number


        declaration_refused(lambda: node("narrow", title="Narrow", description="Passes the number on as it is.")(narrow))
    """
    return f"the limits {limits} don't fit {type}, so change one: {error}"


@DECLARATION_PROBLEMS.problem()
def choices_invalid() -> str:
    """A ``Param``'s choices list the same id twice, or list none at all.

    To fix it, give each choice once, each as a value of the input's type.

    Example:
        from conductor import Choice


        def doubled(pick: Annotated[str, Param(choices=[Choice("s", "Small"), Choice("s", "Short")])]) -> str:
            return pick


        declaration_refused(lambda: node("doubled", title="Doubled", description="Hands back the chosen size.")(doubled))
    """
    return "the choices hold an id twice, or none; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem("choices_invalid", variant="on_a_literal")
def choices_invalid_on_a_literal(*, values: list[Any]) -> str:
    """A ``Param`` gives choices for a ``Literal``, and they don't give each of its values one title.

    On a ``Literal``, the choices are there to give its values titles. So there must be one choice for each value.

    To fix it, give one choice for each value, like ``[Choice("s", "Small"), Choice("m", "Medium")]`` for
    ``Literal["s", "m"]``.

    Args:
        values: The values of the ``Literal``.

    Example:
        from typing import Literal

        from conductor import Choice


        def size(pick: Annotated[Literal["s", "m"], Param(choices=[Choice("s", "Small")])]) -> str:
            return pick


        declaration_refused(lambda: node("size", title="Size", description="Hands back the chosen size.")(size))
    """
    return f"on a Literal the choices title its values, one each: {values}; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem("choices_invalid", variant="not_a_value")
def choices_invalid_not_a_value(*, reason: str) -> str:
    """A ``Param``'s choice isn't a value of the input's type.

    To fix it, give each choice as a value of the input's type, such as a number for an ``int``.

    Args:
        reason: What the type refused, without quoting the value.

    Example:
        from conductor import Choice


        def number_size(pick: Annotated[int, Param(choices=[Choice("s", "Small")])]) -> int:
            return pick


        declaration_refused(lambda: node("number_size", title="Number size", description="Hands back the chosen size.")(number_size))
    """
    return f"a choice is not a value of the field's type: {reason}; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem("choices_invalid", variant="no_json_form")
def choices_invalid_no_json_form() -> str:
    """A ``Param``'s choice can't be written as JSON.

    An editor gets the choices as JSON to show them, and a saved graph stores them as JSON. So each choice must have a
    JSON form.

    To fix it, give each choice as a value of the input's type that can be written as JSON.

    Example:
        from conductor import Choice


        def object_size(pick: Annotated[str, Param(choices=[Choice(object(), "Small")])]) -> str:
            return pick


        declaration_refused(lambda: node("object_size", title="Object size", description="Hands back the chosen size.")(object_size))
    """
    return "a choice has no JSON form; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem()
def compute_added_inputs_without_var_keyword() -> str:
    """A node has a ``compute_added_inputs`` hook, but no ``**`` parameter to take the inputs it adds.

    To fix it, take the added inputs with a ``**`` parameter, like ``**values: float``.

    Example:
        def operands(count: int) -> dict[str, type]:
            return {f"value_{i}": float for i in range(count)}


        def sum_of(count: int) -> float:
            return 0.0


        declaration_refused(lambda: node("sum_of", title="Sum of", description="Adds up numbers.", compute_added_inputs=operands)(sum_of))
    """
    return "a node with compute_added_inputs takes them as **kwargs; add **values to its signature"


@DECLARATION_PROBLEMS.problem()
def context_misplaced() -> str:
    """A ``Context[...]`` is written somewhere other than as the whole of a node's first parameter.

    The host passes the context on every run, so a node can trust it. Anywhere else, it would be an input that a caller
    fills in. So it can't be a later parameter, a keyword-only one, or be inside ``| None``, a list or a return type.

    To fix it, make the context the node's first parameter, like ``def maybe(ctx: Context[HostContext], text: str)``.

    Example:
        def later(text: str, ctx: Context[HostContext]) -> str:
            return text


        declaration_refused(lambda: node("later", title="Later", description="Passes the text on as it is.")(later))
    """
    return "Context is the whole first parameter, def sign(ctx: Context[HostContext], text: str); anywhere else a caller would give its value"


@DECLARATION_PROBLEMS.problem()
def context_not_a_class() -> str:
    """A ``Context[...]`` names something that isn't a class.

    To fix it, name a class: the registry's ``context_type``, a base class of it, or a ``Protocol`` it satisfies.

    Example:
        def listed(ctx: Context[list[str]]) -> str:
            return ""


        declaration_refused(lambda: node("listed", title="Listed", description="Hands back an empty text.")(listed))
    """
    return "Context names the class of the object the host passes; write Context[HostContext]"


@DECLARATION_PROBLEMS.problem()
def context_type_differs(*, subregistry_type: str, registry_type: str) -> str:
    """A registry added to another names a different ``context_type``.

    A registry and each registry added to it must agree on the class of the context. A registry of nodes that don't
    know about any host names no class, and that is fine too.

    To fix it, name the same class on both, or none on the registry being added.

    Args:
        subregistry_type: The class the added registry names.
        registry_type: The class this registry names.

    Example:
        @dataclass(frozen=True)
        class OtherContext:
            caller: Caller


        declaration_refused(lambda: registry.add_subregistry(NodeRegistry(context_type=OtherContext)))  # type: ignore[arg-type]
    """
    return (
        f"the subregistry names context_type={subregistry_type}, and this registry names context_type={registry_type}; "
        "a registry and its subregistries name one class, or the subregistry names none"
    )


@DECLARATION_PROBLEMS.problem("context_type_differs", variant="on_subregistry_only")
def context_type_on_subregistry_only(*, subregistry_type: str) -> str:
    """A registry that names no ``context_type`` is given a registry that names one.

    A registry that names no class is a ``NodeRegistry[None]`` to a type checker. If it took on the added registry's
    class, every ``graph.run(context=...)`` on it would be a type error.

    To fix it, name the class on this registry too.

    Args:
        subregistry_type: The class the added registry names.

    Example:
        @dataclass(frozen=True)
        class ThirdContext:
            caller: Caller


        declaration_refused(lambda: NodeRegistry().add_subregistry(NodeRegistry(context_type=ThirdContext)))  # type: ignore[arg-type]
    """
    return (
        f"the subregistry names context_type={subregistry_type}, and this registry names no context_type; "
        "a registry and its subregistries name one class, or the subregistry names none"
    )


@DECLARATION_PROBLEMS.problem()
def costs_nothing_takes_context() -> str:
    """A node whose ``Policy`` says ``costs_nothing`` takes the run's context.

    ``Policy(costs_nothing=True)`` is for a node whose call costs nothing and touches nothing outside the run.
    ``run.resume(only_costs_nothing=True)`` calls only such nodes. A node that needs the host's objects isn't one of
    them.

    To fix it, drop ``costs_nothing``. Or give the node what it needs as an input.

    Example:
        def stamp(ctx: Context[HostContext], text: str) -> str:
            return text


        declaration_refused(
            lambda: node("stamp", title="Stamp", description="Passes the text on as it is.", policy=Policy(costs_nothing=True))(stamp)
        )
    """
    return (
        "a node that takes the run's context uses the host's objects, so its call is not free; "
        "take what it needs as an input, or drop costs_nothing from its Policy"
    )


@DECLARATION_PROBLEMS.problem()
def default_invalid(*, default: str, type: str, reason: str) -> str:
    """An input's default isn't a value of its type, or can't be written as JSON.

    An editor shows the default, and a saved graph stores it, both as JSON.

    To fix it, give a default of the input's type that can be written as JSON.

    Args:
        default: The default, as Python writes it.
        type: The input's type.
        reason: Why the type refuses it, or why it can't be written as JSON.

    Example:
        def counted(number: int = "many") -> int:  # type: ignore[assignment]
            return number


        declaration_refused(lambda: node("counted", title="Counted", description="Passes the number on as it is.")(counted))
    """
    return f"the default {default} is not a value of {type} that it writes as JSON: {reason}; give one that is"


@DECLARATION_PROBLEMS.problem("default_invalid", variant="none_not_taken")
def default_none_not_taken() -> str:
    """An input's default is ``None``, and its type doesn't take ``None``.

    To fix it, write the type with ``| None``, like ``text: str | None = None``.

    Example:
        def maybe(text: str = None) -> str:  # type: ignore[assignment]
            return text


        declaration_refused(lambda: node("maybe", title="Maybe", description="Passes the text on as it is.")(maybe))
    """
    return "the default is None, and the type does not take None; write the type with | None"


@DECLARATION_PROBLEMS.problem("default_invalid", variant="on_exclusive_group")
def default_on_exclusive_group() -> str:
    """An output in an exclusive group has a default other than ``SKIPPED``.

    A call gives exactly one output of an exclusive group. Every other output in the group is a branch not chosen. An
    output that is left out, of a return or of an answer, is ``SKIPPED``, never a default.

    To fix it, give the output the default ``SKIPPED``, or no default.

    Example:
        @dataclass(frozen=True)
        class Kept:
            keep: Annotated[str, Param(exclusive_group="way")] = "kept"
            drop: Annotated[str, Param(exclusive_group="way")] = SKIPPED


        def keep(text: str) -> Kept:
            return Kept(keep=text)


        declaration_refused(lambda: node("keep", title="Keep", description="Keeps the text.")(keep))
    """
    return "an output of an exclusive group is a branch not chosen when it is left out; give it the default SKIPPED, or none"


@DECLARATION_PROBLEMS.problem("default_invalid", variant="skipped_on_input")
def default_skipped_on_input() -> str:
    """An input's default is ``SKIPPED``.

    ``SKIPPED`` is the value of an output on a branch that wasn't chosen. So only an output can have it as its default.

    To fix it, give the input a value of its type as its default, or no default.

    Example:
        def carried(text: str = SKIPPED) -> str:
            return text


        declaration_refused(lambda: node("carried", title="Carried", description="Passes the text on as it is.")(carried))
    """
    return "the default is SKIPPED, which only an output on a branch not chosen has; give the input a value of its type, or none"


@DECLARATION_PROBLEMS.problem()
def duplicate_node_id(*, id: str, first: str, second: str) -> str:
    """Two nodes added to one registry have the same id.

    To fix it, give the new node an id of its own. Or, to swap a node the registry already has for another, use
    ``registry.replace(newer)``.

    For a node with several versions, only the newest is added to the registry, with ``replaces=``. The older versions
    are plain ``@node`` definitions that are never added.

    Args:
        id: The id.
        first: Where the node the registry already has was declared.
        second: Where the node being added was declared.

    Example:
        def upper_again(text: str) -> str:
            return text


        twice = NodeRegistry()
        twice.add(upper)
        declaration_refused(lambda: twice.node("upper", title="Uppercase", description="Turns a text into capital letters.")(upper_again))
    """
    return f"two nodes have the id {id!r}: {first} and {second}; ids are unique, so swap one for the other with registry.replace(...)"


@DECLARATION_PROBLEMS.problem()
def duplicate_type_id(*, id: str, first: str, second: str) -> str:
    """Two classes given to one registry are declared with the same type id.

    To fix it, give one of the classes another id in its ``@host_type``.

    Args:
        id: The type id.
        first: The class declared first, by its module and name.
        second: The other class.

    Example:
        @host_type("money", title="Amount")
        @dataclass
        class Amount:
            kroner: int


        @host_type("money", title="Price")
        @dataclass
        class Price:
            kroner: float


        declaration_refused(lambda: NodeRegistry(types=[Amount, Price]))
    """
    return f"type id {id!r} is declared by both {first} and {second}; one registry holds one type per id"


@DECLARATION_PROBLEMS.problem()
def empty_title_or_description(
    *, declared_with: str, id: str, empty: Literal["title", "description"]
) -> str:
    """A node is declared with an empty title or description.

    The title and the description are what a person reads about the node in a palette.

    To fix it, give the node both.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The node's id.
        empty: Which of the two is empty.

    Example:
        def whisper(text: str) -> str:
            return text.lower()


        declaration_refused(lambda: node("whisper", title="Whisper", description="")(whisper))
    """
    return f"{declared_with}({id!r}) has an empty {empty}; write the words a person reads for the node in a palette"


@DECLARATION_PROBLEMS.problem()
def exclusive_group_on_input() -> str:
    """An input's ``Param`` names an ``exclusive_group``.

    An exclusive group is a set of outputs, and a call gives exactly one of them. It means nothing on an input.

    To fix it, put ``exclusive_group`` on the outputs that are branches. Those are fields of the frozen dataclass the
    node returns.

    Example:
        def branchy(text: Annotated[str, Param(exclusive_group="x")]) -> str:
            return text


        declaration_refused(lambda: node("branchy", title="Branchy", description="Passes the text on as it is.")(branchy))
    """
    return "exclusive_group is for outputs; put it on an output's Param"


@DECLARATION_PROBLEMS.problem()
def constant_only_on_output() -> str:
    """An output's ``Param`` says ``constant_only``.

    ``constant_only`` says that an input only takes a constant, never an edge or a graph input. An output gives a value
    rather than taking one, so it means nothing there.

    To fix it, put it on the input.

    Example:
        def loud(text: str) -> Annotated[str, Param(constant_only=True)]:
            return text.upper()


        declaration_refused(lambda: node("loud", title="Loud", description="Shouts the text.")(loud))
    """
    return "constant_only is for inputs; put it on an input's Param"


@DECLARATION_PROBLEMS.problem()
def hook_parameter_unknown(*, hook: str, parameter: str, inputs: list[str]) -> str:
    """A hook's parameter doesn't match any input of the node.

    Each parameter of a hook gets the constant or default of the input with the same name.

    To fix it, name each parameter after an input of the node.

    Args:
        hook: ``compute_added_inputs`` or ``compute_outputs``.
        parameter: The parameter.
        inputs: The node's inputs.

    Example:
        def from_names(names: str) -> dict[str, type]:
            return {name: str for name in names.split(",")}


        def named(labels: str) -> dict[str, str]:
            return {}


        declaration_refused(lambda: node("named", title="Named", description="Gives one answer per label.", compute_outputs=from_names)(named))
    """
    return f"{hook} parameter {parameter!r} names no input of the node; it has {inputs}"


@DECLARATION_PROBLEMS.problem("hook_parameter_unknown", variant="not_by_name")
def hook_parameter_not_by_name(*, hook: str, parameter: str) -> str:
    """A hook's parameter is ``*args`` or positional-only.

    A hook is called with each input's value passed by the input's name.

    To fix it, name each parameter after an input of the node, and let it be passed by name.

    Args:
        hook: ``compute_added_inputs`` or ``compute_outputs``.
        parameter: The parameter.

    Example:
        def from_any(*names: str) -> dict[str, type]:
            return {}


        def starred(labels: str) -> dict[str, str]:
            return {}


        declaration_refused(lambda: node("starred", title="Starred", description="Gives one answer per label.", compute_outputs=from_any)(starred))
    """
    return f"{hook} parameter {parameter!r}: a hook's parameters are passed by name; name each after an input of the node"


@DECLARATION_PROBLEMS.problem("hook_parameter_unknown", variant="var_keyword")
def hook_parameter_var_keyword(*, hook: str, parameter: str) -> str:
    """A hook takes a ``**`` parameter where it can't.

    A hook may take ``**`` in one case only: ``**name: type`` on the ``compute_outputs`` of a node that also has
    ``compute_added_inputs``. That ``**`` gets the type arriving on each input ``compute_added_inputs`` added.

    To fix it, annotate the ``**`` parameter ``type`` and give the node a ``compute_added_inputs`` hook. Or name each
    parameter after an input instead.

    Args:
        hook: ``compute_added_inputs`` or ``compute_outputs``.
        parameter: The ``**`` parameter, without its stars.

    Example:
        def from_keywords(**names: str) -> dict[str, type]:
            return {}


        def keyed(labels: str) -> dict[str, str]:
            return {}


        declaration_refused(lambda: node("keyed", title="Keyed", description="Gives one answer per label.", compute_outputs=from_keywords)(keyed))
    """
    return (
        f"{hook} parameter **{parameter}: a hook takes ** only as **{parameter}: type on compute_outputs, for the types arriving on "
        "the inputs compute_added_inputs added; annotate it type and give the node that hook, or name each parameter after an input"
    )


@DECLARATION_PROBLEMS.problem()
def input_without_type() -> str:
    """An input has no type hint.

    The hint is the input's interface. It says what an edge may bring, and what a caller must give.

    To fix it, give the input a type hint.

    Example:
        def untyped(text) -> str:  # type: ignore[no-untyped-def]
            return str(text)


        declaration_refused(lambda: node("untyped", title="Untyped", description="Passes the text on as it is.")(untyped))
    """
    return "an input needs a type hint; write one, as it is the input's interface"


@DECLARATION_PROBLEMS.problem()
def invalid_node_id(*, declared_with: str, id: str) -> str:
    """A node is declared with an id that is empty, or that contains ``.``, ``/``, ``@`` or ``~``.

    Those characters are used to write sources (``a.result``), node paths (``loud/shout``) and waiting ids
    (``a@1~3fa2``). The same rule holds for a node's id in a graph record, and for the names a hook gives inputs and
    outputs.

    To fix it, pick an id without them.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The id given.

    Example:
        def dotted_id(text: str) -> str:
            return text


        declaration_refused(lambda: node("a.b", title="Dotted id", description="Passes the text on as it is.")(dotted_id))
    """
    return (
        f"{declared_with} takes the node's id first, a non-empty string without '.', '/', '@' or '~', which mean something in a source "
        f"or a waiting id: {declared_with}('upper'); got {id!r}"
    )


@DECLARATION_PROBLEMS.problem()
def nested_series() -> str:
    """An input or output is a ``Series`` of ``Series``.

    A ``Series`` has one level of items. The next level comes from the next node: a node that is called once per item
    and returns a ``Series`` of its own.

    To fix it, return one ``Series``.

    Example:
        def nested(text: str) -> Series[Series[str]]:
            return [[text]]


        declaration_refused(lambda: node("nested", title="Nested", description="Wraps a text in lists.")(nested))
    """
    return "a Series of Series; a field holds one level of items, so return one Series"


@DECLARATION_PROBLEMS.problem()
def output_without_type() -> str:
    """A node's function has no return type.

    To fix it, write the return type. ``-> None`` means the node has no outputs.

    Example:
        def silent(text: str):  # type: ignore[no-untyped-def]
            return text


        declaration_refused(lambda: node("silent", title="Silent", description="Passes the text on as it is.")(silent))
    """
    return "a node needs a return type; write one, or -> None for no outputs"


@DECLARATION_PROBLEMS.problem()
def outputs_not_frozen() -> str:
    """A node returns a dataclass that isn't frozen.

    Several calls in a run can be given the same value, so no call may change it.

    To fix it, write ``@dataclass(frozen=True)``.

    Example:
        @dataclass
        class Parts:
            first: str
            last: str


        def parts(name: str) -> Parts:
            return Parts(*name.split(" ", 1))


        declaration_refused(lambda: node("parts", title="Parts", description="Splits a name into first and last name.")(parts))
    """
    return "a dataclass of outputs is frozen; write @dataclass(frozen=True)"


@DECLARATION_PROBLEMS.problem()
def parameter_not_by_name() -> str:
    """A node's function takes an input by position only, before a ``/``, or as ``*args``.

    Conductor passes every input by its name.

    To fix it, drop the ``/`` and the ``*args``.

    Example:
        def positional(text: str, /) -> str:
            return text


        declaration_refused(lambda: node("positional", title="Positional", description="Passes the text on as it is.")(positional))
    """
    return "a node's inputs are passed by name; drop the / and *args"


@DECLARATION_PROBLEMS.problem()
def pydantic_field_in_hint() -> str:
    """An input's or output's hint has pydantic's ``Field(...)`` in it.

    On an input or an output, the title, the description and the limits are written one way: on ``Param``, which takes
    them by the same names as ``Field``. So ``Field(...)`` is refused anywhere in the hint: on the parameter, on the
    return type, on an item (``Series[Annotated[int, Field(ge=0)]]``) and on a column of a table's row type.

    A type that carries its own limits is still fine. ``PositiveInt``, ``constr(min_length=1)`` and
    ``Annotated[int, annotated_types.Gt(0)]`` are read just as pydantic reads them. ``Field(...)`` on the fields of
    your own model or dataclass is fine too, since it belongs to that class.

    To fix it, write the same thing on ``Param``, like ``Annotated[int, Param(ge=0)]``.

    Example:
        from pydantic import Field


        def repeat(text: str, times: Annotated[int, Field(ge=0)] = 1) -> str:
            return text * times


        declaration_refused(lambda: node("repeat", title="Repeat", description="Repeats a text.")(repeat))
    """
    return (
        "pydantic's Field(...) in the hint; an input or output takes its title, description and limits from Param, "
        "so write Annotated[int, Param(ge=0)] where you had Annotated[int, Field(ge=0)]"
    )


@DECLARATION_PROBLEMS.problem("pydantic_field_in_hint", variant="discriminator")
def pydantic_discriminator_in_hint() -> str:
    """An input's or output's hint has pydantic's ``Field(discriminator=...)`` on a union.

    pydantic's ``Discriminator`` says which member of a union a value is, without ``Field``.

    To fix it, write ``Annotated[Cat | Dog, Discriminator("kind")]``.

    Example:
        from typing import Literal

        from pydantic import BaseModel, Field


        class Cat(BaseModel):
            kind: Literal["cat"]


        class Dog(BaseModel):
            kind: Literal["dog"]


        def pet(animal: Annotated[Cat | Dog, Field(discriminator="kind")]) -> str:
            return animal.kind


        declaration_refused(lambda: node("pet", title="Pet", description="Says what kind of animal it is.")(pet))
    """
    return (
        "pydantic's Field(discriminator=...) in the hint; pydantic's Discriminator says which member of a union a value is, "
        'so write Annotated[Cat | Dog, Discriminator("kind")] where you had Annotated[Cat | Dog, Field(discriminator="kind")]'
    )


@DECLARATION_PROBLEMS.problem()
def replaces_another_node(*, declared_with: str, id: str, replaces: str) -> str:
    """A node's ``replaces=`` names a node with a different id.

    A new version replaces an earlier ``@node`` with the same id. A different node gets no ``replaces=``.

    To fix it, name the earlier version of this node, or drop ``replaces=``.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The node's id.
        replaces: The node it names, as Python writes it.

    Example:
        def lower(text: str) -> str:
            return text.lower()


        declaration_refused(lambda: node("lower", title="Lower", description="Turns a text into small letters.", replaces=upper)(lower))
    """
    return f"{declared_with}({id!r}) replaces {replaces}; a node replaces an earlier node of its own id"


@DECLARATION_PROBLEMS.problem()
def type_declaration_invalid(*, id: str, faults: str) -> str:
    """``@host_type`` is given an id, a title or an ``accepts`` it can't take.

    To fix it, give the type:

    - an id that is one word, made of letters, digits, ``_``, ``.`` or ``-``, and that isn't the id of one of
      conductor's own types,
    - a title that isn't empty,
    - only classes in ``accepts``.

    Args:
        id: The id given.
        faults: What is wrong with each argument.

    Example:
        declaration_refused(lambda: host_type("str", title="My text")(type("MyText", (str,), {})))
    """
    return f"@host_type({id!r}): {faults}; give an id that is one word and not one of conductor's own, a title, and classes in accepts"


@DECLARATION_PROBLEMS.problem("type_declaration_invalid", variant="not_on_a_class")
def type_declaration_not_on_a_class(*, id: str, given: str) -> str:
    """``@host_type`` is put above something that isn't a class it can name, such as a function or a ``TypedDict``.

    A ``TypedDict`` is a record. Conductor reads a record by its fields, so it needs no declaration.

    To fix it, put ``@host_type`` above a class.

    Args:
        id: The id given.
        given: What it was put above, as Python writes it.

    Example:
        class Row(TypedDict):
            name: str


        declaration_refused(lambda: host_type("row", title="Row")(Row))
    """
    return f"@host_type({id!r}) goes above a class, and {given} is not one it can name; a TypedDict is a record and needs no declaration"


@DECLARATION_PROBLEMS.problem("type_declaration_invalid", variant="not_writable")
def type_declaration_not_writable(*, id: str, cls: str) -> str:
    """``@host_type`` is put above a class that Python doesn't let anyone change, such as ``int``.

    To fix it, put the value in a class of your own, and declare that class.

    Args:
        id: The id given.
        cls: The class.

    Example:
        declaration_refused(lambda: host_type("whole", title="Whole number")(int))
    """
    return f"@host_type({id!r}) on {cls}: Python lets nobody write on this class; put its value in a class of your own and declare that"


@DECLARATION_PROBLEMS.problem()
def row_field_optional(*, row: str, field: str) -> str:
    """The row type of a ``Table``, or of a ``Series`` of records, lets a field be left out.

    That is a ``NotRequired`` field, or any field of a ``TypedDict`` declared ``total=False``. But every row of a table
    has every column, so every field of its row type is required. A table with a column that some rows leave out isn't
    supported.

    To fix it, declare a field a row may have no value for as ``X | None``. Its column then has ``None`` for that row.

    Args:
        row: The row type's name.
        field: The first field it lets be left out.

    Example:
        from typing import NotRequired


        class Signup(TypedDict):
            email: str
            name: NotRequired[str]


        def signups(text: str) -> Series[Signup]:
            return []


        declaration_refused(lambda: node("signups", title="Signups", description="Reads the signups in a text.")(signups))
    """
    return f"the row type {row} leaves out the field '{field}', and every row of a table has every column; declare it '{field}: X | None' instead"


@DECLARATION_PROBLEMS.problem()
def type_not_carried(*, type: str) -> str:
    """An input's or output's type is one pydantic can't validate or write as JSON, such as a record with a field of a plain
    class.

    To fix it, use a type pydantic can validate and write as JSON: a pydantic model, a dataclass, or a subclass of a
    builtin such as ``class Text(str)``. A class that needs a way of its own can give its own
    ``__get_pydantic_core_schema__``.

    Args:
        type: The type's id.

    Example:
        class Handle:
            pass


        class Opened(TypedDict):
            handle: Handle


        def opened(row: Opened) -> str:
            return ""


        declaration_refused(lambda: node("opened", title="Opened", description="Reads an opened file.")(opened))
    """
    return (
        f"conductor can't check or write {type} as JSON; use a type pydantic knows, such as a pydantic model or a dataclass, "
        "or give the class its own __get_pydantic_core_schema__"
    )


@DECLARATION_PROBLEMS.problem("type_not_carried", variant="declared")
def type_not_carried_declared(*, id: str, cls: str) -> str:
    """``@host_type`` is put above a class that pydantic can't validate or write as JSON.

    To fix it, make the class a pydantic model or a dataclass, with ``@host_type`` above ``@dataclass``. Or give the
    class its own ``__get_pydantic_core_schema__``.

    Args:
        id: The id given.
        cls: The class.

    Example:
        class Plain:
            def __init__(self, path: str) -> None:
                self.path = path


        declaration_refused(lambda: host_type("plain", title="Plain")(Plain))
    """
    return (
        f"@host_type({id!r}) on {cls}: pydantic can't validate it or write it as JSON; make it a pydantic model or a "
        "dataclass (with @host_type above @dataclass), or give it its own __get_pydantic_core_schema__"
    )


@DECLARATION_PROBLEMS.problem()
def type_not_named(*, cls: str, parent: str | None = None) -> str:
    """An input's or output's type is a class of the host's that has no ``@host_type``.

    The id that ``@host_type`` gives is what an editor and a saved graph call the type. A subclass doesn't inherit its
    parent's declaration, so it needs one of its own.

    To fix it, write ``@host_type("...", title="...")`` above the class.

    Args:
        cls: The class, by its module and name.
        parent: A base class that is declared, if there is one.

    Example:
        @dataclass
        class Unnamed:
            kroner: int


        def unnamed_type(values: list[Unnamed]) -> str:
            return ""


        declaration_refused(lambda: node("unnamed-type", title="Unnamed type", description="Hands back an empty text.")(unnamed_type))
    """
    why = f"; {parent}'s declaration is its own and is not inherited" if parent else ""
    return f'{cls} is the type of a field and is not declared; write @host_type("...", title="...") above the class{why}'


@DECLARATION_PROBLEMS.problem()
def type_variable_constrained(*, variable: str) -> str:
    """A type variable is constrained to a few types, like ``T: (int, str)``.

    A bound says what may arrive on the edge, and the variable is then whatever type does arrive.

    To fix it, give the variable a bound, like ``T: int | str``. Or write the union in the signature.

    Args:
        variable: The type variable.

    Example:
        def hand_on[T: (int, str)](value: T) -> T:
            return value


        declaration_refused(lambda: node("hand-on", title="Hand on", description="Hands a value on.")(hand_on))
    """
    return f"the type variable {variable} is constrained to a few types; give it a bound, or write the union"


@DECLARATION_PROBLEMS.problem()
def type_variable_only_in_return(*, variable: str) -> str:
    """A type variable is in the return type and on no input.

    A type variable stands for the type that arrives on an edge, so it has to be on an input. An output whose type
    depends on a constant, not on an edge, gets its type from a ``compute_outputs`` hook.

    To fix it, put the type variable on an input, or use ``compute_outputs``.

    Args:
        variable: The type variable.

    Example:
        def make[T](text: str) -> list[T]:
            return []


        declaration_refused(lambda: node("make", title="Make", description="Makes an empty list.")(make))
    """
    return f"the type variable {variable} is on no input, so no edge can say what it is; put it on an input"


@DECLARATION_PROBLEMS.problem()
def upgrade_missing(*, gone: list[str]) -> str:
    """A node's new version drops an input or output of the version it replaces, and its ``Upgrade`` doesn't say where it
    went.

    To fix it, say where each one went. For an input, use ``upgrade=Upgrade(renamed_inputs={"name": "first"})`` or a
    ``rewrite=``. For an output, use ``renamed_outputs=``, since a rewrite only reads the node's constants and
    defaults.

    Args:
        gone: The inputs and outputs that are gone.

    Example:
        @node("hello", title="Hello v1", description="Says hello to a person.")
        def hello_v1(name: str) -> str:
            return name


        def hello(first: str) -> str:
            return first


        declaration_refused(lambda: node("hello", title="Hello", description="Says hello to a person.", replaces=hello_v1)(hello))
    """
    return (
        f"the version it replaces has {gone}, gone from this one; its Upgrade says where they went"
    )


@DECLARATION_PROBLEMS.problem()
def upgrade_without_replaces(*, declared_with: str, id: str) -> str:
    """A node has an ``upgrade=`` but no ``replaces=``.

    To fix it, name the version it upgrades from with ``replaces=``.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The node's id.

    Example:
        def hi(first: str) -> str:
            return first


        declaration_refused(lambda: node("hi", title="Hi", description="Says hi to a person.", upgrade=Upgrade(renamed_inputs={"name": "first"}))(hi))
    """
    return f"{declared_with}({id!r}) has an upgrade but replaces nothing; name the version it upgrades with replaces="


@DECLARATION_PROBLEMS.problem()
def var_keyword_without_hook() -> str:
    """A node takes a ``**`` parameter, and has no ``compute_added_inputs`` hook.

    The ``**`` parameter is where a node takes the inputs that hook adds.

    To fix it, remove the ``**`` parameter, or add the hook with ``compute_added_inputs=``.

    Example:
        def loose(**values: str) -> str:
            return ""


        declaration_refused(lambda: node("loose", title="Loose", description="Hands back an empty text.")(loose))
    """
    return "**kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs="


@DECLARATION_PROBLEMS.problem()
def version_not_after(*, declared_with: str, id: str, version: int, replaced: int) -> str:
    """A node's ``version=`` isn't higher than the version it replaces.

    To fix it, leave ``version=`` out, which makes it one more than the version replaced. Or give a higher number.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The node's id.
        version: The version given.
        replaced: The version it replaces.

    Example:
        @node("greeting", title="Greeting v1", description="Says hello to a person.")
        def greeting_v1(name: str) -> str:
            return name


        def greeting_again(name: str) -> str:
            return name


        declaration_refused(lambda: node("greeting", title="Greeting", description="Says hello to a person.", replaces=greeting_v1, version=1)(greeting_again))
    """
    return f"{declared_with}({id!r}) is version {version}, not after the version {replaced} it replaces; leave version= out, or give one above {replaced}"


@DECLARATION_PROBLEMS.problem()
def widget_outside_param() -> str:
    """A widget is written in ``Annotated`` next to the type, instead of on ``Param``.

    Other metadata in ``Annotated`` belongs to pydantic, and is read the way pydantic reads it. The one exception is
    ``Field(...)``, which is refused (see ``pydantic_field_in_hint``). A widget there would be ignored, and the input
    would show no control.

    To fix it, put the widget on the input's ``Param``, like ``Annotated[str, Param(widget=Textarea())]``.

    Example:
        from conductor.widgets import Textarea


        def note(text: Annotated[str, Textarea()]) -> str:
            return text


        declaration_refused(lambda: node("note", title="Note", description="Hands a note on.")(note))
    """
    return "a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]"


@DECLARATION_PROBLEMS.problem()
def widget_without_json_form(*, widget: str) -> str:
    """A field's widget has a value that can't be written as JSON, so it can't be sent to an editor.

    A widget is sent to an editor as JSON. A widget field typed ``Any`` takes any value, including one pydantic can't
    write as JSON, such as an object of a class of your own. That is caught here, when the node is declared.

    To fix it, give the widget's fields values that can be written as JSON, such as the id of what the control shows.

    Args:
        widget: The widget's class.

    Example:
        from typing import Any, Literal

        from conductor.widgets import Widget


        class Shelf:
            pass


        class ShelfPicker(Widget):
            type: Literal["shelf-picker"] = "shelf-picker"
            shelf: Any = None


        def pick_from_shelf(book: Annotated[str, Param(widget=ShelfPicker(shelf=Shelf()))]) -> str:
            return book


        declaration_refused(lambda: node("pick_from_shelf", title="Pick a book", description="Picks a book from a shelf.")(pick_from_shelf))
    """
    return f"the widget {widget} holds a value with no JSON form; give its fields values that write as JSON"
