"""The faults a node's or a type's declaration can have, in ``DECLARATION_PROBLEMS``.

These are the mistakes a node's developer can make in code: an ``async def`` node, an input with no type hint, a
limit that doesn't fit its type. They are found as soon as the module that declares the node or the type is
imported. ``@node``, ``subgraph``, ``@host_type``, ``Param`` and ``NodeRegistry`` raise them as a
``NodeDeclarationError``, so only the developer meets them::

    @node("fetch", title="Fetch", description="Fetches a page.")
    async def fetch(url: str) -> str: ...
    # NodeDeclarationError: fetch (version 1): a node is a plain def, not an async def. ...
    # error.problems[0] is async_run() at the loc ("fetch",)

A kind is declared the same way as the other kinds of problem (see ``ProblemCatalogue.problem``), with no part of
``loc`` among its parameters: the code that finds the fault knows where it is, and puts it in the ``loc`` as it
raises (``NodeDeclarationError(problem, *place)``). The ``loc`` is the node, then the parameter or output the fault
is in, if it is in one. A fault that is about no one node, such as a ``@host_type``, has no ``loc``.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``). Its example runs
after the setup at the top of that page. The example's last line, ``declaration_refused(lambda: ...)``, returns the
problem the declaration raises, and a test checks that it is of this kind.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.utils import comma_separated

from .run_problems import ADD_ASKS
from .type import ProblemCatalogue

#: Every kind of problem a declaration can have.
DECLARATION_PROBLEMS = ProblemCatalogue(
    "Declaring a node",
    "These are raised as `NodeDeclarationError` when the module that declares the node or the type is imported. "
    "`error.problems` has the one problem. Its `type` is the code, and its message is written for the node's "
    "developer. `loc` is `(node_id,)` for a fault in the node, or `(node_id, name)` for one in a parameter or an "
    "output, and the error's text puts it in front of the message, with the node's version: "
    "`adds (version 1).value: ...`. It is `()` for a fault "
    "that is about no one node: an id that isn't a valid name, an argument of the wrong type, two registries that "
    "disagree, or a type's own declaration.",
)


@DECLARATION_PROBLEMS.problem
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
        f"the compute_added_inputs parameter {parameter!r} asks for the type arriving on an input, but added inputs are worked out "
        "before any edge is read. Read the arriving type in compute_outputs instead."
    )


@DECLARATION_PROBLEMS.problem
def ask_not_declared_in_source() -> str:
    """A node's function raises ``Ask``, and its ``@node`` doesn't say ``asks=True``.

    A node that may wait for a person says so where it is declared. That way a host knows, before a run, which graphs
    can pause for a person. ``@node`` finds a ``raise Ask(...)`` in the function's own body, where its source can
    be read: a file or a notebook cell, not ``python -c``. A ``raise`` counts even if it never runs, so a function
    that catches its own ``Ask`` still says ``asks=True``. An ``Ask`` raised some other way, such as by a function
    the node calls or defines inside it, is refused when the run meets it (``ask_not_declared``). The reference
    has every case (``python -m conductor.about asking-a-person``).

    To fix it, add ``asks=True`` to the ``@node``.

    Example:
        def confirm_silently(text: str) -> bool:
            raise Ask("Sure?")


        declaration_refused(lambda: node("confirm_silently", title="Confirm", description="Asks a person to confirm.")(confirm_silently))
    """
    return f"the function raises Ask, but its @node doesn't say it may. {ADD_ASKS}"


@DECLARATION_PROBLEMS.problem
def async_run() -> str:
    """A node's function is an ``async def``.

    To fix it, write a plain ``def``. For a slow call, ``Policy(concurrency=8)`` runs several calls at once, on threads.

    Example:
        async def fetch_async(url: str) -> str:
            return url


        declaration_refused(lambda: node("fetch_async", title="Fetch async", description="Fetches a page from an address.")(fetch_async))
    """
    return "a node is a plain def, not an async def. For slow calls, Policy(concurrency=) runs several at once on threads."


@DECLARATION_PROBLEMS.problem
def param_limit_not_supported(*, unknown_limits: list[str], type: str) -> str:
    """A ``Param``, or an annotated-types limit such as ``MaxLen(3)``, sets a limit that pydantic has no rule for on
    the input's type.

    Pydantic would build the type, then fail on every value it is given, or take ``True`` as a number. An
    annotated-types limit is read as the ``Param`` limit of the same name (``Ge(0)`` as ``ge=0``, ``Len(1, 3)`` as
    ``min_length=1, max_length=3``, ``StringConstraints(min_length=1, pattern="x")`` and ``constr(...)`` as
    ``min_length=1, pattern="x"``), so the rule is the same for both. The rest of a ``StringConstraints``, such as
    ``to_lower``, is held to it too.

    A type takes a limit where pydantic has a rule for it: ``ge`` on a number, a date or a duration, ``min_length``
    on a text, a list or a table, ``pattern`` on a text. A host's own type takes it when its schema converts a value
    with a function of its own, since only the host knows whether its values compare. Any other type takes none: a
    ``bool``, a union of two types, ``Any``, a ``Literal``, an enum or a model. A validator beside the limit doesn't
    change that, and a ``PlainValidator`` replaces the check the limit is in, so a limit beside it is refused. The
    same goes for a ``Param`` on a field of a model that a node takes. A host's type takes a limit only when its own
    schema is a function, which pydantic's comparison then wraps: one whose schema is a ``json_or_python_schema``,
    a ``lax_or_strict_schema`` or a ``chain_schema`` is refused, and one that converts text to text takes a number
    limit that only the host can know is meaningful.

    To fix it, use only limits pydantic knows for the type.

    Args:
        unknown_limits: The limits pydantic has no rule for.
        type: The type, as its developer wrote it.

    Example:
        def short(text: Annotated[str, Param(ge=0)]) -> str:
            return text


        declaration_refused(lambda: node("short", title="Short", description="Passes the text on as it is.")(short))
    """
    return f"pydantic has no rule for {comma_separated(unknown_limits)} on {type}. Remove it from the hint, or check the value in the node instead."


@DECLARATION_PROBLEMS.problem
def param_limit_value_invalid(*, limits: str, type: str, reason: str) -> str:
    """A ``Param`` sets a limit to the wrong kind of value for the type, such as ``ge="a"`` on an ``int``.

    To fix it, give each limit a value of the input's type, such as a number for ``ge`` on an ``int``.

    Args:
        limits: The limits, as written.
        type: The type's id.
        reason: pydantic's reason why it can't build the type with those limits.

    Example:
        def narrow(number: Annotated[int, Param(ge="a")]) -> int:  # type: ignore[arg-type]
            return number


        declaration_refused(lambda: node("narrow", title="Narrow", description="Passes the number on as it is.")(narrow))
    """
    return f"the limits {limits} don't fit {type}: {reason}. Give each limit a value that fits the type."


@DECLARATION_PROBLEMS.problem
def choices_invalid(*, type: str, reason: str, errors: list[dict[str, Any]], instead: str) -> str:
    """A ``Param``'s choices can't be used: a choice isn't a value of the input's type, can't be written as JSON or is
    given twice, or there are none.

    An editor gets the choices as JSON to show them, and a saved graph stores them as JSON, so each choice is read from
    its JSON form, as an answer is, and a number in one must be finite. Each is read by every check on the same hint,
    wherever the ``Param`` sits: on the port, a list's item, a dict's value or a field of a host's own model. A check
    that changes a value, like an ``AfterValidator`` that adds ``"!"``, moves a choice off itself, so every value
    would fail on its call (``not_a_choice``).

    ``ctx.errors`` has the fault of each choice that doesn't read, at the choice's index: the type's own code, such as
    ``int_type``, ``finite_number`` or ``not_a_choice``, or ``no_json_form``. Only when every choice reads are repeats
    looked for, so ``choice_twice`` comes alone, as does pydantic's ``too_short`` for no choices, each with ``loc``
    ``[]``.

    To fix it, give each choice once, as a value of the type that can be written as JSON. Where a check changes a
    value (``not_a_choice``), give choices that every check on the hint keeps as they are, or make the change in the
    node. ``ctx.instead`` is the one that fits.

    Args:
        type: The type the ``Param`` is on, as pydantic hands it over: the hint under its metadata, like ``str`` for
            ``Annotated[str, AfterValidator(f), Param(choices=["a"])]``, wherever that hint sits.
        reason: The first fault.
        errors: Every fault, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.
        instead: The fix, as a sentence: for ``not_a_choice``, to give choices every check keeps; for any other
            fault, to give each choice once, as a value of the type that can be written as JSON.

    Example:
        from pydantic import AfterValidator


        def exclaimed(text: str) -> str:
            return text + "!"


        def pick_moods(moods: list[Annotated[str, AfterValidator(exclaimed), Param(choices=["calm", "glad"])]]) -> int:
            return len(moods)


        declaration_refused(lambda: node("pick-moods", title="Pick moods", description="Counts the moods picked.")(pick_moods))
    """
    return f"the choices in {type} can't be used: {reason}. {instead}"


@DECLARATION_PROBLEMS.problem
def choices_on_a_literal(*, type: str) -> str:
    """A ``Param`` gives choices for a ``Literal``.

    A ``Literal``'s values are its choices already. Choices on it would write each value a second time, and the two
    lists would have to agree.

    To fix it, write the values' plain type in place of the ``Literal`` and keep the ``Param``, so each value is
    written once: ``Annotated[str, Param(choices=[Choice("s", "Small"), Choice("m", "Medium")])]``, or
    ``Annotated[str, Param(choices=["s", "m"])]`` without titles. In a ``Series`` or a list, the ``Literal`` is the
    item, and the plain type takes its place there.

    Args:
        type: The plain type of the ``Literal``'s values, such as ``str``, written in place of the ``Literal``.

    Example:
        from typing import Literal

        from conductor import Choice


        def size(pick: Annotated[Literal["s", "m"], Param(choices=[Choice("s", "Small"), Choice("m", "Medium")])]) -> str:
            return pick


        declaration_refused(lambda: node("size", title="Size", description="Hands back the chosen size.")(size))
    """
    return (
        f"a Literal's values are its choices already, so it takes no Param(choices=). Write {type} in place of the"
        " Literal and keep the Param(choices=[...]), each choice a value or a Choice(value, title)."
    )


@DECLARATION_PROBLEMS.problem
def choices_on_a_literal_beside_another_type() -> str:
    """A ``Param`` gives choices for a union of a ``Literal`` with another type, such as ``Literal["auto"] | int``.

    Such a port has no choices: it takes the ``Literal``'s values and any value of the other type. Choices on it
    would limit it to the choices, so the other type would take nothing more, and a ``Literal``'s values can't be
    given titles any other way.

    To fix it, remove the ``Param(choices=)``. The port then takes what its type says, untitled.

    Example:
        from typing import Literal

        from conductor import Choice


        def auto_or_count(count: Annotated[Literal["auto"] | int, Param(choices=[Choice("auto", "Automatic")])]) -> str:
            return str(count)


        declaration_refused(lambda: node("auto_or_count", title="Auto or count", description="Hands back the count.")(auto_or_count))
    """
    return (
        "a Literal beside another type takes no Param(choices=), since choices would limit the port to them. Remove"
        " the Param(choices=)."
    )


@DECLARATION_PROBLEMS.problem
def named_value_invalid(*, value: str, instead: str) -> str:
    """A ``Literal``'s value doesn't read back from its own JSON as itself.

    A type's named values, a ``Literal``'s values and an enum's members, are its choices, and a saved run, a person's
    answer and a description hold each as its JSON. So each must read back from that JSON as itself, or a run of it
    couldn't be saved, described or asked. Text, numbers, yes or no and ``None`` do, and so does the member of an
    enum that is also ``str`` or ``int``, like an ``IntEnum``'s or a ``StrEnum``'s. A plain enum's member, ``bytes``,
    a tuple, a date and ``nan`` don't. This holds at any depth: on the port, a list's or a ``Series``'s item, a
    ``TypedDict``'s field and a table's column.

    To fix it, take the value's type, or the enum itself, as the type, with ``Param(choices=[...])`` for some of its
    values.

    Args:
        value: The value, as Python writes it.
        instead: What to write instead.

    Example:
        import enum
        from typing import Literal


        @host_type("cup", title="Cup")
        class Cup(enum.Enum):
            SMALL = "s"
            LARGE = "l"


        def cup_size(pick: Literal[Cup.SMALL]) -> str:
            return str(pick)


        declaration_refused(lambda: node("cup_size", title="Cup size", description="Hands back the chosen size.")(cup_size))
    """
    return (
        f"the Literal's value {value} doesn't read back from its own JSON as itself, so a run of it couldn't be saved,"
        f" described or asked. {instead}"
    )


@DECLARATION_PROBLEMS.problem
def named_value_invalid_enum(*, enum: str, member: str) -> str:
    """An enum's member doesn't read back from its own JSON as itself, so the enum can't be a type.

    An enum's members are its choices, and the enum reads a member back from its value's JSON. An enum whose values
    are tuples, ``bytes`` or dates reads no member back, so a run of it couldn't be saved, described or asked. This
    holds at any depth.

    To fix it, give the enum's members values that are text, numbers or yes or no. Or take the values' type, such
    as ``str``, with ``Param(choices=[Choice(value, title), ...])``.

    Args:
        enum: The enum's name.
        member: The member, as Python writes it.

    Example:
        import enum


        @host_type("span", title="Span")
        class Span(enum.Enum):
            LOW = (0, 1)


        def pick_span(span: Span) -> str:
            return str(span)


        declaration_refused(lambda: node("pick_span", title="Span", description="Hands back the chosen span.")(pick_span))
    """
    return (
        f"the enum {enum}'s member {member} doesn't read back from its own JSON as itself, so a run of it couldn't be"
        " saved, described or asked. Give its members values that are text, numbers or yes or no, or take the"
        " values' type, such as str, with Param(choices=[...])."
    )


@DECLARATION_PROBLEMS.problem
def named_value_invalid_alike(*, values: list[Any]) -> str:
    """Two values of a ``Literal``, or two members of an enum, are written as the same JSON, like ``inf`` and the text
    ``"Infinity"``.

    A saved run, a person's answer and a graph record hold a value as its JSON. Two values with one JSON form can't be
    told apart there, so one would come back as the other.

    To fix it, give values whose JSON differs.

    Args:
        values: The values of the ``Literal``, or the enum's members.

    Example:
        from typing import Literal


        def cap(limit: Literal[float("inf"), "Infinity"]) -> str:  # type: ignore[valid-type]
            return str(limit)


        declaration_refused(lambda: node("cap", title="Cap", description="Hands back the cap.")(cap))
    """
    return (
        f"two of the values {comma_separated(values)} are written as the same JSON, so a saved run or an answer"
        " can't tell them apart. Give values whose JSON differs."
    )


@DECLARATION_PROBLEMS.problem
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
    return "the node has compute_added_inputs, but no ** parameter to take the inputs it adds. Add one, like **values: float."


@DECLARATION_PROBLEMS.problem
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
    return "Context[...] must be the whole of the node's first parameter, like def sign(ctx: Context[HostContext], text: str). Anywhere else, a caller would have to give its value."


@DECLARATION_PROBLEMS.problem
def context_not_a_class() -> str:
    """A ``Context[...]`` names something that isn't a class.

    To fix it, name a class: the registry's ``context_type``, a base class of it, or a ``Protocol`` it satisfies.

    Example:
        def listed(ctx: Context[list[str]]) -> str:
            return ""


        declaration_refused(lambda: node("listed", title="Listed", description="Hands back an empty text.")(listed))
    """
    return "Context[...] names the class of the object the host passes. Write a class, like Context[HostContext]."


@DECLARATION_PROBLEMS.problem
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
        f"the registry being added names context_type={subregistry_type}, but this registry names context_type={registry_type}. "
        "Name the same class on both, or none on the one being added."
    )


@DECLARATION_PROBLEMS.problem
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
        f"the registry being added names context_type={subregistry_type}, but this registry names none. "
        "Name the same class on this registry too."
    )


@DECLARATION_PROBLEMS.problem
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
        "the node takes the run's context, so it uses the host's objects and its call isn't free. "
        "Drop costs_nothing from its Policy, or give the node what it needs as an input."
    )


@DECLARATION_PROBLEMS.problem
def default_invalid(
    *,
    default: str,
    type: str,
    reason: str,
    errors: list[dict[str, Any]],
    column: str | None = None,
) -> str:
    """An input's default isn't a value of its type, or can't be written as JSON.

    An editor shows the default, and a saved graph stores it, both as JSON. ``ctx.reason`` and ``ctx.errors`` say what
    the type refused, as they do for ``invalid_value``.

    To fix it, give a default of the input's type that can be written as JSON.

    Args:
        default: The default, as Python writes it.
        type: The input's type.
        reason: Why the type refuses it, or why it can't be written as JSON.
        errors: pydantic's list of what it refused, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.
        column: For a table, the column of the first value refused.

    Example:
        def counted(number: int = "many") -> int:  # type: ignore[assignment]
            return number


        declaration_refused(lambda: node("counted", title="Counted", description="Passes the number on as it is.")(counted))
    """
    return f"the default {default} isn't a value of {type} that can be written as JSON: {reason}. Give a default that is."


@DECLARATION_PROBLEMS.problem
def default_none_not_taken() -> str:
    """An input's default is ``None``, and its type doesn't take ``None``.

    To fix it, write the type with ``| None``, like ``text: str | None = None``.

    Example:
        def maybe(text: str = None) -> str:  # type: ignore[assignment]
            return text


        declaration_refused(lambda: node("maybe", title="Maybe", description="Passes the text on as it is.")(maybe))
    """
    return "the default is None, but the type doesn't take None. Write the type with | None, like str | None."


@DECLARATION_PROBLEMS.problem
def default_skipped_on_input() -> str:
    """An input's default is ``SKIPPED``.

    ``SKIPPED`` is the value of an output that was skipped. So only an output can have it as its default.

    To fix it, give the input a value of its type as its default, or no default.

    Example:
        def carried(text: str = SKIPPED) -> str:
            return text


        declaration_refused(lambda: node("carried", title="Carried", description="Passes the text on as it is.")(carried))
    """
    return "an input can't default to SKIPPED, since SKIPPED is only ever the value of a skipped output. Give the input a value of its type as its default, or none."


@DECLARATION_PROBLEMS.problem
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
    return f"two nodes have the id {id!r}: {first} and {second}. Give one of them another id, or swap the old one for the new one with registry.replace(...)."


@DECLARATION_PROBLEMS.problem
def duplicate_type_id(*, id: str, first: str, second: str) -> str:
    """Two classes given to one registry are declared with the same type id.

    To fix it, give one of the classes another id in its ``@host_type``.

    Args:
        id: The type id.
        first: The class declared first, by its module and name.
        second: The other class.

    Example:
        @host_type("money", title="Amount")
        @dataclass(frozen=True)
        class Amount:
            kroner: int


        @host_type("money", title="Price")
        @dataclass(frozen=True)
        class Price:
            kroner: float


        declaration_refused(lambda: NodeRegistry(types=[Amount, Price]))
    """
    return (
        f"the type id {id!r} is declared by both {first} and {second}. Give one of them another id."
    )


@DECLARATION_PROBLEMS.problem
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
    return "constant_only is for inputs, not outputs. Put it on an input's Param."


@DECLARATION_PROBLEMS.problem
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
    return f"the {hook} parameter {parameter!r} isn't an input of the node. Its inputs are {comma_separated(inputs)}."


@DECLARATION_PROBLEMS.problem
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
    return f"the {hook} parameter {parameter!r} can't be passed by name, and a hook gets each input by name. Make it a plain parameter named after an input."


@DECLARATION_PROBLEMS.problem
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
        f"the {hook} parameter **{parameter} isn't allowed here. A hook only takes **{parameter}: type on compute_outputs, for the types "
        "arriving on the inputs compute_added_inputs adds. Annotate it type and add that hook, or name each parameter after an input."
    )


@DECLARATION_PROBLEMS.problem
def input_without_type() -> str:
    """An input has no type hint.

    The hint is the input's interface. It says what an edge may bring, and what a caller must give.

    To fix it, give the input a type hint.

    Example:
        def untyped(text) -> str:  # type: ignore[no-untyped-def]
            return str(text)


        declaration_refused(lambda: node("untyped", title="Untyped", description="Passes the text on as it is.")(untyped))
    """
    return "the input has no type hint. Give it one, since the hint says what the input takes."


@DECLARATION_PROBLEMS.problem
def invalid_declaration_argument(
    *, declared_with: str, reason: str, errors: list[dict[str, Any]]
) -> str:
    """``@node``, ``subgraph``, ``NodeRegistry``, ``@host_type`` or ``Param`` is given an argument it can't take, such
    as ``tags="text"``, ``version=True``, an id with a ``.``, a blank title or a dataclass that isn't frozen.


    Each argument is held to its annotation, strictly: text is a ``str``, ``version`` a whole number of 1 or more and
    never ``True``, ``tags`` a list or tuple of text and never one text, ``asks`` a ``bool``, and ``policy``,
    ``deprecation``, ``replaces`` and ``upgrade`` an instance of their class. ``extra`` is JSON a host is shown, so it
    is a mapping keyed by text whose values are JSON values: a tuple, a date or another object is refused
    (``invalid-json-value``), and so is ``inf`` or ``nan`` (``finite_number``), as any number that enters is. A type
    checker finds most of these faults too.

    Some text has a rule of its own. A node's id is a name, as a node's id in a graph record is: not empty, and none of
    ``.``, ``/``, ``@`` or ``~``, since sources, node paths and waiting ids are written with them (``not_a_name``). A
    type's id, given to ``@host_type``, is one word of letters, digits, ``_``, ``.`` or ``-`` that starts with a letter
    or ``_``, since it is written inside other ids such as ``list[money]`` (``type_id_not_one_word``). It isn't the id
    of one of conductor's own types, such as ``str`` (``conductors_own_type_id``). A title or description is what a
    person reads in a palette, so it isn't empty or only whitespace (``blank_text``), and so is a ``Param``'s. ``@host_type`` put above a class
    conductor names itself, such as ``HttpUrl``, is refused too (``conductors_own_type``, at ``loc`` ``["cls"]``, the
    class): declare a subclass of it to give it an id of your own. So is ``@host_type`` put above a dataclass that
    isn't ``frozen=True``, or a pydantic model whose config isn't (``type_not_frozen``, at ``loc`` ``["cls"]``); why
    is in ``python -m conductor.about types``.

    ``ctx.errors`` has every fault of the arguments, and ``ctx.reason``, which the message quotes, names each after
    its argument, so one run finds them all. Each fault has pydantic's code, one of the codes above, or ``not_a_node``
    for a ``replaces=`` that isn't a node definition, and its ``loc``: the argument, then where in it, such as
    ``["tags", 1]``. A key of ``extra`` that isn't text is not quoted, at any depth: the ``loc`` stops at
    ``"[key]"``, and a fault in the value under it is left out. ``JsonValue``'s own labels, ``dict`` and ``list``, are
    left out of a ``loc``, so it names only keys and indexes. The problem's own ``loc`` is the node's id, and ``()``
    when the id itself is refused, for a registry, and for ``@host_type``. Its text names the node's version after
    the id, unless the version itself is refused.

    To fix it, give each argument the type the signature says, or as its fault says.

    Args:
        declared_with: ``@node``, ``subgraph``, ``NodeRegistry``, ``@host_type`` or ``Param``.
        reason: Every fault, each after the argument it is in, with ``; `` between them.
        errors: Every fault, each with its ``type``, ``loc``, ``msg`` and maybe ``ctx``.

    Example:
        @dataclass
        class Parcel:
            weight: float


        declaration_refused(lambda: host_type("parcel", title="Parcel")(Parcel))
    """
    return (
        f"{declared_with} can't take its arguments: {reason}. Fix each argument as its fault says."
    )


@DECLARATION_PROBLEMS.problem
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
    return "a Series of Series isn't allowed, since a Series has one level of items. Use one Series, and let the next node make the next level."


@DECLARATION_PROBLEMS.problem
def output_without_type() -> str:
    """A node's function has no return type.

    To fix it, write the return type. ``-> None`` means the node has no outputs.

    Example:
        def silent(text: str):  # type: ignore[no-untyped-def]
            return text


        declaration_refused(lambda: node("silent", title="Silent", description="Passes the text on as it is.")(silent))
    """
    return "the node has no return type. Write one, or -> None for no outputs."


@DECLARATION_PROBLEMS.problem
def outputs_not_a_mapping(*, returns: str) -> str:
    """A node whose ``compute_outputs`` hook gives its outputs declares a return type that can't be a mapping.

    The hook says what the outputs are, so the function returns them as a mapping from each output's name to its
    value. A return type like ``str`` says it returns one value, and every call would fail. The return type is held
    to the rule an edge is checked by: a ``dict[str, Any]`` must fit it, as it fits ``Mapping[str, Any]``,
    ``object`` or ``Any``. A ``TypedDict`` is refused, since a mapping of any keys doesn't fit it.

    To fix it, write ``-> dict[str, Any]``.

    Args:
        returns: The return type, written as conductor writes a type, like ``int | None``.

    Example:
        def text_outputs() -> dict[str, type]:
            return {"result": str}


        def one(text: str) -> str:
            return text


        declaration_refused(lambda: node("one", title="One", description="Passes the text on as it is.", compute_outputs=text_outputs)(one))
    """
    return (
        f"the node's compute_outputs hook gives its outputs, so it returns them as a mapping by name, not as {returns}. "
        "Write -> dict[str, Any]."
    )


@DECLARATION_PROBLEMS.problem
def parameter_not_by_name() -> str:
    """A node's function takes an input by position only, before a ``/``, or as ``*args``.

    Conductor passes every input by its name.

    To fix it, drop the ``/`` and the ``*args``.

    Example:
        def positional(text: str, /) -> str:
            return text


        declaration_refused(lambda: node("positional", title="Positional", description="Passes the text on as it is.")(positional))
    """
    return "the node takes an input by position only. Conductor passes every input by name, so drop the / and *args."


@DECLARATION_PROBLEMS.problem
def pydantic_field_in_hint() -> str:
    """An input's or output's hint has pydantic's ``Field(...)`` in it.

    On an input or an output, the title, the description and the limits are written one way: on ``Param``, which takes
    them by the same names as ``Field``. So ``Field(...)`` is refused anywhere in the hint: on the parameter, on the
    return type, on an item (``Series[Annotated[int, Field(ge=0)]]``), on a field of a ``TypedDict`` and on a column of a
    table's row type.

    A type that carries its own limits is still fine. ``PositiveInt``, ``constr(min_length=1)`` and
    ``Annotated[int, annotated_types.Gt(0)]`` are read just as pydantic reads them, the last as ``Param(gt=0)``. ``Field(...)`` on the fields of
    your own model or dataclass is fine too, since it belongs to that class.

    To fix it, write the same thing on ``Param``, like ``Annotated[int, Param(ge=0)]``.

    Example:
        from pydantic import Field


        def repeat(text: str, times: Annotated[int, Field(ge=0)] = 1) -> str:
            return text * times


        declaration_refused(lambda: node("repeat", title="Repeat", description="Repeats a text.")(repeat))
    """
    return (
        "the hint uses pydantic's Field(...). An input or output takes its title, description and limits from Param, "
        "so write Annotated[int, Param(ge=0)] where you had Annotated[int, Field(ge=0)]."
    )


@DECLARATION_PROBLEMS.problem
def timezone_in_hint() -> str:
    """An input's or output's hint has annotated-types' ``Timezone`` in it, such as
    ``Annotated[datetime, Timezone(...)]``.

    Pydantic builds the type and never checks it, so a datetime without a timezone would pass.

    To fix it, take pydantic's ``AwareDatetime`` for a datetime with a timezone, and ``NaiveDatetime`` for one
    without.

    Example:
        from datetime import datetime

        from annotated_types import Timezone


        def meeting(starts: Annotated[datetime, Timezone(...)]) -> datetime:
            return starts


        declaration_refused(lambda: node("meeting", title="Meeting", description="Passes a time on.")(meeting))
    """
    return "pydantic never checks a Timezone. Take pydantic's AwareDatetime, or NaiveDatetime for no timezone."


@DECLARATION_PROBLEMS.problem
def unit_in_hint() -> str:
    """An input's or output's hint has annotated-types' ``Unit`` in it, such as ``Annotated[float, Unit("m")]``.

    A unit is a label, not a rule: pydantic ignores it, so it checks nothing, and a person is never shown it.

    To fix it, put the unit in the port's title or description, like ``Param(title="Distance (m)")``.

    Example:
        from annotated_types import Unit


        def distance(metres: Annotated[float, Unit("m")]) -> float:
            return metres


        declaration_refused(lambda: node("distance", title="Distance", description="Passes a distance on.")(distance))
    """
    return "pydantic ignores a Unit, so it checks nothing and no one sees it. Put the unit in the port's title instead."


@DECLARATION_PROBLEMS.problem
def predicate_alias_in_hint(*, predicate: str) -> str:
    """An input's or output's hint has one of annotated-types' named predicates written as metadata, such as
    ``Annotated[str, LowerCase]``.

    ``LowerCase``, ``IsDigit`` and the others are aliases of a whole type, written with the type in brackets. Written
    as metadata, pydantic ignores them, so every value would pass.

    To fix it, write the alias as the type, like ``LowerCase[str]``.

    Args:
        predicate: The check inside the alias, such as ``Predicate(str.islower)``.

    Example:
        from annotated_types import LowerCase


        def initials(text: Annotated[str, LowerCase]) -> str:
            return text


        declaration_refused(lambda: node("initials", title="Initials", description="Passes the text on.")(initials))
    """
    return f"{predicate} is written as metadata, where pydantic ignores it. Write the alias with its type, like LowerCase[str]."


@DECLARATION_PROBLEMS.problem
def predicate_refuses_every_number(*, predicate: str) -> str:
    """An input's hint asks for a number that isn't finite: ``IsNan``, ``IsInfinite`` or ``IsNotFinite``.

    A number given to a graph from outside is finite, so an input can never be given a value that passes. An output
    may declare one, since a run's own values may be ``inf`` or ``nan``.

    To fix it, remove the predicate from the input, and take ``float | None`` where ``None`` stands for no number.

    Args:
        predicate: The check, such as ``Predicate(math.isnan)``.

    Example:
        from annotated_types import IsNan


        def odd_ratio(ratio: IsNan[float]) -> float:
            return ratio


        declaration_refused(lambda: node("odd_ratio", title="Odd ratio", description="Passes a ratio on.")(odd_ratio))
    """
    return f"{predicate} asks for a number that isn't finite, and an input can never be given one. Remove it."


@DECLARATION_PROBLEMS.problem
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
        'the hint uses pydantic\'s Field(discriminator=...). Write Annotated[Cat | Dog, Discriminator("kind")] where you had '
        'Annotated[Cat | Dog, Field(discriminator="kind")].'
    )


@DECLARATION_PROBLEMS.problem
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
    return f"{declared_with}({id!r}) replaces {replaces}, which is another node. A node only replaces an earlier version of itself, with the same id."


@DECLARATION_PROBLEMS.problem
def type_declaration_not_on_a_class(*, id: str, given: str) -> str:
    """``@host_type`` is put above something that isn't a class it can name, such as a function or a ``TypedDict``.

    Conductor reads a ``TypedDict`` by its fields, so it needs no declaration.

    To fix it, put ``@host_type`` above a class.

    Args:
        id: The id given.
        given: What it was put above, as Python writes it.

    Example:
        class Address(TypedDict):
            street: str


        declaration_refused(lambda: host_type("address", title="Address")(Address))
    """
    return f"@host_type({id!r}) goes above a class, and {given} isn't one it can name. A TypedDict is read by its fields, and needs no declaration."


@DECLARATION_PROBLEMS.problem
def type_declaration_not_writable(*, id: str, cls: str) -> str:
    """``@host_type`` is put above a class that Python doesn't let anyone change, such as ``int``.

    To fix it, put the value in a class of your own, and declare that class.

    Args:
        id: The id given.
        cls: The class.

    Example:
        from collections import OrderedDict


        declaration_refused(lambda: host_type("ordered", title="Ordered mapping")(OrderedDict))
    """
    return f"@host_type({id!r}) can't declare {cls}, since Python doesn't let anyone change that class. Put the value in a class of your own, and declare that."


@DECLARATION_PROBLEMS.problem
def table_row_not_a_typed_dict(*, row: str) -> str:
    """A ``Table``'s row type is neither a ``TypedDict`` nor a type any row fills, such as a dataclass or ``int``.

    A ``TypedDict`` names a table's columns, and a table read for it keeps only those. Any other row type names none,
    so the table would keep every column it was given and check none of them. A bare ``Table`` takes any columns,
    and is written ``Table``, ``Table[Any]`` or ``Table[dict[str, Any]]``. A type variable, as in ``Table[T]``,
    is filled in by a graph.

    To fix it, write the row as a ``TypedDict`` with one field per column, or take a bare ``Table``.

    Args:
        row: The row type, as its developer wrote it.

    Example:
        @host_type("spot", title="Spot")
        @dataclass(frozen=True)
        class Spot:
            x: float
            y: float


        def spots(text: str) -> Table[Spot]:
            return pl.DataFrame({"x": [0.0], "y": [0.0]})


        declaration_refused(lambda: node("spots", title="Spots", description="Reads the spots in a text.")(spots))
    """
    return f"a table's row type names its columns as a TypedDict, and {row} isn't one. Write the row as a TypedDict, or take a bare Table."


@DECLARATION_PROBLEMS.problem
def table_cell_not_kept(*, row: str, column: str, cell: str, reason: str) -> str:
    """A column of a ``Table``'s row type has a type a table doesn't keep as it is.

    A table keeps its cells in polars, which has a column type for Python's own classes like ``int``, ``str`` and
    ``date``, and for lists and structs of them. So a column may be, at any depth: one of those classes; a
    ``Literal`` or an enum whose values are all of one of them; ``list[C]`` or a tuple whose items are all ``C``; a
    model, a dataclass or a ``TypedDict`` of such fields that declares every key a value has; ``C | None``; or
    ``Any``, which a hook gives for a type it doesn't know yet. Any other type would come back from polars changed: a
    union of two types, a dict, a tuple of different items, a ``NotRequired`` field or a config with
    ``extra="allow"``. A ``Series`` of the same rows is a list, and keeps each value as it is.

    A column of ``Any`` is let through here, since its type says nothing of its values. Polars picks one type for
    them when the table is made from rows, and if it would change one, like ``1`` beside ``1.5``, which would become
    ``1.0``, or a model, which would become a dict, the run fails there, with this code in its ``ctx.errors``.

    To fix it, give the column a type a table keeps, such as a model whose fields are all required.

    Args:
        row: The row type's name.
        column: The column whose type a table doesn't keep.
        cell: The type, in the column's type, that a table doesn't keep, such as ``dict[str, int]``.
        reason: Why a table doesn't keep it.

    Example:
        class Scores(TypedDict):
            name: str
            points: dict[str, int]


        def scores(text: str) -> Table[Scores]:
            return pl.DataFrame({"name": ["Ann"], "points": [{"math": 3}]})


        declaration_refused(lambda: node("scores", title="Scores", description="Reads the scores in a text.")(scores))
    """
    return f"in column '{column}' of the row type {row}, a table doesn't keep {cell} as it is: {reason}. Give the column a type a table keeps, such as a model whose fields are all required."


@DECLARATION_PROBLEMS.problem
def row_field_optional(*, row: str, field: str) -> str:
    """The row type of a ``Table``, or of a ``Series`` of rows, lets a field be left out.

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
    return f"the row type {row} lets '{field}' be left out, but every row of a table has every column. Declare it '{field}: X | None' instead."


@DECLARATION_PROBLEMS.problem
def type_not_supported(*, type: str) -> str:
    """An input's or output's type is one pydantic can't validate or write as JSON, such as a hint that is no type at all,
    like ``-> 5`` or ``-> [int]``, or a ``TypedDict`` field written as text that names nothing.

    To fix it, use a type pydantic can validate and write as JSON: a pydantic model, a dataclass, or a subclass of a
    builtin such as ``class Text(str)``. A class that needs a way of its own can give its own
    ``__get_pydantic_core_schema__``.

    Args:
        type: The type's id.

    Example:
        def handles() -> [int]:  # type: ignore[valid-type]
            return []


        declaration_refused(lambda: node("handles", title="Handles", description="Lists the open handles.")(handles))
    """
    return (
        f"conductor can't check {type} or write it as JSON. Use a type pydantic knows, such as a pydantic model or a dataclass, "
        "or give the class its own __get_pydantic_core_schema__."
    )


@DECLARATION_PROBLEMS.problem
def type_not_supported_set(*, type: str) -> str:
    """An input's or output's type is a set, or holds one anywhere inside it, like ``set[str]``, ``frozenset[int]``,
    ``list[set[str]]`` or a field of a host's own model or dataclass typed ``set[str]``.

    A set has no order, so its items could come in a different order each time, and the same set would be written
    as a different JSON text. Every value that enters a graph is read from its JSON form, a node's own default
    included, and a set has none that keeps it the same. A set a node returns would hash differently in each
    process, so a later run could never reuse the call. So conductor doesn't take a set anywhere in a declared type,
    even where a host's serializer writes it as a sorted list.

    To fix it, declare a ``list`` or a ``tuple`` in the set's place, in the field of your own class too, and make a
    set of it inside the node where you need one.

    Args:
        type: Where the set is: the field of a model, dataclass or ``TypedDict`` that holds it, as ``LabelSet.labels``,
            a table's row's included, or else the set itself, as its developer wrote it, like ``set[str]`` inside
            ``list[set[str]]``.

    Example:
        from pydantic import BaseModel


        @host_type("label-set", title="Label set")
        class LabelSet(BaseModel, frozen=True):
            labels: set[str]


        def label_count(given: LabelSet) -> int:
            return len(given.labels)


        declaration_refused(lambda: node("label-count", title="Label count", description="Counts the labels.")(label_count))
    """
    return f"a set has no order, so conductor doesn't take one in a declared type, and {type} is or holds one. Declare a list or a tuple in its place."


@DECLARATION_PROBLEMS.problem
def union_members_alike(*, type: str, member: str, alike: str) -> str:
    """An input's or output's type is or holds a union with two members JSON can't tell apart by their class, like
    ``Heron | Egret``, two models, or ``str | date``.

    A run keeps every value as JSON, which says nothing of a value's class, and pydantic reads a union back as the
    member that takes the value best, the first where two take it as well. So an ``Egret`` a node returned could come
    back as a ``Heron`` from a saved run, or across an edge that reads it again, and a ``date`` as a ``str``. Two
    members are written alike when JSON writes both as the same kind, an object, an array, a text, a number or a
    yes or no, and they are of different classes, where one could read the other's value:

    - two objects: a model, a dataclass, a ``TypedDict``, a dict or a ``Mapping``, generic or not;
    - two texts: ``str``, ``date``, ``datetime``, ``time``, ``timedelta``, ``Decimal``, ``bytes`` or ``UUID``;
    - a number beside an enum or a ``Literal`` of numbers, like ``int | IntEnum``, or ``Literal[1] | float``;
    - an enum beside another enum, a ``Literal`` or the type of its values that shares a value with it;
    - a list beside a tuple, and two lists, or two tuples, whose items are alike, like ``list[Heron] |
      list[Egret]``.

    Such a union is refused even where pydantic might tell its members apart, by their fields or their types.
    ``X | None``, members of different JSON kinds, like ``int | str``, a ``Literal`` beside the type of its values,
    like ``Literal["auto"] | str``, ``int | float``, which JSON tells apart by a float's point, and ``float | str``
    are taken. A float that isn't finite is written as text, so under ``float | str`` it reads back as its text.

    To fix it, give each model a field of its own ``Literal`` and name it with pydantic's ``Discriminator``, as
    ``Annotated[Heron | Egret, Discriminator("kind")]``. For other members, declare one type, or types JSON writes
    as different kinds, like a model holding the value; for an enum, give it values the other member doesn't take.

    Args:
        type: Where the union is: the field of a model, dataclass or ``TypedDict`` that holds it, as
            ``Sighting.bird``, or else the union, as ``Heron | Egret``.
        member: One of the two members, such as ``Heron``.
        alike: The other, such as ``Egret``.

    Example:
        from pydantic import BaseModel


        @host_type("heron", title="Heron")
        class Heron(BaseModel, frozen=True):
            name: str


        @host_type("egret", title="Egret")
        class Egret(BaseModel, frozen=True):
            name: str


        def spot_bird(text: str) -> Heron | Egret:
            return Egret(name=text)


        declaration_refused(lambda: node("spot-bird", title="Spot bird", description="Names the bird in a text.")(spot_bird))
    """
    return f"{member} and {alike} in {type} are written alike as JSON, so a saved value may read back as the other one; such a union is refused even where pydantic might tell them apart. For two models, give each a field of its own Literal and write Annotated[{member} | {alike}, Discriminator('kind')]; else declare one type, or types JSON writes differently, or, for an enum, give it values the other member doesn't take."


@DECLARATION_PROBLEMS.problem
def type_not_supported_secret(*, type: str) -> str:
    """An input's or output's type is one of pydantic's secrets, or holds one, like ``SecretStr``, ``SecretBytes``
    or ``Secret[int]``.

    A secret's JSON form is a mask, ``"**********"``. A run saves its values as JSON and matches a call by its
    arguments' JSON, so a saved run would keep only the mask, and every secret would match every other: a call given
    a new password would reuse what the old one gave.

    To fix it, give the secret to the run in its context (``context=``), which is never saved or matched, and take it
    in the node with ``ctx: Context[X]``.

    Args:
        type: The secret type, as its developer wrote it.

    Example:
        from pydantic import SecretStr


        def log_in(password: SecretStr) -> bool:
            return bool(password.get_secret_value())


        declaration_refused(lambda: node("log-in", title="Log in", description="Logs in.")(log_in))
    """
    return f"a run saves and matches {type} by its JSON, which is a mask. Give the secret to the run in its context."


@DECLARATION_PROBLEMS.problem
def type_not_supported_import_string(*, type: str) -> str:
    """An input's or output's type is pydantic's ``ImportString``, or holds one, like ``ImportString[Callable]``.

    An ``ImportString`` imports the module its value names, and importing a module runs its code. A graph record is
    data that an editor writes, and a run's inputs come from its caller, so either could make the node run any code
    at all. Data must never run code.

    To fix it, take the name as a ``str``, or a ``Literal`` of the names you allow, and look up what it names inside
    the node.

    Args:
        type: The type, as its developer wrote it.

    Example:
        from pydantic import ImportString


        def plugin(entry: ImportString[float]) -> float:
            return entry


        declaration_refused(lambda: node("plugin", title="Plugin", description="Loads a plugin.")(plugin))
    """
    return f"{type} imports what its value names, and importing runs code. Take the name as a str instead."


@DECLARATION_PROBLEMS.problem
def type_not_supported_decoding(*, type: str) -> str:
    """An input's or output's type decodes its value, like pydantic's ``Json[int]``, ``Base64Bytes`` or
    ``Base64Str``, or holds one.

    ``Json[int]`` reads JSON text into an ``int``, and ``Base64Bytes`` reads base64 text into ``bytes``. Conductor
    checks a value by its type wherever it enters or leaves a node, and reads a saved run back by it, so a value that
    was decoded once is decoded again, and the second time it fails.

    To fix it, declare the type the text decodes to: ``int`` for ``Json[int]``, ``bytes`` for ``Base64Bytes`` and
    ``str`` for ``Base64Str``.

    Args:
        type: The type, as its developer wrote it.

    Example:
        from pydantic import Json


        def payload(body: Json[int]) -> int:
            return body


        declaration_refused(lambda: node("payload", title="Payload", description="Reads a payload.")(payload))
    """
    return f"{type} decodes its value, so a value it decoded fails it the next time. Declare the type it decodes to."


@DECLARATION_PROBLEMS.problem
def type_not_supported_bytes_as_text(*, cls: str, field: str) -> str:
    """An input's or output's type holds a pydantic model or a pydantic dataclass of the host's with a ``bytes``
    field, and the class's config reads and writes bytes as text, as pydantic's default config does.

    JSON has no bytes, so conductor writes them as base64 text and reads them back from it. A pydantic model reads
    and writes its own fields by its own config, whatever conductor's says. So bytes given in a mapping for the
    model, such as ``{"data": b"hello"}`` as a run's input or a record's constant, would be written as base64 and
    then read by the model as text: the node would get ``b"aGVsbG8="``.

    A ``TypedDict`` or a plain dataclass given a config with pydantic's ``with_config`` reads its fields by that
    config too, so it is refused the same way.

    To fix it, build the class's config from ``conductor.JSON_CONFIG``, as ``ConfigDict(**JSON_CONFIG, ...)``, or give a
    ``TypedDict`` or dataclass ``with_config(JSON_CONFIG)``. For a class from a library, declare a subclass of it with
    that config, and build the subclass: an instance of the library's class is written by its own config. A library's
    class nested inside it needs a subclass of its own.

    Args:
        cls: The model or pydantic dataclass, by its name.
        field: Its field that holds the bytes.

    Example:
        from pydantic import BaseModel


        @host_type("attachment", title="Attachment")
        class Attachment(BaseModel, frozen=True):
            name: str
            data: bytes


        def attachment_size(file: Attachment) -> int:
            return len(file.data)


        declaration_refused(lambda: node("attachment-size", title="Attachment size", description="Measures a file.")(attachment_size))
    """
    return (
        f"{cls} reads and writes its field {field!r} as text, and conductor carries bytes as base64 text. "
        f"Build the config of {cls} from conductor.JSON_CONFIG, as ConfigDict(**JSON_CONFIG, ...)."
    )


@DECLARATION_PROBLEMS.problem
def host_type_not_supported(*, id: str, cls: str) -> str:
    """``@host_type`` is put above a class that pydantic can't validate or write as JSON.

    To fix it, make the class a frozen pydantic model or a frozen dataclass, with ``@host_type`` above
    ``@dataclass(frozen=True)``. Or give the class its own ``__get_pydantic_core_schema__``.

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
        f"@host_type({id!r}) can't declare {cls}, since pydantic can't check it or write it as JSON. Make it a frozen pydantic model or a "
        "frozen dataclass, with @host_type above @dataclass(frozen=True), or give it its own __get_pydantic_core_schema__."
    )


@DECLARATION_PROBLEMS.problem
def type_not_named(
    *, cls: str, suggested_id: str, suggested_title: str, parent: str | None = None
) -> str:
    """An input's or output's type is a class of the host's that has no ``@host_type``.

    The id that ``@host_type`` gives is what an editor and a saved graph call the type. A subclass doesn't inherit its
    parent's declaration, so it needs one of its own.

    To fix it, write ``@host_type`` above the class. The message suggests an id and a title made from the class's
    name, such as ``@host_type("invoice-line", title="Invoice line")`` for ``InvoiceLine``.

    Args:
        cls: The class, by its module and name.
        suggested_id: An id made from the class's name, such as ``"invoice-line"``.
        suggested_title: A title made from the class's name, such as ``"Invoice line"``.
        parent: A base class that is declared, if there is one.

    Example:
        @dataclass
        class Unnamed:
            kroner: int


        def unnamed_type(values: list[Unnamed]) -> str:
            return ""


        declaration_refused(lambda: node("unnamed-type", title="Unnamed type", description="Hands back an empty text.")(unnamed_type))
    """
    why = (
        f" {parent} is declared, but a subclass doesn't inherit its declaration." if parent else ""
    )
    return (
        f"{cls} is the type of a port, and it has no @host_type. "
        f'Write @host_type("{suggested_id}", title="{suggested_title}") above the class.{why}'
    )


@DECLARATION_PROBLEMS.problem
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
    return f"the type variable {variable} is constrained to a few types. Give it a bound instead, like T: int | str, or write the union."


@DECLARATION_PROBLEMS.problem
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
    return f"the type variable {variable} is on no input, so no edge can say what it is. Put it on an input, or use compute_outputs."


@DECLARATION_PROBLEMS.problem
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
    return f"the new version drops {comma_separated(gone)}, which the version it replaces has. Say where they went in its Upgrade."


@DECLARATION_PROBLEMS.problem
def upgrade_renames_unknown(*, old: list[str], new: list[str]) -> str:
    """A node's ``Upgrade`` renames a port that the version it replaces doesn't declare, or renames one to a port
    that the new version doesn't declare.

    A rename names an old input or output and its new name. A name that isn't a port, such as a typo, moves nothing,
    so a graph upgraded by it would lose what that port got.

    Only declared ports are checked. A ``compute_added_inputs`` hook adds inputs when a graph is compiled, and a
    ``compute_outputs`` hook gives the outputs then. So a version with one of those hooks has its renames of that kind
    of port left unchecked.

    To fix it, spell each old name as the version it replaces declares it, and each new name as the new version does.

    Args:
        old: The old names in ``renamed_inputs`` and ``renamed_outputs`` that the version it replaces doesn't declare.
        new: The new names that the new version doesn't declare.

    Example:
        @node("salute", title="Salute v1", description="Salutes a person.")
        def salute_v1(name: str) -> str:
            return name


        def salute(first: str) -> str:
            return first


        declaration_refused(lambda: node("salute", title="Salute", description="Salutes a person.", replaces=salute_v1, upgrade=Upgrade(renamed_inputs={"nmae": "first"}))(salute))
    """
    return (
        f"the Upgrade renames ports that aren't declared: {comma_separated(old)} in the version it replaces, "
        f"{comma_separated(new)} in the new version. Spell each name as its version declares it."
    )


@DECLARATION_PROBLEMS.problem
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
    return f"{declared_with}({id!r}) has an upgrade= but no replaces=. Name the version it upgrades from with replaces=."


@DECLARATION_PROBLEMS.problem
def var_keyword_without_hook() -> str:
    """A node takes a ``**`` parameter, and has no ``compute_added_inputs`` hook.

    The ``**`` parameter is where a node takes the inputs that hook adds.

    To fix it, remove the ``**`` parameter, or add the hook with ``compute_added_inputs=``.

    Example:
        def loose(**values: str) -> str:
            return ""


        declaration_refused(lambda: node("loose", title="Loose", description="Hands back an empty text.")(loose))
    """
    return "the node takes a ** parameter, but has no compute_added_inputs hook to fill it. Remove the ** parameter, or add compute_added_inputs=."


@DECLARATION_PROBLEMS.problem
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
    return f"{declared_with}({id!r}) is version {version}, which isn't after version {replaced}, the one it replaces. Leave version= out, or give a number above {replaced}."


@DECLARATION_PROBLEMS.problem
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
    return "the widget is written in Annotated next to the type, where it is ignored. Put it on Param: Annotated[..., Param(widget=...)]."


@DECLARATION_PROBLEMS.problem
def widget_without_json_form(*, widget: str) -> str:
    """A port's widget has a value that can't be written as JSON, so it can't be sent to an editor.

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
    return f"the widget {widget} has a value that can't be written as JSON. Give its fields values that can be."
