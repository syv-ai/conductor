"""Every kind of problem a node's or a type's declaration can have, one function each, in ``DECLARATION_PROBLEMS``.

``@node``, ``subgraph``, ``@host_type``, ``Param`` and ``NodeRegistry`` raise them as a ``NodeDeclarationError`` while
the module that declares the node or the type is imported, so only the node's developer meets them::

    @node("fetch", title="Fetch", description="Fetches a page.")
    async def fetch(url: str) -> str: ...
    # NodeDeclarationError: @node('fetch') on fetch: a node is a plain def; ...
    # error.code == "async_run", error.problem == async_run()

A kind is declared as the kinds of ``PROBLEMS`` are (``ProblemCatalogue.problem`` gives the rules): its function
returns the message, and its keyword-only parameters are the keys of ``ctx``. A declaration problem has no ``loc``;
the error's ``where`` says where the fault is (``@node('fetch') on fetch``, then the input's name), and its text puts
that in front of the message. Some codes have several kinds, one per mistake, each with a message of its own: the
decorator names the code and a variant, which the problem's ``ctx.variant`` gives. Each kind's docstring is its entry on the problems page
(``python -m conductor.about problems``), and its example runs after the setup at the top of that page; its last
line, ``declaration_refused(lambda: ...)``, gives the problem the declaration raises, and a test checks it is of
this kind.
"""

from __future__ import annotations

from typing import Any, Literal

from .type import ProblemCatalogue

#: Every kind of problem a declaration can have.
DECLARATION_PROBLEMS = ProblemCatalogue(
    "Declaring a node",
    "Raised as `NodeDeclarationError` when the module that declares the node or the type is imported. `error.code` is "
    "the code and `error.problem` the problem, whose message is written for the node's developer; the error's text puts "
    "where the fault is in front of it: `@node('adds') on adds: value: ...`.",
)


@DECLARATION_PROBLEMS.problem()
def arriving_on_compute_added_inputs(*, parameter: str) -> str:
    """A ``compute_added_inputs`` hook takes the type arriving on an input.

    The inputs a node adds are decided before any edge is read, so nothing has arrived yet. Read the arriving type in
    ``compute_outputs``.

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
    """A node's function is ``async def``.

    Write a plain ``def``. For a slow call, ``Policy(concurrency=8)`` makes several calls at once on threads.

    Example:
        async def fetch_async(url: str) -> str:
            return url


        declaration_refused(lambda: node("fetch_async", title="Fetch async", description="Fetches a page from an address.")(fetch_async))
    """
    return "a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)"


@DECLARATION_PROBLEMS.problem()
def bound_invalid(*, unheld: list[str], type: str) -> str:
    """A ``Param`` sets a limit pydantic holds no rule for on the input's type.

    Use limits pydantic holds on the type: ``ge``, ``gt``, ``le``, ``lt`` and ``multiple_of`` on numbers (a
    ``Decimal`` too), dates, times and durations; ``min_length`` and ``max_length`` on texts, bytes, lists, tuples, sets
    and dicts; ``pattern`` on texts. A table takes none. The same holds for a ``Param`` on a field of a model a node
    takes.

    Args:
        unheld: The limits pydantic holds no rule for.
        type: The type, as its developer wrote it.

    Example:
        def short(text: Annotated[str, Param(ge=0)]) -> str:
            return text


        declaration_refused(lambda: node("short", title="Short", description="Passes the text on as it is.")(short))
    """
    return f"{unheld} can't apply to {type}, as pydantic holds no such limit on it; take them off, or check the value in the node"


@DECLARATION_PROBLEMS.problem("bound_invalid", variant="limit_value")
def bound_invalid_value(*, limits: str, type: str, error: str) -> str:
    """A ``Param`` sets a limit to a value of the wrong kind for the type, such as ``ge="a"`` on an ``int``.

    Give each limit a value of the input's type: a number for ``ge`` on an ``int``.

    Args:
        limits: The limits, as written.
        type: The type's id.
        error: pydantic's account of why it cannot build the type with them.

    Example:
        def narrow(number: Annotated[int, Param(ge="a")]) -> int:  # type: ignore[arg-type]
            return number


        declaration_refused(lambda: node("narrow", title="Narrow", description="Passes the number on as it is.")(narrow))
    """
    return f"the limits {limits} don't fit {type}, so change one: {error}"


@DECLARATION_PROBLEMS.problem()
def choices_invalid() -> str:
    """A ``Param``'s choices give one id twice, or give none.

    Give each choice once, each a value of the input's type.

    Example:
        from conductor import Choice


        def doubled(pick: Annotated[str, Param(choices=[Choice("s", "Small"), Choice("s", "Short")])]) -> str:
            return pick


        declaration_refused(lambda: node("doubled", title="Doubled", description="Hands back the chosen size.")(doubled))
    """
    return "the choices hold an id twice, or none; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem("choices_invalid", variant="on_a_literal")
def choices_invalid_on_a_literal(*, values: list[Any]) -> str:
    """A ``Param``'s choices on a ``Literal`` do not title its values, one each.

    On a ``Literal``, the choices give its values titles: ``[Choice("s", "Small"), Choice("m", "Medium")]`` for
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
    """A ``Param``'s choice is not a value of the input's type.

    Give each choice as a value of the input's type: a number for an ``int``.

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
    """A ``Param``'s choice has no JSON form, so an editor cannot show it nor a saved graph store it.

    Give each choice as a value of the input's type that writes as JSON.

    Example:
        from conductor import Choice


        def object_size(pick: Annotated[str, Param(choices=[Choice(object(), "Small")])]) -> str:
            return pick


        declaration_refused(lambda: node("object_size", title="Object size", description="Hands back the chosen size.")(object_size))
    """
    return "a choice has no JSON form; give each choice once, as a value of the field's type"


@DECLARATION_PROBLEMS.problem()
def compute_added_inputs_without_var_keyword() -> str:
    """A node with a ``compute_added_inputs`` hook has no ``**`` parameter to take the inputs it adds.

    Take the added inputs as ``**values: float``.

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
    """A ``Context[...]`` is written anywhere but as the whole of a node's first parameter.

    Make the context the node's first parameter, ``def maybe(ctx: Context[HostContext], text: str)``, never a later
    one, a keyword-only one, inside ``| None``, a list or a return type. The host passes it on every run, so a node can
    trust it; anywhere else it would be an input a caller fills.

    Example:
        def later(text: str, ctx: Context[HostContext]) -> str:
            return text


        declaration_refused(lambda: node("later", title="Later", description="Passes the text on as it is.")(later))
    """
    return "Context is the whole first parameter, def sign(ctx: Context[HostContext], text: str); anywhere else a caller would give its value"


@DECLARATION_PROBLEMS.problem()
def context_not_a_class() -> str:
    """A ``Context[...]`` names something that is not a class.

    Name a class: the registry's ``context_type``, a base of it, or a ``Protocol`` it satisfies.

    Example:
        def listed(ctx: Context[list[str]]) -> str:
            return ""


        declaration_refused(lambda: node("listed", title="Listed", description="Hands back an empty text.")(listed))
    """
    return "Context names the class of the object the host passes; write Context[HostContext]"


@DECLARATION_PROBLEMS.problem()
def context_type_differs(*, subregistry_type: str, registry_type: str) -> str:
    """A registry added to another names a different ``context_type``.

    Name one class on a registry and on each registry added to it, or none on a subregistry, as a registry of nodes
    that do not know the host does.

    Args:
        subregistry_type: The class the subregistry names.
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
    """A registry that names no ``context_type`` is given a subregistry that names one.

    Name the class on the registry too. A registry that names none is a ``NodeRegistry[None]`` to a type checker, so
    taking the subregistry's class when a run starts would make every ``graph.run(context=...)`` on it a type error.

    Args:
        subregistry_type: The class the subregistry names.

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

    ``Policy(costs_nothing=True)`` is for a node whose call costs nothing and touches nothing outside the run, which
    ``run.estimate()`` calls before any call that costs something. A node that needs the host's
    objects is not that: drop ``costs_nothing``, or give the node what it needs as an input.

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
    """An input's default is not a value of its type, or has no JSON form.

    Give a default of the input's type that writes as JSON: an editor shows it, and a saved graph stores it.

    Args:
        default: The default, as Python writes it.
        type: The input's type.
        reason: Why the type refuses it, or why it has no JSON form.

    Example:
        def counted(number: int = "many") -> int:  # type: ignore[assignment]
            return number


        declaration_refused(lambda: node("counted", title="Counted", description="Passes the number on as it is.")(counted))
    """
    return f"the default {default} is not a value of {type} that it writes as JSON: {reason}; give one that is"


@DECLARATION_PROBLEMS.problem("default_invalid", variant="none_not_taken")
def default_none_not_taken() -> str:
    """An input's default is ``None``, and its type does not take ``None``.

    Write ``text: str | None = None`` for an input that may be ``None``.

    Example:
        def maybe(text: str = None) -> str:  # type: ignore[assignment]
            return text


        declaration_refused(lambda: node("maybe", title="Maybe", description="Passes the text on as it is.")(maybe))
    """
    return "the default is None, and the type does not take None; write the type with | None"


@DECLARATION_PROBLEMS.problem()
def duplicate_node_id(*, id: str, first: str, second: str) -> str:
    """Two nodes added to one registry have the same id.

    For a node the registry already has, swap one for the other with ``registry.replace(newer)``, or give the new
    node an id of its own. The older versions of a chain are plain ``@node`` definitions that are never added; only
    the newest is added, with ``replaces=``.

    Args:
        id: The id.
        first: Where the node the registry has was declared.
        second: Where the node added was declared.

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

    Give one of the classes another id in its ``@host_type``.

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
def empty_title_or_description(*, declared_with: str, id: str, empty: Literal["title", "description"]) -> str:
    """A node is declared with an empty title or description.

    Give the node a title and a description: they are what a person reads for it in a palette.

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

    An exclusive group is a set of outputs of which a call gives at most one. Put ``exclusive_group`` on the outputs
    that are branches, the fields of the frozen dataclass the node returns.

    Example:
        def branchy(text: Annotated[str, Param(exclusive_group="x")]) -> str:
            return text


        declaration_refused(lambda: node("branchy", title="Branchy", description="Passes the text on as it is.")(branchy))
    """
    return "exclusive_group is for outputs; put it on an output's Param"


@DECLARATION_PROBLEMS.problem()
def hook_parameter_unknown(*, hook: str, parameter: str, inputs: list[str]) -> str:
    """A hook's parameter names no input of the node.

    A hook's parameter gets the constant or default of the input of its name, so name each parameter after an input
    of the node.

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

    A hook is called with each input's value by the input's name. Name each parameter after an input of the node,
    and take it by name.

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
    """A hook takes ``**`` other than as ``**name: type`` on ``compute_outputs`` of a node with ``compute_added_inputs``.

    That one ``**`` gets the type arriving on each input ``compute_added_inputs`` added. Annotate it ``type`` and
    give the node that hook, or name each parameter after an input.

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

    Give the input a type hint. The hint is the input's interface: what an edge may bring, and what a caller gives.

    Example:
        def untyped(text) -> str:  # type: ignore[no-untyped-def]
            return str(text)


        declaration_refused(lambda: node("untyped", title="Untyped", description="Passes the text on as it is.")(untyped))
    """
    return "an input needs a type hint; write one, as it is the input's interface"


@DECLARATION_PROBLEMS.problem()
def invalid_node_id(*, declared_with: str, id: str) -> str:
    """A node is declared with an id that is empty or contains ``.``, ``/``, ``@`` or ``~``.

    Those characters are written in addresses (``a.result``), node paths (``loud/shout``) and waiting ids
    (``a@1~3fa2``). Pick an id without them. The same rule holds for a node's id in a graph record, and for the names
    a hook gives inputs and outputs.

    Args:
        declared_with: ``@node`` or ``subgraph``.
        id: The id given.

    Example:
        def dotted_id(text: str) -> str:
            return text


        declaration_refused(lambda: node("a.b", title="Dotted id", description="Passes the text on as it is.")(dotted_id))
    """
    return (
        f"{declared_with} takes the node's id first, a non-empty string without '.', '/', '@' or '~', which mean something in an address "
        f"or a waiting id: {declared_with}('upper'); got {id!r}"
    )


@DECLARATION_PROBLEMS.problem()
def nested_series() -> str:
    """An input or output is a ``Series`` of ``Series``.

    Return one ``Series``. A node after it that is called once per item and returns a ``Series`` of its own makes the
    next level of items.

    Example:
        def nested(text: str) -> Series[Series[str]]:
            return [[text]]


        declaration_refused(lambda: node("nested", title="Nested", description="Wraps a text in lists.")(nested))
    """
    return "a Series of Series; a field holds one level of items, so return one Series"


@DECLARATION_PROBLEMS.problem()
def output_without_type() -> str:
    """A node's function has no return type.

    Write the return type. ``-> None`` declares no outputs.

    Example:
        def silent(text: str):  # type: ignore[no-untyped-def]
            return text


        declaration_refused(lambda: node("silent", title="Silent", description="Passes the text on as it is.")(silent))
    """
    return "a node needs a return type; write one, or -> None for no outputs"


@DECLARATION_PROBLEMS.problem()
def outputs_not_frozen() -> str:
    """A node returns a dataclass that is not frozen.

    Write ``@dataclass(frozen=True)``: several calls in a run can be given the same value, so none may change it.

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
    """A node's function takes an input by position only (before ``/``) or as ``*args``.

    Drop the ``/`` and ``*args``: the engine passes every input by its name.

    Example:
        def positional(text: str, /) -> str:
            return text


        declaration_refused(lambda: node("positional", title="Positional", description="Passes the text on as it is.")(positional))
    """
    return "a node's inputs are passed by name; drop the / and *args"


@DECLARATION_PROBLEMS.problem()
def pydantic_field_in_hint() -> str:
    """An input's or output's hint has pydantic's ``Field(...)`` in it.

    Write the title, the description and the limits on ``Param``, which takes them by the same names:
    ``Annotated[int, Param(ge=0)]``. There is one way to write them on an input or an output, so ``Field(...)`` is
    refused wherever it is written in the hint: on the parameter, on the return type, on an item
    (``Series[Annotated[int, Field(ge=0)]]``) and on a column of a table's row type. A type that has its limits is
    still a type: ``PositiveInt``, ``constr(min_length=1)`` and ``Annotated[int, annotated_types.Gt(0)]`` are read as
    pydantic reads them, and ``Field(...)`` on the fields of your own model or dataclass is that class's.

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

    pydantic's ``Discriminator`` says which member of a union a value is: write
    ``Annotated[Cat | Dog, Discriminator("kind")]``.

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
    """A node's ``replaces=`` names a node of another id.

    A version replaces an earlier ``@node`` of its own id. A different node gets no ``replaces=``.

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
    """``@host_type`` is given an id, title or ``accepts`` it cannot take.

    Give the type an id that is one word (letters, digits, ``_``, ``.`` or ``-``) and is not the id of one of
    conductor's own types, a title that is not empty, and only classes in ``accepts``.

    Args:
        id: The id given.
        faults: What is wrong with each argument.

    Example:
        declaration_refused(lambda: host_type("str", title="My text")(type("MyText", (str,), {})))
    """
    return f"@host_type({id!r}): {faults}; give an id that is one word and not one of conductor's own, a title, and classes in accepts"


@DECLARATION_PROBLEMS.problem("type_declaration_invalid", variant="not_on_a_class")
def type_declaration_not_on_a_class(*, id: str, given: str) -> str:
    """``@host_type`` is put above something that is not a class it can name, such as a function or a ``TypedDict``.

    Put ``@host_type`` above a class. A ``TypedDict`` is a record, which conductor reads by its fields and which needs
    no declaration.

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
    """``@host_type`` is put above a class Python lets nobody change, such as ``int``.

    Put the value in a class of your own and declare that.

    Args:
        id: The id given.
        cls: The class.

    Example:
        declaration_refused(lambda: host_type("whole", title="Whole number")(int))
    """
    return f"@host_type({id!r}) on {cls}: Python lets nobody write on this class; put its value in a class of your own and declare that"


@DECLARATION_PROBLEMS.problem()
def row_field_optional(*, row: str, field: str) -> str:
    """The row type of a ``Table`` or of a ``Series`` of records leaves a field out: a ``NotRequired`` field, or a
    ``TypedDict`` declared ``total=False``.

    Every row of a table has every column, so every field of its row type is required. Declare a field a row may
    have no value for as ``X | None``: its column then holds ``None`` for that row. A table with a column some rows
    leave out is not supported.

    Args:
        row: The row type's name.
        field: The first field it leaves out.

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
    """An input's or output's type is one pydantic cannot validate or write as JSON, such as a record with a field of
    a plain class.

    Use a type pydantic can validate and write as JSON: a pydantic model, a dataclass or a subclass of a builtin
    (``class Text(str)``). A class that needs its own way gives its own ``__get_pydantic_core_schema__``.

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
    """``@host_type`` is put above a class pydantic cannot validate or write as JSON.

    Make it a pydantic model or a dataclass, with ``@host_type`` above ``@dataclass``, or give the class its own
    ``__get_pydantic_core_schema__``.

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
    """An input's or output's type is a class of the host's with no ``@host_type``.

    Write ``@host_type("...", title="...")`` above the class: its id is what an editor and a saved graph call the type.
    A subclass does not inherit its parent's declaration.

    Args:
        cls: The class, by its module and name.
        parent: A base class that is declared, when there is one.

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
    """A type variable is constrained to a few types (``T: (int, str)``).

    Give the variable a bound (``T: int | str``), or write the union in the signature. A bound says what may arrive;
    the variable is then whatever does.

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

    A type variable is what arrives on an edge, so it stands on an input. An output whose type depends on a constant,
    not on an edge, is typed by a ``compute_outputs`` hook.

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
    """A node's newer version has lost an input or output of the version it replaces, and its ``Upgrade`` does not say
    where it went.

    Say where each went: ``upgrade=Upgrade(renamed_inputs={"name": "first"})``, or a ``rewrite=`` for an input; a
    gone output takes ``renamed_outputs=``, since a rewrite reads only the node's constants and defaults.

    Args:
        gone: The inputs and outputs gone.

    Example:
        @node("hello", title="Hello v1", description="Says hello to a person.")
        def hello_v1(name: str) -> str:
            return name


        def hello(first: str) -> str:
            return first


        declaration_refused(lambda: node("hello", title="Hello", description="Says hello to a person.", replaces=hello_v1)(hello))
    """
    return f"the version it replaces has {gone}, gone from this one; its Upgrade says where they went"


@DECLARATION_PROBLEMS.problem()
def upgrade_without_replaces(*, declared_with: str, id: str) -> str:
    """A node has an ``upgrade=`` and no ``replaces=``.

    Name the version it upgrades with ``replaces=``.

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
    """A node takes ``**`` and has no ``compute_added_inputs`` hook.

    The ``**`` parameter takes the inputs that hook adds. Remove it, or add the hook, ``compute_added_inputs=``.

    Example:
        def loose(**values: str) -> str:
            return ""


        declaration_refused(lambda: node("loose", title="Loose", description="Hands back an empty text.")(loose))
    """
    return "**kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs="


@DECLARATION_PROBLEMS.problem()
def version_not_after(*, declared_with: str, id: str, version: int, replaced: int) -> str:
    """A node's ``version=`` is not above the version it replaces.

    Leave ``version=`` out (one past the version replaced), or give a number above it.

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
    """A widget is written in ``Annotated`` beside the type, not on ``Param``.

    Put the widget on the input's ``Param``: ``Annotated[str, Param(widget=Textarea())]``. Other metadata in
    ``Annotated`` is pydantic's and is read as pydantic reads it, except ``Field(...)`` (``pydantic_field_in_hint``);
    a widget there would be passed by, and the input would show no control.

    Example:
        from conductor.widgets import Textarea


        def note(text: Annotated[str, Textarea()]) -> str:
            return text


        declaration_refused(lambda: node("note", title="Note", description="Hands a note on.")(note))
    """
    return "a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]"
