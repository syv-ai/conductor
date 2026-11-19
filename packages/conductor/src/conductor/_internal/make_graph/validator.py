"""``ArgumentsValidator``, the check of what ``make_graph`` is given.

``make_graph`` takes Python values: node definitions, edges as pairs of text, and mappings keyed by text. This checks
that each argument has the shape its annotation says, before anything is written. What the text says, such as
whether ``"shout.mark"`` is a place, is read after it, by ``make_graph`` itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypedDict, cast

from annotated_types import Len
from pydantic import (
    BeforeValidator,
    ConfigDict,
    PlainValidator,
    StrictStr,
    TypeAdapter,
    ValidationError,
    with_config,
)

from conductor._internal.node import FunctionDefinition, NodeDefinition, SubgraphDefinition
from conductor._internal.problem import (
    InvalidCallError,
    call_problems,
    conductor_error,
    shape_error,
)
from conductor._internal.validation import Validator

#: The two classes of a node definition.
_DEFINITIONS = (FunctionDefinition, SubgraphDefinition)


def _node(given: Any) -> NodeDefinition | tuple[str, NodeDefinition]:
    """Takes one node given to ``make_graph``: a definition, or a tuple of an id and a definition, such as
    ``("a", upper)``.

    Raises:
        PydanticCustomError: ``not_a_node``, for anything else, a list ``["a", upper]`` included.
    """
    if isinstance(given, _DEFINITIONS) or (
        isinstance(given, tuple)
        and len(given) == 2
        and isinstance(given[0], str)
        and isinstance(given[1], _DEFINITIONS)
    ):
        return cast("NodeDefinition | tuple[str, NodeDefinition]", given)
    raise conductor_error(
        "not_a_node",
        "A node is a node definition, or a tuple of an id and a definition such as ('a', upper)",
    )


def _one_or_many(given: Any) -> Any:
    """Takes one place given in ``inputs=`` as a list of that one place, so ``"greet.name"`` is ``["greet.name"]``."""
    return [given] if isinstance(given, str) else given


@with_config(ConfigDict(strict=True))
class Arguments(TypedDict):
    """``make_graph``'s arguments, by name, as its signature declares them.

    ``inputs`` gives each graph input a list of places, though one place may be given as text.
    """

    nodes: Sequence[Annotated[NodeDefinition | tuple[str, NodeDefinition], PlainValidator(_node)]]
    edges: Sequence[Annotated[Sequence[StrictStr], Len(2, 2)]] | None
    constants: Mapping[StrictStr, Any]
    defaults: Mapping[StrictStr, Any]
    inputs: Mapping[StrictStr, Annotated[Sequence[StrictStr], BeforeValidator(_one_or_many)]]
    outputs: Mapping[StrictStr, StrictStr]


_ARGUMENTS = TypeAdapter(Arguments)


class ArgumentsValidator(Validator[Mapping[str, Any], Arguments]):
    """The validator for ``make_graph``'s arguments, given by name.

    Each argument is held to its annotation, strictly. A sequence is a list or a tuple, never a set, text or bytes;
    a mapping is any ``Mapping``, a ``MappingProxyType`` included; text is a ``str`` or a subclass of it. So
    ``edges=[["a.result", "b.line"]]`` is taken, and ``edges={("a.result", "b.line")}``, ``outputs=None`` and
    ``make_graph(["a", upper])`` are refused.

    Raises (when called):
        InvalidCallError: One ``invalid_argument`` for each argument refused, with every fault in it in
            ``ctx.errors``.
    """

    def __call__(self, arguments: Mapping[str, Any]) -> Arguments:
        """Checks ``arguments`` and returns them, with each graph input's places as a list.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            return _ARGUMENTS.validate_python(dict(arguments))
        except ValidationError as error:
            faults: dict[str, list[dict[str, Any]]] = {}
            for fault in error.errors():
                argument, *steps = fault["loc"]
                # A key refused for not being text is part of what was given, so the loc stops at it.
                if steps[-1:] == ["[key]"]:
                    steps = [*steps[:-2], "[key]"]
                faults.setdefault(str(argument), []).append(shape_error(fault, steps))
        raise InvalidCallError(
            [
                call_problems.invalid_argument(argument, reason=errors[0]["msg"], errors=errors)
                for argument, errors in faults.items()
            ]
        )
