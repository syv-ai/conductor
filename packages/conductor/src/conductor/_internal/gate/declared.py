"""The declaration door: a default a node author declares, read by its input's or output's type when the node is
declared.

Example: ``shout(line: str, mark: str = "!")`` declares the default ``"!"``, and ``declared_field("mark", str, "!")``
gives the input ``mark`` with that default. ``times: Annotated[int, Param(ge=0)] = -1`` is refused when the module
that declares the node is imported: ``NodeDeclarationError`` with the code ``default_invalid``, naming the input.

A default is read as a Python value is read (``1`` for a ``float`` is ``1.0``). The defaults a graph record gives
are read by the constants door instead, when the graph is compiled.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from conductor._internal.field import Field
from conductor._internal.gate.rules import validator_of
from conductor._internal.markers import REQUIRED, SKIPPED
from conductor._internal.problem import NodeDeclarationError
from conductor._internal.problem import declarations as declaration


def declared_field(name: str, hint: Any, default: Any) -> Field:
    """Returns the input or output ``name`` that a node declares with the type hint ``hint`` and ``default``, the
    default read by the field's type.

    Args:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as written: ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch not taken,
            ``None`` for a blank. These three are kept as they are.

    Raises:
        NodeDeclarationError: Naming the field: what ``Field`` refuses, or a default its type refuses
            (``default_invalid``).
    """
    if default is REQUIRED or default is SKIPPED or default is None:
        return Field(name, hint, default)
    whole = Field(name, hint).whole
    try:
        read = validator_of(whole).validate_python(default)
    except ValidationError as error:
        reason = whole.refusal(error.errors(include_url=False, include_input=False)).reason
        raise NodeDeclarationError(declaration.default_invalid(default=repr(default), type=whole.id, reason=reason), name) from None
    return Field(name, hint, read)
