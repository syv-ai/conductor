"""A node's hook: a function that works out some of the node's inputs or outputs when a graph is compiled.

A node has at most two hooks, given to ``@node`` as ``compute_added_inputs=`` and ``compute_outputs=``. Say a node
takes a list of column names as a constant, and needs one input per column::

    def columns(names: list[str]) -> dict[str, type]:
        return {name: str for name in names}

    @node("pick", title="Pick", description="Picks columns.", compute_added_inputs=columns)
    def pick(names: list[str], **values: str) -> str: ...

``@node`` reads the hook's parameters once, into a ``Hook``. ``names`` names an input, so the hook gets that input's
constant, and the input takes only a constant. A parameter annotated ``type`` gets the type arriving on the input
of its name instead. When a graph is compiled, the compiler calls ``hook.call(...)``, which gives each parameter its
value, calls the function and checks what it returns.

The rules are in the reference: ``python -m conductor.about field-hooks``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar

from conductor._internal.context import ContextParameter
from conductor._internal.field import Field
from conductor._internal.problem import (
    InvalidCallError,
    NodeDeclarationError,
    author_code_problem,
    compile_problems,
    declaration_problems,
)

from .type import HookName
from .validator import HookValidator


@dataclass(frozen=True)
class Hook:
    """One hook of a node, with what each of its parameters gets.

    ``@node`` makes it with ``Hook.read``, and the node's definition keeps it as ``compute_added_inputs`` or
    ``compute_outputs``.

    Attributes:
        name: Which hook it is: ``compute_added_inputs`` or ``compute_outputs``.
        function: The function the node's developer wrote.
        values_of: The inputs whose values the hook gets, each by a parameter of the input's name. Each of them takes
            only a constant, since the hook reads it before any run.
        types_of: The inputs whose arriving type the hook gets, each by a parameter of the input's name annotated
            ``type``. ``Any`` when no edge leads into the input.
        types_of_added: Whether the hook takes ``**added: type``, the type arriving on each input that
            ``compute_added_inputs`` added.
        taken: The names the hook can't give, since the node declares them itself: its inputs and its ``Context``
            parameter, for ``compute_added_inputs``.
        failure_codes: The codes of the problems a call of a hook can give: it raised a ``NodeError``, it raised any
            other exception, or it returned what compiling can't use.
    """

    name: HookName
    function: Callable[..., Mapping[Any, Any]]
    values_of: frozenset[str]
    types_of: frozenset[str]
    types_of_added: bool
    taken: frozenset[str]

    failure_codes: ClassVar[frozenset[str]] = frozenset(
        {
            compile_problems.compute_hook_failed.code,
            compile_problems.compute_hook_raised.code,
            compile_problems.compute_hook_returned_invalid.code,
        }
    )

    @classmethod
    def read(
        cls,
        function: Callable[..., Mapping[Any, Any]],
        name: HookName,
        inputs: Mapping[str, Field],
        context: ContextParameter | None,
        *,
        adds_inputs: bool,
    ) -> Hook:
        """Reads a hook's parameters, and returns the hook.

        Each parameter is passed by name and names an input of the node. A parameter annotated ``type`` gets the
        type arriving on that input. The one exception is ``**added: type`` on ``compute_outputs``, which gets the
        type arriving on each added input.

        Args:
            function: The hook's function.
            name: Which hook it is.
            inputs: The node's declared inputs, by name.
            context: The node's ``Context`` parameter, or ``None``.
            adds_inputs: Whether the node has a ``compute_added_inputs`` hook, so there are added inputs to read.

        Raises:
            NodeDeclarationError: A parameter can't be passed by name, asks for an arriving type in
                ``compute_added_inputs``, or asks for a value and names no declared input. Or it is ``**kwargs``
                anywhere except as ``**added: type`` on the ``compute_outputs`` of a node that adds inputs.
        """
        values_of: set[str] = set()
        types_of: set[str] = set()
        types_of_added = False
        for parameter_name, parameter in inspect.signature(function).parameters.items():
            arriving = parameter.annotation in (type, "type")
            if arriving and name == "compute_added_inputs":
                raise NodeDeclarationError(
                    declaration_problems.arriving_on_compute_added_inputs(parameter=parameter_name)
                )
            if parameter.kind is parameter.VAR_KEYWORD:
                if not (arriving and adds_inputs):
                    raise NodeDeclarationError(
                        declaration_problems.hook_parameter_var_keyword(
                            hook=name, parameter=parameter_name
                        )
                    )
                types_of_added = True
            elif (
                parameter.kind is not parameter.POSITIONAL_OR_KEYWORD
                and parameter.kind is not parameter.KEYWORD_ONLY
            ):
                raise NodeDeclarationError(
                    declaration_problems.hook_parameter_not_by_name(
                        hook=name, parameter=parameter_name
                    )
                )
            elif arriving:
                types_of.add(parameter_name)
            elif parameter_name not in inputs:
                raise NodeDeclarationError(
                    declaration_problems.hook_parameter_unknown(
                        hook=name, parameter=parameter_name, inputs=sorted(inputs)
                    )
                )
            else:
                values_of.add(parameter_name)
        taken: frozenset[str] = frozenset()
        if name == "compute_added_inputs":
            taken = frozenset(inputs) | ({context.name} if context is not None else frozenset())
        return cls(name, function, frozenset(values_of), frozenset(types_of), types_of_added, taken)

    def call(
        self,
        node_id: str,
        values: Mapping[str, Any],
        arriving: Mapping[str, Any] = MappingProxyType({}),
        added: tuple[str, ...] = (),
    ) -> dict[str, Field]:
        """Calls the hook for one node of a graph, and returns the fields it gives, by name.

        Args:
            node_id: The node's id. The problems name it.
            values: The value of each input in ``values_of``: its constant, else its default.
            arriving: The type the edges into each input carry, by input name.
            added: The inputs ``compute_added_inputs`` added.

        Raises:
            InvalidCallError: One problem of a code in ``failure_codes``. ``compute_hook_failed`` or
                ``compute_hook_raised`` when the function raises, as ``author_code_problem`` reports it, and
                ``compute_hook_returned_invalid`` when ``HookValidator`` refuses what it returns.
        """
        kwargs = {name: values[name] for name in self.values_of}
        kwargs |= {name: arriving.get(name, Any) for name in self.types_of}
        if self.types_of_added:
            kwargs |= {name: arriving.get(name, Any) for name in added}
        try:
            returned = self.function(**kwargs)
        except Exception as error:  # noqa: BLE001  the hook is the node author's code, and may raise anything
            raise InvalidCallError(
                [
                    author_code_problem(
                        error,
                        "the %s hook of node %r failed",
                        self.name,
                        node_id,
                        raised=partial(compile_problems.compute_hook_raised, node_id),
                        failed=partial(compile_problems.compute_hook_failed, node_id),
                    )
                ]
            ) from None
        return HookValidator(node_id, self.name, self.taken)(returned)
