"""A node function's source, as ``@node`` reads it.

``@node`` reads the source of the function under it once, with ``read_code``, and gets two things from it. One is
the code as text, which a compiled node hashes into its code fingerprint. The other is whether its own body has
a ``raise Ask(...)``, so a node that asks a person without ``asks=True`` is refused where it is declared.
"""

from __future__ import annotations

import ast
import inspect
from ast import (
    AST,
    AsyncFunctionDef,
    Attribute,
    Call,
    ClassDef,
    FunctionDef,
    Lambda,
    Name,
    Pass,
    Raise,
)
from collections.abc import Callable
from tokenize import TokenError
from types import CodeType
from typing import Any

from conductor._internal.ask import Ask


def read_code(function: Callable[..., Any]) -> tuple[str, bool]:
    """Returns a function's code as text, and whether it raises ``Ask``, from one reading of its source.

    The text is the same on every Python version (see ``_code_text``). A function whose source can't be read (see
    ``_own_definition``) is written from its bytecode instead, which changes with each Python version and keeps
    docstrings. Its ``Ask`` can't be seen, so it counts as raising none.
    """
    function = inspect.unwrap(function)
    definition = _own_definition(function)
    if definition is None:
        return _bytecode_text(function.__code__), False
    # Before the text is written, since writing it strips the definition's name and docstrings.
    raises = _raises_ask(function, definition)
    return _code_text(definition), raises


def _code_text(definition: FunctionDef | AsyncFunctionDef) -> str:
    """Returns a function's code as text that's the same on every Python version, leaving out docstrings.

    The text is the parsed ``def``, written out again by ``ast.unparse``. It leaves out the function's name and
    decorators, and the docstrings of the function and of every function and class inside it. So a comment, a blank
    line, a docstring, the function's name or its ``@node`` line can change, and the text stays the same. A
    docstring that's all of a body is written as ``pass``. It changes ``definition`` in place.
    """
    definition.name, definition.decorator_list = "", []
    for part in ast.walk(definition):
        if (
            isinstance(part, FunctionDef | AsyncFunctionDef | ClassDef)
            and ast.get_docstring(part, clean=False) is not None
        ):
            part.body = part.body[1:] or [Pass()]
    return ast.unparse(definition)


def _raises_ask(function: Callable[..., Any], definition: FunctionDef | AsyncFunctionDef) -> bool:
    """Whether a function's own body has a ``raise`` of conductor's ``Ask``, or of a subclass of it.

    It reads the function's own body only. A function, lambda or class defined inside has names of its own, and may
    never be called, so what it raises is left to the run. The name raised is looked up where the function looks it
    up when it runs: in its closure, then its module. So ``raise A(...)`` after ``from conductor import Ask as A`` at
    the top of the module counts, and so does ``raise conductor.Ask(...)``. A ``raise`` counts wherever it is in the
    body, even under ``if False:``.

    It sees only what the source says. An ``Ask`` raised by a function this one calls, by a function defined inside
    it, or kept in a variable first, and a name the function binds itself, such as an import inside its body, all
    give ``False``. The run refuses such an ``Ask`` when it is raised (``ask_not_declared``).
    """
    code = function.__code__
    closure = dict(zip(code.co_freevars, function.__closure__ or (), strict=True))
    own = {*code.co_varnames, *code.co_cellvars}

    def named(expression: ast.expr) -> object:
        """The value a name or a dotted name in the source stands for, or ``None`` if it can't be told."""
        if isinstance(expression, Attribute):
            return inspect.getattr_static(named(expression.value), expression.attr, None)
        if not isinstance(expression, Name) or expression.id in own:
            return None
        if expression.id in closure:
            try:
                return closure[expression.id].cell_contents
            # A cell that is empty until the enclosing function sets it.
            except ValueError:
                return None
        return function.__globals__.get(expression.id)

    parts: list[AST] = list(definition.body)
    while parts:
        part = parts.pop()
        if isinstance(part, Raise) and part.exc is not None:
            raised = named(part.exc.func if isinstance(part.exc, Call) else part.exc)
            if isinstance(raised, type) and issubclass(raised, Ask):
                return True
        # A scope of its own, whose names the function's own names don't say.
        if not isinstance(part, FunctionDef | AsyncFunctionDef | ClassDef | Lambda):
            parts.extend(ast.iter_child_nodes(part))
    return False


def _own_definition(function: Callable[..., Any]) -> FunctionDef | AsyncFunctionDef | None:
    """Returns the ``def`` of a function, parsed from its source, or ``None`` if its source can't be read.

    That is any function whose source ``inspect.getsource`` can't read: one made by ``exec`` with no file, one typed
    into ``python -c`` or a REPL that keeps no source, or one from a package installed without its ``.py`` files. It
    is also a lambda, whose source is the whole line it's on, which may hold other code. And it is a function whose
    lines at its place aren't Python, hold no def, or start with a def of another name than the one its code was
    compiled with, as when ``exec`` compiles a text under a real file's name. A def of the same name at the same
    place is taken as the function's own.
    """
    # The name the code was compiled with, which a factory that sets ``__name__`` after the def doesn't change.
    name = function.__code__.co_name
    if name == "<lambda>":
        return None
    try:
        source = inspect.getsource(function)
        # A function defined inside another is indented, and a string in it may start at column 0, so the source is
        # parsed inside an "if" rather than dedented.
        tree = ast.parse(f"if True:\n{source}" if source[0].isspace() else source)
    # OSError: no file. TokenError and SyntaxError: the lines read aren't Python, so they aren't this function.
    except (OSError, TokenError, SyntaxError):
        return None
    definition = next(
        (part for part in ast.walk(tree) if isinstance(part, FunctionDef | AsyncFunctionDef)),
        None,
    )
    if definition is None or definition.name != name:
        return None
    return definition


def _bytecode_text(code: CodeType) -> str:
    """Returns a function's bytecode, constants and names as text that's the same in every process.

    A function or class defined inside is written out from its own code.
    """
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in code.co_consts]}:{code.co_names}"


def _constant_text(constant: Any) -> str:
    """Returns a constant as text that's the same in every process.

    A set's members are sorted, since a set's order depends on the process's hash seed.
    """
    if isinstance(constant, CodeType):
        return _bytecode_text(constant)
    if isinstance(constant, frozenset):
        return f"frozenset({sorted(_constant_text(c) for c in constant)})"
    if isinstance(constant, tuple):
        return f"({', '.join(_constant_text(c) for c in constant)},)"
    return repr(constant)
