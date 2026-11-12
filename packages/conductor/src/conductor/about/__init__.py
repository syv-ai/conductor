"""Conductor's reference, shipped in the package: every rule on one page, ``conductor/about/llms.txt``.

The page holds the rules. The names of a public module are not written in it: a line
``<!-- names: conductor.events -->`` stands where they go, and reading the reference writes
them there from the module's ``__all__``, each with the first paragraph of its docstring. The
kinds of problem a catalogue holds are not written in it either: a line
``<!-- problems: conductor.PROBLEMS -->`` stands where they go, and reading the reference writes
each kind there from the function that declares it.

From a shell::

    python -m conductor.about                  # the whole reference
    python -m conductor.about sections         # the section slugs
    python -m conductor.about one-or-many      # one section, by slug or a part of one

The functions here are what that command calls; the module has no public names.
"""

from __future__ import annotations

import inspect
import re
import typing
from importlib import import_module
from importlib.resources import files

from conductor._internal.problem import ProblemCatalogue

_HEADING = re.compile(r"^(##+) (.+)$", re.MULTILINE)
_NAMES = re.compile(r"^<!-- names: ([\w.]+) -->$", re.MULTILINE)
_PROBLEMS = re.compile(r"^<!-- problems: ([\w.]+)\.(\w+) -->$", re.MULTILINE)
_ARGUMENT = re.compile(r"^    (\w+): ", re.MULTILINE)


def _names(module_name: str) -> str:
    """The names a public module exports, as a Markdown list in the order of its ``__all__``.

    A class or a function is followed by the first paragraph of its docstring, on one line and in
    Markdown's backticks; where that paragraph ends by introducing an example (``::``), it ends
    with a full stop. A constant or a type alias has no docstring of its own, so it is listed by
    its name alone.

    Args:
        module_name: The dotted name of the module, as ``"conductor.widgets"``.

    Returns:
        One ``- `Name`: what it is`` line per name.
    """
    module = import_module(module_name)
    lines = []
    for name in module.__all__:
        exported = getattr(module, name)
        has_docstring = inspect.isclass(exported) or inspect.isfunction(exported)
        if has_docstring:
            summary = " ".join(exported.__doc__.split("\n\n")[0].split()).replace("``", "`")
            lines.append(f"- `{name}`: {re.sub(r'::$', '.', summary)}")
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def _problems(catalogue: ProblemCatalogue) -> str:
    """The kinds of problem a catalogue holds, each as a ``####`` section under its code.

    A kind's section has the first paragraph of its docstring, its sentence with a slot for each ``ctx`` value, what
    its ``*loc`` holds when it has one, and a table of its ``ctx`` keys: the type, whether the key is always given,
    and the key's line under ``Args:`` in the docstring.

    Args:
        catalogue: The catalogue.

    Returns:
        The sections, in the order the kinds are declared.

    Raises:
        LookupError: A kind's docstring has no ``Args:`` line for one of its ``ctx`` keys.
    """
    sections = []
    for kind in catalogue:
        summary, _, arguments = inspect.cleandoc(kind.__doc__ or "").partition("\nArgs:\n")
        pieces = _ARGUMENT.split(arguments)
        told = {name: " ".join(text.split()).replace("``", "`") for name, text in zip(pieces[1::2], pieces[2::2])}
        rows = []
        for key, hint in typing.get_type_hints(kind.ctx_type).items():
            if key == "variant":
                rows.append(f"| `variant` | `'{kind.variant}'` | always | Which of the code's sentences this is. |")
                continue
            if key not in told:
                raise LookupError(f"{kind.code}: the docstring of its function has no Args line for the ctx key '{key}'")
            given = "always" if key in kind.ctx_type.__required_keys__ else "when it applies"
            rows.append(f"| `{key}` | `{inspect.formatannotation(hint)}` | {given} | {told[key]} |")
        sections.append("\n\n".join([
            f"#### `{kind.code}`" + (f", variant `{kind.variant}`" if kind.variant else ""),
            " ".join(summary.split("\n\n")[0].split()).replace("``", "`"),
            f"> {kind.sentence}",
            *([f"`loc`: {told['loc']}"] if "loc" in told else []),
            "\n".join(["| `ctx` | type | given | holds |", "|---|---|---|---|", *rows]),
        ]))
    return "\n\n".join(sections)


def _parse_sections(text: str) -> dict[str, str]:
    """Each ``##`` and ``###`` section by its slug.

    A section runs from its heading to the next heading of the same level or
    a higher one (fewer ``#``), so "Retries and Timeouts", a ``###`` inside
    "Core Concepts", ends at the next ``###`` or where "Core Concepts" ends.
    """
    matches = list(_HEADING.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        level = len(m.group(1))        # 2 for ##, 3 for ###
        start = m.start()
        end = len(text)
        for nxt in matches[i + 1:]:
            if len(nxt.group(1)) <= level:
                end = nxt.start()
                break
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
        out[slug] = text[start:end].rstrip() + "\n"
    return out


def get_content() -> str:
    """The whole reference, with each public module's names written in from its ``__all__`` and each catalogue's
    kinds of problem from the functions that declare them."""
    page = files("conductor.about").joinpath("llms.txt").read_text(encoding="utf-8")
    page = _NAMES.sub(lambda marker: _names(marker[1]), page)
    return _PROBLEMS.sub(lambda marker: _problems(getattr(import_module(marker[1]), marker[2])), page)


def list_sections() -> list[str]:
    """The slug of every ``##`` and ``###`` section, in the order the reference has them."""
    return list(_parse_sections(get_content()).keys())


def get_section(name: str) -> str | None:
    """One section by its slug, or the first whose slug contains ``name``; ``None`` when none does."""
    sections = _parse_sections(get_content())
    if name in sections:
        return sections[name]
    for slug, body in sections.items():
        if name.lower() in slug:
            return body
    return None


__all__: list[str] = []
