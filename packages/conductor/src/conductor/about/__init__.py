"""Conductor's reference: one Markdown page, shipped in the package as ``conductor/about/llms.txt``.

The page is written for a developer or a coding assistant that uses conductor, and says how the library behaves.
Two kinds of list in it are not written by hand but made from the code each time the page is read, so they cannot
go out of date:

- A line ``<!-- names: conductor.events -->`` is replaced by the names the module ``conductor.events`` exports,
  in the order of its ``__all__``, each with the first paragraph of its docstring.
- A line ``<!-- problems: conductor.PROBLEMS -->`` is replaced by every kind of problem in the catalogue
  ``conductor.PROBLEMS``: its code, its message and the keys of its ``ctx``, read from the function that
  declares the kind.

From a shell::

    python -m conductor.about                  # the whole reference
    python -m conductor.about sections         # the slug of every section
    python -m conductor.about one-or-many      # one section, by its slug or a part of it

A section's slug is its heading in lower case with a hyphen for each run of other characters: the heading "One or
many" has the slug ``one-or-many``. ``__main__.py`` calls the functions here; the module exports no names.
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

    ``_names("conductor.widgets")`` starts ``"- `Widget`: The base class of every widget. ..."``. A class or a
    function is followed by the first paragraph of its docstring on one line, with reStructuredText's double
    backticks made Markdown's single ones; a paragraph that ends ``::``, to introduce an example, ends with a full
    stop instead. A constant or a type alias has no docstring of its own, so it is listed by its name alone.

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
    """The kinds of problem in a catalogue, each as a ``####`` section headed by its code.

    A kind is made from a function and has the function's docstring, and its section is made from it: the first paragraph of its
    docstring; its message, with a ``{key}`` for each ``ctx`` value; what its ``loc`` is, when the function takes
    a ``*loc`` argument and its docstring says what it is under ``Args:``; and a table of its ``ctx`` keys, with
    each key's type, whether the key is always given, and the key's line under ``Args:``. The key ``variant``,
    which says which of a code's messages this is, needs no line.

    Args:
        catalogue: The catalogue.

    Returns:
        The sections, in the order the kinds are declared.

    Raises:
        LookupError: A kind's docstring has no line under ``Args:`` for one of its ``ctx`` keys.
    """
    sections = []
    for kind in catalogue:
        summary, _, arguments = inspect.cleandoc(kind.__doc__ or "").partition("\nArgs:\n")
        pieces = _ARGUMENT.split(arguments)
        explained = {name: " ".join(text.split()).replace("``", "`") for name, text in zip(pieces[1::2], pieces[2::2])}
        rows = []
        for key, hint in typing.get_type_hints(kind.ctx_type).items():
            if key == "variant":
                rows.append(f"| `variant` | `'{kind.variant}'` | always | Which of the code's sentences this is. |")
                continue
            if key not in explained:
                raise LookupError(f"{kind.code}: the docstring of its function has no Args line for the ctx key '{key}'")
            given = "always" if key in kind.ctx_type.__required_keys__ else "when it applies"
            rows.append(f"| `{key}` | `{inspect.formatannotation(hint)}` | {given} | {explained[key]} |")
        sections.append("\n\n".join([
            f"#### `{kind.code}`" + (f", variant `{kind.variant}`" if kind.variant else ""),
            " ".join(summary.split("\n\n")[0].split()).replace("``", "`"),
            f"> {kind.sentence}",
            *([f"`loc`: {explained['loc']}"] if "loc" in explained else []),
            "\n".join(["| `ctx` | type | given | holds |", "|---|---|---|---|", *rows]),
        ]))
    return "\n\n".join(sections)


def _parse_sections(text: str) -> dict[str, str]:
    """Each section by its slug: every heading of two or more ``#``, with the text under it.

    A section runs from its heading to the next heading with as many ``#`` or fewer. So a ``###`` section inside a
    ``##`` section ends at the next ``###`` heading or where the ``##`` section ends.
    """
    matches = list(_HEADING.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        level = len(m.group(1))
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
    """The whole reference as Markdown, with every ``<!-- names: ... -->`` and ``<!-- problems: ... -->`` line
    replaced by the list it names."""
    page = files("conductor.about").joinpath("llms.txt").read_text(encoding="utf-8")
    page = _NAMES.sub(lambda marker: _names(marker[1]), page)
    return _PROBLEMS.sub(lambda marker: _problems(getattr(import_module(marker[1]), marker[2])), page)


def list_sections() -> list[str]:
    """The slug of every ``##`` and ``###`` section, in the order the reference has them."""
    return list(_parse_sections(get_content()).keys())


def get_section(name: str) -> str | None:
    """One section by its slug, or else the first section whose slug contains ``name``; ``None`` when none does.

    ``get_section("one-or")`` returns the section "One or many", heading included."""
    sections = _parse_sections(get_content())
    if name in sections:
        return sections[name]
    for slug, body in sections.items():
        if name.lower() in slug:
            return body
    return None


__all__: list[str] = []
