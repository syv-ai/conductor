"""Conductor's reference, shipped inside the package.

The reference has two pages. ``llms.txt`` has the rules, and ``problems.md`` has every problem conductor can report.

Both pages are templates, filled in from the code when they are read. That way nothing on them is a copy of something
in the code. Two kinds of line get replaced:

- ``<!-- names: conductor.events -->`` becomes the names the module exports, in the order of its ``__all__``. Each
  name comes with the first paragraph of its docstring.
- ``<!-- problems: conductor.PROBLEMS -->`` becomes every kind of problem in that catalogue. Each entry is written
  from the function that declares the kind: its docstring, its example, its message and its ``ctx`` keys.

From a shell::

    python -m conductor.about                  # the rules
    python -m conductor.about problems         # every problem, with an example of each
    python -m conductor.about sections         # the section slugs of both pages
    python -m conductor.about one-or-many      # one section, by its slug or a part of it

A section's slug is its heading in lower case, with a hyphen in place of any run of other characters. So the
heading "One or many" has the slug ``one-or-many``.

``__main__.py`` is the command, and it calls the functions here. The module exports no names.
"""

from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files

from conductor._internal.problem import ProblemCatalogue, ProblemKind

_HEADING = re.compile(r"^(##+) (.+)$", re.MULTILINE)
_NAMES = re.compile(r"^<!-- names: ([\w.]+) -->$", re.MULTILINE)
_PROBLEMS = re.compile(r"^<!-- problems: ([\w.]+)\.(\w+) -->$", re.MULTILINE)
_ARGUMENT = re.compile(r"^    (\w+): ", re.MULTILINE)
_DOCSTRING_SECTION = re.compile(r"^(Args|Example):\n", re.MULTILINE)


def _names(module_name: str) -> str:
    """Lists the names a public module exports, as Markdown, in the order of its ``__all__``.

    For example, ``_names("conductor.widgets")`` starts with ``"- `Widget`: The base class of every widget."``.

    A class or a function gets the first paragraph of its docstring, joined onto one line. Its double backticks
    become Markdown's single ones. A paragraph that ends in ``::`` to introduce an example ends with a full stop
    instead. A constant or a type alias has no docstring of its own, so it is listed by its name alone.

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


@dataclass(frozen=True)
class _KindDocs:
    """The parts of a problem function's docstring that the problems page prints.

    Attributes:
        text: The paragraphs before ``Args:``, with Markdown's single backticks.
        arguments: Each parameter's line under ``Args:``, joined onto one line, by the parameter's name.
        example: The code under ``Example:``, without its indent. It is empty when there is none.
    """

    text: str
    arguments: dict[str, str]
    example: str


def _kind_docs(kind: ProblemKind[...]) -> _KindDocs:
    """Splits the docstring of the function that declares a kind of problem into its parts.

    The docstring has the kind's text first. Then comes ``Args:``, with a line for each parameter. Last comes
    ``Example:``, with code that makes the problem::

        A node is on a cycle of edges.

        A graph calls each node after the nodes upstream of it, ...

        Args:
            loc: The node.

        Example:
            problems({"nodes": [...], "edges": [...]})
    """
    text, *sections = _DOCSTRING_SECTION.split(inspect.cleandoc(kind.__doc__ or ""))
    by_name = dict(zip(sections[::2], sections[1::2]))
    pieces = _ARGUMENT.split(by_name.get("Args", ""))
    return _KindDocs(
        text=text.strip().replace("``", "`"),
        arguments={
            name: " ".join(line.split()).replace("``", "`")
            for name, line in zip(pieces[1::2], pieces[2::2])
        },
        example=inspect.cleandoc(by_name.get("Example", "")),
    )


def _problems(catalogue: ProblemCatalogue) -> str:
    """Writes every kind of problem in a catalogue as Markdown, for the problems page.

    Each section of the catalogue gets a ``##`` heading, and each kind a ``###`` section under it. A kind's section
    has, in order:

    - the text of its docstring;
    - its example;
    - its message, with a slot for each ``ctx`` value;
    - what its ``loc`` is, when it has one;
    - a table of its ``ctx`` keys, when it has any. Each row gives the key's type, whether it is always given, and
      its line under ``Args:``.

    Args:
        catalogue: The catalogue to write.

    Returns:
        The sections, in the order the kinds are declared.

    Raises:
        LookupError: A kind's docstring has no ``Args:`` line for one of its ``ctx`` keys.
    """
    out = []
    section = ""
    for kind in catalogue:
        if kind.section != section:
            section = kind.section
            out.append(f"## {section}\n\n{catalogue.sections[section]}")
        docs = _kind_docs(kind)
        rows = []
        for key, hint in typing.get_type_hints(kind.ctx_type).items():
            if key == "variant":
                rows.append(
                    f"| `variant` | `'{kind.variant}'` | always | Which of the code's sentences this is. |"
                )
                continue
            if key not in docs.arguments:
                raise LookupError(
                    f"{kind.code}: the docstring of its function has no Args line for the ctx key '{key}'"
                )
            given = "always" if key in kind.ctx_type.__required_keys__ else "when it applies"
            rows.append(
                f"| `{key}` | `{inspect.formatannotation(hint)}` | {given} | {docs.arguments[key]} |"
            )
        out.append(
            "\n\n".join(
                [
                    f"### `{kind.code}`" + (f", variant `{kind.variant}`" if kind.variant else ""),
                    docs.text,
                    *([f"```python\n{docs.example}\n```"] if docs.example else []),
                    f"> {kind.sentence}",
                    *([f"`loc`: {docs.arguments['loc']}"] if "loc" in docs.arguments else []),
                    *(
                        [
                            "\n".join(
                                ["| `ctx` | type | given | meaning |", "|---|---|---|---|", *rows]
                            )
                        ]
                        if rows
                        else []
                    ),
                ]
            )
        )
    return "\n\n".join(out)


def fill_placeholders(page: str) -> str:
    """Fills in a page's placeholders.

    Each ``names`` and ``problems`` placeholder line is replaced by the list or the entries it stands for.

    Args:
        page: The page's Markdown text.

    Returns:
        The page with every placeholder filled in.
    """
    page = _NAMES.sub(lambda line: _names(line[1]), page)
    return _PROBLEMS.sub(lambda line: _problems(getattr(import_module(line[1]), line[2])), page)


def _parse_sections(text: str) -> dict[str, str]:
    """Splits a page into its sections, by slug.

    A section starts at a heading of two or more ``#``. It runs to the next heading with as many ``#`` or fewer. So
    a ``###`` section inside a ``##`` section ends at the next ``###`` heading, or where the ``##`` section ends.

    Two headings can have the same slug, such as a code that is both a problem and a declaration fault. Then only
    the first is kept.
    """
    matches = list(_HEADING.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        level = len(m.group(1))
        start = m.start()
        end = len(text)
        for nxt in matches[i + 1 :]:
            if len(nxt.group(1)) <= level:
                end = nxt.start()
                break
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
        out.setdefault(
            slug, text[start:end].rstrip() + "\n"
        )  # a code with several kinds, or in two catalogues, is found by its first heading
    return out


def get_content() -> str:
    """Returns the rules page, ``llms.txt``, with each public module's names filled in."""
    return fill_placeholders(
        files("conductor.about").joinpath("llms.txt").read_text(encoding="utf-8")
    )


def problems_page() -> str:
    """Returns the problems page, ``problems.md``, with every kind of problem filled in.

    That is the problems of a graph and a run, the faults in a node's declaration, and the problems
    ``check_registry`` returns.
    """
    return fill_placeholders(
        files("conductor.about").joinpath("problems.md").read_text(encoding="utf-8")
    )


def _both_pages() -> list[dict[str, str]]:
    """Returns the sections of the rules page, then those of the problems page."""
    return [_parse_sections(get_content()), _parse_sections(problems_page())]


def list_sections() -> list[str]:
    """Returns the slug of every section of both pages, the rules page first."""
    return [slug for sections in _both_pages() for slug in sections]


def get_section(name: str) -> str | None:
    """Finds one section of the reference, by its slug or a part of it.

    A section whose slug is exactly ``name`` comes first. Otherwise it is the first section whose slug contains
    ``name``. The rules page is searched before the problems page. So ``asking`` finds the rules' section on asking a
    person, and ``cycle`` finds the problem.

    For example, ``get_section("one-or")`` returns the section "One or many", heading included.

    Returns:
        The section's text, or ``None`` when no slug matches.
    """
    pages = _both_pages()
    for sections in pages:
        if name in sections:
            return sections[name]
    for sections in pages:
        for slug, body in sections.items():
            if name.lower() in slug:
                return body
    return None


__all__: list[str] = []
