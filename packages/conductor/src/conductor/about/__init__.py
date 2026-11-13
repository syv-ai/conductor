"""Conductor's reference, shipped in the package: the rules on one page, ``llms.txt``, and every problem on another, ``problems.md``.

Both pages are templates. A line ``<!-- names: conductor.events -->`` is replaced by the names the module exports, from
its ``__all__``, each with the first paragraph of its docstring. A line ``<!-- problems: conductor.PROBLEMS -->`` is
replaced by every kind of problem the catalogue declares, each from the function that declares it: its docstring, its
example, its sentence and its ``ctx`` keys. So nothing on either page is a copy of something in the code.

From a shell::

    python -m conductor.about                  # the rules
    python -m conductor.about problems         # every problem, with an example of each
    python -m conductor.about sections         # the section slugs of both pages
    python -m conductor.about one-or-many      # one section, by its slug or a part of it

A section's slug is its heading in lower case with a hyphen for each run of other characters: the heading "One or
many" has the slug ``one-or-many``. ``__main__.py`` calls the functions here; the module exports no names.
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


@dataclass(frozen=True)
class _KindDocs:
    """The docstring of the function that declares a kind of problem, in the parts the problems page prints.

    Attributes:
        text: The paragraphs before ``Args:``, in Markdown's single backticks.
        arguments: The line under ``Args:`` for each parameter, by the parameter's name, on one line.
        example: The code under ``Example:``, without its indent; empty when there is none.
    """

    text: str
    arguments: dict[str, str]
    example: str


def _kind_docs(kind: ProblemKind[...]) -> _KindDocs:
    """Splits the docstring of the function that declares ``kind``.

    The docstring is the kind's text, then ``Args:`` with a line for each parameter, then ``Example:`` with code that
    makes the problem::

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
        arguments={name: " ".join(line.split()).replace("``", "`") for name, line in zip(pieces[1::2], pieces[2::2])},
        example=inspect.cleandoc(by_name.get("Example", "")),
    )


def _problems(catalogue: ProblemCatalogue) -> str:
    """Every kind of problem a catalogue declares, a ``###`` section each, under a ``##`` heading for each section of
    the catalogue.

    A kind's section has its docstring's text, its example, its sentence with a slot for each ``ctx`` value, what
    its ``*loc`` is when it has one, and, when it has any, a table of its ``ctx`` keys: the type, whether the key is always given,
    and the key's line under ``Args:``.

    Args:
        catalogue: The catalogue.

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
                rows.append(f"| `variant` | `'{kind.variant}'` | always | Which of the code's sentences this is. |")
                continue
            if key not in docs.arguments:
                raise LookupError(f"{kind.code}: the docstring of its function has no Args line for the ctx key '{key}'")
            given = "always" if key in kind.ctx_type.__required_keys__ else "when it applies"
            rows.append(f"| `{key}` | `{inspect.formatannotation(hint)}` | {given} | {docs.arguments[key]} |")
        out.append("\n\n".join([
            f"### `{kind.code}`" + (f", variant `{kind.variant}`" if kind.variant else ""),
            docs.text,
            *([f"```python\n{docs.example}\n```"] if docs.example else []),
            f"> {kind.sentence}",
            *([f"`loc`: {docs.arguments['loc']}"] if "loc" in docs.arguments else []),
            *(["\n".join(["| `ctx` | type | given | meaning |", "|---|---|---|---|", *rows])] if rows else []),
        ]))
    return "\n\n".join(out)


def fill_placeholders(page: str) -> str:
    """``page`` with each ``names`` and ``problems`` placeholder line replaced by what it stands for."""
    page = _NAMES.sub(lambda line: _names(line[1]), page)
    return _PROBLEMS.sub(lambda line: _problems(getattr(import_module(line[1]), line[2])), page)


def _parse_sections(text: str) -> dict[str, str]:
    """Each section by its slug: every heading of two or more ``#``, with the text under it.

    A section runs from its heading to the next heading with as many ``#`` or fewer. So a ``###`` section inside a
    ``##`` section ends at the next ``###`` heading or where the ``##`` section ends. Of two headings with one slug
    (a code of both a problem and a declaration fault would have two), the first is kept.
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
        out.setdefault(slug, text[start:end].rstrip() + "\n")  # a code with several kinds, or in two catalogues, is found by its first
    return out


def get_content() -> str:
    """The rules, ``llms.txt``, with each public module's names written in."""
    return fill_placeholders(files("conductor.about").joinpath("llms.txt").read_text(encoding="utf-8"))


def problems_page() -> str:
    """The problems page, ``problems.md``, with every kind of problem of the core, of a declaration and of ``check_registry``
    written in."""
    return fill_placeholders(files("conductor.about").joinpath("problems.md").read_text(encoding="utf-8"))


def _both_pages() -> list[dict[str, str]]:
    """The sections of the rules, then those of the problems page."""
    return [_parse_sections(get_content()), _parse_sections(problems_page())]


def list_sections() -> list[str]:
    """The slug of every ``##`` and ``###`` section, the rules' first."""
    return [slug for sections in _both_pages() for slug in sections]


def get_section(name: str) -> str | None:
    """One section by its slug, or the first whose slug contains ``name``; ``None`` when none does.

    The rules are searched before the problems page, so ``asking`` finds the rules' section and ``cycle`` the problem.
    ``get_section("one-or")`` returns the section "One or many", heading included.
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
