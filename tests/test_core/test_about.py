"""Tests ``conductor.about``, the reference a reader or an agent prints with ``python -m conductor.about``.

The reference is ``llms.txt`` and the problems page. These tests check that its sections can be found by name, that
the command line prints what it should, and that the lists of public names and problems written into it are right.
"""

from __future__ import annotations

import builtins
import importlib
import re
import subprocess
import sys
import typing
from types import ModuleType

import pytest
from conductor import ProblemCatalogue, about
from pytest import MonkeyPatch

from tests.test_core.test_public_modules import PUBLIC_MODULES


def test_get_content_returns_non_empty_text():
    text = about.get_content()
    assert isinstance(text, str)
    assert len(text) > 1000  # llms.txt is long
    assert "# Conductor" in text


def test_list_sections_includes_core_stanzas():
    slugs = about.list_sections()
    # Top-level sections, under a ##.
    assert "quick-start" in slugs
    assert "core-concepts" in slugs
    assert "api-reference" in slugs
    # Sections under a ### can be asked for by name too.
    assert "retries-and-timeouts" in slugs
    assert "asking-a-person" in slugs


def test_get_section_exact_slug():
    body = about.get_section("retries-and-timeouts")
    assert body is not None
    assert "### Retries and Timeouts" in body
    assert "Policy" in body


def test_get_section_substring_match_resolves_to_full_slug():
    body = about.get_section("asking")
    assert body is not None
    assert "### Asking a person" in body


def test_get_section_unknown_returns_none():
    assert about.get_section("definitely-not-a-section") is None


def test_sections_do_not_bleed_past_same_or_higher_heading():
    retries_body = about.get_section("retries-and-timeouts")
    assert retries_body is not None
    # "Retries and Timeouts" is a ### section, and "Errors" is the ### after it.
    # A section ends at the next heading of the same level or higher.
    assert "### Errors" not in retries_body


def test_cli_no_args_prints_full_content():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "# Conductor" in result.stdout
    assert "## Quick Start" in result.stdout


def test_cli_sections_lists_slugs():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "sections"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    assert "quick-start" in lines
    assert "asking-a-person" in lines


def test_cli_section_filter_emits_only_that_section():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "retries"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "### Retries and Timeouts" in result.stdout
    assert "## Quick Start" not in result.stdout


def test_cli_unknown_section_nonzero_exit_with_help():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "nope-not-real"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "No section matching" in result.stderr
    # It lists the sections there are, so the caller can pick the right one.
    assert "quick-start" in result.stderr


def test_cli_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "python -m conductor.about" in result.stdout


#: The public modules that export names. ``conductor.about`` exports none, so the reference has no section for it.
NAMED_MODULES = [name for name in PUBLIC_MODULES if importlib.import_module(name).__all__]


@pytest.mark.parametrize("module_name", NAMED_MODULES)
def test_a_public_modules_section_lists_every_name_it_exports_in_the_order_it_exports_them(
    module_name,
):
    section = about.get_section(module_name.replace(".", "-"))
    assert section is not None

    listed = re.findall(r"^- `(\w+)`", section, flags=re.MULTILINE)

    assert listed == importlib.import_module(module_name).__all__


def _names_written(module: ModuleType, monkeypatch: MonkeyPatch) -> str:
    """Returns the list the reference writes for ``module``'s names.

    The reference marks where a module's names go with ``<!-- names: ... -->``. This fills in that marker for
    ``module``, the same way the reference fills in its own.
    """
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return about.fill_placeholders(f"<!-- names: {module.__name__} -->")


def test_a_listed_class_or_function_is_followed_by_the_first_paragraph_of_its_docstring(
    monkeypatch: MonkeyPatch,
):
    class Room:
        """A room in a house,
        with ``doors`` and windows.

        Rooms are built by hand.
        """

    def open_door() -> None:
        """Opens a door."""

    module = ModuleType("rooms")
    module.Room = Room  # type: ignore[attr-defined]
    module.open_door = open_door  # type: ignore[attr-defined]
    module.__all__ = ["Room", "open_door"]  # type: ignore[attr-defined]

    # A paragraph over several lines becomes one line, and its ``code`` becomes Markdown's `code`.
    assert _names_written(module, monkeypatch) == (
        "- `Room`: A room in a house, with `doors` and windows.\n- `open_door`: Opens a door."
    )


def test_a_summary_that_ends_by_introducing_an_example_ends_with_a_full_stop(
    monkeypatch: MonkeyPatch,
):
    def build_room() -> None:
        """Builds a room, like this::

        build_room()
        """

    module = ModuleType("rooms")
    module.build_room = build_room  # type: ignore[attr-defined]
    module.__all__ = ["build_room"]  # type: ignore[attr-defined]

    assert _names_written(module, monkeypatch) == "- `build_room`: Builds a room, like this."


def test_a_constant_or_a_type_alias_is_listed_by_its_name_alone():
    section = about.get_section("conductor")
    assert section is not None
    assert "- `SKIPPED`\n" in section
    assert "- `RunStatus`\n" in section


def test_the_cli_prints_a_modules_names():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "conductor-testing"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "- `check_registry`: " in result.stdout


def _library_names() -> set[str]:
    """Returns every name a reader can import: the public modules' names, ``typing``'s and the builtins."""
    names = set(dir(typing)) | set(dir(builtins))
    for module_name in PUBLIC_MODULES:
        names.update(importlib.import_module(module_name).__all__)
    return names


def test_every_name_the_reference_puts_in_backticks_exists():
    """Every name the reference writes in backticks can be imported.

    An agent copies these names into code, so a stale one breaks its next task. The test checks two kinds: every
    name that starts with a capital letter, and every dotted path that starts with ``conductor``. A one-letter name
    like ``X`` stands for any class, so it is left out.
    """
    known = _library_names()
    missing: set[str] = set()
    for span in re.findall(r"`([^`\n]+)`", about.get_content()):
        if re.match(r"(GET|POST) ", span):
            continue
        code = re.sub(r'"[^"]*"', "", span)
        for dotted in re.findall(r"\bconductor(?:\.\w+)+", code):
            module, _, attribute = dotted.rpartition(".")
            try:
                importlib.import_module(dotted)
            except ModuleNotFoundError:
                if not hasattr(importlib.import_module(module), attribute):
                    missing.add(dotted)
        for name in re.findall(r"(?<![.\w])[A-Z][A-Za-z0-9_]+", code):
            if name not in known:
                missing.add(name)

    assert missing == set()


def test_the_problems_page_has_every_kind_of_problem_with_its_sentence_its_ctx_keys_and_its_example():
    from conductor import PROBLEMS
    from conductor.testing.problems import REGISTRY_CHECK_PROBLEMS

    page = about.problems_page()

    assert "<!-- problems:" not in page and "<!-- problems:" not in about.get_content()
    for kind in [*PROBLEMS, *REGISTRY_CHECK_PROBLEMS]:
        assert f"### `{kind.code}`" in page and f"> {kind.sentence}" in page
    for section in PROBLEMS.sections:
        assert f"\n## {section}\n" in page
    entry = about.get_section("context-missing")
    assert entry is not None
    assert (
        "> The node takes the run's context as '{parameter}', of class {needs}, but the call gave no context. Pass context= on every call that starts or resumes the run."
        in entry
    )
    assert "`loc`: The node's path, then its `Context` parameter." in entry
    assert "| `needs` | `str` | always | The name of the class inside `Context[...]`. |" in entry
    assert (
        "```python\nrefused(lambda: Graph.compile(make_graph(who), registry).run())\n```" in entry
    )


def test_the_cli_prints_the_problems_page():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "problems"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == about.problems_page()


def _written(catalogue: ProblemCatalogue, monkeypatch: MonkeyPatch) -> str:
    """Returns the problems page's entries for ``catalogue``.

    The problems page marks where a catalogue goes with ``<!-- problems: ... -->``. This fills in that marker for
    ``catalogue``, the same way the page fills in its own.
    """
    module = ModuleType("rooms")
    module.CATALOGUE = catalogue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return about.fill_placeholders("<!-- problems: rooms.CATALOGUE -->")


def test_a_key_that_may_be_left_out_and_a_variant_are_said_in_a_kinds_entry(
    monkeypatch: MonkeyPatch,
):
    catalogue = ProblemCatalogue()

    @catalogue.problem()
    def room_closed(*loc: str | int, until: str, opener: str | None = None) -> str:
        """A room is closed.

        Come back later.

        Args:
            loc: The room and the door.
            until: When it opens,
                on the wall clock.
            opener: Who opens it, when someone is named.

        Example:
            room_closed("hall", "front", until="noon")
        """
        return f"The room is closed until {until}."

    @catalogue.problem("room_closed", variant="holiday")
    def room_closed_for_a_holiday(*loc: str | int, until: str) -> str:
        """A room is closed for a holiday.

        Args:
            loc: The room and the door.
            until: When it opens.
        """
        return f"The room is closed over the holiday, until {until}."

    entries = _written(catalogue, monkeypatch)

    assert (
        "### `room_closed`\n\nA room is closed.\n\nCome back later.\n\n"
        '```python\nroom_closed("hall", "front", until="noon")\n```\n\n'
        "> The room is closed until {until}.\n\n`loc`: The room and the door."
    ) in entries
    assert "| `until` | `str` | always | When it opens, on the wall clock. |" in entries
    assert (
        "| `opener` | `str` | when it applies | Who opens it, when someone is named. |" in entries
    )
    assert "### `room_closed`, variant `holiday`" in entries
    assert (
        "| `variant` | `'holiday'` | always | Which of the code's sentences this is. |" in entries
    )


def test_a_kind_with_a_key_its_docstring_does_not_explain_fails_when_the_reference_is_read(
    monkeypatch: MonkeyPatch,
):
    catalogue = ProblemCatalogue()

    @catalogue.problem()
    def room_closed(*loc: str | int, until: str) -> str:
        """A room is closed.

        Args:
            loc: The room and the door.
        """
        return f"The room is closed until {until}."

    with pytest.raises(LookupError, match="room_closed.*'until'"):
        _written(catalogue, monkeypatch)
