"""Tests for the conductor.about runnable reference module."""

from __future__ import annotations

import builtins
import importlib
import re
import subprocess
import sys
import typing

import pytest
from conductor import about


def test_get_content_returns_non_empty_text():
    text = about.get_content()
    assert isinstance(text, str)
    assert len(text) > 1000   # llms.txt is substantial
    assert "# Conductor" in text


def test_list_sections_includes_core_stanzas():
    slugs = about.list_sections()
    # H2 — high-level stanzas
    assert "quick-start" in slugs
    assert "core-concepts" in slugs
    assert "api-reference" in slugs
    # H3 — nested concept sections must also be addressable directly
    assert "retries-and-timeouts" in slugs
    assert "legs-and-asking" in slugs


def test_get_section_exact_slug():
    body = about.get_section("retries-and-timeouts")
    assert body is not None
    assert "### Retries and Timeouts" in body
    assert "Policy" in body


def test_get_section_substring_match_resolves_to_full_slug():
    body = about.get_section("asking")
    assert body is not None
    assert "### Legs and Asking" in body


def test_get_section_unknown_returns_none():
    assert about.get_section("definitely-not-a-section") is None


def test_sections_do_not_bleed_past_same_or_higher_heading():
    retries_body = about.get_section("retries-and-timeouts")
    assert retries_body is not None
    # Retries and Timeouts is H3 inside Core Concepts, followed by H3
    # Errors. It must end at the next heading of the same or higher level.
    assert "### Errors" not in retries_body


def test_cli_no_args_prints_full_content():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about"],
        capture_output=True, text=True, check=True,
    )
    assert "# Conductor" in result.stdout
    assert "## Quick Start" in result.stdout


def test_cli_sections_lists_slugs():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "sections"],
        capture_output=True, text=True, check=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    assert "quick-start" in lines
    assert "legs-and-asking" in lines


def test_cli_section_filter_emits_only_that_section():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "retries"],
        capture_output=True, text=True, check=True,
    )
    assert "### Retries and Timeouts" in result.stdout
    assert "## Quick Start" not in result.stdout


def test_cli_unknown_section_nonzero_exit_with_help():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "nope-not-real"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "No section matching" in result.stderr
    # Must list what's available so the caller can correct themselves
    assert "quick-start" in result.stderr


def test_cli_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "python -m conductor.about" in result.stdout


#: The public modules that list names; ``conductor.about`` has none, so it has no section of them.
PUBLIC_MODULES = ("conductor", "conductor.widgets", "conductor.events", "conductor.testing")


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_a_public_modules_section_lists_every_name_it_exports_in_the_order_it_exports_them(module_name):
    section = about.get_section(module_name.replace(".", "-"))
    assert section is not None

    listed = re.findall(r"^- `(\w+)`", section, flags=re.MULTILINE)

    assert listed == importlib.import_module(module_name).__all__


def test_a_listed_class_or_function_is_followed_by_the_first_paragraph_of_its_docstring():
    events = about.get_section("conductor-events")
    assert events is not None
    assert "- `NodeComplete`: Every unit of a node is done.\n" in events
    assert (  # a paragraph over several lines is one line, and its code is in Markdown's single backticks
        "- `NodeProgress`: How many units of a node that runs per item are done, at most four times a second; "
        "`total` `None` until every Series above it is made.\n"
    ) in events


def test_a_summary_that_ends_by_introducing_an_example_ends_with_a_full_stop():
    section = about.get_section("conductor")
    assert section is not None
    assert "- `make_graph`: A graph record written from Python, each node at its version.\n" in section


def test_a_constant_or_a_type_alias_is_listed_by_its_name_alone():
    section = about.get_section("conductor")
    assert section is not None
    assert "- `SKIPPED`\n" in section
    assert "- `RunState`\n" in section


def test_the_cli_prints_a_modules_names():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "conductor-testing"],
        capture_output=True, text=True, check=True,
    )
    assert "- `check_registry`: " in result.stdout


def _library_names() -> set[str]:
    """Every name a reader can import: what the doors list, typing's and the builtins."""
    names = set(dir(typing)) | set(dir(builtins))
    for door in ("conductor", "conductor.widgets", "conductor.events", "conductor.testing"):
        names.update(importlib.import_module(door).__all__)
    return names


def test_every_name_the_reference_puts_in_backticks_exists():
    """A stale name here is broken code on an agent's next task. Every
    class-like name in backticks is one a reader can import (a one-letter
    placeholder like ``X`` aside), and every dotted ``conductor…`` path
    imports."""
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


def test_the_reference_has_every_kind_of_problem_with_its_sentence_and_its_ctx_keys():
    from conductor import PROBLEMS

    text = about.get_content()

    assert "<!-- problems:" not in text
    for kind in PROBLEMS:
        assert f"#### `{kind.code}`" in text and f"> {kind.sentence}" in text
    entry = about.get_section("from-run-missing")
    assert entry is not None
    assert "> '{node_id}' needs a {needs} for '{parameter}', and from_run gives none (it gives {given}); pass from_run with a value for {needs}." in entry
    assert "| `node_id` | `str` | always | The node. |" in entry
    assert "| `given` | `list[str]` | always | The names of the classes `from_run` has values for. |" in entry


def test_a_key_that_may_be_left_out_and_a_variant_are_said_in_a_kinds_entry():
    from conductor import ProblemCatalogue
    from conductor.about import _problems

    catalogue = ProblemCatalogue()

    @catalogue.problem()
    def room_closed(*loc: str | int, until: str, opener: str | None = None) -> str:
        """A room is closed.

        Args:
            loc: The room and the door.
            until: When it opens,
                on the wall clock.
            opener: Who opens it, when someone is named.
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

    entries = _problems(catalogue)

    assert "#### `room_closed`\n\nA room is closed.\n\n> The room is closed until {until}.\n\n`loc`: The room and the door." in entries
    assert "| `until` | `str` | always | When it opens, on the wall clock. |" in entries
    assert "| `opener` | `str` | when it applies | Who opens it, when someone is named. |" in entries
    assert "#### `room_closed`, variant `holiday`" in entries
    assert "| `variant` | `'holiday'` | always | Which of the code's sentences this is. |" in entries


def test_a_kind_with_a_key_its_docstring_does_not_explain_fails_when_the_reference_is_read():
    from conductor import ProblemCatalogue
    from conductor.about import _problems

    catalogue = ProblemCatalogue()

    @catalogue.problem()
    def room_closed(*loc: str | int, until: str) -> str:
        """A room is closed.

        Args:
            loc: The room and the door.
        """
        return f"The room is closed until {until}."

    with pytest.raises(LookupError, match="room_closed.*'until'"):
        _problems(catalogue)
