"""Tests for the conductor.about runnable reference module."""

from __future__ import annotations

import builtins
import importlib
import pkgutil
import re
import subprocess
import sys
import typing

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


def _library_names() -> set[str]:
    """Every name any module of the three packages defines or imports, plus typing's and the builtins."""
    names = set(dir(typing)) | set(dir(builtins))
    for package_name in ("conductor", "conductor_nodes", "conductor_providers"):
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, f"{package_name}."):
            if info.name.endswith("__main__"):
                continue
            names.update(dir(importlib.import_module(info.name)))
    return names


def test_every_name_the_reference_puts_in_backticks_exists():
    """A stale name here is broken code on an agent's next task. Every
    class-like name in backticks is defined somewhere in the library (a
    one-letter placeholder like ``X`` aside), and every dotted
    ``conductor…`` path imports."""
    known = _library_names()
    missing: set[str] = set()
    for span in re.findall(r"`([^`\n]+)`", about.get_content()):
        if re.match(r"(GET|POST) ", span):
            continue
        code = re.sub(r'"[^"]*"', "", span)
        for dotted in re.findall(r"\bconductor(?:_nodes|_providers)?(?:\.\w+)+", code):
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
