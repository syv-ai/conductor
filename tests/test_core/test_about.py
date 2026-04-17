"""Tests for the conductor.about runnable reference module."""

from __future__ import annotations

import subprocess
import sys

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
    assert "retry" in slugs
    assert "shared-references-produce-consume" in slugs


def test_get_section_exact_slug():
    body = about.get_section("retry")
    assert body is not None
    assert "### Retry" in body
    assert "RetryConfig" in body


def test_get_section_substring_match_resolves_to_full_slug():
    body = about.get_section("shared")
    assert body is not None
    assert "Shared References" in body


def test_get_section_unknown_returns_none():
    assert about.get_section("definitely-not-a-section") is None


def test_sections_do_not_bleed_past_same_or_higher_heading():
    retry_body = about.get_section("retry")
    assert retry_body is not None
    # Retry is H3 inside Core Concepts, followed by H3 Data Flow. It must
    # end at the next heading of the same or higher level, so the Data
    # Flow section must not be inside Retry.
    assert "### Data Flow" not in retry_body


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
    assert "shared-references-produce-consume" in lines


def test_cli_section_filter_emits_only_that_section():
    result = subprocess.run(
        [sys.executable, "-m", "conductor.about", "retry"],
        capture_output=True, text=True, check=True,
    )
    assert "### Retry" in result.stdout
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
