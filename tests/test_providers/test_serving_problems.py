"""The problems the router writes: each kind's sentence fills from its ``ctx``, and the page on serving runs lists
every kind."""

import re
from pathlib import Path

from conductor.about import fill_placeholders
from conductor_providers.fastapi import SERVING_PROBLEMS

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"


def test_every_serving_kinds_sentence_fills_from_its_ctx_keys_and_leaves_no_brace() -> None:
    for kind in SERVING_PROBLEMS:
        filled = kind.sentence.format(**{key: "x" for key in kind.ctx_type.__annotations__})

        assert "{" not in filled and "}" not in filled, kind.code


def test_the_page_on_serving_runs_writes_in_every_serving_kind() -> None:
    """The docs build fills the page's marker from the router's catalogue, each kind under a heading of its code."""
    page = PAGE.read_text()
    filled = fill_placeholders(page)

    assert re.findall(r"<!-- problems: ([\w.]+) -->", page) == [
        "conductor_providers.fastapi.SERVING_PROBLEMS"
    ]
    assert [kind.code for kind in SERVING_PROBLEMS if f"### `{kind.code}`" not in filled] == []
