"""Runs mypy on ``typed_setups.py``.

The two setups there must type-check, and mypy must still catch each mistake marked there.
"""

from __future__ import annotations

from pathlib import Path

from mypy import api


def test_the_two_setups_type_check_and_the_marked_mistakes_are_caught() -> None:
    report, errors, status = api.run(
        [str(Path(__file__).with_name("typed_setups.py")), "--strict", "--warn-unused-ignores"]
    )

    assert status == 0, report + errors
