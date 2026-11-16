"""Fixtures that every test under ``tests/`` can use."""

import pytest


@pytest.fixture
def registry():
    """A new, empty ``NodeRegistry`` for each test."""
    from conductor import NodeRegistry

    return NodeRegistry()
