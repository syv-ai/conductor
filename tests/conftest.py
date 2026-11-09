"""Shared fixtures for the engine's tests."""

import pytest


@pytest.fixture
def registry():
    """Fresh NodeRegistry instance."""
    from conductor import NodeRegistry

    return NodeRegistry()
