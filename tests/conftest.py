"""Shared fixtures for the engine's tests."""

import pytest


@pytest.fixture
def registry():
    """Fresh NodeRegistry instance."""
    from conductor.registry import NodeRegistry

    return NodeRegistry()
