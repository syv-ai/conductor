"""Shared fixtures for the engine's tests."""

import pytest


@pytest.fixture
def registry():
    """An empty NodeRegistry, new for each test."""
    from conductor import NodeRegistry

    return NodeRegistry()
