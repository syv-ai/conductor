"""Shared fixtures for flow-engine tests."""

import pytest


@pytest.fixture
def registry():
    """Fresh NodeRegistry instance."""
    from flowengine.registry import NodeRegistry

    return NodeRegistry()
