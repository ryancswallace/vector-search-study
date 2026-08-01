"""Tests for core package behavior."""

import pytest

from vector_search_study import VectorSearchStudyError, greet


def test_greet_uses_default_name() -> None:
    """The default greeting is stable."""
    assert greet() == "Hello, world."


def test_greet_strips_names() -> None:
    """Names are stripped before rendering."""
    assert greet("  Python  ") == "Hello, Python."


def test_greet_rejects_blank_names() -> None:
    """Blank names are rejected."""
    with pytest.raises(VectorSearchStudyError, match="name must not be blank"):
        _ = greet(" ")
