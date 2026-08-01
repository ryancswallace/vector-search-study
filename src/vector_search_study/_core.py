"""Core package behavior."""

from vector_search_study.exceptions import VectorSearchStudyError


def greet(name: str = "world") -> str:
    """Return a friendly greeting.

    Args:
        name: Name to include in the greeting.

    Returns:
        A greeting for `name`.

    Raises:
        VectorSearchStudyError: If `name` is blank.
    """
    cleaned = name.strip()
    if not cleaned:
        raise VectorSearchStudyError("name must not be blank")
    return f"Hello, {cleaned}."
