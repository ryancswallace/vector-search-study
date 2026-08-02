"""Package-specific exceptions."""


class VectorSearchStudyError(ValueError):
    """Raised when Vector Search Study cannot complete an operation."""


class InvalidVectorDataError(VectorSearchStudyError):
    """Raised when a corpus or query matrix violates the vector contract."""


class InvalidSearchParameterError(VectorSearchStudyError):
    """Raised when a search parameter is outside its supported range."""
