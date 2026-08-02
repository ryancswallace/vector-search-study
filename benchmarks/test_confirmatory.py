"""Environment-selected common-identity confirmatory benchmark matrix."""

from __future__ import annotations

import os

import pytest

from benchmarks._confirmatory_harness import install_confirmatory_test

if "VECTOR_SEARCH_CONFIRMATORY_EXPERIMENT" not in os.environ:
    pytest.skip("confirmatory collection requires an explicit experiment", allow_module_level=True)

install_confirmatory_test(globals())
