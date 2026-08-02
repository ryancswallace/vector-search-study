"""Exact CPU PyTorch matrix-multiplication and top-k adapter."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vector_search_study._backend_utils import canonical_result, import_optional
from vector_search_study._base import BaseExactSearcher
from vector_search_study._validation import FloatMatrix
from vector_search_study.api import PreparedQueries, SearchObjective, SearchResult
from vector_search_study.exceptions import InvalidVectorDataError


class TorchTopKSearcher(BaseExactSearcher):
    """Exact CPU PyTorch matmul/topk search for every study objective."""

    _backend_name = "torch_cpu"

    def __init__(
        self,
        corpus: FloatMatrix,
        *,
        objective: SearchObjective | str = SearchObjective.NORMALIZED_COSINE,
    ) -> None:
        """Materialize the corpus tensor outside search timing."""
        super().__init__(corpus, objective=objective)
        self._torch: Any = import_optional("torch", extra="benchmark-backends")
        self._torch.set_num_threads(1)
        self._corpus_tensor: Any = self._torch.from_numpy(np.array(self._corpus, copy=True, order="C"))
        self._corpus_norms: Any | None = None
        if self.objective is SearchObjective.SQUARED_L2:
            self._corpus_norms = (self._corpus_tensor * self._corpus_tensor).sum(dim=1).unsqueeze(0)

    def prepare_queries(self, queries: FloatMatrix) -> PreparedQueries:
        """Validate queries and materialize their CPU tensor outside timing."""
        prepared = super().prepare_queries(queries)
        tensor = self._torch.from_numpy(np.array(prepared.values, copy=True, order="C"))
        return PreparedQueries(
            prepared.values,
            objective=self.objective,
            backend_name=self._backend_name,
            backend_payload=tensor,
        )

    def _search_prepared(self, queries: PreparedQueries, k: int) -> SearchResult:
        """Run matmul and topk on CPU, then canonicalize tied candidates."""
        if queries.backend_name != self._backend_name or queries.backend_payload is None:
            raise InvalidVectorDataError("queries must be prepared by this PyTorch searcher")
        query_tensor = cast(Any, queries.backend_payload)
        scores = self._torch.matmul(query_tensor, self._corpus_tensor.transpose(0, 1))
        if self.objective is SearchObjective.SQUARED_L2:
            query_norms = (query_tensor * query_tensor).sum(dim=1).unsqueeze(1)
            scores = -(query_norms + self._corpus_norms - 2.0 * scores).clamp_min(0.0)
        selected_scores, selected_indices = self._torch.topk(scores, k, dim=1, largest=True, sorted=False)
        return canonical_result(
            selected_indices.detach().cpu().numpy(),
            selected_scores.detach().cpu().numpy(),
        )
