"""
retrieval/hybrid_retriever.py
------------------------------
Combines dense (FAISS) + sparse (BM25) retrieval via weighted score fusion.

alpha = 1.0  →  pure vector search
alpha = 0.0  →  pure BM25
alpha = 0.7  →  default (70% vector, 30% BM25)

The DecisionLayer adjusts alpha at runtime based on query type.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from ingestion.document_loader import Chunk
from retrieval.vector_store import VectorStore
from retrieval.keyword_search import KeywordSearch


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, keyword_search: KeywordSearch, alpha: float = 0.7):
        self.vs = vector_store
        self.ks = keyword_search
        self.alpha = alpha

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int,
        alpha: float = None,
    ) -> List[Tuple[Chunk, float]]:
        a = alpha if alpha is not None else self.alpha
        fetch_k = min(top_k * 2, max(len(self.vs), 1))

        vec_results = self.vs.search(query_vec, fetch_k)
        kw_results  = self.ks.search(query, fetch_k)

        # Weighted score fusion — deduplicate by chunk identity
        scores: Dict[int, Tuple[Chunk, float]] = {}
        for chunk, score in vec_results:
            scores[id(chunk)] = (chunk, a * score)
        for chunk, score in kw_results:
            key = id(chunk)
            if key in scores:
                c, s = scores[key]
                scores[key] = (c, s + (1 - a) * score)
            else:
                scores[key] = (chunk, (1 - a) * score)

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
