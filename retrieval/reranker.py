"""
retrieval/reranker.py
---------------------
Reranks retrieved chunks using fast heuristics (no extra model needed).

Three signals:
  1. Query term coverage  — fraction of query tokens found in the chunk
  2. Length signal        — penalise very short / partial chunks
  3. Position bonus       — earlier chunks in a doc tend to be more definitional

Blended: 70% original retrieval score + 30% heuristic score.
"""

from __future__ import annotations
import re
from typing import List, Tuple
from ingestion.document_loader import Chunk


class HeuristicReranker:
    def rerank(self, query: str, results: List[Tuple[Chunk, float]], top_k: int = None) -> List[Tuple[Chunk, float]]:
        if not results:
            return results
        query_tokens = set(self._tok(query))
        scored = []
        for chunk, base_score in results:
            h = self._heuristic(query_tokens, chunk)
            scored.append((chunk, 0.7 * base_score + 0.3 * h))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if top_k else scored

    def _heuristic(self, query_tokens: set, chunk: Chunk) -> float:
        chunk_tokens = set(self._tok(chunk.text))
        coverage  = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        length_ok = min(len(chunk.text) / 200.0, 1.0)
        pos_bonus = 1.0 / (1.0 + 0.1 * chunk.chunk_id)
        return 0.5 * coverage + 0.3 * length_ok + 0.2 * pos_bonus

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
