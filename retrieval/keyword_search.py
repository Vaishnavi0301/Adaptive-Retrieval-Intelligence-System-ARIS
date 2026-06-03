"""
retrieval/keyword_search.py
---------------------------
BM25 sparse retrieval — great for exact keyword matches, proper nouns, acronyms.
Scores are normalised to [0, 1] so they can be fused with cosine scores.
"""

from __future__ import annotations
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from ingestion.document_loader import Chunk


class KeywordSearch:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        tokenised = [self._tok(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenised)
        print(f"[KeywordSearch] BM25 built over {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        tokens = self._tok(query)
        raw = self.bm25.get_scores(tokens)
        max_score = max(raw) if max(raw) > 0 else 1.0
        norm = raw / max_score
        ranked = sorted(enumerate(norm), key=lambda x: x[1], reverse=True)
        return [
            (self.chunks[i], float(s))
            for i, s in ranked[:top_k]
            if s > 0
        ]

    @staticmethod
    def _tok(text: str) -> List[str]:
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1]
