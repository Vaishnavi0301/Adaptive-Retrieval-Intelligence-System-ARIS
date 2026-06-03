"""
adaptive/cache.py
-----------------
Two cache layers:

1. QueryCache (exact LRU)
   Key = normalised query string. O(1) lookup.

2. SemanticCache (embedding similarity)
   Stores query embeddings. On lookup, finds nearest cached query
   by cosine similarity. Returns cached answer if similarity > threshold.

   Example:
     Cached: "What is ML?"
     Query:  "Explain machine learning"
     Similarity: 0.91 → cache HIT, return same answer instantly.
"""

from __future__ import annotations
import re
import time
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List


# ── Exact LRU Cache ────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    answer: str
    chunks: list
    timestamp: float
    hit_count: int = 0


class QueryCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[CacheEntry]:
        key = self._norm(query)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key].hit_count += 1
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, query: str, answer: str, chunks: list) -> None:
        key = self._norm(query)
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)
        self._store[key] = CacheEntry(answer=answer, chunks=chunks, timestamp=time.time())

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "type": "exact",
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    @staticmethod
    def _norm(query: str) -> str:
        return re.sub(r"\s+", " ", query.lower().strip())


# ── Semantic Cache ─────────────────────────────────────────────────────────

class SemanticCache:
    """
    Finds semantically similar cached queries using cosine similarity.
    If similarity > threshold, returns the cached answer.

    Storage: list of (query_text, embedding, answer, chunks)
    Lookup:  O(N) — fine for small caches (<500 entries)
    """

    def __init__(self, embedder, threshold: float = 0.92, max_size: int = 256):
        self.embedder  = embedder
        self.threshold = threshold
        self.max_size  = max_size
        self._entries: List[dict] = []   # {query, embedding, answer, chunks, ts}
        self.hits   = 0
        self.misses = 0

    def get(self, query: str) -> Optional[dict]:
        if not self._entries:
            self.misses += 1
            return None

        q_vec = self._embed(query)
        stored_vecs = np.array([e["embedding"] for e in self._entries])  # (N, dim)
        sims = stored_vecs @ q_vec                                        # cosine (normalised)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            self.hits += 1
            entry = self._entries[best_idx]
            return {
                "answer":     entry["answer"],
                "chunks":     entry["chunks"],
                "similarity": best_sim,
                "matched_query": entry["query"],
            }
        self.misses += 1
        return None

    def put(self, query: str, query_vec: np.ndarray, answer: str, chunks: list) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)   # evict oldest
        self._entries.append({
            "query":     query,
            "embedding": query_vec / (np.linalg.norm(query_vec) + 1e-9),
            "answer":    answer,
            "chunks":    chunks,
            "ts":        time.time(),
        })

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "type":      "semantic",
            "size":      len(self._entries),
            "threshold": self.threshold,
            "hits":      self.hits,
            "misses":    self.misses,
            "hit_rate":  self.hits / total if total > 0 else 0.0,
        }

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], normalize_embeddings=True)[0]
        return vec.astype(np.float32)
