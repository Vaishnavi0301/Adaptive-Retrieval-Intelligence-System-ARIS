"""
retrieval/vector_store.py
-------------------------
FAISS vector index for dense (semantic) retrieval.
Uses inner product on L2-normalised vectors = cosine similarity.
"""

from __future__ import annotations
import numpy as np
import faiss
from typing import List, Tuple
from ingestion.document_loader import Chunk


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == self.dim
        embs = self._norm(embeddings.astype(np.float32))
        self.index.add(embs)
        self.chunks.extend(chunks)
        print(f"[VectorStore] {len(chunks)} chunks indexed (total={self.index.ntotal})")

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        q = self._norm(query_vec.reshape(1, -1).astype(np.float32))
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    @staticmethod
    def _norm(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def __len__(self):
        return self.index.ntotal
