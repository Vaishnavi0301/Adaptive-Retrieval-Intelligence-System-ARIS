"""
ingestion/document_loader.py
----------------------------
Loads raw text and splits it into overlapping chunks.
Each chunk keeps track of its source document so answers can cite it.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
from config import RAGConfig


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int       # index within the parent document
    doc_index: int      # index in the corpus list
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Chunk(src={self.source}, id={self.chunk_id}, '{self.text[:50]}…')"


class DocumentLoader:
    def __init__(self, config: RAGConfig):
        self.cfg = config

    def load_texts(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Chunk]:
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]
        all_chunks = []
        for doc_idx, (text, source) in enumerate(zip(texts, sources)):
            for c_idx, chunk_text in enumerate(self._split(text)):
                all_chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=c_idx,
                    doc_index=doc_idx,
                ))
        print(f"[Ingestion] {len(texts)} docs → {len(all_chunks)} chunks")
        return all_chunks

    def load_directory(self, dir_path: str) -> List[Chunk]:
        texts, sources = [], []
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                if fname.endswith(".txt"):
                    path = os.path.join(root, fname)
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
                    sources.append(fname)
        return self.load_texts(texts, sources)

    def _split(self, text: str) -> List[str]:
        """Sentence-aware sliding-window chunker."""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.cfg.chunk_size:
            return [text] if text else []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if current and len(current) + 1 + len(sent) > self.cfg.chunk_size:
                chunks.append(current.strip())
                overlap = current[-self.cfg.chunk_overlap:] if len(current) > self.cfg.chunk_overlap else current
                current = overlap + " " + sent
            else:
                current = (current + " " + sent).strip() if current else sent
        if current.strip():
            chunks.append(current.strip())
        return chunks
