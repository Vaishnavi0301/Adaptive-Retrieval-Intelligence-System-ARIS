"""
generation/generator.py
-----------------------
Ollama wrapper with:
  - Model routing  (llama3.2:3b for simple, qwen2.5:7b for complex)
  - Streaming      (generate_stream → calls callback per token)
  - Standard       (generate → returns full string)
"""

from __future__ import annotations
import time
import json
import requests
from typing import List, Tuple, Optional, Callable
from config import RAGConfig
from ingestion.document_loader import Chunk

SYSTEM_PROMPT = """You are a precise, factual question-answering assistant.
You are given retrieved document passages as context.
Answer the user's question using ONLY information from these passages.
If the passages don't contain enough information, say so explicitly.
Cite sources using [Source: <name>] when relevant.
Be concise but complete. Do not speculate beyond the provided context."""


class Generator:
    def __init__(self, config: RAGConfig):
        self.cfg = config
        self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.cfg.ollama_base_url}/api/tags", timeout=5)
            available_base = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            for model in [self.cfg.small_model, self.cfg.large_model]:
                if model.split(":")[0] not in available_base:
                    print(f"[WARN] Model '{model}' not found. Run: ollama pull {model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "\n[ERROR] Cannot connect to Ollama at http://localhost:11434\n"
                "        Open the Ollama app first.\n"
            )

    def route_model(self, query_type: str) -> str:
        if not self.cfg.enable_model_routing:
            return self.cfg.large_model
        return self.cfg.large_model if query_type == "complex" else self.cfg.small_model

    # ── Standard generation ────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        query_type: str = "moderate",
    ) -> Tuple[str, float, str]:
        """Returns (answer, elapsed_seconds, model_used)."""
        model   = self.route_model(query_type)
        context = self._build_context(retrieved)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
        }
        t0 = time.perf_counter()
        r  = requests.post(f"{self.cfg.ollama_base_url}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        answer  = r.json().get("message", {}).get("content", "")
        return answer, elapsed, model

    # ── Streaming generation ───────────────────────────────────────────────

    def generate_stream(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        query_type: str = "moderate",
        callback: Callable[[str], None] = None,
    ) -> str:
        """
        Streams tokens from Ollama.
        callback(token) is called for each token as it arrives.
        Returns the full answer string when done.
        """
        model   = self.route_model(query_type)
        context = self._build_context(retrieved)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "stream": True,
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
        }

        full_answer = []
        with requests.post(
            f"{self.cfg.ollama_base_url}/api/chat",
            json=payload, stream=True, timeout=180
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk_data = json.loads(line)
                    token = chunk_data.get("message", {}).get("content", "")
                    if token:
                        full_answer.append(token)
                        if callback:
                            callback(token)
                    if chunk_data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        return "".join(full_answer)

    @staticmethod
    def _build_context(retrieved: List[Tuple[Chunk, float]]) -> str:
        parts = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            parts.append(f"[{rank}] Source: {chunk.source} | score={score:.3f}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)
