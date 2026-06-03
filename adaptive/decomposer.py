"""
adaptive/decomposer.py
----------------------
Bonus: Breaks complex multi-part queries into simpler sub-questions,
runs retrieval on each, then synthesises a final answer.

Example:
  "Compare dense vs sparse retrieval and explain how hybrid methods
   combine them"
  → Sub-Q1: "What is dense retrieval?"
  → Sub-Q2: "What is sparse / BM25 retrieval?"
  → Sub-Q3: "How does hybrid retrieval combine dense and sparse?"
  → Synthesise → final answer

Only activates when complexity_score >= threshold (default 0.65).
Trade-off: 2-4x more latency but significantly better answers for
multi-aspect questions.
"""

from __future__ import annotations
import re
import json
import requests
from typing import List


DECOMPOSE_PROMPT = """Break the following complex question into 2-4 simple, atomic sub-questions.
Each sub-question must be independently answerable on its own.
Return ONLY a JSON array of strings, no explanation, no markdown fences.
Example output: ["What is X?", "What is Y?", "How do X and Y relate?"]

Question to decompose: {query}"""

SYNTHESIS_PROMPT = """You are given a main question and answers to several sub-questions.
Write a single, coherent, well-structured answer to the main question.
Use only information from the sub-answers. Be concise.

Main question: {main_query}

Sub-question answers:
{sub_answers}

Your synthesised answer:"""


class QueryDecomposer:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def decompose(self, query: str) -> List[str]:
        """Returns list of sub-questions, or [query] if decomposition fails."""
        try:
            raw = self._call(DECOMPOSE_PROMPT.format(query=query))
            raw = re.sub(r"```json|```", "", raw).strip()
            sub_qs = json.loads(raw)
            if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                return [str(q) for q in sub_qs[:4]]
        except Exception as e:
            print(f"[Decomposer] Failed ({e}), using original query.")
        return [query]

    def synthesise(self, main_query: str, sub_qa_pairs: List[tuple]) -> str:
        """Merges sub-answers into one final answer."""
        sub_answers = "\n\n".join(
            f"Sub-question {i+1}: {q}\nAnswer: {a}"
            for i, (q, a) in enumerate(sub_qa_pairs)
        )
        return self._call(
            SYNTHESIS_PROMPT.format(main_query=main_query, sub_answers=sub_answers),
            max_tokens=1024,
        )
