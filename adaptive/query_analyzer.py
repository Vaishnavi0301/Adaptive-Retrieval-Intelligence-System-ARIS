"""
adaptive/query_analyzer.py
--------------------------
Scores query complexity from 0.0 (simple) to 1.0 (complex) using 5 signals.
No LLM needed — pure rule-based logic, runs in microseconds.

Signals:
  word_count      (30%) — more words = likely more complex
  question_depth  (30%) — "how/why/compare/explain" starters, WH-word count
  conjunction     (15%) — "and/but/however" suggest multi-aspect queries
  specificity     (15%) — proper nouns, acronyms, numbers, quoted terms
  clause_count    (10%) — commas / semicolons as sub-question proxies
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List
from config import RAGConfig

COMPLEX_STARTERS = {
    "how", "why", "explain", "compare", "contrast", "describe",
    "elaborate", "analyse", "analyze", "discuss", "evaluate", "differentiate",
}
CONJUNCTIONS = {
    "and", "but", "however", "although", "whereas", "while",
    "moreover", "furthermore", "additionally", "yet",
}
SPECIFICITY_PATTERNS = [
    r"\b\d{4}\b",
    r"\b[A-Z][a-z]+\b",
    r"\b[A-Z]{2,}\b",
    r'["\'](.*?)["\']',
]


@dataclass
class QueryAnalysis:
    query: str
    word_count: int
    complexity_score: float     # [0, 1]
    query_type: str             # "simple" | "moderate" | "complex"
    signals: dict


class QueryAnalyzer:
    def __init__(self, config: RAGConfig):
        self.cfg = config

    def analyse(self, query: str) -> QueryAnalysis:
        words = query.lower().split()
        wc = len(words)

        signals = {
            "word_count_score": self._wc_score(wc),
            "question_depth":   self._depth(query, words),
            "conjunction_load": self._conjunctions(words),
            "specificity":      self._specificity(query),
            "clause_count":     self._clauses(query),
        }
        weights = {
            "word_count_score": 0.30,
            "question_depth":   0.30,
            "conjunction_load": 0.15,
            "specificity":      0.15,
            "clause_count":     0.10,
        }
        score = max(0.0, min(1.0, sum(signals[k] * weights[k] for k in weights)))

        if   score < 0.35: qtype = "simple"
        elif score < 0.65: qtype = "moderate"
        else:              qtype = "complex"

        return QueryAnalysis(query=query, word_count=wc, complexity_score=score,
                             query_type=qtype, signals=signals)

    def _wc_score(self, wc: int) -> float:
        lo, hi = self.cfg.short_query_words, self.cfg.complex_query_words
        return max(0.0, min(1.0, (wc - lo) / max(hi - lo, 1)))

    def _depth(self, query: str, words: List[str]) -> float:
        lowq = query.lower()
        starts_complex = any(lowq.startswith(s) for s in COMPLEX_STARTERS)
        wh_words = {"what", "why", "how", "when", "where", "which", "who"}
        wh_count = len(wh_words & set(words))
        return min((0.6 if starts_complex else 0.0) + wh_count * 0.2, 1.0)

    def _conjunctions(self, words: List[str]) -> float:
        return min(sum(1 for w in words if w in CONJUNCTIONS) * 0.25, 1.0)

    def _specificity(self, query: str) -> float:
        matches = sum(len(re.findall(p, query)) for p in SPECIFICITY_PATTERNS)
        return min(matches * 0.2, 1.0)

    def _clauses(self, query: str) -> float:
        count = query.count(",") + query.count(";") + query.count("—")
        return min(count * 0.33, 1.0)
