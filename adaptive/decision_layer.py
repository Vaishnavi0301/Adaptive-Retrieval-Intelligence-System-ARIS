"""
adaptive/decision_layer.py
--------------------------
Decides top-K and alpha BEFORE every query, based on:
  - Query complexity (from QueryAnalyzer)
  - Live feedback stats (latency + quality from FeedbackLoop)

Rules:
  simple   query  →  min_top_k (2)
  moderate query  →  default_top_k (5)
  complex  query  →  max_top_k (12)

  high latency     →  reduce K (system is slow, cut retrieval depth)
  low quality      →  increase K (need more context)
  high specificity →  lower alpha (lean more on BM25 keyword search)
"""

from __future__ import annotations
from dataclasses import dataclass
from config import RAGConfig
from adaptive.query_analyzer import QueryAnalysis


@dataclass
class RetrievalPlan:
    top_k: int
    alpha: float
    notes: str


class DecisionLayer:
    def __init__(self, config: RAGConfig):
        self.cfg = config
        self._k_offset: int = 0
        self._alpha_offset: float = 0.0

    def plan(self, analysis: QueryAnalysis, feedback_stats: dict) -> RetrievalPlan:
        k     = self._pick_k(analysis, feedback_stats)
        alpha = self._pick_alpha(analysis)
        notes = (
            f"type={analysis.query_type} | "
            f"score={analysis.complexity_score:.2f} | "
            f"K={k} | alpha={alpha:.2f} | "
            f"p50_lat={feedback_stats.get('p50_latency', 0):.2f}s"
        )
        return RetrievalPlan(top_k=k, alpha=alpha, notes=notes)

    def apply_feedback(self, k_delta: int, alpha_delta: float) -> None:
        """Called by the FeedbackLoop to nudge parameters."""
        lo = self.cfg.min_top_k - self.cfg.default_top_k
        hi = self.cfg.max_top_k - self.cfg.default_top_k
        self._k_offset     = max(lo, min(hi, self._k_offset + k_delta))
        self._alpha_offset = max(-0.3, min(0.3, self._alpha_offset + alpha_delta))

    # ── internals ─────────────────────────────────────────────────────────

    def _pick_k(self, analysis: QueryAnalysis, stats: dict) -> int:
        cfg = self.cfg
        # Base K from complexity
        if   analysis.query_type == "simple":  base_k = cfg.min_top_k
        elif analysis.query_type == "complex": base_k = cfg.max_top_k
        else:                                  base_k = cfg.default_top_k

        # Latency pressure → reduce K
        lat_penalty = 0
        p50 = stats.get("p50_latency", 0.0)
        if p50 > cfg.high_latency_threshold:
            excess = p50 - cfg.high_latency_threshold
            lat_penalty = min(int(excess / 0.5) * cfg.k_bump, cfg.k_bump * 2)

        # Quality pressure → increase K
        qual_bonus = 0
        if stats.get("avg_quality", 1.0) < cfg.low_quality_threshold:
            qual_bonus = cfg.k_bump

        raw = base_k + self._k_offset - lat_penalty + qual_bonus
        return max(cfg.min_top_k, min(cfg.max_top_k, raw))

    def _pick_alpha(self, analysis: QueryAnalysis) -> float:
        base = self.cfg.default_alpha
        spec = analysis.signals.get("specificity", 0.0)
        if spec > 0.6:   base -= 0.15   # keyword-heavy → more BM25
        if analysis.query_type == "complex" and spec < 0.4:
            base += 0.1                  # conceptual → more vector
        return max(0.1, min(0.95, base + self._alpha_offset))
