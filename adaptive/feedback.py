"""
adaptive/feedback.py
--------------------
Tracks per-query latency and quality, then nudges the DecisionLayer.

Quality proxy (no ground truth labels needed):
  1. Answer length    (0–0.40)  — very short = likely "I don't know"
  2. Context overlap  (0–0.35)  — do chunk keywords appear in the answer?
  3. Confidence words (0–0.25)  — penalise "I'm not sure", reward "according to"

EMA (Exponential Moving Average) smooths metrics so one bad query doesn't
cause wild swings. alpha=0.25 means recent queries count more but aren't
overwhelming.

Adjustment fires every 3 queries (configurable via _adjust_every).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List
import numpy as np
from config import RAGConfig
from adaptive.decision_layer import DecisionLayer
from ingestion.document_loader import Chunk

LOW_CONF  = ["i'm not sure", "i don't know", "unclear", "cannot determine",
             "no information", "not mentioned", "i am unable"]
HIGH_CONF = ["specifically", "according to", "the document states",
             "as described", "clearly", "in summary"]


@dataclass
class QueryRecord:
    query: str
    top_k: int
    alpha: float
    retrieval_time: float
    generation_time: float
    total_time: float
    answer_length: int
    quality_score: float
    n_chunks: int
    timestamp: float = field(default_factory=time.time)


class FeedbackLoop:
    def __init__(self, config: RAGConfig, decision_layer: DecisionLayer):
        self.cfg = config
        self.dl  = decision_layer
        self.records: List[QueryRecord] = []
        self._ema_lat  = 0.0
        self._ema_qual = 1.0
        self._ema_ret  = 0.0
        self._ema_gen  = 0.0
        self._first    = True
        self._since_adjust = 0
        self._adjust_every = 3

    def record(self, query, top_k, alpha, retrieval_time, generation_time,
               answer, retrieved_chunks) -> QueryRecord:
        total   = retrieval_time + generation_time
        quality = self._quality(answer, retrieved_chunks)
        rec = QueryRecord(
            query=query, top_k=top_k, alpha=alpha,
            retrieval_time=retrieval_time, generation_time=generation_time,
            total_time=total, answer_length=len(answer),
            quality_score=quality, n_chunks=len(retrieved_chunks),
        )
        self.records.append(rec)
        self._update_ema(rec)
        self._since_adjust += 1
        if self._since_adjust >= self._adjust_every:
            self._adjust()
            self._since_adjust = 0
        return rec

    @property
    def stats(self) -> dict:
        return {
            "p50_latency":  self._ema_lat,
            "avg_quality":  self._ema_qual,
            "avg_ret_time": self._ema_ret,
            "avg_gen_time": self._ema_gen,
            "n_queries":    len(self.records),
        }

    def summary(self) -> dict:
        if not self.records:
            return {}
        lat = [r.total_time        for r in self.records]
        ret = [r.retrieval_time    for r in self.records]
        gen = [r.generation_time   for r in self.records]
        return {
            "n_queries":    len(self.records),
            "p50_latency":  float(np.percentile(lat, 50)),
            "p95_latency":  float(np.percentile(lat, 95)),
            "p50_ret_time": float(np.percentile(ret, 50)),
            "p95_ret_time": float(np.percentile(ret, 95)),
            "p50_gen_time": float(np.percentile(gen, 50)),
            "p95_gen_time": float(np.percentile(gen, 95)),
            "avg_quality":  float(np.mean([r.quality_score for r in self.records])),
            "avg_top_k":    float(np.mean([r.top_k         for r in self.records])),
            "avg_alpha":    float(np.mean([r.alpha         for r in self.records])),
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _update_ema(self, rec: QueryRecord):
        a = self.cfg.ema_alpha
        if self._first:
            self._ema_lat, self._ema_qual = rec.total_time, rec.quality_score
            self._ema_ret, self._ema_gen  = rec.retrieval_time, rec.generation_time
            self._first = False
        else:
            self._ema_lat  = a * rec.total_time      + (1-a) * self._ema_lat
            self._ema_qual = a * rec.quality_score   + (1-a) * self._ema_qual
            self._ema_ret  = a * rec.retrieval_time  + (1-a) * self._ema_ret
            self._ema_gen  = a * rec.generation_time + (1-a) * self._ema_gen

    def _adjust(self):
        k_d, a_d = 0, 0.0
        if self._ema_lat  > self.cfg.high_latency_threshold: k_d -= self.cfg.k_bump
        if self._ema_qual < self.cfg.low_quality_threshold:  k_d += self.cfg.k_bump
        tot = self._ema_ret + self._ema_gen
        if tot > 0 and self._ema_ret / tot > 0.6 and self._ema_lat > 1.5:
            k_d -= 1; a_d -= 0.05
        if k_d != 0 or a_d != 0:
            self.dl.apply_feedback(k_d, a_d)
            print(f"[Feedback] k_delta={k_d:+d}, alpha_delta={a_d:+.2f} "
                  f"(ema_lat={self._ema_lat:.2f}s, ema_qual={self._ema_qual:.2f})")

    def _quality(self, answer: str, chunks: List[Chunk]) -> float:
        if not answer or len(answer) < 10:
            return 0.05
        length_score = min(len(answer) / 800.0, 1.0) * 0.40
        if chunks:
            al = answer.lower()
            covered = sum(
                1 for c in chunks[:3]
                if len(set(c.text.lower().split()) & set(al.split())) / max(len(c.text.split()), 1) > 0.1
            )
            util_score = (covered / min(3, len(chunks))) * 0.35
        else:
            util_score = 0.0
        al = answer.lower()
        conf = 0.5
        for p in LOW_CONF:
            if p in al: conf -= 0.2
        for p in HIGH_CONF:
            if p in al: conf += 0.1
        conf_score = max(0.0, min(1.0, conf)) * 0.25
        return length_score + util_score + conf_score
