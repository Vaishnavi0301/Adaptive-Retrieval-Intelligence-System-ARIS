"""
pipeline.py
-----------
Main orchestrator. Features added:
  - Parallel decomposition  (ThreadPoolExecutor, sub-questions run concurrently)
  - Confidence-aware fallback (retry with larger K if quality is low)
  - Streaming generation support
  - Model routing (llama3.2:3b vs qwen2.5:7b)
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from ingestion.document_loader import DocumentLoader, Chunk
from retrieval.vector_store import VectorStore
from retrieval.keyword_search import KeywordSearch
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import HeuristicReranker
from adaptive.query_analyzer import QueryAnalyzer, QueryAnalysis
from adaptive.decision_layer import DecisionLayer, RetrievalPlan
from adaptive.feedback import FeedbackLoop
from adaptive.cache import QueryCache, SemanticCache
from adaptive.decomposer import QueryDecomposer
from generation.generator import Generator


@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: List[Tuple[Chunk, float]]
    plan: RetrievalPlan
    analysis: QueryAnalysis
    retrieval_time: float
    generation_time: float
    total_time: float
    model_used: str = ""
    from_cache: bool = False
    cache_type: str = ""          # "exact" | "semantic" | ""
    sub_questions: List[str] = field(default_factory=list)
    fallback_triggered: bool = False
    quality_score: float = 0.0


class AdaptiveRAGPipeline:
    def __init__(self, config: RAGConfig = None):
        self.cfg = config or RAGConfig()

        print("[Pipeline] Loading embedding model…")
        self.embedder         = SentenceTransformer(self.cfg.embedding_model)
        self.vector_store     = VectorStore(self.cfg.embedding_dim)
        self.keyword_search   = KeywordSearch()
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.keyword_search, self.cfg.default_alpha
        )
        self.reranker         = HeuristicReranker()
        self.query_analyzer   = QueryAnalyzer(self.cfg)
        self.decision_layer   = DecisionLayer(self.cfg)
        self.feedback_loop    = FeedbackLoop(self.cfg, self.decision_layer)

        # Dual cache: exact LRU + semantic similarity
        self.cache          = QueryCache(self.cfg.cache_max_size)
        self.semantic_cache = SemanticCache(
            self.embedder,
            threshold=self.cfg.semantic_cache_threshold,
            max_size=self.cfg.cache_max_size,
        )

        self.generator  = Generator(self.cfg)
        self.decomposer = None
        if self.cfg.enable_decomposition:
            self.decomposer = QueryDecomposer(
                model=self.cfg.large_model,
                base_url=self.cfg.ollama_base_url,
            )
        self._built = False

        # History for dashboard
        self.history: List[RAGResult] = []

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, texts: List[str], sources: List[str] = None) -> None:
        chunks = DocumentLoader(self.cfg).load_texts(texts, sources)
        self._index(chunks)

    def ingest_directory(self, dir_path: str) -> None:
        chunks = DocumentLoader(self.cfg).load_directory(dir_path)
        self._index(chunks)

    def _index(self, chunks: List[Chunk]) -> None:
        print(f"[Pipeline] Embedding {len(chunks)} chunks…")
        embs = self.embedder.encode(
            [c.text for c in chunks],
            show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
        self.vector_store.build(chunks, embs)
        self.keyword_search.build(chunks)
        self._built = True
        print(f"[Pipeline] Ready — {len(chunks)} chunks indexed.\n")

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> RAGResult:
        if not self._built:
            raise RuntimeError("Call ingest() first.")

        t_start = time.perf_counter()

        # 1. Exact cache
        if self.cfg.enable_cache:
            cached = self.cache.get(question)
            if cached:
                print(f"[Cache] Exact HIT")
                result = self._cache_result(question, cached, "exact", t_start)
                self.history.append(result)
                return result

        # 2. Semantic cache
        if self.cfg.enable_semantic_cache:
            sem = self.semantic_cache.get(question)
            if sem:
                print(f"[Cache] Semantic HIT (similarity={sem['similarity']:.3f})")
                result = self._cache_result(question, sem, "semantic", t_start)
                self.history.append(result)
                return result

        # 3. Analyse + Plan
        analysis = self.query_analyzer.analyse(question)
        plan     = self.decision_layer.plan(analysis, self.feedback_loop.stats)
        model    = self.generator.route_model(analysis.query_type)
        print(f"[Plan]  {plan.notes}")
        print(f"[Model] Routing to → {model}  (query_type={analysis.query_type})")

        # 4. Parallel decomposition for complex queries
        if (
            self.decomposer
            and analysis.complexity_score >= self.cfg.decomposition_complexity_threshold
        ):
            sub_qs = self.decomposer.decompose(question)
            if len(sub_qs) > 1:
                result = self._decomposed_parallel(
                    question, sub_qs, plan, analysis, t_start, model
                )
                self.history.append(result)
                return result

        # 5. Standard retrieval
        t_ret  = time.perf_counter()
        q_vec  = self.embedder.encode([question], normalize_embeddings=True)[0]
        raw    = self.hybrid_retriever.search(question, q_vec, plan.top_k, plan.alpha)
        ranked = self.reranker.rerank(question, raw, plan.top_k)
        retrieval_time = time.perf_counter() - t_ret

        # 6. Generate (streaming optional)
        t_gen = time.perf_counter()
        if stream_callback:
            answer = self.generator.generate_stream(question, ranked, analysis.query_type, stream_callback)
        else:
            answer, _, model = self.generator.generate(question, ranked, analysis.query_type)
        generation_time = time.perf_counter() - t_gen

        # 7. Confidence-aware fallback
        quality = self.feedback_loop._quality(answer, [c for c, _ in ranked])
        fallback = False
        if quality < self.cfg.low_quality_threshold and plan.top_k < self.cfg.max_top_k:
            print(f"[Fallback] Low quality ({quality:.2f}) → retrying with K={self.cfg.max_top_k}")
            fallback_k = self.cfg.max_top_k
            raw2    = self.hybrid_retriever.search(question, q_vec, fallback_k, plan.alpha)
            ranked2 = self.reranker.rerank(question, raw2, fallback_k)
            t_gen2  = time.perf_counter()
            answer, _, model = self.generator.generate(question, ranked2, analysis.query_type)
            generation_time += time.perf_counter() - t_gen2
            ranked = ranked2
            fallback = True

        total_time = time.perf_counter() - t_start

        # 8. Store in both caches
        if self.cfg.enable_cache:
            self.cache.put(question, answer, [c for c, _ in ranked])
        if self.cfg.enable_semantic_cache:
            q_vec2 = self.embedder.encode([question], normalize_embeddings=True)[0]
            self.semantic_cache.put(question, q_vec2, answer, [c for c, _ in ranked])

        # 9. Feedback
        rec = self.feedback_loop.record(
            query=question, top_k=plan.top_k, alpha=plan.alpha,
            retrieval_time=retrieval_time, generation_time=generation_time,
            answer=answer, retrieved_chunks=[c for c, _ in ranked],
        )

        result = RAGResult(
            query=question, answer=answer, retrieved_chunks=ranked,
            plan=plan, analysis=analysis,
            retrieval_time=retrieval_time, generation_time=generation_time,
            total_time=total_time, model_used=model,
            fallback_triggered=fallback, quality_score=quality,
        )
        self.history.append(result)
        return result

    # ── Parallel decomposition ─────────────────────────────────────────────

    def _run_subquery(self, sq: str, plan: RetrievalPlan, query_type: str):
        """Single sub-question pipeline — runs in a thread."""
        sq_vec = self.embedder.encode([sq], normalize_embeddings=True)[0]
        raw    = self.hybrid_retriever.search(sq, sq_vec, plan.top_k, plan.alpha)
        ranked = self.reranker.rerank(sq, raw, plan.top_k)
        ans, _, model = self.generator.generate(sq, ranked, query_type)
        return sq, ans, ranked

    def _decomposed_parallel(
        self, main_q, sub_qs, plan, analysis, t_start, model_used
    ) -> RAGResult:
        print(f"[Decomposer] Parallel execution of {len(sub_qs)} sub-questions…")
        sub_qa: List[tuple] = [None] * len(sub_qs)
        all_chunks = []
        t_ret = 0.0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(sub_qs), 4)) as ex:
            future_to_idx = {
                ex.submit(self._run_subquery, sq, plan, analysis.query_type): i
                for i, sq in enumerate(sub_qs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                sq, ans, ranked = future.result()
                sub_qa[idx] = (sq, ans)
                all_chunks.extend(ranked)

        parallel_time = time.perf_counter() - t0
        print(f"[Decomposer] All sub-questions done in {parallel_time:.1f}s (parallel)")

        t_synth = time.perf_counter()
        final   = self.decomposer.synthesise(main_q, sub_qa)
        synth_time = time.perf_counter() - t_synth

        total = time.perf_counter() - t_start

        if self.cfg.enable_cache:
            self.cache.put(main_q, final, [c for c, _ in all_chunks])

        rec = self.feedback_loop.record(
            query=main_q, top_k=plan.top_k, alpha=plan.alpha,
            retrieval_time=t_ret, generation_time=parallel_time + synth_time,
            answer=final, retrieved_chunks=[c for c, _ in all_chunks],
        )

        return RAGResult(
            query=main_q, answer=final, retrieved_chunks=all_chunks,
            plan=plan, analysis=analysis,
            retrieval_time=t_ret, generation_time=parallel_time + synth_time,
            total_time=total, model_used=model_used,
            sub_questions=sub_qs, quality_score=rec.quality_score,
        )

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_result(self, question, cached, cache_type, t_start) -> RAGResult:
        dummy_plan     = RetrievalPlan(self.cfg.default_top_k, self.cfg.default_alpha, "cache_hit")
        dummy_analysis = self.query_analyzer.analyse(question)
        answer  = cached.get("answer", cached.answer) if isinstance(cached, dict) else cached.answer
        chunks  = cached.get("chunks", cached.chunks) if isinstance(cached, dict) else cached.chunks
        return RAGResult(
            query=question, answer=answer,
            retrieved_chunks=[(c, 0.0) for c in chunks],
            plan=dummy_plan, analysis=dummy_analysis,
            retrieval_time=0.0, generation_time=0.0,
            total_time=time.perf_counter() - t_start,
            model_used="(cached)", from_cache=True, cache_type=cache_type,
        )

    def performance_report(self) -> dict:
        report = self.feedback_loop.summary()
        report["cache"]          = self.cache.stats()
        report["semantic_cache"] = self.semantic_cache.stats()
        return report
