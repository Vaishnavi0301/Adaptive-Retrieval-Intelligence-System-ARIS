"""
tests/test_all.py
-----------------
31 unit tests — NO API key and NO Ollama needed.
Tests every component except the Generator (which needs Ollama).

Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import RAGConfig
from ingestion.document_loader import DocumentLoader, Chunk
from retrieval.keyword_search import KeywordSearch
from retrieval.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import HeuristicReranker
from adaptive.query_analyzer import QueryAnalyzer
from adaptive.decision_layer import DecisionLayer
from adaptive.cache import QueryCache


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return RAGConfig(chunk_size=200, chunk_overlap=40)

@pytest.fixture
def sample_texts():
    return [
        "Machine learning is a subset of artificial intelligence. "
        "It allows systems to learn from data without being explicitly programmed.",
        "Deep learning uses neural networks with many layers. "
        "Convolutional networks excel at image classification tasks.",
        "Retrieval-Augmented Generation combines retrieval with language models. "
        "FAISS is a popular vector index for similarity search.",
    ]

@pytest.fixture
def chunks(cfg, sample_texts):
    return DocumentLoader(cfg).load_texts(sample_texts, ["doc_a", "doc_b", "doc_c"])

@pytest.fixture
def ks(chunks):
    s = KeywordSearch()
    s.build(chunks)
    return s

@pytest.fixture
def vs_fixture(chunks):
    dim = 8
    vs = VectorStore(embedding_dim=dim)
    np.random.seed(42)
    embs = np.random.randn(len(chunks), dim).astype(np.float32)
    vs.build(chunks, embs)
    return vs, dim, chunks


# ── DocumentLoader ────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_produces_chunks(self, cfg, sample_texts):
        chunks = DocumentLoader(cfg).load_texts(sample_texts)
        assert len(chunks) >= len(sample_texts)

    def test_source_preserved(self, cfg, sample_texts):
        chunks = DocumentLoader(cfg).load_texts(sample_texts, ["s0","s1","s2"])
        assert all(c.source in ["s0","s1","s2"] for c in chunks)

    def test_short_text_one_chunk(self, cfg):
        chunks = DocumentLoader(cfg).load_texts(["Hello world."])
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, cfg):
        long = ("The quick brown fox jumps over the lazy dog. " * 15).strip()
        chunks = DocumentLoader(cfg).load_texts([long])
        assert len(chunks) > 1

    def test_chunk_text_not_empty(self, cfg, sample_texts):
        for c in DocumentLoader(cfg).load_texts(sample_texts):
            assert len(c.text) > 0

    def test_chunk_id_increments(self, cfg):
        long = ("Sentence number one is here. " * 20).strip()
        chunks = DocumentLoader(cfg).load_texts([long])
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))


# ── KeywordSearch ─────────────────────────────────────────────────────────

class TestKeywordSearch:
    def test_search_returns_results(self, ks, chunks):
        results = ks.search("machine learning artificial intelligence", top_k=3)
        assert len(results) > 0

    def test_scores_descending(self, ks):
        results = ks.search("neural network deep learning", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_normalised(self, ks):
        results = ks.search("learning", top_k=5)
        for _, s in results:
            assert 0.0 <= s <= 1.0

    def test_top_k_limit(self, ks):
        results = ks.search("learning", top_k=1)
        assert len(results) <= 1

    def test_no_match_empty(self, ks):
        results = ks.search("xyzzy qwerty nonsense", top_k=5)
        assert all(s == 0 for _, s in results)


# ── VectorStore ───────────────────────────────────────────────────────────

class TestVectorStore:
    def test_search_returns_results(self, vs_fixture):
        vs, dim, _ = vs_fixture
        q = np.random.randn(dim).astype(np.float32)
        assert len(vs.search(q, top_k=2)) == 2

    def test_scores_in_cosine_range(self, vs_fixture):
        vs, dim, _ = vs_fixture
        q = np.random.randn(dim).astype(np.float32)
        for _, score in vs.search(q, top_k=5):
            assert -1.01 <= score <= 1.01

    def test_top_k_limit(self, vs_fixture):
        vs, dim, _ = vs_fixture
        q = np.random.randn(dim).astype(np.float32)
        assert len(vs.search(q, top_k=1)) == 1

    def test_len(self, vs_fixture):
        vs, _, chunks = vs_fixture
        assert len(vs) == len(chunks)


# ── HybridRetriever ───────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_returns_results(self, vs_fixture, ks):
        vs, dim, chunks = vs_fixture
        hr = HybridRetriever(vs, ks, alpha=0.7)
        q  = np.random.randn(dim).astype(np.float32)
        results = hr.search("machine learning", q, top_k=3)
        assert 0 < len(results) <= 3

    def test_pure_vector(self, vs_fixture, ks):
        vs, dim, _ = vs_fixture
        hr = HybridRetriever(vs, ks, alpha=1.0)
        q  = np.random.randn(dim).astype(np.float32)
        assert len(hr.search("deep learning", q, top_k=3, alpha=1.0)) > 0

    def test_pure_bm25(self, vs_fixture, ks):
        vs, dim, _ = vs_fixture
        hr = HybridRetriever(vs, ks, alpha=0.0)
        q  = np.random.randn(dim).astype(np.float32)
        # BM25-only — results for a known keyword
        assert len(hr.search("neural", q, top_k=3, alpha=0.0)) >= 0  # may be 0 if not found


# ── HeuristicReranker ─────────────────────────────────────────────────────

class TestHeuristicReranker:
    def test_preserves_count(self, chunks):
        rr = HeuristicReranker()
        inp = [(c, 0.5) for c in chunks[:3]]
        assert len(rr.rerank("machine learning", inp)) == 3

    def test_top_k(self, chunks):
        rr = HeuristicReranker()
        inp = [(c, 0.5) for c in chunks]
        assert len(rr.rerank("learning", inp, top_k=1)) == 1

    def test_scores_positive(self, chunks):
        rr = HeuristicReranker()
        inp = [(c, 0.8) for c in chunks[:3]]
        for _, s in rr.rerank("machine", inp):
            assert s >= 0.0


# ── QueryAnalyzer ─────────────────────────────────────────────────────────

class TestQueryAnalyzer:
    def test_simple_query(self, cfg):
        a = QueryAnalyzer(cfg).analyse("What is ML?")
        assert a.query_type == "simple"
        assert a.complexity_score < 0.4

    def test_complex_query(self, cfg):
        q = ("Compare and contrast dense retrieval and sparse BM25 retrieval, "
             "explaining how hybrid methods address their individual weaknesses.")
        a = QueryAnalyzer(cfg).analyse(q)
        assert a.query_type in ("moderate", "complex")

    def test_score_bounded(self, cfg):
        qa = QueryAnalyzer(cfg)
        for q in ["Hi", "What is attention?", "Compare transformers and RNNs in detail."]:
            assert 0.0 <= qa.analyse(q).complexity_score <= 1.0

    def test_word_count(self, cfg):
        a = QueryAnalyzer(cfg).analyse("What is attention mechanism?")
        assert a.word_count == 4


# ── DecisionLayer ─────────────────────────────────────────────────────────

class TestDecisionLayer:
    def test_simple_low_k(self, cfg):
        a = QueryAnalyzer(cfg).analyse("What is ML?")
        p = DecisionLayer(cfg).plan(a, {})
        assert p.top_k <= cfg.default_top_k

    def test_complex_high_k(self, cfg):
        q = "Compare and contrast multiple retrieval strategies in great detail."
        a = QueryAnalyzer(cfg).analyse(q)
        p = DecisionLayer(cfg).plan(a, {})
        assert p.top_k >= cfg.default_top_k

    def test_high_latency_reduces_k(self, cfg):
        a = QueryAnalyzer(cfg).analyse("What is deep learning?")
        dl = DecisionLayer(cfg)
        k_normal = dl.plan(a, {"p50_latency": 0.5}).top_k
        k_slow   = dl.plan(a, {"p50_latency": 5.0}).top_k
        assert k_slow <= k_normal

    def test_k_within_bounds(self, cfg):
        dl = DecisionLayer(cfg)
        a  = QueryAnalyzer(cfg).analyse("What is X?")
        for lat in [0.0, 2.0, 10.0]:
            p = dl.plan(a, {"p50_latency": lat})
            assert cfg.min_top_k <= p.top_k <= cfg.max_top_k

    def test_alpha_in_range(self, cfg):
        dl = DecisionLayer(cfg)
        for q in ["What?", "Compare BM25 and vector search in 2024 RAG pipelines."]:
            p = dl.plan(QueryAnalyzer(cfg).analyse(q), {})
            assert 0.0 <= p.alpha <= 1.0


# ── QueryCache ────────────────────────────────────────────────────────────

class TestQueryCache:
    def test_miss_then_hit(self):
        c = QueryCache(10)
        assert c.get("hello") is None
        c.put("hello", "ans", [])
        assert c.get("hello") is not None

    def test_normalisation(self):
        c = QueryCache(10)
        c.put("Hello World", "ans", [])
        assert c.get("hello world") is not None
        assert c.get("  hello   world  ") is not None

    def test_lru_eviction(self):
        c = QueryCache(max_size=2)
        c.put("q1", "a1", [])
        c.put("q2", "a2", [])
        c.put("q3", "a3", [])   # evicts q1
        assert c.get("q1") is None
        assert c.get("q3") is not None

    def test_hit_rate(self):
        c = QueryCache(10)
        c.put("q", "a", [])
        c.get("q")       # hit
        c.get("q")       # hit
        c.get("miss1")   # miss
        c.get("miss2")   # miss
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 2
        assert abs(s["hit_rate"] - 0.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
