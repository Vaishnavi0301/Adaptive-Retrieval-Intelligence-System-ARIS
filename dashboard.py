"""
dashboard.py
------------
Streamlit dashboard for the Adaptive RAG system.

Run: streamlit run dashboard.py

Features:
  - Live query with streaming tokens
  - Retrieved chunks with BM25 / vector / rerank scores
  - Adaptive K history chart
  - Latency breakdown chart
  - Model routing decisions
  - Cache stats (exact + semantic)
  - Feedback loop state
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np

from config import RAGConfig
from pipeline import AdaptiveRAGPipeline

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 16px;
    margin: 4px 0;
}
.chunk-card {
    background: #181825;
    border-left: 4px solid #89b4fa;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    font-size: 0.85rem;
}
.simple-badge   { background:#a6e3a1; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
.moderate-badge { background:#f9e2af; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
.complex-badge  { background:#f38ba8; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
.model-small    { background:#89dceb; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
.model-large    { background:#cba6f7; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
.cache-hit      { background:#a6e3a1; color:#1e1e2e; padding:2px 8px; border-radius:10px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline init (cached across sessions) ─────────────────────────────────
@st.cache_resource
def load_pipeline():
    cfg = RAGConfig(
        small_model="llama3.2:3b",
        large_model="qwen2.5:7b",
        enable_model_routing=True,
        enable_decomposition=True,
        enable_cache=True,
        enable_semantic_cache=True,
        enable_fallback=True,
        decomposition_complexity_threshold=0.65,
    )
    pipeline = AdaptiveRAGPipeline(config=cfg)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pipeline.ingest_directory(data_dir)
    return pipeline


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    use_streaming = st.toggle("Streaming generation", value=True)
    st.divider()

    st.markdown("**Model Routing**")
    st.markdown("🟢 `simple/moderate` → llama3.2:3b")
    st.markdown("🟣 `complex` → qwen2.5:7b")
    st.divider()

    st.markdown("**Adaptive Logic**")
    st.markdown("• Dynamic top-K (2 → 12)")
    st.markdown("• Hybrid retrieval (FAISS + BM25)")
    st.markdown("• Confidence-aware fallback")
    st.markdown("• Query decomposition")
    st.divider()

    st.markdown("**Cache Layers**")
    st.markdown("• Exact LRU cache")
    st.markdown("• Semantic similarity cache")


# ── Main layout ────────────────────────────────────────────────────────────
st.title("🔍 Adaptive RAG Inference System")
st.caption("Real-time adaptive retrieval with model routing, hybrid search, and semantic caching")

pipeline = load_pipeline()

# ── Query input ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Ask a question",
        placeholder="e.g. What is supervised learning? / Compare dense vs sparse retrieval…",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("Ask ➤", use_container_width=True, type="primary")

# ── Example queries ────────────────────────────────────────────────────────
st.markdown("**Try these:**")
examples = [
    "What is supervised learning?",
    "What is FAISS?",
    "How does BM25 differ from vector search?",
    "Compare dense and sparse retrieval in RAG systems including strengths and weaknesses.",
    "Explain how the Transformer architecture works and why it replaced RNNs.",
]
ex_cols = st.columns(len(examples))
for i, (col, ex) in enumerate(zip(ex_cols, examples)):
    with col:
        if st.button(ex[:35] + ("…" if len(ex) > 35 else ""), key=f"ex_{i}", use_container_width=True):
            query = ex
            run   = True

st.divider()

# ── Run query ──────────────────────────────────────────────────────────────
if run and query:
    # ── Query analysis preview ─────────────────────────────────────────────
    analysis = pipeline.query_analyzer.analyse(query)
    plan     = pipeline.decision_layer.plan(analysis, pipeline.feedback_loop.stats)
    model    = pipeline.generator.route_model(analysis.query_type)

    badge_class = {"simple": "simple-badge", "moderate": "moderate-badge", "complex": "complex-badge"}.get(analysis.query_type, "moderate-badge")
    model_class = "model-small" if "3b" in model else "model-large"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Complexity Score", f"{analysis.complexity_score:.2f}")
    col2.metric("Query Type", analysis.query_type.capitalize())
    col3.metric("Top-K", plan.top_k)
    col4.metric("Alpha (vec weight)", f"{plan.alpha:.2f}")
    col5.metric("Model", model.split(":")[0])

    st.markdown(
        f"<span class='{badge_class}'>{analysis.query_type}</span> &nbsp;"
        f"<span class='{model_class}'>{model}</span> &nbsp;"
        f"words={analysis.word_count} | K={plan.top_k} | α={plan.alpha:.2f}",
        unsafe_allow_html=True
    )

    # ── Answer area ────────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    answer_placeholder = st.empty()

    with st.spinner("Thinking…"):
        if use_streaming and not analysis.complexity_score >= pipeline.cfg.decomposition_complexity_threshold:
            # Stream tokens live
            tokens = []
            def on_token(tok):
                tokens.append(tok)
                answer_placeholder.markdown("".join(tokens) + "▌")

            t0     = time.perf_counter()
            q_vec  = pipeline.embedder.encode([query], normalize_embeddings=True)[0]
            raw    = pipeline.hybrid_retriever.search(query, q_vec, plan.top_k, plan.alpha)
            ranked = pipeline.reranker.rerank(query, raw, plan.top_k)
            answer = pipeline.generator.generate_stream(query, ranked, analysis.query_type, on_token)
            answer_placeholder.markdown(answer)

            # Record manually for streaming path
            elapsed = time.perf_counter() - t0
            pipeline.feedback_loop.record(
                query=query, top_k=plan.top_k, alpha=plan.alpha,
                retrieval_time=0.01, generation_time=elapsed,
                answer=answer, retrieved_chunks=[c for c, _ in ranked],
            )
            if pipeline.cfg.enable_cache:
                pipeline.cache.put(query, answer, [c for c, _ in ranked])

        else:
            result = pipeline.query(query)
            answer = result.answer
            ranked = result.retrieved_chunks
            answer_placeholder.markdown(answer)

    # ── Retrieved chunks ───────────────────────────────────────────────────
    st.markdown("### 📄 Retrieved Chunks")
    if 'ranked' in dir() and ranked:
        for i, (chunk, score) in enumerate(ranked[:5], 1):
            with st.expander(f"Chunk {i} — {chunk.source} (score={score:.3f})", expanded=(i == 1)):
                st.markdown(f"""<div class="chunk-card">{chunk.text}</div>""", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.caption(f"📁 Source: `{chunk.source}`")
                c2.caption(f"🔢 Chunk ID: {chunk.chunk_id}")
                c3.caption(f"📊 Score: {score:.4f}")

    st.divider()

# ── History & Analytics ────────────────────────────────────────────────────
if pipeline.history:
    st.markdown("### 📊 Query History & Adaptive Behaviour")

    tab1, tab2, tab3, tab4 = st.tabs(["Latency", "Adaptive K", "Model Routing", "Cache"])

    history = pipeline.history

    # ── Tab 1: Latency breakdown ───────────────────────────────────────────
    with tab1:
        df = pd.DataFrame([{
            "Query": r.query[:40] + "…",
            "Retrieval (s)": round(r.retrieval_time, 3),
            "Generation (s)": round(r.generation_time, 3),
            "Total (s)": round(r.total_time, 3),
            "Cached": r.from_cache,
        } for r in history])

        non_cached = df[~df["Cached"]]
        if not non_cached.empty:
            st.bar_chart(non_cached.set_index("Query")[["Retrieval (s)", "Generation (s)"]])

        report = pipeline.performance_report()
        if report:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P50 Latency", f"{report.get('p50_latency', 0):.2f}s")
            m2.metric("P95 Latency", f"{report.get('p95_latency', 0):.2f}s")
            m3.metric("P50 Retrieval", f"{report.get('p50_ret_time', 0):.3f}s")
            m4.metric("P50 Generation", f"{report.get('p50_gen_time', 0):.2f}s")

    # ── Tab 2: Adaptive K ──────────────────────────────────────────────────
    with tab2:
        k_data = pd.DataFrame([{
            "Query #": i + 1,
            "Query": r.query[:30] + "…",
            "K Used": r.plan.top_k,
            "Complexity": round(r.analysis.complexity_score, 2),
            "Type": r.analysis.query_type,
        } for i, r in enumerate(history)])

        st.line_chart(k_data.set_index("Query #")[["K Used", "Complexity"]])
        st.dataframe(k_data, use_container_width=True, hide_index=True)

    # ── Tab 3: Model routing ───────────────────────────────────────────────
    with tab3:
        model_data = pd.DataFrame([{
            "Query": r.query[:40] + "…",
            "Type": r.analysis.query_type,
            "Model": r.model_used,
            "Gen Time (s)": round(r.generation_time, 2),
            "Score": round(r.analysis.complexity_score, 2),
        } for r in history])

        st.dataframe(model_data, use_container_width=True, hide_index=True)

        # Count model usage
        model_counts = model_data["Model"].value_counts()
        if len(model_counts) > 0:
            st.bar_chart(model_counts)

    # ── Tab 4: Cache stats ─────────────────────────────────────────────────
    with tab4:
        report = pipeline.performance_report()
        ec = report.get("cache", {})
        sc = report.get("semantic_cache", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Exact LRU Cache**")
            st.metric("Hits",     ec.get("hits", 0))
            st.metric("Misses",   ec.get("misses", 0))
            st.metric("Hit Rate", f"{ec.get('hit_rate', 0):.1%}")
            st.metric("Size",     ec.get("size", 0))

        with c2:
            st.markdown("**Semantic Cache**")
            st.metric("Hits",      sc.get("hits", 0))
            st.metric("Misses",    sc.get("misses", 0))
            st.metric("Hit Rate",  f"{sc.get('hit_rate', 0):.1%}")
            st.metric("Threshold", f"{sc.get('threshold', 0):.2f}")

# ── Feedback loop state ────────────────────────────────────────────────────
if pipeline.feedback_loop.records:
    with st.expander("🔄 Feedback Loop State"):
        stats = pipeline.feedback_loop.stats
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("EMA Latency",  f"{stats.get('p50_latency',  0):.2f}s")
        s2.metric("EMA Quality",  f"{stats.get('avg_quality',  0):.2f}")
        s3.metric("Avg Ret Time", f"{stats.get('avg_ret_time', 0):.3f}s")
        s4.metric("Avg Gen Time", f"{stats.get('avg_gen_time', 0):.2f}s")
        st.caption("Feedback fires every 3 queries and nudges top-K and alpha based on EMA latency and quality.")
