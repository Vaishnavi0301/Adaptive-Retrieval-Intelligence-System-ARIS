"""
retrieval/ann_experiments.py
-----------------------------
ANN (Approximate Nearest Neighbour) tuning experiments.

Compares four FAISS index types:
  1. IndexFlatIP     — exact search (ground truth, baseline)
  2. IndexIVFFlat    — Inverted File Index, nlist clusters
  3. IndexHNSWFlat   — Hierarchical Navigable Small World graph
  4. IndexIVFPQ      — IVF + Product Quantisation (compressed vectors)

Metrics measured per index:
  - Build time      (seconds)
  - Query latency   P50 / P95 over N_QUERIES random queries
  - Recall@K        fraction of exact top-K results returned

Run standalone:
  python retrieval/ann_experiments.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import faiss
from tabulate import tabulate
from typing import List, Tuple, Dict

# ── Experiment settings ────────────────────────────────────────────────────
N_VECTORS   = 10_000    # corpus size to benchmark on
DIM         = 384       # must match embedding dim
N_QUERIES   = 200       # number of random queries to time
TOP_K       = 5         # recall@K

# IVF settings
NLIST       = 100       # number of Voronoi cells (rule of thumb: sqrt(N))
NPROBE_VALUES = [1, 5, 10, 20, 50]  # nprobe tuning sweep

# HNSW settings
HNSW_M      = 32        # connections per node (higher = better recall, more RAM)

# PQ settings
PQ_M        = 32        # number of sub-quantisers (must divide DIM evenly)
PQ_NBITS    = 8         # bits per sub-quantiser


def make_data(n: int, dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2-normalise (cosine similarity via inner product)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def exact_top_k(index_flat: faiss.IndexFlatIP, queries: np.ndarray, k: int) -> np.ndarray:
    _, I = index_flat.search(queries, k)
    return I  # shape (n_queries, k)


def recall_at_k(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    hits = 0
    total = approx_ids.shape[0] * approx_ids.shape[1]
    for i in range(approx_ids.shape[0]):
        hits += len(set(approx_ids[i]) & set(exact_ids[i]))
    return hits / total


def measure_latency(index, queries: np.ndarray, k: int) -> Tuple[float, float]:
    """Returns (P50_ms, P95_ms)."""
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))


# ── Index builders ─────────────────────────────────────────────────────────

def build_flat(corpus: np.ndarray) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(DIM)
    idx.add(corpus)
    return idx


def build_ivf(corpus: np.ndarray, nlist: int, nprobe: int) -> faiss.IndexIVFFlat:
    quantiser = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFFlat(quantiser, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(corpus)
    idx.add(corpus)
    idx.nprobe = nprobe
    return idx


def build_hnsw(corpus: np.ndarray, M: int) -> faiss.IndexHNSWFlat:
    idx = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    idx.add(corpus)
    return idx


def build_ivfpq(corpus: np.ndarray, nlist: int, M: int, nbits: int) -> faiss.IndexIVFPQ:
    quantiser = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFPQ(quantiser, DIM, nlist, M, nbits)
    idx.train(corpus)
    idx.add(corpus)
    idx.nprobe = 10
    return idx


# ── Main experiment ────────────────────────────────────────────────────────

def run_experiments():
    print(f"\n{'='*60}")
    print(f"  ANN Tuning Experiments")
    print(f"  corpus={N_VECTORS} vectors | dim={DIM} | queries={N_QUERIES} | K={TOP_K}")
    print(f"{'='*60}\n")

    corpus  = make_data(N_VECTORS, DIM, seed=0)
    queries = make_data(N_QUERIES, DIM, seed=1)

    # Ground truth from exact search
    flat_idx   = build_flat(corpus)
    exact_ids  = exact_top_k(flat_idx, queries, TOP_K)
    p50_flat, p95_flat = measure_latency(flat_idx, queries, TOP_K)

    rows = []

    # 1. Flat (exact baseline)
    rows.append([
        "IndexFlatIP (exact)",
        "—",
        f"{p50_flat:.3f}",
        f"{p95_flat:.3f}",
        "100.0%",
        "0 (exact)",
    ])

    # 2. IVF with nprobe sweep
    for nprobe in NPROBE_VALUES:
        t0  = time.perf_counter()
        idx = build_ivf(corpus, NLIST, nprobe)
        build_ms = (time.perf_counter() - t0) * 1000
        _, I = idx.search(queries, TOP_K)
        p50, p95 = measure_latency(idx, queries, TOP_K)
        rec = recall_at_k(I, exact_ids)
        rows.append([
            f"IndexIVFFlat",
            f"nprobe={nprobe}",
            f"{p50:.3f}",
            f"{p95:.3f}",
            f"{rec*100:.1f}%",
            f"{build_ms:.0f}ms build",
        ])

    # 3. HNSW
    t0  = time.perf_counter()
    idx = build_hnsw(corpus, HNSW_M)
    build_ms = (time.perf_counter() - t0) * 1000
    _, I = idx.search(queries, TOP_K)
    p50, p95 = measure_latency(idx, queries, TOP_K)
    rec = recall_at_k(I, exact_ids)
    rows.append([
        f"IndexHNSWFlat",
        f"M={HNSW_M}",
        f"{p50:.3f}",
        f"{p95:.3f}",
        f"{rec*100:.1f}%",
        f"{build_ms:.0f}ms build",
    ])

    # 4. IVF + PQ (compressed)
    t0  = time.perf_counter()
    idx = build_ivfpq(corpus, NLIST, PQ_M, PQ_NBITS)
    build_ms = (time.perf_counter() - t0) * 1000
    _, I = idx.search(queries, TOP_K)
    p50, p95 = measure_latency(idx, queries, TOP_K)
    rec = recall_at_k(I, exact_ids)
    rows.append([
        f"IndexIVFPQ",
        f"M={PQ_M},bits={PQ_NBITS}",
        f"{p50:.3f}",
        f"{p95:.3f}",
        f"{rec*100:.1f}%",
        f"{build_ms:.0f}ms build",
    ])

    headers = ["Index Type", "Settings", "P50 (ms)", "P95 (ms)", f"Recall@{TOP_K}", "Notes"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    print("""
Findings:
  IndexFlatIP  — exact, slowest at scale, best for small corpora (<100k)
  IndexIVFFlat — fast with nprobe tuning; nprobe=10 gives good recall/speed tradeoff
  IndexHNSWFlat— fastest queries, high recall, but higher RAM and build time
  IndexIVFPQ   — smallest memory footprint (compressed vectors), lower recall

Recommendation for this project (26 chunks):
  IndexFlatIP is fine — corpus is tiny.
  At 100k+ chunks: use HNSW for best latency with acceptable recall.
  At 1M+ chunks:   use IVFPQ to fit in RAM.
""")


if __name__ == "__main__":
    run_experiments()
