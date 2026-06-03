"""
config.py — All tuneable parameters in one place.
"""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Chunking ---
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- Retrieval ---
    default_top_k: int = 5
    min_top_k: int = 2
    max_top_k: int = 12
    default_alpha: float = 0.7

    # --- Adaptive thresholds ---
    short_query_words: int = 4
    complex_query_words: int = 12
    high_latency_threshold: float = 2.5
    low_quality_threshold: float = 0.35

    # --- Feedback ---
    ema_alpha: float = 0.25
    k_bump: int = 2
    alpha_bump: float = 0.1

    # --- Exact LRU cache ---
    enable_cache: bool = True
    cache_max_size: int = 256

    # --- Semantic cache (BONUS) ---
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.85

    # --- Model routing (BONUS) ---
    small_model: str = "llama3.2:3b"    # simple / moderate queries
    large_model: str = "qwen2.5:7b"     # complex queries
    enable_model_routing: bool = True
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = 1024
    temperature: float = 0.3

    # --- Query decomposition (BONUS) ---
    enable_decomposition: bool = True
    decomposition_complexity_threshold: float = 0.65

    # --- Confidence-aware fallback (BONUS) ---
    enable_fallback: bool = True
