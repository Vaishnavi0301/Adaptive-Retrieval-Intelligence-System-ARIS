"""
main.py - python main.py
Shows: parallel decomposition, semantic cache, fallback, model routing.
"""
import os, sys
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate
colorama_init(autoreset=True)

from config import RAGConfig
from pipeline import AdaptiveRAGPipeline

QUERIES = [
    "What is supervised learning?",
    "What is FAISS?",
    "What is ReLU?",
    "How does BM25 differ from vector search?",
    "What are common techniques to prevent overfitting?",
    "How does batch normalisation help neural network training?",
    (
        "Compare dense retrieval and sparse retrieval in RAG systems, "
        "including their strengths, weaknesses, and how hybrid retrieval combines them."
    ),
    (
        "Explain how the Transformer architecture works and why it replaced "
        "RNNs for sequence modelling tasks."
    ),
]

def sep(title=""):
    print(f"\n{Fore.CYAN}{'─'*65}")
    if title: print(f"  {title}")
    print(f"{'─'*65}{Style.RESET_ALL}")

def print_result(result, idx):
    cache_tag = ""
    if result.from_cache:
        cache_tag = f" {Fore.GREEN}[{result.cache_type.upper()} CACHE HIT]{Style.RESET_ALL}"
    print(f"\n{Fore.YELLOW}[Q{idx+1}]{Style.RESET_ALL} {result.query[:80]}{cache_tag}")
    print(f"  {Fore.BLUE}Plan    :{Style.RESET_ALL} {result.plan.notes}")
    print(f"  {Fore.BLUE}Type    :{Style.RESET_ALL} {result.analysis.query_type} "
          f"(score={result.analysis.complexity_score:.2f})")
    print(f"  {Fore.MAGENTA}Model   :{Style.RESET_ALL} {result.model_used}")
    if result.sub_questions:
        print(f"  {Fore.MAGENTA}Decomposed (parallel) into {len(result.sub_questions)} sub-questions{Style.RESET_ALL}")
    if result.fallback_triggered:
        print(f"  {Fore.RED}Fallback triggered (low quality → retried with K={result.plan.top_k}){Style.RESET_ALL}")
    print(f"  {Fore.BLUE}Timing  :{Style.RESET_ALL} "
          f"retrieval={result.retrieval_time:.2f}s  "
          f"generation={result.generation_time:.2f}s  "
          f"total={result.total_time:.2f}s")
    print(f"\n  {Fore.WHITE}Answer:{Style.RESET_ALL}")
    for line in result.answer.split("\n"):
        print(f"    {line}")

def print_report(pipeline):
    sep("Performance Report")
    report = pipeline.performance_report()
    if not report: return

    timing = [
        ["P50 Total Latency",   f"{report.get('p50_latency',  0):.3f}s"],
        ["P95 Total Latency",   f"{report.get('p95_latency',  0):.3f}s"],
        ["P50 Retrieval Time",  f"{report.get('p50_ret_time', 0):.3f}s"],
        ["P95 Retrieval Time",  f"{report.get('p95_ret_time', 0):.3f}s"],
        ["P50 Generation Time", f"{report.get('p50_gen_time', 0):.3f}s"],
        ["P95 Generation Time", f"{report.get('p95_gen_time', 0):.3f}s"],
    ]
    print(tabulate(timing, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    adaptive = [
        ["Queries run",    report.get("n_queries",  0)],
        ["Avg quality",    f"{report.get('avg_quality', 0):.3f}"],
        ["Avg top-K",      f"{report.get('avg_top_k',  0):.1f}"],
        ["Avg alpha",      f"{report.get('avg_alpha',  0):.3f}"],
    ]
    print(tabulate(adaptive, headers=["Adaptive Metric", "Value"], tablefmt="rounded_outline"))

    ec = report.get("cache", {})
    sc = report.get("semantic_cache", {})
    cache_rows = [
        ["Exact LRU — hits",   ec.get("hits",  0)],
        ["Exact LRU — rate",   f"{ec.get('hit_rate', 0):.1%}"],
        ["Semantic — hits",    sc.get("hits",  0)],
        ["Semantic — rate",    f"{sc.get('hit_rate', 0):.1%}"],
        ["Semantic threshold", f"{sc.get('threshold', 0):.2f}"],
    ]
    print(tabulate(cache_rows, headers=["Cache", "Value"], tablefmt="rounded_outline"))

    sep("Model Routing Summary")
    print(f"  simple / moderate  →  {Fore.GREEN}llama3.2:3b{Style.RESET_ALL}  (fast)")
    print(f"  complex            →  {Fore.YELLOW}qwen2.5:7b{Style.RESET_ALL}   (accurate)")
    print(f"\n  {Fore.CYAN}Streaming:{Style.RESET_ALL} enabled in dashboard (streamlit run dashboard.py)")
    print(f"  {Fore.CYAN}ANN bench:{Style.RESET_ALL} python retrieval/ann_experiments.py")


def main():
    sep("Adaptive RAG System  —  All Features")

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
    pipeline.ingest_directory(os.path.join(os.path.dirname(__file__), "data"))

    sep("Running Queries (parallel decomposition enabled)")
    for i, q in enumerate(QUERIES):
        try:
            result = pipeline.query(q)
            print_result(result, i)
        except Exception as e:
            print(f"{Fore.RED}[Q{i+1}] ERROR: {e}{Style.RESET_ALL}")
            import traceback; traceback.print_exc()

    # Semantic cache demo
    sep("Semantic Cache Demo")
    paraphrase = "Explain what machine learning supervised training is"
    print(f"Original: '{QUERIES[0]}'")
    print(f"Paraphrase: '{paraphrase}'")
    result = pipeline.query(paraphrase)
    print_result(result, 98)

    # Exact cache demo
    sep("Exact Cache Demo")
    result = pipeline.query(QUERIES[0])
    print_result(result, 99)

    print_report(pipeline)

if __name__ == "__main__":
    main()
