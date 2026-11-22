# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers>=6.1,<7",
#   "scikit-learn>=1.8,<2",
#   "numpy>=2.4,<3",
#   "matplotlib>=3.10,<4",
#   "httpx>=0.28,<1",
#   "rank-bm25==0.2.2",
#   "qdrant-client>=1.15,<2",
# ]
# [tool.uv]
# exclude-newer = "2026-10-27T00:00:00Z"
# ///
"""
Homework: Ticket Auto-Categorization with Embeddings

Pipeline:
  embed tickets → k-means clustering → LLM names clusters
  → compare BM25 / cosine / Qdrant → Ticket Helper

Your tasks (5 TODOs + 1 bonus in this file):
  1) get_embeddings(texts)               — sentence-transformers
  2) cluster_tickets(embeddings, n)      — k-means
  3) name_cluster(tickets)               — write the LLM prompt
  4) build_bm25_index(corpus) + bm25_search(index, query, top_k)
                                         — rank_bm25, index built once
  5) upload_to_qdrant(tickets, embeddings) + qdrant_search(client, query_emb, top_k)
                                         — Qdrant :memory: (no Docker needed!)
  BONUS) rerank(query, candidates, model_key) — cross-encoder; evaluated together with
         cosine / BM25 / Fusion on 90 labeled queries (data/queries.json)

Already implemented (read code + docstrings to understand):
  - cosine_search    — brute-force numpy baseline for comparison
  - fusion_search    — Reciprocal Rank Fusion (BM25 + cosine)
  - compute_ari      — clustering quality vs ground-truth categories
  - Ticket Helper, t-SNE visualization, comparison table

Setup:
  export OPENROUTER_API_KEY="sk-or-..."   # for TODO 3 (LLM cluster naming)

Run:
  uv run starter.py --quick               # skip LLM naming; runs demo queries
  uv run starter.py                       # full pipeline with LLM cluster naming
  uv run starter.py --query "can't log in"
  uv run starter.py --quick --rerank      # bonus: hybrid + rerank on 90 labeled queries

Validate:
  uv run eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import httpx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL_LLM          = "google/gemini-3.5-flash-lite"
EMBED_MODEL        = "all-MiniLM-L6-v2"
DEFAULT_N_CLUSTERS = 5      # deliberately suboptimal — your job is to find the best k!
TOP_K              = 5
COLLECTION         = "tickets"
RERANK_CANDIDATES  = 20     # bonus: rerank the cosine top-20, keep the top-5

# Three reranker models for the bonus comparison.
# They are ordered from weakest (fast) to strongest (best quality):
#   ms-marco  — trained on passage-level Q&A; fast on CPU but poor on short texts
#   bge-base  — English + Chinese reranker; better for short documents
#   bge-v2-m3 — multilingual (100+ languages), the strongest of the three
RERANKER_MODELS = {
    "ms-marco":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base":  "BAAI/bge-reranker-base",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}

# Five queries → expected category (used to print category precision@5 after each query).
DEMO_QUERIES: dict[str, str] = {
    "my laptop screen is broken":              "technical",
    "I can't authenticate my identity":        "account",
    "money problems with my purchase":         "billing",
    "package not delivered to my address":     "shipping",
    "want to send the item back for a refund": "returns",
}

# ---------------------------------------------------------------------------
# DATASET — load from data/tickets.json (300 synthetic support tickets)
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent / "data" / "tickets.json"
TICKETS: list[dict] = json.loads(_DATA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PRE-BUILT: MODEL LOADERS (load once, reuse)
# ---------------------------------------------------------------------------
# Loading a model takes seconds, encoding one query takes milliseconds — so models are
# created once here and reused by every call below.

@lru_cache(maxsize=None)
def load_embedder():
    """Return the shared SentenceTransformer (loaded on first call, cached afterwards)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=None)
def load_cross_encoder(model_key: str):
    """Return a cached CrossEncoder for one of RERANKER_MODELS."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODELS[model_key])


# ---------------------------------------------------------------------------
# TODO 1: GET EMBEDDINGS
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.

    Use the pre-built load_embedder() (model 'all-MiniLM-L6-v2') and return a
    numpy array of shape (len(texts), 384), not a tensor.

    Don't create SentenceTransformer inside this function: it runs for every
    query, and loading the model takes seconds.

    Hint: the model has one method that turns a batch of texts into vectors;
    check its parameters for the one that controls the output type.
    """
    # TODO: implement
    raise NotImplementedError("TODO 1: implement get_embeddings")


# ---------------------------------------------------------------------------
# TODO 2: CLUSTER TICKETS
# ---------------------------------------------------------------------------

def cluster_tickets(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster embeddings using K-Means (sklearn).

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        n_clusters: number of clusters

    Returns:
        numpy array of cluster labels, shape (n_samples,)

    Use random_state=42 for reproducibility.

    After implementing, every run prints an "ARI by k" block (k = 3–12) using this
    function. Choose k and justify your choice in results.md.

    Hint: sklearn.cluster.KMeans (already imported). k-means depends on the
    random starting centroids, so run several initialisations (n_init) and let
    sklearn keep the best one; fit and get labels in one call.
    """
    # TODO: implement
    raise NotImplementedError("TODO 2: implement cluster_tickets")


# ---------------------------------------------------------------------------
# TODO 3: NAME CLUSTER VIA LLM
# ---------------------------------------------------------------------------

def name_cluster(tickets: list[str]) -> str:
    """
    Given a list of ticket texts from the same cluster, ask an LLM
    to produce a short category name (2-4 words).

    Your job: write the `prompt` variable below.
    The LLM call and response parsing are already done: a plain HTTP POST to
    OpenRouter's OpenAI-compatible /chat/completions endpoint, no SDK needed.

    The prompt should:
      - include every ticket from `tickets` (one per line reads best);
      - ask for a short name (2-4 words) and nothing else, so the reply can be
        used as a label without parsing;
      - NOT list the categories: clustering is unsupervised, the LLM has to infer
        the topic from the tickets.
    """
    prompt = ""  # <-- replace with your prompt that includes the tickets

    if not prompt.strip():
        raise NotImplementedError(
            "TODO 3: write a prompt — replace the empty string above with a "
            "prompt that includes the ticket texts and asks for a cluster name."
        )

    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": MODEL_LLM,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


# ---------------------------------------------------------------------------
# PRE-BUILT: COSINE SIMILARITY SEARCH (brute-force numpy baseline)
# ---------------------------------------------------------------------------

def cosine_search(
    query_emb: np.ndarray,
    all_embs: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Brute-force cosine similarity search (pre-built — keep for comparison).

    Cosine similarity: cos(a, b) = (a · b) / (||a|| * ||b||)
    Normalized vectors: cosine similarity = dot product.

    Algorithm:
        1. Normalize query to unit length.
        2. Normalize all embeddings row-wise.
        3. Matrix-multiply → all cosine similarities in one step.
        4. Return top_k (index, score) pairs sorted descending.

    This is O(n) per query — fine for hundreds of docs, slow for millions.
    A Qdrant server replaces this with an HNSW index (approximate search).
    Qdrant :memory: (TODO 5) is still a full scan — same API, test scale only.
    """
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    all_norms  = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-10)
    sims       = all_norms @ query_norm
    top_idx    = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# TODO 4: BM25 KEYWORD SEARCH (build the index once, query many times)
# ---------------------------------------------------------------------------

def build_bm25_index(corpus: list[str]) -> "BM25Okapi":  # type: ignore[name-defined]
    """
    TODO 4a: build a BM25 index over the corpus (Okapi BM25, rank_bm25 library).

    BM25 ranks documents by term frequency and inverse document frequency.
    It matches EXACT word tokens — "authenticate" will NOT find "login".
    This difference from cosine is exactly what makes the comparison interesting.

    Build the index ONCE and reuse it for every query, like the Qdrant collection
    in TODO 5. Rebuilding it per query makes BM25 look slow in the latency table.

    Steps:
        1. Tokenize every document: lowercase + split on spaces.
        2. Build rank_bm25.BM25Okapi from the list of token lists and return it.
    """
    # TODO: implement
    raise NotImplementedError("TODO 4a: implement build_bm25_index")


def bm25_search(
    index: "BM25Okapi",  # type: ignore[name-defined]
    query: str,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    TODO 4b: score the query against the index and return the top_k documents.

    Steps:
        1. Tokenize the query EXACTLY like the corpus (lowercase + split).
        2. index.get_scores(tokens) returns one score per document, in corpus order.
        3. Return the top_k (index, score) pairs sorted by score descending,
           as plain int / float (cosine_search does the same sort).
    """
    # TODO: implement
    raise NotImplementedError("TODO 4b: implement bm25_search")


# ---------------------------------------------------------------------------
# TODO 5: QDRANT IN-MEMORY SEARCH
# ---------------------------------------------------------------------------

def upload_to_qdrant(
    tickets: list[dict],
    embeddings: np.ndarray,
) -> "QdrantClient":  # type: ignore[name-defined]
    """
    Upload all tickets to an in-memory Qdrant collection and return the client.

    KEY INSIGHT: QdrantClient(":memory:") runs fully in-process — no Docker,
    no server, no setup required. The API is the same as Qdrant server, so the
    code moves to production by changing one line: QdrantClient(url=...).
    But local mode is a brute-force scan without an HNSW index, meant for tests
    and prototypes (the client warns above 20,000 points).

    Steps:
        a) Create an in-process client with QdrantClient(":memory:").
        b) Create the collection COLLECTION. Its VectorParams need the vector size
           (take it from `embeddings`, don't hardcode 384) and cosine distance.
        c) One PointStruct per ticket: id = the ticket's index in `tickets`
           (qdrant_search returns these ids and the script looks up TICKETS[id]),
           the vector as a plain list, payload with the text and category.
        d) Upload all points in ONE upsert call, not one call per point.
        e) Return the client.

    Imports you'll need:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

    Returns:
        QdrantClient — pass this to qdrant_search() for querying.
    """
    # TODO: implement
    raise NotImplementedError("TODO 5a: implement upload_to_qdrant")


def qdrant_search(
    client: "QdrantClient",  # type: ignore[name-defined]
    query_emb: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Search the in-memory Qdrant collection.

    Steps:
        Call client.query_points() on COLLECTION with the query vector as a list
        and limit=top_k. The response has a .points list; each point has .id and
        .score. Return them as (id, score) tuples.

    Returns:
        List of (index, score) tuples, sorted by score descending (Qdrant does this).

    Compare the results with cosine_search — they should be identical:
    Qdrant :memory: does the same exact cosine scan. Speed and approximation
    differences only appear with a Qdrant server and its HNSW index.
    """
    # TODO: implement
    raise NotImplementedError("TODO 5b: implement qdrant_search")


# ---------------------------------------------------------------------------
# BONUS: RERANKER COMPARISON
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidate_texts: list[str],
    model_key: str = "bge-v2-m3",
) -> list[tuple[str, float]]:
    """
    BONUS: Re-rank candidates using a cross-encoder model.

    A cross-encoder receives (query, document) as a SINGLE input and
    produces one relevance score per pair — it can see attention between
    all query and document tokens, making it more accurate than cosine.
    Trade-off: requires one forward pass per pair → O(n) at query time.

    candidate_texts is the cosine top-RERANK_CANDIDATES (20): the reranker can only
    promote documents that retrieval already found, so the first stage needs depth.

    Get the model from the pre-built load_cross_encoder(model_key) (a cached
    sentence-transformers CrossEncoder). Score all (query, text) pairs in ONE
    predict() call, not one call per candidate, then return every candidate
    with its score, best first.

    Args:
        query:           the search query
        candidate_texts: list of document strings to re-rank
        model_key:       one of "ms-marco", "bge-base", "bge-v2-m3"

    Returns:
        List of (text, score) tuples sorted by relevance score descending.
    """
    # TODO (bonus): implement
    raise NotImplementedError(
        f"BONUS: implement rerank() using load_cross_encoder({model_key!r})"
    )


# ---------------------------------------------------------------------------
# PRE-BUILT: RECIPROCAL RANK FUSION (RRF)
# ---------------------------------------------------------------------------

def fusion_search(
    bm25_results:   list[tuple[int, float]],
    cosine_results: list[tuple[int, float]],
    top_k: int  = 5,
    rrf_k:  int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (pre-built): combine BM25 + cosine rankings.

    score(doc) = Σ  1 / (rrf_k + rank_in_method)
    Documents ranked high by BOTH methods get the highest combined score.
    rrf_k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (idx, _) in enumerate(cosine_results, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, float(s)) for idx, s in sorted_ids]


# ---------------------------------------------------------------------------
# PRE-BUILT: CLUSTERING QUALITY — ARI
# ---------------------------------------------------------------------------

def compute_ari(tickets: list[dict], labels: np.ndarray) -> float:
    """
    Adjusted Rand Index: measures how well clusters match ground-truth categories.

    ARI = 0.0  → clustering is no better than random
    ARI = 1.0  → perfect match with the 7 categories
    ARI < 0    → worse than random (shouldn't happen with reasonable k-means)

    Try different n_clusters (5–10) and observe how ARI changes!
    The 7 ground-truth categories are in tickets[i]["category"].
    """
    true_labels  = [t["category"] for t in tickets]
    return adjusted_rand_score(true_labels, labels)


ARI_SWEEP_K = (3, 4, 5, 6, 7, 8, 9, 10, 12)


def print_ari_sweep(tickets: list[dict], embeddings: np.ndarray) -> None:
    """ARI for every k in ARI_SWEEP_K with your cluster_tickets — one run instead of nine."""
    print("\n  ARI by k (same embeddings, your cluster_tickets):")
    for k in ARI_SWEEP_K:
        print(f"    k={k:<3} ARI = {compute_ari(tickets, cluster_tickets(embeddings, k)):.3f}")
    print("  ↑ Copy this block into results.md Section 1")


# ---------------------------------------------------------------------------
# PRE-BUILT: VISUALIZATION
# ---------------------------------------------------------------------------

def visualize_clusters(
    embeddings:    np.ndarray,
    labels:        np.ndarray,
    cluster_names: dict[int, str] | None = None,
) -> None:
    """Reduce to 2D with t-SNE and plot colored by cluster."""
    print("\n  Running t-SNE for visualization...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(11, 7))
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))

    for label in unique_labels:
        mask = labels == label
        name = cluster_names.get(label, f"Cluster {label}") if cluster_names else f"Cluster {label}"
        plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(label)], label=name, s=45, alpha=0.75)

    plt.legend(fontsize=8, loc="best")
    plt.title("Ticket Clusters — t-SNE (300 tickets)", fontsize=13)
    plt.tight_layout()
    plt.savefig("clusters.png", dpi=150)
    print("  Saved: clusters.png")
    plt.close()


# ---------------------------------------------------------------------------
# PRE-BUILT: COMPARISON TABLE
# ---------------------------------------------------------------------------

def compare_and_print(
    query:             str,
    query_emb:         np.ndarray,
    all_embs:          np.ndarray,
    tickets:           list[dict],
    qdrant_client:     object | None = None,
    expected_category: str | None = None,
    bm25_index:        object | None = None,
) -> dict:
    """Run BM25, cosine, fusion (and optionally Qdrant); print timing, overlap, category precision@5.

    Returns dict of {method_key: {"recall": int|None, "latency_ms": float}}
    for use in the summary table.
    """
    print(f"\n  Query: \"{query}\"")
    print("=" * 72)

    bm25_results: list[tuple[int, float]] | None = None
    t_bm25 = 0.0
    if bm25_index is None:
        print("  [BM25] Not implemented (TODO 4a)")
    else:
        try:
            t0 = time.perf_counter()
            bm25_results = bm25_search(bm25_index, query, TOP_K)
            t_bm25 = (time.perf_counter() - t0) * 1000
        except NotImplementedError:
            print("  [BM25] Not implemented (TODO 4b)")

    t0 = time.perf_counter()
    cosine_results = cosine_search(query_emb, all_embs, TOP_K)
    t_cosine = (time.perf_counter() - t0) * 1000

    fusion_results: list[tuple[int, float]] | None = None
    t_fusion = 0.0
    if bm25_results is not None:
        t0 = time.perf_counter()
        fusion_results = fusion_search(bm25_results, cosine_results, TOP_K)
        t_fusion = t_bm25 + t_cosine + (time.perf_counter() - t0) * 1000  # end-to-end: both searches + merge

    qdrant_results: list[tuple[int, float]] | None = None
    t_qdrant = 0.0
    if qdrant_client is not None:
        try:
            t0 = time.perf_counter()
            qdrant_results = qdrant_search(qdrant_client, query_emb, TOP_K)
            t_qdrant = (time.perf_counter() - t0) * 1000
        except NotImplementedError:
            print("  [Qdrant] Not implemented (TODO 5)")

    methods: list[tuple[str, list[tuple[int, float]], float]] = [
        ("Cosine numpy (pre-built)", cosine_results, t_cosine),
    ]
    if bm25_results is not None:
        methods.insert(0, ("BM25 keyword", bm25_results, t_bm25))
    if fusion_results is not None:
        methods.append(("Fusion RRF (BM25 + cosine + merge)", fusion_results, t_fusion))
    if qdrant_results is not None:
        methods.append(("Qdrant :memory:", qdrant_results, t_qdrant))

    for name, results, elapsed in methods:
        print(f"\n  {name} ({elapsed:.2f} ms):")
        for rank, (idx, score) in enumerate(results, 1):
            cat  = tickets[idx]["category"]
            text = tickets[idx]["text"][:58]
            print(f"    {rank}. [{cat}] {text}... ({score:.4f})")

    if bm25_results and cosine_results:
        bm25_ids   = {idx for idx, _ in bm25_results}
        cosine_ids = {idx for idx, _ in cosine_results}
        print(f"\n  BM25 vs Cosine overlap: {len(bm25_ids & cosine_ids)}/{TOP_K}")
    if qdrant_results and cosine_results:
        q_ids = {idx for idx, _ in qdrant_results}
        c_ids = {idx for idx, _ in cosine_results}
        print(f"  Qdrant vs Cosine overlap: {len(q_ids & c_ids)}/{TOP_K}  (should be 5/5 at this size)")

    def _recall(results: list[tuple[int, float]] | None) -> int | None:
        if results is None or not expected_category:
            return None
        return sum(tickets[idx]["category"] == expected_category for idx, _ in results)

    metrics: dict = {
        "BM25":   {"recall": _recall(bm25_results),   "latency_ms": t_bm25},
        "Cosine": {"recall": _recall(cosine_results),  "latency_ms": t_cosine},
        "Fusion": {"recall": _recall(fusion_results),  "latency_ms": t_fusion},
        "Qdrant": {"recall": _recall(qdrant_results),  "latency_ms": t_qdrant},
    }

    if expected_category:
        parts = []
        for key in ("BM25", "Cosine", "Fusion", "Qdrant"):
            r = metrics[key]["recall"]
            if r is not None:
                parts.append(f"{key}={r}/{TOP_K}")
        print(f"\n  Category P@5 [{expected_category}]: {' | '.join(parts)}")
    print("=" * 72)
    return metrics


# ---------------------------------------------------------------------------
# PRE-BUILT: LEAVE-ONE-OUT RETRIEVAL EVAL (every ticket is a query)
# ---------------------------------------------------------------------------
# Five demo queries show HOW each method behaves; they are too few to say which one is
# better. Here every ticket becomes a query and "relevant" = any OTHER ticket of the same
# category. 300 queries, no labeling work, and a 95% bootstrap interval for every number.

def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean (resample queries with replacement)."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_retrieval_loo(
    tickets:       list[dict],
    embeddings:    np.ndarray,
    qdrant_client: object | None = None,
    top_k:         int = TOP_K,
    bm25_index:    object | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Run every method with each ticket as the query; drop the ticket itself from its results.

    Returns {method: {"p": [precision@k per query], "rr": [reciprocal rank of the first
    same-category hit within top-k, 0 if none]}}.
    """
    corpus = [t["text"] for t in tickets]
    cats   = [t["category"] for t in tickets]
    depth  = 20  # candidates per method before fusion / self-removal
    per: dict[str, dict[str, list[float]]] = {}

    def record(method: str, query_idx: int, results: list[tuple[int, float]]) -> None:
        ids = [idx for idx, _ in results if idx != query_idx][:top_k]
        rel = [cats[idx] == cats[query_idx] for idx in ids]
        per.setdefault(method, {"p": [], "rr": []})
        per[method]["p"].append(sum(rel) / top_k)
        per[method]["rr"].append(next((1.0 / (rank + 1) for rank, ok in enumerate(rel) if ok), 0.0))

    bm25_ready = bm25_index is not None
    for i in range(len(tickets)):
        cos = cosine_search(embeddings[i], embeddings, depth + 1)
        record("Cosine", i, cos)
        if bm25_ready:
            try:
                bm = bm25_search(bm25_index, corpus[i], depth + 1)
            except NotImplementedError:
                bm25_ready = False
            else:
                record("BM25", i, bm)
                fused = fusion_search([r for r in bm if r[0] != i], [r for r in cos if r[0] != i], top_k)
                record("Fusion", i, fused)
        if qdrant_client is not None:
            try:
                record("Qdrant", i, qdrant_search(qdrant_client, embeddings[i], top_k + 1))
            except NotImplementedError:
                qdrant_client = None
    return per


def print_loo_report(per: dict[str, dict[str, list[float]]], top_k: int = TOP_K) -> None:
    """Per-method P@k and MRR@k with intervals, then paired comparisons on the same queries."""
    labels = {"BM25": "BM25 keyword", "Cosine": "Cosine numpy", "Fusion": "Fusion RRF", "Qdrant": "Qdrant :memory:"}
    methods = [m for m in ("BM25", "Cosine", "Fusion", "Qdrant") if m in per]
    n = len(per["Cosine"]["p"])
    print(f"\n{'=' * 72}")
    print(f"  Leave-one-out over all {n} tickets (query = ticket, relevant = same category)")
    print(f"  95% bootstrap intervals over queries")
    print(f"  {'Method':<18} {'P@' + str(top_k):<24} {'MRR@' + str(top_k):<24}")
    print("  " + "─" * 66)
    for m in methods:
        p, rr = per[m]["p"], per[m]["rr"]
        (plo, phi), (rlo, rhi) = _bootstrap_ci(p), _bootstrap_ci(rr)
        print(f"  {labels[m]:<18} {np.mean(p):.3f} [{plo:.3f}, {phi:.3f}]    {np.mean(rr):.3f} [{rlo:.3f}, {rhi:.3f}]")
    pairs = [("BM25", "Cosine"), ("Cosine", "Fusion"), ("Cosine", "Qdrant")]
    print("  " + "─" * 66)
    print(f"  Paired comparison of P@{top_k} (same queries, new minus old)")
    for old, new in pairs:
        if old not in per or new not in per:
            continue
        diffs = [b - a for a, b in zip(per[old]["p"], per[new]["p"])]
        lo, hi = _bootstrap_ci(diffs)
        verdict = ("better" if lo > 0 else "worse" if hi < 0 else "no measurable difference")
        better = sum(d > 0 for d in diffs)
        worse  = sum(d < 0 for d in diffs)
        print(f"  {new:<6} vs {old:<6} {np.mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"better on {better}, worse on {worse} queries -> {verdict}")
    print(f"  {'=' * 70}")
    print("  ↑ Copy this block into results.md Section 3")


# ---------------------------------------------------------------------------
# PRE-BUILT: BONUS EVAL — HYBRID + RERANK ON LABELED QUERIES
# ---------------------------------------------------------------------------
# Leave-one-out above scores "same category": a topic label that favours embeddings, on a
# corpus with no identifiers for BM25 to match exactly. The bonus uses 90 queries with
# per-query relevance labels over the 300 tickets + 40 tickets that carry an identifier
# (order number, error code, invoice, SKU, firmware version...):
#   exact    (40) — the query names an identifier; exactly one ticket is relevant, and three
#                   near-duplicates from the same family differ only by the identifier
#   semantic (50) — a paraphrase of one specific problem; grades from an LLM judge, checked
#                   by hand: 2 = same specific problem, 1 = same topic, everything else 0

_IDS_PATH     = Path(__file__).parent / "data" / "tickets_ids.json"
_QUERIES_PATH = Path(__file__).parent / "data" / "queries.json"
QUERY_TYPES   = ("exact", "semantic")


def _ndcg_at_k(ranked: list[int], rel: dict[int, int], k: int = TOP_K) -> float:
    """Graded nDCG@k: gain 2^grade − 1, log2 position discount, divided by the ideal order."""
    dcg   = sum((2 ** rel.get(d, 0) - 1) / np.log2(pos + 2) for pos, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg  = sum((2 ** g - 1) / np.log2(pos + 2) for pos, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _rr_at_k(ranked: list[int], rel: dict[int, int], k: int = TOP_K) -> float:
    """Reciprocal rank of the first grade-2 ticket within top-k, 0 if there is none."""
    return next((1.0 / (pos + 1) for pos, d in enumerate(ranked[:k]) if rel.get(d) == 2), 0.0)


def evaluate_bonus(base_embeddings: np.ndarray) -> None:
    """Cosine, BM25, Fusion RRF and every reranker on the 90 labeled queries; prints the report.

    Rerankers re-order the cosine top-RERANK_CANDIDATES; Fusion RRF merges BM25 and cosine
    lists of the same depth. nDCG@5 and MRR@5 are reported per query type, with paired
    differences vs cosine on the same queries.
    """
    corpus   = TICKETS + json.loads(_IDS_PATH.read_text(encoding="utf-8"))
    queries  = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    texts    = [t["text"] for t in corpus]
    index_of = {text: i for i, text in enumerate(texts)}
    embs     = np.vstack([base_embeddings, get_embeddings(texts[len(TICKETS):])])
    q_embs   = get_embeddings([q["query"] for q in queries])
    rels     = [{int(d): g for d, g in q["relevant"].items()} for q in queries]
    types    = [q["type"] for q in queries]

    try:
        bm25_index = build_bm25_index(texts)
        bm25_search(bm25_index, queries[0]["query"], 1)
    except NotImplementedError:
        bm25_index = None
        print("  TODO 4 not implemented — BM25 and Fusion rows skipped")

    cosine_results = [cosine_search(e, embs, RERANK_CANDIDATES) for e in q_embs]
    candidates     = [[i for i, _ in res] for res in cosine_results]
    rankings: dict[str, list[list[int]]] = {"Cosine": candidates}
    if bm25_index is not None:
        rankings["BM25"], rankings["Fusion RRF"] = [], []
        for q, cos in zip(queries, cosine_results):
            bm = bm25_search(bm25_index, q["query"], RERANK_CANDIDATES)
            rankings["BM25"].append([i for i, _ in bm])
            rankings["Fusion RRF"].append([i for i, _ in fusion_search(bm, cos, TOP_K)])

    latency: dict[str, float] = {}
    for key in RERANKER_MODELS:
        try:
            rerank(queries[0]["query"], texts[:1], model_key=key)  # warm-up: model loading is not latency
        except NotImplementedError:
            print("  rerank() not implemented — reranker rows skipped")
            break
        print(f"  reranking with {key}...")
        t0 = time.perf_counter()
        rankings[f"Rerank {key}"] = [
            [index_of[text] for text, _ in rerank(q["query"], [texts[i] for i in cand], model_key=key)]
            for q, cand in zip(queries, candidates)
        ]
        latency[f"Rerank {key}"] = (time.perf_counter() - t0) * 1000 / len(queries)

    def per_query(metric, method: str) -> np.ndarray:
        return np.array([metric(r, rel) for r, rel in zip(rankings[method], rels)])

    def mask(qtype: str) -> np.ndarray:
        return np.array([t == qtype for t in types])

    ndcg = {m: per_query(_ndcg_at_k, m) for m in rankings}
    rr   = {m: per_query(_rr_at_k, m) for m in rankings}
    found = np.array([
        sum(rel_d in set(cand) for rel_d, g in rel.items() if g == 2) / sum(g == 2 for g in rel.values())
        for cand, rel in zip(candidates, rels)
    ])

    counts = {t: int(mask(t).sum()) for t in QUERY_TYPES}
    print(f"\n{'=' * 78}")
    print(f"  Bonus: {len(queries)} labeled queries over {len(corpus)} tickets "
          f"({counts['exact']} exact with an identifier, {counts['semantic']} semantic)")
    print(f"  Stage 1: cosine top-{RERANK_CANDIDATES} contains {found.mean():.0%} of the grade-2 tickets "
          f"(exact {found[mask('exact')].mean():.0%}, semantic {found[mask('semantic')].mean():.0%})"
          " — the ceiling for any reranker")
    print(f"\n  {'Method':<18} {'nDCG@5 exact':>13} {'semantic':>9} {'all':>7}   "
          f"{'MRR@5 exact':>12} {'semantic':>9}   {'rerank ms/query':>15}")
    print("  " + "─" * 92)
    for m in rankings:
        ms = f"{latency[m]:.0f}" if m in latency else "—"
        print(f"  {m:<18} {ndcg[m][mask('exact')].mean():>13.3f} {ndcg[m][mask('semantic')].mean():>9.3f} "
              f"{ndcg[m].mean():>7.3f}   {rr[m][mask('exact')].mean():>12.3f} "
              f"{rr[m][mask('semantic')].mean():>9.3f}   {ms:>15}")
    print("  " + "─" * 92)
    print("  Paired nDCG@5 vs Cosine (same queries, method minus cosine), 95% bootstrap interval")
    for m in rankings:
        if m == "Cosine":
            continue
        parts = []
        for t in QUERY_TYPES:
            diffs = list(ndcg[m][mask(t)] - ndcg["Cosine"][mask(t)])
            lo, hi = _bootstrap_ci(diffs)
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "no measurable difference"
            parts.append(f"{t}: {np.mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}] {verdict}")
        print(f"  {m:<18} " + " | ".join(parts))
    print(f"  {'=' * 76}")
    print("  ↑ Copy this block into results.md Section 5")


# ---------------------------------------------------------------------------
# PRE-BUILT: TICKET HELPER DEMO
# ---------------------------------------------------------------------------

def ticket_helper_demo(
    query:         str,
    query_emb:     np.ndarray,
    all_embs:      np.ndarray,
    tickets:       list[dict],
    labels:        np.ndarray,
    cluster_names: dict[int, str],
    top_similar:   int = 3,
    bm25_index:    object | None = None,
) -> None:
    """Show similar past tickets + category hint (majority vote)."""
    print("\n" + "=" * 72)
    print("  Ticket Helper (similar tickets + category hint)")
    print("=" * 72)
    print(f'\n  New ticket: "{query}"')

    try:
        if bm25_index is None:
            raise NotImplementedError
        bm25_r   = bm25_search(bm25_index, query, TOP_K)
        cosine_r = cosine_search(query_emb, all_embs, TOP_K)
        results  = fusion_search(bm25_r, cosine_r, TOP_K)
        method   = "Fusion RRF"
    except NotImplementedError:
        results = cosine_search(query_emb, all_embs, TOP_K)
        method  = "Cosine (BM25 not implemented)"

    print(f"\n  Retrieval: {method}")
    top        = results[:top_similar]
    categories = [tickets[idx]["category"] for idx, _ in top]
    suggested  = max(set(categories), key=categories.count)
    print(f"\n  Suggested category (majority of top-{top_similar}): {suggested}")

    cluster_votes: dict[int, int] = {}
    for idx, _ in top:
        cid = int(labels[idx])
        cluster_votes[cid] = cluster_votes.get(cid, 0) + 1
    best_cid = max(cluster_votes, key=cluster_votes.get)
    print(f'  Cluster (majority of top-{top_similar}): "{cluster_names.get(best_cid, f"Cluster {best_cid}")}"')

    print("\n  Similar past tickets:")
    for rank, (idx, score) in enumerate(top, 1):
        cat  = tickets[idx]["category"]
        text = tickets[idx]["text"][:70]
        print(f"    {rank}. [{cat}] {text}...")
    print("=" * 72)


# ---------------------------------------------------------------------------
# PRE-BUILT: CLUSTER TABLE
# ---------------------------------------------------------------------------

def print_cluster_table(
    tickets:       list[dict],
    labels:        np.ndarray,
    cluster_names: dict[int, str],
) -> None:
    """Print cluster name, size, and example tickets."""
    print("\n" + "=" * 72)
    print("  Cluster Summary")
    print("=" * 72)
    for cluster_id in sorted(cluster_names):
        name    = cluster_names[cluster_id]
        mask    = labels == cluster_id
        ctexts  = [tickets[i]["text"] for i in range(len(tickets)) if mask[i]]
        print(f"\n  [{cluster_id}] {name} ({len(ctexts)} tickets)")
        for t in ctexts[:3]:
            print(f"      - {t[:70]}")
        if len(ctexts) > 3:
            print(f"      ... and {len(ctexts) - 3} more")
    print("=" * 72)


# ---------------------------------------------------------------------------
# VALIDATION — used by eval.py
# ---------------------------------------------------------------------------

def validate_homework() -> tuple[bool, list[str]]:
    """Check that TODOs are filled. Returns (ok, list_of_error_messages)."""
    errors: list[str] = []

    # TODO 1
    try:
        embs = get_embeddings(["hello world", "test sentence"])
        if not isinstance(embs, np.ndarray):
            errors.append("get_embeddings must return a numpy array")
        elif embs.ndim != 2 or embs.shape[0] != 2:
            errors.append(f"get_embeddings shape should be (2, dim), got {embs.shape}")
    except NotImplementedError:
        errors.append("TODO 1: get_embeddings not implemented")
    except Exception as e:
        errors.append(f"get_embeddings raised: {e}")

    # TODO 2
    try:
        dummy  = np.random.randn(10, 8)
        labels = cluster_tickets(dummy, 3)
        if not isinstance(labels, np.ndarray):
            errors.append("cluster_tickets must return a numpy array")
        elif labels.shape != (10,):
            errors.append(f"cluster_tickets shape should be (10,), got {labels.shape}")
    except NotImplementedError:
        errors.append("TODO 2: cluster_tickets not implemented")
    except Exception as e:
        errors.append(f"cluster_tickets raised: {e}")

    # TODO 3 — check prompt is written (don't require API key)
    try:
        result = name_cluster(["test ticket 1", "test ticket 2"])
        if not isinstance(result, str) or not result.strip():
            errors.append("name_cluster must return a non-empty string")
    except NotImplementedError:
        errors.append("TODO 3: name_cluster not implemented — write a prompt")
    except Exception:
        pass  # prompt is written but OPENROUTER_API_KEY not set — that's ok

    # TODO 4
    try:
        docs    = [f"word{i} token" for i in range(10)]
        results = bm25_search(build_bm25_index(docs), "word3 token", 3)
        if not isinstance(results, list) or len(results) != 3:
            errors.append("bm25_search must return a list of length top_k")
        elif not all(isinstance(r, tuple) and len(r) == 2 for r in results):
            errors.append("bm25_search must return list of (index, score) tuples")
    except NotImplementedError:
        errors.append("TODO 4: build_bm25_index / bm25_search not implemented")
    except Exception as e:
        errors.append(f"bm25_search raised: {e}")

    # TODO 5
    try:
        dummy_tickets = [{"text": f"ticket {i}", "category": "test"} for i in range(5)]
        dummy_embs    = np.random.randn(5, 384).astype(np.float32)
        client        = upload_to_qdrant(dummy_tickets, dummy_embs)
        if client is None:
            errors.append("upload_to_qdrant must return a QdrantClient instance")
        else:
            q_emb    = np.random.randn(384).astype(np.float32)
            results  = qdrant_search(client, q_emb, 3)
            if not isinstance(results, list) or len(results) != 3:
                errors.append("qdrant_search must return a list of length top_k")
            elif not all(isinstance(r, tuple) and len(r) == 2 for r in results):
                errors.append("qdrant_search must return list of (index, score) tuples")
    except NotImplementedError:
        errors.append("TODO 5: upload_to_qdrant / qdrant_search not implemented")
    except Exception as e:
        errors.append(f"TODO 5 raised: {e}")

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket Auto-Categorization — HW v2")
    parser.add_argument("--quick",    action="store_true", help="Skip LLM cluster naming")
    parser.add_argument("--query",    type=str,  default=None, help="Single query instead of DEMO_QUERIES")
    parser.add_argument("--clusters", type=int,  default=DEFAULT_N_CLUSTERS, help="Number of clusters")
    parser.add_argument("--rerank",   action="store_true", help="Bonus: hybrid + rerank on 90 labeled queries")
    args = parser.parse_args()

    print("=" * 65)
    print("  Ticket Auto-Categorization with Embeddings (300 tickets)")
    print("=" * 65)

    # Step 1: Embeddings
    print(f"\n[1/5] Generating embeddings for {len(TICKETS)} tickets...")
    texts      = [t["text"] for t in TICKETS]
    embeddings = get_embeddings(texts)
    print(f"  Shape: {embeddings.shape}")

    # Step 2: Clustering
    print(f"\n[2/5] Clustering into {args.clusters} groups...")
    labels = cluster_tickets(embeddings, args.clusters)
    for cid in sorted(set(labels)):
        count = (labels == cid).sum()
        print(f"  Cluster {cid}: {count} tickets")

    ari = compute_ari(TICKETS, labels)
    print(f"\n  ARI vs ground-truth categories: {ari:.3f}  (0=random, 1=perfect)")
    print_ari_sweep(TICKETS, embeddings)

    # Step 3: LLM cluster naming
    cluster_names: dict[int, str] = {}
    if not args.quick:
        print("\n[3/5] Naming clusters via LLM...")
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("  OPENROUTER_API_KEY not set — skipping LLM naming")
            args.quick = True

    if not args.quick:
        for cid in sorted(set(labels)):
            ctexts = [texts[i] for i in range(len(texts)) if labels[i] == cid]
            try:
                name = name_cluster(ctexts)
                cluster_names[cid] = name
                print(f"  Cluster {cid} → \"{name}\"")
            except NotImplementedError:
                print("  TODO 3 not implemented — skipping LLM naming")
                args.quick = True
                break
    else:
        print("\n[3/5] Skipping LLM naming (--quick mode)")

    if args.quick:
        for cid in sorted(set(labels)):
            cluster_names[cid] = f"Cluster {cid}"

    print("\n  Visualizing clusters (t-SNE)...")
    visualize_clusters(embeddings, labels, cluster_names)
    print_cluster_table(TICKETS, labels, cluster_names)

    # Step 4a: search indexes — built once, reused by every query below
    print("\n[4a] Building search indexes (one-time cost, not part of query latency)...")
    bm25_index = None
    try:
        t0 = time.perf_counter()
        bm25_index = build_bm25_index(texts)
        print(f"  BM25 index: {len(texts)} tickets in {(time.perf_counter() - t0) * 1000:.1f} ms")
    except NotImplementedError:
        print("  TODO 4a not implemented — BM25 and Fusion columns will be skipped")

    qdrant_client = None
    try:
        t0 = time.perf_counter()
        qdrant_client = upload_to_qdrant(TICKETS, embeddings)
        print(f"  Qdrant :memory: {len(TICKETS)} tickets in {(time.perf_counter() - t0) * 1000:.1f} ms")
    except NotImplementedError:
        print("  TODO 5 not implemented — Qdrant column will be skipped")

    # Step 4: Search comparison
    if args.query is not None:
        query_pairs = [(args.query, None)]
    else:
        query_pairs = list(DEMO_QUERIES.items())
    print(f"\n[4/5] Search comparison ({len(query_pairs)} queries)...")

    last_q:   str | None       = None
    last_emb: np.ndarray | None = None
    all_metrics: list[dict]     = []

    for i, (query, expected_cat) in enumerate(query_pairs, 1):
        print(f"\n{'=' * 65}")
        print(f"  Query {i}/{len(query_pairs)}: \"{query}\"")
        q_emb   = get_embeddings([query])[0]
        metrics = compare_and_print(query, q_emb, embeddings, TICKETS, qdrant_client, expected_cat,
                                    bm25_index=bm25_index)
        all_metrics.append(metrics)
        last_q   = query
        last_emb = q_emb

    # Summary table (only for DEMO_QUERIES with known expected categories)
    if len(all_metrics) > 1 and all(
        any(v["recall"] is not None for v in m.values()) for m in all_metrics
    ):
        print(f"\n{'=' * 65}")
        print("  Summary — avg over all demo queries")
        print(f"  {'Method':<18} {'Avg cat. P@5':>14} {'Avg Latency':>12}")
        print("  " + "─" * 46)
        for key in ("BM25", "Cosine", "Fusion", "Qdrant"):
            recalls   = [m[key]["recall"]     for m in all_metrics if m[key]["recall"] is not None]
            latencies = [m[key]["latency_ms"] for m in all_metrics if m[key]["latency_ms"] > 0]
            if not latencies:
                continue
            avg_r = sum(recalls) / len(recalls) if recalls else 0
            avg_l = sum(latencies) / len(latencies)
            label = {"BM25": "BM25 keyword", "Cosine": "Cosine numpy", "Fusion": "Fusion RRF", "Qdrant": "Qdrant :memory:"}[key]
            print(f"  {label:<18} {avg_r:>8.1f} / 5   {avg_l:>8.2f} ms")
        print(f"  {'=' * 46}")
        print("  ↑ 5 demo queries only — see the leave-one-out block below for the real comparison")

    # Step 4b: leave-one-out eval over all tickets (only in demo mode)
    if args.query is None:
        print(f"\n[4b] Leave-one-out retrieval eval: every ticket is a query ({len(TICKETS)} queries)...")
        print_loo_report(evaluate_retrieval_loo(TICKETS, embeddings, qdrant_client, bm25_index=bm25_index))

    # Bonus: hybrid + rerank on labeled queries
    if args.rerank:
        print("\n[BONUS] Hybrid + rerank on 90 labeled queries (data/queries.json)...")
        evaluate_bonus(embeddings)

    # Step 5: Ticket Helper
    print("\n[5/5] Ticket Helper demo...")
    if last_q is not None and last_emb is not None:
        ticket_helper_demo(last_q, last_emb, embeddings, TICKETS, labels, cluster_names,
                           bm25_index=bm25_index)

    print("\nDone! Check clusters.png for the t-SNE visualization.")


if __name__ == "__main__":
    main()
