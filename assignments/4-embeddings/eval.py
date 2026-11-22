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
Homework v2 validation: structural checks, unit tests, category smoke tests, leave-one-out eval.

1. Structural: validate_homework() — types, shapes, edge cases.
2. Unit tests: small deterministic checks for each TODO function (+ rerank, once implemented).
3. Category smoke tests: run cosine + BM25 on 5 queries; check that
   ≥3/5 top results belong to the expected category (category precision@5).
   → These tests survive any corpus changes, unlike hardcoded index sets.
4. Leave-one-out eval: every one of the 300 tickets is a query, relevant = same category.
   Checks mean precision@5 against thresholds and that Qdrant matches cosine.
   → 300 queries instead of 5: the comparison between methods is not noise.
5. Bonus data: the 90 labeled queries and 40 ID tickets load and are consistent.

Run:
  uv run eval.py
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np

try:
    import starter  # type: ignore
except ImportError as e:
    print("Could not import starter.py:", e)
    print("Run this script from the homework directory, next to starter.py: uv run eval.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Category smoke-test queries
#
# For each query, we know which category is most relevant.
# A good retrieval system should return ≥ 3 results from that category in top-5.
#
# These queries are semantically rich — they test understanding, not keyword matching.
# The corpus has 300 balanced tickets so each category has ~43 relevant docs.
# ---------------------------------------------------------------------------

EVAL_QUERIES: dict[str, str] = {
    "my laptop screen is broken":              "technical",
    "I can't authenticate my identity":        "account",
    "money problems with my purchase":         "billing",
    "package not delivered to my address":     "shipping",
    "want to send the item back for a refund": "returns",
}

# Category precision@5 thresholds for the 5 smoke queries
COSINE_THRESHOLD = 3   # ≥ N of top-5 results must be in expected category
BM25_THRESHOLD   = 2   # BM25 is weaker on semantic queries — looser threshold
QDRANT_THRESHOLD = 3   # same as cosine (Qdrant :memory: is exact COSINE)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_unit_tests() -> tuple[int, int, list[str]]:
    """Run targeted unit tests per function. Returns (passed, total, failures)."""
    passed   = 0
    total    = 0
    failures: list[str] = []

    def check(name: str, condition: bool, msg: str) -> None:
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"    [{name}] {msg}: ok")
        else:
            failures.append(f"[{name}] {msg}")
            print(f"    [{name}] {msg}: FAIL")

    # --- get_embeddings ---
    try:
        embs = starter.get_embeddings(["hello world", "hello world", "something completely different"])
        check("get_embeddings", embs.shape[1] == 384,
              f"embedding dim == 384 (got {embs.shape[1]})")
        cos_same = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))
        check("get_embeddings", cos_same > 0.99,
              f"identical texts similarity ~1.0 (got {cos_same:.4f})")
        cos_diff = float(np.dot(embs[0], embs[2]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[2])))
        check("get_embeddings", cos_diff < 0.9,
              f"different texts similarity < 0.9 (got {cos_diff:.4f})")
        check("get_embeddings", embs.dtype in (np.float32, np.float64),
              f"returns float array (got {embs.dtype})")
    except NotImplementedError:
        total += 4
        failures.append("[get_embeddings] not implemented")
        print("    [get_embeddings] not implemented — skipping")
    except Exception as e:
        total += 4
        failures.append(f"[get_embeddings] error: {e}")
        print(f"    [get_embeddings] error: {e}")

    # --- cluster_tickets ---
    try:
        np.random.seed(0)
        dummy  = np.random.randn(20, 8)
        labels = starter.cluster_tickets(dummy, 4)
        check("cluster_tickets", len(set(labels)) == 4,
              f"4 clusters → 4 unique labels (got {len(set(labels))})")
        labels2 = starter.cluster_tickets(dummy, 4)
        check("cluster_tickets", np.array_equal(labels, labels2),
              "deterministic with random_state=42")
        check("cluster_tickets", labels.shape == (20,),
              f"shape (20,) (got {labels.shape})")
    except NotImplementedError:
        total += 3
        failures.append("[cluster_tickets] not implemented")
        print("    [cluster_tickets] not implemented — skipping")
    except Exception as e:
        total += 3
        failures.append(f"[cluster_tickets] error: {e}")
        print(f"    [cluster_tickets] error: {e}")

    # --- name_cluster: prompt must be written ---
    try:
        result = starter.name_cluster(["test 1", "test 2"])
        total += 1
        if isinstance(result, str) and result.strip():
            passed += 1
            print(f"    [name_cluster] implemented, returned name: ok")
        else:
            failures.append("[name_cluster] must return a non-empty string")
            print("    [name_cluster] returned empty or wrong type: FAIL")
    except NotImplementedError:
        total += 1
        failures.append("[name_cluster] not implemented — write a prompt (TODO 3)")
        print("    [name_cluster] not implemented (TODO 3): FAIL")
    except Exception:
        total += 1
        passed += 1  # prompt written, API key needed to run fully
        print("    [name_cluster] implemented (needs OPENROUTER_API_KEY to run): ok")

    # --- bm25_search ---
    try:
        corpus  = ["the cat sat on the mat", "the dog ran in the park", "a bird flew over the lake"]
        results = starter.bm25_search(starter.build_bm25_index(corpus), "cat sat", 2)
        check("bm25_search", results[0][0] == 0,
              f'top-1 for "cat sat" is doc 0 (got {results[0][0]})')
        check("bm25_search", len(results) == 2,
              f"top_k=2 returns 2 results (got {len(results)})")
        scores = [s for _, s in results]
        check("bm25_search", all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
              "scores sorted descending")
        # Critical: query must be lowercased — mismatch causes wrong results
        r_upper = starter.bm25_search(starter.build_bm25_index(["CAT SAT ON MAT", "dog ran"]), "cat sat", 1)
        r_lower = starter.bm25_search(starter.build_bm25_index(["cat sat on mat", "dog ran"]), "cat sat", 1)
        check("bm25_search", r_upper[0][0] == r_lower[0][0],
              "case-insensitive tokenisation (query + corpus both lowercased)")
    except NotImplementedError:
        total += 4
        failures.append("[bm25_search] not implemented")
        print("    [bm25_search] not implemented — skipping")
    except Exception as e:
        total += 4
        failures.append(f"[bm25_search] error: {e}")
        print(f"    [bm25_search] error: {e}")

    # --- upload_to_qdrant + qdrant_search ---
    try:
        dummy_tickets = [{"text": f"ticket number {i}", "category": "test"} for i in range(5)]
        dummy_embs    = np.random.randn(5, 384).astype(np.float32)
        client        = starter.upload_to_qdrant(dummy_tickets, dummy_embs)
        check("upload_to_qdrant", client is not None,
              "returns a non-None client")
        q_emb   = np.random.randn(384).astype(np.float32)
        results = starter.qdrant_search(client, q_emb, 3)
        check("qdrant_search", isinstance(results, list) and len(results) == 3,
              f"returns list of length 3 (got {len(results) if isinstance(results, list) else type(results)})")
        check("qdrant_search", all(isinstance(r, tuple) and len(r) == 2 for r in results),
              "each result is a (index, score) tuple")
        scores = [s for _, s in results]
        check("qdrant_search", all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
              "results sorted by score descending")
    except NotImplementedError:
        total += 5
        failures.append("[TODO 5] upload_to_qdrant / qdrant_search not implemented")
        print("    [TODO 5] not implemented — skipping")
    except Exception as e:
        total += 5
        failures.append(f"[TODO 5] error: {e}")
        print(f"    [TODO 5] error: {e}")

    # --- rerank (bonus): skipped until implemented; the first run downloads the model ---
    try:
        model_key = next(iter(starter.RERANKER_MODELS))
        docs      = ["The weather is sunny today",
                     "Reset your password from the login page",
                     "Our office is closed on Sunday"]
        ranked    = starter.rerank("how do I reset my password", docs, model_key=model_key)
        check("rerank", sorted(text for text, _ in ranked) == sorted(docs),
              "returns every candidate exactly once")
        check("rerank", ranked[0][0] == docs[1],
              f'top-1 for "reset my password" is the password doc ({model_key})')
        scores = [float(s) for _, s in ranked]
        check("rerank", all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
              "scores sorted descending")
    except NotImplementedError:
        print("    [rerank] bonus not implemented — skipping")
    except Exception as e:
        total += 3
        failures.append(f"[rerank] error: {e}")
        print(f"    [rerank] error: {e}")

    return passed, total, failures


# Leave-one-out thresholds (reference: cosine 0.646, BM25 0.521, Qdrant == cosine)
LOO_COSINE_MIN = 0.55
LOO_BM25_MIN   = 0.40


# ---------------------------------------------------------------------------
# Category smoke tests (5 queries)
# ---------------------------------------------------------------------------

def run_recall_tests() -> tuple[int, int, list[str]]:
    """
    For each query in EVAL_QUERIES, check that top-5 retrieval results
    contain at least N tickets from the expected category.

    This metric survives corpus changes — unlike hardcoded index sets.
    """
    texts      = [t["text"] for t in starter.TICKETS]
    embeddings = starter.get_embeddings(texts)

    passed   = 0
    total    = 0
    failures: list[str] = []
    try:
        bm25_index = starter.build_bm25_index(texts)
    except NotImplementedError:
        bm25_index = None

    for query, expected_cat in EVAL_QUERIES.items():
        q_emb       = starter.get_embeddings([query])[0]
        short_q     = query[:40]

        # --- Cosine recall ---
        cos_results = starter.cosine_search(q_emb, embeddings, 5)
        cos_cats    = [starter.TICKETS[idx]["category"] for idx, _ in cos_results]
        cos_hits    = sum(c == expected_cat for c in cos_cats)
        total += 1
        if cos_hits >= COSINE_THRESHOLD:
            passed += 1
            print(f"    [cosine] \"{short_q}\" → {cos_hits}/5 {expected_cat}: ok")
        else:
            msg = f'[cosine] "{short_q}" → {cos_hits}/5 {expected_cat} (need ≥{COSINE_THRESHOLD})'
            failures.append(msg)
            print(f"    {msg}: FAIL")
            print(f"      got: {cos_cats}")

        # --- BM25 recall (skip if not implemented) ---
        try:
            if bm25_index is None:
                raise NotImplementedError
            bm25_results = starter.bm25_search(bm25_index, query, 5)
            bm25_cats    = [starter.TICKETS[idx]["category"] for idx, _ in bm25_results]
            bm25_hits    = sum(c == expected_cat for c in bm25_cats)
            total += 1
            if bm25_hits >= BM25_THRESHOLD:
                passed += 1
                print(f"    [bm25]   \"{short_q}\" → {bm25_hits}/5 {expected_cat}: ok")
            else:
                msg = f'[bm25] "{short_q}" → {bm25_hits}/5 {expected_cat} (need ≥{BM25_THRESHOLD})'
                failures.append(msg)
                print(f"    {msg}: FAIL")
                print(f"      got: {bm25_cats}")
        except NotImplementedError:
            print(f"    [bm25]   skipped (TODO 4 not implemented)")

        # --- Qdrant recall (skip if not implemented) ---
        try:
            dummy_client = starter.upload_to_qdrant(starter.TICKETS, embeddings)
            q_results    = starter.qdrant_search(dummy_client, q_emb, 5)
            q_cats       = [starter.TICKETS[idx]["category"] for idx, _ in q_results]
            q_hits       = sum(c == expected_cat for c in q_cats)
            total += 1
            if q_hits >= QDRANT_THRESHOLD:
                passed += 1
                print(f"    [qdrant] \"{short_q}\" → {q_hits}/5 {expected_cat}: ok")
            else:
                msg = f'[qdrant] "{short_q}" → {q_hits}/5 {expected_cat} (need ≥{QDRANT_THRESHOLD})'
                failures.append(msg)
                print(f"    {msg}: FAIL")
                print(f"      got: {q_cats}")
        except NotImplementedError:
            print(f"    [qdrant] skipped (TODO 5 not implemented)")

    return passed, total, failures


def run_loo_checks() -> list[str]:
    """Leave-one-out over all tickets via starter.evaluate_retrieval_loo; returns failure messages."""
    embeddings = starter.get_embeddings([t["text"] for t in starter.TICKETS])
    try:
        client = starter.upload_to_qdrant(starter.TICKETS, embeddings)
    except NotImplementedError:
        client = None
    try:
        bm25_index = starter.build_bm25_index([t["text"] for t in starter.TICKETS])
    except NotImplementedError:
        bm25_index = None
    per = starter.evaluate_retrieval_loo(starter.TICKETS, embeddings, client, bm25_index=bm25_index)
    failures: list[str] = []
    means = {m: float(np.mean(v["p"])) for m, v in per.items()}
    for method, mean in means.items():
        print(f"    [{method.lower():<6}] mean P@5 = {mean:.3f}")
    if means["Cosine"] < LOO_COSINE_MIN:
        failures.append(f"cosine mean P@5 {means['Cosine']:.3f} < {LOO_COSINE_MIN} — check get_embeddings (TODO 1)")
    if "BM25" not in per:
        print("    [bm25]   skipped (TODO 4 not implemented)")
    elif means["BM25"] < LOO_BM25_MIN:
        failures.append(f"BM25 mean P@5 {means['BM25']:.3f} < {LOO_BM25_MIN} — check tokenization in build_bm25_index / bm25_search (TODO 4)")
    if "Qdrant" not in per:
        print("    [qdrant] skipped (TODO 5 not implemented)")
    elif abs(means["Qdrant"] - means["Cosine"]) > 0.01:
        failures.append(f"Qdrant P@5 {means['Qdrant']:.3f} differs from cosine {means['Cosine']:.3f} — "
                        "check vectors / distance in upload_to_qdrant (TODO 5)")
    return failures


def run_bonus_data_checks() -> list[str]:
    """Bonus files load and are consistent; rerankers are not downloaded here."""
    import json

    id_tickets = json.loads(starter._IDS_PATH.read_text(encoding="utf-8"))
    queries    = json.loads(starter._QUERIES_PATH.read_text(encoding="utf-8"))
    n_corpus   = len(starter.TICKETS) + len(id_tickets)
    failures: list[str] = []
    for q in queries:
        rel = {int(d): g for d, g in q["relevant"].items()}
        if q["type"] not in starter.QUERY_TYPES:
            failures.append(f"{q['qid']}: unknown type {q['type']!r}")
        if not all(0 <= d < n_corpus for d in rel):
            failures.append(f"{q['qid']}: relevant index outside 0..{n_corpus - 1}")
        if 2 not in rel.values():
            failures.append(f"{q['qid']}: no grade-2 ticket")
        if q["type"] == "exact" and len(rel) != 1:
            failures.append(f"{q['qid']}: exact query must have exactly one relevant ticket")
    counts = {t: sum(q["type"] == t for q in queries) for t in starter.QUERY_TYPES}
    print(f"    {len(queries)} queries ({counts['exact']} exact, {counts['semantic']} semantic) "
          f"over {n_corpus} tickets")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 58)
    print("  Homework v2 — validation")
    print("=" * 58)

    ok, errors = starter.validate_homework()

    if not ok:
        print("\n  Structural checks failed:")
        for e in errors:
            print(f"    - {e}")
        print("\n  Fix the TODOs and run uv run eval.py again.")
        print("=" * 58)
        sys.exit(1)

    print("\n  Structural checks passed.")
    if any("TODO 3" in e for e in errors):
        print("  Note: TODO 3 (name_cluster) needs OPENROUTER_API_KEY to run fully.")

    # Unit tests
    print("\n  Unit tests:")
    try:
        ut_passed, ut_total, ut_failures = run_unit_tests()
        if ut_passed == ut_total:
            print(f"\n  {ut_passed}/{ut_total} unit tests passed.")
        else:
            print(f"\n  {ut_passed}/{ut_total} unit tests passed. Failures:")
            for f in ut_failures:
                print(f"    - {f}")
    except Exception as e:
        print(f"  Unit tests could not run: {e}")

    # Category smoke tests
    print("\n  Category smoke tests, precision@5 on 5 queries (≥3/5 cosine, ≥2/5 BM25, ≥3/5 Qdrant):")
    try:
        r_passed, r_total, r_failures = run_recall_tests()
        if r_passed == r_total:
            print(f"\n  All {r_total} smoke checks passed.")
        else:
            print(f"\n  {r_passed}/{r_total} smoke checks passed.")
            if r_failures:
                print("  Failures:")
                for f in r_failures:
                    print(f"    - {f}")
    except NotImplementedError:
        print("  Smoke tests require TODO 1 (get_embeddings) to be implemented.")
    except Exception as e:
        print(f"  Smoke tests could not run: {e}")

    # Leave-one-out eval over all tickets
    print(f"\n  Leave-one-out eval over all {len(starter.TICKETS)} tickets "
          f"(cosine P@5 ≥ {LOO_COSINE_MIN}, BM25 P@5 ≥ {LOO_BM25_MIN}, Qdrant == cosine):")
    try:
        loo_failures = run_loo_checks()
        if loo_failures:
            print("  Failures:")
            for f in loo_failures:
                print(f"    - {f}")
        else:
            print("  All leave-one-out checks passed.")
    except NotImplementedError:
        print("  Leave-one-out eval requires TODO 1 (get_embeddings) to be implemented.")
    except Exception as e:
        print(f"  Leave-one-out eval could not run: {e}")

    print("\n  Bonus data (data/tickets_ids.json + data/queries.json):")
    try:
        bonus_failures = run_bonus_data_checks()
        if bonus_failures:
            print("  Failures:")
            for f in bonus_failures:
                print(f"    - {f}")
        else:
            print("  Bonus data checks passed.")
    except Exception as e:
        print(f"  Bonus data could not load: {e}")

    print(f"\n  Run the full pipeline: uv run starter.py --quick")
    print(f"  Bonus (after rerank()): uv run starter.py --quick --rerank")
    print("=" * 58)
    sys.exit(0)


if __name__ == "__main__":
    main()
