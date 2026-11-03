"""Retrieval metrics, the proposed Aspect Coverage Score and a paired significance test.

All embedding arguments are expected to be L2-normalised, so a dot product is
a cosine similarity.
"""
import numpy as np

from . import config


def precision_at_k(retrieved, relevant, k):
    return sum(1 for i in retrieved[:k] if i in relevant) / k


def reciprocal_rank(retrieved, relevant):
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved, relevant, k):
    return sum(
        relevant.get(idx, 0) / np.log2(rank + 1)
        for rank, idx in enumerate(retrieved[:k], start=1)
    )


def ndcg_at_k(retrieved, relevant, k):
    ideal = sorted(relevant, key=relevant.get, reverse=True)
    ideal_dcg = dcg_at_k(ideal, relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def diversity_score(doc_embs):
    """1 - mean pairwise cosine similarity of the selected documents."""
    n = len(doc_embs)
    if n < 2:
        return 0.0
    sims = doc_embs @ doc_embs.T
    return float(1.0 - sims[np.triu_indices(n, k=1)].mean())


def covered_aspects(aspect_embs, doc_embs, threshold=config.ASPECT_SIM_THRESHOLD):
    """Boolean vector: is each aspect covered by at least one document?"""
    if len(aspect_embs) == 0 or len(doc_embs) == 0:
        return np.zeros(len(aspect_embs), dtype=bool)
    return (aspect_embs @ doc_embs.T).max(axis=1) >= threshold


def aspect_coverage_score(aspect_embs, doc_embs, threshold=config.ASPECT_SIM_THRESHOLD):
    """ACS = covered aspects / total aspects (0.0 when the query has no aspects)."""
    if len(aspect_embs) == 0:
        return 0.0
    return float(covered_aspects(aspect_embs, doc_embs, threshold).mean())


def paired_randomization_test(a, b, n_resamples=10000, seed=config.SEED):
    """Two-sided paired sign-flip test on per-query scores.

    Returns ``(mean difference a - b, p-value)``.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = diff.mean()
    if np.allclose(diff, 0):
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_resamples + 1)
    return float(observed), float(p)
