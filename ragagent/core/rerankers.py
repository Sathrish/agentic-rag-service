"""Second-stage re-rankers. Both return positions into the candidate list."""
import numpy as np

from . import config


def cross_encoder_scores(cross_encoder, query, candidate_texts):
    pairs = [[query, doc] for doc in candidate_texts]
    return np.asarray(cross_encoder.predict(pairs, show_progress_bar=False), dtype=np.float32)


def cross_encoder_rerank(ce_scores, top_k=config.TOP_K):
    """Sort candidates by cross-encoder relevance."""
    return list(np.argsort(-ce_scores, kind="stable")[:top_k])


def aspect_aware_rerank(
    ce_scores,
    cand_embs,
    aspect_embs,
    top_k=config.TOP_K,
    lambda_param=config.LAMBDA,
    threshold=config.ASPECT_SIM_THRESHOLD,
):
    """Greedy selection balancing relevance against coverage of new query aspects.

    At every step each remaining candidate is scored as

        lambda * CE_relevance + (1 - lambda) * (newly covered aspects / total aspects)

    where CE_relevance is the cross-encoder score min-max normalised over the
    candidate set, and a document "covers" an aspect when their cosine
    similarity is >= ``threshold``. ``cand_embs`` and ``aspect_embs`` must be
    L2-normalised; the candidate embeddings are looked up, never re-encoded.
    """
    ce_scores = np.asarray(ce_scores, dtype=np.float64)
    relevance = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-9)

    n_aspects = len(aspect_embs)
    if n_aspects:
        covers = (cand_embs @ aspect_embs.T) >= threshold      # (candidates, aspects)
    else:
        covers = np.zeros((len(ce_scores), 0), dtype=bool)

    covered = np.zeros(n_aspects, dtype=bool)
    remaining = list(range(len(ce_scores)))
    selected = []
    for _ in range(min(top_k, len(remaining))):
        new_cov = (covers[remaining] & ~covered).sum(axis=1) / max(n_aspects, 1)
        combined = lambda_param * relevance[remaining] + (1 - lambda_param) * new_cov
        best = remaining[int(np.argmax(combined))]   # ties -> earlier candidate
        selected.append(best)
        remaining.remove(best)
        covered |= covers[best]
    return selected
