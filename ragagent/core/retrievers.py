"""First-stage retrievers: BM25 (lexical) and dense bi-encoder + FAISS."""
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    """Lowercase alphanumeric tokens, so "cancer?" and "cancer" match."""
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, doc_texts):
        self.index = BM25Okapi([tokenize(d) for d in doc_texts])

    def search(self, query: str, k: int):
        scores = self.index.get_scores(tokenize(query))
        return np.argsort(-scores, kind="stable")[:k]


def encode_documents(encoder, doc_texts, cache_path=None, batch_size=64):
    """Encode all documents once (L2-normalised float32), caching to disk."""
    if cache_path is not None and Path(cache_path).exists():
        embs = np.load(cache_path)
        if embs.shape[0] == len(doc_texts):
            return embs
    embs = encoder.encode(
        doc_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
    return embs


class DenseRetriever:
    """Exact inner-product search over normalised embeddings (= cosine similarity)."""

    def __init__(self, doc_embs):
        self.index = faiss.IndexFlatIP(doc_embs.shape[1])
        self.index.add(doc_embs)

    def search(self, query_emb, k: int):
        _, idx = self.index.search(query_emb.reshape(1, -1).astype(np.float32), k)
        return idx[0]
