"""Memory-lean BM25 that reproduces rank_bm25.BM25Okapi scores exactly, using flat NumPy posting lists.

rank_bm25 keeps a Python dict per document (~65 MB for NFCorpus); this keeps three arrays (a few MB)
and avoids importing scipy, which matters on a 512 MB free host. tests/test_search.py checks the
scores match rank_bm25.
"""
from collections import Counter

import numpy as np

from .core.retrievers import tokenize


class LeanBM25:
    def __init__(self, doc_texts, k1=1.5, b=0.75, epsilon=0.25):
        docs = [tokenize(d) for d in doc_texts]
        vocab, rows, cols, vals = {}, [], [], []
        for i, toks in enumerate(docs):
            for tok, tf in Counter(toks).items():
                rows.append(i)
                cols.append(vocab.setdefault(tok, len(vocab)))
                vals.append(tf)
        self.vocab = vocab
        rows, cols = np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)
        tf = np.asarray(vals, dtype=np.float64)
        doc_len = np.array([len(t) for t in docs], dtype=np.float64)
        n_docs = len(docs)

        # Same IDF as BM25Okapi: negative values are floored to epsilon * mean IDF.
        df = np.bincount(cols, minlength=len(vocab)).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * (idf.sum() / len(idf))
        self.idf = idf

        # Pre-compute the BM25 weight of every (doc, term) pair and store it as posting lists by term.
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        weight = tf * (k1 + 1) / (tf + norm[rows])
        order = np.argsort(cols, kind="stable")
        self.post_docs, self.post_w = rows[order], weight[order]
        self.indptr = np.concatenate([[0], np.cumsum(df.astype(np.int64))])
        self.n_docs = n_docs

    def get_scores(self, query_tokens):
        scores = np.zeros(self.n_docs)
        for tok in query_tokens:            # repeated query tokens count again, as in rank_bm25
            j = self.vocab.get(tok)
            if j is not None:
                lo, hi = self.indptr[j], self.indptr[j + 1]
                np.add.at(scores, self.post_docs[lo:hi], self.idf[j] * self.post_w[lo:hi])
        return scores

    def search(self, query: str, k: int):
        scores = self.get_scores(tokenize(query))
        return np.argsort(-scores, kind="stable")[:k]
