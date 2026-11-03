"""Search engine: BM25 + dense retrieval fused with Reciprocal Rank Fusion, then
cross-encoder or aspect-aware re-ranking. Models are injected, so tests can use fakes."""
from dataclasses import dataclass, field

import numpy as np

from . import settings
from .core import config
from .core.aspects import extract_aspects
from .core.metrics import covered_aspects
from .core.rerankers import aspect_aware_rerank, cross_encoder_rerank, cross_encoder_scores
from .core.retrievers import BM25Retriever, DenseRetriever, encode_documents

RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class Doc:
    idx: int
    doc_id: str
    title: str
    text: str
    score: float = 0.0


@dataclass
class SearchResult:
    query: str
    docs: list
    aspects: list = field(default_factory=list)
    covered: list = field(default_factory=list)    # one bool per aspect

    @property
    def coverage(self) -> float:
        return float(np.mean(self.covered)) if self.covered else 0.0

    @property
    def uncovered_aspects(self):
        return [a for a, c in zip(self.aspects, self.covered) if not c]


def reciprocal_rank_fusion(rankings, k=RRF_K, limit=None):
    """Fuse several ranked lists of ids: score(d) = sum over lists of 1 / (k + rank)."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda i: (-scores[i], i))
    return fused[:limit] if limit else fused


class SearchEngine:
    FIRST_STAGES = ("bm25", "dense", "hybrid")
    RERANKERS = ("none", "cross-encoder", "aspect")

    def __init__(self, doc_ids, titles, texts, encoder, cross_encoder, cache_path=None):
        self.doc_ids = list(doc_ids)
        self.titles = list(titles)
        self.texts = list(texts)
        self.doc_texts = [f"{t}. {x}" for t, x in zip(self.titles, self.texts)]
        self.encoder = encoder
        self.cross_encoder = cross_encoder
        self.bm25 = BM25Retriever(self.doc_texts)
        self.doc_embs = encode_documents(encoder, self.doc_texts, cache_path)
        self.dense = DenseRetriever(self.doc_embs)

    @classmethod
    def from_nfcorpus(cls):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        from .data import load_nfcorpus

        corpus, _, _ = load_nfcorpus()
        encoder = SentenceTransformer(config.BI_ENCODER, device="cpu")
        cross = CrossEncoder(config.CROSS_ENCODER, device="cpu")
        ids = list(corpus)
        return cls(ids, [corpus[d]["title"] for d in ids], [corpus[d]["text"] for d in ids],
                   encoder, cross, cache_path=settings.CACHE_DIR / "nfcorpus_minilm.npy")

    def embed(self, texts):
        return np.asarray(self.encoder.encode(list(texts), convert_to_numpy=True,
                                              normalize_embeddings=True), dtype=np.float32)

    def candidates(self, query, n=config.N_CANDIDATES, first_stage="hybrid"):
        if first_stage not in self.FIRST_STAGES:
            raise ValueError(f"first_stage must be one of {self.FIRST_STAGES}")
        lexical = list(self.bm25.search(query, n)) if first_stage != "dense" else []
        dense = list(self.dense.search(self.embed([query])[0], n)) if first_stage != "bm25" else []
        if first_stage == "bm25":
            return [int(i) for i in lexical]
        if first_stage == "dense":
            return [int(i) for i in dense]
        return reciprocal_rank_fusion([lexical, dense], limit=n)

    def search(self, query, aspects=None, first_stage="hybrid", reranker="aspect",
               top_k=config.TOP_K, n_candidates=config.N_CANDIDATES, exclude=()):
        """Retrieve ``top_k`` documents. ``aspects`` defaults to the rule-based extractor."""
        if reranker not in self.RERANKERS:
            raise ValueError(f"reranker must be one of {self.RERANKERS}")
        aspects = list(aspects) if aspects else extract_aspects(query)
        cand = [i for i in self.candidates(query, n_candidates, first_stage) if i not in set(exclude)]
        if not cand:
            return SearchResult(query, [], aspects, [False] * len(aspects))

        if reranker == "none":
            order, scores = list(range(min(top_k, len(cand)))), np.zeros(len(cand))
        else:
            scores = cross_encoder_scores(self.cross_encoder, query, [self.doc_texts[i] for i in cand])
            if reranker == "cross-encoder" or not aspects:
                order = cross_encoder_rerank(scores, top_k)
            else:
                order = aspect_aware_rerank(scores, self.doc_embs[cand], self.embed(aspects), top_k)

        chosen = [cand[j] for j in order]
        docs = [Doc(i, self.doc_ids[i], self.titles[i], self.texts[i], float(scores[j]))
                for i, j in zip(chosen, order)]
        covered = (covered_aspects(self.embed(aspects), self.doc_embs[chosen]).tolist()
                   if aspects else [])
        return SearchResult(query, docs, aspects, covered)
