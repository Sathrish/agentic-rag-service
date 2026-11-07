"""ONNX versions of the two models (via fastembed), so the service fits a 512 MB free host without PyTorch.

They expose the small slice of the sentence-transformers API that SearchEngine uses:
``encoder.encode(texts, ...)`` and ``cross_encoder.predict(pairs, ...)``.
"""
import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


THREADS = 1       # more threads and bigger batches make onnxruntime's memory pool grow past 512 MB


class OnnxEncoder:
    def __init__(self, model_name=EMBED_MODEL, cache_dir=None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=cache_dir, threads=THREADS)

    def encode(self, texts, batch_size=16, normalize_embeddings=True, **_):
        batch_size = min(batch_size, 16)      # callers may ask for 64; keep memory bounded
        embs = np.asarray(list(self.model.embed(list(texts), batch_size=batch_size)), dtype=np.float32)
        if normalize_embeddings:
            embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return embs


class OnnxCrossEncoder:
    def __init__(self, model_name=RERANK_MODEL, cache_dir=None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name, cache_dir=cache_dir, threads=THREADS)

    def predict(self, pairs, batch_size=1, **_):
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        query = pairs[0][0]
        if all(q == query for q, _ in pairs):          # the engine always scores one query at a time
            return np.asarray(list(self.model.rerank(query, [d for _, d in pairs], batch_size=batch_size)), dtype=np.float32)
        return np.asarray([next(iter(self.model.rerank(q, [d]))) for q, d in pairs], dtype=np.float32)
