"""Download NFCorpus and both models, and cache the document embeddings.

Run once (the Docker build does this) so the server starts in seconds."""
from .search import SearchEngine

if __name__ == "__main__":
    engine = SearchEngine.from_nfcorpus()
    print(f"Ready: {len(engine.doc_ids)} documents indexed, embeddings {engine.doc_embs.shape}")
