"""Retrieval hyper-parameters, kept identical to the aspect-aware-rag-reranking study
(github.com/Sathrish/aspect-aware-rag-reranking) so results stay comparable."""

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEED = 42
TOP_K = 5                     # documents kept after re-ranking
N_CANDIDATES = 20             # first-stage candidates handed to the re-rankers
LAMBDA = 0.7                  # relevance weight in the Aspect-Aware score
ASPECT_SIM_THRESHOLD = 0.25   # cosine threshold: "document covers aspect"
MAX_ASPECTS = 4
