"""Runtime settings, read from environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"

# Any OpenAI-compatible endpoint works. Default: Groq's free tier.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")   # gpt-oss only: low | medium | high
MODEL_RUNTIME = os.getenv("MODEL_RUNTIME", "torch")                  # "torch" (benchmarks) or "onnx" (small hosts)

MAX_REWRITES = int(os.getenv("MAX_REWRITES", "1"))      # extra search rounds the agent may take
MIN_RELEVANT_DOCS = int(os.getenv("MIN_RELEVANT_DOCS", "2"))
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.5"))   # share of aspects the evidence must cover
DOC_CHARS = int(os.getenv("DOC_CHARS", "900"))          # characters of each document shown to the LLM
