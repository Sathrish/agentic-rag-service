"""FastAPI service: POST /api/ask runs the agent; / serves the demo page."""
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import settings

STATIC = settings.ROOT / "static"
RATE_LIMIT = 8            # questions per client per minute, protects the free LLM quota


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=400)
    mode: str = Field("agent", pattern="^(agent|single)$")


class Service:
    """Holds the engine and agent; built once, in the background, at startup."""

    def __init__(self, agent=None):
        self.agent = agent
        self.error = None
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def load(self):
        try:
            from .agent import RagAgent
            from .llm import ChatLLM
            from .search import SearchEngine

            engine = SearchEngine.from_nfcorpus()
            self.agent = RagAgent(engine, ChatLLM())
        except Exception as exc:  # surfaced through /api/health instead of crashing the server
            self.error = f"{type(exc).__name__}: {exc}"

    def allow(self, client):
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= RATE_LIMIT:
                return False
            hits.append(now)
            return True


def create_app(service: Service | None = None, autoload: bool = True) -> FastAPI:
    svc = service or Service()

    @asynccontextmanager
    async def lifespan(app):
        if autoload and svc.agent is None and svc.error is None:
            threading.Thread(target=svc.load, daemon=True).start()
        yield

    app = FastAPI(title="Agentic RAG", version="1.0", lifespan=lifespan)
    app.state.service = svc

    @app.get("/api/health")
    def health():
        status = "ready" if svc.agent else ("error" if svc.error else "loading")
        return {"status": status, "error": svc.error, "model": settings.LLM_MODEL}

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request):
        if svc.agent is None:
            raise HTTPException(503, svc.error or "The search index is still loading, try again shortly.")
        client = request.client.host if request.client else "unknown"
        if not svc.allow(client):
            raise HTTPException(429, f"Limit is {RATE_LIMIT} questions per minute on this free demo.")
        run = svc.agent.run if req.mode == "agent" else svc.agent.run_single_pass
        try:
            return run(req.question)
        except Exception as exc:
            raise HTTPException(502, f"The language model call failed: {type(exc).__name__}") from exc

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
