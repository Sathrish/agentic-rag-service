from fastapi.testclient import TestClient

from fakes import FakeLLM
from ragagent.agent import RagAgent
from ragagent.api import RATE_LIMIT, Service, create_app


def client_for(engine):
    return TestClient(create_app(Service(RagAgent(engine, FakeLLM(aspects=["statins breast cancer"])))))


def test_health_and_ask(engine):
    with client_for(engine) as c:
        assert c.get("/api/health").json()["status"] == "ready"
        r = c.post("/api/ask", json={"question": "Do statins cause breast cancer?"})
        assert r.status_code == 200
        body = r.json()
        assert body["sources"] and body["answer"] and body["trace"]
        assert c.post("/api/ask", json={"question": "Do statins cause breast cancer?",
                                        "mode": "single"}).json()["rounds"] == 1


def test_validation_and_rate_limit(engine):
    with client_for(engine) as c:
        assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
        assert c.post("/api/ask", json={"question": "valid question", "mode": "x"}).status_code == 422
        codes = [c.post("/api/ask", json={"question": "Do statins cause cancer?"}).status_code
                 for _ in range(RATE_LIMIT + 1)]
        assert codes[-1] == 429 and set(codes[:-1]) == {200}


def test_not_ready_returns_503():
    svc = Service()
    svc.error = "RuntimeError: Set GROQ_API_KEY"
    with TestClient(create_app(svc, autoload=False)) as c:
        assert c.get("/api/health").json()["status"] == "error"
        assert c.post("/api/ask", json={"question": "Do statins cause cancer?"}).status_code == 503
