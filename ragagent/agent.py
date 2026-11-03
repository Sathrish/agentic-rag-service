"""LangGraph agent: plan -> retrieve -> grade -> (rewrite -> retrieve)* -> generate -> verify.

The LLM decides whether a question needs retrieval, splits it into aspects, judges
which retrieved documents are relevant, rewrites the query when evidence is thin,
and finally answers with citations that a separate verification step checks.
"""
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import settings
from .core import config
from .core.aspects import extract_aspects
from .core.metrics import covered_aspects

PLAN_SYSTEM = """You plan searches for a question-answering assistant. Its knowledge base is
NFCorpus: abstracts of medical and nutrition research (diet, diseases, drugs, supplements).
Return JSON: {"route": "retrieve" or "direct", "aspects": [...], "reply": "..."}.
- route "direct" only for greetings or questions unrelated to health, nutrition or medicine;
  then "reply" briefly says what the assistant can help with and "aspects" is [].
- otherwise route "retrieve" and list 1 to 4 short search phrases ("aspects"), each covering
  a distinct part of the question. Leave "reply" empty."""

GRADE_SYSTEM = """You judge search results. For the question, decide which numbered documents
contain information that helps answer it. Return JSON: {"relevant": [document numbers]}."""

REWRITE_SYSTEM = """The search results so far miss parts of the question. Write ONE new search
query (under 15 words) that targets the missing aspects. Return JSON: {"query": "..."}."""

ANSWER_SYSTEM = """Answer the question using ONLY the numbered sources. Cite sources inline like [1]
or [2][3] after each claim. Write 3 to 6 sentences. If the sources do not answer part of the
question, say so plainly instead of guessing. Do not give personal medical advice."""

VERIFY_SYSTEM = """Check an answer against its sources. For each numbered answer sentence,
decide whether the cited sources support it. Return JSON: {"unsupported": [sentence numbers]}."""

NO_EVIDENCE = ("I couldn't find research in the knowledge base that answers this question, "
               "so I won't guess.")

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_CITATION = re.compile(r"\[(\d+)\]")


def split_sentences(text):
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


_ODD_CITATION = re.compile(r"[【\[]\s*(\d+)\s*[】\]]")
_PUNCT = str.maketrans({"‑": "-", "‐": "-", "–": "-", "—": " - ", " ": " ", " ": " "})


def normalise_answer(text):
    """Map model-specific citation brackets (e.g. 【3】) to [3] and odd dashes/spaces to ASCII."""
    return _ODD_CITATION.sub(r"[\1]", text.translate(_PUNCT)).strip()


def cited_numbers(text):
    return sorted({int(n) for n in _CITATION.findall(text)})


def excerpt(text, chars=settings.DOC_CHARS):
    """Opening and ending of an abstract: the ending usually holds the conclusion."""
    if len(text) <= chars:
        return text
    head = int(chars * 0.45)
    return f"{text[:head].rstrip()} ... {text[-(chars - head):].lstrip()}"


def format_sources(docs, chars=settings.DOC_CHARS):
    return "\n\n".join(f"[{i}] {d.title}. {excerpt(d.text, chars)}" for i, d in enumerate(docs, start=1))


class AgentState(TypedDict, total=False):
    question: str
    route: str
    reply: str
    aspects: list
    query: str
    rounds: int
    seen: list          # corpus indices already shown to the grader
    evidence: list      # Doc objects judged relevant, in the order found
    retrieved: list     # Doc objects from the latest search
    coverage: float
    answer: str
    sources: list
    unsupported: list
    faithfulness: float
    trace: Annotated[list, operator.add]


class RagAgent:
    def __init__(self, engine, llm, max_rewrites=settings.MAX_REWRITES,
                 min_docs=settings.MIN_RELEVANT_DOCS, min_coverage=settings.MIN_COVERAGE):
        self.engine = engine
        self.llm = llm
        self.max_rewrites = max_rewrites
        self.min_docs = min_docs
        self.min_coverage = min_coverage
        self.graph = self._build()
        self.search_graph = self._build(answer=False)

    # ---- nodes -------------------------------------------------------------------------
    def plan(self, state):
        q = state["question"]
        try:
            out = self.llm.json(PLAN_SYSTEM, f"Question: {q}")
            route = "direct" if out.get("route") == "direct" else "retrieve"
            aspects = [str(a).strip() for a in out.get("aspects") or [] if str(a).strip()]
            reply = str(out.get("reply") or "").strip()
        except Exception as exc:  # fall back to rule-based aspects rather than failing the request
            route, aspects, reply = "retrieve", [], ""
            note = f"planner failed ({type(exc).__name__}); using rule-based aspects"
        else:
            note = f"route={route}"
        if route == "retrieve" and not aspects:
            aspects = extract_aspects(q)
        aspects = aspects[:config.MAX_ASPECTS]
        return {"route": route, "reply": reply, "aspects": aspects, "query": q, "rounds": 0,
                "seen": [], "evidence": [],
                "trace": [{"step": "plan", "detail": note, "aspects": aspects}]}

    def retrieve(self, state):
        res = self.engine.search(state["query"], aspects=state["aspects"], exclude=state["seen"])
        return {"retrieved": res.docs, "rounds": state["rounds"] + 1,
                "seen": state["seen"] + [d.idx for d in res.docs],
                "trace": [{"step": "retrieve", "query": state["query"],
                           "docs": [d.doc_id for d in res.docs]}]}

    def grade(self, state):
        docs = state["retrieved"]
        relevant, note = docs, None
        if docs:
            listing = format_sources(docs, chars=400)
            try:
                out = self.llm.json(GRADE_SYSTEM, f"Question: {state['question']}\n\nDocuments:\n{listing}")
                keep = {int(n) for n in out.get("relevant", []) if str(n).isdigit()}
                relevant = [d for i, d in enumerate(docs, start=1) if i in keep]
            except Exception as exc:
                relevant = docs   # keep everything if the grader fails
                note = f"grader failed ({type(exc).__name__}); kept all documents"
        evidence = state["evidence"] + relevant
        coverage = self._coverage(state["aspects"], evidence)
        step = {"step": "grade", "relevant": [d.doc_id for d in relevant],
                "evidence": len(evidence), "coverage": round(coverage, 2)}
        if note:
            step["detail"] = note
        return {"evidence": evidence, "coverage": coverage, "trace": [step]}

    def rewrite(self, state):
        uncovered = self._uncovered(state["aspects"], state["evidence"]) or state["aspects"]
        try:
            out = self.llm.json(REWRITE_SYSTEM, f"Question: {state['question']}\n"
                                                f"Previous query: {state['query']}\n"
                                                f"Missing aspects: {'; '.join(uncovered)}")
            query = str(out.get("query") or "").strip() or " ".join(uncovered)
        except Exception:
            query = " ".join(uncovered)
        return {"query": query, "trace": [{"step": "rewrite", "query": query, "missing": uncovered}]}

    def generate(self, state):
        if state.get("route") == "direct":
            return {"answer": state["reply"] or "I answer questions about nutrition and medical research.",
                    "sources": [], "trace": [{"step": "answer", "detail": "direct reply"}]}
        docs = state["evidence"][:6]
        if not docs:
            return {"answer": NO_EVIDENCE, "sources": [], "trace": [{"step": "answer", "detail": "no evidence"}]}
        answer = normalise_answer(self.llm.text(
            ANSWER_SYSTEM, f"Sources:\n{format_sources(docs)}\n\nQuestion: {state['question']}"))
        return {"answer": answer, "sources": docs,
                "trace": [{"step": "answer", "cited": cited_numbers(answer)}]}

    def verify(self, state):
        docs, answer = state.get("sources") or [], state.get("answer", "")
        sentences = split_sentences(answer)
        if not docs or not sentences:
            return {"unsupported": [], "faithfulness": None}
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        try:
            out = self.llm.json(VERIFY_SYSTEM, f"Sources:\n{format_sources(docs)}\n\nAnswer sentences:\n{numbered}")
            bad = sorted({int(n) for n in out.get("unsupported", []) if str(n).isdigit()
                          and 1 <= int(n) <= len(sentences)})
        except Exception:
            return {"unsupported": [], "faithfulness": None,
                    "trace": [{"step": "verify", "detail": "verifier failed"}]}
        faith = 1 - len(bad) / len(sentences)
        return {"unsupported": [sentences[i - 1] for i in bad], "faithfulness": faith,
                "trace": [{"step": "verify", "faithfulness": round(faith, 2)}]}

    # ---- routing -----------------------------------------------------------------------
    def after_plan(self, state):
        return "generate" if state["route"] == "direct" else "retrieve"

    def after_grade(self, state):
        enough = len(state["evidence"]) >= self.min_docs and state["coverage"] >= self.min_coverage
        if enough or state["rounds"] > self.max_rewrites:
            return "generate"
        return "rewrite"

    def _build(self, answer=True):
        """Full graph, or (answer=False) only the search loop, for retrieval benchmarks."""
        g = StateGraph(AgentState)
        nodes = ["plan", "retrieve", "grade", "rewrite"] + (["generate", "verify"] if answer else [])
        for name in nodes:
            g.add_node(name, getattr(self, name))
        done = "generate" if answer else END
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self.after_plan, {"generate": done, "retrieve": "retrieve"})
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self.after_grade, {"generate": done, "rewrite": "rewrite"})
        g.add_edge("rewrite", "retrieve")
        if answer:
            g.add_edge("generate", "verify")
            g.add_edge("verify", END)
        return g.compile()

    # ---- helpers -----------------------------------------------------------------------
    def _covered(self, aspects, docs):
        if not aspects or not docs:
            return [False] * len(aspects)
        return covered_aspects(self.engine.embed(aspects), self.engine.doc_embs[[d.idx for d in docs]]).tolist()

    def _coverage(self, aspects, docs):
        cov = self._covered(aspects, docs)
        return sum(cov) / len(cov) if cov else 0.0

    def _uncovered(self, aspects, docs):
        return [a for a, c in zip(aspects, self._covered(aspects, docs)) if not c]

    # ---- public API --------------------------------------------------------------------
    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        calls0, tokens0 = getattr(self.llm, "calls", 0), getattr(self.llm, "tokens", 0)
        state = self.graph.invoke({"question": question.strip(), "trace": []})
        return self._result(state, t0, calls0, tokens0)

    def search_only(self, question: str) -> dict:
        """Plan, retrieve, grade and rewrite, but do not answer (for retrieval evaluation)."""
        t0 = time.perf_counter()
        calls0, tokens0 = getattr(self.llm, "calls", 0), getattr(self.llm, "tokens", 0)
        state = self.search_graph.invoke({"question": question.strip(), "trace": []})
        return self._result(state, t0, calls0, tokens0)

    def run_single_pass(self, question: str) -> dict:
        """Baseline: one hybrid + aspect-aware search, no planning, grading or rewriting."""
        t0 = time.perf_counter()
        calls0, tokens0 = getattr(self.llm, "calls", 0), getattr(self.llm, "tokens", 0)
        res = self.engine.search(question)
        state = {"question": question, "route": "retrieve", "aspects": res.aspects,
                 "evidence": res.docs, "rounds": 1, "trace": []}
        state.update(self.generate(state))
        state.update(self.verify(state))
        return self._result(state, t0, calls0, tokens0)

    def _result(self, state, t0, calls0, tokens0=0):
        docs = state.get("sources") or []
        ranking = [d.idx for d in state.get("evidence") or []]
        ranking += [d.idx for d in state.get("retrieved") or [] if d.idx not in ranking]
        return {
            "question": state["question"],
            "route": state.get("route"),
            "answer": state.get("answer", ""),
            "sources": [{"n": i, "doc_id": d.doc_id, "title": d.title, "snippet": d.text[:300]}
                        for i, d in enumerate(docs, start=1)],
            "aspects": state.get("aspects", []),
            "rounds": state.get("rounds", 0),
            "coverage": state.get("coverage"),
            "faithfulness": state.get("faithfulness"),
            "unsupported": state.get("unsupported", []),
            "ranking": ranking,
            "trace": state.get("trace", []),
            "llm_calls": getattr(self.llm, "calls", 0) - calls0,
            "llm_tokens": getattr(self.llm, "tokens", 0) - tokens0,
            "seconds": round(time.perf_counter() - t0, 2),
        }
