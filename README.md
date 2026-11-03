# Agentic RAG for medical research

A **LangGraph agent** that answers medical and nutrition questions from 3,633 research abstracts (NFCorpus). It plans its searches, judges which documents are relevant, rewrites the query when the evidence is thin, answers with inline citations, and then **verifies** each sentence against the sources. It runs behind a **FastAPI** service with a small web demo and is packaged with **Docker**.

It extends my earlier study, [aspect-aware-rag-reranking](https://github.com/Sathrish/aspect-aware-rag-reranking), whose retrieval code is reused unchanged. The question this repo asks is not "can I build an agent?" but **"does the agent actually retrieve and answer better than a simpler pipeline, and what does it cost?"**

## How the agent works

```
question ─► plan ─┬─► (greeting / off-topic) ─────────────► direct reply
                  └─► retrieve ─► grade ─┬─ enough evidence ─► answer ─► verify ─► response
                        ▲                └─ thin evidence ─► rewrite ─┘ (bounded: 1 extra round)
                        └──────────────────────────────────────────────┘
```

| Step | What it does |
|---|---|
| **plan** | LLM routes the question (retrieve vs direct reply) and splits it into 1–4 *aspects*, e.g. "statins breast cancer risk", "statins breast cancer survival". Falls back to a rule-based splitter if the LLM fails. |
| **retrieve** | Hybrid search: BM25 + dense MiniLM embeddings (FAISS), fused with Reciprocal Rank Fusion, then **aspect-aware re-ranking** (cross-encoder relevance balanced against covering new aspects). Never re-shows a document already graded. |
| **grade** | LLM marks which retrieved documents actually help. The agent then measures **aspect coverage**: the share of aspects covered by the relevant evidence. |
| **rewrite** | If fewer than 2 relevant documents or under 50% coverage, the LLM writes a new query aimed at the *missing* aspects. Capped at one extra round. |
| **answer** | LLM answers from the evidence only, citing `[n]` after each claim, and says so when the sources don't cover something. |
| **verify** | A second LLM pass checks each sentence against its cited sources and reports a **faithfulness** score plus the unsupported sentences. |

The UI shows every step (the agent trace), the sources, and the metrics for each answer.

## Results

### Retrieval baselines (all 323 NFCorpus test queries, human relevance labels)

![NDCG@5 of the non-agentic baselines](results/retrieval_ndcg_baselines.png)

| System | NDCG@5 | MRR |
|---|---|---|
| BM25 | 0.3344 | 0.500 |
| Hybrid (BM25 + dense, RRF) | 0.3673 | 0.538 |
| Hybrid + cross-encoder | **0.3919** | **0.567** |
| Hybrid + aspect-aware | 0.3889 | 0.560 |

- **Reproducibility check:** BM25 scores exactly 0.3344, the same as my original study, so the reused code behaves identically.
- **Hybrid fusion helps:** +9.9% NDCG@5 over BM25 (paired randomisation test, 10,000 resamples, p < 0.001).
- **Aspect-aware re-ranking does not beat the cross-encoder here:** −0.8% NDCG@5 (p = 0.03). Its purpose is covering more aspects of a question, which NDCG does not reward, so the agent uses it for evidence coverage and this repo reports the trade-off rather than hiding it.

### Agent vs baselines

**In progress.** The full benchmark (agent retrieval on 100 fixed random queries, plus answer quality on 15 reference questions: ROUGE-L, faithfulness, citation rate, LLM calls, tokens and latency, single-pass RAG vs agent) is being run within the free API tier's daily token cap. Results will appear here, whichever way they come out.

What I have measured so far, on single questions:

| Question | Search rounds | Aspect coverage | Faithfulness | LLM calls | Seconds |
|---|---|---|---|---|---|
| Is coffee good or bad for the liver? | 1 | 100% | 100% | 4 | 7.0 |
| Do statins increase breast cancer risk, and do they affect survival? | 1 | 100% | 50% | 4 | 10.1 |

The second row is the verifier doing its job: it flagged 2 of 4 sentences as not fully supported by the source excerpts.

## Engineering notes and problems I hit

- **Provider-agnostic LLM layer.** Any OpenAI-compatible API works through three environment variables. The default is Groq's free tier with `openai/gpt-oss-120b`.
- **Free-tier limits shaped the design.** Groq's free tier allows 1,000 requests and 200,000 tokens per model per day. My first benchmark run hit the daily token cap after 100 queries and, because results were only saved at the end, lost them. The evaluation now **saves after every query and resumes** where it stopped. Setting `reasoning_effort=low` for gpt-oss cut hidden reasoning tokens by about 20x (225 → 11 on a grading call). The API also rate-limits each client to protect the shared demo quota.
- **Retired model.** My first default model had been retired by the provider; the code now lists available models instead of assuming.
- **Citation format.** gpt-oss cites with `【3】` rather than `[3]`, which made the citation counter report zero. Answers are normalised before scoring.
- **Truncation was hiding conclusions.** Feeding the first 700 characters of each abstract cut off the conclusion, so the agent wrongly called the statin survival evidence "uncertain". It now sends each abstract's opening **and** ending.
- **Dataset download.** The original BEIR server intermittently fails Python's TLS verification, so the loader falls back to the identical Hugging Face mirror (verified: 3,633 documents, 323 test queries).
- **Robustness.** If the planner, grader or verifier returns bad JSON, the agent degrades gracefully (rule-based aspects, keep all documents, skip verification) and records it in the trace rather than failing the request.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (use source .venv/bin/activate elsewhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env                                # then paste a free key from https://console.groq.com/keys
python -m ragagent.prepare                            # downloads NFCorpus + models, caches embeddings
uvicorn ragagent.api:app --port 7860                  # open http://localhost:7860
```

Tests use fake models and a fake LLM, so they run offline with no API key:

```bash
python -m pytest -q          # 20 tests: search, agent routing, rewrite cap, verifier, API, rate limit
```

Benchmarks:

```bash
python -m eval.run_eval --skip-agent             # baselines only, no API key needed
python -m eval.run_eval --limit 100 --rpm 8      # agent + answer quality; resumes if interrupted
```

Docker (also used for the free Hugging Face Spaces deployment):

```bash
docker build -t agentic-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=... agentic-rag
```

## API

`POST /api/ask` with `{"question": "...", "mode": "agent" | "single"}` returns the answer, sources, aspects, search rounds, aspect coverage, faithfulness, unsupported sentences, the step-by-step trace, LLM calls, tokens and latency. `GET /api/health` reports whether the index is loaded.

## Project layout

```
ragagent/
  core/        retrieval code reused unchanged from aspect-aware-rag-reranking
  search.py    hybrid BM25 + dense search with RRF, cross-encoder / aspect-aware re-ranking
  agent.py     LangGraph state graph: plan, retrieve, grade, rewrite, answer, verify
  llm.py       OpenAI-compatible client (Groq by default), JSON parsing, token tracking
  api.py       FastAPI service with per-client rate limiting
eval/run_eval.py   benchmarks with paired randomisation tests, resumable
static/index.html  demo page with the agent trace
tests/             offline tests with fake models
```

## Limitations

- NFCorpus abstracts are short and the relevance labels are partial, so absolute scores are modest for every system.
- The answer benchmark has only 15 reference questions; treat its numbers as indicative.
- Faithfulness is judged by the same LLM family that wrote the answer, which can be lenient. A different judge model would be a stronger test.
- This is a research demo, not medical advice.

---
Built by [Sathrish S](https://sathrish.github.io), with Claude Code as a coding assistant.
