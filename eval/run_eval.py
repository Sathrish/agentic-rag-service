"""Benchmark the agent against non-agentic baselines.

Part A, retrieval: NDCG@5 and MRR on NFCorpus test queries with human relevance labels (qrels).
Part B, answers: ROUGE-L against reference answers, faithfulness (verifier) and citation rate
on the 15 questions in eval/rag_qa.json, for single-pass RAG vs the agent.

Usage:
    python -m eval.run_eval --limit 100            # retrieval on 100 queries + all answer questions
    python -m eval.run_eval --skip-agent           # baselines only (no API key needed)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragagent import settings  # noqa: E402
from ragagent.agent import cited_numbers, split_sentences  # noqa: E402
from ragagent.core.config import SEED, TOP_K  # noqa: E402
from ragagent.core.metrics import ndcg_at_k, paired_randomization_test, reciprocal_rank  # noqa: E402
from ragagent.data import load_nfcorpus, load_rag_qa  # noqa: E402

BASELINES = {
    "BM25": dict(first_stage="bm25", reranker="none"),
    "Hybrid (BM25+Dense, RRF)": dict(first_stage="hybrid", reranker="none"),
    "Hybrid + Cross-Encoder": dict(first_stage="hybrid", reranker="cross-encoder"),
    "Hybrid + Aspect-Aware": dict(first_stage="hybrid", reranker="aspect"),
}
AGENT = "Agent (LLM aspects + grading + rewrite)"


class Throttle:
    """Keep LLM calls under a requests-per-minute budget (free tiers are rate limited)."""

    def __init__(self, rpm):
        self.gap = 60.0 / rpm if rpm else 0.0
        self.last = 0.0

    def wait(self, calls=1):
        delay = self.last + self.gap * calls - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def _load_partial(path, key):
    """Rows already finished in an interrupted run, so a quota stop never loses work."""
    if path.exists():
        done = pd.read_csv(path)
        return done.to_dict("records"), set(done[key].astype(str))
    return [], set()


def evaluate_retrieval(engine, agent, queries, qrels, throttle, path):
    idx_of = {d: i for i, d in enumerate(engine.doc_ids)}
    rows, done = _load_partial(path, "query_id")
    for n, (qid, text) in enumerate(queries.items(), start=1):
        if str(qid) in done:
            continue
        relevant = {idx_of[d]: g for d, g in qrels[qid].items()}
        row = {"query_id": qid, "query": text}
        for name, kw in BASELINES.items():
            ranking = [d.idx for d in engine.search(text, top_k=TOP_K, **kw).docs]
            row[f"{name} | ndcg@5"] = ndcg_at_k(ranking, relevant, TOP_K)
            row[f"{name} | mrr"] = reciprocal_rank(ranking, relevant)
        if agent is not None:
            out = agent.search_only(text)
            ranking = out["ranking"][:TOP_K]
            row[f"{AGENT} | ndcg@5"] = ndcg_at_k(ranking, relevant, TOP_K)
            row[f"{AGENT} | mrr"] = reciprocal_rank(ranking, relevant)
            row["agent_rounds"] = out["rounds"]
            row["agent_llm_calls"] = out["llm_calls"]
            row["agent_llm_tokens"] = out["llm_tokens"]
            throttle.wait(out["llm_calls"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        if n % 25 == 0:
            print(f"  retrieval: {n}/{len(queries)} queries")
    return pd.DataFrame(rows)


def evaluate_answers(agent, qa, throttle, path):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows, _ = _load_partial(path, "query")
    finished = {(r["system"], r["query"]) for r in rows}
    for item in qa:
        for system, fn in (("Single-pass RAG", agent.run_single_pass), ("Agent", agent.run)):
            if (system, item["query"]) in finished:
                continue
            out = fn(item["query"])
            throttle.wait(out["llm_calls"])
            sentences = split_sentences(out["answer"])
            rows.append({
                "system": system, "query": item["query"],
                "rougeL": scorer.score(item["reference"], out["answer"])["rougeL"].fmeasure,
                "faithfulness": out["faithfulness"],
                "citation_rate": (sum(bool(cited_numbers(s)) for s in sentences) / len(sentences)) if sentences else 0.0,
                "rounds": out["rounds"], "llm_calls": out["llm_calls"], "llm_tokens": out["llm_tokens"],
                "seconds": out["seconds"], "answer": out["answer"],
            })
            pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def summarise(ret, ans):
    summary = {"retrieval": {}, "answers": {}}
    systems = list(BASELINES) + ([AGENT] if f"{AGENT} | ndcg@5" in ret else [])
    for s in systems:
        summary["retrieval"][s] = {"ndcg@5": round(ret[f"{s} | ndcg@5"].mean(), 4),
                                   "mrr": round(ret[f"{s} | mrr"].mean(), 4)}
    pairs = [("Hybrid (BM25+Dense, RRF)", "BM25"), ("Hybrid + Aspect-Aware", "Hybrid + Cross-Encoder")]
    if AGENT in systems:
        pairs += [(AGENT, "Hybrid + Aspect-Aware"), (AGENT, "Hybrid + Cross-Encoder")]
    summary["significance_ndcg@5"] = []
    for a, b in pairs:
        diff, p = paired_randomization_test(ret[f"{a} | ndcg@5"], ret[f"{b} | ndcg@5"])
        rel = diff / ret[f"{b} | ndcg@5"].mean() if ret[f"{b} | ndcg@5"].mean() else float("nan")
        summary["significance_ndcg@5"].append({"a": a, "b": b, "mean_diff": round(diff, 4),
                                               "relative": round(rel, 4), "p_value": round(p, 4)})
    summary["n_queries"] = int(len(ret))
    if "agent_rounds" in ret:
        summary["agent_rewrite_rate"] = round(float((ret["agent_rounds"] > 1).mean()), 4)
        summary["agent_llm_calls_per_query"] = round(float(ret["agent_llm_calls"].mean()), 2)
        if "agent_llm_tokens" in ret:
            summary["agent_llm_tokens_per_query"] = round(float(ret["agent_llm_tokens"].mean()), 1)
    if ans is not None and len(ans):
        for s, g in ans.groupby("system"):
            summary["answers"][s] = {k: round(float(g[k].mean()), 4) for k in
                                     ("rougeL", "faithfulness", "citation_rate", "llm_calls", "llm_tokens", "seconds")}
        a = ans[ans.system == "Agent"].set_index("query")
        b = ans[ans.system == "Single-pass RAG"].set_index("query").loc[a.index]
        for metric in ("rougeL", "faithfulness"):
            diff, p = paired_randomization_test(a[metric].fillna(0), b[metric].fillna(0))
            summary["answers"][f"agent_vs_single_{metric}"] = {"mean_diff": round(diff, 4), "p_value": round(p, 4)}
        summary["n_answer_questions"] = int(len(a))
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary["retrieval"])
    vals = [summary["retrieval"][n]["ndcg@5"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(names[::-1], vals[::-1], color=["#818cf8" if "Agent" in n else "#5eead4" for n in names[::-1]])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel(f"NDCG@5 on NFCorpus test ({summary['n_queries']} queries)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Retrieval quality: baselines vs agent" if any("Agent" in n for n in names) else "Retrieval quality: non-agentic baselines")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random sample of test queries")
    ap.add_argument("--skip-agent", action="store_true", help="baselines only, no LLM calls")
    ap.add_argument("--skip-answers", action="store_true")
    ap.add_argument("--rpm", type=float, default=25, help="LLM requests per minute budget")
    ap.add_argument("--fresh", action="store_true", help="discard saved progress and start over")
    args = ap.parse_args()

    from ragagent.search import SearchEngine

    engine = SearchEngine.from_nfcorpus()
    _, queries, qrels = load_nfcorpus()
    if args.limit:
        random.seed(SEED)
        keep = sorted(random.sample(sorted(queries), min(args.limit, len(queries))))
        queries = {q: queries[q] for q in keep}

    agent = None
    if not args.skip_agent:
        from ragagent.agent import RagAgent
        from ragagent.llm import ChatLLM
        agent = RagAgent(engine, ChatLLM())
    throttle = Throttle(args.rpm)

    out = settings.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    tag = "agent" if agent is not None else "baselines"
    ret_path, ans_path = out / f"retrieval_{tag}_{len(queries)}.csv", out / "answers.csv"
    if args.fresh:
        ret_path.unlink(missing_ok=True)
        ans_path.unlink(missing_ok=True)

    print(f"Retrieval evaluation on {len(queries)} queries (agent: {agent is not None})")
    try:
        ret = evaluate_retrieval(engine, agent, queries, qrels, throttle, ret_path)
        ans = None
        if agent is not None and not args.skip_answers:
            print("Answer evaluation on eval/rag_qa.json")
            ans = evaluate_answers(agent, load_rag_qa(), throttle, ans_path)
    except Exception as exc:
        if "rate limit" in str(exc).lower() or "429" in str(exc):
            print("Stopped by the LLM provider's rate limit; progress is saved. "
                  f"Rerun the same command later to resume.\n{exc}")
            sys.exit(2)
        raise

    summary = summarise(ret, ans)
    summary["llm_model"] = settings.LLM_MODEL if agent else None
    summary["llm_reasoning_effort"] = settings.LLM_REASONING_EFFORT if agent else None
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    plot(summary, out / f"retrieval_ndcg_{tag}.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
